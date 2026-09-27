"""Generate E1 controls through the same AgentLoop async-vLLM path as C.

Modes:
  fixed: generate eight roots independently of C and seven independent
         continuations from response prefix 128;
  gstar: generate 32 independent root-only reference trajectories.

All prompts are retained; the prompt region includes the fixed 128-token prefix.
"""

import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

LLM_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if LLM_DIR not in sys.path:
    sys.path.insert(0, LLM_DIR)

import hydra
import numpy as np
import pandas as pd
import ray
import torch
from omegaconf import OmegaConf

from experiments.RQ3.gradients import _prompt_ids, _reward_text_task
from experiments.RQ2.search import request_sampling_seed
from verl.protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask
from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker


BRANCH_PREFIX = 128
N_SUFFIXES = 7


def _select(config, path, default=None):
    value = OmegaConf.select(config, path)
    return default if value is None else value


def _required(config, path):
    value = OmegaConf.select(config, path)
    if value is None:
        raise ValueError(f"missing required config field: {path}")
    return value




def _request_batch(requests, tokenizer, prompt_length):
    """Build the continuation-mode DataProto consumed by AgentLoopManager."""
    batch_size = len(requests)
    input_ids = torch.full((batch_size, prompt_length), tokenizer.pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, prompt_length), dtype=torch.long)
    raw_prompts = np.empty(batch_size, dtype=object)
    raw_prompt_lens = np.empty(batch_size, dtype=np.int64)
    seeds = np.empty(batch_size, dtype=np.int64)
    for index, request in enumerate(requests):
        ids = request["prompt_ids"]
        if not 0 < len(ids) <= prompt_length:
            raise ValueError(f"invalid continuation prompt length {len(ids)} (limit {prompt_length})")
        input_ids[index, -len(ids):] = torch.as_tensor(ids, dtype=torch.long)
        attention_mask[index, -len(ids):] = 1
        raw_prompts[index] = request["raw_prompt"]
        raw_prompt_lens[index] = int(request["raw_prompt_len"])
        if not 0 < raw_prompt_lens[index] <= len(ids):
            raise ValueError(
                f"invalid raw prompt length {raw_prompt_lens[index]} "
                f"for continuation length {len(ids)}"
            )
        seeds[index] = request["seed"]
    batch = DataProto.from_single_dict(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": compute_position_id_with_mask(attention_mask),
        }
    )
    batch.non_tensor_batch = {
        "raw_prompt": raw_prompts,
        "raw_prompt_len": raw_prompt_lens,
        "sampling_seed": seeds,
    }
    batch.meta_info = {"validate": False, "temperature": 1.0}
    return batch


def _generate(manager, requests, tokenizer, rollout_config, sleep_after=False):
    if not requests:
        return []
    batch = _request_batch(requests, tokenizer, int(rollout_config.prompt_length))
    divisor = int(rollout_config.agent.num_workers)
    padded, pad_size = pad_dataproto_to_divisor(batch, divisor)
    output = unpad_dataproto(
        manager.generate_sequences(padded, sleep_after=sleep_after), pad_size=pad_size
    )
    response_mask = output.batch["response_mask"].cpu().bool()
    responses = output.batch["responses"].cpu()
    return [
        np.asarray(response[mask].tolist(), dtype=np.int32)
        for response, mask in zip(responses, response_mask)
    ]


def _reward_records(records, prompt_frame, tokenizer, reward_procs, fixed):
    tasks = []
    for record in records:
        global_prompt = record["prompt_idx"]
        row = prompt_frame.loc[global_prompt]
        data_source = row["data_source"]
        ground_truth = row["reward_model"]["ground_truth"]
        if fixed:
            root_text = tokenizer.decode(record["backbone_ids"], skip_special_tokens=True)
            tasks.append(((global_prompt, record["group"], -1), data_source, root_text, ground_truth))
            prefix = record["backbone_ids"][:BRANCH_PREFIX]
            for suffix_idx, suffix in enumerate(record["suffix_ids"]):
                if len(suffix):
                    text = tokenizer.decode(np.concatenate([prefix, suffix]), skip_special_tokens=True)
                    tasks.append(((global_prompt, record["group"], suffix_idx), data_source, text, ground_truth))
        else:
            text = tokenizer.decode(record["backbone_ids"], skip_special_tokens=True)
            tasks.append(((global_prompt, record["group"], -1), data_source, text, ground_truth))
    scores = {}
    with Pool(processes=reward_procs) as pool:
        for key, score in pool.imap_unordered(_reward_text_task, tasks, chunksize=32):
            scores[key] = score
    for record in records:
        key = (record["prompt_idx"], record["group"])
        if fixed:
            record["backbone_reward"] = np.float32(scores[(*key, -1)])
            record["suffix_rewards"] = np.asarray(
                [scores.get((*key, suffix_idx), 0.0) for suffix_idx in range(N_SUFFIXES)],
                dtype=np.float32,
            )
        else:
            record["backbone_reward"] = np.float32(scores[(*key, -1)])




def _run_fixed(config, manager, tokenizer, rollout_config, prompt_frame):
    shard = int(_required(config, "data.shard"))
    shard_size = int(_required(config, "data.shard_size"))
    seed_base = int(_required(config, "data.rollout_seed_base"))
    lo = shard * shard_size
    hi = lo + shard_size
    prompt_indices = list(range(lo, hi))
    root_requests = []
    for group in range(8):
        for global_prompt in prompt_indices:
            prompt_ids = _prompt_ids(prompt_frame.loc[global_prompt], tokenizer)
            root_requests.append(
                {
                    "key": (global_prompt, group),
                    "prompt_ids": prompt_ids,
                    "raw_prompt": prompt_frame.loc[global_prompt, "prompt"],
                    "raw_prompt_len": len(prompt_ids),
                    "seed": request_sampling_seed(seed_base, global_prompt, group, 0),
                }
            )
    started = time.time()
    root_responses = _generate(manager, root_requests, tokenizer, rollout_config)
    empty = np.asarray([], dtype=np.int32)
    records = []
    for request, response in zip(root_requests, root_responses):
        global_prompt, group = request["key"]
        records.append(
            {
                "prompt_idx": global_prompt,
                "group": group,
                "prefix_len": min(BRANCH_PREFIX, len(response)),
                "backbone_ids": response,
                "backbone_finish": "unified_async_independent",
                "backbone_reward": np.float32(0.0),
                "suffix_ids": [empty] * N_SUFFIXES,
                "suffix_finish": ["short_root"] * N_SUFFIXES,
                "suffix_rewards": np.zeros(N_SUFFIXES, dtype=np.float32),
                "backbone_sampling_seed": request["seed"],
                "suffix_sampling_seeds": np.full(N_SUFFIXES, -1, dtype=np.int64),
            }
        )
    print(
        f"[fixed] roots: {len(root_requests)} independent requests in "
        f"{time.time() - started:.1f}s",
        flush=True,
    )
    by_key = {(record["prompt_idx"], record["group"]): record for record in records}

    for suffix_idx in range(N_SUFFIXES):
        requests = []
        for record in records:
            if len(record["backbone_ids"]) <= BRANCH_PREFIX:
                continue
            global_prompt, group = record["prompt_idx"], record["group"]
            original = _prompt_ids(prompt_frame.loc[global_prompt], tokenizer)
            branch_prompt = original + record["backbone_ids"][:BRANCH_PREFIX].tolist()
            requests.append(
                {
                    "key": (global_prompt, group),
                    "prompt_ids": branch_prompt,
                    "raw_prompt": prompt_frame.loc[global_prompt, "prompt"],
                    "raw_prompt_len": len(original),
                    "seed": request_sampling_seed(seed_base, global_prompt, group, suffix_idx + 1),
                }
            )
        started = time.time()
        responses = _generate(
            manager, requests, tokenizer, rollout_config, sleep_after=suffix_idx == N_SUFFIXES - 1
        )
        for request, response in zip(requests, responses):
            record = by_key[request["key"]]
            record["suffix_ids"][suffix_idx] = response
            record["suffix_finish"][suffix_idx] = "unified_async"
            record["suffix_sampling_seeds"][suffix_idx] = request["seed"]
        print(
            f"[fixed] suffix {suffix_idx + 1}/{N_SUFFIXES}: {len(requests)} requests "
            f"in {time.time() - started:.1f}s",
            flush=True,
        )
    return records


def _run_gstar(config, manager, tokenizer, rollout_config, prompt_frame):
    shard = int(_required(config, "data.shard"))
    shard_size = int(_required(config, "data.shard_size"))
    seed_base = int(_required(config, "data.rollout_seed_base"))
    groups_per_call = int(_select(config, "data.groups_per_call", 8))
    lo = shard * shard_size
    hi = lo + shard_size
    prompt_indices = list(range(lo, hi))
    records = []
    empty = np.asarray([], dtype=np.int32)
    for group_start in range(0, 32, groups_per_call):
        groups = range(group_start, min(32, group_start + groups_per_call))
        requests = []
        for group in groups:
            for global_prompt in prompt_indices:
                prompt_ids = _prompt_ids(prompt_frame.loc[global_prompt], tokenizer)
                requests.append(
                    {
                        "key": (global_prompt, group),
                        "prompt_ids": prompt_ids,
                        "raw_prompt": prompt_frame.loc[global_prompt, "prompt"],
                        "raw_prompt_len": len(prompt_ids),
                        "seed": request_sampling_seed(seed_base, global_prompt, group, 0),
                    }
                )
        started = time.time()
        responses = _generate(
            manager,
            requests,
            tokenizer,
            rollout_config,
            sleep_after=group_start + groups_per_call >= 32,
        )
        for request, response in zip(requests, responses):
            global_prompt, group = request["key"]
            records.append(
                {
                    "prompt_idx": global_prompt,
                    "group": group,
                    "prefix_len": min(BRANCH_PREFIX, len(response)),
                    "backbone_ids": response,
                    "backbone_finish": "unified_async",
                    "backbone_reward": np.float32(0.0),
                    "suffix_ids": [empty] * N_SUFFIXES,
                    "suffix_finish": ["root_only"] * N_SUFFIXES,
                    "suffix_rewards": np.zeros(N_SUFFIXES, dtype=np.float32),
                    "backbone_sampling_seed": request["seed"],
                }
            )
        print(
            f"[gstar] groups {group_start}-{group_start + len(list(groups)) - 1}: "
            f"{len(requests)} requests in {time.time() - started:.1f}s",
            flush=True,
        )
    return records


@hydra.main(config_path="pkg://verl.trainer.config", config_name="ppo_trainer", version_base=None)
def main(config):
    if not ray.is_initialized():
        ray_kwargs = OmegaConf.to_container(config.ray_kwargs.get("ray_init", {}), resolve=True)
        ray.init(**ray_kwargs)
    ray.get(main_task.remote(config))


@ray.remote(num_cpus=1)
def main_task(config):
    OmegaConf.resolve(config)
    mode = str(_required(config, "data.control_mode"))
    output_path = Path(str(_required(config, "data.output_path")))
    prompt_path = str(_required(config, "data.prompt_source"))
    reward_procs = int(_select(config, "data.reward_procs", 32))
    model_path = str(_required(config, "actor_rollout_ref.model.path"))
    rollout_config = config.actor_rollout_ref.rollout
    tokenizer = hf_tokenizer(copy_to_local(model_path), trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompt_frame = pd.read_parquet(prompt_path)
    prompt_frame.index = np.arange(len(prompt_frame))
    limit = int(rollout_config.prompt_length)
    longest = max(len(_prompt_ids(row, tokenizer)) for _, row in prompt_frame.iterrows())
    if longest + BRANCH_PREFIX > limit:
        raise ValueError(f'Increase prompt_length to at least {longest + BRANCH_PREFIX}; no prompts may be dropped')

    resource_pool = RayResourcePool(process_on_nodes=[1], use_gpu=True, max_colocate_count=1)
    actor_cls = RayClassWithInitArgs(
        cls=ray.remote(AsyncActorRolloutRefWorker),
        config=config.actor_rollout_ref,
        role="rollout",
    )
    worker_cls = create_colocated_worker_cls(class_dict={"actor_rollout": actor_cls})
    worker_group = RayWorkerGroup(
        resource_pool=resource_pool, ray_cls_with_init=worker_cls, device_name=config.trainer.device
    ).spawn(prefix_set={"actor_rollout": actor_cls})["actor_rollout"]
    worker_group.init_model()
    from verl.experimental.agent_loop import AgentLoopManager

    manager = AgentLoopManager(config=config, worker_group=worker_group, rm_resource_pool=None)
    if mode == "fixed":
        records = _run_fixed(config, manager, tokenizer, rollout_config, prompt_frame)
        fixed = True
    elif mode == "gstar":
        records = _run_gstar(config, manager, tokenizer, rollout_config, prompt_frame)
        fixed = False
    else:
        raise ValueError(f"unknown control mode: {mode}")

    _reward_records(records, prompt_frame, tokenizer, reward_procs, fixed=fixed)
    result = pd.DataFrame(records).sort_values(["prompt_idx", "group"]).reset_index(drop=True)
    shard_size = int(_required(config, "data.shard_size"))
    shard = int(_required(config, "data.shard"))
    groups = 8 if fixed else 32
    expected = shard_size * groups
    if len(result) != expected or result.duplicated(["prompt_idx", "group"]).any():
        raise ValueError(f"output grid mismatch: rows={len(result)}, expected={expected}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    result.to_parquet(temporary)
    os.replace(temporary, output_path)
    print(f"saved mode={mode} rows={len(result)} -> {output_path}", flush=True)


if __name__ == "__main__":
    main()
