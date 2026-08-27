# Copyright 2025 Junyu Lu (Julian Lou). All rights reserved.

"""
RQ2: OPTS tree-count scaling. Reward mode + zero OTRC baseline, two phases:
  - call 0: open all trees (trees_per_prompt * dataset_size prompts, one drawn
    prompt = one tree, uid = tree ID) and generate all roots in one batch;
  - the next n_branch_rounds calls: branch all OTRC-qualifying positions of the
    existing trees in one batch per round; no new trees.
After branch rounds in opts_avg_snapshot_rounds, snapshot each tree's greedy
terminal response into column opts_avg_responses_s{round} for opts-avg@k eval.
"""

import os
import sys
from collections import defaultdict

LLM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if LLM_DIR not in sys.path:
    sys.path.insert(0, LLM_DIR)

import hydra
import numpy as np
import ray

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

from pprint import pprint

import torch
import pandas as pd
from omegaconf import OmegaConf

from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.hdfs_io import makedirs
from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker

from trainer.opts_ttpo.core_algos import (
    compute_treegae_advantage_return,
)
from trainer.opts_ttpo.ray_trainer import (
    PromptBuffer,
    compute_episodic_returns,
    compute_response_mask,
    decode_response_strs,
    merge_batches,
    prepare_next_round_input,
    refresh_tree_search_states,
    set_opts_ttpo_info,
    select_next_states,
    selected_to_branch_points,
)
from verl.trainer.ppo.reward import compute_reward

REWARD_MODE = "reward"
OTRC_BASELINE_MODE = "zero"


def _select_first(config, *paths, default=None):
    for path in paths:
        value = OmegaConf.select(config, path)
        if value is not None:
            return value
    return default


def _require_config_value(config, *paths):
    value = _select_first(config, *paths)
    if value is None:
        joined_paths = ", ".join(paths)
        raise ValueError(f"Missing required config value. Tried: {joined_paths}")
    return value


def _get_actor_worker_config(config):
    return _select_first(config, "actor_rollout_ref", default=config)


def _get_rollout_config(config):
    return _require_config_value(config, "actor_rollout_ref.rollout", "rollout")


@hydra.main(config_path="pkg://verl.trainer.config", config_name="ppo_trainer", version_base=None)
def main(config):
    run_generation(config)


def run_generation(config) -> None:
    if not ray.is_initialized():
        default_runtime_env = {"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}}
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    ray.get(main_task.remote(config))


@ray.remote(num_cpus=1)
def main_task(config):
    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    actor_worker_config = _get_actor_worker_config(config)
    rollout_config = _get_rollout_config(config)

    model_path = _require_config_value(config, "actor_rollout_ref.model.path", "model.path")
    data_path = _require_config_value(config, "data.path", "data.val_files", "data.train_files")
    batch_size = _require_config_value(config, "data.batch_size", "data.val_batch_size", "data.train_batch_size")
    trees_per_prompt = int(_select_first(config, "data.trees_per_prompt", default=32))
    output_path = _require_config_value(config, "data.output_path")
    trust_remote_code = _select_first(
        config,
        "actor_rollout_ref.model.trust_remote_code",
        "model.trust_remote_code",
        "data.trust_remote_code",
        default=False,
    )

    local_path = copy_to_local(model_path)
    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)

    assert batch_size >= 1, f"batch_size must be >= 1, got {batch_size}"
    assert trees_per_prompt >= 1, f"trees_per_prompt must be >= 1, got {trees_per_prompt}"

    prompt_length = rollout_config.prompt_length
    response_length = rollout_config.response_length
    max_search_per_tree = rollout_config.get("max_search_per_tree", 1)
    n_branch_rounds = int(_select_first(config, "data.n_branch_rounds", default=max_search_per_tree))
    assert n_branch_rounds >= 0, f"n_branch_rounds must be >= 0, got {n_branch_rounds}"
    raw_snapshot_rounds = _select_first(config, "data.opts_avg_snapshot_rounds", default=[1, 3, 7, 15])
    if isinstance(raw_snapshot_rounds, str):
        raw_snapshot_rounds = raw_snapshot_rounds.strip("[]").replace(",", " ").split()
    elif isinstance(raw_snapshot_rounds, int):
        raw_snapshot_rounds = [raw_snapshot_rounds]
    opts_avg_snapshot_rounds = sorted({int(r) for r in raw_snapshot_rounds if int(r) >= 1})
    gamma = config.algorithm.gamma
    lam = config.algorithm.lam

    dataset = pd.read_parquet(data_path)
    total_samples = len(dataset)
    trees_target = trees_per_prompt * total_samples

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    from utils.reward_fn import compute_score
    from verl.workers.reward_manager.naive import NaiveRewardManager
    reward_fn = NaiveRewardManager(
        tokenizer=tokenizer,
        num_examine=0,
        compute_score=compute_score,
    )

    resource_pool = RayResourcePool(
        process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        use_gpu=True,
        max_colocate_count=2,
    )

    async_mode = rollout_config.get("mode", "sync") == "async"
    actor_worker_impl = AsyncActorRolloutRefWorker if async_mode else ActorRolloutRefWorker
    actor_rollout_cls = RayClassWithInitArgs(
        cls=ray.remote(actor_worker_impl),
        config=actor_worker_config,
        role="rollout",
    )
    critic_cls = RayClassWithInitArgs(cls=ray.remote(CriticWorker), config=config.critic)

    class_dict = {"actor_rollout": actor_rollout_cls, "critic": critic_cls}
    worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)

    wg_dict = RayWorkerGroup(
        resource_pool=resource_pool,
        ray_cls_with_init=worker_dict_cls,
        device_name=config.trainer.device,
    )
    spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
    wg = spawn_wg["actor_rollout"]
    critic_wg = spawn_wg["critic"]

    wg.init_model()
    critic_wg.init_model()

    async_rollout_manager = None
    if async_mode:
        from verl.experimental.agent_loop import AgentLoopManager

        async_rollout_manager = AgentLoopManager(config=config, worker_group=wg, rm_resource_pool=None)

    from torchdata.stateful_dataloader import StatefulDataLoader
    from verl.trainer.main_ppo import create_rl_dataset
    from verl.utils.dataset.rl_dataset import collate_fn as rl_collate_fn

    dataset_config = OmegaConf.create(OmegaConf.to_container(config.data, resolve=False))
    dataset_config.val_files = data_path
    dataset_config.max_prompt_length = prompt_length
    dataset_config.shuffle = False
    dataset_config.validation_shuffle = False

    rl_dataset = create_rl_dataset(
        data_path,
        dataset_config,
        tokenizer,
        processor=None,
        max_samples=dataset_config.get("val_max_samples", -1),
    )
    dataloader = StatefulDataLoader(
        dataset=rl_dataset,
        batch_size=batch_size,
        num_workers=dataset_config.get("dataloader_num_workers", 0),
        shuffle=False,
        drop_last=False,
        collate_fn=rl_collate_fn,
    )
    prompt_buffer = PromptBuffer(dataloader)
    if len(rl_dataset) != total_samples:
        raise ValueError(
            "PromptBuffer dataset size does not match the raw parquet size. "
            "This usually means the reused training data pipeline filtered or reordered samples. "
            f"prompt_buffer.total_samples={len(rl_dataset)}, total_samples={total_samples}"
        )

    idx_responses = defaultdict(list)
    idx_sample_counter = defaultdict(int)
    global_sample_counter = 0

    print(f"Starting RQ2 OPTS tree-count scaling: "
          f"trees_per_prompt={trees_per_prompt}, trees_target={trees_target}, "
          f"batch_size={batch_size}, n_branch_rounds={n_branch_rounds}, "
          f"reward_mode={REWARD_MODE}, otrc_baseline={OTRC_BASELINE_MODE}, "
          f"max_search_per_tree={max_search_per_tree}")

    global_batch = None
    next_states = {}
    search_count = {}
    max_otrc_scores = {}
    tree_search_state_by_uid = {}
    uid_to_dataset_idx = {}
    prompt_cursor = 0
    trees_opened = 0
    response_by_rid = {}
    tree_seq = {}
    opts_avg_snapshots = {r: [[] for _ in range(total_samples)] for r in opts_avg_snapshot_rounds}

    total_calls = 1 + n_branch_rounds
    for call_idx in range(total_calls):
        root_phase = call_idx == 0
        print(f"[call {call_idx + 1}/{total_calls}] Start. "
              f"phase={'root' if root_phase else 'branch'}, "
              f"trees_opened={trees_opened}/{trees_target}, "
              f"continuations={len(next_states)}.")

        if root_phase:
            batch = prompt_buffer.draw(trees_target)
            for uid in batch.non_tensor_batch["uid"]:
                uid_to_dataset_idx[uid] = prompt_cursor
                prompt_cursor = (prompt_cursor + 1) % total_samples
                tree_seq[str(uid)] = len(tree_seq)
            trees_opened += len(batch.non_tensor_batch["uid"])
        else:
            if not next_states:
                print(f"[call {call_idx + 1}/{total_calls}] No OTRC candidates left; stopping early.")
                break
            batch = prepare_next_round_input(
                global_batch=global_batch,
                next_states=next_states,
                pad_token_id=tokenizer.pad_token_id,
            )
        batch.meta_info["temperature"] = rollout_config.temperature

        if async_mode:
            size_divisor = rollout_config.agent.num_workers
            batch_padded, pad_size = pad_dataproto_to_divisor(batch, size_divisor)
            output = unpad_dataproto(
                async_rollout_manager.generate_sequences(
                    batch_padded, sleep_after=(call_idx == total_calls - 1)
                ),
                pad_size=pad_size,
            )
            # AgentLoopManager rebuilds non_tensor_batch; keep input metadata order.
            for key in ("uid", "raw_prompt_len", "data_source", "reward_model", "extra_info", "raw_prompt"):
                if key in batch.non_tensor_batch:
                    output.non_tensor_batch[key] = batch.non_tensor_batch[key]
        else:
            batch_padded, pad_size = pad_dataproto_to_divisor(batch, wg.world_size)
            output_padded = wg.generate_sequences(batch_padded)
            output = unpad_dataproto(output_padded, pad_size=pad_size)

        if "response_mask" not in output.batch.keys():
            output.batch["response_mask"] = compute_response_mask(output)

        values_input_padded, values_pad_size = pad_dataproto_to_divisor(output, critic_wg.world_size)
        values_output_padded = critic_wg.compute_values(values_input_padded)
        values_output = unpad_dataproto(values_output_padded, pad_size=values_pad_size)
        output = output.union(values_output)

        full_response_strs = decode_response_strs(output, tokenizer, prompt_length, response_length)
        if "extra_info" not in output.non_tensor_batch:
            output.non_tensor_batch["extra_info"] = np.array(
                [{} for _ in range(len(full_response_strs))], dtype=object)
        for i, s in enumerate(full_response_strs):
            output.non_tensor_batch["extra_info"][i]["full_response_str"] = s
        reward_tensor, _ = compute_reward(output, reward_fn)
        output.batch["token_level_rewards"] = reward_tensor

        new_sample_indices = set_opts_ttpo_info(output, global_batch, next_states, call_idx)
        output.non_tensor_batch["episodic_returns"] = compute_episodic_returns(output, global_batch)
        cur_batch_size, cur_response_len = output.batch["responses"].shape
        output.batch["state_branches"] = torch.ones(cur_batch_size, cur_response_len)
        output.batch["advantages"] = torch.zeros(cur_batch_size, cur_response_len)
        output.batch["returns"] = torch.zeros(cur_batch_size, cur_response_len)

        if global_batch is None:
            global_batch = output
        else:
            global_batch = merge_batches(global_batch, output)

        advantages, returns = compute_treegae_advantage_return(
            token_level_rewards=global_batch.batch["token_level_rewards"],
            values=global_batch.batch["values"],
            response_mask=global_batch.batch["response_mask"],
            attention_mask=global_batch.batch["attention_mask"],
            gamma=gamma,
            lam=lam,
            rid=list(global_batch.non_tensor_batch["rid"]),
            pid=list(global_batch.non_tensor_batch["pid"]),
            branch_pos=list(global_batch.non_tensor_batch["branch_pos"]),
            cid=list(global_batch.non_tensor_batch["cid"]),
            new_sample_indices=new_sample_indices,
            raw_prompt_len=global_batch.non_tensor_batch["raw_prompt_len"],
            max_prompt_len=global_batch.batch["attention_mask"].shape[1] - global_batch.batch["response_mask"].shape[1],
            advantages=global_batch.batch["advantages"],
        )
        global_batch.batch["advantages"] = advantages
        global_batch.batch["returns"] = returns

        response_strs = decode_response_strs(output, tokenizer, prompt_length, response_length)
        output_uids = output.non_tensor_batch["uid"]
        current_rids = output.non_tensor_batch["rid"]
        current_pids = output.non_tensor_batch["pid"]
        current_branch_pos = output.non_tensor_batch["branch_pos"]

        for local_idx, (resp, uid) in enumerate(zip(response_strs, output_uids)):
            dataset_idx = uid_to_dataset_idx[uid]
            idx_sample_counter[dataset_idx] += 1
            global_sample_counter += 1
            pid = current_pids[local_idx]
            rid_str = str(current_rids[local_idx])
            response_by_rid[rid_str] = resp
            idx_responses[dataset_idx].append({
                "response": resp,
                "uid": str(uid),
                "sample_index": idx_sample_counter[dataset_idx],
                "global_index": global_sample_counter,
                "rid": rid_str,
                "pid": None if pid is None else str(pid),
                "branch_pos": int(current_branch_pos[local_idx]),
            })

        refresh_tree_search_states(
            batch=global_batch,
            affected_uids=set(output_uids),
            tree_search_state_by_uid=tree_search_state_by_uid,
            gamma=gamma,
            max_prompt_length=prompt_length,
            tokenizer=tokenizer,
            round_idx=call_idx,
        )

        # States persist across calls, so unbranched trees keep their latest
        # terminal — the snapshot is exactly the "as of this round" view.
        if call_idx in opts_avg_snapshots:
            snapshot_entries = defaultdict(list)
            for uid, state in tree_search_state_by_uid.items():
                dataset_idx = uid_to_dataset_idx[uid]
                snapshot_entries[dataset_idx].append((tree_seq[uid], response_by_rid[state.terminal_rid]))
            for dataset_idx, entries in snapshot_entries.items():
                assert len(entries) == trees_per_prompt, (
                    f"snapshot s{call_idx}: prompt {dataset_idx} has {len(entries)} trees, "
                    f"expected {trees_per_prompt}."
                )
                entries.sort(key=lambda item: item[0])
                opts_avg_snapshots[call_idx][dataset_idx] = [response for _, response in entries]

        print(f"[call {call_idx + 1}/{total_calls}] Done. "
              f"Collected {sum(len(v) for v in idx_responses.values())} total responses, "
              f"{trees_opened}/{trees_target} trees opened.")

        # search_batch_size=trees_target: every qualifying tree may branch once per round.
        if call_idx < total_calls - 1:
            selected_states = select_next_states(
                batch=global_batch,
                search_count=search_count,
                max_otrc_scores=max_otrc_scores,
                max_search_per_tree=max_search_per_tree,
                tree_search_state_by_uid=tree_search_state_by_uid,
                max_searched_tree_ratio=1.0,
                search_batch_size=trees_target,
                otrc_baseline_mode=OTRC_BASELINE_MODE,
            )
            next_states = selected_to_branch_points(selected_states, global_batch)

    assert trees_opened == trees_target, (
        f"Tree-count invariant violated: opened {trees_opened}, target {trees_target}."
    )

    final_advantages_by_rid = {}
    final_advantages = global_batch.batch["advantages"].detach().cpu()
    final_response_mask = global_batch.batch["response_mask"].detach().cpu().bool()
    for rid, advantages, valid_mask in zip(
        global_batch.non_tensor_batch["rid"],
        final_advantages,
        final_response_mask,
    ):
        final_advantages_by_rid[str(rid)] = [float(x) for x in advantages[valid_mask].tolist()]

    output_responses = [[] for _ in range(total_samples)]
    output_sample_indices = [[] for _ in range(total_samples)]
    output_global_indices = [[] for _ in range(total_samples)]
    output_tree_uids = [[] for _ in range(total_samples)]
    output_tree_rids = [[] for _ in range(total_samples)]
    output_tree_pids = [[] for _ in range(total_samples)]
    output_tree_branch_pos = [[] for _ in range(total_samples)]
    output_tree_advantages = [[] for _ in range(total_samples)]

    for dataset_idx, resp_list in idx_responses.items():
        for record in resp_list:
            output_responses[dataset_idx].append(record["response"])
            output_sample_indices[dataset_idx].append(record["sample_index"])
            output_global_indices[dataset_idx].append(record["global_index"])
            output_tree_uids[dataset_idx].append(record["uid"])
            output_tree_rids[dataset_idx].append(record["rid"])
            output_tree_pids[dataset_idx].append(record["pid"])
            output_tree_branch_pos[dataset_idx].append(record["branch_pos"])
            output_tree_advantages[dataset_idx].append(final_advantages_by_rid[record["rid"]])

    dataset["responses"] = output_responses
    dataset["sample_indices"] = output_sample_indices
    dataset["global_indices"] = output_global_indices
    dataset["tree_uids"] = output_tree_uids
    dataset["tree_rids"] = output_tree_rids
    dataset["tree_pids"] = output_tree_pids
    dataset["tree_branch_pos"] = output_tree_branch_pos
    dataset["tree_advantages"] = output_tree_advantages
    for r, snapshots in opts_avg_snapshots.items():
        if any(len(s) > 0 for s in snapshots):
            dataset[f"opts_avg_responses_s{r}"] = snapshots
    dataset["opts_reward_mode"] = REWARD_MODE

    output_dir = os.path.dirname(output_path)
    makedirs(output_dir, exist_ok=True)
    dataset.to_parquet(output_path)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
