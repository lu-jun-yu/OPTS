# Copyright 2025 Junyu Lu (Julian Lou). All rights reserved.

"""
Inference-time scaling/search experiment for OPTS.

Two modes:
  - reward-guided (reward_mode="reward"): uses actual reward_fn for OTRC guidance
  - value-guided  (reward_mode="value"):  uses critic's last-position value as reward (bounded if critic.value_head_activation=sigmoid)

Total inference budget = dataset_size * n_samples responses, matching pass@k
cost. Each run performs n_samples rounds of tree-structured sampling with
OTRC-based search over the full evaluation set.

Results include sample_indices and global_indices. reward-mode opts@k should
truncate by global_indices <= k * dataset_size. value-mode opts@k uses saved
online greedy-path response snapshots such as value_opts_responses_k32.
"""

import os
import sys
from collections import defaultdict

# Add LLM directory to Python path for correct imports
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
from trainer.opts_ttpo.ray_trainer_exp1 import (
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
    requested_batch_size = _require_config_value(config, "data.batch_size", "data.val_batch_size", "data.train_batch_size")
    n_samples = _require_config_value(config, "data.n_samples")
    output_path = _require_config_value(config, "data.output_path")
    reward_mode = _select_first(config, "data.reward_mode", default="value")
    raw_opts_snapshot_ks = _select_first(config, "data.opts_snapshot_ks", default=[8, 16, 32, 64, 128])
    if isinstance(raw_opts_snapshot_ks, str):
        raw_opts_snapshot_ks = raw_opts_snapshot_ks.strip("[]").replace(",", " ").split()
    elif isinstance(raw_opts_snapshot_ks, int):
        raw_opts_snapshot_ks = [raw_opts_snapshot_ks]
    opts_snapshot_ks = sorted({int(k) for k in raw_opts_snapshot_ks if int(k) >= 1})
    trust_remote_code = _select_first(
        config,
        "actor_rollout_ref.model.trust_remote_code",
        "model.trust_remote_code",
        "data.trust_remote_code",
        default=False,
    )

    local_path = copy_to_local(model_path)
    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)

    assert n_samples >= 1, "n_samples should always >= 1"
    if rollout_config.temperature == 0.0:
        assert n_samples == 1, "When temperature=0, n_samples must be 1."

    assert reward_mode in ("reward", "value"), f"reward_mode must be 'reward' or 'value', got {reward_mode}"

    prompt_length = rollout_config.prompt_length
    response_length = rollout_config.response_length
    max_search_per_tree = rollout_config.get("max_search_per_tree", 1)
    gamma = config.algorithm.gamma
    lam = config.algorithm.lam

    # Read dataset
    dataset = pd.read_parquet(data_path)
    total_samples = len(dataset)

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Setup reward_fn for reward-guided mode
    reward_fn = None
    if reward_mode == "reward":
        from utils.reward_fn import compute_score
        from verl.workers.reward_manager.naive import NaiveRewardManager
        reward_fn = NaiveRewardManager(
            tokenizer=tokenizer,
            num_examine=0,
            compute_score=compute_score,
        )

    # Create resource pool with colocated workers
    resource_pool = RayResourcePool(
        process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        use_gpu=True,
        max_colocate_count=2,
    )

    # Create actor rollout and critic workers. In async mode the OPTS continuation
    # path runs inside the agent loop, driven by AgentLoopManager.
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

    if requested_batch_size < total_samples:
        raise ValueError(
            "OPTS evaluation expects one round to cover the full evaluation set. "
            f"Got batch_size={requested_batch_size} but dataset_size={total_samples}. "
            "Increase data.val_batch_size/data.batch_size or split the dataset explicitly."
        )

    effective_batch_size = min(requested_batch_size, total_samples)

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
        batch_size=effective_batch_size,
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

    # Result collection: dataset_idx -> list of response records.
    idx_responses = defaultdict(list)
    # Sample counter per dataset row (per-prompt, 1-based).
    idx_sample_counter = defaultdict(int)
    # Global counter across the entire run (1-based). Enables reconstructing the
    # chronological order for opts@k, where the OPTS run is truncated to budget
    # k * dataset_size responses and then grouped by prompt.
    global_sample_counter = 0

    print(f"Starting inference-time search: "
          f"n_samples={n_samples}, batch_size={effective_batch_size}, "
          f"requested_batch_size={requested_batch_size}, "
          f"reward_mode={reward_mode}, max_search_per_tree={max_search_per_tree}, "
          f"opts_snapshot_ks={opts_snapshot_ks}")

    global_batch = None
    next_states = {}
    search_count = {}
    max_otrc_scores = {}
    tree_search_state_by_uid = {}
    uid_to_dataset_idx = {}
    prompt_cursor = 0
    response_by_rid = {}
    global_index_by_rid = {}
    value_opts_snapshots = {k: [[] for _ in range(total_samples)] for k in opts_snapshot_ks}

    for round_idx in range(n_samples):
        print(f"[round {round_idx + 1}/{n_samples}] Start.")
        # === Construct this round's batch ===
        if round_idx == 0:
            batch = prompt_buffer.draw(effective_batch_size)
            for uid in batch.non_tensor_batch["uid"]:
                uid_to_dataset_idx[uid] = prompt_cursor
                prompt_cursor = (prompt_cursor + 1) % total_samples
            batch.meta_info["temperature"] = rollout_config.temperature
        else:
            k = len(next_states)
            parts = []
            if k > 0:
                continued = prepare_next_round_input(
                    global_batch=global_batch,
                    next_states=next_states,
                    pad_token_id=tokenizer.pad_token_id,
                )
                parts.append(continued)
            remaining = effective_batch_size - k
            if remaining > 0:
                new_prompts = prompt_buffer.draw(remaining)
                for uid in new_prompts.non_tensor_batch["uid"]:
                    uid_to_dataset_idx[uid] = prompt_cursor
                    prompt_cursor = (prompt_cursor + 1) % total_samples
                parts.append(new_prompts)

            if len(parts) == 2:
                batch = merge_batches(parts[0], parts[1])
            elif len(parts) == 1:
                batch = parts[0]
            batch.meta_info["temperature"] = rollout_config.temperature

        # === Generate sequences ===
        if async_mode:
            size_divisor = rollout_config.agent.num_workers
            batch_padded, pad_size = pad_dataproto_to_divisor(batch, size_divisor)
            output = unpad_dataproto(
                async_rollout_manager.generate_sequences(batch_padded), pad_size=pad_size
            )
            # AgentLoopManager rebuilds non_tensor_batch; downstream tree
            # bookkeeping uses the input metadata order.
            for key in ("uid", "raw_prompt_len", "data_source", "reward_model", "extra_info", "raw_prompt"):
                if key in batch.non_tensor_batch:
                    output.non_tensor_batch[key] = batch.non_tensor_batch[key]
        else:
            batch_padded, pad_size = pad_dataproto_to_divisor(batch, wg.world_size)
            output_padded = wg.generate_sequences(batch_padded)
            output = unpad_dataproto(output_padded, pad_size=pad_size)

        if "response_mask" not in output.batch.keys():
            output.batch["response_mask"] = compute_response_mask(output)

        # === Compute values ===
        values_input_padded, values_pad_size = pad_dataproto_to_divisor(output, critic_wg.world_size)
        values_output_padded = critic_wg.compute_values(values_input_padded)
        values_output = unpad_dataproto(values_output_padded, pad_size=values_pad_size)
        output = output.union(values_output)

        # === Compute rewards ===
        if reward_mode == "reward":
            # Decode full response into extra_info, then compute reward via reward_fn
            full_response_strs = decode_response_strs(output, tokenizer, prompt_length, response_length)
            if "extra_info" not in output.non_tensor_batch:
                output.non_tensor_batch["extra_info"] = np.array(
                    [{} for _ in range(len(full_response_strs))], dtype=object)
            for i, s in enumerate(full_response_strs):
                output.non_tensor_batch["extra_info"][i]["full_response_str"] = s
            reward_tensor, _ = compute_reward(output, reward_fn)
            output.batch["token_level_rewards"] = reward_tensor
        else:
            # Value-guided: use last valid position's value as reward
            response_mask = output.batch["response_mask"]
            values = output.batch["values"]
            last_pos = (response_mask.sum(dim=1) - 1).clamp(min=0).long()
            token_level_rewards = torch.zeros_like(values)
            batch_indices = torch.arange(values.size(0), device=values.device)
            token_level_rewards[batch_indices, last_pos] = values[batch_indices, last_pos]
            output.batch["token_level_rewards"] = token_level_rewards

        # === Tree structure bookkeeping ===
        new_sample_indices = set_opts_ttpo_info(output, global_batch, next_states, round_idx)
        output.non_tensor_batch["episodic_returns"] = compute_episodic_returns(output, global_batch)
        cur_batch_size, cur_response_len = output.batch["responses"].shape
        output.batch["state_branches"] = torch.ones(cur_batch_size, cur_response_len)
        output.batch["advantages"] = torch.zeros(cur_batch_size, cur_response_len)
        output.batch["returns"] = torch.zeros(cur_batch_size, cur_response_len)

        if global_batch is None:
            global_batch = output
        else:
            global_batch = merge_batches(global_batch, output)

        # === Compute TreeGAE advantages ===
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
            state_branches=global_batch.batch["state_branches"],
            new_sample_indices=new_sample_indices,
            raw_prompt_len=global_batch.non_tensor_batch["raw_prompt_len"],
            max_prompt_len=global_batch.batch["attention_mask"].shape[1] - global_batch.batch["response_mask"].shape[1],
            advantages=global_batch.batch["advantages"],
        )
        global_batch.batch["advantages"] = advantages
        global_batch.batch["returns"] = returns

        # === Collect decoded responses with sample_index + global_index ===
        response_strs = decode_response_strs(output, tokenizer, prompt_length, response_length)
        output_uids = output.non_tensor_batch["uid"]
        current_rids = output.non_tensor_batch["rid"]
        current_pids = output.non_tensor_batch["pid"]
        current_branch_pos = output.non_tensor_batch["branch_pos"]
        indexed_rids = global_batch.non_tensor_batch["rid"][new_sample_indices]
        if not (len(response_strs) == len(output_uids) == len(current_rids)):
            raise RuntimeError(
                f"Round {round_idx}: inconsistent output lengths: "
                f"responses={len(response_strs)}, uids={len(output_uids)}, rids={len(current_rids)}"
            )
        indexed_rid_strs = [str(rid) for rid in indexed_rids]
        current_rid_strs = [str(rid) for rid in current_rids]
        if indexed_rid_strs != current_rid_strs:
            mismatches = [
                (i, indexed_rid_strs[i], current_rid_strs[i])
                for i in range(min(len(indexed_rid_strs), len(current_rid_strs)))
                if indexed_rid_strs[i] != current_rid_strs[i]
            ]
            raise RuntimeError(
                f"Round {round_idx}: global_batch[new_sample_indices] rids do not match output rids. "
                f"indexed_len={len(indexed_rid_strs)}, output_len={len(current_rid_strs)}, "
                f"first_mismatches={mismatches[:5]}"
            )

        for local_idx, (resp, uid) in enumerate(zip(response_strs, output_uids)):
            dataset_idx = uid_to_dataset_idx[uid]
            idx_sample_counter[dataset_idx] += 1
            global_sample_counter += 1
            pid = current_pids[local_idx]
            rid_str = str(current_rids[local_idx])
            response_by_rid[rid_str] = resp
            global_index_by_rid[rid_str] = global_sample_counter
            idx_responses[dataset_idx].append({
                "response": resp,
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
            round_idx=round_idx,
        )

        snapshot_k = round_idx + 1
        if reward_mode == "value" and snapshot_k in value_opts_snapshots:
            snapshot_entries = defaultdict(list)
            for uid, state in tree_search_state_by_uid.items():
                dataset_idx = uid_to_dataset_idx[uid]
                response = response_by_rid[state.terminal_rid]
                global_index = global_index_by_rid[state.terminal_rid]
                snapshot_entries[dataset_idx].append((global_index, response, state.terminal_rid))
            for dataset_idx, entries in snapshot_entries.items():
                entries.sort(key=lambda item: item[0])
                value_opts_snapshots[snapshot_k][dataset_idx] = [response for _, response, _ in entries]

        # === OTRC selection for next round ===
        if round_idx < n_samples - 1:
            selected_states = select_next_states(
                batch=global_batch,
                search_count=search_count,
                max_otrc_scores=max_otrc_scores,
                max_search_per_tree=max_search_per_tree,
                tree_search_state_by_uid=tree_search_state_by_uid,
                gamma=gamma,
                max_prompt_length=prompt_length,
                batch_size=effective_batch_size,
                tokenizer=tokenizer,
            )
            next_states = selected_to_branch_points(selected_states, global_batch)

        print(f"[round {round_idx + 1}/{n_samples}] Done. "
              f"Collected {sum(len(v) for v in idx_responses.values())} total responses.")

    expected_response_count = total_samples * n_samples
    if global_sample_counter != expected_response_count:
        raise RuntimeError(
            f"Unexpected OPTS generation count: got {global_sample_counter}, "
            f"expected {expected_response_count} (= dataset_size * n_samples)."
        )

    # === Assemble output ===
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
    output_tree_rids = [[] for _ in range(total_samples)]
    output_tree_pids = [[] for _ in range(total_samples)]
    output_tree_branch_pos = [[] for _ in range(total_samples)]
    output_tree_advantages = [[] for _ in range(total_samples)]

    for dataset_idx, resp_list in idx_responses.items():
        for record in resp_list:
            output_responses[dataset_idx].append(record["response"])
            output_sample_indices[dataset_idx].append(record["sample_index"])
            output_global_indices[dataset_idx].append(record["global_index"])
            output_tree_rids[dataset_idx].append(record["rid"])
            output_tree_pids[dataset_idx].append(record["pid"])
            output_tree_branch_pos[dataset_idx].append(record["branch_pos"])
            output_tree_advantages[dataset_idx].append(final_advantages_by_rid.get(record["rid"], []))

    dataset["responses"] = output_responses
    dataset["sample_indices"] = output_sample_indices
    dataset["global_indices"] = output_global_indices
    dataset["tree_rids"] = output_tree_rids
    dataset["tree_pids"] = output_tree_pids
    dataset["tree_branch_pos"] = output_tree_branch_pos
    dataset["tree_advantages"] = output_tree_advantages
    for k in opts_snapshot_ks:
        dataset[f"value_opts_responses_k{k}"] = value_opts_snapshots[k]
    dataset["opts_reward_mode"] = reward_mode

    # Write output
    output_dir = os.path.dirname(output_path)
    makedirs(output_dir, exist_ok=True)
    dataset.to_parquet(output_path)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
