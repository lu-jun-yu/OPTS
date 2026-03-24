# Copyright 2025 Junyu Lu (Julian Lou). All rights reserved.

"""
Inference-time scaling/search experiment for OPTS.

Two modes:
  - reward-guided (reward_mode="reward"): uses actual reward_fn for TUCT guidance
  - value-guided  (reward_mode="value"):  uses critic's last-position value as reward

Total inference budget = dataset_size * n_samples / batch_size steps,
matching pass@k cost. Each step performs n_samples rounds of tree-structured
sampling with TUCT-based search.

Results include sample_index per response for scaling analysis:
  filter sample_index <= k to get n_samples=k results.
"""

import os
import uuid
from collections import defaultdict

import hydra
import numpy as np
import ray

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

from pprint import pprint

import torch
import pandas as pd
from omegaconf import OmegaConf

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.hdfs_io import makedirs
from verl.utils.model import compute_position_id_with_mask
from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker

from .opts_ttpo.core_algos import (
    compute_treegae_advantage_return,
    compute_branch_weight,
)
from .opts_ttpo.ray_trainer import (
    compute_aggregated_returns,
    compute_episodic_returns,
    compute_response_mask,
    merge_batches,
    prepare_next_round_input,
    set_opts_ttpo_info,
    select_next_states,
    selected_to_branch_points,
)
from verl.trainer.ppo.reward import compute_reward


class InferencePromptBuffer:
    """Buffer that draws prompts from a dataset, cycling through all prompts.

    Each draw assigns a fresh uid but tracks which original dataset row it came from,
    enabling result aggregation by prompt. Fresh uids are needed because the same
    prompt may appear multiple times in the same step's global_batch.
    """

    def __init__(self, dataset, tokenizer, prompt_key, prompt_length, apply_chat_template_kwargs=None):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.prompt_key = prompt_key
        self.prompt_length = prompt_length
        self.apply_chat_template_kwargs = apply_chat_template_kwargs or {}
        self.total_samples = len(dataset)

        # Track dataset fields for reward computation
        self.data_sources = dataset["data_source"].tolist() if "data_source" in dataset.columns else [None] * self.total_samples
        self.reward_models = dataset["reward_model"].tolist() if "reward_model" in dataset.columns else [None] * self.total_samples

        # Cursor and cycling
        self.cursor = 0
        self.has_cycled = False
        self._just_cycled = False  # set True on the draw that causes cycling

        # Map uid -> original dataset index (populated on each draw)
        self.uid_to_idx = {}

    def draw(self, n: int) -> DataProto:
        """Draw n prompts with fresh uids. Cycles through the dataset."""
        self._just_cycled = False
        indices = []
        for _ in range(n):
            indices.append(self.cursor)
            self.cursor += 1
            if self.cursor >= self.total_samples:
                self.cursor = 0
                if not self.has_cycled:
                    self.has_cycled = True
                    self._just_cycled = True

        # Assign fresh uids and record mapping
        uids = [str(uuid.uuid4()) for _ in indices]
        for uid, idx in zip(uids, indices):
            self.uid_to_idx[uid] = idx

        # Tokenize selected prompts
        chat_lst = self.dataset[self.prompt_key].tolist()
        batch_chats = [chat_lst[i].tolist() if hasattr(chat_lst[i], 'tolist') else chat_lst[i] for i in indices]

        inputs = self.tokenizer.apply_chat_template(
            batch_chats,
            add_generation_prompt=True,
            padding=True,
            truncation=True,
            max_length=self.prompt_length,
            return_tensors="pt",
            return_dict=True,
            tokenize=True,
            **self.apply_chat_template_kwargs,
        )

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        position_ids = compute_position_id_with_mask(attention_mask)

        batch = DataProto.from_dict({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        })

        raw_prompt_lens = attention_mask.sum(dim=1).cpu().numpy()
        batch.non_tensor_batch["raw_prompt_len"] = raw_prompt_lens
        batch.non_tensor_batch["uid"] = np.array(uids, dtype=object)

        # Set data_source and reward_model for reward computation
        batch.non_tensor_batch["data_source"] = np.array([self.data_sources[i] for i in indices], dtype=object)
        batch.non_tensor_batch["reward_model"] = np.array([self.reward_models[i] for i in indices], dtype=object)

        return batch

    @property
    def just_cycled(self):
        return self._just_cycled


def set_full_response_str(batch: DataProto, tokenizer, prompt_length: int, response_length: int):
    """Decode full response string and store in extra_info for reward computation."""
    batch_size = batch.batch["input_ids"].shape[0]

    if "extra_info" not in batch.non_tensor_batch:
        batch.non_tensor_batch["extra_info"] = np.array([{} for _ in range(batch_size)], dtype=object)

    for i in range(batch_size):
        raw_prompt_len = int(batch.non_tensor_batch["raw_prompt_len"][i])
        valid_prompt_len = int(batch.batch["attention_mask"][i, :prompt_length].sum().item())
        pad_len = prompt_length - valid_prompt_len
        start_pos = pad_len + raw_prompt_len
        end_pos = start_pos + response_length
        full_response_ids = batch.batch["input_ids"][i, start_pos:end_pos]
        full_response_str = tokenizer.decode(full_response_ids, skip_special_tokens=True)
        batch.non_tensor_batch["extra_info"][i]["full_response_str"] = full_response_str


def decode_responses(batch: DataProto, tokenizer) -> list[str]:
    """Decode responses from batch, returning list of response strings."""
    batch_size = batch.batch["input_ids"].shape[0]
    responses = []
    for i in range(batch_size):
        data_item = batch[i]
        prompt_length = int(data_item.non_tensor_batch["raw_prompt_len"])
        valid_response_length = int(data_item.batch["attention_mask"][prompt_length:].sum().item())
        valid_response_ids = data_item.batch["input_ids"][prompt_length: prompt_length + valid_response_length]
        response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        responses.append(response_str)
    return responses


@hydra.main(config_path="pkg://verl.trainer.config", config_name="generation", version_base=None)
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

    local_path = copy_to_local(config.model.path)
    trust_remote_code = config.data.get("trust_remote_code", False)
    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)

    n_samples = config.data.n_samples
    assert n_samples >= 1, "n_samples should always >= 1"
    if config.rollout.temperature == 0.0:
        assert n_samples == 1, "When temperature=0, n_samples must be 1."

    reward_mode = config.data.get("reward_mode", "value")
    assert reward_mode in ("reward", "value"), f"reward_mode must be 'reward' or 'value', got {reward_mode}"

    batch_size = config.data.batch_size
    prompt_length = config.rollout.prompt_length
    response_length = config.rollout.response_length
    c_tuct = config.rollout.get("c", 1.0)
    max_search_per_tree = config.rollout.get("max_search_per_tree", 1)
    gamma = config.algorithm.gamma
    lam = config.algorithm.lam

    # Read dataset
    dataset = pd.read_parquet(config.data.path)
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

    # Create actor rollout and critic workers
    actor_rollout_cls = RayClassWithInitArgs(cls=ray.remote(ActorRolloutRefWorker), config=config, role="rollout")
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

    # Build prompt buffer
    apply_chat_template_kwargs = config.data.get("apply_chat_template_kwargs", {})
    prompt_buffer = InferencePromptBuffer(
        dataset=dataset,
        tokenizer=tokenizer,
        prompt_key=config.data.prompt_key,
        prompt_length=prompt_length,
        apply_chat_template_kwargs=apply_chat_template_kwargs,
    )

    # Total steps = dataset_size * n_samples / batch_size (matching pass@k cost)
    total_steps = -(-total_samples * n_samples // batch_size)  # ceil division

    # Global state
    prev_mean_return = None
    all_returns = []

    # Result collection: dataset_idx -> list of (response_str, sample_index)
    idx_responses = defaultdict(list)
    # Sample counter per dataset row
    idx_sample_counter = defaultdict(int)

    print(f"Starting inference-time search: total_steps={total_steps}, "
          f"n_samples={n_samples}, batch_size={batch_size}, "
          f"reward_mode={reward_mode}, c={c_tuct}, max_search_per_tree={max_search_per_tree}")

    for step_idx in range(total_steps):
        print(f"[step {step_idx + 1}/{total_steps}] Start.")
        global_batch = None
        next_states = {}
        search_count = {}

        for round_idx in range(n_samples):
            # === Construct this round's batch ===
            if round_idx == 0:
                batch = prompt_buffer.draw(batch_size)
                batch.meta_info["temperature"] = config.rollout.temperature
            else:
                k = len(next_states)
                parts = []
                if k > 0:
                    continued = prepare_next_round_input(
                        global_batch=global_batch,
                        next_states=next_states,
                    )
                    parts.append(continued)
                remaining = batch_size - k
                if remaining > 0:
                    new_prompts = prompt_buffer.draw(remaining)
                    # new_prompts have fresh uids assigned by the buffer
                    parts.append(new_prompts)

                if len(parts) == 2:
                    batch = merge_batches(parts[0], parts[1])
                elif len(parts) == 1:
                    batch = parts[0]
                else:
                    break  # no data to process
                batch.meta_info["temperature"] = config.rollout.temperature

            # Check if buffer just cycled → compute return threshold
            if prompt_buffer.just_cycled and all_returns:
                prev_mean_return = sum(all_returns) / len(all_returns)
                print(f"[step {step_idx + 1}] Buffer cycled. prev_mean_return={prev_mean_return:.4f}")

            batch.meta_info["round_idx"] = round_idx

            # === Generate sequences ===
            batch_padded, pad_size = pad_dataproto_to_divisor(batch, wg.world_size)
            output_padded = wg.generate_sequences(batch_padded)
            output = unpad_dataproto(output_padded, pad_size=pad_size)

            if "response_mask" not in output.batch.keys():
                output.batch["response_mask"] = compute_response_mask(output)

            # === Compute values ===
            values_output = critic_wg.compute_values(output)
            output = output.union(values_output)

            # === Compute rewards ===
            if reward_mode == "reward":
                # Decode full response and compute reward via reward_fn
                set_full_response_str(output, tokenizer, prompt_length, response_length)
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
            output.non_tensor_batch["new_sample_indices"] = new_sample_indices
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
                gamma=gamma,
                lam=lam,
                rid=list(global_batch.non_tensor_batch["rid"]),
                pid=list(global_batch.non_tensor_batch["pid"]),
                cid=list(global_batch.non_tensor_batch["cid"]),
                state_branches=global_batch.batch["state_branches"],
                new_sample_indices=global_batch.non_tensor_batch["new_sample_indices"],
                next_states=next_states,
                advantages=global_batch.batch["advantages"],
            )
            global_batch.batch["advantages"] = advantages
            global_batch.batch["returns"] = returns

            # === Collect decoded responses with sample_index ===
            response_strs = decode_responses(output, tokenizer)
            output_uids = output.non_tensor_batch["uid"]
            for resp, uid in zip(response_strs, output_uids):
                dataset_idx = prompt_buffer.uid_to_idx.get(uid)
                if dataset_idx is not None:
                    idx_sample_counter[dataset_idx] += 1
                    idx_responses[dataset_idx].append((resp, idx_sample_counter[dataset_idx]))

            # === TUCT selection for next round ===
            if round_idx < n_samples - 1:
                selected_states = select_next_states(
                    batch=global_batch,
                    search_count=search_count,
                    max_search_per_tree=max_search_per_tree,
                    c=c_tuct,
                    gamma=gamma,
                    return_threshold=prev_mean_return,
                    max_prompt_length=prompt_length,
                    batch_size=batch_size,
                    tokenizer=tokenizer,
                )
                next_states = selected_to_branch_points(selected_states, global_batch)

        # === Post-step: compute aggregated returns for threshold ===
        if global_batch is not None:
            branch_weight = compute_branch_weight(
                state_branches=global_batch.batch["state_branches"],
                pid=global_batch.non_tensor_batch["pid"],
                rid=global_batch.non_tensor_batch["rid"],
                uid=global_batch.non_tensor_batch["uid"],
                branch_pos=global_batch.non_tensor_batch["branch_pos"],
            )
            global_batch.batch["branch_weight"] = branch_weight
            step_returns = compute_aggregated_returns(global_batch)
            all_returns.extend(step_returns)

        print(f"[step {step_idx + 1}/{total_steps}] Done. "
              f"Collected {sum(len(v) for v in idx_responses.values())} total responses.")

    # === Assemble output ===
    output_responses = [[] for _ in range(total_samples)]
    output_sample_indices = [[] for _ in range(total_samples)]

    for dataset_idx, resp_list in idx_responses.items():
        for resp_str, sample_idx in resp_list:
            output_responses[dataset_idx].append(resp_str)
            output_sample_indices[dataset_idx].append(sample_idx)

    dataset["responses"] = output_responses
    dataset["sample_indices"] = output_sample_indices

    # Write output
    output_dir = os.path.dirname(config.data.output_path)
    makedirs(output_dir, exist_ok=True)
    dataset.to_parquet(config.data.output_path)
    print(f"Results saved to {config.data.output_path}")


if __name__ == "__main__":
    main()
