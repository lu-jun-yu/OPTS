# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Generate responses given a dataset of prompts
"""

import os
import uuid

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
)
from .opts_ttpo.ray_trainer import (
    compute_response_mask,
    merge_batches,
    prepare_next_round_input,
    set_opts_ttpo_info,
    select_next_states,
)


@hydra.main(config_path="pkg://verl.trainer.config", config_name="generation", version_base=None)
def main(config):
    run_generation(config)


def run_generation(config) -> None:
    if not ray.is_initialized():
        # this is for local ray cluster
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

    if config.rollout.temperature == 0.0:
        assert config.data.n_samples == 1, "When temperature=0, n_samples must be 1."
    assert config.data.n_samples >= 1, "n_samples should always >= 1"

    # read dataset
    dataset = pd.read_parquet(config.data.path)
    chat_lst = dataset[config.data.prompt_key].tolist()
    chat_lst = [chat.tolist() for chat in chat_lst]

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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

    total_samples = len(dataset)
    config_batch_size = config.data.batch_size
    apply_chat_template_kwargs = config.data.get("apply_chat_template_kwargs", {})
    num_batch = -(-total_samples // config_batch_size)
    output_lst = [[] for _ in range(total_samples)]

    for batch_idx in range(num_batch):
        print(f"[{batch_idx + 1}/{num_batch}] Start to process.")
        batch_chat_lst = chat_lst[batch_idx * config_batch_size : (batch_idx + 1) * config_batch_size]
        actual_batch_size = len(batch_chat_lst)
        inputs = tokenizer.apply_chat_template(
            batch_chat_lst,
            add_generation_prompt=True,
            padding=True,
            truncation=True,
            max_length=config.rollout.prompt_length,
            return_tensors="pt",
            return_dict=True,
            tokenize=True,
            **apply_chat_template_kwargs,
        )
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        position_ids = compute_position_id_with_mask(attention_mask)
        batch_dict = {"input_ids": input_ids, "attention_mask": attention_mask, "position_ids": position_ids}

        batch: DataProto = DataProto.from_dict(batch_dict)
        raw_prompt_lens = batch.batch["attention_mask"].sum(dim=1).cpu().numpy()
        batch.non_tensor_batch["raw_prompt_len"] = raw_prompt_lens
        batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(actual_batch_size)], dtype=object)
        global_batch = None
        next_states = {}

        # START MULTI-ROUND GENERATION
        print(f"[{batch_idx + 1}/{num_batch}] Start to generate.")
        for round_idx in range(config.data.n_rounds):
            batch.meta_info["round_idx"] = round_idx
            batch_repeated = batch.repeat(repeat_times=config.data.n_samples, interleave=True)
            batch_repeated_padded, pad_size = pad_dataproto_to_divisor(batch_repeated, wg.world_size)

            output_padded = wg.generate_sequences(batch_repeated_padded)
            output = unpad_dataproto(output_padded, pad_size=pad_size)

            if "response_mask" not in output.batch.keys():
                output.batch["response_mask"] = compute_response_mask(output)

            # Compute values using critic model
            values_output = critic_wg.compute_values(output)
            output = output.union(values_output)

            # Compute token-level rewards from values: use last valid position's value as reward
            response_mask = output.batch["response_mask"]
            values = output.batch["values"]
            last_pos = (response_mask.sum(dim=1) - 1).clamp(min=0).long()
            token_level_rewards = torch.zeros_like(values)
            batch_indices = torch.arange(values.size(0), device=values.device)
            token_level_rewards[batch_indices, last_pos] = values[batch_indices, last_pos]
            output.batch["token_level_rewards"] = token_level_rewards

            new_sample_indices = set_opts_ttpo_info(output, global_batch, next_states, round_idx)
            output.non_tensor_batch["new_sample_indices"] = new_sample_indices
            batch_size, response_len = output.batch["responses"].shape
            output.batch["state_branches"] = torch.ones(batch_size, response_len)
            output.batch["advantages"] = torch.zeros(batch_size, response_len)
            output.batch["returns"] = torch.zeros(batch_size, response_len)
            if global_batch is None:
                global_batch = output
            else:
                global_batch = merge_batches(global_batch, output)

            # compute advantages
            advantages, returns = compute_treegae_advantage_return(
                token_level_rewards=global_batch.batch["token_level_rewards"],
                values=global_batch.batch["values"],
                response_mask=global_batch.batch["response_mask"],
                gamma=config.algorithm.gamma,
                lam=config.algorithm.lam,
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

            for i in range(batch_size):
                data_item = output[i]
                prompt_length = int(data_item.non_tensor_batch["raw_prompt_len"])
                valid_response_length = int(data_item.batch["attention_mask"][prompt_length:].sum().item())
                valid_response_ids = data_item.batch["input_ids"][prompt_length: prompt_length + valid_response_length]
                response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)

                idx = batch_idx * config_batch_size + i // config.data.n_samples
                output_lst[idx].append(response_str)

            if round_idx < config.data.n_rounds - 1:
                next_states = select_next_states(
                    batch=global_batch,
                    root_tuct=config.rollout.root_tuct,
                    round_idx=round_idx,
                    n_samples_per_round=config.data.n_samples,
                    max_prompt_length=config.rollout.prompt_length,
                )
                batch = prepare_next_round_input(
                    global_batch=global_batch,
                    next_states=next_states,
                )

    # add to the data frame
    dataset["responses"] = output_lst

    # write to a new parquet
    output_dir = os.path.dirname(config.data.output_path)
    makedirs(output_dir, exist_ok=True)
    dataset.to_parquet(config.data.output_path)


if __name__ == "__main__":
    main()
