# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import logging
import os
import time
import uuid
from collections import defaultdict, OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from pprint import pprint
from typing import Any, Optional, List, Tuple, Dict

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.import_utils import load_class_from_fqn
from verl.utils.metric import reduce_metrics
from verl.utils.py_functional import rename_dict
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.config import FSDPEngineConfig
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding

# Import RayPPOTrainer from verl for inheritance
from verl.trainer.ppo.ray_trainer import RayPPOTrainer

# Import OPTS_TTPO specific functions from local core_algos
from .core_algos import (
    AdvantageEstimator,
    agg_loss,
    compute_decay_factor,
    compute_partree_branches,
    compute_branch_weight_factors,
)


# Setup logging for batch tracking
logger_batch = logging.getLogger("opts_ttpo.batch_tracker")
logger_batch.setLevel(logging.DEBUG)
logger_batch.propagate = False  # Prevent duplicate logs from propagating to root logger
if not logger_batch.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] %(message)s')
    handler.setFormatter(formatter)
    logger_batch.addHandler(handler)


@contextmanager
def timed_block(name: str, step: int = -1, round_idx: int = -1):
    """Context manager to log the execution time of a code block.

    Args:
        name: Name of the code block being timed.
        step: Global training step number.
        round_idx: OPTS round index (for multi-round generation).
    """
    step_info = f"step={step}" if step >= 0 else ""
    round_info = f"round={round_idx}" if round_idx >= 0 else ""
    prefix = f"[{step_info}][{round_info}]" if step_info or round_info else ""

    logger_batch.info(f"{prefix}[{name}] >>> START")
    start_time = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start_time
        logger_batch.info(f"{prefix}[{name}] <<< END (elapsed: {elapsed:.3f}s)")


def log_batch_state(batch: DataProto, stage: str, step: int = -1, round_idx: int = -1) -> None:
    """Log the state of a batch for debugging purposes.

    Args:
        batch: The DataProto batch to log.
        stage: A string describing the current processing stage.
        step: Global training step number.
        round_idx: OPTS round index (for multi-round generation).
    """
    if round_idx == 0:
        return
    step_info = f"step={step}" if step >= 0 else ""
    round_info = f"round={round_idx}" if round_idx >= 0 else ""
    prefix = f"[{step_info}][{round_info}][{stage}]"

    # Basic batch info
    batch_size = batch.batch.batch_size[0] if hasattr(batch.batch, 'batch_size') else len(batch.batch.get('input_ids', []))

    logger_batch.info(f"{prefix} === Batch State ===")
    logger_batch.info(f"{prefix} batch_size: {batch_size}")

    # Tensor keys and shapes
    tensor_keys = list(batch.batch.keys())
    logger_batch.info(f"{prefix} tensor_keys: {tensor_keys}")
    for key in tensor_keys:
        tensor = batch.batch[key]
        if hasattr(tensor, 'shape'):
            logger_batch.info(f"{prefix}   {key}: shape={tensor.shape}, dtype={tensor.dtype}")
        else:
            logger_batch.info(f"{prefix}   {key}: type={type(tensor)}")

    # Non-tensor batch keys
    non_tensor_keys = list(batch.non_tensor_batch.keys()) if hasattr(batch, 'non_tensor_batch') else []
    logger_batch.info(f"{prefix} non_tensor_keys: {non_tensor_keys}")
    for key in non_tensor_keys:
        val = batch.non_tensor_batch[key]
        if isinstance(val, np.ndarray):
            logger_batch.info(f"{prefix}   {key}: shape={val.shape}, dtype={val.dtype}")
        else:
            logger_batch.info(f"{prefix}   {key}: type={type(val)}, len={len(val) if hasattr(val, '__len__') else 'N/A'}")

    # Meta info
    meta_keys = list(batch.meta_info.keys()) if hasattr(batch, 'meta_info') else []
    logger_batch.info(f"{prefix} meta_info_keys: {meta_keys}")

    # Key statistics for important tensors
    if 'attention_mask' in batch.batch:
        mask = batch.batch['attention_mask']
        seq_lens = mask.sum(dim=-1)
        logger_batch.info(f"{prefix} seq_lens: min={seq_lens.min().item()}, max={seq_lens.max().item()}, mean={seq_lens.float().mean().item():.1f}")

    if 'response_mask' in batch.batch:
        resp_mask = batch.batch['response_mask']
        resp_lens = resp_mask.sum(dim=-1)
        logger_batch.info(f"{prefix} response_lens: min={resp_lens.min().item()}, max={resp_lens.max().item()}, mean={resp_lens.float().mean().item():.1f}")

    if 'token_level_scores' in batch.batch:
        scores = batch.batch['token_level_scores']
        total_scores = scores.sum(dim=-1)
        logger_batch.info(f"{prefix} total_scores: min={total_scores.min().item():.4f}, max={total_scores.max().item():.4f}, mean={total_scores.mean().item():.4f}")

    if 'token_level_rewards' in batch.batch:
        rewards = batch.batch['token_level_rewards']
        total_rewards = rewards.sum(dim=-1)
        logger_batch.info(f"{prefix} total_rewards: min={total_rewards.min().item():.4f}, max={total_rewards.max().item():.4f}, mean={total_rewards.mean().item():.4f}")

    if 'advantages' in batch.batch:
        adv = batch.batch['advantages']
        logger_batch.info(f"{prefix} advantages: min={adv.min().item():.4f}, max={adv.max().item():.4f}, mean={adv.mean().item():.4f}, std={adv.std().item():.4f}")

    if 'values' in batch.batch:
        vals = batch.batch['values']
        logger_batch.info(f"{prefix} values: min={vals.min().item():.4f}, max={vals.max().item():.4f}, mean={vals.mean().item():.4f}")

    if 'old_log_probs' in batch.batch:
        old_lp = batch.batch['old_log_probs']
        logger_batch.info(f"{prefix} old_log_probs: min={old_lp.min().item():.4f}, max={old_lp.max().item():.4f}, mean={old_lp.mean().item():.4f}")

    if 'ref_log_prob' in batch.batch:
        ref_lp = batch.batch['ref_log_prob']
        logger_batch.info(f"{prefix} ref_log_prob: min={ref_lp.min().item():.4f}, max={ref_lp.max().item():.4f}, mean={ref_lp.mean().item():.4f}")

    logger_batch.info(f"{prefix} === End Batch State ===")


def log_sample_generations(batch: DataProto, tokenizer, step: int = 1, round_idx: int = 1, num_samples: int = 2) -> None:
    """Log decoded sample prompts and responses for debugging.

    Args:
        batch: The DataProto batch containing prompts and responses.
        tokenizer: Tokenizer for decoding token ids to text.
        step: Global training step number.
        round_idx: OPTS round index.
        num_samples: Number of samples to log.
    """
    if "prompts" not in batch.batch or "responses" not in batch.batch:
        return

    num_samples = min(num_samples, batch.batch["prompts"].shape[0])
    logger_batch.info(f"[step={step}][round={round_idx}] === Sample Generations ===")

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    for i in range(num_samples):
        prompt_ids = batch.batch["prompts"][i]
        response_ids = batch.batch["responses"][i]

        # Remove leading pads from prompt (left-padded)
        nonpad_mask = prompt_ids != pad_token_id
        if nonpad_mask.any():
            first_nonpad = nonpad_mask.nonzero()[0].item()
            prompt_ids = prompt_ids[first_nonpad:]
        else:
            prompt_ids = prompt_ids[:0]

        # Remove trailing pads from response (right-padded)
        nonpad_mask = response_ids != pad_token_id
        if nonpad_mask.any():
            last_nonpad = nonpad_mask.nonzero()[-1].item()
            response_ids = response_ids[: last_nonpad + 1]
        else:
            response_ids = response_ids[:0]

        prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=False)
        response_text = tokenizer.decode(response_ids, skip_special_tokens=False)
        # Truncate if too long
        if len(prompt_text) > 500:
            prompt_text = prompt_text[:250] + "...[truncated]..." + prompt_text[-250:]
        if len(response_text) > 500:
            response_text = response_text[:250] + "...[truncated]..." + response_text[-250:]
        logger_batch.info(f"[step={step}][round={round_idx}] Sample {i} PROMPT:\n{prompt_text}")
        logger_batch.info(f"[step={step}][round={round_idx}] Sample {i} RESPONSE:\n{response_text}")

    logger_batch.info(f"[step={step}][round={round_idx}] === End Sample Generations ===")


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create Ray resource pools for distributed training.

        Initializes resource pools based on the resource pool specification,
        with each pool managing GPU resources across multiple nodes.
        For FSDP backend, uses max_colocate_count=1 to merge WorkerGroups.
        For Megatron backend, uses max_colocate_count>1 for different models.
        """
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, using max_colocate_count=3: actor_critic_ref, rollout, reward model (optional)
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=3, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray._private.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
    next_states: Optional[Dict] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]

        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.TreeGAE:
        # TreeGAE for OPTS_TTPO: compute advantages on tree-structured trajectories
        from .core_algos import compute_treegae_advantage_return

        advantages, returns, advantages_mean = compute_treegae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
            rid=list(data.non_tensor_batch["rid"]),
            pid=list(data.non_tensor_batch["pid"]),
            cid=list(data.non_tensor_batch["cid"]),
            state_branches=data.batch["state_branches"],
            new_sample_indices=data.non_tensor_batch["new_sample_indices"],
            next_states=next_states,
            advantages=data.batch["advantages"],
            advantages_mean=data.batch["advantages_mean"],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        data.batch["advantages_mean"] = advantages_mean
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


def merge_batches(batch1: DataProto, batch2: DataProto) -> DataProto:
    """Merge two DataProto batches along batch dimension."""
    # Merge tensor batch
    batch1_keys = set(batch1.batch.keys())
    batch2_keys = set(batch2.batch.keys())
    common_keys = batch1_keys & batch2_keys
    merged_batch = {k: torch.cat([batch1.batch[k], batch2.batch[k]], dim=0)
                    for k in common_keys}

    # Merge non-tensor batch
    merged_non_tensor = {}
    for k in batch1.non_tensor_batch:
        if k not in batch2.non_tensor_batch:
            continue
        v1, v2 = batch1.non_tensor_batch[k], batch2.non_tensor_batch[k]
        merged_non_tensor[k] = np.concatenate([v1, v2], axis=0)

    result = DataProto.from_single_dict(merged_batch)
    result.non_tensor_batch = merged_non_tensor
    result.meta_info = batch1.meta_info.copy()
    return result


def prepare_next_round_input(
    global_batch: DataProto,
    next_states: Dict[str, Tuple[int, int]],
) -> DataProto:
    """Prepare input batch for next round based on selected states.

    Args:
        global_batch: Global batch containing all trajectories.
        next_states: Dict mapping uid to (parent_idx, branch_pos).
    """
    sel_indices = [idx for idx, _ in next_states.values()]
    sel_positions = [pos for _, pos in next_states.values()]

    # Compute valid prompt lengths
    prompt_len = global_batch.batch["input_ids"].shape[1] - global_batch.batch["responses"].shape[1]
    prompt_masks = global_batch.batch["attention_mask"][sel_indices, :prompt_len]
    valid_prompt_lens = prompt_masks.sum(dim=1).int()

    # Fill tensors: extract valid tokens only, left-pad
    # Each sample has different start/end indices, so we must loop
    bs = len(next_states)
    padded_ids = torch.zeros(bs, prompt_len, dtype=torch.long)
    padded_mask = torch.zeros(bs, prompt_len, dtype=torch.long)

    for i in range(bs):
        sel_idx = sel_indices[i]
        pos = sel_positions[i]
        valid_prompt_len = valid_prompt_lens[i].item()

        # Compute indices for this sample
        start_idx = prompt_len - valid_prompt_len
        end_idx = prompt_len + pos + 1
        valid_len = valid_prompt_len + pos + 1
        pad_l = prompt_len - valid_len

        # Copy valid tokens with left-padding
        padded_ids[i, pad_l:] = global_batch.batch["input_ids"][sel_idx, start_idx:end_idx]
        padded_mask[i, pad_l:] = 1

    from verl.utils.model import compute_position_id_with_mask
    padded_pos = compute_position_id_with_mask(padded_mask)

    new_batch = DataProto.from_single_dict({
        "input_ids": padded_ids, "attention_mask": padded_mask, "position_ids": padded_pos
    })

    for key in ["uid", "data_source", "reward_model", "extra_info", "raw_prompt", "raw_prompt_len"]:
        if key in global_batch.non_tensor_batch:
            new_batch.non_tensor_batch[key] = global_batch.non_tensor_batch[key][sel_indices]

    new_batch.meta_info = global_batch.meta_info.copy()
    return new_batch


def set_opts_ttpo_info(
    local_batch: DataProto,
    global_batch: Optional[DataProto],
    next_states: Dict[str, Tuple[int, int]],
    round_idx: int,
) -> np.ndarray:
    """Set OPTS_TTPO tree structure info: rid, pid, branch_pos, cid.

    Args:
        local_batch: Current round's local batch.
        global_batch: Accumulated global batch (None for first round).
        next_states: Dict {uid: (parent_idx, branch_pos)}.
        round_idx: Current round index (0-based).

    Returns:
        new_sample_indices: Indices for new samples in global batch after merge.
    """
    batch_size = local_batch.batch['responses'].shape[0]
    uid = local_batch.non_tensor_batch['uid']

    # Generate rid
    rid = np.array([f"r{round_idx}_{i}" for i in range(batch_size)], dtype=object)

    # Set pid and branch_pos
    if round_idx == 0:
        pid = np.array([None] * batch_size, dtype=object)
        branch_pos = np.full(batch_size, -1, dtype=np.int32)
    else:
        # next_states format: {uid: (parent_idx, branch_pos)}
        # Get parent_rid from parent_idx
        global_rid = global_batch.non_tensor_batch['rid']

        pid = np.array([global_rid[next_states[u][0]] if next_states[u][1] != -1 else None for u in uid], dtype=object)
        branch_pos = np.array([next_states[u][1] for u in uid], dtype=np.int32)

    # Initialize cid as empty OrderedDict for each sample
    cid = np.array([OrderedDict() for _ in range(batch_size)], dtype=object)

    # Update global_batch's cid with new children
    if round_idx > 0:
        global_cid = global_batch.non_tensor_batch['cid']
        rid2idx = {r: i for i, r in enumerate(global_batch.non_tensor_batch['rid'])}

        for i in range(batch_size):
            if pid[i] is not None:
                p_idx, bp = rid2idx[pid[i]], int(branch_pos[i])
                if bp not in global_cid[p_idx]:
                    global_cid[p_idx][bp] = []
                global_cid[p_idx][bp].append(rid[i])

    # Update local_batch
    local_batch.non_tensor_batch['rid'] = rid
    local_batch.non_tensor_batch['pid'] = pid
    local_batch.non_tensor_batch['branch_pos'] = branch_pos
    local_batch.non_tensor_batch['cid'] = cid

    # Compute new_sample_indices
    global_size = len(global_batch.non_tensor_batch['rid']) if global_batch is not None else 0
    return np.arange(global_size, global_size + batch_size)


def compute_forward_values(
    batch: DataProto,
    global_batch: Optional[DataProto],
    next_states: Dict[str, Tuple[int, int]],
    gamma: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute forward values for OPTS_TTPO: gamma_t, trajectory_reward.

    Args:
        batch: Current round's batch containing token_level_rewards, response_mask, pid, uid.
        global_batch: Global batch from previous rounds (None for first round).
        next_states: Dict mapping uid to (parent_idx, branch_pos).
        gamma: Discount factor.

    Returns:
        gamma_t: shape (batch_size, response_len)
        trajectory_reward: shape (batch_size, response_len)
    """
    token_level_rewards = batch.batch["token_level_rewards"]
    response_mask = batch.batch["response_mask"]
    pid = batch.non_tensor_batch["pid"]
    uid = batch.non_tensor_batch["uid"]

    batch_size, response_len = token_level_rewards.shape

    # Build rid2idx from global_batch
    rid2idx = {}
    if global_batch is not None:
        rid2idx = {r: i for i, r in enumerate(global_batch.non_tensor_batch["rid"])}

    # Compute parent_branch_pos from next_states
    parent_branch_pos = [next_states.get(u, (None, 0))[1] if u in next_states else 0 for u in uid]

    # Build parent indices and mask
    has_parent = torch.tensor([p is not None and p in rid2idx for p in pid])
    parent_indices = torch.tensor([rid2idx.get(p, 0) if p is not None else 0 for p in pid], dtype=torch.long)
    branch_pos = torch.tensor(parent_branch_pos, dtype=torch.long)

    # Initialize from parent or default values
    if global_batch is not None and has_parent.any():
        parent_gamma_t = global_batch.batch["gamma_t"]
        parent_trajectory_reward = global_batch.batch["trajectory_reward"]
        init_gamma = torch.where(has_parent, parent_gamma_t[parent_indices, branch_pos], torch.tensor(1.0 / gamma))
        init_traj_reward = torch.where(has_parent, parent_trajectory_reward[parent_indices, branch_pos], torch.tensor(0.0))
    else:
        init_gamma = torch.full((batch_size,), 1.0 / gamma)
        init_traj_reward = torch.zeros(batch_size)

    # Forward loop (vectorized over batch, sequential over time)
    gamma_t_list, traj_reward_list = [], []
    last_gamma, last_traj = init_gamma, init_traj_reward

    for t in range(response_len):
        mask_t = response_mask[:, t]
        # Only accumulate when response_mask is 1
        new_gamma = last_gamma * gamma
        new_traj = last_traj + new_gamma * token_level_rewards[:, t]

        last_gamma = torch.where(mask_t > 0, new_gamma, last_gamma)
        last_traj = torch.where(mask_t > 0, new_traj, last_traj)

        gamma_t_list.append(last_gamma)
        traj_reward_list.append(last_traj)

    gamma_t = torch.stack(gamma_t_list, dim=1)
    trajectory_reward = torch.stack(traj_reward_list, dim=1)

    return gamma_t, trajectory_reward


def select_next_states(
    batch: DataProto,
    lam: float,
    c: float,
    alpha: float,
    round_idx: int,
    n_samples_per_round: int,
    max_prompt_length: int,
) -> Dict[str, Tuple[int, int]]:
    """Select next states for expansion using TUCT.

    This function updates batch's subtree_branches and state_branches in-place.

    Args:
        batch: DataProto containing all required tensors and non-tensor data
        lam: GAE lambda
        c: TUCT exploration constant
        alpha: Decay factor for TUCT
        round_idx: current round index
        n_samples_per_round: number of samples per round (n)
        max_prompt_length: maximum allowed prompt length

    Returns:
        next_states: Dict mapping uid to (parent_idx, branch_pos)
    """
    # Extract tensors from batch (these are references, modifications are in-place)
    rewards = batch.batch["token_level_rewards"]
    values = batch.batch["values"]
    advantages = batch.batch["advantages"]
    advantages_mean = batch.batch["advantages_mean"]
    trajectory_reward = batch.batch["trajectory_reward"]
    gamma_t = batch.batch["gamma_t"]
    subtree_branches = batch.batch["subtree_branches"]
    state_branches = batch.batch["state_branches"]
    response_mask = batch.batch["response_mask"]

    # Extract non-tensor data
    uid = batch.non_tensor_batch["uid"]
    rid = batch.non_tensor_batch["rid"]
    pid = batch.non_tensor_batch["pid"]
    cid = list(batch.non_tensor_batch["cid"])
    parent_branch_pos = batch.non_tensor_batch["branch_pos"]

    # Compute prompt_lengths
    prompt_len = batch.batch["input_ids"].shape[1] - batch.batch["responses"].shape[1]
    prompt_lengths = batch.batch["attention_mask"][:, :prompt_len].sum(dim=1)

    batch_size, response_len = values.shape
    rid2idx = {r: i for i, r in enumerate(rid)}

    # 1) Compute GVE and expected trajectory reward
    gve = values[:, 1:] + lam * advantages_mean
    expected_traj_reward = trajectory_reward[:, :-1] + gamma_t[:, :-1] * gve

    # 2) Compute partree_branches and decay_factor
    partree_branches = compute_partree_branches(
        cid=cid, pid=pid, subtree_branches=subtree_branches,
        round_idx=round_idx, n_samples_per_round=n_samples_per_round,
        rid2idx=rid2idx, parent_branch_pos=parent_branch_pos,
    )
    decay_factor = compute_decay_factor(cid=cid, response_mask=response_mask, alpha=alpha)

    # 3) Compute TUCT: exploitation + exploration, scaled by decay_factor
    exploration = c * torch.sqrt(torch.log(partree_branches[:, :-1])) / (subtree_branches[:, :-1] + 1e-8)
    tuct = (expected_traj_reward + exploration) * decay_factor[:, :-1]
    tuct = torch.where(response_mask[:, 1:] > 0, tuct, torch.tensor(-float('inf')))

    # Mask positions that would exceed max_prompt_length
    pos_indices = torch.arange(response_len - 1).unsqueeze(0)
    pos_mask = (pos_indices + prompt_lengths.unsqueeze(1)) < max_prompt_length
    tuct = torch.where(pos_mask, tuct, torch.tensor(-float('inf')))

    extra_response_len = pos_mask.sum(dim=1)
    logger_batch.info(f"[select_next_states] extra_response_len: min={extra_response_len.min().item()}, max={extra_response_len.max().item()}, mean={extra_response_len.float().mean().item():.2f}")

    # 4) Select best state per uid, directly return next_states format
    unique_uids = np.unique(uid)
    logger_uid = unique_uids[0]

    def _select_for_uid(u) -> Tuple[str, Tuple[int, int]]:
        """Select the best state for a single uid, return (uid, (parent_idx, branch_pos))."""
        uid_indices = np.where(uid == u)[0]

        # Find max TUCT across all trajectories for this uid
        uid_tuct = tuct[uid_indices]
        max_val, max_flat_idx = uid_tuct.view(-1).max(dim=0)
        traj_idx = max_flat_idx.item() // (response_len - 1)
        pos_idx = max_flat_idx.item() % (response_len - 1)
        best_tuct = max_val.item()
        best_idx = uid_indices[traj_idx]
        best_pos = pos_idx

        # Compare with root TUCT (from prompt start)
        root_indices = [i for i in uid_indices if pid[i] is None]
        root_advs = advantages[root_indices, 0]
        root_gve = values[root_indices[0], 0] + lam * root_advs.mean()
        root_tuct = root_gve.item() + c
        if root_tuct > best_tuct:
            best_idx = root_indices[0]
            best_pos = -1

        if logger_uid == u:
            logger_batch.info(f"[select_next_states] root_advs: {root_advs}")
            logger_batch.info(f"[select_next_states] root_gve: {root_gve}")
            logger_batch.info(f"[select_next_states] root_tuct: {root_tuct}")

        return (u, (best_idx, best_pos))

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(_select_for_uid, unique_uids))

    next_states = dict(results)
    max_states = sorted(next_states.values(), key=lambda x: -x[1])[0]
    if max_states[1] != -1:
        start_idx = max(max_states[1] - 10, 0)
        end_idx = start_idx + 100
        logger_batch.info(f"[select_next_states] rewards[max_states[0]]: {rewards[max_states[0]][start_idx:end_idx].tolist()}")
        logger_batch.info(f"[select_next_states] advantages_mean[max_states[0]]: {advantages_mean[max_states[0]][start_idx:end_idx].tolist()}")
        logger_batch.info(f"[select_next_states] gve[max_states[0]]: {gve[max_states[0]][start_idx:end_idx].tolist()}")
        logger_batch.info(f"[select_next_states] trajectory_reward[max_states[0]][-1]: {trajectory_reward[max_states[0]][-1]}")
        logger_batch.info(f"[select_next_states] expected_traj_reward[max_states[0]]: {expected_traj_reward[max_states[0]][start_idx:end_idx].tolist()}")
        logger_batch.info(f"[select_next_states] tuct[max_states[0]]: {tuct[max_states[0]][start_idx:end_idx].tolist()}")

    # 5) Update subtree_branches and state_branches in-place (parallel)
    def _update_branches(item: Tuple[str, Tuple[int, int]]) -> None:
        u, (idx, pos) = item
        if pos >= 0:
            # Update state_branches at selected position
            state_branches[idx, pos] += n_samples_per_round

            # Propagate subtree_branches upward
            current_idx = idx
            bp = pos
            while True:
                subtree_branches[current_idx, :bp + 1] += n_samples_per_round
                parent_rid = pid[current_idx]
                if parent_rid is None or parent_rid not in rid2idx:
                    break
                current_idx = rid2idx[parent_rid]
                bp = parent_branch_pos[current_idx]

    with ThreadPoolExecutor() as executor:
        list(executor.map(_update_branches, next_states.items()))

    return next_states
    

class RayOPTSTTPOTrainer(RayPPOTrainer):
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping or Role.ActorRolloutRef in role_worker_mapping, (
                f"{role_worker_mapping.keys()=}"
            )

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.role_worker_mapping)
        # legacy reward model implementation
        self.use_rm = need_reward_model(self.role_worker_mapping)
        self.use_reward_loop = self.config.reward_model.use_reward_loop

        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = (
            config.actor_rollout_ref.model.get("lora_rank", 0) > 0
            or config.actor_rollout_ref.model.get("lora_adapter_path") is not None
        )

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self.use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("train_max_samples", -1),
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("val_max_samples", -1),
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _log_rollout_data(
        self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str
    ):
        """Log rollout data to disk.
        Args:
            batch (DataProto): The batch containing rollout data
            reward_extra_infos_dict (dict): Additional reward information to log
            timing_raw (dict): Timing information for profiling
            rollout_data_dir (str): Directory path to save the rollout data
        """
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]

            reward_extra_infos_to_dump = reward_extra_infos_dict.copy()
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_dict.setdefault(
                    "request_id",
                    batch.non_tensor_batch["request_id"].tolist(),
                )

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=rollout_data_dir,
            )

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _compute_or_extract_reward(
        self,
        batch: DataProto,
        reward_fn=None,
        return_dict: bool = False,
        sum_reward: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]] | torch.Tensor | dict[str, Any]:
        """
        Compute or extract reward from batch.

        When use_reward_loop=True, rewards are already computed during generate_sequences
        and stored in rm_scores. This method directly extracts them instead of calling
        reward functions which would only perform format conversion.

        Args:
            batch: DataProto containing the batch data
            reward_fn: Reward function to use if rm_scores doesn't exist (for training/validation)
            return_dict: Whether to return dict format with reward_extra_info (for validation)
            sum_reward: Whether to sum reward tensor along last dimension (for REMAX baseline)

        Returns:
            If return_dict=True: dict with "reward_tensor" and "reward_extra_info"
            If return_dict=False and sum_reward=True: summed reward_tensor (1D tensor)
            If return_dict=False and sum_reward=False: reward_tensor (2D tensor)
        """
        # When rm_scores already exists, extract it directly (format conversion only)
        if "rm_scores" in batch.batch.keys():
            reward_tensor = batch.batch["rm_scores"]
            if sum_reward:
                reward_tensor = reward_tensor.sum(dim=-1)

            if return_dict:
                # Extract reward_extra_info if available
                reward_extra_keys = batch.meta_info.get("reward_extra_keys", [])
                reward_extra_info = (
                    {key: batch.non_tensor_batch[key] for key in reward_extra_keys} if reward_extra_keys else {}
                )
                return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
            else:
                # If sum_reward=True, only return tensor (for REMAX baseline)
                if sum_reward:
                    return reward_tensor
                # Otherwise, return tuple with reward_extra_info (for training loop)
                reward_extra_keys = batch.meta_info.get("reward_extra_keys", [])
                reward_extra_infos_dict = (
                    {key: batch.non_tensor_batch[key] for key in reward_extra_keys} if reward_extra_keys else {}
                )
                return reward_tensor, reward_extra_infos_dict

        # Otherwise, compute reward using reward_fn
        if reward_fn is None:
            raise ValueError("reward_fn must be provided when rm_scores is not available.")

        if return_dict:
            result = reward_fn(batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            if sum_reward:
                reward_tensor = reward_tensor.sum(dim=-1)
            reward_extra_info = result.get("reward_extra_info", {})
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        else:
            reward_tensor, reward_extra_infos_dict = compute_reward(batch, reward_fn)
            if sum_reward:
                reward_tensor = reward_tensor.sum(dim=-1)
            return reward_tensor, reward_extra_infos_dict

    def _set_full_response_str(self, batch: DataProto) -> None:
        """Decode full response and store in extra_info for OPTS_TTPO mode.

        This method computes the full response string for each sample by extracting
        tokens from raw_prompt_len onwards and decoding them.

        Args:
            batch: DataProto containing input_ids, attention_mask, and raw_prompt_len.
        """
        batch_size = batch.batch["input_ids"].shape[0]

        if "extra_info" not in batch.non_tensor_batch:
            batch.non_tensor_batch["extra_info"] = np.array([{} for _ in range(batch_size)], dtype=object)

        for i in range(batch_size):
            raw_prompt_len = int(batch.non_tensor_batch["raw_prompt_len"][i])
            valid_prompt_len = int(batch.batch["attention_mask"][i, :self.config.data.max_prompt_length].sum().item())
            pad_len = self.config.data.max_prompt_length - valid_prompt_len
            start_pos = pad_len + raw_prompt_len
            full_response_ids = batch.batch["input_ids"][i, start_pos:]
            full_response_str = self.tokenizer.decode(full_response_ids, skip_special_tokens=True)
            batch.non_tensor_batch["extra_info"][i]["full_response_str"] = full_response_str

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid", "raw_prompt_len"}) & batch.non_tensor_batch.keys()

        # pop those keys for generation
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
                "round_idx": 0,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # evaluate using reward_function
            result = self._compute_or_extract_reward(test_batch, reward_fn=self.val_reward_fn, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            reward_extra_info = result.get("reward_extra_info", {})
            for key, values in reward_extra_info.items():
                if key not in reward_extra_infos_dict:
                    reward_extra_infos_dict[key] = []
                if isinstance(values, np.ndarray):
                    reward_extra_infos_dict[key].extend(values.tolist())
                else:
                    reward_extra_infos_dict[key].extend(values if isinstance(values, list) else [values])

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        actor_role = Role.ActorRolloutRef if Role.ActorRolloutRef in self.role_worker_mapping else Role.ActorRollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[actor_role],
                config=self.config.actor_rollout_ref,
                role=str(actor_role),
            )
            self.resource_pool_to_cls[resource_pool][str(actor_role)] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)

            from verl.workers.config import CriticConfig

            critic_cfg: CriticConfig = omega_conf_to_dataclass(self.config.critic)

            if self.use_legacy_worker_impl == "disable":
                # convert critic_cfg into TrainingWorkerConfig
                from verl.workers.engine_workers import TrainingWorkerConfig

                orig_critic_cfg = critic_cfg
                if orig_critic_cfg.strategy == "fsdp":
                    engine_config: FSDPEngineConfig = orig_critic_cfg.model.fsdp_config
                    engine_config.infer_max_token_len_per_gpu = critic_cfg.ppo_infer_max_token_len_per_gpu
                    engine_config.max_token_len_per_gpu = critic_cfg.ppo_max_token_len_per_gpu
                else:
                    raise NotImplementedError(f"Unknown strategy {orig_critic_cfg.strategy=}")

                critic_cfg = TrainingWorkerConfig(
                    model_type="value_model",
                    model_config=orig_critic_cfg.model_config,
                    engine_config=engine_config,
                    optimizer_config=orig_critic_cfg.optim,
                    checkpoint_config=orig_critic_cfg.checkpoint,
                )

            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy and Role.RefPolicy in self.role_worker_mapping:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

        # create a reward model if reward_fn is None
        # for legacy discriminative reward model, we create a reward model worker here
        # for reward loop discriminative reward model, we create a reward loop manager here
        if not self.use_reward_loop:
            # legacy reward model only handle reward-model based scenario
            if self.use_rm:
                # we create a RM here
                resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
                rm_cls = RayClassWithInitArgs(
                    self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model
                )
                self.resource_pool_to_cls[resource_pool][str(Role.RewardModel)] = rm_cls
        else:
            # reward loop handle hybrid reward scenario (rule, disrm, genrm, ...)
            can_reward_loop_parallelize = self.config.actor_rollout_ref.rollout.mode == "async" and (
                not self.use_rm or self.config.reward_model.enable_resource_pool
            )
            # judge if we can asynchronously parallelize reward model with actor rollout
            # two condition that we can parallelize reward model with actor rollout:
            # 1. reward model is not enabled (rule-based reward can parallelize)
            # 2. reward model is enabled but extra resource pool is enabled
            # If we cannot parallelize, we should enable synchronous mode here, and launch a reward loop manager here
            # else for parallelize mode, we launch a reward worker for each rollout worker (in agent loop, not here)
            if not can_reward_loop_parallelize:
                from verl.experimental.reward_loop import RewardLoopManager

                self.config.reward_model.n_gpus_per_node = self.config.trainer.n_gpus_per_node
                resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
                self.reward_loop_manager = RewardLoopManager(
                    config=self.config,
                    rm_resource_pool=resource_pool,
                )

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            if self.use_legacy_worker_impl == "disable":
                self.critic_wg.reset()
                # assign critic loss
                from functools import partial

                from verl.workers.utils.losses import value_loss

                value_loss_ = partial(value_loss, config=orig_critic_cfg)
                self.critic_wg.set_loss_fn(value_loss_)
            else:
                self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            if str(Role.RefPolicy) in all_wg:
                self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
                self.ref_policy_wg.init_model()
            else:
                # Model engine: ActorRolloutRefWorker
                assert str(Role.ActorRolloutRef) in all_wg, f"{all_wg.keys()=}"
                self.ref_policy_wg = all_wg[str(Role.ActorRolloutRef)]

        self.rm_wg = None
        # initalization of rm_wg will be deprecated in the future
        if self.use_rm and not self.use_reward_loop:
            self.rm_wg = all_wg[str(Role.RewardModel)]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg[str(actor_role)]
        self.actor_rollout_wg.init_model()

        if self.ref_in_actor:
            self.ref_policy_wg = self.actor_rollout_wg

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            # Support custom AgentLoopManager via config
            manager_class_fqn = self.config.actor_rollout_ref.rollout.get("agent", {}).get("agent_loop_manager_class")
            if manager_class_fqn:
                AgentLoopManager = load_class_from_fqn(manager_class_fqn, "AgentLoopManager")
            else:
                from verl.experimental.agent_loop import AgentLoopManager

            self.async_rollout_mode = True
            if self.config.reward_model.enable and self.config.reward_model.enable_resource_pool:
                rm_resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            else:
                rm_resource_pool = None

            self.async_rollout_manager = AgentLoopManager(
                config=self.config,
                worker_group=self.actor_rollout_wg,
                rm_resource_pool=rm_resource_pool,
            )

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", str(Role.Critic)
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        if (
            hasattr(self.config.actor_rollout_ref.actor.checkpoint, "async_save")
            and self.config.actor_rollout_ref.actor.checkpoint.async_save
        ) or (
            "async_save" in self.config.actor_rollout_ref.actor.checkpoint
            and self.config.actor_rollout_ref.actor.checkpoint["async_save"]
        ):
            print("skip write latest_checkpointed_iteration.txt when async_save is True")
            return
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, str(Role.Critic))
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)
            if self.use_rm and not self.use_reward_loop:
                self.rm_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()
            if self.use_rm and not self.use_reward_loop:
                self.rm_wg.stop_profile()

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen", keep_minibatch=False):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1)  # (train_batch_size,)
        workload_lst = calculate_workload(global_seqlen_lst)
        world_size = self.actor_rollout_wg.world_size
        if keep_minibatch:
            # Decouple the DP balancing and mini-batching.
            minibatch_size = self.config.actor_rollout_ref.actor.get("ppo_mini_batch_size")
            minibatch_num = len(workload_lst) // minibatch_size
            global_partition_lst = [[] for _ in range(world_size)]
            for i in range(minibatch_num):
                rearrange_minibatch_lst = get_seqlen_balanced_partitions(
                    workload_lst[i * minibatch_size : (i + 1) * minibatch_size],
                    k_partitions=world_size,
                    equal_size=True,
                )
                for j, part in enumerate(rearrange_minibatch_lst):
                    global_partition_lst[j].extend([x + minibatch_size * i for x in part])
        else:
            global_partition_lst = get_seqlen_balanced_partitions(
                workload_lst, k_partitions=world_size, equal_size=True
            )
        # Place smaller micro-batches at both ends to reduce the bubbles in pipeline parallel.
        for idx, partition in enumerate(global_partition_lst):
            partition.sort(key=lambda x: (workload_lst[x], x))
            ordered_partition = partition[::2] + partition[1::2][::-1]
            global_partition_lst[idx] = ordered_partition
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _compute_values(self, batch: DataProto) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to nopadding
            batch_td = left_right_2_no_padding(batch_td)
            # step 3: add meta info
            tu.assign_non_tensor(batch_td, compute_loss=False)
            output = self.critic_wg.infer_batch(batch_td)
            output = output.get()
            values = tu.get(output, "values")
            values = no_padding_2_padding(values, batch_td)
            values = tu.get_tensordict({"values": values.float()})
            values = DataProto.from_tensordict(values)
        else:
            values = self.critic_wg.compute_values(batch)
        return values

    def _compute_ref_log_prob(self, batch: DataProto) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            # step 1: convert dataproto to tensordict.
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to nopadding
            batch_td = left_right_2_no_padding(batch_td)
            # step 3: add meta info
            tu.assign_non_tensor(batch_td, calculate_entropy=False, compute_loss=False)
            output = self.ref_policy_wg.compute_ref_log_prob(batch_td)
            # gather output
            log_probs = tu.get(output, "log_probs")
            # step 4. No padding to padding
            log_probs = no_padding_2_padding(log_probs, batch_td)
            # step 5: rebuild a tensordict and convert to dataproto
            ref_log_prob = tu.get_tensordict({"ref_log_prob": log_probs.float()})
            ref_log_prob = DataProto.from_tensordict(ref_log_prob)
        else:
            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)

        return ref_log_prob

    def _compute_old_log_prob(self, batch: DataProto):
        if self.use_legacy_worker_impl == "disable":
            # TODO: remove step 1, 2, 4 after we make the whole training tensordict and padding free
            # step 1: convert dataproto to tensordict.
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to nopadding
            batch_td = left_right_2_no_padding(batch_td)
            # step 3: add meta info
            tu.assign_non_tensor(batch_td, calculate_entropy=True, compute_loss=False)
            output = self.actor_rollout_wg.compute_log_prob(batch_td)
            # gather output
            entropy = tu.get(output, "entropy")
            log_probs = tu.get(output, "log_probs")
            old_log_prob_mfu = tu.get(output, "metrics")["mfu"]
            # step 4. No padding to padding
            entropy = no_padding_2_padding(entropy, batch_td)
            log_probs = no_padding_2_padding(log_probs, batch_td)
            # step 5: rebuild a tensordict and convert to dataproto
            old_log_prob = tu.get_tensordict({"old_log_probs": log_probs.float(), "entropys": entropy.float()})
            old_log_prob = DataProto.from_tensordict(old_log_prob)
        else:
            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
            old_log_prob_mfu = 0
        return old_log_prob, old_log_prob_mfu

    def _update_actor(self, batch: DataProto) -> DataProto:
        rollout_config = self.config.actor_rollout_ref.rollout
        batch.meta_info["multi_turn"] = rollout_config.multi_turn.enable
        # TODO: Make "temperature" single source of truth from generation.
        batch.meta_info["temperature"] = rollout_config.temperature
        # update actor
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to no-padding
            batch_td = left_right_2_no_padding(batch_td)
            calculate_entropy = self.config.actor_rollout_ref.actor.entropy_coeff != 0.0
            ppo_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
            ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
            ppo_epochs = self.config.actor_rollout_ref.actor.ppo_epochs
            seed = self.config.actor_rollout_ref.actor.data_loader_seed
            shuffle = self.config.actor_rollout_ref.actor.shuffle
            tu.assign_non_tensor(
                batch_td,
                calculate_entropy=calculate_entropy,
                global_batch_size=ppo_mini_batch_size,
                mini_batch_size=ppo_mini_batch_size,
                epochs=ppo_epochs,
                seed=seed,
                dataloader_kwargs={"shuffle": shuffle},
            )

            actor_output = self.actor_rollout_wg.update_actor(batch_td)
            actor_output = tu.get(actor_output, "metrics")
            actor_output = rename_dict(actor_output, "actor/")
            # modify key name
            actor_output["perf/mfu/actor"] = actor_output.pop("actor/mfu")
            actor_output = DataProto.from_single_dict(data={}, meta_info={"metrics": actor_output})
        else:
            actor_output = self.actor_rollout_wg.update_actor(batch)
        return actor_output

    def _update_critic(self, batch: DataProto) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to no-padding
            batch_td = left_right_2_no_padding(batch_td)
            ppo_mini_batch_size = self.config.critic.ppo_mini_batch_size
            ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
            ppo_epochs = self.config.critic.ppo_epochs
            seed = self.config.critic.data_loader_seed
            shuffle = self.config.critic.shuffle
            tu.assign_non_tensor(
                batch_td,
                global_batch_size=ppo_mini_batch_size,
                mini_batch_size=ppo_mini_batch_size,
                epochs=ppo_epochs,
                seed=seed,
                dataloader_kwargs={"shuffle": shuffle},
            )

            output = self.critic_wg.train_mini_batch(batch_td)
            output = output.get()
            output = tu.get(output, "metrics")
            output = rename_dict(output, "critic/")
            # modify key name
            output["perf/mfu/critic"] = output.pop("critic/mfu")
            critic_output = DataProto.from_single_dict(data={}, meta_info={"metrics": output})
        else:
            critic_output = self.critic_wg.update_critic(batch)
        return critic_output

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        current_epoch = self.global_steps // len(self.train_dataloader)

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                # Log initial batch state
                log_batch_state(batch, stage="initial_batch", step=self.global_steps)

                # OPTS_TTPO setup: get parameters and initialize global batch
                opts_ttpo_mode = self.config.actor_rollout_ref.rollout.search == "opts"
                global_batch = None
                next_states = {}

                # OPTS_TTPO: Initialize raw_prompt_len before round loop
                if opts_ttpo_mode:
                    raw_prompt_lens = batch.batch["attention_mask"].sum(dim=1).cpu().numpy()
                    batch.non_tensor_batch["raw_prompt_len"] = raw_prompt_lens

                with marked_timer("step", timing_raw):
                    step_start_time = time.perf_counter()
                    logger_batch.info(f"[step={self.global_steps}] ========== STEP START ==========")
                    for round_idx in range(self.config.actor_rollout_ref.rollout.g):
                        round_start_time = time.perf_counter()
                        logger_batch.info(f"[step={self.global_steps}][round={round_idx}] ----- ROUND START -----")
                        gen_batch = self._get_gen_batch(batch)

                        # pass global_steps to trace
                        gen_batch.meta_info["global_steps"] = self.global_steps
                        gen_batch.meta_info["round_idx"] = round_idx
                        gen_batch_output = gen_batch.repeat(
                            repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                        )

                        is_last_step = self.global_steps >= self.total_training_steps

                        # generate a batch
                        with marked_timer("gen", timing_raw, color="red"):
                            with timed_block("generate_sequences", step=self.global_steps, round_idx=round_idx):
                                log_batch_state(gen_batch_output, stage="before_generate", step=self.global_steps, round_idx=round_idx)
                                if not self.async_rollout_mode:
                                    gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
                                else:
                                    gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)
                                log_batch_state(gen_batch_output, stage="after_generate", step=self.global_steps, round_idx=round_idx)
                                if round_idx > 1:
                                    log_sample_generations(gen_batch_output, self.tokenizer, step=self.global_steps, round_idx=round_idx)

                            timing_raw.update(gen_batch_output.meta_info["timing"])
                            gen_batch_output.meta_info.pop("timing", None)

                        if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                            if self.reward_fn is None:
                                raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                            with marked_timer("gen_max", timing_raw, color="purple"):
                                gen_baseline_batch = deepcopy(gen_batch)
                                gen_baseline_batch.meta_info["do_sample"] = False
                                if not self.async_rollout_mode:
                                    gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                                else:
                                    gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                                batch = batch.union(gen_baseline_output)
                                # compute reward model score on batch
                                rm_scores = None
                                if self.use_rm and "rm_scores" not in batch.batch.keys():
                                    if not self.use_reward_loop:
                                        rm_scores = self.rm_wg.compute_rm_score(batch)
                                    else:
                                        assert self.reward_loop_manager is not None, "RewardLoopManager is None"
                                        rm_scores = self.reward_loop_manager.compute_rm_score(batch)
                                    batch = batch.union(rm_scores)

                                # Compute or extract reward for REMAX baseline
                                reward_baseline_tensor = self._compute_or_extract_reward(
                                    batch, reward_fn=self.reward_fn, sum_reward=True
                                )

                                keys_to_pop = set(gen_baseline_output.batch.keys())
                                if rm_scores is not None:
                                    keys_to_pop.update(rm_scores.batch.keys())
                                batch.pop(batch_keys=list(keys_to_pop))

                                batch.batch["reward_baselines"] = reward_baseline_tensor

                                del rm_scores, gen_baseline_batch, gen_baseline_output
                        # repeat to align with repeated responses in rollout
                        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                        # In OPTS mode, pop input_ids/attention_mask/position_ids before union to avoid conflict
                        batch = batch.union(gen_batch_output)
                        log_batch_state(batch, stage="after_union_gen_output", step=self.global_steps, round_idx=round_idx)

                        if "response_mask" not in batch.batch.keys():
                            batch.batch["response_mask"] = compute_response_mask(batch)
                        # Balance the number of valid tokens across DP ranks.
                        # NOTE: This usually changes the order of data in the `batch`,
                        # which won't affect the advantage calculation (since it's based on uid),
                        # but might affect the loss calculation (due to the change of mini-batching).
                        if self.config.trainer.balance_batch:
                            self._balance_batch(batch, metrics=metrics)

                        # compute global_valid tokens
                        batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                        with marked_timer("reward", timing_raw, color="yellow"):
                            # compute reward model score
                            if self.use_rm and "rm_scores" not in batch.batch.keys():
                                with timed_block("compute_rm_score", step=self.global_steps, round_idx=round_idx):
                                    if not self.use_reward_loop:
                                        reward_tensor = self.rm_wg.compute_rm_score(batch)
                                    else:
                                        assert self.reward_loop_manager is not None, "RewardLoopManager is None"
                                        reward_tensor = self.reward_loop_manager.compute_rm_score(batch)
                                    batch = batch.union(reward_tensor)

                            # OPTS_TTPO: Decode full response and store in extra_info
                            if opts_ttpo_mode:
                                self._set_full_response_str(batch)

                            # Compute or extract reward for training
                            if self.config.reward_model.launch_reward_fn_async:
                                future_reward = compute_reward_async.remote(
                                    data=batch, config=self.config, tokenizer=self.tokenizer
                                )
                            else:
                                with timed_block("compute_reward", step=self.global_steps, round_idx=round_idx):
                                    reward_tensor, reward_extra_infos_dict = self._compute_or_extract_reward(
                                        batch, reward_fn=self.reward_fn, return_dict=False
                                    )

                        # Operating Mode Selection:
                        # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ)
                        # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: π_rollout, π_old, π_θ)
                        #   Note: π_old computed once per data batch, serves as stable reference during mini-batch updates
                        rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                        bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                        if bypass_recomputing_logprobs:  # Use `rollout_log_probs`
                            from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode

                            apply_bypass_mode(
                                batch=batch,
                                rollout_corr_config=rollout_corr_config,
                                policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                            )
                        else:  # Recompute old_log_probs
                            with marked_timer("old_log_prob", timing_raw, color="blue"):
                                with timed_block("compute_old_log_prob", step=self.global_steps, round_idx=round_idx):
                                    old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
                                    entropys = old_log_prob.batch["entropys"]
                                    response_masks = batch.batch["response_mask"]
                                    actor_config = self.config.actor_rollout_ref.actor
                                    entropy_agg = agg_loss(
                                        loss_mat=entropys,
                                        loss_mask=response_masks,
                                        loss_agg_mode=actor_config.loss_agg_mode,
                                        loss_scale_factor=actor_config.loss_scale_factor,
                                    )
                                    old_log_prob_metrics = {
                                        "actor/entropy": entropy_agg.detach().item(),
                                        "perf/mfu/actor_infer": old_log_prob_mfu,
                                    }
                                    metrics.update(old_log_prob_metrics)
                                    old_log_prob.batch.pop("entropys")
                                    batch = batch.union(old_log_prob)
                                if "rollout_log_probs" in batch.batch.keys():
                                    # TODO: we may want to add diff of probs too.
                                    from verl.utils.debug.metrics import calculate_debug_metrics

                                    metrics.update(calculate_debug_metrics(batch))

                        assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

                        if self.use_reference_policy:
                            # compute reference log_prob
                            with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                                with timed_block("compute_ref_log_prob", step=self.global_steps, round_idx=round_idx):
                                    ref_log_prob = self._compute_ref_log_prob(batch)
                                    batch = batch.union(ref_log_prob)

                        # compute values
                        if self.use_critic:
                            with marked_timer("values", timing_raw, color="cyan"):
                                with timed_block("compute_values", step=self.global_steps, round_idx=round_idx):
                                    values = self._compute_values(batch)
                                    batch = batch.union(values)

                        with marked_timer("adv", timing_raw, color="brown"):
                            # we combine with rule-based rm
                            reward_extra_infos_dict: dict[str, list]
                            if self.config.reward_model.launch_reward_fn_async:
                                reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                            batch.batch["token_level_scores"] = reward_tensor

                            if reward_extra_infos_dict:
                                batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                            # compute rewards. apply_kl_penalty if available
                            if self.config.algorithm.use_kl_in_reward:
                                batch, kl_metrics = apply_kl_penalty(
                                    batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                                )
                                metrics.update(kl_metrics)
                            else:
                                batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                            # Compute rollout correction: IS weights, rejection sampling, and metrics
                            # Only runs in decoupled mode (computes once per batch using stable π_old)
                            # In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout
                            if (
                                rollout_corr_config is not None
                                and "rollout_log_probs" in batch.batch
                                and not bypass_recomputing_logprobs  # Only in decoupled mode
                            ):
                                from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                                # Compute IS weights, apply rejection sampling, compute metrics
                                batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                                # IS and off-policy metrics already have rollout_corr/ prefix
                                metrics.update(is_metrics)

                            # compute advantages, executed on the driver process
                            norm_adv_by_std_in_grpo = self.config.algorithm.get(
                                "norm_adv_by_std_in_grpo", True
                            )  # GRPO adv normalization factor

                            # OPTS_TTPO: set tree structure info before advantage computation
                            if opts_ttpo_mode:
                                # Set tree structure info (rid, pid, branch_pos, cid)
                                new_sample_indices = set_opts_ttpo_info(batch, global_batch, next_states, round_idx)
                                batch.non_tensor_batch["new_sample_indices"] = new_sample_indices

                                # Initialize state_branches, subtree_branches as all ones
                                batch_size, response_len = batch.batch["responses"].shape
                                batch.batch["state_branches"] = torch.ones(batch_size, response_len)
                                batch.batch["subtree_branches"] = torch.ones(batch_size, response_len)

                                # Initialize advantages, advantages_mean, returns as all zeros
                                batch.batch["advantages"] = torch.zeros(batch_size, response_len)
                                batch.batch["advantages_mean"] = torch.zeros(batch_size, response_len - 1)
                                batch.batch["returns"] = torch.zeros(batch_size, response_len)

                                # Compute forward values (gamma_t, trajectory_reward)
                                with timed_block("compute_forward_values", step=self.global_steps, round_idx=round_idx):
                                    gamma_t, trajectory_reward = compute_forward_values(
                                        batch=batch,
                                        global_batch=global_batch,
                                        next_states=next_states,
                                        gamma=self.config.algorithm.gamma,
                                    )
                                batch.batch["gamma_t"] = gamma_t
                                batch.batch["trajectory_reward"] = trajectory_reward

                                # Merge local_batch to global_batch
                                if global_batch is None:
                                    global_batch = batch
                                else:
                                    global_batch = merge_batches(global_batch, batch)
                                # log_batch_state(global_batch, stage="after_merge_to_global_batch", step=self.global_steps, round_idx=round_idx)

                            # For OPTS_TTPO, compute advantage on global_batch; otherwise on batch
                            target_batch = global_batch if opts_ttpo_mode else batch
                            with timed_block("compute_advantage", step=self.global_steps, round_idx=round_idx):
                                target_batch = compute_advantage(
                                    target_batch,
                                    adv_estimator=self.config.algorithm.adv_estimator,
                                    gamma=self.config.algorithm.gamma,
                                    lam=self.config.algorithm.lam,
                                    num_repeat=self.config.actor_rollout_ref.rollout.n,
                                    norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                                    config=self.config.algorithm,
                                    next_states=next_states if opts_ttpo_mode else None,
                                )
                            if opts_ttpo_mode:
                                global_batch = target_batch
                            else:
                                batch = target_batch
                            log_batch_state(target_batch, stage="after_advantage", step=self.global_steps, round_idx=round_idx)

                        # OPTS_TTPO: select next states and prepare next round input
                        if opts_ttpo_mode:
                            # Select next states if not last round
                            if round_idx < self.config.actor_rollout_ref.rollout.g - 1:
                                with timed_block("select_next_states", step=self.global_steps, round_idx=round_idx):
                                    next_states = select_next_states(
                                        batch=global_batch,
                                        lam=self.config.algorithm.lam,
                                        c=self.config.actor_rollout_ref.rollout.c,
                                        alpha=self.config.actor_rollout_ref.rollout.alpha,
                                        round_idx=round_idx,
                                        n_samples_per_round=self.config.actor_rollout_ref.rollout.n,
                                        max_prompt_length=self.config.data.max_prompt_length,
                                    )
                                    sorted_states = sorted(next_states.values(), key=lambda x: -x[1])
                                    logger_batch.info(f"[step={self.global_steps}][round={round_idx}][after_select_next_states] next_states count={len(next_states)}, max_pos={sorted_states[:5]}, min_pos={sorted_states[-5:]}")

                                # Prepare next round input
                                with timed_block("prepare_next_round_input", step=self.global_steps, round_idx=round_idx):
                                    batch = prepare_next_round_input(
                                        global_batch=global_batch,
                                        next_states=next_states,
                                    )

                        # Log round completion
                        round_elapsed = time.perf_counter() - round_start_time
                        logger_batch.info(f"[step={self.global_steps}][round={round_idx}] ----- ROUND END (elapsed: {round_elapsed:.3f}s) -----")

                    if opts_ttpo_mode:
                        with timed_block("opts_ttpo_final_processing", step=self.global_steps):
                            import verl.utils.torch_functional as verl_F
                            # Use global_batch for update in OPTS_TTPO mode
                            batch = global_batch
                            batch.batch["advantages"] = verl_F.masked_whiten(batch.batch["advantages"], batch.batch["response_mask"])

                            # Compute branch_weight_factor for TTPO gradient correction
                            branch_weight_factor = compute_branch_weight_factors(
                                state_branches=batch.batch["state_branches"],
                                pid=batch.non_tensor_batch["pid"],
                                rid=batch.non_tensor_batch["rid"],
                                branch_pos=batch.non_tensor_batch["branch_pos"],
                            )
                            batch.batch["branch_weight_factor"] = branch_weight_factor
                            log_batch_state(batch, stage="opts_ttpo_final_batch_before_update", step=self.global_steps)
                        
                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            with timed_block("update_critic", step=self.global_steps):
                                critic_output = self._update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            with timed_block("update_actor", step=self.global_steps):
                                actor_output = self._update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                    # Log step completion with timing summary
                    step_elapsed = time.perf_counter() - step_start_time
                    logger_batch.info(f"[step={self.global_steps}] ========== STEP END (total: {step_elapsed:.3f}s) ==========")
                    logger_batch.info(f"[step={self.global_steps}] Timing summary: {timing_raw}")

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        with timed_block("validation", step=self.global_steps):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                # Check if the conditions for saving a checkpoint are met.
                # The conditions include a mandatory condition (1) and
                # one of the following optional conditions (2/3/4):
                # 1. The save frequency is set to a positive value.
                # 2. It's the last training step.
                # 3. The current step number is a multiple of the save frequency.
                # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        with timed_block("save_checkpoint", step=self.global_steps):
                            self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)
