# Copyright 2025 Anonymous authors. All rights reserved.

"""TreeRL trainer implemented on the repository's native VeRL stack."""

from __future__ import annotations

import copy
from pprint import pprint
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.ppo import core_algos as verl_core_algos
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
)
from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.model import compute_position_id_with_mask
from verl.utils.rollout_skip import RolloutSkip

from trainer.opts_ttpo.ray_trainer import (
    PromptBuffer,
    RayOPTSTTPOTrainer,
    compute_response_mask,
)

from .core_algos import (
    BranchJob,
    RolloutPlan,
    TreeLeaf,
    compute_process_advantages,
    plan_rollout_budget,
    select_entropy_branch_jobs,
)


class RayTreeRLTrainer(RayOPTSTTPOTrainer):
    """Generate EPTrees and optimize the actor with TreeRL process rewards."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.use_critic:
            raise ValueError("TreeRL is policy-only; set critic.enable=False")
        if self.config.algorithm.use_kl_in_reward:
            raise ValueError("The matched OPTS-TTPO protocol sets algorithm.use_kl_in_reward=False")
        if self.config.actor_rollout_ref.actor.use_kl_loss:
            raise ValueError("The matched OPTS-TTPO protocol sets actor.use_kl_loss=False")
        if self.config.actor_rollout_ref.rollout.get("search", None) != "treerl":
            raise ValueError("TreeRL requires actor_rollout_ref.rollout.search=treerl")
        if not self.config.actor_rollout_ref.rollout.calculate_log_probs:
            raise ValueError("EPTree fork selection requires rollout.calculate_log_probs=True")
        if not self.use_reward_loop:
            raise ValueError("TreeRL requires reward_model.use_reward_loop=True to score reconstructed full paths")

    def init_workers(self):
        """Initialize VeRL workers plus a driver-side async reward loop.

        Async rollout workers score the compact generation they receive.  A
        TreeRL branch is generated from a continuation prompt, however, so that
        compact score only sees the suffix.  Training rewards must instead be
        computed after the driver has reconstructed every complete root-to-leaf
        response.  A dedicated RewardLoopManager provides the same async custom
        verifier used by rollout generation for those reconstructed paths.
        """

        super().init_workers()
        if not hasattr(self, "reward_loop_manager"):
            from verl.experimental.reward_loop import RewardLoopManager

            self.reward_loop_manager = RewardLoopManager(config=self.config)

    def _rollout_plan(self) -> RolloutPlan:
        cfg = self.config.algorithm.treerl
        plan = plan_rollout_budget(
            total_rollouts=int(cfg.total_rollouts),
            roots_per_prompt=int(cfg.m),
            fork_points=int(cfg.n),
            iterations=int(cfg.l),
            branches_per_fork=int(cfg.t),
        )
        configured_batch = int(self.config.data.train_batch_size)
        if configured_batch != plan.total_rollouts:
            raise ValueError(
                "For a matched completed-rollout budget, data.train_batch_size "
                f"({configured_batch}) must equal algorithm.treerl.total_rollouts ({plan.total_rollouts})"
            )
        return plan

    def _expand_root_prompts(self, prompts: DataProto, plan: RolloutPlan) -> DataProto:
        indices: list[int] = []
        for prompt_index in range(plan.tree_prompt_count):
            indices.extend([prompt_index] * plan.roots_per_prompt)
        if plan.filler_prompt_count:
            indices.extend([plan.tree_prompt_count] * plan.filler_root_count)
        roots = prompts[indices]
        if "extra_info" in roots.non_tensor_batch:
            roots.non_tensor_batch["extra_info"] = np.array(
                [copy.copy(value) for value in roots.non_tensor_batch["extra_info"]], dtype=object
            )
        # TreeRL rewards are only valid after every continuation suffix has
        # been attached to its inherited prefix on the driver.
        roots.non_tensor_batch["treerl_defer_reward"] = np.ones(len(roots), dtype=np.bool_)
        roots.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
        return roots

    def _set_root_metadata(self, roots: DataProto) -> None:
        size = len(roots)
        roots.non_tensor_batch["rid"] = np.array(
            [f"step{self.global_steps}_root{i}" for i in range(size)], dtype=object
        )
        roots.non_tensor_batch["parent_rid"] = np.array([None] * size, dtype=object)
        roots.non_tensor_batch["branch_pos"] = np.full(size, -1, dtype=np.int32)

    def _prepare_branch_inputs(self, roots: DataProto, jobs: list[BranchJob]) -> DataProto:
        if not jobs:
            raise ValueError("TreeRL received no branch jobs")

        base_prompt_width = roots.batch["prompts"].shape[1]
        response_width = roots.batch["responses"].shape[1]
        continuation_width = base_prompt_width + response_width - 1
        device = roots.batch["input_ids"].device
        dtype = roots.batch["input_ids"].dtype

        input_ids = torch.full(
            (len(jobs), continuation_width),
            self.tokenizer.pad_token_id,
            dtype=dtype,
            device=device,
        )
        attention_mask = torch.zeros((len(jobs), continuation_width), dtype=torch.long, device=device)

        parent_indices = [job.parent_index for job in jobs]
        for row, job in enumerate(jobs):
            parent = job.parent_index
            raw_prompt_len = int(roots.non_tensor_batch["raw_prompt_len"][parent])
            prompt_mask = roots.batch["attention_mask"][parent, :base_prompt_width].bool()
            raw_prompt_ids = roots.batch["prompts"][parent][prompt_mask]
            if len(raw_prompt_ids) != raw_prompt_len:
                raise ValueError(
                    f"raw prompt length mismatch for root {parent}: metadata={raw_prompt_len}, tensor={len(raw_prompt_ids)}"
                )

            inherited_len = job.branch_pos + 1
            inherited_ids = roots.batch["responses"][parent, :inherited_len]
            if inherited_len and not bool(roots.batch["response_mask"][parent, :inherited_len].all()):
                raise ValueError("TreeRL selected a fork beyond the valid root response")
            continuation = torch.cat([raw_prompt_ids, inherited_ids])
            if len(continuation) > continuation_width:
                raise ValueError("TreeRL continuation prompt exceeds the configured context budget")
            input_ids[row, -len(continuation) :] = continuation
            attention_mask[row, -len(continuation) :] = 1

        branch_inputs = DataProto.from_single_dict(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": compute_position_id_with_mask(attention_mask),
            }
        )
        copied_keys = {
            "uid",
            "data_source",
            "reward_model",
            "extra_info",
            "raw_prompt",
            "raw_prompt_len",
        }
        for key in copied_keys & set(roots.non_tensor_batch):
            values = roots.non_tensor_batch[key][parent_indices]
            if key == "extra_info":
                values = np.array([copy.copy(value) for value in values], dtype=object)
            branch_inputs.non_tensor_batch[key] = values
        branch_inputs.non_tensor_batch["treerl_is_continuation"] = np.ones(len(jobs), dtype=np.bool_)
        branch_inputs.meta_info = roots.meta_info.copy()
        branch_inputs.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
        return branch_inputs

    def _set_branch_metadata(self, branches: DataProto, jobs: list[BranchJob]) -> None:
        if len(branches) != len(jobs):
            raise ValueError(f"generated {len(branches)} branches for {len(jobs)} TreeRL jobs")
        branches.non_tensor_batch["rid"] = np.array(
            [f"step{self.global_steps}_branch{i}" for i in range(len(jobs))], dtype=object
        )
        branches.non_tensor_batch["parent_rid"] = np.array([job.parent_rid for job in jobs], dtype=object)
        branches.non_tensor_batch["branch_pos"] = np.array([job.branch_pos for job in jobs], dtype=np.int32)
        branches.non_tensor_batch["selected_token_pos"] = np.array(
            [job.selected_token_pos for job in jobs], dtype=np.int32
        )
        branches.non_tensor_batch["fork_surprisal"] = np.array(
            [job.surprisal for job in jobs], dtype=np.float32
        )

    def _generate(self, batch: DataProto, *, sleep_after: bool) -> DataProto:
        if not self.async_rollout_mode:
            raise ValueError("TreeRL continuation currently requires actor_rollout_ref.rollout.mode=async")
        batch.non_tensor_batch["treerl_defer_reward"] = np.ones(len(batch), dtype=np.bool_)
        gen_batch = self._get_gen_batch(batch)
        gen_batch.meta_info["global_steps"] = self.global_steps
        worker_count = int(self.config.actor_rollout_ref.rollout.agent.num_workers)
        padded_gen_batch, pad_size = pad_dataproto_to_divisor(gen_batch, worker_count)
        output = self.async_rollout_manager.generate_sequences(padded_gen_batch, sleep_after=sleep_after)
        output = unpad_dataproto(output, pad_size=pad_size)
        output.meta_info.pop("timing", None)
        if "rm_scores" in output.batch:
            raise AssertionError("TreeRL compact roots/suffixes were scored before full-path reconstruction")
        # Agent-loop generation always publishes this bookkeeping key, even
        # when deferred scoring correctly produced no reward fields.  It is
        # local to that generation call and must not leak into the next call.
        batch.meta_info.pop("reward_extra_keys", None)
        output.meta_info.pop("reward_extra_keys", None)
        batch = batch.union(output)
        if "response_mask" not in batch.batch:
            batch.batch["response_mask"] = compute_response_mask(batch)
        return batch

    @staticmethod
    def _valid_response_ids(batch: DataProto, row: int) -> torch.Tensor:
        mask = batch.batch["response_mask"][row].bool()
        return batch.batch["responses"][row][mask]

    def _eligible_entropy_token_counts(self, roots: DataProto) -> list[int]:
        """Apply the released EPTree end-of-answer branch mask.

        The reference implementation keeps the token containing ``answer`` or
        ``conclusion`` eligible, then masks every later token so that expansion
        does not merely revisit an already stated final answer.
        """

        counts: list[int] = []
        for row in range(len(roots)):
            response_ids = self._valid_response_ids(roots, row)
            eligible_count = len(response_ids)
            for token_pos, token_id in enumerate(response_ids):
                token_piece = self.tokenizer.decode([int(token_id)], skip_special_tokens=False).lower()
                if "answer" in token_piece or "conclusion" in token_piece:
                    eligible_count = token_pos + 1
                    break
            counts.append(eligible_count)
        return counts

    def _expand_complete_leaves(
        self,
        roots: DataProto,
        branches: DataProto,
        plan: RolloutPlan,
    ) -> tuple[DataProto, dict[str, float]]:
        prompt_width = int(self.config.data.max_prompt_length)
        response_width = int(self.config.data.max_response_length)
        root_by_rid = {rid: i for i, rid in enumerate(roots.non_tensor_batch["rid"])}

        descriptors: list[tuple[DataProto, int, str, str | None, int, torch.Tensor, torch.Tensor]] = []
        generated_suffix_tokens = 0

        for row in range(len(roots)):
            raw_prompt_len = int(roots.non_tensor_batch["raw_prompt_len"][row])
            source_prompt_width = roots.batch["prompts"].shape[1]
            prompt_mask = roots.batch["attention_mask"][row, :source_prompt_width].bool()
            raw_prompt_ids = roots.batch["prompts"][row][prompt_mask]
            if len(raw_prompt_ids) != raw_prompt_len:
                raise ValueError("root raw-prompt metadata no longer matches the generated tensor")
            response_ids = self._valid_response_ids(roots, row)[:response_width]
            generated_suffix_tokens += len(response_ids)
            descriptors.append(
                (
                    roots,
                    row,
                    roots.non_tensor_batch["rid"][row],
                    None,
                    -1,
                    raw_prompt_ids,
                    response_ids,
                )
            )

        continuation_prompt_width = branches.batch["prompts"].shape[1]
        for row in range(len(branches)):
            parent_rid = branches.non_tensor_batch["parent_rid"][row]
            if parent_rid not in root_by_rid:
                raise ValueError(f"TreeRL branch references unknown root {parent_rid!r}")
            raw_prompt_len = int(branches.non_tensor_batch["raw_prompt_len"][row])
            prompt_mask = branches.batch["attention_mask"][row, :continuation_prompt_width].bool()
            continuation_ids = branches.batch["prompts"][row][prompt_mask]
            raw_prompt_ids = continuation_ids[:raw_prompt_len]
            inherited_ids = continuation_ids[raw_prompt_len:]
            suffix_ids = self._valid_response_ids(branches, row)
            generated_suffix_tokens += len(suffix_ids)
            response_ids = torch.cat([inherited_ids, suffix_ids])[:response_width]
            descriptors.append(
                (
                    branches,
                    row,
                    branches.non_tensor_batch["rid"][row],
                    parent_rid,
                    int(branches.non_tensor_batch["branch_pos"][row]),
                    raw_prompt_ids,
                    response_ids,
                )
            )

        if len(descriptors) != plan.total_rollouts:
            raise AssertionError(
                f"TreeRL produced {len(descriptors)} complete leaves, expected {plan.total_rollouts}"
            )

        device = roots.batch["input_ids"].device
        token_dtype = roots.batch["input_ids"].dtype
        prompts = torch.full(
            (len(descriptors), prompt_width), self.tokenizer.pad_token_id, dtype=token_dtype, device=device
        )
        responses = torch.full(
            (len(descriptors), response_width), self.tokenizer.pad_token_id, dtype=token_dtype, device=device
        )
        prompt_mask = torch.zeros((len(descriptors), prompt_width), dtype=torch.long, device=device)
        response_mask = torch.zeros((len(descriptors), response_width), dtype=torch.long, device=device)

        for out_row, (_, _, _, _, _, raw_prompt_ids, response_ids) in enumerate(descriptors):
            if len(raw_prompt_ids) > prompt_width:
                raise ValueError("TreeRL raw prompt exceeds data.max_prompt_length")
            prompts[out_row, -len(raw_prompt_ids) :] = raw_prompt_ids
            prompt_mask[out_row, -len(raw_prompt_ids) :] = 1
            if len(response_ids):
                responses[out_row, : len(response_ids)] = response_ids
                response_mask[out_row, : len(response_ids)] = 1

        input_ids = torch.cat([prompts, responses], dim=1)
        attention_mask = torch.cat([prompt_mask, response_mask], dim=1)
        full_batch = DataProto.from_single_dict(
            {
                "prompts": prompts,
                "responses": responses,
                "response_mask": response_mask,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": compute_position_id_with_mask(attention_mask),
            }
        )

        metadata_keys = {"uid", "data_source", "reward_model", "extra_info", "raw_prompt", "raw_prompt_len"}
        for key in metadata_keys:
            values = []
            for source, source_row, *_ in descriptors:
                if key not in source.non_tensor_batch:
                    raise KeyError(f"TreeRL source batch is missing required metadata key {key!r}")
                value = source.non_tensor_batch[key][source_row]
                values.append(copy.copy(value) if key == "extra_info" else value)
            full_batch.non_tensor_batch[key] = np.array(values, dtype=object if key != "raw_prompt_len" else None)

        full_batch.non_tensor_batch["rid"] = np.array([item[2] for item in descriptors], dtype=object)
        full_batch.non_tensor_batch["parent_rid"] = np.array([item[3] for item in descriptors], dtype=object)
        full_batch.non_tensor_batch["branch_pos"] = np.array([item[4] for item in descriptors], dtype=np.int32)
        full_batch.non_tensor_batch["__num_turns__"] = np.full(len(descriptors), 2, dtype=np.int32)
        full_batch.meta_info = roots.meta_info.copy()

        completed_response_tokens = int(response_mask.sum().item())
        rollout_metrics = {
            "treerl/tree_prompt_count": float(plan.tree_prompt_count),
            "treerl/filler_prompt_count": float(plan.filler_prompt_count),
            "treerl/root_rollouts": float(plan.root_rollouts),
            "treerl/branch_rollouts": float(plan.branch_rollouts),
            "treerl/generated_suffix_tokens": float(generated_suffix_tokens),
            "treerl/completed_response_tokens": float(completed_response_tokens),
            "treerl/reused_prefix_tokens": float(completed_response_tokens - generated_suffix_tokens),
        }
        return full_batch, rollout_metrics

    def _score_and_assign_advantages(self, batch: DataProto) -> tuple[DataProto, dict[str, Any], dict[str, float]]:
        self._set_full_response_str(batch)

        reward_worker_count = len(self.reward_loop_manager.reward_loop_workers)
        reward_input, reward_pad_size = pad_dataproto_to_divisor(batch, reward_worker_count)
        reward_output = self.reward_loop_manager.compute_rm_score(reward_input)
        reward_output = unpad_dataproto(reward_output, pad_size=reward_pad_size)
        batch = batch.union(reward_output)

        reward_result = self._compute_or_extract_reward(batch, return_dict=True)
        reward_tensor = reward_result["reward_tensor"]
        reward_extra_infos = reward_result["reward_extra_info"]
        batch.batch["token_level_scores"] = reward_tensor
        batch.batch["token_level_rewards"] = reward_tensor

        rewards = reward_tensor.sum(dim=-1).detach().cpu().tolist()
        leaves = []
        response_lengths = batch.batch["response_mask"].sum(dim=-1).long().cpu().tolist()
        for row, response_len in enumerate(response_lengths):
            token_ids = tuple(batch.batch["responses"][row, :response_len].detach().cpu().tolist())
            leaves.append(
                TreeLeaf(
                    uid=batch.non_tensor_batch["uid"][row],
                    rid=batch.non_tensor_batch["rid"][row],
                    parent_rid=batch.non_tensor_batch["parent_rid"][row],
                    branch_pos=int(batch.non_tensor_batch["branch_pos"][row]),
                    token_ids=token_ids,
                    reward=float(rewards[row]),
                )
            )

        process_advantages, process_metrics = compute_process_advantages(leaves)
        advantages = torch.zeros_like(reward_tensor, dtype=torch.float32)
        for row, leaf in enumerate(leaves):
            values = torch.as_tensor(process_advantages[leaf.rid], dtype=torch.float32, device=advantages.device)
            advantages[row, : len(values)] = values
        batch.batch["advantages"] = advantages
        batch.batch["returns"] = advantages.clone()
        batch.meta_info["batch_num_tokens"] = float(batch.batch["response_mask"].sum().item())
        batch.meta_info["global_token_num"] = batch.batch["attention_mask"].sum(dim=-1).tolist()
        return batch, reward_extra_infos, process_metrics

    def _collect_tree_batch(self, timing_raw: dict[str, float]) -> tuple[DataProto, dict[str, Any], dict[str, float]]:
        plan = self._rollout_plan()
        base_prompts = self.prompt_buffer.draw(plan.base_prompt_count)
        roots = self._expand_root_prompts(base_prompts, plan)

        with marked_timer("generate_roots", timing_raw, color="red"):
            roots = self._generate(roots, sleep_after=plan.branch_rollouts == 0)
        self._set_root_metadata(roots)

        standard_root_count = plan.tree_prompt_count * plan.roots_per_prompt
        jobs = select_entropy_branch_jobs(
            sampled_log_probs=roots.batch["rollout_log_probs"],
            response_mask=roots.batch["response_mask"],
            raw_prompt_lens=roots.non_tensor_batch["raw_prompt_len"],
            uids=roots.non_tensor_batch["uid"],
            rids=roots.non_tensor_batch["rid"],
            eligible_indices=range(standard_root_count),
            fork_points=plan.fork_points,
            branches_per_fork=plan.branches_per_fork,
            max_prompt_length=int(self.config.data.max_prompt_length)
            + int(self.config.data.max_response_length)
            - 1,
            eligible_token_counts=self._eligible_entropy_token_counts(roots),
        )
        if len(jobs) != plan.branch_rollouts:
            raise AssertionError(f"planned {plan.branch_rollouts} TreeRL branches but selected {len(jobs)}")

        with marked_timer("generate_branches", timing_raw, color="red"):
            branch_inputs = self._prepare_branch_inputs(roots, jobs)
            branches = self._generate(branch_inputs, sleep_after=True)
        self._set_branch_metadata(branches, jobs)

        batch, rollout_metrics = self._expand_complete_leaves(roots, branches, plan)
        with marked_timer("reward", timing_raw, color="yellow"):
            batch, reward_extra_infos, process_metrics = self._score_and_assign_advantages(batch)
        rollout_metrics.update(process_metrics)
        return batch, reward_extra_infos, rollout_metrics

    def fit(self):
        """Run TreeRL with exact completed-rollout budget matching."""
        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()
        current_epoch = self.global_steps // len(self.train_dataloader)

        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            RolloutSkip(self.config, self.actor_rollout_wg).wrap_generate_sequences()

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="TreeRL Training")
        self.global_steps += 1
        self.max_steps_duration = 0
        last_val_metrics = None
        self.prompt_buffer = PromptBuffer(self.train_dataloader)
        self.prompt_buffer.load_state_dict(getattr(self, "_prompt_buffer_state_to_load", None))
        self._prompt_buffer_state_to_load = None

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for _ in range(len(self.train_dataloader)):
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
                metrics: dict[str, Any] = {}
                timing_raw: dict[str, float] = {}
                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )

                with marked_timer("step", timing_raw):
                    batch, reward_extra_infos, tree_metrics = self._collect_tree_batch(timing_raw)
                    metrics.update(tree_metrics)

                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
                        entropys = old_log_prob.batch.pop("entropys")
                        actor_config = self.config.actor_rollout_ref.actor
                        entropy = verl_core_algos.agg_loss(
                            loss_mat=entropys,
                            loss_mask=batch.batch["response_mask"],
                            loss_agg_mode=actor_config.loss_agg_mode,
                            loss_scale_factor=actor_config.loss_scale_factor,
                        )
                        metrics["actor/entropy"] = entropy.detach().item()
                        metrics["perf/mfu/actor_infer"] = old_log_prob_mfu
                        batch = batch.union(old_log_prob)

                    with marked_timer("update_actor", timing_raw, color="red"):
                        actor_output = self._update_actor(batch)
                    metrics.update(reduce_metrics(actor_output.meta_info["metrics"]))

                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos, timing_raw, rollout_data_dir)

                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
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

                self.max_steps_duration = max(self.max_steps_duration, timing_raw["step"])
                metrics.update({"training/global_step": self.global_steps, "training/epoch": epoch})
                metrics.update(compute_data_metrics(batch=batch, use_critic=False))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                metrics.update(
                    compute_throughout_metrics(
                        batch=batch,
                        timing_raw=timing_raw,
                        n_gpus=self.resource_pool_manager.get_n_gpus(),
                    )
                )

                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)
                logger.log(data=metrics, step=self.global_steps)
                progress_bar.update(1)
                self.global_steps += 1

                if is_last_step:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                if hasattr(self.train_dataset, "on_batch_end"):
                    self.train_dataset.on_batch_end(batch=batch)
