# Copyright 2026 Junyu Lu (Julian Lou). All rights reserved.

"""Evaluate every actor checkpoint in one initialized VERL/vLLM process.

The checkpoint root and result directory are supplied through
``CHECKPOINT_EVAL_ROOT`` and ``CHECKPOINT_EVAL_RESULTS``.  One JSON file is
written per completed checkpoint so an interrupted job can be resumed safely.
"""

import json
import os
import re
import socket
from pathlib import Path
from pprint import pprint

import hydra
import ray
from omegaconf import OmegaConf

from verl.trainer.main_ppo import TaskRunner, create_rl_dataset, create_rl_sampler, run_ppo
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.reward import load_reward_manager
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config
from verl.utils.device import auto_set_ascend_device_name
from verl.utils.fs import copy_to_local


def _checkpoint_step(path: Path) -> int:
    match = re.fullmatch(r"global_step_(\d+)", path.name)
    if match is None:
        raise ValueError(f"invalid checkpoint directory: {path}")
    return int(match.group(1))


class CheckpointEvalTaskRunner(TaskRunner):
    """TaskRunner that reuses workers while evaluating a checkpoint series."""

    def run(self, config):
        checkpoint_root = Path(os.environ["CHECKPOINT_EVAL_ROOT"]).resolve()
        results_dir = Path(os.environ["CHECKPOINT_EVAL_RESULTS"]).resolve()
        results_dir.mkdir(parents=True, exist_ok=True)

        checkpoints = sorted(checkpoint_root.glob("global_step_*"), key=_checkpoint_step)
        checkpoints = [path for path in checkpoints if (path / "actor").is_dir()]
        if not checkpoints:
            raise FileNotFoundError(f"no actor checkpoints found under {checkpoint_root}")

        pending = [path for path in checkpoints if not (results_dir / f"step_{_checkpoint_step(path)}.json").exists()]
        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        print(f"Checkpoint root: {checkpoint_root}")
        print(f"Results dir: {results_dir}")
        print(f"Checkpoints: {len(checkpoints)}, pending: {len(pending)}")
        pprint(OmegaConf.to_container(config, resolve=True))
        if not pending:
            return

        OmegaConf.resolve(config)
        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        self.add_critic_worker(config)
        self.add_reward_model_worker(config)
        self.add_ref_policy_worker(config, actor_rollout_cls)

        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(self.role_worker_mapping),
            use_critic=need_critic(config),
        )

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )
        from verl.utils import hf_processor, hf_tokenizer
        from verl.utils.dataset.rl_dataset import collate_fn

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)
        reward_fn = load_reward_manager(
            config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
        )
        val_reward_fn = load_reward_manager(
            config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {})
        )
        resource_pool_manager = self.init_resource_pool_mgr(config)
        train_dataset = create_rl_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor,
            is_train=True,
            max_samples=config.data.get("train_max_samples", -1),
        )
        val_dataset = create_rl_dataset(
            config.data.val_files,
            config.data,
            tokenizer,
            processor,
            is_train=False,
            max_samples=config.data.get("val_max_samples", -1),
        )
        train_sampler = create_rl_sampler(config.data, train_dataset)
        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )
        trainer.init_workers()

        for checkpoint in pending:
            step = _checkpoint_step(checkpoint)
            output_path = results_dir / f"step_{step}.json"
            print(f"BACKFILL_START step={step} checkpoint={checkpoint}", flush=True)
            trainer.global_steps = step
            trainer.actor_rollout_wg.load_checkpoint(
                str(checkpoint / "actor"),
                del_local_after_load=False,
            )
            metrics = trainer._validate()
            record = {
                "checkpoint": str(checkpoint),
                "step": step,
                "metrics": metrics,
            }
            temporary_path = output_path.with_suffix(".json.tmp")
            with temporary_path.open("w", encoding="utf-8") as stream:
                json.dump(record, stream, ensure_ascii=False, sort_keys=True, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, output_path)
            print(f"BACKFILL_DONE step={step} metrics={json.dumps(metrics, sort_keys=True)}", flush=True)


@hydra.main(config_path="pkg://verl.trainer.config", config_name="ppo_trainer", version_base=None)
def main(config):
    auto_set_ascend_device_name(config)
    task_runner_class = ray.remote(num_cpus=1)(CheckpointEvalTaskRunner)
    run_ppo(config, task_runner_class=task_runner_class)


if __name__ == "__main__":
    main()
