"""Save training checkpoints for evaluation and reproducibility.

Each checkpoint contains policy and optimizer state, observation/reward
normalization statistics, and RNG state. The job output directory is provided
by the experiment worker via the AXAP_JOB_DIR environment variable.
"""
from pathlib import Path
import os
import numpy as np
import torch
import gymnasium as gym


def _wrapper(root, desired):
    while root is not None:
        if isinstance(root, desired):
            return root
        root = getattr(root, "env", None)
    raise RuntimeError(f"Missing training wrapper {desired.__name__}")


def save_checkpoint(agent, optimizer, args, iteration, global_step, envs):
    target = os.environ.get("AXAP_JOB_DIR")
    if not target:
        raise RuntimeError("Missing AXAP_JOB_DIR; configure a checkpoint output directory")
    if args.checkpoint_every < 1:
        raise ValueError("checkpoint_every must be positive")
    env0 = envs.envs[0] if hasattr(envs, "envs") else envs[0]
    obs = _wrapper(env0, gym.wrappers.NormalizeObservation).obs_rms
    ret = _wrapper(env0, gym.wrappers.NormalizeReward).return_rms
    payload = {
        "model_state_dict": agent.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "obs_rms_mean": np.array(obs.mean, copy=True),
        "obs_rms_var": np.array(obs.var, copy=True),
        "obs_rms_count": float(obs.count),
        "ret_rms_mean": np.array(ret.mean, copy=True),
        "ret_rms_var": np.array(ret.var, copy=True),
        "ret_rms_count": float(ret.count),
        "iteration": iteration,
        "global_step": global_step,
        "env_id": args.env_id,
        "train_seed": args.seed,
        "timeout_bootstrap": args.timeout_bootstrap,
        "args": dict(vars(args)),
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
    }
    directory = Path(target) / "checkpoints"
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / f"iteration_{iteration:04d}_step_{global_step:08d}.pt"
    pending = output.with_suffix(".pt.part")
    torch.save(payload, pending)
    os.replace(pending, output)
    print(f"AXAP_CHECKPOINT={output}", flush=True)
