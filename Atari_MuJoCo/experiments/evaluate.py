"""Evaluate the last saved policy checkpoint without training-time tree search.

Evaluation freezes observation normalization, uses deterministic actor-mean
actions, and records undiscounted environment return for each episode.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import json
import sys

import gymnasium as gym
import numpy as np
import torch

from mujoco_config import ALGORITHMS, atomic_json, sha256

sys.path.insert(0, str(ALGORITHMS))
from ppo_continuous_action import Agent


def evaluate(job_dir, episodes=10, output=None):
    """Evaluate the final checkpoint in a job directory and save episode returns."""
    job_dir = Path(job_dir).resolve()
    if episodes < 1:
        raise ValueError("episodes must be positive")
    output = Path(output) if output is not None else job_dir / "independent_eval" / "results.json"
    if output.exists():
        raise FileExistsError(f"Evaluation already exists; use --output for a separate result: {output}")
    checkpoints = sorted((job_dir / "checkpoints").glob("iteration_*.pt"),
                         key=lambda p: int(p.name.split("_")[1]))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints under {job_dir}")
    checkpoint_path = checkpoints[-1]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    env_id, seed = checkpoint["env_id"], checkpoint["train_seed"]
    config_path = job_dir / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        expected_step = config["expected_iterations"] * checkpoint["args"]["num_steps"] * checkpoint["args"]["num_envs"]
        if (env_id != config["env"] or seed != config["seed"] or
                bool(checkpoint["timeout_bootstrap"]) != (config["mode"] == "bootstrap") or
                checkpoint["iteration"] != config["expected_iterations"] or checkpoint["global_step"] != expected_step):
            raise ValueError("Final checkpoint does not match the job configuration")
    obs_mean = np.asarray(checkpoint["obs_rms_mean"], dtype=np.float64)
    obs_var = np.asarray(checkpoint["obs_rms_var"], dtype=np.float64)
    if (not np.all(np.isfinite(obs_mean)) or not np.all(np.isfinite(obs_var)) or
            np.any(obs_var < 0) or obs_mean.shape != obs_var.shape):
        raise ValueError("Invalid saved observation normalization")
    env = gym.wrappers.ClipAction(gym.wrappers.FlattenObservation(gym.make(env_id)))
    history = []
    try:
        if env.spec.max_episode_steps != 1000:
            raise ValueError(f"Expected the original 1000-step evaluation horizon, got {env.spec.max_episode_steps}")
        if obs_mean.shape != env.observation_space.shape:
            raise ValueError("Checkpoint normalization shape does not match the environment")
        agent = Agent(SimpleNamespace(single_observation_space=env.observation_space,
                                      single_action_space=env.action_space)).cpu()
        agent.load_state_dict(checkpoint["model_state_dict"], strict=True)
        agent.eval()
        for ep in range(episodes):
            independent_seed = 100000 + seed * 100 + ep
            observation, _ = env.reset(seed=independent_seed)
            total = 0.0
            for steps in range(1, 1001):
                normalized = np.clip((np.asarray(observation, dtype=np.float64) - obs_mean) /
                                     np.sqrt(obs_var + 1e-8), -10, 10).astype(np.float32)
                with torch.no_grad():
                    action = agent.actor_mean(torch.from_numpy(normalized).unsqueeze(0)).squeeze(0).numpy()
                if not np.all(np.isfinite(action)):
                    raise ValueError("Policy produced a nonfinite evaluation action")
                observation, reward, terminated, truncated, _ = env.step(action)
                total += float(reward)
                if not np.isfinite(total):
                    raise ValueError("Nonfinite independent evaluation return")
                if terminated or truncated:
                    history.append({"seed": independent_seed, "return": total, "length": steps,
                                    "terminated": bool(terminated), "truncated": bool(truncated)})
                    break
            else:
                raise RuntimeError("Environment did not terminate or truncate at its 1000-step limit")
    finally:
        env.close()
    result = {
        "metric": "independent_raw_environment_return",
        "policy": "deterministic_actor_mean_no_tree_search",
        "training_seed": seed, "source_checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path), "checkpoint_step": checkpoint["global_step"],
        "env_id": env_id, "timeout_bootstrap": checkpoint["timeout_bootstrap"], "episodes": history,
        "mean_return": float(np.mean([x["return"] for x in history])),
        "std_return": float(np.std([x["return"] for x in history], ddof=1)) if len(history) > 1 else 0.0,
    }
    atomic_json(output, result)
    print(f"INDEPENDENT_EVAL {env_id} seed={seed} mean={result['mean_return']:.3f}", flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_dir", type=Path)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    evaluate(args.job_dir, args.episodes, args.output)
