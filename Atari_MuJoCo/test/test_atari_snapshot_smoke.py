"""Low-resource Atari snapshot determinism check.

This verifies that AtariStateSnapshotWrapper can restore a saved state well
enough that a fixed future action sequence produces the same trajectory.
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
CLEANRL_ROOT = REPO_ROOT / "cleanrl"
OPTS_ATARI_PATH = CLEANRL_ROOT / "cleanrl" / "opts_ttpo_atari.py"


def load_opts_ttpo_atari():
    sys.path.insert(0, str(CLEANRL_ROOT))
    spec = importlib.util.spec_from_file_location("opts_ttpo_atari_under_test", OPTS_ATARI_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {OPTS_ATARI_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def step_signature(obs, reward, terminated, truncated, info):
    episode = info.get("episode")
    if episode is None:
        episode_sig = None
    else:
        ep_r = episode.get("r")
        ep_l = episode.get("l")
        episode_sig = (
            float(ep_r[0] if hasattr(ep_r, "__getitem__") else ep_r),
            int(ep_l[0] if hasattr(ep_l, "__getitem__") else ep_l),
        )
    return {
        "obs": np.asarray(obs).copy(),
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "episode": episode_sig,
    }


def rollout(env, actions):
    trajectory = []
    for action in actions:
        obs, reward, terminated, truncated, info = env.step(int(action))
        trajectory.append(step_signature(obs, reward, terminated, truncated, info))
        if terminated or truncated:
            break
    return trajectory


def compare_trajectories(a, b):
    if len(a) != len(b):
        return False, f"trajectory length differs: {len(a)} != {len(b)}"
    for idx, (left, right) in enumerate(zip(a, b), start=1):
        if not np.array_equal(left["obs"], right["obs"]):
            diff = np.abs(left["obs"].astype(np.int16) - right["obs"].astype(np.int16)).max()
            return False, f"obs differs at branch step {idx}, max_abs_pixel_diff={diff}"
        for key in ("reward", "terminated", "truncated", "episode"):
            if left[key] != right[key]:
                return False, f"{key} differs at branch step {idx}: {left[key]} != {right[key]}"
    return True, "matched"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default="BreakoutNoFrameskip-v4")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--warmup-steps", type=int, default=8)
    parser.add_argument("--branch-steps", type=int, default=32)
    parser.add_argument("--capture-video", action="store_true")
    args = parser.parse_args()

    opts_atari = load_opts_ttpo_atari()
    env = opts_atari.make_env(
        args.env_id,
        idx=0,
        capture_video=args.capture_video,
        run_name="test_atari_snapshot_smoke",
    )()

    try:
        action_n = env.action_space.n
        for trial in range(args.trials):
            rng = np.random.default_rng(args.seed + trial)
            env.reset(seed=args.seed + trial)

            warmup_actions = rng.integers(0, action_n, size=args.warmup_steps)
            warmup = rollout(env, warmup_actions)
            if warmup and (warmup[-1]["terminated"] or warmup[-1]["truncated"]):
                env.reset(seed=args.seed + trial + 10_000)

            snapshot = env.clone_state()
            branch_actions = rng.integers(0, action_n, size=args.branch_steps)

            first = rollout(env, branch_actions)
            env.restore_state(snapshot)
            second = rollout(env, branch_actions)

            ok, message = compare_trajectories(first, second)
            status = "PASS" if ok else "FAIL"
            print(f"[{status}] trial={trial} env={args.env_id} steps={len(first)}: {message}")
            if not ok:
                raise SystemExit(1)
    finally:
        env.close()

    print("All snapshot restore checks passed.")


if __name__ == "__main__":
    main()
