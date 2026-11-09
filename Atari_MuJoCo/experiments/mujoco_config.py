"""Shared MuJoCo experiment settings and helpers used by the job runner."""
from contextlib import contextmanager
from pathlib import Path
import hashlib
import json
import os
import sys

HERE = Path(__file__).resolve().parent
ALGORITHMS = HERE.parent / "cleanrl" / "cleanrl"
SETTINGS = {
    "2048_1m": (2048, 1_000_000),
    "4096_2m": (4096, 2_000_000),
    "8192_4m": (8192, 4_000_000),
}
# The 8192/4M configuration is opt-in; the default sweep uses the two smaller budgets.
DEFAULT_SETTINGS = ("2048_1m", "4096_2m")
ENVS = ("Humanoid-v4", "Ant-v4", "HalfCheetah-v4", "Walker2d-v4", "Hopper-v4")
MODES = ("zero", "bootstrap")
SCRIPTS = {
    "ppo": "ppo_continuous_action.py",
    "full": "opts_ttpo_continuous_action_wEqual-bMax-nLen.py",
    "unweighted": "opts_ttpo_continuous_action_wEqual-bMax-nLen.py",
    "random_mean": "opts_ttpo_continuous_action_wEqual-bMean-nLen.py",
    "guided_mean": "opts_ttpo_continuous_action_wEqual-bMean-nLen.py",
}
EXTRA = {
    "ppo": [],
    "full": ["--baseline", "mean", "--xi", "0.6", "--max-search-per-tree", "1"],
    "unweighted": ["--aggregation", "none", "--baseline", "mean", "--xi", "0.6", "--max-search-per-tree", "1"],
    "random_mean": ["--branch-selection", "random", "--max-search-per-tree", "1"],
    "guided_mean": ["--branch-selection", "performance_difference", "--max-search-per-tree", "1"],
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".part")
    pending.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(pending, path)


def job_spec(setting, mode, variant, env, seed, smoke=False, eval_episodes=10):
    """Validate a run configuration and derive its exact training/evaluation sizes."""
    if setting not in SETTINGS or mode not in MODES or variant not in SCRIPTS or env not in ENVS:
        raise ValueError(f"Unknown experiment: {setting}, {mode}, {variant}, {env}")
    if seed < 1 or eval_episodes < 1:
        raise ValueError("Training seed and evaluation episode count must be positive")
    if not smoke and eval_episodes != 10:
        raise ValueError("Formal runs require the matched 10-episode evaluation protocol")
    num_steps, total_steps = SETTINGS[setting]
    if smoke:
        total_steps = 2 * num_steps
    iterations = total_steps // num_steps
    return {
        "setting": setting, "mode": mode, "variant": variant, "env": env, "seed": seed,
        "smoke": smoke, "num_steps": num_steps, "total_timesteps": total_steps,
        "num_envs": 1, "num_minibatches": 32, "update_epochs": 10,
        "minibatch_size": num_steps // 32, "expected_iterations": iterations,
        "expected_final_step": iterations * num_steps,
        "optimizer_steps": iterations * 32 * 10, "checkpoint_every": 1 if smoke else 50,
        "independent_eval_episodes": eval_episodes,
        "eval_policy": "deterministic_actor_mean_no_tree_search",
        "training_summary": ("ordinary_training_episode_return" if variant == "ppo" else
                             "branch_weighted_return_within_tree_then_equal_mean_across_trees"),
    }


def job_dir(results_root, spec):
    return (Path(results_root).resolve() / ("smoke" if spec["smoke"] else "jobs") /
            spec["setting"] / spec["mode"] / spec["variant"] / spec["env"] / f"seed_{spec['seed']:02d}")


def training_command(spec, directory):
    """Build a command that runs the corresponding CleanRL algorithm entrypoint."""
    command = [sys.executable, str(ALGORITHMS / SCRIPTS[spec["variant"]]),
               "--env-id", spec["env"], "--total-timesteps", str(spec["total_timesteps"]),
               "--num-steps", str(spec["num_steps"]), "--num-envs", "1",
               "--num-minibatches", "32", "--update-epochs", "10", "--no-cuda",
               "--seed", str(spec["seed"]), "--results-root", str(directory / "train"),
               "--checkpoint-every", str(spec["checkpoint_every"])]
    if spec["mode"] == "bootstrap":
        command.append("--timeout-bootstrap")
    return command + EXTRA[spec["variant"]]


def source_hashes(variant):
    """Identify the implementation and helper files used for this method."""
    files = [ALGORITHMS / SCRIPTS[variant], ALGORITHMS / SCRIPTS["ppo"],
             HERE / "axap_checkpoint.py", HERE / "evaluate.py", HERE / "worker.py", HERE / "mujoco_config.py"]
    if variant != "ppo":
        core = "bMax" if variant in ("full", "unweighted") else "bMean"
        files.append(ALGORITHMS / f"opts_ttpo_core_wEqual_{core}.py")
    return {p.relative_to(HERE.parent).as_posix(): sha256(p) for p in files}


def cpu_environment(directory):
    """Give each worker its own output directory and a single numeric CPU thread."""
    result = os.environ.copy()
    result.update(AXAP_JOB_DIR=str(directory), CUDA_VISIBLE_DEVICES="", PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "TORCH_NUM_THREADS"):
        result[key] = "1"
    return result


@contextmanager
def exclusive_lock(path):
    """Nonblocking OS lock; lock ownership is released even after a crashed process."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        if os.name == "nt":
            import msvcrt
            if path.stat().st_size == 0:
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if os.name == "nt":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)
