"""Run and validate one MuJoCo training configuration on CPU.

The worker writes a training log, periodic checkpoints, and an independent
evaluation of the final policy. Use run_queue.py to schedule multiple workers.
"""
from pathlib import Path
import argparse
import json
import math
import subprocess
import sys
import time
import traceback

from mujoco_config import (HERE, SETTINGS, MODES, SCRIPTS, ENVS, atomic_json, cpu_environment,
                           exclusive_lock, job_dir, job_spec, sha256, source_hashes, training_command)


def validate_artifacts(directory, spec):
    """Reject missing or inconsistent curves, checkpoints, and evaluation results."""
    curves = list((directory / "train").rglob(f"{spec['env']}_{spec['seed']}.json"))
    if len(curves) != 1:
        raise ValueError(f"Expected exactly one training curve, found {curves}")
    curve = json.loads(curves[0].read_text(encoding="utf-8"))
    expected = list(range(spec["num_steps"], spec["expected_final_step"] + 1, spec["num_steps"]))
    if [int(row["step"]) for row in curve] != expected:
        raise ValueError("Incomplete or inconsistent training steps")
    if any(not math.isfinite(float(row["mean_return"])) for row in curve):
        raise ValueError("Nonfinite training return")
    iterations = sorted(set(range(spec["checkpoint_every"], spec["expected_iterations"] + 1,
                                  spec["checkpoint_every"])) | {spec["expected_iterations"]})
    checkpoints = [directory / "checkpoints" / f"iteration_{i:04d}_step_{i * spec['num_steps']:08d}.pt" for i in iterations]
    actual = set((directory / "checkpoints").glob("iteration_*.pt"))
    if actual != set(checkpoints) or any(p.stat().st_size == 0 for p in actual):
        raise ValueError("Missing, empty, or unexpected checkpoint files")
    evaluation = directory / "independent_eval" / "results.json"
    result = json.loads(evaluation.read_text(encoding="utf-8"))
    expected_identity = {"env_id": spec["env"], "training_seed": spec["seed"],
                         "checkpoint_step": spec["expected_final_step"],
                         "timeout_bootstrap": spec["mode"] == "bootstrap",
                         "metric": "independent_raw_environment_return",
                         "policy": "deterministic_actor_mean_no_tree_search"}
    for key, value in expected_identity.items():
        if result[key] != value:
            raise ValueError(f"Evaluation identity mismatch for {key}")
    episodes = result["episodes"]
    if len(episodes) != spec["independent_eval_episodes"]:
        raise ValueError("Independent evaluation episode count mismatch")
    for index, episode in enumerate(episodes):
        if (episode["seed"] != 100000 + spec["seed"] * 100 + index or
                not 1 <= episode["length"] <= 1000 or
                not (episode["terminated"] or episode["truncated"]) or
                not math.isfinite(float(episode["return"]))):
            raise ValueError("Invalid evaluation episode or mismatched evaluation seed")
    mean = sum(float(ep["return"]) for ep in episodes) / len(episodes)
    if not math.isclose(mean, result["mean_return"], rel_tol=1e-12, abs_tol=1e-9):
        raise ValueError("Evaluation mean differs from per-episode returns")
    return curves[0], checkpoints, evaluation


def run(spec, results_root, retry_failed=False):
    """Run one job; verify completed jobs before skipping or retrying them."""
    directory = job_dir(results_root, spec)
    directory.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(directory / "worker.lock"):
        identity = {**spec, "source_sha256": source_hashes(spec["variant"])}
        config_path = directory / "config.json"
        completed = directory / "complete.json"
        if completed.exists():
            saved = json.loads(config_path.read_text(encoding="utf-8"))
            if any(saved.get(key) != value for key, value in identity.items()):
                raise ValueError(f"Existing completed job has different settings or source: {directory}")
            curve, checkpoints, evaluation = validate_artifacts(directory, spec)
            complete = json.loads(completed.read_text(encoding="utf-8"))
            expected_files = {p.relative_to(directory).as_posix(): sha256(p) for p in [curve, *checkpoints, evaluation]}
            if complete["artifact_sha256"] != expected_files:
                raise ValueError(f"Completed-job artifacts changed: {directory}")
            print(f"VERIFIED_SKIP {directory}", flush=True)
            return
        leftovers = [p for p in directory.iterdir() if p.name not in ("worker.lock", "previous_attempts")]
        if leftovers:
            if not retry_failed:
                raise FileExistsError(f"Incomplete attempt exists at {directory}; inspect it, then use --retry-failed")
            archive = directory / "previous_attempts" / str(time.time_ns())
            archive.mkdir(parents=True)
            for path in leftovers:
                path.rename(archive / path.name)
        command = training_command(spec, directory)
        atomic_json(config_path, {**identity, "source": SCRIPTS[spec["variant"]], "command": command,
                                  "started_at": time.time()})
        environment = cpu_environment(directory)
        try:
            with (directory / "training.log").open("w", encoding="utf-8") as log:
                subprocess.run(command, cwd=directory, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
            eval_command = [sys.executable, str(HERE / "evaluate.py"), str(directory),
                            "--episodes", str(spec["independent_eval_episodes"])]
            with (directory / "evaluation.log").open("w", encoding="utf-8") as log:
                subprocess.run(eval_command, cwd=directory, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
            curve, checkpoints, evaluation = validate_artifacts(directory, spec)
            atomic_json(completed, {
                "completed_at": time.time(), "train_curve": curve.relative_to(directory).as_posix(),
                "checkpoints": len(checkpoints), "final_checkpoint": checkpoints[-1].relative_to(directory).as_posix(),
                "independent_eval": evaluation.relative_to(directory).as_posix(),
                "artifact_sha256": {p.relative_to(directory).as_posix(): sha256(p) for p in [curve, *checkpoints, evaluation]},
            })
            print(f"COMPLETE {directory}", flush=True)
        except Exception as error:
            atomic_json(directory / "failure.json", {"error": str(error), "timestamp": time.time(),
                                                       "traceback": traceback.format_exc()})
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("setting", choices=SETTINGS)
    parser.add_argument("mode", choices=MODES)
    parser.add_argument("variant", choices=SCRIPTS)
    parser.add_argument("env", choices=ENVS)
    parser.add_argument("seed", type=int)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    spec = job_spec(args.setting, args.mode, args.variant, args.env, args.seed, args.smoke, args.eval_episodes)
    if args.dry_run:
        print(json.dumps({**spec, "command": training_command(spec, job_dir(args.results_root, spec))}, indent=2))
    else:
        run(spec, args.results_root, args.retry_failed)


if __name__ == "__main__":
    main()
