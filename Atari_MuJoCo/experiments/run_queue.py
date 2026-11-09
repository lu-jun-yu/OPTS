"""Schedule MuJoCo training and evaluation jobs with bounded CPU concurrency.

Choose environments, methods, rollout budgets, timeout handling, and seeds via
command-line flags. Use --dry-run to inspect the planned workload first.
"""
from collections import Counter, deque
from pathlib import Path
import argparse
import json
import os
import signal
import subprocess
import sys
import time

from mujoco_config import HERE, SETTINGS, DEFAULT_SETTINGS, MODES, ENVS, SCRIPTS, atomic_json, exclusive_lock, job_spec


def planned_jobs(args):
    """Expand the selected configurations into one job per method/task/seed."""
    if not 1 <= args.max_workers <= 80:
        raise ValueError("max-workers must be between 1 and 80")
    if not 1 <= args.seed_start <= args.seed_end:
        raise ValueError("Invalid seed range")
    for values in (args.settings, args.modes, args.variants, args.envs):
        if len(values) != len(set(values)):
            raise ValueError("Duplicate setting/mode/variant/environment in queue")
    return [job_spec(setting, mode, variant, env, seed, args.smoke, args.eval_episodes)
            for setting in args.settings for seed in range(args.seed_start, args.seed_end + 1)
            for env in args.envs for mode in args.modes for variant in args.variants]


def stop_worker(process):
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if os.name != "nt":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.wait()


def run(args, jobs):
    """Execute jobs without exceeding max_workers, including evaluation time."""
    root = args.results_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(root / "queue.lock"):
        run_id = str(time.time_ns())
        atomic_json(root / f"queue_manifest_{run_id}.json", jobs)
        pending = deque(jobs)
        active = {}
        done = 0
        failed = []
        logs = root / "queue_logs"
        logs.mkdir(exist_ok=True)
        try:
            while pending or active:
                paused = (root / "PAUSE_NEW_JOBS").exists()
                while pending and len(active) < args.max_workers and not paused:
                    spec = pending.popleft()
                    command = [sys.executable, str(HERE / "worker.py"), spec["setting"], spec["mode"],
                               spec["variant"], spec["env"], str(spec["seed"]), "--results-root", str(root),
                               "--eval-episodes", str(args.eval_episodes)]
                    if args.smoke:
                        command.append("--smoke")
                    if args.retry_failed:
                        command.append("--retry-failed")
                    label = "_".join(str(spec[k]) for k in ("setting", "mode", "variant", "env", "seed"))
                    log = (logs / f"{run_id}_{label}.log").open("w", encoding="utf-8")
                    try:
                        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                                   start_new_session=os.name != "nt")
                    except BaseException:
                        log.close()
                        raise
                    active[process.pid] = (process, spec, log)
                for pid, (process, spec, log) in list(active.items()):
                    code = process.poll()
                    if code is not None:
                        del active[pid]
                        log.close()
                        if code == 0:
                            done += 1
                        else:
                            failed.append({"job": spec, "returncode": code})
                        print(f"EXIT pid={pid} rc={code} complete={done}/{len(jobs)} failed={len(failed)}", flush=True)
                atomic_json(root / "queue_status.json", {"pid": os.getpid(), "total": len(jobs), "done": done,
                            "failed": failed, "active": len(active), "pending": len(pending),
                            "max_workers": args.max_workers, "paused": paused, "updated_at": time.time()})
                if paused and not active:
                    print("Queue drained; remove PAUSE_NEW_JOBS and rerun to continue.", flush=True)
                    return 2
                if pending or active:
                    time.sleep(1)
        finally:
            for process, spec, log in active.values():
                stop_worker(process)
                log.close()
        return 1 if failed else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--settings", nargs="+", choices=SETTINGS, default=list(DEFAULT_SETTINGS))
    parser.add_argument("--modes", nargs="+", choices=MODES, default=["zero"])
    parser.add_argument("--variants", nargs="+", choices=SCRIPTS, default=list(SCRIPTS))
    parser.add_argument("--envs", nargs="+", choices=ENVS, default=list(ENVS))
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--seed-end", type=int, default=50)
    parser.add_argument("--max-workers", type=int, default=80)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    jobs = planned_jobs(args)
    if args.dry_run:
        print(json.dumps({"total": len(jobs), "by_setting": dict(Counter(s["setting"] for s in jobs)),
                          "by_variant": dict(Counter(s["variant"] for s in jobs)), "first": jobs[0], "last": jobs[-1]}, indent=2))
        return 0
    return run(args, jobs)


if __name__ == "__main__":
    sys.exit(main())
