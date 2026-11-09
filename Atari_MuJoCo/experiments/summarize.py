"""Summarize completed MuJoCo runs with NumPy-only statistics.

Writes summary.json and summary.md to a new --output-dir. Both report training
tail returns separately from independent policy-evaluation returns. This tool
reads results only; it does not train or evaluate policies.

Examples (from this script's directory)::

    python summarize.py --input exported_results.json --output-dir fresh_summary
    python summarize.py --results-root runs --output-dir fresh_summary

Seed standard deviations use ddof=1, so at least two matched seeds are required.
Blocks contain ten consecutive requested seeds, starting at --seed-start. Any
remaining seeds are explicitly listed and retained in the all-seed analysis.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


SETTINGS = {"2048_1m": 2048, "4096_2m": 4096}
ENVIRONMENTS = ("Humanoid-v4", "Ant-v4", "HalfCheetah-v4", "Walker2d-v4", "Hopper-v4")
VARIANTS = ("ppo", "full", "unweighted", "random_mean", "guided_mean")
METRICS = ("training_tail", "independent_eval")
PAIRS = (("full", "ppo"), ("unweighted", "ppo"), ("full", "unweighted"),
         ("full", "random_mean"), ("full", "guided_mean"), ("guided_mean", "random_mean"))
ITERATIONS = 488
TAIL_POINTS = 100
EVAL_EPISODES = 10


class ValidationError(ValueError):
    """Results are missing, inconsistent, or unsuitable for the requested analysis."""


@dataclass(frozen=True)
class Run:
    setting: str
    mode: str
    variant: str
    env: str
    seed: int
    training_tail: float
    independent_eval: float

    @property
    def key(self) -> tuple[str, str, str, str, int]:
        return self.setting, self.mode, self.variant, self.env, self.seed


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def field(data: dict, name: str, context: str) -> Any:
    require(name in data, f"{context}: missing required field {name!r}")
    return data[name]


def mapping(data: Any, context: str) -> dict:
    require(isinstance(data, dict), f"{context}: expected a JSON object")
    return data


def finite(value: Any, context: str) -> float:
    require(not isinstance(value, bool), f"{context}: boolean is not a numeric metric")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError(f"{context}: expected a finite number, got {value!r}") from exc
    require(math.isfinite(result), f"{context}: non-finite value {value!r}")
    return result


def integer(value: Any, context: str) -> int:
    result = finite(value, context)
    require(result.is_integer(), f"{context}: expected an integer, got {value!r}")
    return int(result)


def assert_equal(value: Any, expected: Any, context: str) -> None:
    require(value == expected, f"{context}: expected {expected!r}, got {value!r}")


def assert_close(value: float, expected: float, context: str) -> None:
    require(math.isclose(value, expected, rel_tol=1e-10, abs_tol=1e-8),
            f"{context}: reported {value!r} disagrees with computed {expected!r}")


def reject_nonfinite(data: Any, context: str) -> None:
    if isinstance(data, float):
        require(math.isfinite(data), f"{context}: non-finite JSON number")
    elif isinstance(data, dict):
        for key, value in data.items():
            reject_nonfinite(value, f"{context}.{key}")
    elif isinstance(data, list):
        for index, value in enumerate(data):
            reject_nonfinite(value, f"{context}[{index}]")


def read_json(path: Path) -> Any:
    def unique_keys(items: list[tuple[str, Any]]) -> dict:
        result = {}
        for key, value in items:
            require(key not in result, f"{path}: duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=unique_keys)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"Cannot read JSON {path}: {exc}") from exc
    reject_nonfinite(data, str(path))
    return data


def normalized_setting(value: Any, mode: str, context: str) -> str:
    require(isinstance(value, str), f"{context}: setting must be a string")
    result = value.removeprefix(mode + "_")
    require(result in SETTINGS, f"{context}: unknown setting {value!r}; expected {list(SETTINGS)}")
    return result


def identity(data: dict, context: str) -> tuple[str, str, str, str, int]:
    mode = field(data, "mode", context)
    require(isinstance(mode, str) and bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", mode)),
            f"{context}: invalid mode {mode!r}")
    setting = normalized_setting(field(data, "setting", context), mode, context)
    variant, env = field(data, "variant", context), field(data, "env", context)
    require(variant in VARIANTS, f"{context}: unknown variant {variant!r}")
    require(env in ENVIRONMENTS, f"{context}: unknown environment {env!r}")
    seed = integer(field(data, "seed", context), f"{context}.seed")
    require(seed >= 1, f"{context}: seed must be positive")
    return setting, mode, variant, env, seed


def numeric_list(data: Any, count: int, context: str) -> list[float]:
    require(isinstance(data, list) and len(data) == count,
            f"{context}: expected exactly {count} entries")
    return [finite(value, f"{context}[{index}]") for index, value in enumerate(data)]


def load_snapshot(path: Path, mode: str, seed_start: int, seed_end: int) -> tuple[list[Run], dict]:
    data = mapping(read_json(path), str(path))
    for name, expected in (("environments", ENVIRONMENTS), ("variants", VARIANTS)):
        actual = field(data, name, str(path))
        require(isinstance(actual, list) and sorted(actual) == sorted(expected),
                f"{path}: {name} must contain each of {list(expected)} exactly once")
    rows = field(data, "runs", str(path))
    require(isinstance(rows, list), f"{path}.runs: expected a list")
    if "validated_jobs" in data:
        assert_equal(integer(data["validated_jobs"], "validated_jobs"), len(rows), "validated_jobs")
    runs, seen = [], set()
    excluded_mode = excluded_seeds = 0
    for index, raw in enumerate(rows):
        context = f"{path.name}:runs[{index}]"
        row = mapping(raw, context)
        key = identity(row, context)
        require(key not in seen, f"{context}: duplicate run/seed identity {key}")
        seen.add(key)
        setting, row_mode, variant, env, seed = key
        if row_mode != mode:
            excluded_mode += 1
            continue
        if not seed_start <= seed <= seed_end:
            excluded_seeds += 1
            continue
        assert_equal(integer(field(row, "iterations", context), context + ".iterations"),
                     ITERATIONS, context + ".iterations")
        assert_equal(integer(field(row, "final_step", context), context + ".final_step"),
                     ITERATIONS * SETTINGS[setting], context + ".final_step")
        returns = numeric_list(field(row, "eval_episode_returns", context), EVAL_EPISODES,
                               context + ".eval_episode_returns")
        evaluation = finite(field(row, "independent_eval", context), context + ".independent_eval")
        assert_close(evaluation, float(np.mean(returns)), context + ".independent_eval")
        if "eval_episode_lengths" in row:
            lengths = numeric_list(row["eval_episode_lengths"], EVAL_EPISODES,
                                   context + ".eval_episode_lengths")
            require(all(x.is_integer() and 1 <= x <= 1000 for x in lengths),
                    f"{context}: evaluation lengths must be integers in [1, 1000]")
        for name in ("eval_terminated_count", "eval_truncated_count"):
            if name in row:
                value = integer(row[name], context + "." + name)
                require(0 <= value <= EVAL_EPISODES, f"{context}.{name}: outside [0, 10]")
        runs.append(Run(setting, row_mode, variant, env, seed,
                        finite(field(row, "training_tail", context), context + ".training_tail"),
                        evaluation))
    source = {
        "kind": "snapshot", "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_records": len(rows), "excluded_other_modes": excluded_mode,
        "excluded_outside_seed_range": excluded_seeds,
        "validation": "Checked run identities, iterations, final steps, finite returns, and ten-episode means. "
                      "The imported snapshot contains summary metrics but no raw curves or checkpoint payloads.",
    }
    return runs, source


def resolve_artifact(value: Any, root: Path, job: Path, context: str) -> Path:
    """Resolve job-relative, root-relative, or relocated absolute artifact paths."""
    require(isinstance(value, str) and bool(value.strip()), f"{context}: expected a nonempty path")
    portable = value.replace("\\", "/")
    raw = Path(portable)
    candidates = [raw] if raw.is_absolute() else [job / raw, root / raw, Path.cwd() / raw]
    # Results copied between machines can contain absolute paths from their source location.
    marker = job.relative_to(root).as_posix() + "/"
    marked = "/" + portable.lstrip("/")
    suffix = marked.split("/" + marker, 1)
    if len(suffix) == 2:
        candidates.append(job / suffix[1])
    existing = {candidate.resolve() for candidate in candidates if candidate.is_file()}
    require(bool(existing), f"{context}: artifact does not exist: {value!r}")
    outside = [str(candidate) for candidate in existing if not candidate.is_relative_to(job)]
    require(not outside, f"{context}: artifact belongs outside this job {job}: {outside}")
    require(len(existing) == 1, f"{context}: ambiguous artifact path {value!r}: {sorted(map(str, existing))}")
    return existing.pop()


def load_job(job: Path, root: Path, key: tuple[str, str, str, str, int]) -> Run:
    setting, mode, variant, env, seed = key
    context = job.relative_to(root).as_posix()
    config = mapping(read_json(job / "config.json"), context + "/config.json")
    complete = mapping(read_json(job / "complete.json"), context + "/complete.json")
    assert_equal(identity(config, context + "/config.json"), key, context + ": config identity")
    assert_equal(integer(field(config, "expected_iterations", context), context + ".expected_iterations"),
                 ITERATIONS, context + ".expected_iterations")
    for name, expected in zip(("setting", "mode", "variant", "env", "seed"), key):
        if name in complete:
            actual = complete[name]
            if name == "setting":
                actual = normalized_setting(actual, mode, context)
            elif name == "seed":
                actual = integer(actual, context + ".complete.seed")
            assert_equal(actual, expected, context + ".complete." + name)
    final_step = ITERATIONS * SETTINGS[setting]
    for obj, label in ((config, "config"), (complete, "complete")):
        for name, expected in (("iterations", ITERATIONS), ("expected_iterations", ITERATIONS),
                               ("final_step", final_step), ("expected_final_step", final_step)):
            if name in obj:
                assert_equal(integer(obj[name], context + "." + label + "." + name), expected,
                             context + "." + label + "." + name)
    for name in ("returncode", "return_code"):
        if name in complete:
            assert_equal(integer(complete[name], context + "." + name), 0, context + "." + name)

    paths = {name: resolve_artifact(field(complete, name, context), root, job, context + "." + name)
             for name in ("train_curve", "final_checkpoint", "independent_eval")}
    curves = list((job / "train").rglob(f"{env}_{seed}.json"))
    require(len(curves) == 1, f"{context}: expected one training curve {env}_{seed}.json, found {len(curves)}")
    assert_equal(paths["train_curve"], curves[0].resolve(), context + ": completion training curve")
    assert_equal(paths["independent_eval"], (job / "independent_eval/results.json").resolve(),
                 context + ": completion evaluation path")
    curve = read_json(paths["train_curve"])
    require(isinstance(curve, list) and len(curve) == ITERATIONS,
            f"{context}: training curve must contain exactly {ITERATIONS} points")
    returns = []
    for index, raw in enumerate(curve, 1):
        point_context = f"{context}:training point {index}"
        point = mapping(raw, point_context)
        assert_equal(integer(field(point, "step", point_context), point_context + ".step"),
                     index * SETTINGS[setting], point_context + ".step")
        returns.append(finite(field(point, "mean_return", point_context), point_context + ".mean_return"))
        for name in ("min_return", "max_return"):
            if name in point:
                finite(point[name], point_context + "." + name)

    evaluation = mapping(read_json(paths["independent_eval"]), context + ":evaluation")
    for name, expected in (("env_id", env), ("metric", "independent_raw_environment_return"),
                           ("policy", "deterministic_actor_mean_no_tree_search")):
        assert_equal(field(evaluation, name, context), expected, context + ".evaluation." + name)
    for name, expected in (("training_seed", seed), ("checkpoint_step", final_step)):
        assert_equal(integer(field(evaluation, name, context), context + ".evaluation." + name),
                     expected, context + ".evaluation." + name)
    checkpoint = resolve_artifact(field(evaluation, "source_checkpoint", context), root, job,
                                  context + ".evaluation.source_checkpoint")
    assert_equal(checkpoint, paths["final_checkpoint"], context + ": evaluation/completion checkpoint")
    timeout = field(evaluation, "timeout_bootstrap", context)
    require(isinstance(timeout, bool), f"{context}: timeout_bootstrap must be boolean")
    if mode == "zero":
        assert_equal(timeout, False, context + ".evaluation.timeout_bootstrap")
    elif mode == "bootstrap":
        assert_equal(timeout, True, context + ".evaluation.timeout_bootstrap")
    if "timeout_bootstrap" in config:
        assert_equal(timeout, config["timeout_bootstrap"], context + ": timeout mode")
    episodes = field(evaluation, "episodes", context)
    require(isinstance(episodes, list) and len(episodes) == EVAL_EPISODES,
            f"{context}: independent evaluation must have exactly {EVAL_EPISODES} episodes")
    episode_returns = []
    episode_seeds = []
    for index, raw in enumerate(episodes):
        ep_context = f"{context}:evaluation episode {index}"
        episode = mapping(raw, ep_context)
        episode_returns.append(finite(field(episode, "return", ep_context), ep_context + ".return"))
        episode_seeds.append(integer(field(episode, "seed", ep_context), ep_context + ".seed"))
        require(1 <= integer(field(episode, "length", ep_context), ep_context + ".length") <= 1000,
                f"{ep_context}: length must be within the 1000-step limit")
        for name in ("terminated", "truncated"):
            require(isinstance(field(episode, name, ep_context), bool), f"{ep_context}.{name}: expected boolean")
        require(episode["terminated"] or episode["truncated"], f"{ep_context}: episode is unfinished")
    expected_seeds = [100000 + seed * 100 + index for index in range(EVAL_EPISODES)]
    assert_equal(sorted(episode_seeds), expected_seeds, context + ": evaluation reset seeds (including duplicates)")
    mean = finite(field(evaluation, "mean_return", context), context + ".evaluation.mean_return")
    assert_close(mean, float(np.mean(episode_returns)), context + ".evaluation.mean_return")
    if "std_return" in evaluation:
        assert_close(finite(evaluation["std_return"], context + ".evaluation.std_return"),
                     float(np.std(episode_returns, ddof=1)), context + ".evaluation.std_return")
    return Run(setting, mode, variant, env, seed, float(np.mean(returns[-TAIL_POINTS:])), mean)


def load_results_root(root: Path, mode: str, seed_start: int, seed_end: int) -> tuple[list[Run], dict]:
    root = root.resolve()
    require((root / "jobs").is_dir(), f"Missing jobs directory: {root / 'jobs'}")
    runs, missing = [], []
    excluded_seeds = 0
    for setting in SETTINGS:
        for variant in VARIANTS:
            for env in ENVIRONMENTS:
                parent = root / "jobs" / setting / mode / variant / env
                jobs: dict[int, Path] = {}
                if parent.is_dir():
                    for path in sorted(parent.glob("seed_*")):
                        match = re.fullmatch(r"seed_(\d+)", path.name)
                        require(bool(match) and path.is_dir(), f"Invalid seed directory: {path}")
                        seed = int(match.group(1))
                        require(seed not in jobs, f"Duplicate seed {seed} under {parent}: {jobs.get(seed)} and {path}")
                        jobs[seed] = path
                for seed, path in jobs.items():
                    if not seed_start <= seed <= seed_end:
                        excluded_seeds += 1
                        continue
                    assert_equal(path.name, f"seed_{seed:02d}", f"{path}: canonical seed directory")
                    require(path.resolve().is_relative_to(root), f"Job directory escapes results root: {path}")
                for seed in range(seed_start, seed_end + 1):
                    if seed not in jobs:
                        missing.append(f"{setting}/{mode}/{variant}/{env}/seed_{seed:02d}")
                    else:
                        runs.append(load_job(jobs[seed].resolve(), root, (setting, mode, variant, env, seed)))
    require(not missing, f"Missing {len(missing)} required jobs/seeds; first entries: {', '.join(missing[:12])}")
    return runs, {
        "kind": "results_root", "excluded_outside_seed_range": excluded_seeds,
        "validation": "Config/completion identities, artifact references, 488 exact training steps, "
                      "ten complete evaluation episodes, reset seeds, final checkpoint step and means checked. "
                      "Checkpoint payloads are not deserialized.",
    }


def validate_matching(runs: list[Run], mode: str, seeds: list[int]) -> dict:
    lookup = {}
    for run in runs:
        require(run.key not in lookup, f"Duplicate run/seed identity {run.key}")
        for metric in METRICS:
            finite(getattr(run, metric), f"{run.key}: {metric}")
        lookup[run.key] = run
    expected = {(setting, mode, variant, env, seed) for setting in SETTINGS
                for variant in VARIANTS for env in ENVIRONMENTS for seed in seeds}
    missing, extra = expected - lookup.keys(), lookup.keys() - expected
    require(not missing and not extra,
            f"Required matched seeds/tasks are incomplete: {len(missing)} missing, {len(extra)} unexpected; "
            f"missing examples={sorted(missing)[:8]}; unexpected examples={sorted(extra)[:8]}")
    return lookup


def seed_groups(seed_start: int, seed_end: int) -> tuple[list[tuple[int, int]], list[int]]:
    blocks = [(start, start + 9) for start in range(seed_start, seed_end - 8, 10)]
    groups = list(dict.fromkeys([(seed_start, seed_end)] + blocks))
    remainder_start = seed_start + 10 * len(blocks)
    return groups, list(range(remainder_start, seed_end + 1))


def group_indices(mode: str, setting: str, lo: int, hi: int, bootstrap: int, bootstrap_seed: int) -> list[np.ndarray]:
    # Use a stable group identifier so bootstrap resampling is reproducible across processes.
    key = f"group:{mode}_{setting}:{lo}:{hi}"
    words = np.frombuffer(hashlib.sha256(key.encode("utf-8")).digest()[:16], dtype="<u4")
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([bootstrap_seed] + words.tolist())))
    return [rng.integers(0, hi - lo + 1, size=(bootstrap, hi - lo + 1)) for _ in ENVIRONMENTS]


def interval(values: np.ndarray) -> list[float]:
    require(bool(np.isfinite(values).all()), "Non-finite bootstrap statistic")
    return np.quantile(values, [0.025, 0.975]).tolist()


def summarize_metric(values: np.ndarray, indices: list[np.ndarray], seeds: list[int], context: str) -> dict:
    require(values.shape == (len(ENVIRONMENTS), len(VARIANTS), len(seeds)), f"{context}: invalid matched array shape")
    require(len(seeds) >= 2, f"{context}: at least two seeds are required for sample standard deviation")
    require(bool(np.isfinite(values).all()), f"{context}: non-finite input return")
    means = values.mean(axis=2)
    stds = values.std(axis=2, ddof=1)
    require(bool(np.isfinite(means).all() and np.isfinite(stds).all()), f"{context}: non-finite mean/std")
    draws = np.stack([np.take(values[ei], indices[ei], axis=1).mean(axis=2)
                      for ei in range(len(ENVIRONMENTS))])
    require(bool(np.isfinite(draws).all()), f"{context}: non-finite bootstrap mean")
    result = {"seed_range": [seeds[0], seeds[-1]], "seeds": seeds, "n_seeds": len(seeds),
              "by_environment": {}, "macro": {}}
    for ei, env in enumerate(ENVIRONMENTS):
        result["by_environment"][env] = {
            "variants": {variant: {"mean": float(means[ei, vi]), "std": float(stds[ei, vi]),
                                   "n_seeds": len(seeds)} for vi, variant in enumerate(VARIANTS)},
            "comparisons": {},
        }
    for lhs, rhs in PAIRS:
        li, ri = VARIANTS.index(lhs), VARIANTS.index(rhs)
        key = f"{lhs}_vs_{rhs}"
        for ei, env in enumerate(ENVIRONMENTS):
            require(means[ei, ri] > 0, f"{context}/{env}/{key}: nonpositive comparison denominator {means[ei, ri]}")
            invalid = np.flatnonzero(draws[ei, ri] <= 0)
            require(not len(invalid), f"{context}/{env}/{key}: nonpositive bootstrap denominator "
                    f"in {len(invalid)} resamples; no resamples may be dropped")
        relative = 100.0 * (means[:, li] / means[:, ri] - 1.0)
        relative_draws = 100.0 * (draws[:, li] / draws[:, ri] - 1.0)
        require(bool(np.isfinite(relative).all()), f"{context}/{key}: non-finite relative comparison")
        result["macro"][key] = {
            "relative_percent": float(relative.mean()), "ci95_percent": interval(relative_draws.mean(axis=0)),
            "n_tasks": len(ENVIRONMENTS), "task_weight": 1.0 / len(ENVIRONMENTS),
            "denominator_variant": rhs,
            "denominator_means": {env: float(means[ei, ri]) for ei, env in enumerate(ENVIRONMENTS)},
        }
        for ei, env in enumerate(ENVIRONMENTS):
            diffs = values[ei, li] - values[ei, ri]
            result["by_environment"][env]["comparisons"][key] = {
                "lhs": lhs, "rhs": rhs, "lhs_mean": float(means[ei, li]),
                "denominator_mean": float(means[ei, ri]), "denominator_n_seeds": len(seeds),
                "relative_percent": float(relative[ei]), "ci95_percent": interval(relative_draws[ei]),
                "paired_difference_mean": float(diffs.mean()),
                "paired_difference_ci95": interval(draws[ei, li] - draws[ei, ri]),
            }
    return result


def summarize_runs(runs: list[Run], mode: str = "zero", seed_start: int = 1, seed_end: int = 50,
                   bootstrap: int = 5000, bootstrap_seed: int = 20271109) -> dict:
    require(seed_start >= 1 and seed_end > seed_start, "Require 1 <= seed-start < seed-end (sample std uses ddof=1)")
    require(bootstrap >= 1 and bootstrap_seed >= 0, "bootstrap must be positive and bootstrap-seed nonnegative")
    seeds = list(range(seed_start, seed_end + 1))
    lookup = validate_matching(runs, mode, seeds)
    groups, remainder = seed_groups(seed_start, seed_end)
    report = {
        "schema_version": 1, "mode": mode, "seed_range": [seed_start, seed_end], "seeds": seeds,
        "settings": {setting: {"num_steps": steps, "iterations": ITERATIONS,
                               "final_step": ITERATIONS * steps,
                               "nominal_timesteps": 1000000 if setting == "2048_1m" else 2000000}
                     for setting, steps in SETTINGS.items()},
        "environments": list(ENVIRONMENTS), "variants": list(VARIANTS), "validated_jobs": len(runs),
        "metric_definitions": {
            "training_tail": "Within each training seed, mean of the final 100 of 488 logged mean_return points. "
                             "For tree variants, including unweighted, these are branch-weighted per-tree returns; "
                             "this is not an independent policy evaluation.",
            "independent_eval": "Within each training seed, mean raw return of 10 deterministic actor-mean "
                                "episodes, without tree search, from the final checkpoint.",
        },
        "statistics": {
            "std_ddof": 1, "unit": "training seed", "training_tail_points": TAIL_POINTS,
            "evaluation_episodes_per_seed": EVAL_EPISODES,
            "task_comparison": "100 * (mean(lhs seed returns) / mean(rhs seed returns) - 1)",
            "absolute_difference": "mean(lhs seed return - rhs seed return) on matched seeds",
            "aggregate": "equal arithmetic mean of the five task comparison percentages; task weight 1/5",
            "bootstrap_replicates": bootstrap, "bootstrap_seed": bootstrap_seed,
            "bootstrap_method": "paired seed resampling within each task; independent resampling across tasks; "
                                "shared indices across methods and both metrics; no pooling of tasks or episodes",
            "bootstrap_rng": "NumPy PCG64/SeedSequence([base_seed, SHA256 group-key first four little-endian uint32s])",
            "bootstrap_group_key": "group:{mode}_{setting}:{first_seed}:{last_seed}",
            "ci": "95% percentile interval, quantiles 0.025 and 0.975, linear interpolation; not multiplicity adjusted",
            "denominator_policy": "positive observed and every bootstrap rhs mean; reject, never clip or drop",
            "blocks": "non-overlapping blocks of 10 starting at seed-start; remainder retained in all-seed group",
            "block_remainder_seeds": remainder,
        },
        "groups": {},
    }
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        for setting in SETTINGS:
            report["groups"][setting] = {}
            for lo, hi in groups:
                group_seeds = list(range(lo, hi + 1))
                indices = group_indices(mode, setting, lo, hi, bootstrap, bootstrap_seed)
                metrics = {}
                for metric in METRICS:
                    values = np.array([[[getattr(lookup[setting, mode, variant, env, seed], metric)
                                         for seed in group_seeds] for variant in VARIANTS]
                                       for env in ENVIRONMENTS], dtype=np.float64)
                    metrics[metric] = summarize_metric(values, indices, group_seeds,
                                                       f"{setting}/{mode}/seeds{lo}_{hi}/{metric}")
                report["groups"][setting][f"seeds{lo}_{hi}"] = metrics
    return report


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
                     + ["| " + " | ".join(row) + " |" for row in rows])


def render_markdown(report: dict) -> str:
    stats = report["statistics"]
    lines = ["# MuJoCo matched-seed summary", "",
             f"Mode `{report['mode']}`; seeds {report['seed_range'][0]}-{report['seed_range'][1]} inclusive; "
             f"{report['validated_jobs']} validated jobs; five tasks, five variants, two settings.", "",
             "Settings: `2048_1m` = 488 iterations / final step 999424; "
             "`4096_2m` = 488 iterations / final step 1998848.", "",
             report["metric_definitions"]["training_tail"], "",
             report["metric_definitions"]["independent_eval"], "",
             "Mean +/- SD is across training seeds (sample SD, ddof=1). Each seed contributes one value per metric. "
             "Evaluation episodes and training points are never counted as independent seed samples.", "",
             "Task gain = 100 * (lhs mean / rhs mean - 1); the exact denominator is the rhs task/seed-group mean. "
             "The macro gain is the equal mean of five task gains (weight 1/5), not a ratio of pooled returns. "
             "Task comparison cells show percent gain / absolute paired difference.", "",
             f"95% percentile CIs: {stats['bootstrap_replicates']} paired seed resamples per task; "
             f"base seed {stats['bootstrap_seed']}; independently sampled tasks, shared method/metric indices. "
             "No nonpositive observed or resampled denominator is accepted. Full-precision denominators and "
             "per-task percentage/difference CIs are in summary.json. Intervals are not multiplicity adjusted.", "",
             "Blocks start at the requested first seed and do not overlap. "
             f"Remainder seeds included only in the all-seed group: {stats['block_remainder_seeds']}.", ""]
    if "source" in report:
        lines.extend([f"Source: `{report['source']['path']}` (relative to this output directory when possible).",
                      report["source"]["validation"], ""])
    for setting, groups in report["groups"].items():
        for group, metrics in groups.items():
            for metric, result in metrics.items():
                lo, hi = result["seed_range"]
                lines.extend([f"## {setting} / {metric} / seeds {lo}-{hi} (n={result['n_seeds']})", ""])
                rows = [[env] + [f"{result['by_environment'][env]['variants'][v]['mean']:.6g} +/- "
                                 f"{result['by_environment'][env]['variants'][v]['std']:.6g}" for v in VARIANTS]
                        for env in ENVIRONMENTS]
                lines.extend([markdown_table(["Task: mean +/- SD"] + list(VARIANTS), rows), ""])
                rows = []
                for lhs, rhs in PAIRS:
                    key = f"{lhs}_vs_{rhs}"
                    macro = result["macro"][key]
                    low, high = macro["ci95_percent"]
                    row = [f"{lhs} / {rhs}", f"{macro['relative_percent']:+.3f}% [{low:+.3f}, {high:+.3f}]"]
                    for env in ENVIRONMENTS:
                        comp = result["by_environment"][env]["comparisons"][key]
                        row.append(f"{comp['relative_percent']:+.3f}% / {comp['paired_difference_mean']:+.3f}")
                    rows.append(row)
                lines.extend([markdown_table(["lhs / rhs", "Five-task gain [95% CI]"] + list(ENVIRONMENTS), rows), ""])
    return "\n".join(lines).rstrip() + "\n"


def write_outputs(report: dict, output_dir: Path) -> None:
    # Serialize before creating any outputs; an existing directory is never reused.
    content = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    markdown = render_markdown(report)
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise ValidationError(f"Refusing to overwrite existing output directory: {output_dir}") from exc
    (output_dir / "summary.json").write_text(content, encoding="utf-8")
    (output_dir / "summary.md").write_text(markdown, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="exported results snapshot JSON")
    source.add_argument("--results-root", type=Path, help="new run root containing jobs/{setting}/{mode}/...")
    parser.add_argument("--output-dir", type=Path, required=True, help="NEW output directory; existing directories are refused")
    parser.add_argument("--mode", default="zero")
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--seed-end", type=int, default=50)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20271109)
    args = parser.parse_args(argv)
    try:
        require(bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.mode)), "mode must be a single path component")
        require(1 <= args.seed_start < args.seed_end, "Require 1 <= seed-start < seed-end (sample std uses ddof=1)")
        require(args.bootstrap > 0 and args.bootstrap_seed >= 0, "bootstrap must be positive and bootstrap-seed nonnegative")
        require(not args.output_dir.exists(), f"Refusing to overwrite existing output directory: {args.output_dir}")
        source_path = (args.input if args.input is not None else args.results_root).resolve()
        loader = load_snapshot if args.input is not None else load_results_root
        runs, provenance = loader(source_path, args.mode, args.seed_start, args.seed_end)
        report = summarize_runs(runs, args.mode, args.seed_start, args.seed_end, args.bootstrap, args.bootstrap_seed)
        try:
            provenance["path"] = Path(os.path.relpath(source_path, args.output_dir.resolve())).as_posix()
        except ValueError:  # Different Windows drives cannot have a relative path.
            provenance["path"] = source_path.as_posix()
        report["source"] = provenance
        write_outputs(report, args.output_dir)
    except (ValidationError, OSError, FloatingPointError) as exc:
        parser.error(str(exc))
    print(f"Validated {len(runs)} matched jobs; wrote {args.output_dir / 'summary.json'} and {args.output_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
