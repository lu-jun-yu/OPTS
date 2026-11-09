"""Check statistical summaries and input validation with temporary test data."""
from __future__ import annotations

import contextlib
import copy
import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

import summarize as s


def put_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def fixture_job(root: Path, setting: str = "2048_1m", variant: str = "ppo",
                env: str = "Humanoid-v4", seed: int = 1) -> Path:
    job = root / "jobs" / setting / "zero" / variant / env / f"seed_{seed:02d}"
    identity = dict(setting=setting, mode="zero", variant=variant, env=env, seed=seed)
    put_json(job / "config.json", dict(identity, expected_iterations=s.ITERATIONS))
    curve = [{"step": str(i * s.SETTINGS[setting]), "mean_return": str(i + seed),
              "min_return": str(i + seed - 1), "max_return": str(i + seed + 1)}
             for i in range(1, s.ITERATIONS + 1)]
    train = job / "train" / "fixture" / f"{env}_{seed}.json"
    put_json(train, curve)
    checkpoint = job / "checkpoints/final.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fixture: checkpoint is intentionally not deserialized")
    episodes = [{"seed": 100000 + 100 * seed + i, "return": float(100 + seed + i),
                 "length": 10, "terminated": True, "truncated": False} for i in range(10)]
    put_json(job / "independent_eval/results.json", {
        "metric": "independent_raw_environment_return", "policy": "deterministic_actor_mean_no_tree_search",
        "training_seed": seed, "env_id": env, "timeout_bootstrap": False,
        "checkpoint_step": s.ITERATIONS * s.SETTINGS[setting],
        "source_checkpoint": "checkpoints/final.pt", "episodes": episodes,
        "mean_return": float(np.mean([ep["return"] for ep in episodes])),
        "std_return": float(np.std([ep["return"] for ep in episodes], ddof=1)),
    })
    put_json(job / "complete.json", dict(identity, train_curve=train.relative_to(job).as_posix(),
                                        final_checkpoint="checkpoints/final.pt",
                                        independent_eval="independent_eval/results.json"))
    return job


def fixture_runs() -> list[s.Run]:
    return [s.Run(setting, "zero", variant, env, seed,
                  float(100 + 10 * vi + seed), float(200 + 20 * vi + seed))
            for setting in s.SETTINGS for vi, variant in enumerate(s.VARIANTS)
            for env in s.ENVIRONMENTS for seed in (1, 2)]


def fixture_snapshot() -> dict:
    rows = []
    for run in fixture_runs():
        row = dict(run.__dict__)
        row.update(setting="zero_" + run.setting, iterations=s.ITERATIONS,
                   final_step=s.ITERATIONS * s.SETTINGS[run.setting],
                   eval_episode_returns=[run.independent_eval] * 10)
        rows.append(row)
    return {"environments": list(s.ENVIRONMENTS), "variants": list(s.VARIANTS),
            "validated_jobs": len(rows), "runs": rows}


class StatisticsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="mujoco_statistics_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def metric(self, values: np.ndarray, indices: list[np.ndarray] | None = None) -> dict:
        if indices is None:
            indices = s.group_indices("zero", "2048_1m", 1, values.shape[-1], 100, 20271109)
        return s.summarize_metric(values, indices, list(range(1, values.shape[-1] + 1)), "fixture")

    def test_matching_requires_every_seed_and_task(self) -> None:
        runs = fixture_runs()
        self.assertEqual(len(s.validate_matching(runs, "zero", [1, 2])), 100)
        with self.assertRaisesRegex(s.ValidationError, "1 missing"):
            s.validate_matching(runs[:-1], "zero", [1, 2])
        with self.assertRaisesRegex(s.ValidationError, "20 missing"):
            s.validate_matching([r for r in runs if r.env != "Hopper-v4"], "zero", [1, 2])

    def test_duplicate_and_unexpected_seeds_fail(self) -> None:
        runs = fixture_runs()
        with self.assertRaisesRegex(s.ValidationError, "Duplicate"):
            s.validate_matching(runs + [runs[0]], "zero", [1, 2])
        with self.assertRaisesRegex(s.ValidationError, "unexpected"):
            s.validate_matching(runs + [replace(runs[0], seed=3)], "zero", [1, 2])

    def test_ratio_of_means_and_absolute_difference(self) -> None:
        values = np.full((5, 5, 2), 5.0)
        values[:, 0] = [1.0, 9.0]
        values[:, 1] = [2.0, 9.0]
        report = self.metric(values)
        comp = report["by_environment"]["Ant-v4"]["comparisons"]["full_vs_ppo"]
        self.assertAlmostEqual(comp["relative_percent"], 10.0)
        self.assertEqual(comp["denominator_mean"], 5.0)
        self.assertEqual(comp["denominator_n_seeds"], 2)
        self.assertEqual(comp["paired_difference_mean"], 0.5)
        self.assertAlmostEqual(report["by_environment"]["Ant-v4"]["variants"]["ppo"]["std"],
                               np.std([1.0, 9.0], ddof=1))

    def test_equal_task_weight_and_pairing(self) -> None:
        values = np.empty((5, 5, 4))
        for task in range(5):
            values[task] = (10.0 ** task) * np.arange(1.0, 5.0)
            values[task, 1] = values[task, 0] * (1 + task / 10)
        report = self.metric(values)
        comp = report["macro"]["full_vs_ppo"]
        self.assertAlmostEqual(comp["relative_percent"], 20.0)
        # Perfect within-task proportionality yields zero bootstrap ratio variation.
        np.testing.assert_allclose(comp["ci95_percent"], [20, 20], atol=1e-12)

    def test_bootstrap_matches_small_explicit_reference(self) -> None:
        rng = np.random.default_rng(3)
        values = rng.uniform(1, 9, size=(5, 5, 4))
        indices = s.group_indices("zero", "2048_1m", 1, 4, 200, 20271109)
        report = self.metric(values, indices)
        relative, difference = [], []
        for draw in range(200):
            task_relative, task_difference = [], []
            for task in range(5):
                lhs = np.mean(values[task, 1, indices[task][draw]])
                rhs = np.mean(values[task, 0, indices[task][draw]])
                task_relative.append(100 * (lhs / rhs - 1))
                task_difference.append(lhs - rhs)
            relative.append(task_relative)
            difference.append(task_difference)
        relative, difference = np.asarray(relative), np.asarray(difference)
        np.testing.assert_allclose(report["macro"]["full_vs_ppo"]["ci95_percent"],
                                   np.quantile(relative.mean(axis=1), [0.025, 0.975]))
        for ei, env in enumerate(s.ENVIRONMENTS):
            comp = report["by_environment"][env]["comparisons"]["full_vs_ppo"]
            np.testing.assert_allclose(comp["ci95_percent"], np.quantile(relative[:, ei], [0.025, 0.975]))
            np.testing.assert_allclose(comp["paired_difference_ci95"],
                                       np.quantile(difference[:, ei], [0.025, 0.975]))

    def test_task_resamples_are_independent_and_reproducible(self) -> None:
        a = s.group_indices("zero", "2048_1m", 1, 10, 100, 7)
        b = s.group_indices("zero", "2048_1m", 1, 10, 100, 7)
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)
        self.assertFalse(np.array_equal(a[0], a[1]))

    def test_nonpositive_point_denominator_fails(self) -> None:
        for denominator in (0.0, -1.0):
            with self.subTest(denominator=denominator):
                values = np.ones((5, 5, 2))
                values[0, 0] = denominator
                with self.assertRaisesRegex(s.ValidationError, "nonpositive comparison denominator"):
                    self.metric(values)

    def test_nonpositive_bootstrap_denominator_fails_without_dropping(self) -> None:
        values = np.ones((5, 5, 2))
        values[0, 0] = [-1, 3]  # Observed mean is positive, first resample is not.
        indices = [np.array([[0, 0], [1, 1]]) for _ in range(5)]
        with self.assertRaisesRegex(s.ValidationError, "nonpositive bootstrap denominator"):
            self.metric(values, indices)

    def test_nan_metric_fails(self) -> None:
        runs = fixture_runs()
        runs[0] = replace(runs[0], training_tail=float("nan"))
        with self.assertRaisesRegex(s.ValidationError, "non-finite"):
            s.validate_matching(runs, "zero", [1, 2])

    def test_block_remainder_is_explicit_and_not_discarded(self) -> None:
        groups, remainder = s.seed_groups(3, 24)
        self.assertEqual(groups, [(3, 24), (3, 12), (13, 22)])
        self.assertEqual(remainder, [23, 24])

    def test_snapshot_validation_and_cli(self) -> None:
        path = self.root / "snapshot.json"
        put_json(path, fixture_snapshot())
        output = self.root / "summary"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(s.main(["--input", str(path), "--output-dir", str(output),
                                     "--seed-start", "1", "--seed-end", "2", "--bootstrap", "50"]), 0)
        report = s.read_json(output / "summary.json")
        self.assertEqual(report["validated_jobs"], 100)
        self.assertEqual(set(report["groups"]["2048_1m"]["seeds1_2"]), set(s.METRICS))
        self.assertEqual(report["source"]["path"], "../snapshot.json")
        markdown = (output / "summary.md").read_text(encoding="utf-8")
        self.assertIn("including unweighted", markdown)
        self.assertIn("guided_mean / random_mean", markdown)
        before = (output / "summary.json").read_bytes()
        with self.assertRaisesRegex(s.ValidationError, "Refusing to overwrite"):
            s.write_outputs(report, output)
        self.assertEqual(before, (output / "summary.json").read_bytes())

    def test_snapshot_duplicate_and_eval_mean_fail(self) -> None:
        data = fixture_snapshot()
        data["runs"].append(copy.deepcopy(data["runs"][0]))
        data["validated_jobs"] += 1
        path = self.root / "snapshot.json"
        put_json(path, data)
        with self.assertRaisesRegex(s.ValidationError, "duplicate run/seed"):
            s.load_snapshot(path, "zero", 1, 2)
        data = fixture_snapshot()
        data["runs"][0]["independent_eval"] += 1
        put_json(path, data)
        with self.assertRaisesRegex(s.ValidationError, "disagrees"):
            s.load_snapshot(path, "zero", 1, 2)

    def test_job_training_tail_is_separate_from_eval(self) -> None:
        job = fixture_job(self.root)
        run = s.load_job(job, self.root, ("2048_1m", "zero", "ppo", "Humanoid-v4", 1))
        self.assertAlmostEqual(run.training_tail, np.mean(np.arange(389, 489) + 1))
        self.assertAlmostEqual(run.independent_eval, 105.5)
        self.assertNotEqual(run.training_tail, run.independent_eval)

    def test_config_and_completion_identity_failures(self) -> None:
        job = fixture_job(self.root)
        key = ("2048_1m", "zero", "ppo", "Humanoid-v4", 1)
        for filename in ("config.json", "complete.json"):
            with self.subTest(filename=filename):
                data = s.read_json(job / filename)
                original = copy.deepcopy(data)
                data["seed"] = 2
                put_json(job / filename, data)
                with self.assertRaisesRegex(s.ValidationError, "identity|complete.seed"):
                    s.load_job(job, self.root, key)
                put_json(job / filename, original)

    def test_bad_training_step_count_and_nonfinite_string_fail(self) -> None:
        job = fixture_job(self.root)
        path = job / "train/fixture/Humanoid-v4_1.json"
        curve = s.read_json(path)
        key = ("2048_1m", "zero", "ppo", "Humanoid-v4", 1)
        for kind, message in (("count", "488 points"), ("step", "point 2.step"), ("nan", "non-finite")):
            with self.subTest(kind=kind):
                changed = copy.deepcopy(curve)
                if kind == "count":
                    changed.pop()
                elif kind == "step":
                    changed[1]["step"] = changed[0]["step"]
                else:
                    changed[0]["mean_return"] = "nan"
                put_json(path, changed)
                with self.assertRaisesRegex(s.ValidationError, message):
                    s.load_job(job, self.root, key)
        put_json(path, curve)

    def test_eval_count_step_and_duplicate_reset_seed_fail(self) -> None:
        job = fixture_job(self.root)
        path = job / "independent_eval/results.json"
        evaluation = s.read_json(path)
        key = ("2048_1m", "zero", "ppo", "Humanoid-v4", 1)
        for kind, message in (("count", "exactly 10"), ("step", "checkpoint_step"), ("seed", "reset seeds")):
            with self.subTest(kind=kind):
                changed = copy.deepcopy(evaluation)
                if kind == "count":
                    changed["episodes"].pop()
                elif kind == "step":
                    changed["checkpoint_step"] -= 1
                else:
                    changed["episodes"][1]["seed"] = changed["episodes"][0]["seed"]
                put_json(path, changed)
                with self.assertRaisesRegex(s.ValidationError, message):
                    s.load_job(job, self.root, key)

    def test_absolute_root_relative_and_relocated_paths(self) -> None:
        job = fixture_job(self.root)
        artifact = job / "checkpoints/final.pt"
        paths = [str(artifact), artifact.relative_to(self.root).as_posix(),
                 "/old/machine/" + artifact.relative_to(self.root).as_posix(),
                 "Z:\\old\\machine\\" + str(artifact.relative_to(self.root)).replace("/", "\\")]
        for path in paths:
            with self.subTest(path=path):
                self.assertEqual(s.resolve_artifact(path, self.root, job, "fixture"), artifact.resolve())
        outside = self.root / "wrong.pt"
        outside.write_bytes(b"not this job")
        with self.assertRaisesRegex(s.ValidationError, "outside this job"):
            s.resolve_artifact(str(outside), self.root, job, "fixture")

    def test_missing_and_duplicate_seed_directories_fail(self) -> None:
        job = fixture_job(self.root)
        with self.assertRaisesRegex(s.ValidationError, "Missing 99 required jobs"):
            s.load_results_root(self.root, "zero", 1, 2)
        (job.parent / "seed_1").mkdir()
        with self.assertRaisesRegex(s.ValidationError, "Duplicate seed 1"):
            s.load_results_root(self.root, "zero", 1, 2)

    def test_duplicate_curves_and_missing_checkpoint_fail(self) -> None:
        job = fixture_job(self.root)
        key = ("2048_1m", "zero", "ppo", "Humanoid-v4", 1)
        other = job / "train/other/Humanoid-v4_1.json"
        put_json(other, s.read_json(job / "train/fixture/Humanoid-v4_1.json"))
        with self.assertRaisesRegex(s.ValidationError, "found 2"):
            s.load_job(job, self.root, key)
        other.unlink()
        (job / "checkpoints/final.pt").unlink()
        with self.assertRaisesRegex(s.ValidationError, "artifact does not exist"):
            s.load_job(job, self.root, key)

    def test_new_folder_cli_all_tasks_variants_and_settings(self) -> None:
        for setting in s.SETTINGS:
            for variant in s.VARIANTS:
                for env in s.ENVIRONMENTS:
                    for seed in (1, 2):
                        fixture_job(self.root, setting, variant, env, seed)
        output = self.root / "summary"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(s.main(["--results-root", str(self.root), "--output-dir", str(output),
                                     "--seed-start", "1", "--seed-end", "2", "--bootstrap", "50"]), 0)
        report = s.read_json(output / "summary.json")
        self.assertEqual(report["validated_jobs"], 100)
        for setting in s.SETTINGS:
            for metric in s.METRICS:
                group = report["groups"][setting]["seeds1_2"][metric]
                self.assertEqual(len(group["by_environment"]), 5)
                self.assertEqual(len(group["macro"]), 6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
