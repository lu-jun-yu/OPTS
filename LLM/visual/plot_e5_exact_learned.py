#!/usr/bin/env python3
"""Build the paired exact/learned E5 figures from completed search trees.

The learned-critic panels mirror the four exact-critic quantities:
reward-guided and value-guided search, each evaluated by the full-reward
lambda objective and by verifier return.  All learned curves use lambda=.999.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXACT = REPO_ROOT / "LLM/results/e5_exact_deterministic/summary.json"
DEFAULT_REWARD = (
    REPO_ROOT
    / "LLM/results/step400/rq2/opts_ttpo_exp8_3_0810_n8_rq2_opts_s15_t32_bs902.parquet"
)
DEFAULT_VALUE = (
    REPO_ROOT
    / "LLM/results/step400/e5_learned_critic/opts_ttpo_exp8_3_0810_n8_reward-text-roots_s15_t32_bs902_value.parquet"
)
DEFAULT_REWARD_EVAL = (
    REPO_ROOT
    / "LLM/results/step400/rq2/eval_20260915/opts_ttpo_exp8_3_0810_n8_rq2_opts_s15_t32_bs902__opts-avg_s0-1-3-7-15_k32.json"
)
DEFAULT_VALUE_EVAL = (
    REPO_ROOT
    / "LLM/results/step400/e5_learned_critic/eval/opts_ttpo_exp8_3_0810_n8_reward-text-roots_s15_t32_bs902_value__opts_ttpo_exp8_3_0810_n8_reward-text-roots_s15_t32_bs902_value_opts-avg_s0-1-3-7-15_k32.json"
)
DEFAULT_METRICS = (
    REPO_ROOT / "LLM/results/step400/e5_learned_critic/e5_learned_critic_metrics.json"
)
DEFAULT_MAIN = REPO_ROOT / "paper/figures/e5_search_budget_exact_learned.pdf"
DEFAULT_APPENDIX = REPO_ROOT / "paper/figures/e5_learned_critic_by_dataset.pdf"

ROUNDS = (0, 1, 3, 7, 15)
ROUND_POSITIONS = tuple(math.log2(value + 1) for value in ROUNDS)
LAMBDA = 0.999
DATASETS = (
    "hiyouga/math12k",
    "math-ai/aime24",
    "math-ai/aime25",
    "math-ai/aime26",
    "math-ai/amc23",
    "math-ai/minervamath",
)
DATASET_NAMES = {
    "hiyouga/math12k": "MATH500",
    "math-ai/aime24": "AIME24",
    "math-ai/aime25": "AIME25",
    "math-ai/aime26": "AIME26",
    "math-ai/amc23": "AMC23",
    "math-ai/minervamath": "MinervaMath",
}


def rid_round(rid: str) -> int:
    return int(str(rid).split("_", 1)[0][1:])


def child_map(rids: list[str], pids: list[str | None], branch_pos: list[int]) -> dict:
    rid_to_index = {str(rid): index for index, rid in enumerate(rids)}
    children = defaultdict(list)
    for child, (pid, pos) in enumerate(zip(pids, branch_pos, strict=True)):
        if pid is not None:
            if str(pid) not in rid_to_index:
                raise ValueError(f"Missing parent rid {pid}")
            children[(rid_to_index[str(pid)], int(pos))].append(child)
    return dict(children)


def local_deltas_from_backed_up(
    advantages: list[list[float]], children: dict, lam: float
) -> list[np.ndarray]:
    """Recover local TD residuals from a fully backed-up max TreeGAE tree."""

    arrays = [np.asarray(values, dtype=np.float64) for values in advantages]
    deltas = []
    for node, values in enumerate(arrays):
        delta = values.copy()
        for position in range(len(values)):
            successors = []
            if position + 1 < len(values):
                successors.append(float(values[position + 1]))
            successors.extend(float(arrays[child][0]) for child in children.get((node, position), ()))
            if successors:
                delta[position] -= lam * max(successors)
        deltas.append(delta)
    return deltas


def backed_up_advantages(
    deltas: list[np.ndarray], children: dict, rounds: list[int], budget: int, lam: float
) -> list[np.ndarray | None]:
    """Recompute max TreeGAE on the nodes retained at one search budget."""

    included = [index for index, round_index in enumerate(rounds) if round_index <= budget]
    output: list[np.ndarray | None] = [None] * len(deltas)
    for node in sorted(included, key=lambda index: rounds[index], reverse=True):
        values = np.empty_like(deltas[node])
        for position in range(len(values) - 1, -1, -1):
            successors = []
            if position + 1 < len(values):
                successors.append(float(values[position + 1]))
            successors.extend(
                float(output[child][0])
                for child in children.get((node, position), ())
                if rounds[child] <= budget
            )
            values[position] = deltas[node][position] + (
                lam * max(successors) if successors else 0.0
            )
        output[node] = values
    return output


def ordered_uids(
    uids: list[str], rids: list[str], global_indices: list[int]
) -> list[str]:
    first_root = {}
    for uid, rid, global_index in zip(uids, rids, global_indices, strict=True):
        if rid_round(rid) == 0:
            first_root.setdefault(str(uid), int(global_index))
            first_root[str(uid)] = min(first_root[str(uid)], int(global_index))
    return sorted(first_root, key=first_root.__getitem__)


def select_root_and_terminal(
    uid: str,
    uids: list[str],
    pids: list[str | None],
    rounds: list[int],
    children: dict,
    advantages: list[np.ndarray | None],
    budget: int,
) -> tuple[int, int]:
    roots = [
        index
        for index, (candidate_uid, pid, round_index) in enumerate(zip(uids, pids, rounds, strict=True))
        if str(candidate_uid) == uid and pid is None and round_index <= budget
    ]
    if not roots:
        raise ValueError(f"No root for uid={uid} at s={budget}")
    root = max(roots, key=lambda index: float(advantages[index][0]))
    node = root
    position = 0
    while True:
        current = advantages[node]
        continuation = float(current[position + 1]) if position + 1 < len(current) else 0.0
        candidates = [
            child for child in children.get((node, position), ()) if rounds[child] <= budget
        ]
        if candidates:
            child = max(candidates, key=lambda index: float(advantages[index][0]))
            if float(advantages[child][0]) > continuation:
                node, position = child, 0
                continue
        if position + 1 >= len(current):
            return root, node
        position += 1


def target_sum_along_guidance(
    root: int,
    children: dict,
    rounds: list[int],
    guidance: list[np.ndarray | None],
    target_deltas: list[np.ndarray],
    budget: int,
    lam: float,
) -> tuple[float, int]:
    node = root
    position = 0
    discount = 1.0
    total = 0.0
    while True:
        total += discount * float(target_deltas[node][position])
        current = guidance[node]
        continuation = float(current[position + 1]) if position + 1 < len(current) else 0.0
        candidates = [
            child for child in children.get((node, position), ()) if rounds[child] <= budget
        ]
        if candidates:
            child = max(candidates, key=lambda index: float(guidance[index][0]))
            if float(guidance[child][0]) > continuation:
                node, position = child, 0
                discount *= lam
                continue
        if position + 1 >= len(current):
            return total, node
        position += 1
        discount *= lam


def true_reward_deltas(values: list[list[float]], rewards: list[float]) -> list[np.ndarray]:
    output = []
    for raw_values, reward in zip(values, rewards, strict=True):
        value = np.clip(np.asarray(raw_values, dtype=np.float64), 0.0, 1.0)
        delta = np.empty_like(value)
        if len(value) == 0:
            raise ValueError("Empty response value sequence")
        delta[:-1] = value[1:] - value[:-1]
        delta[-1] = float(reward) - value[-1]
        output.append(delta)
    return output


def read_eval(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if payload.get("opts_avg_slices") != list(ROUNDS):
        raise ValueError(f"Unexpected search rounds in {path}")
    return payload["results"]


def true_return_improvements(results: dict) -> dict[str, list[float]]:
    series = {}
    for dataset in DATASETS + ("_all",):
        values = [float(results[dataset][f"opts-avg_s{budget}@32"]) for budget in ROUNDS]
        series[dataset] = [value - values[0] for value in values]
    return series


def aggregate_tree_values(per_dataset: dict[str, dict[int, list[float]]]) -> dict:
    output = {}
    all_values = {budget: [] for budget in ROUNDS}
    for dataset in DATASETS:
        output[dataset] = []
        for budget in ROUNDS:
            values = per_dataset[dataset][budget]
            output[dataset].append(float(np.mean(values)))
            all_values[budget].extend(values)
    output["_all"] = [float(np.mean(all_values[budget])) for budget in ROUNDS]
    return output


def analyze_reward(path: Path) -> tuple[dict, dict]:
    import pyarrow.parquet as pq

    columns = [
        "data_source", "responses", "global_indices", "tree_uids", "tree_rids",
        "tree_pids", "tree_branch_pos", "tree_advantages",
        "opts_avg_responses_s1", "opts_avg_responses_s3", "opts_avg_responses_s7",
        "opts_avg_responses_s15",
    ]
    rows = pq.read_table(path, columns=columns).to_pylist()
    improvements = defaultdict(lambda: {budget: [] for budget in ROUNDS})
    max_reconstruction_error = 0.0
    snapshot_mismatches = 0
    tree_count = 0

    for row_index, row in enumerate(rows):
        rids = [str(value) for value in row["tree_rids"]]
        pids = [None if value is None else str(value) for value in row["tree_pids"]]
        uids = [str(value) for value in row["tree_uids"]]
        rounds = [rid_round(rid) for rid in rids]
        children = child_map(rids, pids, row["tree_branch_pos"])
        final = [np.asarray(values, dtype=np.float64) for values in row["tree_advantages"]]
        deltas = local_deltas_from_backed_up(row["tree_advantages"], children, LAMBDA)
        order = ordered_uids(uids, rids, row["global_indices"])
        tree_count += len(order)
        baseline = {}

        for budget in ROUNDS:
            guidance = backed_up_advantages(deltas, children, rounds, budget, LAMBDA)
            if budget == ROUNDS[-1]:
                for expected, actual in zip(final, guidance, strict=True):
                    max_reconstruction_error = max(
                        max_reconstruction_error,
                        float(np.max(np.abs(expected - actual))),
                    )
            selected_texts = []
            for uid in order:
                root, terminal = select_root_and_terminal(
                    uid, uids, pids, rounds, children, guidance, budget
                )
                value = float(guidance[root][0])
                if budget == 0:
                    baseline[uid] = value
                improvements[row["data_source"]][budget].append(value - baseline[uid])
                selected_texts.append(row["responses"][terminal])
            if budget > 0:
                expected_texts = list(row[f"opts_avg_responses_s{budget}"])
                snapshot_mismatches += sum(a != b for a, b in zip(selected_texts, expected_texts, strict=True))
        if (row_index + 1) % 100 == 0:
            print(f"reward analysis: {row_index + 1}/{len(rows)}", flush=True)

    audit = {
        "rows": len(rows),
        "trees": tree_count,
        "max_final_advantage_reconstruction_error": max_reconstruction_error,
        "snapshot_terminal_mismatches": snapshot_mismatches,
    }
    return aggregate_tree_values(improvements), audit


def analyze_value(path: Path, expected_true_returns: dict) -> tuple[dict, dict]:
    import pyarrow.parquet as pq

    columns = [
        "data_source", "responses", "global_indices", "tree_uids", "tree_rids",
        "tree_pids", "tree_branch_pos", "tree_rounds", "tree_values", "tree_rewards",
        *[f"tree_advantages_s{budget}" for budget in ROUNDS],
        *[f"opts_avg_responses_s{budget}" for budget in ROUNDS],
    ]
    rows = pq.read_table(path, columns=columns).to_pylist()
    improvements = defaultdict(lambda: {budget: [] for budget in ROUNDS})
    realized = defaultdict(lambda: {budget: [] for budget in ROUNDS})
    snapshot_mismatches = 0
    tree_count = 0

    for row_index, row in enumerate(rows):
        rids = [str(value) for value in row["tree_rids"]]
        pids = [None if value is None else str(value) for value in row["tree_pids"]]
        uids = [str(value) for value in row["tree_uids"]]
        rounds = [int(value) for value in row["tree_rounds"]]
        children = child_map(rids, pids, row["tree_branch_pos"])
        deltas = true_reward_deltas(row["tree_values"], row["tree_rewards"])
        order = ordered_uids(uids, rids, row["global_indices"])
        tree_count += len(order)
        baseline = {}

        for budget in ROUNDS:
            guidance = [
                np.asarray(values, dtype=np.float64) if len(values) else None
                for values in row[f"tree_advantages_s{budget}"]
            ]
            selected_texts = []
            for uid in order:
                root, terminal = select_root_and_terminal(
                    uid, uids, pids, rounds, children, guidance, budget
                )
                target, target_terminal = target_sum_along_guidance(
                    root, children, rounds, guidance, deltas, budget, LAMBDA
                )
                if terminal != target_terminal:
                    raise AssertionError("Guidance traversal returned inconsistent terminals")
                if budget == 0:
                    baseline[uid] = target
                improvements[row["data_source"]][budget].append(target - baseline[uid])
                realized[row["data_source"]][budget].append(float(row["tree_rewards"][terminal]))
                selected_texts.append(row["responses"][terminal])
            expected_texts = list(row[f"opts_avg_responses_s{budget}"])
            snapshot_mismatches += sum(a != b for a, b in zip(selected_texts, expected_texts, strict=True))
        if (row_index + 1) % 100 == 0:
            print(f"value analysis: {row_index + 1}/{len(rows)}", flush=True)

    realized_aggregate = aggregate_tree_values(realized)
    max_true_return_error = max(
        abs(realized_aggregate[dataset][index] - expected_true_returns[dataset][index])
        for dataset in DATASETS + ("_all",)
        for index in range(len(ROUNDS))
    )
    audit = {
        "rows": len(rows),
        "trees": tree_count,
        "snapshot_terminal_mismatches": snapshot_mismatches,
        "max_true_return_eval_error": max_true_return_error,
    }
    return aggregate_tree_values(improvements), audit


def analyze(args: argparse.Namespace) -> dict:
    reward_eval = read_eval(args.reward_eval)
    value_eval = read_eval(args.value_eval)
    reward_true = true_return_improvements(reward_eval)
    value_true = true_return_improvements(value_eval)
    reward_lambda, reward_audit = analyze_reward(args.reward_parquet)
    value_lambda, value_audit = analyze_value(
        args.value_parquet,
        {
            dataset: [float(value_eval[dataset][f"opts-avg_s{budget}@32"]) for budget in ROUNDS]
            for dataset in DATASETS + ("_all",)
        },
    )
    payload = {
        "protocol": {
            "lambda": LAMBDA,
            "search_rounds": list(ROUNDS),
            "aggregation": "micro average across 902 prompts and 32 trees per prompt",
            "j_lambda": "full verifier-reward lambda return under the frozen learned critic",
            "j": "answer-only verifier return of the guidance-greedy terminal response",
        },
        "series": {
            "reward": {"j_lambda": reward_lambda, "j": reward_true},
            "value": {"j_lambda": value_lambda, "j": value_true},
        },
        "audit": {"reward": reward_audit, "value": value_audit},
        "sources": {
            "reward_parquet": str(args.reward_parquet),
            "value_parquet": str(args.value_parquet),
            "reward_eval": str(args.reward_eval),
            "value_eval": str(args.value_eval),
        },
    }
    if reward_audit["snapshot_terminal_mismatches"]:
        raise AssertionError(f"Reward snapshot mismatch: {reward_audit}")
    if reward_audit["max_final_advantage_reconstruction_error"] > 2e-6:
        raise AssertionError(f"Reward reconstruction mismatch: {reward_audit}")
    if value_audit["snapshot_terminal_mismatches"]:
        raise AssertionError(f"Value snapshot mismatch: {value_audit}")
    if value_audit["max_true_return_eval_error"] > 1e-12:
        raise AssertionError(f"Value verifier mismatch: {value_audit}")
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def style_axis(axis, x_labels: bool = True) -> None:
    from matplotlib.ticker import FormatStrFormatter, MaxNLocator

    axis.set_xlim(-0.18, ROUND_POSITIONS[-1] + 0.18)
    axis.set_xticks(ROUND_POSITIONS)
    axis.set_xticklabels([str(value) for value in ROUNDS] if x_labels else [])
    axis.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3, steps=[1, 2, 2.5, 5, 10]))
    axis.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    axis.axhline(0.0, color="#7E858A", linestyle="--", linewidth=0.75, zorder=2)
    axis.grid(axis="y", color="#C8D0D5", linestyle=(0, (3, 3)), linewidth=0.65, alpha=0.78)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#A5ADB3")
    axis.spines["bottom"].set_color("#A5ADB3")
    axis.spines["left"].set_linewidth(0.7)
    axis.spines["bottom"].set_linewidth(0.7)
    axis.tick_params(axis="both", colors="#5D6870", labelsize=8.4, length=2.8, width=0.65, pad=2.2)


def padded_limits(values: list[float], lower_zero: bool = False) -> tuple[float, float]:
    low, high = min(values), max(values)
    span = max(high - low, 0.02)
    lower = low - 0.12 * span
    upper = high + 0.13 * span
    if lower_zero and low >= -1e-12:
        lower = -0.015 * max(upper, 0.02)
    return lower, upper


def exact_curves(path: Path) -> dict:
    payload = json.loads(path.read_text())
    curves = {}
    for guidance in ("full", "truncated"):
        curves[guidance] = {}
        for lam in ("0", "0.3", "0.6", "0.95"):
            records = [
                record
                for record in payload["per_configuration"]
                if record["guidance"] == guidance and record["lambda"] == lam
            ]
            baseline = np.asarray([record["baseline"] for record in records], dtype=np.float64)
            curves[guidance][lam] = {}
            for key in ("j_lambda", "true_return"):
                values = np.asarray([record[key] for record in records], dtype=np.float64)
                improvements = values - baseline[:, None]
                curves[guidance][lam][key] = {
                    "mean": improvements.mean(axis=0).tolist(),
                    "std": improvements.std(axis=0, ddof=1).tolist(),
                }
    return curves


def plot_main(exact_path: Path, learned: dict, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif", "mathtext.fontset": "cm", "font.size": 9.4,
        "text.color": "#303030", "axes.labelcolor": "#303030",
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    exact = exact_curves(exact_path)
    figure, axes = plt.subplots(2, 4, figsize=(10.8, 4.35))
    figure.subplots_adjust(left=0.073, right=0.995, bottom=0.105, top=0.79, wspace=0.20, hspace=0.14)
    panels = (
        ("full", "j_lambda", r"$J_\lambda(\pi_j^S)-J(\pi)$"),
        ("full", "true_return", r"$J(\pi_j^S)-J(\pi)$"),
        ("truncated", "j_lambda", r"$J_\lambda(\pi_j^S)-J(\pi)$"),
        ("truncated", "true_return", r"$J(\pi_j^S)-J(\pi)$"),
    )
    colors = {"0": "#4C78A8", "0.3": "#B49A68", "0.6": "#91809E", "0.95": "#B5475D"}
    handles = []
    exact_bounds = {"full": [], "truncated": []}
    for column, (guidance, metric, title) in enumerate(panels):
        axis = axes[0, column]
        style_axis(axis, x_labels=False)
        axis.set_title(f"({chr(97 + column)})  {title}", fontsize=11.0, fontweight="bold", pad=5.5)
        for lam in colors:
            curve = exact[guidance][lam][metric]
            mean = np.asarray(curve["mean"])
            std = np.asarray(curve["std"])
            exact_bounds[guidance].extend((mean - std).tolist() + (mean + std).tolist())
            exact_positions = np.log2(np.arange(len(mean), dtype=np.float64) + 1.0)
            axis.fill_between(exact_positions, mean - std, mean + std, color=colors[lam], alpha=0.13, linewidth=0)
            handle, = axis.plot(exact_positions, mean, color=colors[lam], linewidth=1.6, solid_capstyle="round", label=rf"$\lambda={lam}$")
            if column == 0:
                handles.append(handle)
    for column, (guidance, _, _) in enumerate(panels):
        axes[0, column].set_ylim(*padded_limits(exact_bounds[guidance], lower_zero=True))

    learned_panels = (
        ("reward", "j_lambda"), ("reward", "j"), ("value", "j_lambda"), ("value", "j")
    )
    learned_colors = {"reward": "#4C78A8", "value": "#B5475D"}
    learned_values = {"reward": [], "value": []}
    for column, ((guidance, metric), (_, _, title)) in enumerate(zip(learned_panels, panels, strict=True)):
        axis = axes[1, column]
        style_axis(axis, x_labels=True)
        values = learned["series"][guidance][metric]["_all"]
        learned_values[guidance].extend(values)
        axis.plot(ROUND_POSITIONS, values, color=learned_colors[guidance], linewidth=2.0, solid_capstyle="round")
        axis.set_xlabel(r"Search budget $j$", fontsize=11.5, labelpad=2.5)
    for column, (guidance, _) in enumerate(learned_panels):
        axes[1, column].set_ylim(*padded_limits(learned_values[guidance], lower_zero=False))

    # Keep paired panels close, with room for the next group's y tick labels.
    group_gap = 0.045
    panel_width = (0.995 - 0.073 - 2 * 0.024 - group_gap) / 4
    column_lefts = (
        0.073,
        0.073 + panel_width + 0.024,
        0.073 + 2 * panel_width + 0.024 + group_gap,
        0.073 + 3 * panel_width + 2 * 0.024 + group_gap,
    )
    for row in axes:
        for column, axis in enumerate(row):
            axis.tick_params(axis="both", labelsize=9.8)
            position = axis.get_position()
            axis.set_position([column_lefts[column], position.y0, panel_width, position.height])
        for first, second in ((0, 1), (2, 3)):
            row[second].sharey(row[first])
            row[second].tick_params(axis="y", left=False, labelleft=False)

    axes[0, 0].set_ylabel("Exact-value\nimprovement", fontsize=11.0, fontweight="bold", labelpad=6)
    axes[1, 0].set_ylabel("Learned-critic\nimprovement", fontsize=11.0, fontweight="bold", labelpad=6)
    figure.legend(handles=handles, labels=[h.get_label() for h in handles], frameon=False, ncol=4,
                  loc="upper center", bbox_to_anchor=(0.5, 0.995), fontsize=13.0,
                  handlelength=2.5, columnspacing=1.55)
    group_centers = [(column_lefts[first] + column_lefts[first + 1] + panel_width) / 2
                     for first in (0, 2)]
    for center, title in zip(group_centers, ("Reward-guided OPTS", "Value-guided OPTS")):
        figure.text(center, 0.878, title, ha="center", va="center", fontsize=12.0, fontweight="bold")

    output.parent.mkdir(parents=True, exist_ok=True)
    for path in (output, output.with_suffix(".png")):
        figure.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.035)
    plt.close(figure)


def plot_appendix(learned: dict, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    plt.rcParams.update({
        "font.family": "serif", "mathtext.fontset": "cm", "font.size": 8.6,
        "text.color": "#303030", "axes.labelcolor": "#303030",
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    figure, axes = plt.subplots(4, 6, figsize=(12.2, 7.4), sharey="row")
    figure.subplots_adjust(left=0.095, right=0.995, bottom=0.095, top=0.94, wspace=0.14, hspace=0.20)
    rows = (
        ("reward", "j_lambda", "Reward-guided", r"$\Delta J_\lambda$"),
        ("reward", "j", "Reward-guided", r"$\Delta J$"),
        ("value", "j_lambda", "Value-guided", r"$\Delta J_\lambda$"),
        ("value", "j", "Value-guided", r"$\Delta J$"),
    )
    colors = {"reward": "#4C78A8", "value": "#B5475D"}
    for row_index, (guidance, metric, label, formula) in enumerate(rows):
        row_values = [value for dataset in DATASETS for value in learned["series"][guidance][metric][dataset]]
        ylim = padded_limits(row_values, lower_zero=False)
        for column, dataset in enumerate(DATASETS):
            axis = axes[row_index, column]
            style_axis(axis, x_labels=row_index == len(rows) - 1)
            axis.tick_params(axis="both", labelsize=10.2)
            axis.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3, steps=[1, 2, 5, 10]))
            axis.set_ylim(*ylim)
            if column > 0:
                axis.tick_params(axis="y", left=False, labelleft=False)
            axis.plot(ROUND_POSITIONS, learned["series"][guidance][metric][dataset],
                      color=colors[guidance], linewidth=1.8, solid_capstyle="round")
            if row_index == 0:
                axis.set_title(DATASET_NAMES[dataset], fontsize=11.5, fontweight="bold", pad=7)
        axes[row_index, 0].set_ylabel(f"{label}\n{formula}", fontsize=10.5, fontweight="bold", labelpad=8)
    figure.text(0.545, 0.025, r"Maximum OPTS search rounds $S_{\max}$", ha="center", va="center", fontsize=11.5)
    output.parent.mkdir(parents=True, exist_ok=True)
    for path in (output, output.with_suffix(".png")):
        figure.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.035)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact", type=Path, default=DEFAULT_EXACT)
    parser.add_argument("--reward-parquet", type=Path, default=DEFAULT_REWARD)
    parser.add_argument("--value-parquet", type=Path, default=DEFAULT_VALUE)
    parser.add_argument("--reward-eval", type=Path, default=DEFAULT_REWARD_EVAL)
    parser.add_argument("--value-eval", type=Path, default=DEFAULT_VALUE_EVAL)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--main-output", type=Path, default=DEFAULT_MAIN)
    parser.add_argument("--appendix-output", type=Path, default=DEFAULT_APPENDIX)
    parser.add_argument("--reuse-metrics", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.reuse_metrics:
        learned = json.loads(args.metrics.read_text())
    else:
        learned = analyze(args)
    plot_main(args.exact, learned, args.main_output)
    plot_appendix(learned, args.appendix_output)
    print(json.dumps(learned["audit"], indent=2))
    print(f"Wrote {args.metrics}")
    print(f"Wrote {args.main_output} and {args.appendix_output}")


if __name__ == "__main__":
    main()
