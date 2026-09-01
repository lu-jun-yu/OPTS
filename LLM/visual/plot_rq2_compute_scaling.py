import argparse
import gc
import json
import math
import os
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, FuncFormatter, MaxNLocator


os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = REPO_ROOT / "LLM/results/step400/eval"
GEN_DIR = REPO_ROOT / "LLM/results/step400/gen"
METHOD_TAG = "opts_ttpo_exp8_3_0810_n8"

DEFAULT_REWARD_IID = (
    EVAL_DIR / f"{METHOD_TAG}_iid_n128__task2_iid_pass_k8-16-32-64-128.json"
)
DEFAULT_REWARD_OPTS = (
    EVAL_DIR
    / f"{METHOD_TAG}_opts_reward_s3_n128__task2_reward_opts_k8-16-32-64-128.json"
)
DEFAULT_VALUE_IID = (
    EVAL_DIR / f"{METHOD_TAG}_iid_n128__task3_iid_cons_k8-16-32-64-128.json"
)
DEFAULT_VALUE_OPTS = (
    EVAL_DIR
    / f"{METHOD_TAG}_opts_value_s3_n128__task3_value_opts_k8-16-32-64-128.json"
)

DEFAULT_IID_PARQUET = GEN_DIR / f"{METHOD_TAG}_iid_n128.parquet"
DEFAULT_REWARD_PARQUET = GEN_DIR / f"{METHOD_TAG}_opts_reward_s3_n128.parquet"
DEFAULT_VALUE_PARQUET = GEN_DIR / f"{METHOD_TAG}_opts_value_s3_n128.parquet"
DEFAULT_TOKENIZER = REPO_ROOT / "LLM/models/Qwen3-1.7B"

DEFAULT_PERFORMANCE_OUTPUT = (
    REPO_ROOT / "paper/figures/rq2_compute_scaling_performance.pdf"
)
DEFAULT_PERFORMANCE_PNG_OUTPUT = (
    REPO_ROOT / "paper/figures/rq2_compute_scaling_performance.png"
)
DEFAULT_TOKEN_OUTPUT = REPO_ROOT / "paper/figures/rq2_compute_scaling.pdf"
DEFAULT_TOKEN_PNG_OUTPUT = REPO_ROOT / "paper/figures/rq2_compute_scaling.png"
DEFAULT_DATASET_OUTPUT = (
    REPO_ROOT / "paper/figures/rq2_compute_scaling_by_dataset.pdf"
)
DEFAULT_DATASET_PNG_OUTPUT = (
    REPO_ROOT / "paper/figures/rq2_compute_scaling_by_dataset.png"
)

K_VALUES = [8, 16, 32, 64, 128]
NUM_PROMPTS = 902
TOTAL_ROLLOUTS = NUM_PROMPTS * max(K_VALUES)
DATASETS = (
    ("hiyouga/math12k", "Math12K"),
    ("math-ai/aime24", "AIME24"),
    ("math-ai/aime25", "AIME25"),
    ("math-ai/aime26", "AIME26"),
    ("math-ai/amc23", "AMC23"),
    ("math-ai/minervamath", "MinervaMath"),
)
EXPECTED_DATASETS = {dataset for dataset, _ in DATASETS} | {"_all"}

IID_COLOR = "#7A8791"
OPTS_COLOR = "#B5475D"
TEXT_COLOR = "#263238"
MUTED_TEXT_COLOR = "#59636B"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot pooled performance for reward- and value-guided OPTS under "
            "matched rollout budgets, with optional token-cost panels."
        )
    )
    parser.add_argument("--reward-iid", type=Path, default=DEFAULT_REWARD_IID)
    parser.add_argument("--reward-opts", type=Path, default=DEFAULT_REWARD_OPTS)
    parser.add_argument("--value-iid", type=Path, default=DEFAULT_VALUE_IID)
    parser.add_argument("--value-opts", type=Path, default=DEFAULT_VALUE_OPTS)
    parser.add_argument("--iid-parquet", type=Path, default=DEFAULT_IID_PARQUET)
    parser.add_argument(
        "--reward-parquet", type=Path, default=DEFAULT_REWARD_PARQUET
    )
    parser.add_argument("--value-parquet", type=Path, default=DEFAULT_VALUE_PARQUET)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--tokenizer-batch-size", type=int, default=512)
    parser.add_argument(
        "--with-token-panels",
        action="store_true",
        help="Also compute token costs and render the legacy four-panel figure.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--png-output", type=Path)
    parser.add_argument(
        "--dataset-output", type=Path, default=DEFAULT_DATASET_OUTPUT
    )
    parser.add_argument(
        "--dataset-png-output", type=Path, default=DEFAULT_DATASET_PNG_OUTPUT
    )
    args = parser.parse_args()
    if args.output is None:
        args.output = (
            DEFAULT_TOKEN_OUTPUT
            if args.with_token_panels
            else DEFAULT_PERFORMANCE_OUTPUT
        )
    if args.png_output is None:
        args.png_output = (
            DEFAULT_TOKEN_PNG_OUTPUT
            if args.with_token_panels
            else DEFAULT_PERFORMANCE_PNG_OUTPUT
        )
    return args


def load_evaluation_series(path, metric):
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    if data.get("metrics") != [metric]:
        raise ValueError(
            f"Expected metrics=[{metric!r}] in {path}, found {data.get('metrics')}"
        )
    if data.get("k") != K_VALUES:
        raise ValueError(f"Expected k={K_VALUES} in {path}, found {data.get('k')}")
    if data.get("opts_avg_slices") is not None:
        raise ValueError(
            f"Expected opts_avg_slices=null in {path}, "
            f"found {data.get('opts_avg_slices')}"
        )

    results = data.get("results")
    if not isinstance(results, dict):
        raise ValueError(f"Missing results object in {path}")
    if set(results) != EXPECTED_DATASETS:
        missing = sorted(EXPECTED_DATASETS - set(results))
        extra = sorted(set(results) - EXPECTED_DATASETS)
        raise ValueError(
            f"Dataset mismatch in {path}; missing={missing}, extra={extra}"
        )

    expected_keys = {f"{metric}@{k_value}" for k_value in K_VALUES}
    series = {}
    for dataset, dataset_results in results.items():
        if not isinstance(dataset_results, dict):
            raise ValueError(f"Invalid result for {dataset} in {path}")
        if set(dataset_results) != expected_keys:
            missing = sorted(expected_keys - set(dataset_results))
            extra = sorted(set(dataset_results) - expected_keys)
            raise ValueError(
                f"Metric mismatch for {dataset} in {path}; "
                f"missing={missing}, extra={extra}"
            )
        values = []
        for k_value in K_VALUES:
            value = float(dataset_results[f"{metric}@{k_value}"])
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"Invalid {dataset} {metric}@{k_value}={value} in {path}"
                )
            values.append(value)
        series[dataset] = values
    return series


def list_array(table, name, path):
    array = table[name].combine_chunks()
    if array.null_count:
        raise ValueError(f"Null outer lists in {name} of {path}")
    return array


def outer_offsets(array):
    return array.offsets.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)


def tokenize_lengths(texts, tokenizer, batch_size, label):
    if batch_size < 1:
        raise ValueError(f"tokenizer_batch_size must be positive, found {batch_size}")
    if any(not isinstance(text, str) for text in texts):
        raise ValueError(f"Non-string response encountered while tokenizing {label}")

    lengths = np.empty(len(texts), dtype=np.int64)
    backend = tokenizer.backend_tokenizer
    next_report = 20_000
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encodings = backend.encode_batch(batch, add_special_tokens=False)
        lengths[start : start + len(encodings)] = [
            len(encoding.ids) for encoding in encodings
        ]
        completed = start + len(encodings)
        if completed >= next_report:
            print(f"[{label}] tokenized {completed:,}/{len(texts):,}", flush=True)
            next_report += 20_000
    return lengths


def compute_iid_token_cost(path, tokenizer, batch_size):
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(path)
    if parquet_file.metadata.num_rows != NUM_PROMPTS:
        raise ValueError(
            f"Expected {NUM_PROMPTS} prompts in {path}, "
            f"found {parquet_file.metadata.num_rows}"
        )

    table = pq.read_table(path, columns=["responses"])
    responses = list_array(table, "responses", path)
    row_lengths = np.diff(outer_offsets(responses))
    if not np.all(row_lengths == max(K_VALUES)):
        counts = Counter(row_lengths.tolist())
        raise ValueError(f"Expected 128 IID responses per prompt in {path}: {counts}")

    texts = responses.values.to_pylist()
    if len(texts) != TOTAL_ROLLOUTS:
        raise ValueError(
            f"Expected {TOTAL_ROLLOUTS} IID responses in {path}, found {len(texts)}"
        )
    lengths = tokenize_lengths(texts, tokenizer, batch_size, "IID")
    lengths = lengths.reshape(NUM_PROMPTS, max(K_VALUES))

    costs = []
    for k_value in K_VALUES:
        selected_count = lengths[:, :k_value].size
        expected_count = NUM_PROMPTS * k_value
        if selected_count != expected_count:
            raise ValueError(
                f"IID k={k_value}: selected {selected_count}, expected {expected_count}"
            )
        costs.append(float(lengths[:, :k_value].sum() / NUM_PROMPTS))

    del table, responses, texts, lengths
    gc.collect()
    return costs


def difference_summary(differences):
    return {
        "count": int(differences.size),
        "min": int(differences.min()),
        "p01": float(np.quantile(differences, 0.01)),
        "median": float(np.median(differences)),
        "p99": float(np.quantile(differences, 0.99)),
        "max": int(differences.max()),
        "mean": float(differences.mean()),
        "within_1": float((np.abs(differences) <= 1).mean()),
        "within_2": float((np.abs(differences) <= 2).mean()),
        "within_4": float((np.abs(differences) <= 4).mean()),
        "common": Counter(differences.tolist()).most_common(8),
    }


def print_difference_summary(label, scope, summary):
    print(
        f"[{label} token audit: {scope}] "
        f"n={summary['count']:,}, min={summary['min']}, p01={summary['p01']:.1f}, "
        f"median={summary['median']:.1f}, p99={summary['p99']:.1f}, "
        f"max={summary['max']}, mean={summary['mean']:.4f}; "
        f"|diff|<=1: {100 * summary['within_1']:.2f}%, "
        f"<=2: {100 * summary['within_2']:.2f}%, "
        f"<=4: {100 * summary['within_4']:.2f}%; "
        f"most_common={summary['common']}",
        flush=True,
    )


def compute_opts_token_cost(path, tokenizer, batch_size, label):
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(path)
    if parquet_file.metadata.num_rows != NUM_PROMPTS:
        raise ValueError(
            f"Expected {NUM_PROMPTS} prompts in {path}, "
            f"found {parquet_file.metadata.num_rows}"
        )

    columns = [
        "responses",
        "global_indices",
        "tree_rids",
        "tree_pids",
        "tree_branch_pos",
        "tree_advantages",
    ]
    print(f"[{label}] reading tree responses and audit lengths", flush=True)
    table = pq.read_table(path, columns=columns)
    arrays = {name: list_array(table, name, path) for name in columns}
    response_offsets = outer_offsets(arrays["responses"])
    for name in columns[1:]:
        if not np.array_equal(outer_offsets(arrays[name]), response_offsets):
            raise ValueError(f"Per-prompt list alignment differs for {name} in {path}")

    responses = arrays["responses"].values.to_pylist()
    global_indices = np.asarray(
        arrays["global_indices"].values.to_pylist(), dtype=np.int64
    )
    rids = arrays["tree_rids"].values.to_pylist()
    pids = arrays["tree_pids"].values.to_pylist()
    branch_positions = np.asarray(
        arrays["tree_branch_pos"].values.to_pylist(), dtype=np.int64
    )

    advantages = arrays["tree_advantages"].values
    advantage_offsets = advantages.offsets.to_numpy(zero_copy_only=False)
    advantage_lengths = np.diff(advantage_offsets).astype(np.int64, copy=False)

    lengths = [
        len(responses),
        len(global_indices),
        len(rids),
        len(pids),
        len(branch_positions),
        len(advantage_lengths),
    ]
    if any(length != TOTAL_ROLLOUTS for length in lengths):
        raise ValueError(
            f"Expected {TOTAL_ROLLOUTS} aligned OPTS rollouts in {path}, "
            f"found lengths={lengths}"
        )
    if len(set(rids)) != TOTAL_ROLLOUTS:
        raise ValueError(f"tree_rids are not globally unique in {path}")
    if not np.array_equal(
        np.sort(global_indices), np.arange(1, TOTAL_ROLLOUTS + 1, dtype=np.int64)
    ):
        raise ValueError(f"global_indices do not cover 1..{TOTAL_ROLLOUTS} in {path}")

    rid_to_index = {rid: index for index, rid in enumerate(rids)}
    shared_prefix = np.full(TOTAL_ROLLOUTS, -1, dtype=np.int64)
    depths = np.zeros(TOTAL_ROLLOUTS, dtype=np.int64)
    for index in np.argsort(global_indices):
        pid = pids[index]
        branch_position = int(branch_positions[index])
        if pid is None:
            if branch_position != -1:
                raise ValueError(
                    f"Root {rids[index]} has branch_pos={branch_position} in {path}"
                )
            shared_prefix[index] = 0
            continue

        if pid not in rid_to_index:
            raise ValueError(f"Missing parent rid={pid} for {rids[index]} in {path}")
        parent_index = rid_to_index[pid]
        if global_indices[parent_index] >= global_indices[index]:
            raise ValueError(
                f"Parent {pid} does not precede child {rids[index]} in {path}"
            )
        if shared_prefix[parent_index] < 0:
            raise ValueError(f"Unresolved parent chain for {rids[index]} in {path}")
        if not 0 <= branch_position < advantage_lengths[parent_index]:
            raise ValueError(
                f"branch_pos={branch_position} falls outside parent advantage length "
                f"{advantage_lengths[parent_index]} for {rids[index]} in {path}"
            )

        shared_prefix[index] = (
            shared_prefix[parent_index] + branch_position + 1
        )
        depths[index] = depths[parent_index] + 1

    full_lengths = tokenize_lengths(responses, tokenizer, batch_size, label)
    new_token_lengths = full_lengths - shared_prefix
    negative_count = int((new_token_lengths < 0).sum())
    if negative_count:
        raise ValueError(
            f"{label}: prefix subtraction produced {negative_count} negative lengths; "
            "refusing to plot token cost"
        )

    differences = new_token_lengths - advantage_lengths
    overall_summary = difference_summary(differences)
    child_mask = depths > 0
    child_summary = difference_summary(differences[child_mask])
    print_difference_summary(label, "all", overall_summary)
    print_difference_summary(label, "children", child_summary)
    aggregate_difference = float(differences.sum() / advantage_lengths.sum())
    print(
        f"[{label} token audit] nonroots={int(child_mask.sum()):,}, "
        f"max_depth={int(depths.max())}, new_token_range="
        f"[{int(new_token_lengths.min())}, {int(new_token_lengths.max())}], "
        f"aggregate diff={100 * aggregate_difference:+.4f}%",
        flush=True,
    )

    if overall_summary["within_4"] < 0.99 or abs(aggregate_difference) > 0.005:
        raise ValueError(
            f"{label}: tokenizer-derived suffix lengths are clearly inconsistent "
            "with tree_advantages lengths; refusing to plot token cost"
        )

    costs = []
    for k_value in K_VALUES:
        selected = global_indices <= NUM_PROMPTS * k_value
        selected_count = int(selected.sum())
        expected_count = NUM_PROMPTS * k_value
        if selected_count != expected_count:
            raise ValueError(
                f"{label} k={k_value}: selected {selected_count}, "
                f"expected {expected_count}"
            )
        costs.append(float(new_token_lengths[selected].sum() / NUM_PROMPTS))

    audit = {
        "overall": overall_summary,
        "children": child_summary,
        "aggregate_difference": aggregate_difference,
        "nonroots": int(child_mask.sum()),
        "max_depth": int(depths.max()),
    }

    del (
        table,
        arrays,
        responses,
        global_indices,
        rids,
        pids,
        branch_positions,
        advantages,
        advantage_offsets,
        advantage_lengths,
        rid_to_index,
        shared_prefix,
        depths,
        full_lengths,
        new_token_lengths,
        differences,
    )
    gc.collect()
    return costs, audit


def dynamic_performance_ylim(*series):
    values = [value for values in series for value in values]
    low = min(values)
    high = max(values)
    span = high - low
    pad = max(0.15 * span, 0.004)
    quantum = 0.005
    lower = max(0.0, math.floor((low - pad) / quantum) * quantum)
    upper = min(1.0, math.ceil((high + pad) / quantum) * quantum)
    return lower, upper


def count_strict_wins(opts_values, iid_values, tolerance=1e-12):
    return sum(
        opts_value > iid_value + tolerance
        for opts_value, iid_value in zip(opts_values, iid_values)
    )


def style_axis(axis, ylim, ylabel, token_axis=False):
    axis.set_xlim(-0.18, len(K_VALUES) - 0.82)
    axis.set_ylim(*ylim)
    axis.set_xticks(range(len(K_VALUES)))
    axis.set_xticklabels([str(k_value) for k_value in K_VALUES])
    axis.yaxis.set_major_locator(
        MaxNLocator(nbins=4, min_n_ticks=3, steps=[1, 2, 2.5, 5, 10])
    )
    if token_axis:
        axis.yaxis.set_major_formatter(
            FuncFormatter(lambda value, _position: f"{value / 1000:.0f}k")
        )
    else:
        axis.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    axis.grid(
        axis="y",
        color="#CBD1D6",
        linestyle=(0, (4, 3)),
        linewidth=0.65,
        alpha=0.78,
    )
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#A5ADB3")
    axis.spines["bottom"].set_color("#A5ADB3")
    axis.spines["left"].set_linewidth(0.7)
    axis.spines["bottom"].set_linewidth(0.7)
    axis.tick_params(
        axis="both",
        colors=MUTED_TEXT_COLOR,
        labelsize=9.4,
        length=3.0,
        width=0.65,
        pad=2.5,
    )
    axis.set_ylabel(ylabel, fontsize=10.2, labelpad=5.0)


def plot_method_pair(axis, iid_values, opts_values):
    x_values = range(len(K_VALUES))
    iid_line, = axis.plot(
        x_values,
        iid_values,
        color=IID_COLOR,
        linestyle=(0, (4, 2.5)),
        linewidth=1.85,
        marker="o",
        markersize=5.2,
        markerfacecolor="white",
        markeredgecolor=IID_COLOR,
        markeredgewidth=1.15,
        label="IID baseline",
        zorder=3,
    )
    opts_line, = axis.plot(
        x_values,
        opts_values,
        color=OPTS_COLOR,
        linewidth=2.35,
        marker="D",
        markersize=5.3,
        markerfacecolor=OPTS_COLOR,
        markeredgecolor="white",
        markeredgewidth=0.75,
        solid_capstyle="round",
        label=r"OPTS ($S_{\max}=3$)",
        zorder=4,
    )
    return iid_line, opts_line


def add_badge(axis, text):
    axis.text(
        0.035,
        0.94,
        text,
        transform=axis.transAxes,
        ha="left",
        va="top",
        color=OPTS_COLOR,
        fontsize=9.7,
        fontweight="semibold",
        bbox={
            "boxstyle": "round,pad=0.26",
            "facecolor": "#EEF4F8",
            "edgecolor": "#B8CBD9",
            "linewidth": 0.6,
        },
        zorder=5,
    )


def plot_performance_panel(axis, iid_values, opts_values, title, ylabel):
    handles = plot_method_pair(axis, iid_values, opts_values)
    style_axis(
        axis,
        dynamic_performance_ylim(iid_values, opts_values),
        ylabel,
    )
    axis.set_title(
        title,
        color=TEXT_COLOR,
        fontsize=12.0,
        fontweight="semibold",
        linespacing=0.92,
        pad=8.0,
    )
    wins = count_strict_wins(opts_values, iid_values)
    return handles, wins


def plot_token_panel(axis, iid_values, opts_values, title, token_ylim):
    plot_method_pair(axis, iid_values, opts_values)
    style_axis(axis, token_ylim, "Tokens / prompt", token_axis=True)
    axis.set_title(
        title,
        color=TEXT_COLOR,
        fontsize=12.0,
        fontweight="semibold",
        linespacing=0.92,
        pad=8.0,
    )
    percent_change = 100.0 * (opts_values[-1] / iid_values[-1] - 1.0)
    add_badge(axis, rf"At $k=128$: {percent_change:+.1f}% tokens")
    return percent_change


def plot_performance_figure(reward_iid, reward_opts, value_iid, value_opts):
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10.2,
            "text.color": TEXT_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    figure.subplots_adjust(
        left=0.085,
        right=0.992,
        bottom=0.30,
        top=0.80,
        wspace=0.30,
    )

    handles, reward_wins = plot_performance_panel(
        axes[0],
        reward_iid,
        reward_opts,
        "(a) Reward-Guided Performance",
        "pass@k",
    )
    _, value_wins = plot_performance_panel(
        axes[1],
        value_iid,
        value_opts,
        "(b) Value-Guided Performance",
        "cons@k",
    )
    for axis in axes:
        axis.set_xlabel(r"Rollout budget $k$", fontsize=10.2, labelpad=5.0)

    figure.legend(
        handles,
        ["IID baseline", r"OPTS ($S_{\max}=3$)"],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=2,
        frameon=False,
        fontsize=10.1,
        handlelength=2.7,
        columnspacing=2.2,
    )
    return figure, reward_wins, value_wins


def style_dataset_axis(axis, ylim):
    axis.set_xlim(-0.18, len(K_VALUES) - 0.82)
    axis.set_ylim(*ylim)
    axis.set_xticks(range(len(K_VALUES)))
    axis.set_xticklabels([str(k_value) for k_value in K_VALUES])
    axis.yaxis.set_major_locator(
        MaxNLocator(nbins=3, min_n_ticks=3, steps=[1, 2, 2.5, 5, 10])
    )
    precision = 3 if ylim[1] - ylim[0] < 0.05 else 2
    axis.yaxis.set_major_formatter(FormatStrFormatter(f"%.{precision}f"))
    axis.grid(
        axis="y",
        color="#CBD1D6",
        linestyle=(0, (4, 3)),
        linewidth=0.55,
        alpha=0.78,
    )
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#A5ADB3")
    axis.spines["bottom"].set_color("#A5ADB3")
    axis.spines["left"].set_linewidth(0.65)
    axis.spines["bottom"].set_linewidth(0.65)
    axis.tick_params(
        axis="both",
        colors=MUTED_TEXT_COLOR,
        labelsize=7.8,
        length=2.6,
        width=0.6,
        pad=2.0,
    )


def plot_dataset_method_pair(axis, iid_values, opts_values):
    x_values = range(len(K_VALUES))
    iid_line, = axis.plot(
        x_values,
        iid_values,
        color=IID_COLOR,
        linestyle=(0, (4, 2.5)),
        linewidth=1.35,
        marker="o",
        markersize=3.7,
        markerfacecolor="white",
        markeredgecolor=IID_COLOR,
        markeredgewidth=0.9,
        label="IID baseline",
        zorder=3,
    )
    opts_line, = axis.plot(
        x_values,
        opts_values,
        color=OPTS_COLOR,
        linewidth=1.75,
        marker="D",
        markersize=3.8,
        markerfacecolor=OPTS_COLOR,
        markeredgecolor="white",
        markeredgewidth=0.55,
        solid_capstyle="round",
        label=r"OPTS ($S_{\max}=3$)",
        zorder=4,
    )
    return iid_line, opts_line


def plot_dataset_figure(
    reward_iid_series,
    reward_opts_series,
    value_iid_series,
    value_opts_series,
):
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9.2,
            "text.color": TEXT_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    figure, axes = plt.subplots(2, len(DATASETS), figsize=(10.8, 4.6))
    figure.subplots_adjust(
        left=0.080,
        right=0.995,
        bottom=0.185,
        top=0.855,
        hspace=0.58,
        wspace=0.40,
    )

    handles = None
    for column, (dataset, display_name) in enumerate(DATASETS):
        reward_iid = reward_iid_series[dataset]
        reward_opts = reward_opts_series[dataset]
        value_iid = value_iid_series[dataset]
        value_opts = value_opts_series[dataset]
        handles = plot_dataset_method_pair(
            axes[0, column], reward_iid, reward_opts
        )
        plot_dataset_method_pair(axes[1, column], value_iid, value_opts)
        style_dataset_axis(
            axes[0, column], dynamic_performance_ylim(reward_iid, reward_opts)
        )
        style_dataset_axis(
            axes[1, column], dynamic_performance_ylim(value_iid, value_opts)
        )
        axes[0, column].set_title(
            display_name,
            color=TEXT_COLOR,
            fontsize=10.2,
            fontweight="semibold",
            pad=5.0,
        )

    axes[0, 0].set_ylabel("pass@k", fontsize=9.2, labelpad=6.0)
    axes[1, 0].set_ylabel("cons@k", fontsize=9.2, labelpad=6.0)
    figure.text(
        0.535,
        0.965,
        "Reward-Guided OPTS",
        ha="center",
        va="center",
        color=OPTS_COLOR,
        fontsize=11.2,
        fontweight="semibold",
    )
    figure.text(
        0.535,
        0.505,
        "Value-Guided OPTS",
        ha="center",
        va="center",
        color=OPTS_COLOR,
        fontsize=11.2,
        fontweight="semibold",
    )
    figure.text(
        0.535,
        0.112,
        r"Rollout budget $k$",
        ha="center",
        va="center",
        color=MUTED_TEXT_COLOR,
        fontsize=9.5,
    )
    figure.legend(
        handles,
        ["IID baseline", r"OPTS ($S_{\max}=3$)"],
        loc="lower center",
        bbox_to_anchor=(0.535, 0.006),
        ncol=2,
        frameon=False,
        fontsize=9.2,
        handlelength=2.6,
        columnspacing=2.1,
    )
    return figure


def plot_figure(
    reward_iid,
    reward_opts,
    value_iid,
    value_opts,
    iid_tokens,
    reward_tokens,
    value_tokens,
):
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10.2,
            "text.color": TEXT_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    figure, axes = plt.subplots(1, 4, figsize=(10.8, 3.2))
    figure.subplots_adjust(
        left=0.055,
        right=0.993,
        bottom=0.255,
        top=0.79,
        wspace=0.36,
    )

    handles, reward_wins = plot_performance_panel(
        axes[0],
        reward_iid,
        reward_opts,
        "(a) Reward-Guided\nPerformance",
        "pass@k",
    )

    max_token_cost = max(iid_tokens + reward_tokens + value_tokens)
    token_upper = math.ceil((1.10 * max_token_cost) / 10_000) * 10_000
    token_ylim = (0.0, float(token_upper))
    reward_token_change = plot_token_panel(
        axes[1],
        iid_tokens,
        reward_tokens,
        "(b) Reward-Guided\nToken Cost",
        token_ylim,
    )

    _, value_wins = plot_performance_panel(
        axes[2],
        value_iid,
        value_opts,
        "(c) Value-Guided\nPerformance",
        "cons@k",
    )
    value_token_change = plot_token_panel(
        axes[3],
        iid_tokens,
        value_tokens,
        "(d) Value-Guided\nToken Cost",
        token_ylim,
    )

    figure.text(
        0.5,
        0.155,
        r"Rollout budget $k$",
        ha="center",
        va="center",
        color=MUTED_TEXT_COLOR,
        fontsize=10.6,
    )
    figure.legend(
        handles,
        ["IID baseline", r"OPTS ($S_{\max}=3$)"],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.002),
        ncol=2,
        frameon=False,
        fontsize=10.1,
        handlelength=2.7,
        columnspacing=2.2,
    )
    return (
        figure,
        reward_wins,
        value_wins,
        reward_token_change,
        value_token_change,
    )


def main():
    args = parse_args()
    reward_iid_series = load_evaluation_series(args.reward_iid, "pass")
    reward_opts_series = load_evaluation_series(args.reward_opts, "opts")
    value_iid_series = load_evaluation_series(args.value_iid, "cons")
    value_opts_series = load_evaluation_series(args.value_opts, "opts")
    reward_iid = reward_iid_series["_all"]
    reward_opts = reward_opts_series["_all"]
    value_iid = value_iid_series["_all"]
    value_opts = value_opts_series["_all"]

    if not args.with_token_panels:
        figure, reward_wins, value_wins = plot_performance_figure(
            reward_iid,
            reward_opts,
            value_iid,
            value_opts,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            args.output,
            bbox_inches="tight",
            pad_inches=0.035,
            metadata={"Title": "Compute Scaling of OPTS"},
        )
        args.png_output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            args.png_output,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.035,
        )
        plt.close(figure)

        dataset_figure = plot_dataset_figure(
            reward_iid_series,
            reward_opts_series,
            value_iid_series,
            value_opts_series,
        )
        args.dataset_output.parent.mkdir(parents=True, exist_ok=True)
        dataset_figure.savefig(
            args.dataset_output,
            bbox_inches="tight",
            pad_inches=0.035,
            metadata={"Title": "Compute Scaling of OPTS by Dataset"},
        )
        args.dataset_png_output.parent.mkdir(parents=True, exist_ok=True)
        dataset_figure.savefig(
            args.dataset_png_output,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.035,
        )
        plt.close(dataset_figure)
        print(f"Reward-guided pooled wins: {reward_wins}/{len(K_VALUES)}")
        print(f"Value-guided pooled wins: {value_wins}/{len(K_VALUES)}")
        print(f"Wrote {args.output}")
        print(f"Wrote {args.png_output}")
        print(f"Wrote {args.dataset_output}")
        print(f"Wrote {args.dataset_png_output}")
        return

    from transformers import AutoTokenizer

    print(f"Loading tokenizer from {args.tokenizer}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=True,
        local_files_only=True,
    )
    iid_tokens = compute_iid_token_cost(
        args.iid_parquet,
        tokenizer,
        args.tokenizer_batch_size,
    )
    reward_tokens, reward_audit = compute_opts_token_cost(
        args.reward_parquet,
        tokenizer,
        args.tokenizer_batch_size,
        "Reward OPTS",
    )
    value_tokens, value_audit = compute_opts_token_cost(
        args.value_parquet,
        tokenizer,
        args.tokenizer_batch_size,
        "Value OPTS",
    )

    print(f"IID tokens/prompt: {iid_tokens}", flush=True)
    print(f"Reward OPTS tokens/prompt: {reward_tokens}", flush=True)
    print(f"Value OPTS tokens/prompt: {value_tokens}", flush=True)

    (
        figure,
        reward_wins,
        value_wins,
        reward_token_change,
        value_token_change,
    ) = plot_figure(
        reward_iid,
        reward_opts,
        value_iid,
        value_opts,
        iid_tokens,
        reward_tokens,
        value_tokens,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        args.output,
        bbox_inches="tight",
        pad_inches=0.035,
        metadata={"Title": "Compute Scaling of OPTS"},
    )
    args.png_output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        args.png_output,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.035,
    )
    plt.close(figure)

    print(f"Reward-guided pooled wins: {reward_wins}/{len(K_VALUES)}")
    print(f"Value-guided pooled wins: {value_wins}/{len(K_VALUES)}")
    print(f"Reward OPTS token change at k=128: {reward_token_change:+.2f}%")
    print(f"Value OPTS token change at k=128: {value_token_change:+.2f}%")
    print(
        "Reward audit aggregate diff vs tree_advantages: "
        f"{100 * reward_audit['aggregate_difference']:+.4f}%"
    )
    print(
        "Value audit aggregate diff vs tree_advantages: "
        f"{100 * value_audit['aggregate_difference']:+.4f}%"
    )
    print(f"Wrote {args.output}")
    print(f"Wrote {args.png_output}")


if __name__ == "__main__":
    main()
