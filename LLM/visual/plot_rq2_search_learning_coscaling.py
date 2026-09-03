import argparse
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEARCH_DIR = REPO_ROOT / "LLM/results/step400/rq2/eval"
DEFAULT_LEARNING_DIR = REPO_ROOT / "LLM/results/rq2_learn_scaling/eval"
DEFAULT_OUTPUT = REPO_ROOT / "paper/figures/rq2_search_learning_coscaling.pdf"
DEFAULT_PNG_OUTPUT = REPO_ROOT / "paper/figures/rq2_search_learning_coscaling.png"
DEFAULT_LEARNING_OUTPUT = REPO_ROOT / "paper/figures/rq2_base_policy_scaling.pdf"
DEFAULT_LEARNING_PNG_OUTPUT = REPO_ROOT / "paper/figures/rq2_base_policy_scaling.png"

DATASET_ORDER = [
    "hiyouga/math12k",
    "math-ai/aime24",
    "math-ai/aime25",
    "math-ai/aime26",
    "math-ai/amc23",
    "math-ai/minervamath",
]

DATASET_NAMES = {
    "hiyouga/math12k": "Math12K",
    "math-ai/aime24": "AIME24",
    "math-ai/aime25": "AIME25",
    "math-ai/aime26": "AIME26",
    "math-ai/amc23": "AMC23",
    "math-ai/minervamath": "MinervaMath",
}

SEARCH_ROUNDS = [0, 1, 3, 7, 15]
TRAINING_STEPS = [20, 40, 80, 160, 320]
METRIC = "opts-avg"
K = 32
FIXED_SEARCH_ROUND = 3

SEARCH_COLOR = "#426F9D"
LEARNING_COLOR = "#C86F4A"
TEXT_COLOR = "#263238"
MUTED_TEXT_COLOR = "#59636B"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot separate 1x6 figures for search-budget scaling and "
            "base-policy scaling from evaluation JSON files."
        )
    )
    parser.add_argument(
        "--search-json",
        type=Path,
        default=None,
        help=(
            "Search-budget scaling JSON. By default, require exactly one JSON "
            f"under {DEFAULT_SEARCH_DIR}."
        ),
    )
    parser.add_argument(
        "--learning-dir",
        type=Path,
        default=DEFAULT_LEARNING_DIR,
        help="Directory containing one fixed S_max=3 JSON for each training step.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output PDF path.",
    )
    parser.add_argument(
        "--png-output",
        type=Path,
        default=DEFAULT_PNG_OUTPUT,
        help="PNG preview path.",
    )
    parser.add_argument(
        "--learning-output",
        type=Path,
        default=DEFAULT_LEARNING_OUTPUT,
        help="Output PDF path for base-policy scaling.",
    )
    parser.add_argument(
        "--learning-png-output",
        type=Path,
        default=DEFAULT_LEARNING_PNG_OUTPUT,
        help="PNG preview path for base-policy scaling.",
    )
    return parser.parse_args()


def read_json(path):
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def require_exact_sequence(data, field, expected, path):
    actual = data.get(field)
    if actual != expected:
        raise ValueError(f"Expected {field}={expected} in {path}, found {actual}")


def validate_common_schema(data, path, expected_slices):
    require_exact_sequence(data, "metrics", [METRIC], path)
    require_exact_sequence(data, "k", [K], path)
    require_exact_sequence(data, "opts_avg_slices", expected_slices, path)

    results = data.get("results")
    if not isinstance(results, dict):
        raise ValueError(f"Missing results object in {path}")

    expected_datasets = set(DATASET_ORDER) | {"_all"}
    actual_datasets = set(results)
    if actual_datasets != expected_datasets:
        missing = sorted(expected_datasets - actual_datasets)
        extra = sorted(actual_datasets - expected_datasets)
        raise ValueError(
            f"Dataset mismatch in {path}; missing={missing}, extra={extra}"
        )

    return results


def read_score(results, dataset, key, path):
    dataset_result = results.get(dataset)
    if not isinstance(dataset_result, dict) or key not in dataset_result:
        raise KeyError(f"Missing {key} for {dataset} in {path}")
    value = float(dataset_result[key])
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"Invalid score {value} for {dataset}/{key} in {path}")
    return value


def discover_search_json(explicit_path):
    if explicit_path is not None:
        return explicit_path
    matches = sorted(DEFAULT_SEARCH_DIR.glob("*.json"))
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one JSON under {DEFAULT_SEARCH_DIR}, found {matches}"
        )
    return matches[0]


def load_search_scaling(path):
    data = read_json(path)
    results = validate_common_schema(data, path, SEARCH_ROUNDS)
    series = {}
    for dataset in DATASET_ORDER:
        series[dataset] = [
            read_score(results, dataset, f"{METRIC}_s{s_value}@{K}", path)
            for s_value in SEARCH_ROUNDS
        ]
    return series


def training_step_from_filename(path):
    matches = {int(match) for match in re.findall(r"(?:^|_)step(\d+)(?:_|$)", path.stem)}
    if len(matches) != 1:
        raise ValueError(f"Could not identify one training step from {path.name}")
    return matches.pop()


def load_learning_scaling(directory):
    paths_by_step = {}
    for path in sorted(directory.glob("*.json")):
        step = training_step_from_filename(path)
        if step in paths_by_step:
            raise ValueError(
                f"Multiple learning-scaling JSONs for step {step}: "
                f"{paths_by_step[step]} and {path}"
            )
        paths_by_step[step] = path

    if set(paths_by_step) != set(TRAINING_STEPS):
        missing = sorted(set(TRAINING_STEPS) - set(paths_by_step))
        extra = sorted(set(paths_by_step) - set(TRAINING_STEPS))
        raise ValueError(
            f"Training-step mismatch under {directory}; missing={missing}, extra={extra}"
        )

    values_by_step = {}
    for step in TRAINING_STEPS:
        path = paths_by_step[step]
        data = read_json(path)
        results = validate_common_schema(data, path, [FIXED_SEARCH_ROUND])
        values_by_step[step] = {
            dataset: read_score(
                results,
                dataset,
                f"{METRIC}_s{FIXED_SEARCH_ROUND}@{K}",
                path,
            )
            for dataset in DATASET_ORDER
        }

    return {
        dataset: [values_by_step[step][dataset] for step in TRAINING_STEPS]
        for dataset in DATASET_ORDER
    }


def is_monotonic(values, tolerance=1e-12):
    return all(
        next_value + tolerance >= value
        for value, next_value in zip(values, values[1:])
    )


def shared_column_ylim(top_values, bottom_values):
    values = top_values + bottom_values
    low = min(values)
    high = max(values)
    span = high - low
    pad = max(0.12 * span, 0.018)
    return max(0.0, low - pad), min(1.0, high + pad)


def style_axis(axis, x_labels, ylim):
    axis.set_xlim(-0.18, len(x_labels) - 0.82)
    axis.set_ylim(*ylim)
    axis.set_xticks(range(len(x_labels)))
    axis.set_xticklabels([str(value) for value in x_labels])
    axis.yaxis.set_major_locator(
        MaxNLocator(nbins=3, min_n_ticks=3, steps=[1, 2, 2.5, 5, 10])
    )
    axis.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    axis.grid(
        axis="y",
        color="#CBD1D6",
        linestyle=(0, (3, 3)),
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
        labelsize=10.5,
        length=3.0,
        width=0.65,
        pad=2.5,
    )


def plot_series(axis, values, color):
    axis.plot(
        range(len(values)),
        values,
        color=color,
        linewidth=2.25,
        marker="o",
        markersize=5.4,
        markerfacecolor=color,
        markeredgecolor="white",
        markeredgewidth=0.8,
        solid_capstyle="round",
        zorder=3,
    )
    delta_pp = 100.0 * (values[-1] - values[0])
    axis.text(
        0.04,
        0.91,
        rf"$\Delta$ {delta_pp:+.1f} pp",
        transform=axis.transAxes,
        ha="left",
        va="top",
        color=color,
        fontsize=11.0,
        fontweight="semibold",
        bbox={
            "boxstyle": "round,pad=0.20",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.88,
        },
        zorder=4,
    )


def plot_row(series, x_values, color, title, xlabel, ylims):
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11.5,
            "axes.titlesize": 13.0,
            "axes.titleweight": "semibold",
            "text.color": TEXT_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    figure, axes = plt.subplots(1, 6, figsize=(12.2, 2.85), squeeze=False)
    figure.subplots_adjust(
        left=0.065,
        right=0.992,
        bottom=0.235,
        top=0.735,
        wspace=0.34,
    )

    for column, dataset in enumerate(DATASET_ORDER):
        axis = axes[0, column]
        style_axis(axis, x_values, ylims[dataset])
        plot_series(axis, series[dataset], color)
        axis.set_title(DATASET_NAMES[dataset], color=TEXT_COLOR, pad=8.0)

    figure.text(
        0.5,
        0.90,
        title,
        ha="center",
        va="center",
        color=color,
        fontsize=15.0,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.075,
        xlabel,
        ha="center",
        va="center",
        color=MUTED_TEXT_COLOR,
        fontsize=11.8,
    )
    figure.text(
        0.018,
        0.45,
        "Avg@32",
        ha="center",
        va="center",
        rotation=90,
        color=MUTED_TEXT_COLOR,
        fontsize=11.8,
        fontweight="semibold",
    )

    return figure


def main():
    args = parse_args()
    search_path = discover_search_json(args.search_json)
    search_series = load_search_scaling(search_path)
    learning_series = load_learning_scaling(args.learning_dir)

    ylims = {
        dataset: shared_column_ylim(search_series[dataset], learning_series[dataset])
        for dataset in DATASET_ORDER
    }
    search_figure = plot_row(
        search_series,
        SEARCH_ROUNDS,
        SEARCH_COLOR,
        r"More Search $\rightarrow$ Stronger Search Policy",
        r"Maximum OPTS search rounds $S_{\max}$",
        ylims,
    )
    learning_figure = plot_row(
        learning_series,
        TRAINING_STEPS,
        LEARNING_COLOR,
        r"Stronger Policy $\rightarrow$ Stronger Search",
        r"Training step (fixed search budget: $S_{\max}=3$)",
        ylims,
    )
    monotonic_count = sum(
        is_monotonic(search_series[dataset]) for dataset in DATASET_ORDER
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    search_figure.savefig(
        args.output,
        bbox_inches="tight",
        pad_inches=0.035,
        metadata={"Title": "Search-Budget Scaling of OPTS"},
    )

    if args.png_output:
        args.png_output.parent.mkdir(parents=True, exist_ok=True)
        search_figure.savefig(
            args.png_output,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.035,
        )

    args.learning_output.parent.mkdir(parents=True, exist_ok=True)
    learning_figure.savefig(
        args.learning_output,
        bbox_inches="tight",
        pad_inches=0.035,
        metadata={"Title": "Base-Policy Scaling of OPTS"},
    )

    if args.learning_png_output:
        args.learning_png_output.parent.mkdir(parents=True, exist_ok=True)
        learning_figure.savefig(
            args.learning_png_output,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.035,
        )

    plt.close(search_figure)
    plt.close(learning_figure)
    print(f"Search scaling: {search_path}")
    print(f"Learning scaling: {args.learning_dir}")
    print(f"Top-row monotonic datasets: {monotonic_count}/{len(DATASET_ORDER)}")
    print(f"Wrote {args.output}")
    if args.png_output:
        print(f"Wrote {args.png_output}")
    print(f"Wrote {args.learning_output}")
    if args.learning_png_output:
        print(f"Wrote {args.learning_png_output}")


if __name__ == "__main__":
    main()
