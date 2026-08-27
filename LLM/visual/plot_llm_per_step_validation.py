#!/usr/bin/env python3
"""Plot the 2x6 per-step validation grid for Qwen3-1.7B runs."""

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from scipy.ndimage import uniform_filter1d


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO_ROOT / "LLM/results/step400/per-step"
DEFAULT_PDF = REPO_ROOT / "paper/figures/llm_per_step_validation.pdf"
DEFAULT_PNG = REPO_ROOT / "paper/figures/llm_per_step_validation.png"

BENCHMARKS = (
    ("val-core/hiyouga/math12k", "MATH500"),
    ("val-core/math-ai/minervamath", "MinervaMath"),
    ("val-core/math-ai/amc23", "AMC23"),
    ("val-core/math-ai/aime24", "AIME24"),
    ("val-core/math-ai/aime25", "AIME25"),
    ("val-core/math-ai/aime26", "AIME26"),
)

METRICS = (
    ("acc/avg@32", "Avg@32"),
    ("acc/pass@32", "Pass@32"),
)

AVAILABLE_METRICS = {"acc/avg@32", "acc/pass@32", "acc/cons@32"}

RAW_METHODS = {
    "ppo_0704_n8_1.7B": "PPO",
    "dapo_0703_n8_1.7B": "DAPO",
    "reinforce_pp_baseline_0703_n8_1.7B": "REINFORCE++",
    "opts_ttpo_exp8_3_0810_n8_1.7B": "OPTS-TTPO",
}

METHODS = ("PPO", "DAPO", "REINFORCE++", "OPTS-TTPO")
METHOD_STYLE = {
    "PPO": {
        "color": "#4c72b0",
        "linewidth": 1.30,
        "linestyle": (0, (4.0, 2.4)),
        "zorder": 3,
    },
    "DAPO": {
        "color": "#2ca02c",
        "linewidth": 1.40,
        "linestyle": "-",
        "zorder": 3,
    },
    "REINFORCE++": {
        "color": "#ff7f0e",
        "linewidth": 1.40,
        "linestyle": "-",
        "zorder": 3,
    },
    "OPTS-TTPO": {
        "color": "#c44e52",
        "linewidth": 1.85,
        "linestyle": "-",
        "zorder": 5,
    },
}

EXPECTED_STEPS = tuple(range(20, 401, 20))
TEXT_COLOR = "#263238"
MUTED_TEXT = "#5F6B73"
SPINE_COLOR = "#A5ADB3"
GRID_COLOR = "#CBD1D6"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot avg@32 and pass@32 for six benchmarks."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--png-output", type=Path, default=DEFAULT_PNG)
    return parser.parse_args()


def parse_value_header(header):
    raw_method, metric_spec = header.split(" - ", 1)
    raw_method = raw_method.removeprefix("Name: ")
    if raw_method not in RAW_METHODS:
        raise ValueError(f"Unknown run name in CSV header: {raw_method}")

    for benchmark, _ in BENCHMARKS:
        prefix = f"{benchmark}/"
        if metric_spec.startswith(prefix):
            metric = metric_spec[len(prefix) :]
            if metric not in AVAILABLE_METRICS:
                raise ValueError(f"Unknown validation metric: {metric}")
            return benchmark, metric, RAW_METHODS[raw_method]
    raise ValueError(f"Unknown benchmark in CSV header: {metric_spec}")


def load_curves(input_dir):
    curves = {}
    sources = {}

    for csv_path in sorted(input_dir.glob("*.csv")):
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            value_headers = [
                header
                for header in fieldnames
                if header != "Step"
                and not header.endswith("__MIN")
                and not header.endswith("__MAX")
            ]
            if len(value_headers) != len(METHODS):
                raise ValueError(
                    f"Expected {len(METHODS)} center-value columns in {csv_path}, "
                    f"found {len(value_headers)}"
                )

            parsed = [parse_value_header(header) for header in value_headers]
            benchmark_metric = {(benchmark, metric) for benchmark, metric, _ in parsed}
            if len(benchmark_metric) != 1:
                raise ValueError(f"Mixed benchmark or metric columns in {csv_path}")
            benchmark, metric = next(iter(benchmark_metric))
            if metric not in {value for value, _ in METRICS}:
                continue
            key = (benchmark, metric)
            if key in sources:
                raise ValueError(f"Duplicate export for {key}: {sources[key]} and {csv_path}")

            rows = list(reader)
            steps = tuple(int(float(row["Step"])) for row in rows)
            if steps != EXPECTED_STEPS:
                raise ValueError(f"Unexpected steps in {csv_path}: {steps}")

            method_headers = {method: header for header, (_, _, method) in zip(value_headers, parsed)}
            if set(method_headers) != set(METHODS):
                raise ValueError(f"Incomplete method set in {csv_path}: {sorted(method_headers)}")

            curves[key] = {}
            for method in METHODS:
                header = method_headers[method]
                values = []
                for row in rows:
                    raw_value = row[header].strip()
                    value = math.nan if raw_value == "" else float(raw_value)
                    if math.isfinite(value) and not 0.0 <= value <= 1.0:
                        raise ValueError(
                            f"Out-of-range value for {method} in {csv_path}: {value}"
                        )
                    values.append(value)
                curves[key][method] = values
            sources[key] = csv_path

    expected_keys = {
        (benchmark, metric)
        for benchmark, _ in BENCHMARKS
        for metric, _ in METRICS
    }
    if set(curves) != expected_keys:
        missing = sorted(expected_keys - set(curves))
        extra = sorted(set(curves) - expected_keys)
        raise ValueError(f"Incomplete 2x6 grid; missing={missing}, extra={extra}")
    return curves


def finite_bounds(method_values):
    values = [
        value
        for series in method_values.values()
        for value in series
        if math.isfinite(value)
    ]
    lower = min(values)
    upper = max(values)
    span = max(upper - lower, 0.08)
    padding = 0.09 * span
    return max(0.0, lower - padding), min(1.0, upper + padding)


def smooth_curve(values, window=5):
    finite_points = [
        (step, value)
        for step, value in zip(EXPECTED_STEPS, values)
        if math.isfinite(value)
    ]
    steps, finite_values = zip(*finite_points)
    finite_values = np.asarray(finite_values, dtype=float)
    smoothed = uniform_filter1d(
        finite_values,
        size=min(window, len(finite_values)),
        mode="nearest",
    )
    smoothed[0] = finite_values[0]
    smoothed[-1] = finite_values[-1]
    return steps, smoothed


def style_axis(axis, row, column):
    axis.set_facecolor("white")
    axis.set_xlim(15, 405)
    axis.set_xticks((20, 200, 400))
    if row < len(METRICS) - 1:
        axis.tick_params(axis="x", labelbottom=False)
    axis.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
    axis.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    axis.grid(
        axis="y",
        color=GRID_COLOR,
        linestyle=(0, (3, 3)),
        linewidth=0.60,
        alpha=0.72,
    )
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(SPINE_COLOR)
        axis.spines[side].set_linewidth(0.7)
    axis.tick_params(
        axis="both",
        colors=MUTED_TEXT,
        labelsize=9.0,
        length=2.5,
        width=0.65,
        pad=1.6,
    )
    if column == 0:
        axis.tick_params(axis="y", labelsize=9.1)
        axis.set_ylabel(
            METRICS[row][1],
            fontsize=11.3,
            fontweight="semibold",
            color=TEXT_COLOR,
            labelpad=3.0,
        )


def plot_grid(curves):
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10.0,
            "text.color": TEXT_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    figure, axes = plt.subplots(2, 6, figsize=(12.2, 5.9), squeeze=False)
    figure.subplots_adjust(
        left=0.067,
        right=0.995,
        bottom=0.090,
        top=0.885,
        wspace=0.32,
        hspace=0.28,
    )

    for column, (_, benchmark_name) in enumerate(BENCHMARKS):
        axes[0, column].set_title(
            benchmark_name,
            fontsize=12.3,
            fontweight="semibold",
            pad=8.0,
        )

    for row, (metric, _) in enumerate(METRICS):
        for column, (benchmark, _) in enumerate(BENCHMARKS):
            axis = axes[row, column]
            style_axis(axis, row, column)
            panel = curves[(benchmark, metric)]
            axis.set_ylim(*finite_bounds(panel))

            for method in METHODS:
                style = METHOD_STYLE[method]
                finite_steps, smoothed_values = smooth_curve(panel[method])
                axis.plot(
                    finite_steps,
                    smoothed_values,
                    color=style["color"],
                    linewidth=style["linewidth"],
                    linestyle=style["linestyle"],
                    alpha=1.0 if method == "OPTS-TTPO" else 0.90,
                    solid_capstyle="round",
                    zorder=style["zorder"],
                )

    figure.text(
        0.535,
        0.027,
        "Training step",
        ha="center",
        va="center",
        fontsize=10.8,
        color=MUTED_TEXT,
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=METHOD_STYLE[method]["color"],
            linewidth=METHOD_STYLE[method]["linewidth"] + 0.15,
            linestyle=METHOD_STYLE[method]["linestyle"],
            label=method,
        )
        for method in METHODS
    ]
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.535, 0.975),
        ncol=4,
        frameon=False,
        fontsize=10.6,
        handlelength=2.6,
        handletextpad=0.55,
        columnspacing=1.7,
    )
    return figure


def save_figure(figure, pdf_path, png_path):
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(pdf_path, bbox_inches="tight", pad_inches=0.025)
    print(f"Saved PDF: {pdf_path}", flush=True)
    if png_path is not None:
        png_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            png_path,
            dpi=240,
            bbox_inches="tight",
            pad_inches=0.025,
        )
        print(f"Saved PNG: {png_path}", flush=True)
    plt.close(figure)


def main():
    args = parse_args()
    curves = load_curves(args.input_dir)
    figure = plot_grid(curves)
    save_figure(figure, args.output, args.png_output)


if __name__ == "__main__":
    main()
