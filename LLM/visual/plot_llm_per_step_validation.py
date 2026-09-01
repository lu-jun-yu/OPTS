#!/usr/bin/env python3
"""Plot the 4x3 per-step validation grid for Qwen3-1.7B runs."""

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
        "color": "#4776A8",
        "linewidth": 1.25,
        "linestyle": ":",
        "marker": "^",
        "alpha": 0.92,
        "zorder": 2,
    },
    "DAPO": {
        "color": "#4E9A6A",
        "linewidth": 1.25,
        "linestyle": "-.",
        "marker": "s",
        "alpha": 0.92,
        "zorder": 2,
    },
    "REINFORCE++": {
        "color": "#E39A32",
        "linewidth": 1.25,
        "linestyle": "--",
        "marker": "D",
        "alpha": 0.92,
        "zorder": 2,
    },
    "OPTS-TTPO": {
        "color": "#D13F4A",
        "linewidth": 2.05,
        "linestyle": "-",
        "marker": "o",
        "alpha": 1.0,
        "zorder": 6,
    },
}

EXPECTED_STEPS = tuple(range(20, 401, 20))
TEXT_COLOR = "#22262A"
MUTED_TEXT = "#525A61"
SPINE_COLOR = "#858D94"
GRID_COLOR = "#C9CFD4"


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
    steps, finite_values = finite_curve(values)
    smoothed = uniform_filter1d(
        finite_values,
        size=min(window, len(finite_values)),
        mode="nearest",
    )
    smoothed[0] = finite_values[0]
    smoothed[-1] = finite_values[-1]
    return steps, smoothed


def finite_curve(values):
    finite_points = [
        (step, value)
        for step, value in zip(EXPECTED_STEPS, values)
        if math.isfinite(value)
    ]
    steps, finite_values = zip(*finite_points)
    return np.asarray(steps), np.asarray(finite_values, dtype=float)


def style_axis(axis, metric_label, column, show_x_labels):
    axis.set_facecolor("white")
    axis.set_xlim(15, 405)
    axis.set_xticks((20, 100, 200, 300, 400))
    if not show_x_labels:
        axis.tick_params(axis="x", labelbottom=False)
    axis.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
    axis.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    axis.grid(
        axis="both",
        color=GRID_COLOR,
        linestyle="-",
        linewidth=0.55,
        alpha=0.62,
    )
    axis.set_axisbelow(True)
    for side in ("left", "right", "top", "bottom"):
        axis.spines[side].set_color(SPINE_COLOR)
        axis.spines[side].set_linewidth(0.72)
    axis.tick_params(
        axis="both",
        colors=MUTED_TEXT,
        labelsize=8.1,
        length=2.8,
        width=0.70,
        pad=1.4,
        direction="out",
    )
    if column == 0:
        axis.tick_params(axis="y", labelsize=8.4)
        axis.set_ylabel(
            metric_label,
            fontsize=9.4,
            fontweight="semibold",
            color=TEXT_COLOR,
            labelpad=2.5,
        )


def plot_grid(curves):
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 9.5,
            "text.color": TEXT_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    figure = plt.figure(figsize=(7.4, 7.75), facecolor="white")
    outer_grid = figure.add_gridspec(
        2,
        1,
        left=0.105,
        right=0.990,
        bottom=0.073,
        top=0.925,
        hspace=0.255,
    )
    axes = np.empty((4, 3), dtype=object)
    for block in range(2):
        block_grid = outer_grid[block, 0].subgridspec(
            2,
            3,
            wspace=0.23,
            hspace=0.10,
        )
        for local_row in range(2):
            for column in range(3):
                axes[2 * block + local_row, column] = figure.add_subplot(
                    block_grid[local_row, column]
                )

    metric_labels = dict(METRICS)
    panel_rows = (
        ("acc/avg@32", BENCHMARKS[:3]),
        ("acc/pass@32", BENCHMARKS[:3]),
        ("acc/avg@32", BENCHMARKS[3:]),
        ("acc/pass@32", BENCHMARKS[3:]),
    )

    for row, (metric, benchmarks) in enumerate(panel_rows):
        for column, (benchmark, benchmark_name) in enumerate(benchmarks):
            axis = axes[row, column]
            style_axis(
                axis,
                metric_labels[metric],
                column,
                show_x_labels=row in (1, 3),
            )
            if row in (0, 2):
                axis.set_title(
                    benchmark_name,
                    fontsize=10.2,
                    fontweight="semibold",
                    color=TEXT_COLOR,
                    pad=4.2,
                )
            panel = curves[(benchmark, metric)]
            axis.set_ylim(*finite_bounds(panel))

            for method in METHODS:
                style = METHOD_STYLE[method]
                raw_steps, raw_values = finite_curve(panel[method])
                finite_steps, smoothed_values = smooth_curve(panel[method])
                axis.plot(
                    raw_steps,
                    raw_values,
                    color=style["color"],
                    linewidth=0.75 if method == "OPTS-TTPO" else 0.60,
                    linestyle="-",
                    alpha=0.24 if method == "OPTS-TTPO" else 0.16,
                    zorder=1,
                )
                axis.plot(
                    finite_steps,
                    smoothed_values,
                    color=style["color"],
                    linewidth=style["linewidth"],
                    linestyle=style["linestyle"],
                    alpha=style["alpha"],
                    solid_capstyle="round",
                    dash_capstyle="round",
                    marker=style["marker"],
                    markersize=3.4 if method == "OPTS-TTPO" else 2.9,
                    markevery=2,
                    markerfacecolor=style["color"],
                    markeredgecolor="white",
                    markeredgewidth=0.45,
                    zorder=style["zorder"],
                )

    figure.text(
        0.545,
        0.020,
        "Training steps",
        ha="center",
        va="center",
        fontsize=9.3,
        color=MUTED_TEXT,
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=METHOD_STYLE[method]["color"],
            linewidth=METHOD_STYLE[method]["linewidth"] + 0.15,
            linestyle=METHOD_STYLE[method]["linestyle"],
            marker=METHOD_STYLE[method]["marker"],
            markersize=3.8,
            markerfacecolor=METHOD_STYLE[method]["color"],
            markeredgecolor="white",
            markeredgewidth=0.5,
            label=method,
        )
        for method in METHODS
    ]
    legend = figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.545, 0.989),
        ncol=4,
        frameon=True,
        fancybox=True,
        framealpha=1.0,
        facecolor="white",
        edgecolor="#CDD3D8",
        fontsize=8.8,
        handlelength=2.4,
        handletextpad=0.50,
        columnspacing=1.25,
        borderpad=0.45,
    )
    legend.get_frame().set_linewidth(0.65)
    for text_item in legend.get_texts():
        if text_item.get_text() == "OPTS-TTPO":
            text_item.set_fontweight("semibold")
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
