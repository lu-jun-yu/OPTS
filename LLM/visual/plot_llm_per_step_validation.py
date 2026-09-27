#!/usr/bin/env python3
"""Plot marker-free learning curves in two rows and six benchmark columns."""

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
        "linewidth": 1.46,
        "linestyle": (0, (4, 2.5)),
        "alpha": 0.95,
        "zorder": 2,
    },
    "DAPO": {
        "color": "#4E9A6A",
        "linewidth": 1.46,
        "linestyle": (0, (4, 2, 1, 2)),
        "alpha": 0.95,
        "zorder": 2,
    },
    "REINFORCE++": {
        "color": "#E39A32",
        "linewidth": 1.46,
        "linestyle": (0, (1, 2)),
        "alpha": 0.95,
        "zorder": 2,
    },
    "OPTS-TTPO": {
        "color": "#D13F4A",
        "linewidth": 1.87,
        "linestyle": "-",
        "alpha": 1.0,
        "zorder": 6,
    },
}

EXPECTED_STEPS = tuple(range(20, 401, 20))
TEXT_COLOR = "#263238"
MUTED_TEXT = "#414B53"
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
        raise ValueError(f"Incomplete benchmark grid; missing={missing}, extra={extra}")
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
    start = min(finite_curve(series)[1][0] for series in method_values.values())
    return start, min(1.0, upper + padding)


def smooth_curve(values, window=3):
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


def style_axis(axis, show_x_labels):
    axis.set_facecolor("white")
    axis.set_xlim(15, 405)
    axis.set_xticks((20, 200, 400))
    axis.yaxis.set_major_locator(
        MaxNLocator(nbins=3, min_n_ticks=3, steps=(1, 2, 4, 5, 10))
    )
    axis.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    axis.grid(
        axis="y",
        color=GRID_COLOR,
        linestyle=(0, (4, 3)),
        linewidth=0.55,
        alpha=0.65,
    )
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(SPINE_COLOR)
        axis.spines[side].set_linewidth(0.65)
    axis.tick_params(
        axis="both",
        colors=MUTED_TEXT,
        labelsize=13.0,
        length=2.5,
        width=0.60,
        pad=2.0,
        direction="out",
    )
    axis.tick_params(axis="x", labelbottom=show_x_labels)


def layout_panels(figure, axes):
    """Center complete panels, including tick labels, inside separated columns."""
    left, right, gutter = 0.045, 0.992, 0.005
    width = (right - left - (axes.shape[1] - 1) * gutter) / axes.shape[1]
    height = 1.70 / figure.get_figheight()
    bottoms = (2.48 / figure.get_figheight(), 0.52 / figure.get_figheight())

    # Measure the decorations rather than treating the axes rectangle as a panel.
    for _ in range(3):
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        left_pad = right_pad = 0.0
        for axis in axes.flat:
            body = axis.get_tightbbox(renderer).transformed(figure.transFigure.inverted())
            plot = axis.get_position()
            left_pad = max(left_pad, plot.x0 - body.x0)
            right_pad = max(right_pad, body.x1 - plot.x1)
        plot_width = width - left_pad - right_pad
        for row in range(axes.shape[0]):
            for column in range(axes.shape[1]):
                x = left + column * (width + gutter) + left_pad
                axes[row, column].set_position((x, bottoms[row], plot_width, height))

    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    for column, (_, name) in enumerate(BENCHMARKS):
        bodies = [
            axis.get_tightbbox(renderer).transformed(figure.transFigure.inverted())
            for axis in axes[:, column]
        ]
        center = (min(body.x0 for body in bodies) + max(body.x1 for body in bodies)) / 2
        figure.text(
            center, 4.32 / figure.get_figheight(), name,
            ha="center", va="bottom", fontsize=14.0,
            fontweight="semibold", color=TEXT_COLOR,
        )
    for row, (_, label) in enumerate(METRICS):
        figure.text(
            0.029, bottoms[row] + height / 2, label,
            ha="center", va="center", rotation=90, fontsize=13.0,
        )


def plot_grid(curves):
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 9.5,
            "text.color": TEXT_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    figure = plt.figure(figsize=(11.4, 5.10), facecolor="white")
    axes = np.empty((len(METRICS), len(BENCHMARKS)), dtype=object)
    for row in range(len(METRICS)):
        for column in range(len(BENCHMARKS)):
            axes[row, column] = figure.add_axes((0.10, 0.15, 0.10, 0.33))

    for column, (benchmark, benchmark_name) in enumerate(BENCHMARKS):
        for metric_index, (metric, metric_label) in enumerate(METRICS):
            axis = axes[metric_index, column]
            style_axis(axis, show_x_labels=metric_index == 1)
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
                    alpha=style["alpha"],
                    solid_capstyle="round",
                    dash_capstyle="round",
                    marker=None,
                    zorder=style["zorder"],
                )

    layout_panels(figure, axes)

    figure.text(
        0.523,
        0.035,
        "Training steps",
        ha="center",
        va="center",
        fontsize=13.0,
        color=MUTED_TEXT,
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=METHOD_STYLE[method]["color"],
            linewidth=METHOD_STYLE[method]["linewidth"],
            linestyle=METHOD_STYLE[method]["linestyle"],
            marker=None,
            label=method,
        )
        for method in METHODS
    ]
    legend = figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.523, 0.985),
        ncol=4,
        frameon=False,
        fontsize=13.0,
        handlelength=2.3,
        handletextpad=0.60,
        columnspacing=1.30,
        borderpad=0.0,
    )
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
