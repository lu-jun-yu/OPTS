#!/usr/bin/env python3
"""Avg@32 versus OPTS search rounds for performance-difference, random and midpoint rebranching (appendix figure)."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = REPO_ROOT / "LLM/results/step400/appendix3_20260915/eval"
DEFAULT_PDF = REPO_ROOT / "paper/figures/rebranch_position_ablation.pdf"
DEFAULT_PNG = REPO_ROOT / "paper/figures/rebranch_position_ablation.png"
ROUNDS = (0, 1, 3, 7)
BENCHMARKS = (
    ("hiyouga/math12k", "MATH500"), ("math-ai/minervamath", "MinervaMath"), ("math-ai/amc23", "AMC23"),
    ("math-ai/aime24", "AIME24"), ("math-ai/aime25", "AIME25"), ("math-ai/aime26", "AIME26"),
)
RULES = (
    ("perf_diff", "Performance-difference", {"color": "#D13F4A", "linewidth": 1.9, "linestyle": "-", "marker": "o", "zorder": 6}),
    ("random", "Uniform random", {"color": "#4776A8", "linewidth": 1.5, "linestyle": (0, (4, 2.5)), "marker": "s", "zorder": 3}),
    ("midpoint", "Fixed midpoint", {"color": "#4E9A6A", "linewidth": 1.5, "linestyle": (0, (4, 2, 1, 2)), "marker": "^", "zorder": 3}),
)
TEXT_COLOR, MUTED_TEXT, SPINE_COLOR, GRID_COLOR = "#263238", "#414B53", "#A5ADB3", "#CBD1D6"


def load(eval_dir):
    curves = {}
    for key, _, _ in RULES:
        path = next(eval_dir.glob(f"*_{key}_s7_*_k32.json"), None)
        if path is None and key == "perf_diff":
            # Result files from runs predating the rule rename use the old filename tag.
            path = next(eval_dir.glob("*_otrc_s7_*_k32.json"))
        results = json.load(open(path))["results"]
        curves[key] = {b: [results[b][f"opts-avg_s{s}@32"] for s in ROUNDS] for b, _ in BENCHMARKS}
    return curves


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", type=Path, default=EVAL_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--png-output", type=Path, default=DEFAULT_PNG)
    args = parser.parse_args()
    curves = load(args.eval_dir)

    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
                         "font.size": 9.5, "text.color": TEXT_COLOR, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, len(BENCHMARKS), figsize=(11.4, 2.35), facecolor="white")
    for axis, (bench, name) in zip(axes, BENCHMARKS):
        for key, _, style in RULES:
            axis.plot(range(len(ROUNDS)), curves[key][bench], markersize=3.6, markeredgewidth=0.0, **style)
        axis.set_title(name, fontsize=12.0, fontweight="semibold", color=TEXT_COLOR, pad=6)
        axis.set_xticks(range(len(ROUNDS)))
        axis.set_xticklabels([str(s) for s in ROUNDS])
        axis.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=(1, 2, 5, 10)))
        axis.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        axis.grid(axis="y", color=GRID_COLOR, linestyle=(0, (4, 3)), linewidth=0.55, alpha=0.65)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color(SPINE_COLOR)
            axis.spines[side].set_linewidth(0.65)
        axis.tick_params(axis="both", colors=MUTED_TEXT, labelsize=10.0, length=2.5, width=0.6, pad=2.0)
    axes[0].set_ylabel("Avg@32", fontsize=11.0, color=TEXT_COLOR)
    handles = [Line2D([0], [0], label=label, markersize=3.6, markeredgewidth=0.0, **style) for _, label, style in RULES]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=3, frameon=False, fontsize=11.0,
               handlelength=2.4, columnspacing=1.6)
    fig.tight_layout(rect=(0, 0.0, 1, 0.96))
    y0 = min(axis.get_position().y0 for axis in axes)
    fig.text(0.5, y0 - 0.19, r"Maximum OPTS search rounds $S_{\max}$", ha="center", va="top",
             fontsize=11.0, color=MUTED_TEXT)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(args.png_output, dpi=240, bbox_inches="tight", pad_inches=0.03)
    print(f"Saved {args.output} and {args.png_output}")


if __name__ == "__main__":
    main()
