#!/usr/bin/env python3
"""xi x s hyper-parameter grid on the two MuJoCo development tasks."""

import argparse
import glob
import json
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = Path("/data/results/1_2048")
PREFIX = "opts_ttpo_continuous_action_wEqual-bMax-nLen_"
TASKS = ["Hopper-v4", "Humanoid-v4"]
XIS = [round(0.1 * i, 1) for i in range(11)]
SS = list(range(1, 9))
MAIN = (0.6, 1)
TAIL_POINTS = 100
TEXT_COLOR = "#263238"


def load_grid(results_dir):
    """returns dict[(task, xi, s)] -> (full_mean, tail_mean) averaged over seeds; also seed counts."""
    grid, counts = {}, {}
    for d in sorted(glob.glob(str(results_dir / f"{PREFIX}*_s*_*"))):
        m = re.search(r"xi([\d.]+)_s(\d+)_\d{8}$", d)
        if not m:
            continue
        xi, s = float(m.group(1)), int(m.group(2))
        for task in TASKS:
            full, tail = [], []
            for f in glob.glob(f"{d}/{task}_*.json"):
                rows = [float(r["mean_return"]) for r in json.load(open(f)) if "mean_return" in r]
                if not rows:
                    continue
                full.append(np.mean(rows))
                tail.append(np.mean(rows[-TAIL_POINTS:]))
            if full:
                grid[(task, xi, s)] = (float(np.mean(full)), float(np.mean(tail)))
                counts[(task, xi, s)] = len(full)
    return grid, counts


def matrix(grid, task, which):
    m = np.full((len(XIS), len(SS)), np.nan)
    for i, xi in enumerate(XIS):
        for j, s in enumerate(SS):
            if (task, xi, s) in grid:
                m[i, j] = grid[(task, xi, s)][which]
    return m


def normalized(mats):
    out = np.zeros_like(mats[0])
    for m in mats:
        lo, hi = np.nanmin(m), np.nanmax(m)
        out += (m - lo) / max(hi - lo, 1e-9)
    return out / len(mats)


def draw(ax, m, title, cmap, fmt):
    im = ax.imshow(m, cmap=cmap, aspect="auto", origin="lower")
    ax.set_xticks(range(len(SS)))
    ax.set_xticklabels([str(s) for s in SS], fontsize=8)
    ax.set_yticks(range(len(XIS)))
    ax.set_yticklabels([f"{x:.1f}" for x in XIS], fontsize=8)
    ax.set_title(title, fontsize=10.5, fontweight="semibold", color=TEXT_COLOR, pad=5)
    i, j = XIS.index(MAIN[0]), SS.index(MAIN[1])
    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="black", linewidth=1.8))
    bi, bj = np.unravel_index(np.nanargmax(m), m.shape)
    ax.add_patch(Rectangle((bj - 0.5, bi - 0.5), 1, 1, fill=False, edgecolor="white", linewidth=1.4, linestyle="--"))
    ax.tick_params(length=2, colors="#414B53")
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=7)
    cb.formatter = matplotlib.ticker.FormatStrFormatter(fmt)
    cb.update_ticks()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    ap.add_argument("--output", type=Path, default=REPO_ROOT / "paper/figures/mujoco_xi_s_grid.pdf")
    args = ap.parse_args()
    grid, counts = load_grid(args.results_dir)
    expected = len(TASKS) * len(XIS) * len(SS)
    print(f"grid cells: {len(grid)}/{expected}; seeds per cell: min {min(counts.values())} max {max(counts.values())}")

    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
                         "text.color": TEXT_COLOR, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, len(TASKS) + 1, figsize=(15.5, 6.4), facecolor="white")
    summary = {}
    for row, (which, label) in enumerate(((0, "Full-training mean"), (1, "Tail mean"))):
        mats = [matrix(grid, task, which) for task in TASKS]
        for col, (task, m) in enumerate(zip(TASKS, mats)):
            draw(axes[row, col], m, task.replace("-v4", ""), "viridis", "%.0f")
        norm = normalized(mats)
        summary[label] = norm
        draw(axes[row, -1], norm, "Normalized (2 dev. tasks)", "magma", "%.2f")
        axes[row, 0].set_ylabel(label + "\n" + r"length-penalty exponent $\xi$", fontsize=9.5, color="#414B53")
        bi, bj = np.unravel_index(np.nanargmax(norm), norm.shape)
        print(f"{label}: best cross-task cell xi={XIS[bi]}, s={SS[bj]} ({norm[bi, bj]:.3f}); main-text (0.6,1) = {norm[XIS.index(0.6), 0]:.3f}, "
              f"rank {int((norm > norm[XIS.index(0.6), 0]).sum()) + 1}/{norm.size}")
    for ax in axes[-1]:
        ax.set_xlabel(r"Maximum search count $s$", fontsize=9.5, color="#414B53")
    fig.tight_layout(w_pad=1.2, h_pad=1.4)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(args.output.with_suffix(".png"), dpi=200, bbox_inches="tight", pad_inches=0.03)
    print(f"Saved {args.output}")
    np.savez(args.output.with_suffix(".npz"), **{f"{t}_{w}": matrix(grid, t, i) for t in TASKS for i, w in ((0, "full"), (1, "tail"))},
             norm_full=summary["Full-training mean"], norm_tail=summary["Tail mean"])


if __name__ == "__main__":
    main()
