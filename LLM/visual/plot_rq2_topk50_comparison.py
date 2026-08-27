"""Compare unrestricted and Top-K=50 OPTS compute-scaling results."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_rq2_compute_scaling import (
    EVAL_DIR,
    IID_COLOR,
    K_VALUES,
    METHOD_TAG,
    MUTED_TEXT_COLOR,
    OPTS_COLOR,
    TEXT_COLOR,
    dynamic_performance_ylim,
    load_evaluation_series,
    style_axis,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT = REPO_ROOT / "paper/figures/rq2_compute_scaling_topk50_comparison.pdf"
PNG_OUTPUT = REPO_ROOT / "paper/figures/rq2_compute_scaling_topk50_comparison.png"
TOPK50_COLOR = "#D66A3A"

REWARD_IID = EVAL_DIR / (
    f"{METHOD_TAG}_iid_n128__task2_iid_pass_k8-16-32-64-128.json"
)
VALUE_IID = EVAL_DIR / (
    f"{METHOD_TAG}_iid_n128__task3_iid_cons_k8-16-32-64-128.json"
)
REWARD_UNRESTRICTED = EVAL_DIR / (
    f"{METHOD_TAG}_opts_reward_s3_n128__task2_reward_opts_k8-16-32-64-128.json"
)
VALUE_UNRESTRICTED = EVAL_DIR / (
    f"{METHOD_TAG}_opts_value_s3_n128__task3_value_opts_k8-16-32-64-128.json"
)
REWARD_TOPK50 = EVAL_DIR / (
    f"{METHOD_TAG}_opts_reward_s3_topk50_n128__task2_reward_opts_k8-16-32-64-128.json"
)
VALUE_TOPK50 = EVAL_DIR / (
    f"{METHOD_TAG}_opts_value_s3_topk50_n128__task3_value_opts_k8-16-32-64-128.json"
)


def pooled(path, metric):
    return load_evaluation_series(path, metric)["_all"]


def plot_panel(axis, iid, unrestricted, topk50, title, ylabel):
    x_values = range(len(K_VALUES))
    iid_line, = axis.plot(
        x_values,
        iid,
        color=IID_COLOR,
        linestyle=(0, (4, 2.5)),
        linewidth=1.75,
        marker="o",
        markersize=5.0,
        markerfacecolor="white",
        markeredgecolor=IID_COLOR,
        markeredgewidth=1.1,
        label="IID (Top-K=50)",
        zorder=2,
    )
    unrestricted_line, = axis.plot(
        x_values,
        unrestricted,
        color=OPTS_COLOR,
        linewidth=2.15,
        marker="D",
        markersize=5.1,
        markerfacecolor=OPTS_COLOR,
        markeredgecolor="white",
        markeredgewidth=0.7,
        label="OPTS (unrestricted)",
        zorder=3,
    )
    topk50_line, = axis.plot(
        x_values,
        topk50,
        color=TOPK50_COLOR,
        linewidth=2.15,
        marker="s",
        markersize=5.0,
        markerfacecolor=TOPK50_COLOR,
        markeredgecolor="white",
        markeredgewidth=0.7,
        label="OPTS (Top-K=50)",
        zorder=4,
    )
    style_axis(
        axis,
        dynamic_performance_ylim(iid, unrestricted, topk50),
        ylabel,
    )
    axis.set_title(
        title,
        color=TEXT_COLOR,
        fontsize=12.0,
        fontweight="semibold",
        pad=8.0,
    )
    axis.set_xlabel(r"Rollout budget $k$", fontsize=10.2, labelpad=5.0)
    return iid_line, unrestricted_line, topk50_line


def main():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10.2,
            "text.color": TEXT_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "xtick.color": MUTED_TEXT_COLOR,
            "ytick.color": MUTED_TEXT_COLOR,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    reward_iid = pooled(REWARD_IID, "pass")
    value_iid = pooled(VALUE_IID, "cons")
    reward_unrestricted = pooled(REWARD_UNRESTRICTED, "opts")
    value_unrestricted = pooled(VALUE_UNRESTRICTED, "opts")
    reward_topk50 = pooled(REWARD_TOPK50, "opts")
    value_topk50 = pooled(VALUE_TOPK50, "opts")

    figure, axes = plt.subplots(1, 2, figsize=(7.4, 2.85))
    figure.subplots_adjust(
        left=0.083,
        right=0.992,
        bottom=0.31,
        top=0.80,
        wspace=0.29,
    )
    handles = plot_panel(
        axes[0],
        reward_iid,
        reward_unrestricted,
        reward_topk50,
        "(a) Reward-Guided Performance",
        "pass@k",
    )
    plot_panel(
        axes[1],
        value_iid,
        value_unrestricted,
        value_topk50,
        "(b) Value-Guided Performance",
        "cons@k",
    )
    figure.legend(
        handles,
        ["IID (Top-K=50)", "OPTS (unrestricted)", "OPTS (Top-K=50)"],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=3,
        frameon=False,
        fontsize=8.9,
        handlelength=2.45,
        columnspacing=1.45,
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        OUTPUT,
        bbox_inches="tight",
        pad_inches=0.035,
        metadata={"Title": "OPTS Top-K=50 Comparison"},
    )
    figure.savefig(
        PNG_OUTPUT,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.035,
    )
    plt.close(figure)

    reward_delta = [100 * (new - old) for new, old in zip(reward_topk50, reward_unrestricted)]
    value_delta = [100 * (new - old) for new, old in zip(value_topk50, value_unrestricted)]
    print("Reward Top-K=50 minus unrestricted (pp):", [round(x, 3) for x in reward_delta])
    print("Value Top-K=50 minus unrestricted (pp):", [round(x, 3) for x in value_delta])
    print(f"Wrote {OUTPUT}")
    print(f"Wrote {PNG_OUTPUT}")


if __name__ == "__main__":
    main()
