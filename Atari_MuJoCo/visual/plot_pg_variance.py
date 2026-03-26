"""
验证1可视化：正/负轨迹策略梯度方差 vs batch_size 折线图
1×5 子图，横轴 batch_size，纵轴 PG 方差 (log scale)
"""
import os
import json
import sys
import matplotlib.pyplot as plt
import numpy as np


TARGET_TASKS = ["Ant-v4", "HalfCheetah-v4", "Hopper-v4", "Humanoid-v4", "Walker2d-v4"]
SEEDS = [1, 2, 3, 4, 5]

COLOR_POS = "#1f77b4"
COLOR_NEG = "#d62728"


def main():
    input_dir = sys.argv[1] if len(sys.argv) > 1 else "results/variance/verify1"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "visual"

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    for idx, task in enumerate(TARGET_TASKS):
        ax = axes[idx]
        all_pos = []  # list of arrays, one per seed
        all_neg = []
        batch_sizes = None

        for seed in SEEDS:
            path = os.path.join(input_dir, f"{task}_seed{seed}.json")
            if not os.path.exists(path):
                continue
            with open(path, "r") as f:
                result = json.load(f)
            batch_sizes = result["batch_sizes"]
            pos_v = result["pos_variances"]
            neg_v = result["neg_variances"]
            # filter out None entries (batch_size exceeded available steps)
            all_pos.append([v if v is not None else np.nan for v in pos_v])
            all_neg.append([v if v is not None else np.nan for v in neg_v])

        if not all_pos or batch_sizes is None:
            ax.set_title(task, fontsize=12)
            ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
            continue

        all_pos = np.array(all_pos)  # (num_seeds, num_batch_sizes)
        all_neg = np.array(all_neg)
        x = np.array(batch_sizes)

        # Compute mean and std in log space so error bands are visible on log scale
        log_pos = np.log(all_pos + 1e-30)  # avoid log(0)
        log_neg = np.log(all_neg + 1e-30)

        pos_log_mean = np.nanmean(log_pos, axis=0)
        pos_log_std = np.nanstd(log_pos, axis=0)
        neg_log_mean = np.nanmean(log_neg, axis=0)
        neg_log_std = np.nanstd(log_neg, axis=0)

        pos_mean = np.exp(pos_log_mean)
        pos_upper = np.exp(pos_log_mean + pos_log_std)
        pos_lower = np.exp(pos_log_mean - pos_log_std)
        neg_mean = np.exp(neg_log_mean)
        neg_upper = np.exp(neg_log_mean + neg_log_std)
        neg_lower = np.exp(neg_log_mean - neg_log_std)

        ax.plot(x, pos_mean, 'o-', color=COLOR_POS, linewidth=1.5, markersize=5, label="Positive traj.")
        ax.fill_between(x, pos_lower, pos_upper, color=COLOR_POS, alpha=0.2)
        ax.plot(x, neg_mean, 's-', color=COLOR_NEG, linewidth=1.5, markersize=5, label="Negative traj.")
        ax.fill_between(x, neg_lower, neg_upper, color=COLOR_NEG, alpha=0.2)

        ax.set_yscale('log')
        ax.set_xscale('log', base=2)
        ax.set_xticks(x)
        ax.set_xticklabels([str(b) for b in batch_sizes], fontsize=8)
        ax.set_title(task, fontsize=12)
        ax.set_xlabel("Batch Size", fontsize=10)
        if idx == 0:
            ax.set_ylabel("PG Variance", fontsize=10)
        ax.grid(True, alpha=0.3, which='both')

    # shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=11,
               bbox_to_anchor=(0.5, 1.08), frameon=True)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "pg_variance_verify1.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    plt.close()


if __name__ == "__main__":
    main()
