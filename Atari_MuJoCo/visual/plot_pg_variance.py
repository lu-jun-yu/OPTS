"""
验证1可视化：正/负轨迹策略梯度方差对比
2×3 grid（5 任务 + legend），复用 plot_mujoco.py 配色风格
"""
import os
import json
import sys
import matplotlib.pyplot as plt
import numpy as np


TARGET_TASKS = ["Ant-v4", "HalfCheetah-v4", "Hopper-v4", "Humanoid-v4", "Walker2d-v4"]
SEEDS = [1, 2, 3, 4, 5]

COLOR_POS = "#1f77b4"  # 蓝（冷色）
COLOR_NEG = "#d62728"  # 红（暖色）


def main():
    input_dir = sys.argv[1] if len(sys.argv) > 1 else "results/variance/verify1"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "visual"

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for idx, task in enumerate(TARGET_TASKS):
        ax = axes[idx]
        seeds_found, pos_vars, neg_vars = [], [], []

        for seed in SEEDS:
            path = os.path.join(input_dir, f"{task}_seed{seed}.json")
            if not os.path.exists(path):
                continue
            with open(path, "r") as f:
                result = json.load(f)
            seeds_found.append(seed)
            pos_vars.append(result["positive_variance"])
            neg_vars.append(result["negative_variance"])

        if not seeds_found:
            ax.set_title(task, fontsize=12)
            ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
            continue

        x = np.arange(len(seeds_found))
        ax.plot(seeds_found, pos_vars, 'o-', color=COLOR_POS, linewidth=1.5, markersize=6)
        ax.plot(seeds_found, neg_vars, 's-', color=COLOR_NEG, linewidth=1.5, markersize=6)

        ax.set_title(task, fontsize=12)
        ax.set_xticks(seeds_found)
        ax.set_xticklabels([f"S{s}" for s in seeds_found])
        ax.set_xlabel("Seed")
        ax.set_ylabel("PG Variance")
        ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
        ax.grid(True, alpha=0.3)

    # legend panel
    axes[5].axis('off')
    handles = [
        plt.Line2D([0], [0], color=COLOR_POS, marker='o', linewidth=2),
        plt.Line2D([0], [0], color=COLOR_NEG, marker='s', linewidth=2),
    ]
    axes[5].legend(handles, ["Positive trajectories", "Negative trajectories"],
                   loc='center', fontsize=11, frameon=True)

    plt.suptitle("Policy Gradient Variance: Positive vs Negative Trajectories", fontsize=14)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "pg_variance_verify1.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    plt.close()


if __name__ == "__main__":
    main()
