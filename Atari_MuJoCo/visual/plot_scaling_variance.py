"""
验证2可视化：PPO vs OPTS_TTPO 策略梯度方差随 batch_size 的缩放
1×5 子图，横轴 batch_size，纵轴 PG 方差 (log scale)，跨 seed 聚合 mean ± std
"""
import os
import json
import sys
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


TARGET_TASKS = ["Hopper-v4", "Walker2d-v4", "HalfCheetah-v4", "Ant-v4", "Humanoid-v4"]
SEEDS = [1, 2, 3, 4, 5]

COLOR_PPO = "#1f77b4"
COLOR_OPTS = "#d62728"


def main():
    input_dir = sys.argv[1] if len(sys.argv) > 1 else "results/variance/verify2"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "visual"

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    for idx, task in enumerate(TARGET_TASKS):
        ax = axes[idx]
        ppo_by_bs = defaultdict(list)
        opts_by_bs = defaultdict(list)

        for seed in SEEDS:
            path = os.path.join(input_dir, f"{task}_seed{seed}.json")
            if not os.path.exists(path):
                continue
            with open(path, "r") as f:
                result = json.load(f)
            for bs_str, var in result.get("ppo_variance", {}).items():
                v = var["mean"] if isinstance(var, dict) else var
                ppo_by_bs[int(bs_str)].append(v)
            for bs_str, var in result.get("opts_variance", {}).items():
                v = var["mean"] if isinstance(var, dict) else var
                opts_by_bs[int(bs_str)].append(v)

        all_bs = sorted(set(ppo_by_bs.keys()) & set(opts_by_bs.keys()))
        if not all_bs:
            ax.set_title(task, fontsize=12)
            ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
            continue

        bs = np.array(all_bs)
        ppo_m = np.array([np.mean(ppo_by_bs[b]) for b in all_bs])
        ppo_s = np.array([np.std(ppo_by_bs[b]) for b in all_bs])
        opts_m = np.array([np.mean(opts_by_bs[b]) for b in all_bs])
        opts_s = np.array([np.std(opts_by_bs[b]) for b in all_bs])

        ax.plot(bs, ppo_m, 'o-', color=COLOR_PPO, linewidth=1.5, markersize=5, label="PPO")
        ax.fill_between(bs, ppo_m - ppo_s, ppo_m + ppo_s, color=COLOR_PPO, alpha=0.2)
        ax.plot(bs, opts_m, 's-', color=COLOR_OPTS, linewidth=1.5, markersize=5, label="OPTS_TTPO")
        ax.fill_between(bs, opts_m - opts_s, opts_m + opts_s, color=COLOR_OPTS, alpha=0.2)

        ax.set_yscale('log')
        ax.set_xscale('log', base=2)
        ax.set_xticks(bs)
        ax.set_xticklabels([str(b) for b in bs], fontsize=8)
        ax.set_title(task, fontsize=12)
        ax.set_xlabel("Batch Size", fontsize=10)
        if idx == 0:
            ax.set_ylabel("PG Variance (log scale)", fontsize=10)
        ax.grid(True, alpha=0.3, which='major')

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=11,
               bbox_to_anchor=(0.5, 1.08), frameon=True)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "scaling_variance_verify2.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    plt.close()


if __name__ == "__main__":
    main()
