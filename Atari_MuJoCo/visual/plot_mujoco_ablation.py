"""
OPTS-TTPO 连续控制 MuJoCo 超参数消融：仅绘制不同 tau/s 配置的收敛曲线（1×5 子图）。

用法：
 python visual/plot_mujoco_ablation.py results/1_4096 20260410 → 绘制该日期下所有 opts_ttpo_continuous_action_tau*_s*_* 目录。

    python visual/plot_mujoco_ablation.py results/1_4096 20260410 tau0.0_s2,tau0.4_s6
        → 仅绘制指定 tau/s 标签对应的目录。

目录名需匹配：opts_ttpo_continuous_action_{tau0.x_sN}_{YYYYMMDD}

分层消融（固定 τ 下对所有搜索次数 s 的曲线做逐点平均，再比 τ；再在最优 τ 下比 s）：
 python visual/plot_mujoco_ablation.py results/1_4096 20260410 --hierarchical
输出：all_tasks_mujoco_ablation_{date}_mean_over_s.png、 all_tasks_mujoco_ablation_{date}_best_tau_s.png，并在终端打印推荐 τ 与 s。
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_mujoco import (
    COLOR_OPTS_TTPO,
    COLOR_PPO,
    COLOR_RPO,
    EXTRA_ALGO_COLORS,
    TARGET_TASKS,
    aggregate_seed_results,
    load_episodic_returns,
    parse_result_path,
    smooth_data,
)

# 与 plot_mujoco 主图一致的颜色集，按序循环分配给多条消融曲线
ABLATION_LINE_COLORS = [
    COLOR_OPTS_TTPO,
    COLOR_PPO,
    COLOR_RPO,
    *EXTRA_ALGO_COLORS,
]

# 消融图：统一使用更大的平滑窗口（比 plot_mujoco 默认更“重”）
DEFAULT_ABLATION_SMOOTH_WINDOW = 21

ABLATION_DIR_RE = re.compile(
    r"^opts_ttpo_continuous_action_(tau[\d.]+_s\d+)_(\d{8})$"
)
# 与目录标签一致：tau0.4_s6
TAU_S_FROM_TAG_RE = re.compile(r"^tau([\d.]+)_s(\d+)$")


def discover_ablation_run_dirs(
    results_subdir: Path,
    date: str,
    hyper_tags: list[str] | None,
) -> list[Path]:
    """列出 results_subdir 下符合命名规则的 OPTS-TTPO 消融运行目录（已排序）。"""
    if not results_subdir.is_dir():
        print(f"Results subdirectory does not exist: {results_subdir}")
        return []

    chosen: list[tuple[str, Path]] = []
    for d in sorted(results_subdir.iterdir()):
        if not d.is_dir():
            continue
        m = ABLATION_DIR_RE.match(d.name)
        if not m:
            continue
        tag, dir_date = m.group(1), m.group(2)
        if dir_date != date:
            continue
        if hyper_tags is not None and tag not in hyper_tags:
            continue
        chosen.append((tag, d))

    chosen.sort(key=lambda x: x[0])
    return [p for _, p in chosen]


def collect_aggregated_per_task(
    run_dirs: list[Path],
    seed_filters: set[int] | None,
) -> dict[str, dict[tuple[str, str], dict]]:
    """
    从给定运行目录中聚合各任务的曲线数据。
    返回: task_name -> (algo_name, date) -> {steps, mean, std}
    """
    # task -> algo_key -> seed -> run tuple
    per_task: dict[str, dict[tuple[str, str], dict]] = defaultdict(
        lambda: defaultdict(dict)
    )

    for run_dir in run_dirs:
        for filepath in run_dir.rglob("*.json"):
            parsed = parse_result_path(filepath)
            if parsed is None:
                continue
            task, algo_name, date, seed = parsed
            if task not in TARGET_TASKS:
                continue

            if seed_filters is not None:
                if seed not in seed_filters:
                    continue
            else:
                if seed < 1 or seed > 10:
                    continue

            step_values, mean_return_values, max_return_values, min_return_values = (
                load_episodic_returns(filepath)
            )
            if not mean_return_values:
                continue
            algo_key = (algo_name, date)
            per_task[task][algo_key][seed] = (
                step_values,
                mean_return_values,
                max_return_values,
                min_return_values,
            )

    all_tasks_data: dict[str, dict[tuple[str, str], dict]] = {
        t: {} for t in TARGET_TASKS
    }
    for task_name in TARGET_TASKS:
        for algo_key, seed_data in per_task[task_name].items():
            aggregated_steps, aggregated_mean, aggregated_std = aggregate_seed_results(
                seed_data
            )
            if aggregated_mean:
                all_tasks_data[task_name][algo_key] = {
                    "steps": aggregated_steps,
                    "mean": aggregated_mean,
                    "std": aggregated_std,
                }

    return all_tasks_data


def _display_name_for_ablation(algo_name: str) -> str:
    """从 opts_ttpo_continuous_action_tauX_sY 得到图例标签。"""
    prefix = "opts_ttpo_continuous_action_"
    if algo_name.startswith(prefix):
        rest = algo_name[len(prefix) :]
        if rest.endswith("_mean_over_s"):
            return f"{rest[: -len('_mean_over_s')]} (mean over s)"
        return rest
    return algo_name


def parse_tau_s_from_algo_name(algo_name: str) -> tuple[float, int] | None:
    """opts_ttpo_continuous_action_tau0.4_s6 -> (0.4, 6)；跨 s 平均曲线 -> (0.4, -1)。"""
    prefix = "opts_ttpo_continuous_action_"
    if not algo_name.startswith(prefix):
        return None
    rest = algo_name[len(prefix) :]
    if rest.endswith("_mean_over_s"):
        inner = rest[: -len("_mean_over_s")]
        m_only_tau = re.match(r"^tau([\d.]+)$", inner)
        if not m_only_tau:
            return None
        return float(m_only_tau.group(1)), -1
    m = TAU_S_FROM_TAG_RE.match(rest)
    if not m:
        return None
    return float(m.group(1)), int(m.group(2))


def _align_mean_curves(
    series: list[dict],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """对多条 (steps, mean, std) 按索引截断到最短长度，返回 steps、逐点 mean 与 std（跨曲线）。"""
    if not series:
        return np.array([]), np.array([]), np.array([])
    lengths = [len(s["mean"]) for s in series]
    m = min(lengths)
    if m == 0:
        return np.array([]), np.array([]), np.array([])
    steps = np.array(series[0]["steps"][:m], dtype=float)
    means = np.stack([np.array(s["mean"][:m], dtype=float) for s in series], axis=0)
    cross_mean = np.mean(means, axis=0)
    cross_std = np.std(means, axis=0, ddof=0)
    return steps, cross_mean, cross_std


def build_cross_s_average_by_tau(
    all_tasks_data: dict[str, dict[tuple[str, str], dict]],
    date: str,
) -> dict[str, dict[tuple[str, str], dict]]:
    """
    每个任务、每个 τ：对该 τ 下所有 s 的（已跨 seed 聚合的）mean 曲线逐点对齐后取平均。
    阴影为跨 s 的 std（不同搜索次数之间的离散程度）。
    """
    out: dict[str, dict[tuple[str, str], dict]] = {t: {} for t in TARGET_TASKS}

    for task_name in TARGET_TASKS:
        by_tau: dict[float, list[dict]] = defaultdict(list)
        for algo_key, data in all_tasks_data[task_name].items():
            _algo, d = algo_key
            if d != date:
                continue
            ts = parse_tau_s_from_algo_name(_algo)
            if ts is None:
                continue
            tau, _s = ts
            by_tau[tau].append(data)

        for tau in sorted(by_tau.keys()):
            steps, cross_mean, cross_std = _align_mean_curves(by_tau[tau])
            if len(cross_mean) == 0:
                continue
            # 与 parse_tau_s_from_algo_name 中 _mean_over_s 分支一致
            tag = f"tau{tau:g}_mean_over_s"
            synth_algo = f"opts_ttpo_continuous_action_{tag}"
            out[task_name][(synth_algo, date)] = {
                "steps": list(steps),
                "mean": list(cross_mean),
                "std": list(cross_std),
            }
    return out


def score_smoothed_tail(
    mean_list: list[float],
    smooth_window: int,
    tail_frac: float = 0.2,
) -> float:
    """平滑后对序列最后 tail_frac 比例取均值，作为单任务上的标量得分。"""
    if not mean_list:
        return float("-inf")
    sm = smooth_data(mean_list, smooth_window)
    arr = np.asarray(sm, dtype=float)
    n = len(arr)
    k = max(1, int(np.ceil(n * tail_frac)))
    return float(np.mean(arr[-k:]))


def mean_score_across_tasks(
    per_task_data: dict[str, dict],
    smooth_window: int,
    tail_frac: float = 0.2,
) -> float:
    scores = []
    for task_name in TARGET_TASKS:
        if task_name not in per_task_data:
            continue
        d = per_task_data[task_name]
        scores.append(
            score_smoothed_tail(d["mean"], smooth_window, tail_frac=tail_frac)
        )
    if not scores:
        return float("-inf")
    return float(np.mean(scores))


def pick_best_tau(
    tau_avg_data: dict[str, dict[tuple[str, str], dict]],
    date: str,
    smooth_window: int,
) -> tuple[float, dict[float, float]]:
    """返回最优 τ 及各 τ 的跨任务平均尾部得分。"""
    per_tau_task_payload: dict[float, dict[str, dict]] = defaultdict(dict)
    for task_name in TARGET_TASKS:
        for algo_key, data in tau_avg_data[task_name].items():
            _algo, d = algo_key
            if d != date:
                continue
            ts = parse_tau_s_from_algo_name(_algo)
            if ts is None or ts[1] != -1:
                continue
            tau, _s = ts
            per_tau_task_payload[tau][task_name] = data

    tau_scores: dict[float, float] = {}
    for tau, tasks_map in per_tau_task_payload.items():
        tau_scores[tau] = mean_score_across_tasks(tasks_map, smooth_window)

    if not tau_scores:
        return float("nan"), {}
    best_tau = max(tau_scores.keys(), key=lambda t: tau_scores[t])
    return best_tau, tau_scores


def filter_by_tau(
    all_tasks_data: dict[str, dict[tuple[str, str], dict]],
    date: str,
    tau: float,
) -> dict[str, dict[tuple[str, str], dict]]:
    """只保留指定 τ 下的 (tau,s) 运行（用于第二步画 s 消融）。"""
    out: dict[str, dict[tuple[str, str], dict]] = {t: {} for t in TARGET_TASKS}
    for task_name in TARGET_TASKS:
        for algo_key, data in all_tasks_data[task_name].items():
            _algo, d = algo_key
            if d != date:
                continue
            ts = parse_tau_s_from_algo_name(_algo)
            if ts is None:
                continue
            t_val, _s = ts
            if abs(t_val - tau) > 1e-9:
                continue
            out[task_name][algo_key] = data
    return out


def pick_best_s_at_tau(
    filtered: dict[str, dict[tuple[str, str], dict]],
    smooth_window: int,
) -> tuple[int, dict[int, float]]:
    """在已按 τ 过滤的数据上，按跨任务平均尾部得分选最优 s。"""
    s_scores: dict[int, float] = {}
    per_s_tasks: dict[int, dict[str, dict]] = defaultdict(dict)

    for task_name in TARGET_TASKS:
        for algo_key, data in filtered[task_name].items():
            _algo, _d = algo_key
            ts = parse_tau_s_from_algo_name(_algo)
            if ts is None:
                continue
            _tau, s_val = ts
            per_s_tasks[s_val][task_name] = data

    for s_val, tasks_map in per_s_tasks.items():
        s_scores[s_val] = mean_score_across_tasks(tasks_map, smooth_window)

    if not s_scores:
        return -1, {}
    best_s = max(s_scores.keys(), key=lambda s: s_scores[s])
    return best_s, s_scores


def _distinct_colors(algo_keys: list[tuple[str, str]]) -> dict[tuple[str, str], str]:
    """使用 plot_mujoco 同款配色，按曲线顺序循环分配。"""
    pal = ABLATION_LINE_COLORS
    return {k: pal[i % len(pal)] for i, k in enumerate(algo_keys)}


def plot_ablation_all_tasks(
    all_tasks_data: dict[str, dict[tuple[str, str], dict]],
    output_path: Path,
    smooth_window: int = DEFAULT_ABLATION_SMOOTH_WINDOW,
    show_std: bool = True,
) -> None:
    all_algos: set[tuple[str, str]] = set()
    for task_data in all_tasks_data.values():
        for algo_key in task_data.keys():
            all_algos.add(algo_key)
    sorted_keys = sorted(all_algos, key=lambda x: (x[0], x[1]))
    algo_colors = _distinct_colors(sorted_keys)

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    for idx, task_name in enumerate(TARGET_TASKS):
        ax = axes[idx]
        if task_name not in all_tasks_data or not all_tasks_data[task_name]:
            ax.set_title(task_name.replace("-v4", "-v1"), fontsize=12)
            ax.set_xlabel("Timesteps", fontsize=10)
            if idx == 0:
                ax.set_ylabel("Mean Episodic Return", fontsize=10)
            continue

        task_data = all_tasks_data[task_name]
        for algo_key, data in sorted(task_data.items(), key=lambda x: x[0][0]):
            algo_name, date = algo_key
            color = algo_colors[algo_key]
            display_name = _display_name_for_ablation(algo_name)

            steps = np.array(data["steps"])
            mean_values = smooth_data(data["mean"], smooth_window)
            if show_std:
                std_values = smooth_data(data["std"], smooth_window)
                ax.fill_between(
                    steps[: len(mean_values)],
                    mean_values - std_values,
                    mean_values + std_values,
                    color=color,
                    alpha=0.2,
                )
            ax.plot(
                steps[: len(mean_values)],
                mean_values,
                color=color,
                label=display_name,
                linewidth=1.5,
            )

        ax.set_title(task_name, fontsize=12)
        ax.set_xlabel("Timesteps", fontsize=10)
        if idx == 0:
            ax.set_ylabel("Mean Episodic Return", fontsize=10)

        if task_data:
            all_steps: list[float] = []
            for data in task_data.values():
                all_steps.extend(data["steps"])
            if all_steps:
                max_step = max(all_steps)
                max_step_rounded = int(round(max_step / 1_000_000) * 1_000_000)
                if max_step_rounded == 0:
                    max_step_rounded = 1_000_000
                ax.set_xlim(0, max_step_rounded)
                ax.set_xticks([0, max_step_rounded])
                ax.set_xticklabels(["0", f"{max_step_rounded // 1_000_000}M"])

        ax.grid(True, alpha=0.3, which="major")

    handles, labels = [], []
    for algo_key in sorted(algo_colors.keys()):
        algo_name, _date = algo_key
        display_name = _display_name_for_ablation(algo_name)
        handles.append(plt.Line2D([0], [0], color=algo_colors[algo_key], linewidth=2))
        labels.append(display_name)

    ncol = min(len(handles), 4) if handles else 1
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=ncol,
        fontsize=11,
        bbox_to_anchor=(0.5, 1.08),
        frameon=True,
    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Ablation plot saved to: {output_path}")
    plt.close()


def _parse_seed_filters(argv: list[str]) -> tuple[list[str], set[int] | None]:
    """支持 --seeds 1,2,3，返回剩余 argv 与 seed 集合。"""
    seed_filters: set[int] | None = None
    out: list[str] = []
    i = 0
    while i < len(argv):
        if argv[i] == "--seeds" and i + 1 < len(argv):
            seed_filters = {int(s.strip()) for s in argv[i + 1].split(",")}
            i += 2
            continue
        out.append(argv[i])
        i += 1
    return out, seed_filters


def main() -> None:
    raw_argv = list(sys.argv[1:])
    hierarchical = "--hierarchical" in raw_argv
    raw_argv = [a for a in raw_argv if a != "--hierarchical"]

    rest, seed_filters = _parse_seed_filters(raw_argv)

    if len(rest) < 2:
        print(
            "用法: python visual/plot_mujoco_ablation.py <results_subdir> <YYYYMMDD> "
            "[tau0.0_s2,tau0.4_s6] [--hierarchical] [--seeds 1,2,3]\n"
            "  --hierarchical先对每个 τ 跨所有 s 逐点平均再比 τ；打印推荐 τ/s，并另存两张图。"
        )
        sys.exit(1)

    results_subdir = Path(rest[0]).resolve()
    date = rest[1]
    if not re.fullmatch(r"\d{8}", date):
        print(f"日期应为 8 位数字，例如 20260410，收到: {date}")
        sys.exit(1)

    hyper_tags: list[str] | None = None
    if len(rest) >= 3:
        hyper_tags = [t.strip() for t in rest[2].split(",") if t.strip()]

    run_dirs = discover_ablation_run_dirs(results_subdir, date, hyper_tags)
    if not run_dirs:
        tag_msg = f"（标签过滤: {hyper_tags}）" if hyper_tags else ""
        print(
            f"未找到匹配的运行目录: {results_subdir}，日期 {date}{tag_msg}\n"
            f"期望目录名形如: opts_ttpo_continuous_action_tau0.4_s6_{date}"
        )
        sys.exit(1)

    print(f"将绘制 {len(run_dirs)} 组配置: {[d.name for d in run_dirs]}")

    all_tasks_data = collect_aggregated_per_task(run_dirs, seed_filters)
    if not any(all_tasks_data[t] for t in TARGET_TASKS):
        print("未读到任何任务的 JSON 数据，请检查结果目录。")
        sys.exit(1)

    script_dir = Path(__file__).resolve().parent
    sw = DEFAULT_ABLATION_SMOOTH_WINDOW

    if hierarchical:
        tau_avg_data = build_cross_s_average_by_tau(all_tasks_data, date)
        if not any(tau_avg_data[t] for t in TARGET_TASKS):
            print("分层模式：无法构造跨 s 平均曲线（请检查各 τ 下是否至少有一条运行）。")
            sys.exit(1)

        path_mean_s = script_dir / f"all_tasks_mujoco_ablation_{date}_mean_over_s.png"
        plot_ablation_all_tasks(tau_avg_data, path_mean_s, smooth_window=sw)

        best_tau, tau_scores = pick_best_tau(tau_avg_data, date, sw)
        print("\n=== 按 τ：对每个 τ 跨所有 s 逐点平均后的曲线 ===")
        print("各 τ 得分（各任务上平滑后最后 20% 时间步均值，再对任务取平均）：")
        for t in sorted(tau_scores.keys()):
            print(f"  tau={t:g}  score={tau_scores[t]:.4f}")
        print(f"\n推荐 τ* = {best_tau:g}")

        filtered = filter_by_tau(all_tasks_data, date, best_tau)
        if not any(filtered[t] for t in TARGET_TASKS):
            print(f"在 τ={best_tau:g} 下未找到任何 (tau,s) 运行目录。")
            sys.exit(1)

        best_s, s_scores = pick_best_s_at_tau(filtered, sw)
        print(f"\n=== 在 τ*={best_tau:g} 下按 s 比较（同上得分）===")
        for s in sorted(s_scores.keys()):
            print(f"  s={s}  score={s_scores[s]:.4f}")
        print(f"\n推荐 s* = {best_s}（在 τ* 下）")

        path_s = script_dir / f"all_tasks_mujoco_ablation_{date}_best_tau_s.png"
        plot_ablation_all_tasks(
            filtered, path_s, smooth_window=sw, show_std=False
        )
    else:
        output_path = script_dir / f"all_tasks_mujoco_ablation_{date}.png"
        plot_ablation_all_tasks(all_tasks_data, output_path, smooth_window=sw)


if __name__ == "__main__":
    main()
