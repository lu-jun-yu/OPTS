"""
绘制57个 Atari 任务下不同算法的 episodic_return 收敛曲线
从 ../cleanrl/results/ 目录中读取数据文件
目录结构：results/{num_envs}_{num_steps}/{algo_name}_{date}/{env_id}_{seed}.json
3个随机种子，不聚合 mean/std，而是以相同颜色画出每条种子曲线
布局：10行6列，共57张子图（最后一行3张空白）
"""
import os
import json
import re
import sys
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy.ndimage import uniform_filter1d


# 57个 Atari 任务（按字母序，与 run_all_baselines_atari.sh 一致）
TARGET_TASKS = [
    "AlienNoFrameskip-v4",
    "AmidarNoFrameskip-v4",
    "AssaultNoFrameskip-v4",
    "AsterixNoFrameskip-v4",
    "AsteroidsNoFrameskip-v4",
    "AtlantisNoFrameskip-v4",
    "BankHeistNoFrameskip-v4",
    "BattleZoneNoFrameskip-v4",
    "BeamRiderNoFrameskip-v4",
    "BerzerkNoFrameskip-v4",
    "BowlingNoFrameskip-v4",
    "BoxingNoFrameskip-v4",
    "BreakoutNoFrameskip-v4",
    "CentipedeNoFrameskip-v4",
    "ChopperCommandNoFrameskip-v4",
    "CrazyClimberNoFrameskip-v4",
    "DefenderNoFrameskip-v4",
    "DemonAttackNoFrameskip-v4",
    "DoubleDunkNoFrameskip-v4",
    "EnduroNoFrameskip-v4",
    "FishingDerbyNoFrameskip-v4",
    "FreewayNoFrameskip-v4",
    "FrostbiteNoFrameskip-v4",
    "GopherNoFrameskip-v4",
    "GravitarNoFrameskip-v4",
    "HeroNoFrameskip-v4",
    "IceHockeyNoFrameskip-v4",
    "JamesbondNoFrameskip-v4",
    "KangarooNoFrameskip-v4",
    "KrullNoFrameskip-v4",
    "KungFuMasterNoFrameskip-v4",
    "MontezumaRevengeNoFrameskip-v4",
    "MsPacmanNoFrameskip-v4",
    "NameThisGameNoFrameskip-v4",
    "PhoenixNoFrameskip-v4",
    "PitfallNoFrameskip-v4",
    "PongNoFrameskip-v4",
    "PrivateEyeNoFrameskip-v4",
    "QbertNoFrameskip-v4",
    "RiverraidNoFrameskip-v4",
    "RoadRunnerNoFrameskip-v4",
    "RobotankNoFrameskip-v4",
    "SeaquestNoFrameskip-v4",
    "SkiingNoFrameskip-v4",
    "SolarisNoFrameskip-v4",
    "SpaceInvadersNoFrameskip-v4",
    "StarGunnerNoFrameskip-v4",
    "SurroundNoFrameskip-v4",
    "TennisNoFrameskip-v4",
    "TimePilotNoFrameskip-v4",
    "TutankhamNoFrameskip-v4",
    "UpNDownNoFrameskip-v4",
    "VentureNoFrameskip-v4",
    "VideoPinballNoFrameskip-v4",
    "WizardOfWorNoFrameskip-v4",
    "YarsRevengeNoFrameskip-v4",
    "ZaxxonNoFrameskip-v4",
]

NCOLS = 6
NROWS = 10  # ceil(57/6) = 10

# 冷暖色穿插的调色板
COOL_COLORS = [
    "#1f77b4",  # 蓝
    "#2ca02c",  # 绿
    "#9467bd",  # 紫
    "#17becf",  # 青
]
WARM_COLORS = [
    "#d62728",  # 红
    "#ff7f0e",  # 橙
    "#8c564b",  # 棕
    "#e377c2",  # 粉
]


def get_alternating_colors(n):
    colors = []
    for i in range(n):
        if i % 2 == 0:
            colors.append(COOL_COLORS[(i // 2) % len(COOL_COLORS)])
        else:
            colors.append(WARM_COLORS[(i // 2) % len(WARM_COLORS)])
    return colors


def assign_algo_colors(algo_keys):
    sorted_keys = sorted(algo_keys)
    colors = get_alternating_colors(len(sorted_keys))
    return {algo_key: colors[i] for i, algo_key in enumerate(sorted_keys)}


def smooth_data(data, window_size=5):
    if len(data) < window_size:
        return np.array(data)
    return uniform_filter1d(np.array(data), size=window_size, mode='nearest')


def load_episodic_returns(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        step_values = []
        mean_return_values = []

        for item in data:
            if isinstance(item, dict) and 'mean_return' in item and 'step' in item:
                step_values.append(float(item['step']))
                mean_return_values.append(float(item['mean_return']))

        return step_values, mean_return_values
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return [], []


def parse_result_path(filepath):
    """
    解析结果文件路径
    目录结构：results/{num_envs}_{num_steps}/{algo_name}_{date}/{env_id}_{seed}.json
    """
    path = Path(filepath)
    filename = path.stem  # e.g., "BreakoutNoFrameskip-v4_1"
    algo_dir = path.parent.name  # e.g., "ppo_atari_20260302"

    # Parse seed from filename: {env_id}_{seed}
    seed_match = re.search(r'_(\d+)$', filename)
    if not seed_match:
        return None
    seed = int(seed_match.group(1))
    task = filename[:seed_match.start()]

    # Parse algo_name and date from directory name: {algo_name}_{date}
    date_match = re.search(r'_(\d{8})$', algo_dir)
    if not date_match:
        return None
    date = date_match.group(1)
    algo_name = algo_dir[:date_match.start()]

    return (task, algo_name, date, seed)


USE_SHORT_NAME = False


def get_display_name(algo_name, date=None):
    if algo_name == "ppo_atari":
        return "PPO"
    if algo_name == "a2c_atari":
        return "A2C"
    if algo_name.startswith("opts_ttpo") and USE_SHORT_NAME:
        return "OPTS-TTPO"
    # 默认显示完整名称（包含日期）
    if date:
        return f"{algo_name}_{date}"
    return algo_name


def load_algo_filters_from_config(task_name, config_filename="algo_select_atari.json"):
    try:
        script_dir = Path(__file__).resolve().parent
        config_path = script_dir / config_filename
        if not config_path.exists():
            return None
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        algo_list = cfg.get(task_name)
        if isinstance(algo_list, list):
            return [str(a) for a in algo_list]
    except Exception as e:
        print(f"Warning: failed to load algo filters from config '{config_filename}': {e}")
    return None


def plot_all_tasks(results_dir="../cleanrl/results", output_dir="./visual",
                   algo_filters=None, smooth_window=200, seed_filters=None):
    """
    绘制57个 Atari 任务的收敛曲线（10行6列布局）
    每个算法的不同种子以相同颜色画出（不聚合 mean/std）
    """
    # 递归查找所有 JSON 文件
    files = list(Path(results_dir).rglob("*.json"))
    if not files:
        print(f"No results files found in {results_dir}")
        return

    # 收集数据: {task: {algo_key: {seed: (steps, returns)}}}
    all_data = defaultdict(lambda: defaultdict(dict))

    for filepath in files:
        parsed = parse_result_path(filepath)
        if parsed is None:
            continue

        task, algo_name, date, seed = parsed
        if task not in TARGET_TASKS:
            continue

        if algo_filters is not None:
            algo_id = algo_name
            algo_id_with_date = f"{algo_name}_{date}"
            if (algo_id not in algo_filters) and (algo_id_with_date not in algo_filters):
                continue

        if seed_filters is not None and seed not in seed_filters:
            continue

        algo_key = (algo_name, date)
        step_values, mean_return_values = load_episodic_returns(filepath)
        if mean_return_values:
            all_data[task][algo_key][seed] = (step_values, mean_return_values)

    if not all_data:
        print("No data found for any Atari task")
        return

    # 收集所有出现的算法，统一分配颜色
    all_algos = set()
    for task_data in all_data.values():
        for algo_key in task_data.keys():
            all_algos.add(algo_key)

    algo_colors = assign_algo_colors(all_algos)

    # 创建 10行6列 子图
    fig, axes = plt.subplots(NROWS, NCOLS, figsize=(NCOLS * 4, NROWS * 3))

    for idx, task_name in enumerate(TARGET_TASKS):
        row, col = divmod(idx, NCOLS)
        ax = axes[row][col]

        # 简短标题：去掉 NoFrameskip-v4 后缀
        short_name = task_name.replace("NoFrameskip-v4", "")
        ax.set_title(short_name, fontsize=10)

        if task_name not in all_data:
            ax.text(0.5, 0.5, "No data", ha='center', va='center',
                    transform=ax.transAxes, fontsize=9, color='gray')
            ax.grid(True, alpha=0.3)
            continue

        task_data = all_data[task_name]

        for algo_key in sorted(task_data.keys()):
            seed_data = task_data[algo_key]
            algo_name, date = algo_key
            color = algo_colors[algo_key]
            display_name = get_display_name(algo_name, date)

            # 各 seed 的 step 条数不一致时，统一到最短长度再可视化
            min_len = min(
                (len(steps) for steps, _ in seed_data.values()),
                default=0,
            )

            for i, (seed, (steps, returns)) in enumerate(sorted(seed_data.items())):
                steps_trunc = steps[:min_len]
                returns_trunc = returns[:min_len]
                steps_arr = np.array(steps_trunc)
                smoothed = smooth_data(returns_trunc, smooth_window)
                # 只在第一条种子曲线加 label（避免图例重复）
                label = display_name if i == 0 else None
                ax.plot(steps_arr[:len(smoothed)], smoothed,
                        color=color, label=label, linewidth=0.8, alpha=0.8)

        # x轴：只显示0和终点
        all_steps = []
        for seed_data in task_data.values():
            for steps, _ in seed_data.values():
                all_steps.extend(steps)
        if all_steps:
            max_step = max(all_steps)
            max_step_rounded = int(round(max_step / 1000000) * 1000000)
            if max_step_rounded == 0:
                max_step_rounded = 1000000
            ax.set_xlim(0, max_step_rounded)
            ax.set_xticks([0, max_step_rounded])
            ax.set_xticklabels(['0', f'{max_step_rounded:.0e}'], fontsize=7)

        ax.tick_params(axis='y', labelsize=7)
        ax.grid(True, alpha=0.3)

    # 隐藏多余的子图（57个任务，最后3格为空）
    for idx in range(len(TARGET_TASKS), NROWS * NCOLS):
        row, col = divmod(idx, NCOLS)
        axes[row][col].axis('off')

    # 在最后一个空位放统一图例
    legend_ax = axes[NROWS - 1][NCOLS - 1]
    legend_ax.axis('off')
    handles, labels = [], []
    for algo_key in sorted(algo_colors.keys()):
        algo_name, date = algo_key
        display_name = get_display_name(algo_name, date)
        handles.append(plt.Line2D([0], [0], color=algo_colors[algo_key], linewidth=2))
        labels.append(display_name)
    legend_ax.legend(handles, labels, loc='center', fontsize=9, frameon=True)

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "all_tasks_atari.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Atari convergence curves saved to: {output_path}")
    plt.close()


def main():
    """
    用法：
        python plot_atari.py [--short-name] [--seeds 1,2,3] [results_dir] [algo1 algo2 ...]

        --short-name        OPTS_TTPO 使用简称 "OPTS-TTPO"（默认显示全称）
        --seeds 1,2,3       只可视化指定随机种子的数据（逗号分隔，默认全部）
    """
    global USE_SHORT_NAME

    seed_filters = None
    raw_args = sys.argv[1:]
    filtered_args = []
    i = 0
    while i < len(raw_args):
        if raw_args[i] == "--short-name":
            USE_SHORT_NAME = True
        elif raw_args[i] == "--seeds":
            if i + 1 < len(raw_args):
                seed_filters = set(int(s) for s in raw_args[i + 1].split(","))
                i += 1
            else:
                print("Error: --seeds requires an argument (e.g., --seeds 1,2,3)")
                return
        else:
            filtered_args.append(raw_args[i])
        i += 1

    results_dir = filtered_args[0] if filtered_args else "../cleanrl/results"
    algo_filters = filtered_args[1:] if len(filtered_args) > 1 else None

    if os.path.exists(results_dir):
        seed_info = f" (seeds: {sorted(seed_filters)})" if seed_filters else ""
        print(f"Plotting Atari convergence curves for {len(TARGET_TASKS)} tasks{seed_info}...")
        script_dir = str(Path(__file__).resolve().parent)
        plot_all_tasks(results_dir, output_dir=script_dir, algo_filters=algo_filters, seed_filters=seed_filters)
    else:
        print(f"Results directory {results_dir} does not exist")


if __name__ == "__main__":
    main()
