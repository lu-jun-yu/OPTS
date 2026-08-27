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
from matplotlib.ticker import FuncFormatter, MaxNLocator
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
    "ALE_Surround-v5",
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

COLOR_PPO = "#4c72b0"
COLOR_A2C = "#2ca02c"
OPTS_TTPO_COLORS = [
    "#c44e52",  # red
    "#ff7f0e",  # orange
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#17becf",  # cyan
    "#bcbd22",  # olive
    "#7f7f7f",  # gray
]
EXTRA_ALGO_COLORS = ["#4c72b0", "#2ca02c", "#17becf", "#8c564b", "#e377c2", "#bcbd22", "#7f7f7f"]

TEXT_COLOR = "#263238"
MUTED_TEXT_COLOR = "#59636b"
SPINE_COLOR = "#a5adb3"
GRID_COLOR = "#d9e1e8"


def build_algo_colors(algo_keys):
    sorted_keys = sorted(algo_keys)
    colors = {}

    opts_keys = [k for k in sorted_keys if k[0].startswith("opts_ttpo")]
    opts_color_map = {
        k: OPTS_TTPO_COLORS[i % len(OPTS_TTPO_COLORS)]
        for i, k in enumerate(opts_keys)
    }

    used_colors = set()

    for algo_key in sorted_keys:
        algo_name, _ = algo_key
        if algo_name.startswith("ppo_atari"):
            colors[algo_key] = COLOR_PPO
        elif algo_name == "a2c_atari":
            colors[algo_key] = COLOR_A2C
        elif algo_name.startswith("opts_ttpo"):
            colors[algo_key] = opts_color_map[algo_key]
        else:
            continue
        used_colors.add(colors[algo_key])

    extra_i = 0
    for algo_key in sorted_keys:
        if algo_key in colors:
            continue

        while EXTRA_ALGO_COLORS[extra_i % len(EXTRA_ALGO_COLORS)] in used_colors:
            extra_i += 1

        colors[algo_key] = EXTRA_ALGO_COLORS[extra_i % len(EXTRA_ALGO_COLORS)]
        used_colors.add(colors[algo_key])
        extra_i += 1

    return colors


def get_curve_zorder(algo_name):
    return 3 if algo_name.startswith("opts_ttpo") else 2


def get_curve_linestyle(algo_name):
    return "--" if algo_name.startswith("ppo_atari") else "-"


def get_algo_sort_key(algo_key):
    algo_name, date = algo_key
    if algo_name.startswith("ppo_atari"):
        return (0, algo_name, date)
    if algo_name.startswith("opts_ttpo"):
        return (1, algo_name, date)
    return (2, algo_name, date)


def format_return_tick(value, _position):
    abs_value = abs(value)
    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:g}M"
    if abs_value >= 1_000:
        scaled = value / 1_000
        return f"{scaled:.1f}k" if abs(scaled) < 10 else f"{scaled:.0f}k"
    if abs_value >= 10:
        return f"{value:.0f}"
    if abs_value >= 1:
        return f"{value:.1f}".rstrip("0").rstrip(".")
    return f"{value:.1f}"


def style_axis(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.45, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(SPINE_COLOR)
        ax.spines[side].set_linewidth(0.45)
    ax.tick_params(
        axis="both",
        colors=MUTED_TEXT_COLOR,
        labelsize=4.5,
        length=1.6,
        width=0.45,
        pad=1.0,
    )
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.yaxis.set_major_formatter(FuncFormatter(format_return_tick))


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
    if algo_name.startswith("ppo_atari"):
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
                   algo_filters=None, smooth_window=1000, seed_filters=None,
                   output_path=None, png_preview=False):
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

    algo_colors = build_algo_colors(all_algos)

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 5.2,
        "text.color": TEXT_COLOR,
        "axes.labelcolor": TEXT_COLOR,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    # 按论文整页附录图的实际尺寸绘制，避免大画布在 LaTeX 中被过度缩小。
    fig, axes = plt.subplots(NROWS, NCOLS, figsize=(7.4, 9.5), squeeze=False)
    fig.subplots_adjust(
        left=0.070,
        right=0.992,
        bottom=0.052,
        top=0.948,
        wspace=0.26,
        hspace=0.52,
    )

    for idx, task_name in enumerate(TARGET_TASKS):
        row, col = divmod(idx, NCOLS)
        ax = axes[row][col]
        style_axis(ax)

        # 简短标题：去掉 NoFrameskip-v4 后缀
        short_name = task_name.replace("NoFrameskip-v4", "")
        if task_name == "ALE_Surround-v5":
            short_name = "Surround"
        ax.set_title(short_name, fontsize=5.8, fontweight="semibold", pad=1.5)

        if task_name not in all_data:
            ax.text(0.5, 0.5, "No data", ha='center', va='center',
                    transform=ax.transAxes, fontsize=5.0, color=MUTED_TEXT_COLOR)
            continue

        task_data = all_data[task_name]

        for algo_key in sorted(task_data.keys(), key=get_algo_sort_key):
            seed_data = task_data[algo_key]
            algo_name, date = algo_key
            color = algo_colors[algo_key]
            display_name = get_display_name(algo_name, date)

            for i, (seed, (steps, returns)) in enumerate(sorted(seed_data.items())):
                steps_arr = np.array(steps)
                smoothed = smooth_data(returns, smooth_window)
                # 只在第一条种子曲线加 label（避免图例重复）
                label = display_name if i == 0 else None
                ax.plot(steps_arr[:len(smoothed)], smoothed,
                        color=color, linestyle=get_curve_linestyle(algo_name),
                        label=label, linewidth=0.60, alpha=0.8,
                        zorder=get_curve_zorder(algo_name))

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
            ax.set_xticklabels(
                ["0", f"{max_step_rounded // 1_000_000}M"],
                fontsize=4.5,
            )

    # 隐藏多余的子图（57个任务，最后3格为空）
    for idx in range(len(TARGET_TASKS), NROWS * NCOLS):
        row, col = divmod(idx, NCOLS)
        axes[row][col].axis('off')

    handles, labels = [], []
    seen_labels = set()
    for algo_key in sorted(algo_colors.keys(), key=get_algo_sort_key):
        algo_name, date = algo_key
        display_name = get_display_name(algo_name, date)
        if display_name in seen_labels:
            continue
        seen_labels.add(display_name)
        handles.append(plt.Line2D(
            [0], [0],
            color=algo_colors[algo_key],
            linestyle=get_curve_linestyle(algo_name),
            linewidth=1.3,
        ))
        labels.append(display_name)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.53, 0.995),
        ncol=max(1, len(labels)),
        fontsize=6.2,
        frameon=False,
        handlelength=2.3,
        columnspacing=1.5,
    )
    fig.supxlabel("Environment steps", fontsize=6.2, x=0.53, y=0.010)
    fig.supylabel("Mean return", fontsize=6.2, x=0.018, y=0.50)

    if output_path is None:
        output_path = Path(output_dir) / "all_tasks_atari.pdf"
    else:
        output_path = Path(output_path)
    if not output_path.suffix:
        output_path = output_path.with_suffix(".pdf")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_kwargs = {"bbox_inches": "tight", "pad_inches": 0.02}
    if output_path.suffix.lower() == ".png":
        save_kwargs["dpi"] = 300
    fig.savefig(output_path, **save_kwargs)
    print(f"Atari convergence curves saved to: {output_path}")

    if png_preview and output_path.suffix.lower() != ".png":
        preview_path = output_path.with_suffix(".png")
        fig.savefig(preview_path, dpi=220, bbox_inches="tight", pad_inches=0.02)
        print(f"Atari PNG preview saved to: {preview_path}")

    plt.close()


def main():
    """
    用法：
        python plot_atari.py [--short-name] [--seeds 1,2,3]
            [--output path.pdf] [--png-preview]
            [results_dir] [algo1 algo2 ...]

        --short-name        OPTS_TTPO 使用简称 "OPTS-TTPO"（默认显示全称）
        --seeds 1,2,3       只可视化指定随机种子的数据（逗号分隔，默认全部）
        --output path.pdf   指定主输出路径（默认 visual/all_tasks_atari.pdf）
        --png-preview       同时在主输出旁生成同名 PNG 预览
    """
    global USE_SHORT_NAME

    seed_filters = None
    output_path = None
    png_preview = False
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
        elif raw_args[i] == "--output":
            if i + 1 < len(raw_args):
                output_path = raw_args[i + 1]
                i += 1
            else:
                print("Error: --output requires a path (e.g., figure.pdf)")
                return
        elif raw_args[i] == "--png-preview":
            png_preview = True
        else:
            filtered_args.append(raw_args[i])
        i += 1

    results_dir = filtered_args[0] if filtered_args else "../cleanrl/results"
    algo_filters = filtered_args[1:] if len(filtered_args) > 1 else None

    if os.path.exists(results_dir):
        seed_info = f" (seeds: {sorted(seed_filters)})" if seed_filters else ""
        print(f"Plotting Atari convergence curves for {len(TARGET_TASKS)} tasks{seed_info}...")
        script_dir = str(Path(__file__).resolve().parent)
        plot_all_tasks(
            results_dir,
            output_dir=script_dir,
            algo_filters=algo_filters,
            seed_filters=seed_filters,
            output_path=output_path,
            png_preview=png_preview,
        )
    else:
        print(f"Results directory {results_dir} does not exist")


if __name__ == "__main__":
    main()
