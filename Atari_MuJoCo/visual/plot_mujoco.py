"""
绘制相同 task 下不同算法的 episodic_return 收敛曲线
从 ./results/ 目录中读取数据文件
新格式目录结构：results/{num_envs}_{num_steps}/{algo_name}_{date}/{env_id}_{seed}.json
对于相同任务和相同算法，聚合 seed 1-10 的结果，计算同一行的均值
支持平滑曲线和 max/min_return 阴影区域绘制
"""
import os
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy.ndimage import uniform_filter1d


# 要绘制的5个任务
TARGET_TASKS = ["Hopper-v4", "Walker2d-v4", "HalfCheetah-v4", "Ant-v4", "Humanoid-v4"]

# 指定任务的平滑窗口大小（其他任务使用默认值）
TASK_SMOOTH_WINDOWS = {
    "Ant-v4": 9,
    "Humanoid-v4": 9,
}

# PPO 蓝、RPO 绿、OPTS-TTPO 红；其余算法用补充色
COLOR_PPO = "#1f77b4"
COLOR_RPO = "#2ca02c"
COLOR_OPTS_TTPO = "#d62728"
EXTRA_ALGO_COLORS = ["#9467bd", "#ff7f0e", "#17becf", "#8c564b", "#e377c2"]


def build_algo_colors(algo_keys):
    """
    为算法分配颜色：PPO 蓝、RPO 绿、OPTS-TTPO（opts_ttpo*）红；
    其他算法按排序稳定地依次使用 EXTRA_ALGO_COLORS。
    """
    sorted_keys = sorted(algo_keys)
    colors = {}
    extra_i = 0
    for algo_key in sorted_keys:
        algo_name, _ = algo_key
        if algo_name == "ppo_continuous_action":
            colors[algo_key] = COLOR_PPO
        elif algo_name == "rpo_continuous_action":
            colors[algo_key] = COLOR_RPO
        elif algo_name.startswith("opts_ttpo"):
            colors[algo_key] = COLOR_OPTS_TTPO
        else:
            colors[algo_key] = EXTRA_ALGO_COLORS[extra_i % len(EXTRA_ALGO_COLORS)]
            extra_i += 1
    return colors


def get_task_smooth_window(task_name, default_window):
    """
    获取指定任务的平滑窗口大小。
    """
    return TASK_SMOOTH_WINDOWS.get(task_name, default_window)


def smooth_data(data, window_size=5):
    """
    使用移动平均对数据进行平滑
    
    Args:
        data: 输入数据列表或数组
        window_size: 平滑窗口大小
    
    Returns:
        平滑后的数据数组
    """
    if len(data) < window_size:
        return np.array(data)
    return uniform_filter1d(np.array(data), size=window_size, mode='nearest')


def load_episodic_returns(filepath):
    """
    从 JSON 文件中读取 episodic_return 值

    Args:
        filepath: JSON文件路径

    Returns:
        (step值列表, mean_return值列表, max_return值列表, min_return值列表) 元组
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        step_values = []
        mean_return_values = []
        max_return_values = []
        min_return_values = []

        for item in data:
            if isinstance(item, dict) and 'mean_return' in item:
                if 'step' in item:
                    step_values.append(float(item['step']))
                else:
                    continue

                mean_return_values.append(float(item['mean_return']))

                if 'max_return' in item:
                    max_return_values.append(float(item['max_return']))
                else:
                    max_return_values.append(float(item['mean_return']))

                if 'min_return' in item:
                    min_return_values.append(float(item['min_return']))
                else:
                    min_return_values.append(float(item['mean_return']))

        return step_values, mean_return_values, max_return_values, min_return_values
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return [], [], [], []


def parse_result_path(filepath):
    """
    解析新格式的结果文件路径
    目录结构：results/{num_envs}_{num_steps}/{algo_name}_{date}/{env_id}_{seed}.json
    例如：results/1_2048/opts_ttpo_continuous_action_20260225/HalfCheetah-v4_1.json

    Returns:
        (task, algo_name, date, seed) 或 None
    """
    import re
    path = Path(filepath)
    filename = path.stem  # e.g., "HalfCheetah-v4_1"
    algo_dir = path.parent.name  # e.g., "opts_ttpo_continuous_action_20260225"

    # Parse seed from filename: {env_id}_{seed}
    seed_match = re.search(r'_(\d+)$', filename)
    if not seed_match:
        return None
    seed = int(seed_match.group(1))
    task = filename[:seed_match.start()]  # e.g., "HalfCheetah-v4"

    # Parse algo_name and date from directory name: {algo_name}_{date}
    date_match = re.search(r'_(\d{8})$', algo_dir)
    if not date_match:
        return None
    date = date_match.group(1)
    algo_name = algo_dir[:date_match.start()]  # e.g., "opts_ttpo_continuous_action"

    return (task, algo_name, date, seed)


def aggregate_seed_results(all_seed_data):
    """
    聚合多个 seed 的结果，对同一行计算 mean 和 std

    各 seed 的 step 条数不一致时，取最短长度截断后再按索引对齐聚合。

    Args:
        all_seed_data: 字典，key 是 seed，value 是 (step列表, mean_return列表, max_return列表, min_return列表) 元组

    Returns:
        (aggregated_steps, aggregated_mean, aggregated_std) 元组
        其中 std 是跨 seed 的 mean_return 的标准差
    """
    if not all_seed_data:
        return [], [], []

    lengths = []
    for values in all_seed_data.values():
        if isinstance(values, tuple) and len(values) >= 4 and values[0]:
            lengths.append(len(values[0]))

    if not lengths:
        return [], [], []

    min_length = min(lengths)
    if min_length == 0:
        return [], [], []

    # 步数轴采用排序后第一个 seed 的 step（截断到最短长度）
    sorted_seeds = sorted(all_seed_data.keys())
    first_values = all_seed_data[sorted_seeds[0]]
    if not isinstance(first_values, tuple) or len(first_values) < 4:
        return [], [], []
    aggregated_steps = list(first_values[0][:min_length])

    aggregated_mean_values = []
    aggregated_std_values = []

    for i in range(min_length):
        row_values = []
        for seed in sorted_seeds:
            values = all_seed_data[seed]
            if isinstance(values, tuple) and len(values) >= 4:
                mean_returns = values[1]
                if i < len(mean_returns):
                    row_values.append(mean_returns[i])

        if row_values:
            aggregated_mean_values.append(np.mean(row_values))
            aggregated_std_values.append(np.std(row_values))

    aggregated_steps = aggregated_steps[:len(aggregated_mean_values)]

    return aggregated_steps, aggregated_mean_values, aggregated_std_values


def load_algo_filters_from_config(task_name, config_filename="algo_select.json"):
    """
    从配置文件中读取指定 task 的算法名列表。
    """
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


USE_SHORT_NAME = False


def get_display_name(algo_name, date=None):
    """
    获取算法的显示名称
    PPO/A2C/RPO 使用简短名称
    OPTS_TTPO 根据 USE_SHORT_NAME 决定是否显示简称
    """
    if algo_name == "ppo_continuous_action":
        return "PPO"
    if algo_name == "a2c_continuous_action":
        return "A2C"
    if algo_name == "rpo_continuous_action":
        return "RPO"
    if algo_name.startswith("opts_ttpo") and USE_SHORT_NAME:
        return "OPTS-TTPO"

    # 默认显示完整名称（包含日期）
    if date:
        return f"{algo_name}_{date}"
    return algo_name


def plot_all_tasks_convergence(results_dir="../cleanrl/results", output_dir=".",
                                algo_filters=None, smooth_window=5, seed_filters=None):
    """
    绘制所有5个任务的收敛曲线在一张图上（1行5列布局）
    
    Args:
        results_dir: results 目录路径
        output_dir: 输出图片的目录路径
        algo_filters: 要可视化的算法标识列表
        smooth_window: 平滑窗口大小
    """
    # 收集所有任务的数据
    all_tasks_data = {}
    
    for task_name in TARGET_TASKS:
        # 递归查找所有 JSON 文件
        files = list(Path(results_dir).rglob("*.json"))

        if not files:
            print(f"No results files found for task: {task_name}")
            continue

        algorithms_data = defaultdict(lambda: defaultdict(list))

        for filepath in files:
            parsed = parse_result_path(filepath)

            if parsed is None:
                continue

            task, algo_name, date, seed = parsed

            if task != task_name:
                continue

            if algo_filters is not None:
                algo_id = algo_name
                algo_id_with_date = f"{algo_name}_{date}"
                if (algo_id not in algo_filters) and (algo_id_with_date not in algo_filters):
                    continue

            if seed_filters is not None:
                if seed not in seed_filters:
                    continue
            else:
                if seed < 1 or seed > 10:
                    continue

            algo_key = (algo_name, date)

            step_values, mean_return_values, max_return_values, min_return_values = load_episodic_returns(filepath)
            if mean_return_values:
                algorithms_data[algo_key][seed] = (step_values, mean_return_values, max_return_values, min_return_values)

        if algorithms_data:
            aggregated_data = {}
            for algo_key, seed_data in algorithms_data.items():
                aggregated_steps, aggregated_mean, aggregated_std = aggregate_seed_results(seed_data)
                if aggregated_mean:
                    aggregated_data[algo_key] = {
                        'steps': aggregated_steps,
                        'mean': aggregated_mean,
                        'std': aggregated_std,
                    }
            all_tasks_data[task_name] = aggregated_data
    
    if not all_tasks_data:
        print("No data found for any task")
        return
    
    # 收集所有算法，为每个算法分配颜色
    all_algos = set()
    for task_data in all_tasks_data.values():
        for algo_key in task_data.keys():
            all_algos.add(algo_key)
    
    algo_colors = build_algo_colors(all_algos)

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    # 绘制每个任务
    for idx, task_name in enumerate(TARGET_TASKS):
        ax = axes[idx]
        
        if task_name not in all_tasks_data:
            ax.set_title(task_name.replace('-v4', '-v1'), fontsize=12)
            ax.set_xlabel("Timesteps", fontsize=10)
            if idx == 0:
                ax.set_ylabel("Mean Episodic Return", fontsize=10)
            continue
        
        task_data = all_tasks_data[task_name]
        
        task_smooth_window = get_task_smooth_window(task_name, smooth_window)
        for algo_key, data in task_data.items():
            algo_name, date = algo_key
            color = algo_colors[algo_key]
            display_name = get_display_name(algo_name, date)
            
            steps = np.array(data['steps'])
            mean_values = smooth_data(data['mean'], task_smooth_window)
            std_values = smooth_data(data['std'], task_smooth_window)

            # 绘制阴影区域（mean ± std）
            ax.fill_between(steps[:len(mean_values)],
                          mean_values - std_values,
                          mean_values + std_values,
                          color=color, alpha=0.2)

            # 绘制均值曲线
            ax.plot(steps[:len(mean_values)], mean_values,
                   color=color, label=display_name, linewidth=1.5)

        ax.set_title(task_name, fontsize=12)
        ax.set_xlabel("Timesteps", fontsize=10)
        if idx == 0:
            ax.set_ylabel("Mean Episodic Return", fontsize=10)

        # 设置x轴范围和刻度（只显示0和终点）
        if task_data:
            all_steps = []
            for data in task_data.values():
                all_steps.extend(data['steps'])
            if all_steps:
                max_step = max(all_steps)
                # 四舍五入到最近的1000000整数倍
                max_step_rounded = int(round(max_step / 1000000) * 1000000)
                if max_step_rounded == 0:
                    max_step_rounded = 1000000
                ax.set_xlim(0, max_step_rounded)
                ax.set_xticks([0, max_step_rounded])
                ax.set_xticklabels(["0", f"{max_step_rounded // 1_000_000}M"])
        
        ax.grid(True, alpha=0.3, which='major')

    handles, labels = [], []
    for algo_key in sorted(algo_colors.keys()):
        algo_name, date = algo_key
        display_name = get_display_name(algo_name, date)
        handles.append(plt.Line2D([0], [0], color=algo_colors[algo_key], linewidth=2))
        labels.append(display_name)

    ncol = min(len(handles), 4) if handles else 1
    fig.legend(handles, labels, loc='upper center', ncol=ncol, fontsize=11,
               bbox_to_anchor=(0.5, 1.08), frameon=True)

    plt.tight_layout()
    
    # 保存图片
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "all_tasks_mujoco.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Combined convergence curves saved to: {output_path}")
    plt.close()


def plot_convergence_curves(task_name, results_dir="./results", output_dir=".",
                            algo_filters=None, smooth_window=5, seed_filters=None):
    """
    绘制单个 task 下不同算法的收敛曲线（保留原有功能）
    
    Args:
        task_name: 任务名称
        results_dir: results 目录路径
        output_dir: 输出图片的目录路径
        algo_filters: 要可视化的算法标识列表
        smooth_window: 平滑窗口大小
    """
    # 递归查找所有 JSON 文件
    files = list(Path(results_dir).rglob("*.json"))

    if not files:
        print(f"No results files found for task: {task_name}")
        return

    algorithms_data = defaultdict(lambda: defaultdict(list))

    for filepath in files:
        parsed = parse_result_path(filepath)

        if parsed is None:
            continue

        task, algo_name, date, seed = parsed

        if task != task_name:
            continue

        if algo_filters is not None:
            algo_id = algo_name
            algo_id_with_date = f"{algo_name}_{date}"
            if (algo_id not in algo_filters) and (algo_id_with_date not in algo_filters):
                continue

        if seed_filters is not None:
            if seed not in seed_filters:
                continue
        else:
            if seed < 1 or seed > 10:
                continue

        algo_key = (algo_name, date)

        step_values, mean_return_values, max_return_values, min_return_values = load_episodic_returns(filepath)
        if mean_return_values:
            algorithms_data[algo_key][seed] = (step_values, mean_return_values, max_return_values, min_return_values)
    
    if not algorithms_data:
        print(f"Could not find valid data for task: {task_name}")
        return
    
    plt.figure(figsize=(12, 7))
    task_smooth_window = get_task_smooth_window(task_name, smooth_window)
    
    algo_keys = sorted(algorithms_data.keys())
    algo_colors = build_algo_colors(algo_keys)
    
    for algo_key in algo_keys:
        seed_data = algorithms_data[algo_key]
        algo_name, date = algo_key
        color = algo_colors[algo_key]
        display_name = get_display_name(algo_name, date)
        
        aggregated_steps, aggregated_mean, aggregated_std = aggregate_seed_results(seed_data)

        if not aggregated_mean:
            continue

        steps = np.array(aggregated_steps)
        mean_values = smooth_data(aggregated_mean, task_smooth_window)
        std_values = smooth_data(aggregated_std, task_smooth_window)

        # 绘制阴影区域（mean ± std）
        plt.fill_between(steps[:len(mean_values)],
                        mean_values - std_values,
                        mean_values + std_values,
                        color=color, alpha=0.2)

        # 绘制均值曲线
        plt.plot(steps[:len(mean_values)], mean_values,
                color=color, label=display_name, linewidth=2)

    plt.xlabel('Timesteps', fontsize=12)
    plt.ylabel('Mean Episodic Return (Averaged over seeds)', fontsize=12)
    plt.title(f'Convergence Curves for {task_name}', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # 设置x轴刻度（只显示0和终点）
    ax = plt.gca()
    xlim = ax.get_xlim()
    max_step = xlim[1]
    # 四舍五入到最近的1000000整数倍
    max_step_rounded = int(round(max_step / 1000000) * 1000000)
    if max_step_rounded == 0:
        max_step_rounded = 1000000
    ax.set_xlim(0, max_step_rounded)
    ax.set_xticks([0, max_step_rounded])
    ax.set_xticklabels(["0", f"{max_step_rounded // 1_000_000}M"])
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{task_name}_convergence.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Convergence curve saved to: {output_path}")
    plt.close()


def main():
    """主函数：绘制5个MuJoCo任务的收敛曲线合并图

    用法：
        python plot_convergence.py [--short-name] [--seeds 1,2,3] [results_dir] [algo1 algo2 ...]

        --short-name        OPTS_TTPO 使用简称 "OPTS-TTPO"（默认显示全称）
        --seeds 1,2,3       只可视化指定随机种子的数据（逗号分隔，默认 seed 1-10）
    """
    import sys
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
    global_algo_filters = filtered_args[1:] if len(filtered_args) > 1 else None

    if os.path.exists(results_dir):
        seed_info = f" (seeds: {sorted(seed_filters)})" if seed_filters else ""
        print(f"Plotting combined convergence curves for: {TARGET_TASKS}{seed_info}")
        script_dir = str(Path(__file__).resolve().parent)
        plot_all_tasks_convergence(results_dir, output_dir=script_dir,
                                   algo_filters=global_algo_filters,
                                   seed_filters=seed_filters)
    else:
        print(f"Results directory {results_dir} does not exist")
        print("Please run training first to generate results files.")


if __name__ == "__main__":
    main()
