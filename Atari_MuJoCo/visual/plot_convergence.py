"""
绘制相同 task 下不同算法的 episodic_return 收敛曲线
从 ./results/ 目录中读取数据文件，格式：{task}_{算法名}_{日期}_{seed}.txt
对于相同任务和相同算法，聚合 seed 1-10 的结果，计算同一行的均值
"""
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict


def load_episodic_returns(filepath):
    """从文件中读取 episodic_return 值（每行一个值）"""
    try:
        with open(filepath, 'r') as f:
            values = [float(line.strip()) for line in f if line.strip()]
        return values
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return []


def parse_filename(filename):
    """
    解析文件名，提取 task、算法名、日期和 seed
    格式：{task}_{算法名}_{日期}_{seed}.txt
    例如：HalfCheetah-v4_opts_ttpo_continuous_action_20260117_1.txt
    
    Returns:
        (task, algo_name, date, seed) 或 None
    """
    import re
    name_without_ext = filename.replace('.txt', '')
    
    # 匹配格式：task_algo_date_seed
    # task 可能包含 -，如 HalfCheetah-v4
    # seed 是最后一个数字
    
    # 先找到最后一个下划线后的数字（seed）
    seed_match = re.search(r'_(\d+)$', name_without_ext)
    if not seed_match:
        return None
    
    seed = int(seed_match.group(1))
    remaining = name_without_ext[:seed_match.start()]  # 移除 _seed 部分
    
    # 现在 remaining 格式是：task_algo_date
    # 找到日期（8位数字）
    date_match = re.search(r'_(\d{8})$', remaining)
    if not date_match:
        return None
    
    date = date_match.group(1)
    remaining = remaining[:date_match.start()]  # 移除 _date 部分
    
    # 现在 remaining 格式是：task_algo
    # task 是第一部分（可能包含 -）
    # 算法名是剩余部分
    parts = remaining.split('_', 1)
    if len(parts) < 2:
        return None
    
    task = parts[0]
    algo_name = parts[1]
    
    return (task, algo_name, date, seed)


def aggregate_seed_results(all_seed_data):
    """
    聚合多个 seed 的结果，对同一行（相同 iteration）计算均值
    
    Args:
        all_seed_data: 字典，key 是 seed，value 是该 seed 的所有 return 值列表
    
    Returns:
        aggregated_values: 聚合后的均值列表
    """
    if not all_seed_data:
        return []
    
    # 找到最长的序列长度
    max_length = max(len(values) for values in all_seed_data.values() if values)
    
    if max_length == 0:
        return []
    
    # 对每一行计算均值
    aggregated_values = []
    for i in range(max_length):
        # 收集所有 seed 在第 i 行的值（如果该 seed 有这一行）
        row_values = []
        for seed, values in all_seed_data.items():
            if i < len(values):
                row_values.append(values[i])
        
        # 计算均值
        if row_values:
            mean_value = np.mean(row_values)
            aggregated_values.append(mean_value)
        else:
            break  # 如果没有 seed 有这一行，停止
    
    return aggregated_values


def plot_convergence_curves(task_name, results_dir="./results", output_dir="./visual"):
    """
    绘制相同 task 下不同算法的收敛曲线
    对于相同任务和相同算法，聚合 seed 1-10 的结果，计算同一行的均值
    
    Args:
        task_name: 任务名称（如 "HalfCheetah-v4" 或 "BreakoutNoFrameskip-v4"）
        results_dir: results 目录路径
        output_dir: 输出图片的目录路径
    """
    # 查找该 task 的所有结果文件
    pattern = os.path.join(results_dir, f"{task_name}_*.txt")
    files = glob.glob(pattern)
    
    if not files:
        print(f"No results files found for task: {task_name}")
        print(f"Searched pattern: {pattern}")
        return
    
    # 按 (task, algo_name) 分组，收集所有 seed 的数据
    # 结构：algorithms_data[algo_name] = {seed: [values]}
    algorithms_data = defaultdict(lambda: defaultdict(list))
    
    for filepath in files:
        filename = os.path.basename(filepath)
        parsed = parse_filename(filename)
        
        if parsed is None:
            print(f"Warning: Could not parse filename: {filename}")
            continue
        
        task, algo_name, date, seed = parsed
        
        # 只处理指定 task 和 seed 1-10 的文件
        if task != task_name:
            continue
        
        if seed < 1 or seed > 10:
            continue
        
        # 加载数据
        values = load_episodic_returns(filepath)
        if values:
            algorithms_data[algo_name][seed] = values
    
    if not algorithms_data:
        print(f"Could not find valid data for task: {task_name}")
        print(f"Files found: {files[:5]}...")  # 只显示前5个文件
        return
    
    plt.figure(figsize=(10, 6))
    
    # 对每种算法，聚合所有 seed 的结果
    aggregated_data = {}
    for algo_name, seed_data in algorithms_data.items():
        # 聚合 seed 1-10 的结果
        aggregated_values = aggregate_seed_results(seed_data)
        if aggregated_values:
            aggregated_data[algo_name] = aggregated_values
    
    if not aggregated_data:
        print(f"No aggregated data available for task: {task_name}")
        return
    
    # 计算横坐标（假设每个值对应一次 iteration）
    max_iterations = max(len(values) for values in aggregated_data.values())
    total_timesteps = 1000000
    # 假设步数均匀分布
    x_values = np.linspace(0, total_timesteps, max_iterations)
    
    # 绘制每种算法的曲线
    for algo_name, values in aggregated_data.items():
        if len(values) < max_iterations:
            # 线性插值到 max_iterations 个点
            y_interp = np.interp(x_values, np.linspace(0, total_timesteps, len(values)), values)
            plt.plot(x_values, y_interp, label=algo_name, linewidth=2)
        else:
            plt.plot(x_values[:len(values)], values, label=algo_name, linewidth=2)
    
    plt.xlabel('Timesteps', fontsize=12)
    plt.ylabel('Mean Episodic Return (Averaged over seeds)', fontsize=12)
    plt.title(f'Convergence Curves for {task_name} (Aggregated over seeds 1-10)', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # 设置横坐标范围 0-1000000，只显示 0 和 1000000
    plt.xlim(0, total_timesteps)
    plt.xticks([0, total_timesteps], ['0', '1000000'])
    
    # 保存图片
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{task_name}_convergence.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Convergence curve saved to: {output_path}")
    print(f"  Algorithms: {list(aggregated_data.keys())}")
    for algo_name, values in aggregated_data.items():
        num_seeds = len(algorithms_data[algo_name])
        print(f"    {algo_name}: {num_seeds} seeds, {len(values)} iterations")
    plt.close()


def main():
    """主函数：可以为多个 task 绘制收敛曲线
    
    用法：
        python plot_convergence.py                                    # 自动检测所有 task 并绘制
        python plot_convergence.py HalfCheetah-v4                     # 绘制指定 task 的收敛曲线
        python plot_convergence.py HalfCheetah-v4 ../cleanrl/results  # 指定 results 目录
        
    注意：results_dir 默认为 "./results"，如果从 visual 目录运行，需要指定正确的路径，
         例如 "../cleanrl/results" 或使用绝对路径
    """
    import sys
    
    # 默认 results 目录为当前目录下的 results
    # 如果从 visual 目录运行，可以指定 ../cleanrl/results
    if len(sys.argv) > 2:
        results_dir = sys.argv[2]
    else:
        results_dir = "./results"
    
    # 如果提供了命令行参数，只绘制指定的 task
    if len(sys.argv) > 1:
        task_name = sys.argv[1]
        plot_convergence_curves(task_name, results_dir)
        return
    
    # 自动检测所有 task
    if os.path.exists(results_dir):
        files = glob.glob(os.path.join(results_dir, "*.txt"))
        tasks = set()
        for filepath in files:
            filename = os.path.basename(filepath)
            parsed = parse_filename(filename)
            if parsed is not None:
                task, algo_name, date, seed = parsed
                tasks.add(task)
        
        if tasks:
            print(f"Found tasks: {tasks}")
            for task in sorted(tasks):
                print(f"\nPlotting convergence curve for {task}...")
                plot_convergence_curves(task, results_dir)
        else:
            print(f"No task files found in {results_dir}")
    else:
        print(f"Results directory {results_dir} does not exist")
        print("Please run training first to generate results files.")


if __name__ == "__main__":
    main()
