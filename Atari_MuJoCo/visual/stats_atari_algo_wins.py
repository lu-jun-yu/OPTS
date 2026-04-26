"""
比较两个 Atari 算法在各任务上的「胜负」计数（基于 JSON 日志中的 mean_return 序列）。

用法（与 plot_atari.py 一致）：
    cd Atari_MuJoCo
    python visual/stats_atari_algo_wins.py results/8_128 ppo_atari opts_ttpo_atari_tau0.4_s6

指标说明（由 cleanrl 写入的 JSON：每条为一次迭代的统计）：
- 全训练平均：对该任务下所有日志点的 mean_return 取平均。
- 尾部平均（对应论文里常见的 last-100-episodes 思想的可行近似）：
  日志中**没有逐 episode 的 return**，因此用「最后 LAST_LOG_POINTS 条记录的
  mean_return 再平均」近似训练末期表现；每条记录本身是当轮内已完成回合的
  平均回报。可用 --last-log-points 修改窗口大小（默认 100）。

胜负规则：对每个任务分别汇总两个算法（匹配到的所有日期目录、所有种子）
的上述标量后再比较；分数高者在该任务上记 1 胜。分数相同为平局（仅汇总输出
两侧胜场数，不展示平局项）。

可选参数：
    --seeds 1,2,3       只统计指定种子
    --tasks env1,env2  只统计列出的任务（默认同 plot_atari 的 57 个）
    --last-log-points N 尾部指标使用的日志条数（默认 100）
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Sequence, Tuple

# 与 visual/plot_atari.py 保持一致（避免依赖 matplotlib/numpy）
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


def load_episodic_returns(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        step_values = []
        mean_return_values = []
        for item in data:
            if isinstance(item, dict) and "mean_return" in item and "step" in item:
                step_values.append(float(item["step"]))
                mean_return_values.append(float(item["mean_return"]))
        return step_values, mean_return_values
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return [], []


def parse_result_path(filepath):
    path = Path(filepath)
    filename = path.stem
    algo_dir = path.parent.name
    seed_match = re.search(r"_(\d+)$", filename)
    if not seed_match:
        return None
    seed = int(seed_match.group(1))
    task = filename[: seed_match.start()]
    date_match = re.search(r"_(\d{8})$", algo_dir)
    if not date_match:
        return None
    date = date_match.group(1)
    algo_name = algo_dir[: date_match.start()]
    return (task, algo_name, date, seed)


def _parse_seed_list(s: str) -> Optional[set]:
    if not s.strip():
        return None
    return {int(x.strip()) for x in s.split(",") if x.strip()}


def _algo_matches_filter(algo_name: str, date: str, filt: str) -> bool:
    algo_id_with_date = f"{algo_name}_{date}"
    return filt == algo_name or filt == algo_id_with_date


def _aggregate_seed_curves_stepwise(
    seed_curves: Dict[int, Tuple[List[float], List[float]]],
    task_name: Optional[str] = None,
    algo_name: Optional[str] = None,
) -> Tuple[List[float], List[float]]:
    """
    参考 plot_mujoco.py：各 seed 曲线长度不一致时，先截断到最短长度，
    再按索引（step-level）跨 seed 逐点求平均。
    """
    if not seed_curves:
        return [], []

    sorted_seeds = sorted(seed_curves.keys())
    valid = []
    for s in sorted_seeds:
        steps, returns = seed_curves[s]
        if steps and returns:
            valid.append((s, steps, returns))
    if not valid:
        return [], []

    step_lens = {s: len(steps) for s, steps, _ in valid}
    ret_lens = {s: len(rets) for s, _, rets in valid}
    aligned_lens = {s: min(step_lens[s], ret_lens[s]) for s, _, _ in valid}
    min_len = min(aligned_lens.values())
    if min_len <= 0:
        return [], []

    if len(set(aligned_lens.values())) > 1:
        ctx_task = task_name if task_name is not None else "<unknown_task>"
        ctx_algo = algo_name if algo_name is not None else "<unknown_algo>"
        detail = ", ".join(
            f"seed{s}:{aligned_lens[s]}(step={step_lens[s]},return={ret_lens[s]})"
            for s in sorted(aligned_lens.keys())
        )
        print(
            f"[WARNING] step-level 截断触发: task={ctx_task}, algo={ctx_algo}, "
            f"min_len={min_len}, lengths=[{detail}]"
        )

    aggregated_steps = list(valid[0][1][:min_len])
    aggregated_returns = []
    for i in range(min_len):
        row = []
        for _, _, rets in valid:
            row.append(float(rets[i]))
        aggregated_returns.append(float(mean(row)))
    return aggregated_steps, aggregated_returns


def _collect_task_algo_scores(
    results_dir: Path,
    algo_filters: Tuple[str, str],
    task_allow: Optional[set],
    seed_filters: Optional[set],
    last_log_points: int,
) -> Dict[str, Dict[str, List[float]]]:
    """
    返回 task -> algo_key_str -> [score]
    其中 score 来自“先按 seed 做 step-level 平均（长度不对齐时截断最短），
    再对聚合曲线求全局均值/尾部均值”。
    """
    # task -> algo(0/1) -> seed -> (date, steps, returns)
    raw: Dict[str, Dict[int, Dict[int, Tuple[str, List[float], List[float]]]]] = defaultdict(
        lambda: {0: {}, 1: {}}
    )

    for filepath in results_dir.rglob("*.json"):
        parsed = parse_result_path(filepath)
        if parsed is None:
            continue
        task, algo_name, date, seed = parsed
        if task_allow is not None and task not in task_allow:
            continue
        if task not in TARGET_TASKS:
            continue
        if seed_filters is not None and seed not in seed_filters:
            continue

        which: Optional[int] = None
        for i, filt in enumerate(algo_filters):
            if _algo_matches_filter(algo_name, date, filt):
                which = i
                break
        if which is None:
            continue

        steps, returns = load_episodic_returns(filepath)
        if not returns:
            continue

        # 同一 task/algo/seed 若有多条记录，优先保留日期更晚的一条
        prev = raw[task][which].get(seed)
        if prev is None or date > prev[0]:
            raw[task][which][seed] = (date, [float(x) for x in steps], [float(x) for x in returns])

    out: Dict[str, Dict[str, List[float]]] = {}
    for task, sides in raw.items():
        seed_curves_a = {s: (v[1], v[2]) for s, v in sides[0].items()}
        seed_curves_b = {s: (v[1], v[2]) for s, v in sides[1].items()}

        _, agg_a = _aggregate_seed_curves_stepwise(seed_curves_a, task, algo_filters[0])
        _, agg_b = _aggregate_seed_curves_stepwise(seed_curves_b, task, algo_filters[1])

        all_a: List[float] = []
        all_b: List[float] = []
        tail_a: List[float] = []
        tail_b: List[float] = []
        if agg_a:
            all_a = [float(mean(agg_a))]
            tail_n_a = min(last_log_points, len(agg_a))
            tail_a = [float(mean(agg_a[-tail_n_a:]))]
        if agg_b:
            all_b = [float(mean(agg_b))]
            tail_n_b = min(last_log_points, len(agg_b))
            tail_b = [float(mean(agg_b[-tail_n_b:]))]

        out[task] = {
            algo_filters[0]: all_a,
            algo_filters[1]: all_b,
            f"{algo_filters[0]}::__tail__": tail_a,
            f"{algo_filters[1]}::__tail__": tail_b,
        }
    return out


def _aggregate_scores(pairs: List[float]) -> float:
    if not pairs:
        return float("nan")
    return float(mean(pairs))


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    seed_filters: Optional[set] = {1, 2, 3}
    last_log_points = 100
    task_subset: Optional[set] = None

    filtered: List[str] = []
    i = 0
    while i < len(argv):
        if argv[i] == "--seeds":
            if i + 1 >= len(argv):
                print("错误: --seeds 需要参数，例如 --seeds 1,2,3", file=sys.stderr)
                return 1
            seed_filters = _parse_seed_list(argv[i + 1])
            i += 2
            continue
        if argv[i] == "--last-log-points":
            if i + 1 >= len(argv):
                print("错误: --last-log-points 需要整数参数", file=sys.stderr)
                return 1
            last_log_points = max(1, int(argv[i + 1]))
            i += 2
            continue
        if argv[i] == "--tasks":
            if i + 1 >= len(argv):
                print("错误: --tasks 需要逗号分隔的环境 id 列表", file=sys.stderr)
                return 1
            task_subset = {t.strip() for t in argv[i + 1].split(",") if t.strip()}
            i += 2
            continue
        filtered.append(argv[i])
        i += 1

    if len(filtered) < 3:
        print(
            "用法: python visual/stats_atari_algo_wins.py "
            "[--seeds 1,2,3] [--last-log-points 100] [--tasks env1,env2] "
            "<results_dir> <algo_a> <algo_b>",
            file=sys.stderr,
        )
        return 1

    results_dir = Path(filtered[0])
    algo_a, algo_b = filtered[1], filtered[2]

    if not results_dir.is_dir():
        print(f"结果目录不存在: {results_dir}", file=sys.stderr)
        return 1

    task_allow = task_subset
    collected = _collect_task_algo_scores(
        results_dir,
        (algo_a, algo_b),
        task_allow,
        seed_filters,
        last_log_points,
    )

    tasks_eval = sorted(
        collected.keys(),
        key=lambda t: TARGET_TASKS.index(t) if t in TARGET_TASKS else 999,
    )

    wins_a_all = wins_b_all = 0
    wins_a_tail = wins_b_tail = 0
    missing_both = 0
    n_compared = 0

    for task in tasks_eval:
        d = collected[task]
        scores_a_all = d.get(algo_a, [])
        scores_b_all = d.get(algo_b, [])
        scores_a_tail = d.get(f"{algo_a}::__tail__", [])
        scores_b_tail = d.get(f"{algo_b}::__tail__", [])

        if not scores_a_all or not scores_b_all:
            missing_both += 1
            continue

        ma = _aggregate_scores(scores_a_all)
        mb = _aggregate_scores(scores_b_all)
        ta = _aggregate_scores(scores_a_tail)
        tb = _aggregate_scores(scores_b_tail)

        n_compared += 1
        if ma > mb:
            wins_a_all += 1
        elif mb > ma:
            wins_b_all += 1

        if ta > tb:
            wins_a_tail += 1
        elif tb > ta:
            wins_b_tail += 1

    total_with_data = n_compared

    print(f"结果目录: {results_dir.resolve()}")
    print(f"算法 A: {algo_a}")
    print(f"算法 B: {algo_b}")
    if seed_filters is not None:
        print(f"种子过滤: {sorted(seed_filters)}")
    print(f"尾部窗口: 最后 {last_log_points} 条日志记录的 mean_return 平均")
    print()
    print("=== 汇总：在同时有两个算法数据的任务上 ===")
    print(f"可比任务数: {total_with_data}")
    print()
    print("1) 全训练期平均（所有日志点 mean_return 的平均）")
    print(f"    {algo_a} 胜: {wins_a_all}")
    print(f"    {algo_b} 胜: {wins_b_all}")
    print()
    print("2) 尾部平均（最后若干条迭代日志的 mean_return 平均，见上文说明）")
    print(f"    {algo_a} 胜: {wins_a_tail}")
    print(f"    {algo_b} 胜: {wins_b_tail}")
    if missing_both:
        print()
        print(f"（另有 {missing_both} 个任务因缺少任一侧数据未计入胜负）")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
