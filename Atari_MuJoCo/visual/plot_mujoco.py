# PPO 蓝、RPO 绿；OPTS-TTPO 多版本使用一组不同颜色；其余算法用补充色
COLOR_PPO = "#1f77b4"
COLOR_RPO = "#2ca02c"
OPTS_TTPO_COLORS = [
    "#d62728",  # red
    "#ff7f0e",  # orange
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#17becf",  # cyan
    "#bcbd22",  # olive
    "#7f7f7f",  # gray
]
EXTRA_ALGO_COLORS = ["#1f77b4", "#2ca02c", "#17becf", "#8c564b", "#e377c2", "#bcbd22", "#7f7f7f"]


def build_algo_colors(algo_keys):
    """
    为算法分配颜色：
    - PPO 固定蓝色；
    - RPO 固定绿色；
    - 多个 OPTS-TTPO（opts_ttpo*）按排序稳定地分配不同颜色；
    - 其他算法按排序稳定地依次使用 EXTRA_ALGO_COLORS，并避开已使用颜色。
    """
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
        if algo_name == "ppo_continuous_action":
            colors[algo_key] = COLOR_PPO
        elif algo_name == "rpo_continuous_action":
            colors[algo_key] = COLOR_RPO
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