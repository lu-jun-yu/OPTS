import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator


# ==========================================================================
# ALGORITHMS TO PLOT — method tags in the eval JSON filenames.
# Edit this list to switch/extend the algorithms.
ALGORITHMS = ["opts_ttpo_exp8_3_0810_n8"]
# ==========================================================================

OPTS_METHOD = ALGORITHMS[0]

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVAL_DIR = REPO_ROOT / "LLM/results/step400/eval"
DEFAULT_TASK2_IID = DEFAULT_EVAL_DIR / f"{OPTS_METHOD}_iid_n128__task2_iid_pass_k8-16-32-64-128.json"
DEFAULT_TASK2_OPTS = DEFAULT_EVAL_DIR / f"{OPTS_METHOD}_opts_reward_n128__task2_reward_opts_k8-16-32-64-128.json"
DEFAULT_TASK3_IID = DEFAULT_EVAL_DIR / f"{OPTS_METHOD}_iid_n128__task3_iid_cons_k8-16-32-64-128.json"
DEFAULT_TASK3_OPTS = DEFAULT_EVAL_DIR / f"{OPTS_METHOD}_opts_value_n128__task3_value_opts_k8-16-32-64-128.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "paper/figures"

# --- Display names (edit here to relabel the figures) -----------------------
IID_LABEL = "IID"                  # baseline curve in the s-figures
OPTS_LABEL_TEMPLATE = "OPTS (s{})"  # per-s curve label
REWARD_TITLE = "Reward-guided OPTS"
VALUE_TITLE = "Value-guided OPTS"
# Labels for the optional 1x6 task figures (--task-figures).
TASK2_IID_LABEL = "IID sampling"
TASK2_OPTS_LABEL = "Reward-guided OPTS"
TASK3_IID_LABEL = "IID cons"
TASK3_OPTS_LABEL = "Value OPTS"
# ---------------------------------------------------------------------------

DATASET_NAME_MAP = {
    "hiyouga/math12k": "MATH500",
    "math-ai/aime24": "AIME24",
    "math-ai/aime25": "AIME25",
    "math-ai/aime26": "AIME26",
    "math-ai/amc23": "AMC23",
    "math-ai/minervamath": "MinervaMath",
}

DATASET_ORDER = [
    "hiyouga/math12k",
    "math-ai/aime24",
    "math-ai/aime25",
    "math-ai/aime26",
    "math-ai/amc23",
    "math-ai/minervamath",
]

# Styles for the max-search-per-tree figures (the s{n} tag in eval
# filenames; s = maximum number of searches per tree).
S_BASELINE_STYLE = {
    "color": "#4E79A7",
    "marker": "o",
    "linestyle": "--",
    "linewidth": 2.0,
    "markersize": 5.5,
    "markeredgecolor": "white",
    "markeredgewidth": 0.6,
}

S_VALUE_STYLES = {
    1: {
        "color": "#F2BE4A",
        "marker": "s",
        "linewidth": 2.4,
        "markersize": 6.5,
        "markeredgecolor": "white",
        "markeredgewidth": 0.6,
    },
    3: {
        "color": "#EF9330",
        "marker": "D",
        "linewidth": 2.4,
        "markersize": 6.0,
        "markeredgecolor": "white",
        "markeredgewidth": 0.6,
    },
    7: {
        "color": "#D95F59",
        "marker": "^",
        "linewidth": 2.4,
        "markersize": 6.5,
        "markeredgecolor": "white",
        "markeredgewidth": 0.6,
    },
}


S_RCPARAMS = {
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9.5,
    "axes.edgecolor": "#999999",
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


def set_s_figure_style():
    plt.rcParams.update(S_RCPARAMS)


def finish_s_axis(axis, k_values, all_values, fixed_y):
    axis.set_xticks(range(len(k_values)))
    axis.set_xticklabels([str(k_value) for k_value in k_values])
    axis.tick_params(axis="both", colors="#555555", length=3.5, width=0.7)
    axis.grid(
        True,
        axis="y",
        color="#CCCCCC",
        linestyle=(0, (4, 3)),
        linewidth=0.7,
        alpha=0.7,
    )
    axis.set_axisbelow(True)
    axis.margins(x=0.05)
    if fixed_y:
        axis.set_ylim(0.0, 1.0)
    else:
        set_dynamic_ylim(axis, all_values)
    axis.yaxis.set_major_locator(MaxNLocator(nbins=4))
    axis.yaxis.set_major_formatter(FormatStrFormatter("%.3g"))


LINE_STYLES = [
    {
        "color": "#456FA6",
        "marker": "o",
        "linewidth": 3.0,
        "markersize": 8,
        "markeredgecolor": "white",
        "markeredgewidth": 0.9,
    },
    {
        "color": "#D95F59",
        "marker": "s",
        "linewidth": 3.0,
        "markersize": 8,
        "markeredgecolor": "white",
        "markeredgewidth": 0.9,
    },
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot step-400 eval scores at k as 1x6 PDF figures."
    )
    parser.add_argument("--task2-iid", default=DEFAULT_TASK2_IID)
    parser.add_argument("--task2-opts", default=DEFAULT_TASK2_OPTS)
    parser.add_argument("--task3-iid", default=DEFAULT_TASK3_IID)
    parser.add_argument("--task3-opts", default=DEFAULT_TASK3_OPTS)
    parser.add_argument(
        "--s-values",
        type=int,
        nargs="+",
        default=[1, 3, 7],
        help="max search-per-tree values (the s{n} tag in eval filenames) to "
        "plot against the IID baselines (default: 1 3 7). "
        "Missing s{n} JSONs are skipped.",
    )
    parser.add_argument(
        "--task-figures",
        action="store_true",
        help="Also write the per-benchmark 1x6 task2/task3 figures "
        "(step400_task2_*/step400_task3_*). Off by default.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for output files (default: paper/figures).",
    )
    parser.add_argument(
        "--format",
        default="pdf",
        choices=["png", "pdf", "svg"],
        help="Output image format (default: pdf).",
    )
    parser.add_argument(
        "--fixed-y",
        action="store_true",
        help="Use a fixed y-axis range of [0, 1] for every subplot.",
    )
    return parser.parse_args()


def find_s_eval(eval_dir, mode, s_value):
    """Locate the s{n}-tagged OPTS eval JSON, tolerating varying k tags."""
    task = "task2" if mode == "reward" else "task3"
    pattern = f"{OPTS_METHOD}_opts_{mode}_s{s_value}_n128__{task}_{mode}_opts_k*.json"
    matches = sorted(Path(eval_dir).glob(pattern))
    if len(matches) > 1:
        raise ValueError(f"Multiple matches for {pattern}: {matches}")
    return matches[0] if matches else None


def load_eval(path):
    eval_path = Path(path)
    with eval_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    metrics = data.get("metrics", [])
    if not metrics:
        raise ValueError(f"No metrics found in {eval_path}")

    return {
        "path": eval_path,
        "metric": metrics[0],
        "k": [int(k) for k in data["k"]],
        "results": data["results"],
    }


def score_series(eval_data, dataset, k_values=None):
    metric = eval_data["metric"]
    result = eval_data["results"].get(dataset, {})
    values = []
    for k_value in k_values if k_values is not None else eval_data["k"]:
        key = f"{metric}@{k_value}"
        if key not in result:
            raise KeyError(f"Missing {key} for {dataset} in {eval_data['path']}")
        values.append(float(result[key]))
    return values


def common_datasets(first_eval, second_eval):
    first = set(first_eval["results"])
    second = set(second_eval["results"])
    available = [name for name in DATASET_ORDER if name in first and name in second]
    extras = sorted((first & second) - set(DATASET_ORDER) - {"_all"})
    return available + extras


def set_dynamic_ylim(axis, values):
    low = min(values)
    high = max(values)
    if high == low:
        pad = 0.03
    else:
        pad = max((high - low) * 0.18, 0.015)
    axis.set_ylim(max(0.0, low - pad), min(1.0, high + pad))


def plot_pair(first_eval, second_eval, first_label, second_label, output_path, fixed_y):
    if first_eval["k"] != second_eval["k"]:
        shared_k = [k for k in first_eval["k"] if k in set(second_eval["k"])]
        if not shared_k:
            raise ValueError(
                f"k values do not overlap: {first_eval['path']} has {first_eval['k']}, "
                f"{second_eval['path']} has {second_eval['k']}"
            )
        print(
            f"k values differ; using shared k={shared_k} for {output_path}"
        )
    else:
        shared_k = first_eval["k"]

    datasets = common_datasets(first_eval, second_eval)
    if len(datasets) != len(DATASET_ORDER):
        raise ValueError(
            f"Expected {len(DATASET_ORDER)} datasets, found {len(datasets)}: {datasets}"
        )

    k_values = shared_k
    x_values = list(range(len(k_values)))

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 18,
            "axes.titlesize": 22,
            "axes.labelsize": 20,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 18,
            "axes.edgecolor": "#777777",
            "axes.linewidth": 0.9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(
        1,
        len(datasets),
        figsize=(3.75 * len(datasets), 4.4),
        sharex=True,
    )
    legend_handles = []
    legend_labels = []

    for axis, dataset in zip(axes, datasets):
        first_values = score_series(first_eval, dataset, k_values)
        second_values = score_series(second_eval, dataset, k_values)
        all_values = first_values + second_values

        for values, label, style in zip(
            [first_values, second_values],
            [first_label, second_label],
            LINE_STYLES,
        ):
            line, = axis.plot(x_values, values, label=label, **style)
            if label not in legend_labels:
                legend_handles.append(line)
                legend_labels.append(label)

        axis.set_title(
            DATASET_NAME_MAP.get(dataset, dataset),
            fontweight="semibold",
            pad=14,
        )
        axis.set_xticks(x_values)
        axis.set_xticklabels([str(k_value) for k_value in k_values])
        axis.tick_params(axis="x", labelbottom=True)
        axis.tick_params(axis="both", colors="#333333", length=4, width=0.8)
        axis.grid(
            True,
            axis="y",
            color="#AEB6BF",
            linestyle=(0, (3, 3)),
            linewidth=0.8,
            alpha=0.45,
        )
        axis.set_axisbelow(True)
        axis.margins(x=0.04)
        if fixed_y:
            axis.set_ylim(0.0, 1.0)
        else:
            set_dynamic_ylim(axis, all_values)
        axis.yaxis.set_major_locator(MaxNLocator(nbins=5))

    axes[0].set_ylabel("Success rate", labelpad=10)
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
        handlelength=2.2,
        columnspacing=1.8,
    )
    fig.tight_layout(rect=(0.005, 0.02, 0.995, 0.90), w_pad=1.6)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_file.resolve()}")


def plot_s_overview(panels, output_path, fixed_y):
    """Plot per-s (max search per tree) OPTS curves on the aggregated test
    set (_all).

    panels: list of two entries (left, right), each either None or
    (iid_eval, [(s_value, eval), ...], title).
    """
    set_s_figure_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.7))

    for axis, panel in zip(axes, panels):
        if panel is None:
            axis.axis("off")
            continue
        iid_eval, s_evals, title = panel

        k_values = sorted(
            {k for eval_data in [iid_eval] + [ev for _, ev in s_evals] for k in eval_data["k"]}
        )
        k_index = {k_value: i for i, k_value in enumerate(k_values)}

        curves = [(iid_eval, IID_LABEL, S_BASELINE_STYLE)] + [
            (s_eval, OPTS_LABEL_TEMPLATE.format(s_value), S_VALUE_STYLES.get(s_value, LINE_STYLES[1]))
            for s_value, s_eval in s_evals
        ]
        all_values = []
        for eval_data, label, style in curves:
            values = score_series(eval_data, "_all", eval_data["k"])
            axis.plot(
                [k_index[k] for k in eval_data["k"]],
                values,
                label=label,
                **style,
            )
            all_values.extend(values)

        axis.set_title(title, fontweight="bold", pad=8)
        axis.set_xlabel("k")
        axis.set_ylabel(f"{iid_eval['metric']}@k")
        finish_s_axis(axis, k_values, all_values, fixed_y)
        axis.legend(frameon=False, loc="upper left", borderaxespad=0.1)

    fig.tight_layout()

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_file.resolve()}")


def plot_s_grid(panels, output_path, fixed_y):
    """Plot per-s (max search per tree) OPTS curves split by benchmark in a
    2x6 grid.

    Top row: reward-guided OPTS; bottom row: value-guided OPTS. Each column
    is one benchmark. panels has the same structure as in plot_s_overview.
    """
    set_s_figure_style()
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 12,
        }
    )
    datasets = DATASET_ORDER
    fig, axes = plt.subplots(
        2,
        len(datasets),
        figsize=(2.9 * len(datasets), 7.6),
        sharex=True,
    )

    shared_handles = []
    shared_labels = []

    for row, (panel, row_axes) in enumerate(zip(panels, axes)):
        if panel is None:
            for axis in row_axes:
                axis.axis("off")
            continue
        iid_eval, s_evals, row_title = panel

        k_values = sorted(
            {k for eval_data in [iid_eval] + [ev for _, ev in s_evals] for k in eval_data["k"]}
        )
        k_index = {k_value: i for i, k_value in enumerate(k_values)}

        curves = [(iid_eval, IID_LABEL, S_BASELINE_STYLE)] + [
            (s_eval, OPTS_LABEL_TEMPLATE.format(s_value), S_VALUE_STYLES.get(s_value, LINE_STYLES[1]))
            for s_value, s_eval in s_evals
        ]

        for axis, dataset in zip(row_axes, datasets):
            all_values = []
            for eval_data, label, style in curves:
                values = score_series(eval_data, dataset, eval_data["k"])
                (line,) = axis.plot(
                    [k_index[k] for k in eval_data["k"]],
                    values,
                    label=label,
                    **style,
                )
                all_values.extend(values)
                if row == 0 and label not in shared_labels:
                    shared_handles.append(line)
                    shared_labels.append(label)

            if row == 0:
                axis.set_title(
                    DATASET_NAME_MAP.get(dataset, dataset),
                    fontweight="bold",
                    pad=8,
                )
                axis.tick_params(axis="x", labelbottom=False)
            else:
                axis.set_xlabel("k")
            finish_s_axis(axis, k_values, all_values, fixed_y)

        row_axes[0].set_ylabel(f"{iid_eval['metric']}@k")
        fig.text(
            0.010,
            1.0 if row == 0 else 0.505,
            row_title,
            va="top",
            ha="left",
            fontsize=13,
            fontweight="bold",
        )

    if shared_handles:
        fig.legend(
            shared_handles,
            shared_labels,
            loc="upper center",
            ncol=len(shared_labels),
            frameon=False,
            bbox_to_anchor=(0.5, 1.0),
            handlelength=2.6,
            columnspacing=1.8,
        )

    fig.subplots_adjust(
        left=0.045, right=0.995, top=0.90, bottom=0.08, hspace=0.48, wspace=0.32
    )

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_file.resolve()}")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)

    task2_iid = load_eval(args.task2_iid)

    task3_iid_path = Path(args.task3_iid)
    task3_opts_path = Path(args.task3_opts)
    if args.task_figures:
        task2_opts = load_eval(args.task2_opts)
        plot_pair(
            task2_iid,
            task2_opts,
            TASK2_IID_LABEL,
            TASK2_OPTS_LABEL,
            output_dir / f"step400_task2_iid_pass_vs_reward_opts.{args.format}",
            args.fixed_y,
        )

        if task3_iid_path.is_file() and task3_opts_path.is_file():
            plot_pair(
                load_eval(task3_iid_path),
                load_eval(task3_opts_path),
                TASK3_IID_LABEL,
                TASK3_OPTS_LABEL,
                output_dir / f"step400_task3_iid_cons_vs_value_opts.{args.format}",
                args.fixed_y,
            )
        else:
            print(
                "Skipped task 3: both value-OPTS and IID-consistency JSON files "
                "are required."
            )

    # max_search_per_tree (s{n}) variants: reward-guided (left) and
    # value-guided (right) panels in one figure. Missing s{n} eval JSONs are
    # skipped.
    task3_iid = load_eval(task3_iid_path) if task3_iid_path.is_file() else None
    panels = []
    for mode, iid_eval, title in [
        ("reward", task2_iid, REWARD_TITLE),
        ("value", task3_iid, VALUE_TITLE),
    ]:
        s_evals = []
        for s_value in args.s_values:
            s_path = find_s_eval(DEFAULT_EVAL_DIR, mode, s_value)
            if s_path is None:
                print(f"Skipped {mode} s{s_value}: eval JSON not found.")
                continue
            s_evals.append((s_value, load_eval(s_path)))
        if iid_eval is None or not s_evals:
            panels.append(None)
        else:
            panels.append((iid_eval, s_evals, title))
    if any(panel is not None for panel in panels):
        plot_s_overview(
            panels,
            output_dir / f"step400_search_per_tree_overview.{args.format}",
            args.fixed_y,
        )
        plot_s_grid(
            panels,
            output_dir / f"step400_search_per_tree_per_benchmark.{args.format}",
            args.fixed_y,
        )


if __name__ == "__main__":
    main()
