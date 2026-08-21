import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVAL_DIR = REPO_ROOT / "LLM/results/step400/eval"
DEFAULT_TASK2_IID = DEFAULT_EVAL_DIR / "opts_ttpo_exp8_3_0810_n8_iid_n128__task2_iid_pass_k8-16-32-64-128.json"
DEFAULT_TASK2_OPTS = DEFAULT_EVAL_DIR / "opts_ttpo_opts_reward_n128__task2_reward_opts_k8-16-32-64-128.json"
DEFAULT_TASK3_IID = DEFAULT_EVAL_DIR / "opts_ttpo_exp8_3_0810_n8_iid_n128__task3_iid_cons_k8-16-32-64-128.json"
DEFAULT_TASK3_OPTS = DEFAULT_EVAL_DIR / "opts_ttpo_opts_value_n128__task3_value_opts_k8-16-32-64-128.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "paper/figures"

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


def score_series(eval_data, dataset):
    metric = eval_data["metric"]
    result = eval_data["results"].get(dataset, {})
    values = []
    for k_value in eval_data["k"]:
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
        raise ValueError(
            f"k values differ: {first_eval['path']} has {first_eval['k']}, "
            f"{second_eval['path']} has {second_eval['k']}"
        )

    datasets = common_datasets(first_eval, second_eval)
    if len(datasets) != len(DATASET_ORDER):
        raise ValueError(
            f"Expected {len(DATASET_ORDER)} datasets, found {len(datasets)}: {datasets}"
        )

    k_values = first_eval["k"]
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
        first_values = score_series(first_eval, dataset)
        second_values = score_series(second_eval, dataset)
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


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)

    task2_iid = load_eval(args.task2_iid)
    task2_opts = load_eval(args.task2_opts)
    plot_pair(
        task2_iid,
        task2_opts,
        "IID sampling",
        "Reward-guided OPTS",
        output_dir / f"step400_task2_iid_pass_vs_reward_opts.{args.format}",
        args.fixed_y,
    )

    task3_iid_path = Path(args.task3_iid)
    task3_opts_path = Path(args.task3_opts)
    if task3_iid_path.is_file() and task3_opts_path.is_file():
        plot_pair(
            load_eval(task3_iid_path),
            load_eval(task3_opts_path),
            "IID cons",
            "Value OPTS",
            output_dir / f"step400_task3_iid_cons_vs_value_opts.{args.format}",
            args.fixed_y,
        )
    else:
        print(
            "Skipped task 3: both value-OPTS and IID-consistency JSON files "
            "are required."
        )


if __name__ == "__main__":
    main()
