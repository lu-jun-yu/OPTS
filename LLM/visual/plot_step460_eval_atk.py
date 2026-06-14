import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


DEFAULT_TASK2_IID = "LLM/outputs/step460/eval/opts_ttpo_iid_n128__task2_iid_pass_k8-16-32-64-128.json"
DEFAULT_TASK2_OPTS = "LLM/outputs/step460/eval/opts_ttpo_opts_reward_n128__task2_reward_opts_k8-16-32-64-128.json"
DEFAULT_TASK3_IID = "LLM/outputs/step460/eval/opts_ttpo_iid_n128__task3_iid_cons_k8-16-32-64-128.json"
DEFAULT_TASK3_OPTS = "LLM/outputs/step460/eval/opts_ttpo_opts_value_n128__task3_value_opts_k8-16-32-64-128.json"

DATASET_NAME_MAP = {
    "hiyouga/math12k": "Math12k",
    "math-ai/aime25": "AIME25",
    "math-ai/amc23": "AMC23",
    "math-ai/minervamath": "MinervaMath",
}

DATASET_ORDER = [
    "hiyouga/math12k",
    "math-ai/aime25",
    "math-ai/amc23",
    "math-ai/minervamath",
]

LINE_STYLES = [
    {
        "color": "#4C78A8",
        "marker": "o",
        "linewidth": 2.2,
        "markersize": 5,
    },
    {
        "color": "#E45756",
        "marker": "s",
        "linewidth": 2.2,
        "markersize": 5,
    },
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot step-460 eval scores at k as two 1x4 figures."
    )
    parser.add_argument("--task2-iid", default=DEFAULT_TASK2_IID)
    parser.add_argument("--task2-opts", default=DEFAULT_TASK2_OPTS)
    parser.add_argument("--task3-iid", default=DEFAULT_TASK3_IID)
    parser.add_argument("--task3-opts", default=DEFAULT_TASK3_OPTS)
    parser.add_argument(
        "--output-dir",
        default="LLM/visual",
        help="Directory for output PNG/PDF files.",
    )
    parser.add_argument(
        "--format",
        default="png",
        choices=["png", "pdf", "svg"],
        help="Output image format.",
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


def plot_pair(first_eval, second_eval, first_label, second_label, title, output_path, fixed_y):
    if first_eval["k"] != second_eval["k"]:
        raise ValueError(
            f"k values differ: {first_eval['path']} has {first_eval['k']}, "
            f"{second_eval['path']} has {second_eval['k']}"
        )

    datasets = common_datasets(first_eval, second_eval)
    if len(datasets) != 4:
        raise ValueError(f"Expected 4 datasets, found {len(datasets)}: {datasets}")

    k_values = first_eval["k"]
    x_values = list(range(len(k_values)))

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 3.8), sharex=True)
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

        axis.set_title(DATASET_NAME_MAP.get(dataset, dataset), fontsize=11, pad=8)
        axis.set_xticks(x_values)
        axis.set_xticklabels([f"@{k_value}" for k_value in k_values])
        axis.set_xlabel("@k")
        axis.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
        if fixed_y:
            axis.set_ylim(0.0, 1.0)
        else:
            set_dynamic_ylim(axis, all_values)

    axes[0].set_ylabel("Score")
    fig.suptitle(title, fontsize=13, y=1.03)
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88), w_pad=1.4)

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
    task3_iid = load_eval(args.task3_iid)
    task3_opts = load_eval(args.task3_opts)

    plot_pair(
        task2_iid,
        task2_opts,
        "IID pass",
        "Reward OPTS",
        "Task 2: IID pass vs reward OPTS",
        output_dir / f"step460_task2_iid_pass_vs_reward_opts.{args.format}",
        args.fixed_y,
    )
    plot_pair(
        task3_iid,
        task3_opts,
        "IID cons",
        "Value OPTS",
        "Task 3: IID cons vs value OPTS",
        output_dir / f"step460_task3_iid_cons_vs_value_opts.{args.format}",
        args.fixed_y,
    )


if __name__ == "__main__":
    main()
