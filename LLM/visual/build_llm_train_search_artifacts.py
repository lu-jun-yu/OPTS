import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


METHOD_NAME_MAP = {
    "ppo_0516_1.7B": "PPO",
    "dapo_0516_1.7B": "DAPO",
    "gpg_0516_1.7B": "GPG",
    "reinforce_pp_baseline_0516_1.7B": "REINFORCE++",
    "opts_ttpo_lam999_0515_1.7B": "OPTS-TTPO",
}

METHOD_ORDER = ["PPO", "DAPO", "GPG", "REINFORCE++", "OPTS-TTPO"]

METHOD_STYLE = {
    "PPO": {"color": "#4C78A8", "linewidth": 2.0, "alpha": 0.95},
    "DAPO": {"color": "#72B7B2", "linewidth": 2.0, "alpha": 0.95},
    "GPG": {"color": "#54A24B", "linewidth": 2.0, "alpha": 0.95},
    "REINFORCE++": {"color": "#F58518", "linewidth": 2.0, "alpha": 0.95},
    "OPTS-TTPO": {"color": "#E45756", "linewidth": 2.6, "alpha": 1.0},
}

BENCHMARK_NAME_MAP = {
    "val-core/math-ai/minervamath": "MinervaMath",
    "val-core/math-ai/amc23": "AMC23",
    "val-core/math-ai/aime25": "AIME25",
    "val-core/hiyouga/math12k": "Math12k",
}

BENCHMARK_ORDER = [
    "val-core/math-ai/minervamath",
    "val-core/math-ai/amc23",
    "val-core/math-ai/aime25",
    "val-core/hiyouga/math12k",
]

SUMMARY_METRICS = ["acc/avg@32", "acc/pass@32", "acc/cons@32"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build reward-curve figures and step-460 summaries for LLM training-time search."
    )
    parser.add_argument(
        "--results-dir",
        default="LLM/results",
        help="Directory containing the exported W&B CSV files.",
    )
    parser.add_argument(
        "--summary-step",
        type=int,
        default=460,
        help="Checkpoint step used for the final-result summary.",
    )
    parser.add_argument(
        "--summary-csv",
        default="LLM/visual/step460_summary.csv",
        help="Output CSV path for the final-step summary.",
    )
    parser.add_argument(
        "--output-image",
        default="paper/figures/llm_reward_mean32.png",
        help="Output image path for the 1x4 reward-mean curves.",
    )
    return parser.parse_args()


def parse_metric_header(header):
    _, metric_spec = header.split(" - ", 1)
    parts = metric_spec.split("/")
    benchmark = "/".join(parts[:3])
    metric = "/".join(parts[3:])
    return benchmark, metric


def parse_method_name(header):
    raw_name = header.split(" - ", 1)[0].replace("Name: ", "")
    return METHOD_NAME_MAP.get(raw_name, raw_name)


def load_results(results_dir):
    curves = defaultdict(lambda: defaultdict(dict))
    summary = defaultdict(lambda: defaultdict(dict))

    for csv_path in sorted(Path(results_dir).glob("*.csv")):
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            value_headers = [
                name
                for name in fieldnames
                if name != "Step" and not name.endswith("__MIN") and not name.endswith("__MAX")
            ]
            if not value_headers:
                continue

            benchmark, metric = parse_metric_header(value_headers[0])
            for row in reader:
                step = int(row["Step"])
                for header in value_headers:
                    method = parse_method_name(header)
                    value_str = row[header]
                    if value_str == "":
                        continue
                    value = float(value_str)
                    curves[benchmark][metric].setdefault(method, []).append((step, value))
                    summary[benchmark][metric][step] = summary[benchmark][metric].get(step, {})
                    summary[benchmark][metric][step][method] = value

    return curves, summary


def write_summary_csv(summary, summary_step, output_path):
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["benchmark", "method", "avg@32", "pass@32", "cons@32"])
        for benchmark in BENCHMARK_ORDER:
            if benchmark not in summary:
                continue
            for method in METHOD_ORDER:
                row = [BENCHMARK_NAME_MAP[benchmark], method]
                for metric in SUMMARY_METRICS:
                    metric_values = summary[benchmark].get(metric, {}).get(summary_step, {})
                    row.append(metric_values.get(method, ""))
                writer.writerow(row)


def plot_reward_curves(curves, output_path):
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 4, figsize=(16.0, 3.8), sharex=True)
    legend_handles = {}

    for axis, benchmark in zip(axes, BENCHMARK_ORDER):
        axis.set_title(BENCHMARK_NAME_MAP[benchmark], fontsize=11, pad=8)
        axis.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
        axis.set_xlabel("Training step")
        if axis is axes[0]:
            axis.set_ylabel("Reward mean@32")

        reward_curves = curves.get(benchmark, {}).get("reward/mean@32", {})
        for method in METHOD_ORDER:
            if method not in reward_curves:
                continue
            steps, values = zip(*reward_curves[method])
            style = METHOD_STYLE[method]
            line, = axis.plot(
                steps,
                values,
                label=method,
                color=style["color"],
                linewidth=style["linewidth"],
                alpha=style["alpha"],
            )
            axis.scatter(
                steps[-1],
                values[-1],
                color=style["color"],
                s=18 if method == "OPTS-TTPO" else 14,
                zorder=3,
            )
            legend_handles[method] = line

    fig.legend(
        [legend_handles[method] for method in METHOD_ORDER if method in legend_handles],
        [method for method in METHOD_ORDER if method in legend_handles],
        loc="upper center",
        ncol=len(legend_handles),
        frameon=False,
        bbox_to_anchor=(0.5, 1.08),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95), w_pad=1.4)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    curves, summary = load_results(args.results_dir)
    write_summary_csv(summary, args.summary_step, args.summary_csv)
    plot_reward_curves(curves, args.output_image)
    print(f"Wrote summary CSV to {Path(args.summary_csv).resolve()}")
    print(f"Wrote reward figure to {Path(args.output_image).resolve()}")


if __name__ == "__main__":
    main()
