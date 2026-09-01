"""Diagnose the new s=7, Top-K=50 Figure-3 rerun without editing the paper."""

from __future__ import annotations

import gc
import json
import os
import re
from pathlib import Path

import matplotlib
import numpy as np
import pyarrow.parquet as pq

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator, PercentFormatter


os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

REPO_ROOT = Path(__file__).resolve().parents[2]
GEN_DIR = REPO_ROOT / "LLM/results/step400/gen"
EVAL_DIR = REPO_ROOT / "LLM/results/step400/eval"
LOG_DIR = REPO_ROOT / "LLM/logs/step400"
OUTPUT_DIR = REPO_ROOT / "LLM/visual"

METHOD_TAG = "opts_ttpo_exp8_3_0810_n8"
K_VALUES = [8, 16, 32, 64, 128]
NUM_PROMPTS = 902
TOTAL_ROLLOUTS = NUM_PROMPTS * max(K_VALUES)

IID_PARQUET = GEN_DIR / f"{METHOD_TAG}_iid_n128.parquet"
TOKENIZER_PATH = REPO_ROOT / "LLM/models/Qwen3-1.7B"

RUNS = {
    "reward": {
        "paper_parquet": GEN_DIR / f"{METHOD_TAG}_opts_reward_s3_n128.parquet",
        "paper_log": LOG_DIR / f"{METHOD_TAG}_opts_reward_s3.log",
        "new_parquet": GEN_DIR / f"{METHOD_TAG}_opts_reward_s7_topk50_n128.parquet",
        "new_log": LOG_DIR / f"{METHOD_TAG}_opts_reward_s7_topk50.log",
        "iid_eval": EVAL_DIR / f"{METHOD_TAG}_iid_n128__task2_iid_pass_k8-16-32-64-128.json",
        "paper_eval": EVAL_DIR / f"{METHOD_TAG}_opts_reward_s3_n128__task2_reward_opts_k8-16-32-64-128.json",
        "topk50_s3_eval": EVAL_DIR / f"{METHOD_TAG}_opts_reward_s3_topk50_n128__task2_reward_opts_k8-16-32-64-128.json",
        "new_eval": EVAL_DIR / f"{METHOD_TAG}_opts_reward_s7_topk50_n128__task2_reward_opts_k8-16-32-64-128.json",
        "iid_metric": "pass",
        "opts_metric": "opts",
        "ylabel": "pass@k",
        "row_title": "Reward-guided OPTS",
    },
    "value": {
        "paper_parquet": GEN_DIR / f"{METHOD_TAG}_opts_value_s3_n128.parquet",
        "paper_log": LOG_DIR / f"{METHOD_TAG}_opts_value_s3.log",
        "new_parquet": GEN_DIR / f"{METHOD_TAG}_opts_value_s7_topk50_n128.parquet",
        "new_log": LOG_DIR / f"{METHOD_TAG}_opts_value_s7_topk50.log",
        "iid_eval": EVAL_DIR / f"{METHOD_TAG}_iid_n128__task3_iid_cons_k8-16-32-64-128.json",
        "paper_eval": EVAL_DIR / f"{METHOD_TAG}_opts_value_s3_n128__task3_value_opts_k8-16-32-64-128.json",
        "topk50_s3_eval": EVAL_DIR / f"{METHOD_TAG}_opts_value_s3_topk50_n128__task3_value_opts_k8-16-32-64-128.json",
        "new_eval": EVAL_DIR / f"{METHOD_TAG}_opts_value_s7_topk50_n128__task3_value_opts_k8-16-32-64-128.json",
        "iid_metric": "cons",
        "opts_metric": "opts",
        "ylabel": "cons@k",
        "row_title": "Value-guided OPTS",
    },
}

PDF_OUTPUT = OUTPUT_DIR / "rq2_topk50_s7_diagnostics.pdf"
PNG_OUTPUT = OUTPUT_DIR / "rq2_topk50_s7_diagnostics.png"
JSON_OUTPUT = OUTPUT_DIR / "rq2_topk50_s7_diagnostics.json"

TEXT = "#253238"
MUTED = "#64717A"
GRID = "#D8DEE2"
IID = "#909AA2"
PAPER = "#4E718C"
TOPK50_S3 = "#D19545"
NEW = "#B5475D"


def _list_array(table, name, path):
    array = table[name].combine_chunks()
    if array.null_count:
        raise ValueError(f"Null outer list in {name}: {path}")
    return array


def _outer_offsets(array):
    return array.offsets.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)


def parse_selected_counts(path: Path) -> list[int]:
    pattern = re.compile(r"\[select_next_states\].*?\bselected=(\d+)\b")
    counts = [int(match.group(1)) for match in pattern.finditer(path.read_text(errors="ignore"))]
    if len(counts) != max(K_VALUES) - 1:
        raise ValueError(f"Expected 127 selection counts in {path}, found {len(counts)}")
    return counts


def load_tree_diagnostics(parquet_path: Path, log_path: Path, label: str):
    columns = [
        "global_indices",
        "tree_rids",
        "tree_pids",
        "tree_branch_pos",
        "tree_advantages",
    ]
    print(f"[{label}] reading tree metadata", flush=True)
    table = pq.read_table(parquet_path, columns=columns)
    arrays = {name: _list_array(table, name, parquet_path) for name in columns}
    offsets = _outer_offsets(arrays["global_indices"])
    for name in columns[1:]:
        if not np.array_equal(_outer_offsets(arrays[name]), offsets):
            raise ValueError(f"Outer offsets differ for {name}: {parquet_path}")

    global_indices = np.asarray(arrays["global_indices"].values.to_pylist(), dtype=np.int64)
    rids = arrays["tree_rids"].values.to_pylist()
    pids = arrays["tree_pids"].values.to_pylist()
    branch_pos = np.asarray(arrays["tree_branch_pos"].values.to_pylist(), dtype=np.int64)
    advantages = arrays["tree_advantages"].values
    advantage_lengths = np.diff(
        advantages.offsets.to_numpy(zero_copy_only=False)
    ).astype(np.int64, copy=False)

    lengths = [len(global_indices), len(rids), len(pids), len(branch_pos), len(advantage_lengths)]
    if lengths != [TOTAL_ROLLOUTS] * len(lengths):
        raise ValueError(f"Expected {TOTAL_ROLLOUTS} aligned rollouts, found {lengths}")
    if len(set(rids)) != TOTAL_ROLLOUTS:
        raise ValueError(f"RIDs are not unique: {parquet_path}")
    if not np.array_equal(
        np.sort(global_indices), np.arange(1, TOTAL_ROLLOUTS + 1, dtype=np.int64)
    ):
        raise ValueError(f"Global indices are incomplete: {parquet_path}")

    rid_to_index = {rid: index for index, rid in enumerate(rids)}
    retained_prefix = np.zeros(TOTAL_ROLLOUTS, dtype=np.int64)
    for index in np.argsort(global_indices):
        pid = pids[index]
        if pid is None:
            if branch_pos[index] != -1:
                raise ValueError(f"Root with branch_pos={branch_pos[index]}: {rids[index]}")
            continue
        if pid not in rid_to_index:
            raise ValueError(f"Missing parent {pid}: {parquet_path}")
        parent_index = rid_to_index[pid]
        if global_indices[parent_index] >= global_indices[index]:
            raise ValueError(f"Parent appears after child {rids[index]}")
        if not 0 <= branch_pos[index] < advantage_lengths[parent_index]:
            raise ValueError(f"Invalid branch position for {rids[index]}")
        retained_prefix[index] = retained_prefix[parent_index] + branch_pos[index] + 1

    selected_counts = parse_selected_counts(log_path)
    rid_pattern = re.compile(r"^r(\d+)_(\d+)$")
    rounds = np.empty(TOTAL_ROLLOUTS, dtype=np.int64)
    local_indices = np.empty(TOTAL_ROLLOUTS, dtype=np.int64)
    for index, rid in enumerate(rids):
        match = rid_pattern.fullmatch(rid)
        if match is None:
            raise ValueError(f"Unexpected RID {rid}: {parquet_path}")
        rounds[index], local_indices[index] = map(int, match.groups())

    continuation = np.zeros(TOTAL_ROLLOUTS, dtype=bool)
    for round_index in range(1, max(K_VALUES)):
        round_mask = rounds == round_index
        continuation[round_mask] = local_indices[round_mask] < selected_counts[round_index - 1]
        found = int(continuation[round_mask].sum())
        if found != selected_counts[round_index - 1]:
            raise ValueError(
                f"Round {round_index}: recovered {found} continuations, "
                f"expected {selected_counts[round_index - 1]}"
            )

    expected_selections = int(sum(selected_counts))
    if int(continuation.sum()) != expected_selections:
        raise ValueError(f"Recovered continuation count differs in {parquet_path}")

    prompt_root = continuation & np.equal(pids, None)
    explicit_child = continuation & np.not_equal(pids, None)
    if int(explicit_child.sum()) != int(np.not_equal(pids, None).sum()):
        raise ValueError(f"Found non-continuation child in {parquet_path}")

    normalized_positions = np.zeros(expected_selections, dtype=np.float64)
    child_indices = np.flatnonzero(explicit_child)
    parent_indices = np.asarray([rid_to_index[pids[index]] for index in child_indices])
    parent_path_lengths = retained_prefix[parent_indices] + advantage_lengths[parent_indices]
    child_positions = retained_prefix[child_indices] / parent_path_lengths
    if np.any((child_positions <= 0) | (child_positions > 1)):
        raise ValueError(f"Normalized branch position outside (0,1]: {parquet_path}")
    normalized_positions[int(prompt_root.sum()) :] = child_positions

    token_cost = []
    for k_value in K_VALUES:
        selected = global_indices <= NUM_PROMPTS * k_value
        if int(selected.sum()) != NUM_PROMPTS * k_value:
            raise ValueError(f"Wrong rollout count at k={k_value}: {parquet_path}")
        token_cost.append(float(advantage_lengths[selected].sum() / NUM_PROMPTS))

    nonroot = normalized_positions[normalized_positions > 0]
    summary = {
        "selections": expected_selections,
        "prompt_root_selections": int(prompt_root.sum()),
        "prompt_root_fraction": float(prompt_root.sum() / expected_selections),
        "explicit_child_selections": int(explicit_child.sum()),
        "normalized_position_nonroot": {
            "mean": float(nonroot.mean()),
            "p25": float(np.quantile(nonroot, 0.25)),
            "median": float(np.median(nonroot)),
            "p75": float(np.quantile(nonroot, 0.75)),
        },
        "retained_prefix_tokens_nonroot": {
            "mean": float(retained_prefix[child_indices].mean()),
            "median": float(np.median(retained_prefix[child_indices])),
            "p90": float(np.quantile(retained_prefix[child_indices], 0.90)),
        },
        "tokens_per_prompt": dict(zip(map(str, K_VALUES), token_cost)),
    }

    del table, arrays, advantages, rid_to_index, rounds, local_indices
    gc.collect()
    return normalized_positions, token_cost, summary


def load_score_series(path: Path, metric: str) -> list[float]:
    data = json.loads(path.read_text())
    if data.get("k") != K_VALUES or data.get("metrics") != [metric]:
        raise ValueError(f"Unexpected evaluation schema: {path}")
    pooled = data["results"]["_all"]
    return [float(pooled[f"{metric}@{k_value}"]) for k_value in K_VALUES]


def load_iid_token_cost(tokenizer) -> list[float]:
    parquet_file = pq.ParquetFile(IID_PARQUET)
    if parquet_file.metadata.num_rows != NUM_PROMPTS:
        raise ValueError(f"Expected {NUM_PROMPTS} IID rows")

    totals = np.zeros(len(K_VALUES), dtype=np.float64)
    rows_seen = 0
    backend = tokenizer.backend_tokenizer
    print("[IID] tokenizing decoded responses", flush=True)
    for batch in parquet_file.iter_batches(batch_size=16, columns=["responses"]):
        response_lists = batch.column(0).to_pylist()
        if any(len(responses) != max(K_VALUES) for responses in response_lists):
            raise ValueError("IID rows do not contain exactly 128 responses")
        texts = [response for responses in response_lists for response in responses]
        encodings = backend.encode_batch(texts, add_special_tokens=False)
        lengths = np.asarray([len(encoding.ids) for encoding in encodings], dtype=np.int64)
        lengths = lengths.reshape(len(response_lists), max(K_VALUES))
        cumulative = np.cumsum(lengths, axis=1)
        for index, k_value in enumerate(K_VALUES):
            totals[index] += cumulative[:, k_value - 1].sum()
        rows_seen += len(response_lists)
        print(f"[IID] {rows_seen}/{NUM_PROMPTS} prompts", flush=True)

    if rows_seen != NUM_PROMPTS:
        raise ValueError(f"Expected {NUM_PROMPTS} IID rows, found {rows_seen}")
    return (totals / NUM_PROMPTS).tolist()


def ecdf(values):
    x = np.sort(values)
    y = np.arange(1, len(x) + 1, dtype=np.float64) / len(x)
    return x, y


def style_axis(axis):
    axis.grid(axis="y", color=GRID, linewidth=0.65, alpha=0.78)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#A8B0B6")
    axis.spines["bottom"].set_color("#A8B0B6")
    axis.tick_params(labelsize=8.8, colors=MUTED, length=3, width=0.65, pad=2.5)


def plot_branch_panel(axis, paper_positions, new_positions, paper_summary, new_summary):
    for values, color, linestyle, linewidth in (
        (paper_positions, PAPER, (0, (4, 2.5)), 1.8),
        (new_positions, NEW, "-", 2.25),
    ):
        x, y = ecdf(values)
        axis.step(x, y, where="post", color=color, linestyle=linestyle, linewidth=linewidth)
    style_axis(axis)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1.015)
    axis.xaxis.set_major_locator(MaxNLocator(5))
    axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axis.set_xlabel("Normalized branch position", fontsize=9.5, labelpad=4)
    axis.set_ylabel("Cumulative fraction", fontsize=9.5, labelpad=5)
    axis.text(
        0.97,
        0.08,
        "Prompt-root rebranch\n"
        f"paper: {paper_summary['prompt_root_fraction']:.1%}\n"
        f"new: {new_summary['prompt_root_fraction']:.1%}",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.0,
        color=TEXT,
        bbox={"boxstyle": "round,pad=0.28", "fc": "white", "ec": "#CCD3D8", "lw": 0.6},
    )


def plot_budget_panel(axis, iid, paper, new, ylabel, token_axis=False):
    x = np.arange(len(K_VALUES))
    axis.plot(
        x,
        iid,
        color=IID,
        linestyle=(0, (4, 2.5)),
        marker="o",
        markersize=4.5,
        markerfacecolor="white",
        markeredgewidth=1.0,
        linewidth=1.6,
    )
    axis.plot(
        x,
        paper,
        color=PAPER,
        linestyle=(0, (4, 2.5)),
        marker="s",
        markersize=4.4,
        markerfacecolor="white",
        markeredgewidth=1.0,
        linewidth=1.75,
    )
    axis.plot(
        x,
        new,
        color=NEW,
        marker="D",
        markersize=4.7,
        markeredgecolor="white",
        markeredgewidth=0.6,
        linewidth=2.2,
    )
    style_axis(axis)
    axis.set_xticks(x)
    axis.set_xticklabels(K_VALUES)
    axis.set_xlabel(r"Rollout budget $k$", fontsize=9.5, labelpad=4)
    axis.set_ylabel(ylabel, fontsize=9.5, labelpad=5)
    if token_axis:
        axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000:.0f}k"))
        delta = 100 * (new[-1] / paper[-1] - 1)
        axis.text(
            0.04,
            0.92,
            f"new vs paper @128: {delta:+.1f}%",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=8.1,
            color=NEW,
            fontweight="semibold",
        )
    else:
        axis.yaxis.set_major_locator(MaxNLocator(5))


def plot_score_panel(axis, iid, paper, topk50_s3, new, ylabel):
    plot_budget_panel(axis, iid, paper, new, ylabel, token_axis=False)
    x = np.arange(len(K_VALUES))
    axis.plot(
        x,
        topk50_s3,
        color=TOPK50_S3,
        linestyle=(0, (1.5, 2.0)),
        marker="^",
        markersize=4.3,
        markerfacecolor="white",
        markeredgewidth=0.9,
        linewidth=1.55,
        zorder=2,
    )
    delta_paper = 100 * (new[-1] - paper[-1])
    delta_topk50 = 100 * (new[-1] - topk50_s3[-1])
    axis.text(
        0.04,
        0.92,
        f"@128: {delta_paper:+.2f} pp vs paper\n"
        f"          {delta_topk50:+.2f} pp vs s=3 Top-K=50",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=7.8,
        color=NEW,
        fontweight="semibold",
    )


def make_figure(results):
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9.5,
            "text.color": TEXT,
            "axes.labelcolor": TEXT,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(2, 3, figsize=(11.6, 5.75))
    figure.subplots_adjust(left=0.075, right=0.992, bottom=0.205, top=0.84, hspace=0.55, wspace=0.31)

    for row, mode in enumerate(("reward", "value")):
        result = results[mode]
        plot_branch_panel(
            axes[row, 0],
            result["paper_positions"],
            result["new_positions"],
            result["paper_branch"],
            result["new_branch"],
        )
        plot_budget_panel(
            axes[row, 1],
            results["iid_tokens"],
            result["paper_tokens"],
            result["new_tokens"],
            "Generated tokens / prompt",
            token_axis=True,
        )
        plot_score_panel(
            axes[row, 2],
            result["iid_score"],
            result["paper_score"],
            result["topk50_s3_score"],
            result["new_score"],
            RUNS[mode]["ylabel"],
        )
        axes[row, 0].text(
            -0.30,
            0.50,
            RUNS[mode]["row_title"],
            transform=axes[row, 0].transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontsize=11.0,
            fontweight="semibold",
            color=TEXT,
        )

    for column, title in enumerate(("Branch-position distribution", "Cumulative token cost", "Pooled performance")):
        axes[0, column].set_title(title, fontsize=11.3, fontweight="semibold", pad=8, color=TEXT)

    handles = [
        Line2D([0], [0], color=IID, linestyle=(0, (4, 2.5)), marker="o", markerfacecolor="white", label="IID (Top-K=50)"),
        Line2D([0], [0], color=PAPER, linestyle=(0, (4, 2.5)), marker="s", markerfacecolor="white", label="Paper: OPTS s=3, unrestricted"),
        Line2D([0], [0], color=TOPK50_S3, linestyle=(0, (1.5, 2.0)), marker="^", markerfacecolor="white", label="OPTS s=3, Top-K=50 (score only)"),
        Line2D([0], [0], color=NEW, marker="D", markeredgecolor="white", label="New: OPTS s=7, Top-K=50"),
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.535, 0.965),
        ncol=4,
        frameon=False,
        fontsize=8.7,
        handlelength=2.5,
        columnspacing=1.5,
    )
    figure.text(
        0.535,
        0.055,
        "Diagnostic comparison only: the new run changes both the per-tree search budget "
        "(s=3 to s=7) and sampling (unrestricted to Top-K=50).",
        ha="center",
        va="center",
        fontsize=8.4,
        color=MUTED,
    )
    return figure


def main():
    from transformers import AutoTokenizer

    results = {}
    for mode, config in RUNS.items():
        paper_positions, paper_tokens, paper_branch = load_tree_diagnostics(
            config["paper_parquet"], config["paper_log"], f"{mode} paper"
        )
        new_positions, new_tokens, new_branch = load_tree_diagnostics(
            config["new_parquet"], config["new_log"], f"{mode} new"
        )
        iid_score = load_score_series(config["iid_eval"], config["iid_metric"])
        paper_score = load_score_series(config["paper_eval"], config["opts_metric"])
        topk50_s3_score = load_score_series(config["topk50_s3_eval"], config["opts_metric"])
        new_score = load_score_series(config["new_eval"], config["opts_metric"])
        results[mode] = {
            "paper_positions": paper_positions,
            "new_positions": new_positions,
            "paper_tokens": paper_tokens,
            "new_tokens": new_tokens,
            "paper_branch": paper_branch,
            "new_branch": new_branch,
            "iid_score": iid_score,
            "paper_score": paper_score,
            "topk50_s3_score": topk50_s3_score,
            "new_score": new_score,
            "score_delta_pp_new_minus_paper": [100 * (new - old) for new, old in zip(new_score, paper_score)],
            "score_delta_pp_new_minus_s3_topk50": [100 * (new - old) for new, old in zip(new_score, topk50_s3_score)],
            "token_change_pct_new_minus_paper": [100 * (new / old - 1) for new, old in zip(new_tokens, paper_tokens)],
        }

    print(f"Loading tokenizer from {TOKENIZER_PATH}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_PATH,
        trust_remote_code=True,
        local_files_only=True,
    )
    results["iid_tokens"] = load_iid_token_cost(tokenizer)

    figure = make_figure(results)
    PDF_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(PDF_OUTPUT, bbox_inches="tight", pad_inches=0.04, metadata={"Title": "Top-K=50 s=7 diagnostics"})
    figure.savefig(PNG_OUTPUT, dpi=260, bbox_inches="tight", pad_inches=0.04)
    plt.close(figure)

    serializable = {
        "configurations": {
            "iid": {"top_k": 50},
            "paper": {"max_search_per_tree": 3, "top_k": -1},
            "score_only_s3_topk50": {"max_search_per_tree": 3, "top_k": 50},
            "new": {"max_search_per_tree": 7, "top_k": 50},
        },
        "k": K_VALUES,
        "iid_tokens_per_prompt": results["iid_tokens"],
    }
    for mode in ("reward", "value"):
        serializable[mode] = {key: value for key, value in results[mode].items() if not key.endswith("_positions")}
    JSON_OUTPUT.write_text(json.dumps(serializable, indent=2) + "\n")

    for mode in ("reward", "value"):
        result = serializable[mode]
        print(f"[{mode}] new-vs-paper score delta (pp): {[round(x, 3) for x in result['score_delta_pp_new_minus_paper']]}")
        print(f"[{mode}] new-vs-paper token delta (%): {[round(x, 2) for x in result['token_change_pct_new_minus_paper']]}")
        print(f"[{mode}] new branch summary: {result['new_branch']}")
    print(f"Wrote {PDF_OUTPUT}")
    print(f"Wrote {PNG_OUTPUT}")
    print(f"Wrote {JSON_OUTPUT}")


if __name__ == "__main__":
    main()
