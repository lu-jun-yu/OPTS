# Copyright 2025 Junyu Lu (Julian Lou). All rights reserved.

"""
Offline evaluation for math reasoning benchmarks.

Two modes:
  A) Generate + score (default): run vLLM on test sets, score with math_verify,
     compute unbiased pass@k.
  B) Score pre-generated parquet (--pregenerated_parquet): read a parquet produced
     by verl.trainer.main_generation / main_opts_generation and compute any of
     avg@k / pass@k / cons@k / opts@k. Correctness uses only answer match (no
     format reward); `pass@k` here is the strict "any correct in first k";
     `opts@k` filters by sample_indices <= k and requires a main_opts_generation
     parquet.

Test sets (from dataset_survey.md):
  - math500      500  high-school baseline      (data/math12k/test.parquet)
  - minervamath  272  college-level OOD          (data/minervamath/test.parquet)
  - amc23         40  mid-level competition      (data/amc23/test.parquet)
  - aime25        30  hard competition ceiling   (data/aime25/test.parquet)

Usage:
    # Greedy pass@1
    python -m trainer.main_eval \
        --model_path Qwen/Qwen3-4B \
        --datasets math500 minervamath amc23 aime25

    # pass@k (n=64, temperature=1.0)
    python -m trainer.main_eval \
        --model_path output/rlvr/checkpoint-200 \
        --n 64 \
        --temperature 1.0

    # Score a pre-generated parquet: avg@32, pass@32, cons@32 of first 32 responses
    python -m trainer.main_eval \
        --pregenerated_parquet outputs/step300/gen/dapo_gen128.parquet \
        --metrics avg pass cons --k 32 \
        --output_dir outputs/step300/eval

    # pass@k / opts@k for k in 8 16 32 64 128 on an OPTS generation parquet
    python -m trainer.main_eval \
        --pregenerated_parquet outputs/step300/gen/opts_ttpo_opts_gen128.parquet \
        --metrics pass opts --k 8 16 32 64 128 \
        --output_dir outputs/step300/eval
"""

import argparse
import json
import math
import os
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from math_verify.metric import math_metric
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

    HAS_MATH_VERIFY = True
except ImportError:
    HAS_MATH_VERIFY = False
    print(
        "Warning: math_verify not installed. "
        "Install via `pip install math-verify` for accurate scoring. "
        "Falling back to exact \\boxed{} match."
    )

# Project extractor/validator (reused so scoring semantics match training).
# We only use answer correctness here — format reward is intentionally excluded.
from utils.reward_fn import extract_answer as project_extract_answer
from utils.reward_fn import validate_answer as project_validate_answer

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------
DATASET_PATHS = {
    "math500": "data/math12k/test.parquet",
    "minervamath": "data/minervamath/test.parquet",
    "amc23": "data/amc23/test.parquet",
    "aime25": "data/aime25/test.parquet",
}


# ---------------------------------------------------------------------------
# Scoring (consistent with verl training reward)
# ---------------------------------------------------------------------------
def compute_score(model_output: str, ground_truth: str) -> float:
    """Score a single response against ground truth."""
    if HAS_MATH_VERIFY:
        return _score_math_verify(model_output, ground_truth)
    return _score_exact_match(model_output, ground_truth)


def _score_math_verify(model_output: str, ground_truth: str) -> float:
    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )
    ground_truth_boxed = "\\boxed{" + ground_truth + "}"
    try:
        score, _ = verify_func([ground_truth_boxed], [model_output])
        return float(score)
    except Exception:
        return 0.0


def _score_exact_match(model_output: str, ground_truth: str) -> float:
    pred = _extract_boxed(model_output)
    return 1.0 if pred == ground_truth.strip() else 0.0


def _extract_boxed(text: str) -> str:
    """Extract the last \\boxed{...} from text."""
    matches = []
    i = 0
    while i < len(text):
        pos = text.find("\\boxed{", i)
        if pos == -1:
            break
        depth, j = 0, pos + 7
        while j < len(text):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                if depth == 0:
                    matches.append(text[pos + 7 : j])
                    break
                depth -= 1
            j += 1
        i = j + 1
    return matches[-1].strip() if matches else ""


def strip_thinking(text: str) -> str:
    """Strip <think>...</think> block, return only the final answer part.

    The model outputs: <think>reasoning...</think>\n\nThe answer is \\boxed{...}
    We only score the part after </think> to avoid intermediate \\boxed{} inside thinking.
    """
    idx = text.rfind("</think>")
    if idx != -1:
        return text[idx + len("</think>"):].strip()
    return text


# ---------------------------------------------------------------------------
# pass@k estimator (Codex / Chen et al., 2021)
# ---------------------------------------------------------------------------
def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator: pass@k = 1 - C(n-c, k) / C(n, k)."""
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


# ---------------------------------------------------------------------------
# Answer-only correctness (parquet-mode metrics — format reward is excluded)
# ---------------------------------------------------------------------------
def is_answer_correct(response: str, ground_truth: str) -> bool:
    """True iff the \\boxed{...} after </think> matches ground truth."""
    answer = project_extract_answer(response)
    if answer is None:
        return False
    try:
        return bool(project_validate_answer(answer, ground_truth))
    except Exception:
        return False


def _first_k(seq, k: int):
    return seq[: min(k, len(seq))]


def avg_at_k(correct_flags: list[bool], k: int) -> float:
    """Mean correctness over the first k responses."""
    subset = _first_k(correct_flags, k)
    if not subset:
        return 0.0
    return float(np.mean([1.0 if c else 0.0 for c in subset]))


def strict_pass_at_k(correct_flags: list[bool], k: int) -> float:
    """1 if any of the first k responses is correct, else 0."""
    return 1.0 if any(_first_k(correct_flags, k)) else 0.0


def cons_at_k(responses: list[str], ground_truth: str, k: int) -> float:
    """Self-consistency (majority vote) over the first k responses."""
    subset = _first_k(responses, k)
    answers = [project_extract_answer(r) for r in subset]
    answers = [a for a in answers if a is not None]
    if not answers:
        return 0.0
    majority_ans = Counter(answers).most_common(1)[0][0]
    try:
        return 1.0 if project_validate_answer(majority_ans, ground_truth) else 0.0
    except Exception:
        return 0.0


def opts_at_k_row(correct_flags: list[bool], global_indices, k: int, dataset_size: int) -> float:
    """Per-prompt contribution for opts@k.

    opts@k truncates the full OPTS run (budget = max_n_samples * dataset_size)
    to its chronological prefix of budget = k * dataset_size responses, groups
    them by prompt, and asks: does this prompt have any correct response in
    that prefix? The final opts@k averages this indicator over all prompts
    (so prompts that receive no response within the budget count as 0).
    """
    threshold = k * dataset_size
    return 1.0 if any(c for c, gi in zip(correct_flags, global_indices) if int(gi) <= threshold) else 0.0


# ---------------------------------------------------------------------------
# Parquet-mode evaluator
# ---------------------------------------------------------------------------
def evaluate_pregenerated_parquet(
    parquet_path: str,
    metrics: list[str],
    k_values: list[int],
    response_key: str = "responses",
    reward_model_key: str = "reward_model",
    data_source_key: str = "data_source",
    global_indices_key: str = "global_indices",
) -> dict:
    """Score a generation parquet and aggregate per data_source + global."""
    df = pd.read_parquet(parquet_path)
    dataset_size = len(df)
    has_global = global_indices_key in df.columns
    if "opts" in metrics and not has_global:
        raise ValueError(
            f"opts@k requires column '{global_indices_key}', not found in {parquet_path}. "
            "Regenerate with the updated main_opts_generation.py (writes global_indices)."
        )

    print(
        f"Eval parquet: {parquet_path}  rows={dataset_size}  metrics={metrics}  k={k_values}"
    )

    per_source: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for _, row in tqdm(df.iterrows(), total=dataset_size, desc="scoring"):
        responses = list(row[response_key])
        gt = str(row[reward_model_key]["ground_truth"])
        ds = row[data_source_key] if data_source_key in df.columns else "default"
        correct_flags = [is_answer_correct(r, gt) for r in responses]
        global_indices = list(row[global_indices_key]) if has_global else None

        for k in k_values:
            if "avg" in metrics:
                per_source[ds][f"avg@{k}"].append(avg_at_k(correct_flags, k))
            if "pass" in metrics:
                per_source[ds][f"pass@{k}"].append(strict_pass_at_k(correct_flags, k))
            if "cons" in metrics:
                per_source[ds][f"cons@{k}"].append(cons_at_k(responses, gt, k))
            if "opts" in metrics:
                per_source[ds][f"opts@{k}"].append(
                    opts_at_k_row(correct_flags, global_indices, k, dataset_size)
                )

    summary: dict = {}
    print("=" * 64)
    for ds in sorted(per_source):
        summary[ds] = {}
        for name in sorted(per_source[ds]):
            summary[ds][name] = float(np.mean(per_source[ds][name]))
            print(f"  [{ds}] {name} = {summary[ds][name]:.4f}  (n={len(per_source[ds][name])})")

    flat: dict[str, list[float]] = defaultdict(list)
    for metric_dict in per_source.values():
        for name, vals in metric_dict.items():
            flat[name].extend(vals)
    summary["_all"] = {name: float(np.mean(vals)) for name, vals in flat.items()}
    print("-" * 64)
    for name in sorted(summary["_all"]):
        print(f"  [ALL] {name} = {summary['_all'][name]:.4f}  (n={len(flat[name])})")
    print("=" * 64)

    return summary


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_test_data(dataset_names: list[str], data_root: str) -> dict[str, pd.DataFrame]:
    data = {}
    for name in dataset_names:
        path = os.path.join(data_root, DATASET_PATHS[name])
        if not os.path.exists(path):
            print(f"  [SKIP] {path} not found. Run data_preprocess first.")
            continue
        df = pd.read_parquet(path)
        data[name] = df
        print(f"  {name}: {len(df)} problems")
    return data


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate math reasoning model.")
    parser.add_argument("--model_path", default=None, help="Model path or HuggingFace ID (generation mode).")
    parser.add_argument(
        "--datasets", nargs="+", default=list(DATASET_PATHS.keys()),
        choices=list(DATASET_PATHS.keys()),
        help="Test sets to evaluate on (generation mode).",
    )
    parser.add_argument("--data_root", default=".", help="Root dir for data paths.")
    parser.add_argument("--n", type=int, default=1, help="Samples per problem (>1 for pass@k).")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature (forced 0 when n=1).")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling.")
    parser.add_argument("--max_tokens", type=int, default=4096, help="Max new tokens.")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="TP size for vLLM.")
    parser.add_argument("--output_dir", default="eval/results", help="Output directory.")
    # Parquet-mode (skip vLLM, just score a generated parquet).
    parser.add_argument(
        "--pregenerated_parquet", default=None,
        help="If set, skip vLLM generation and score this parquet (avg@k/pass@k/cons@k/opts@k).",
    )
    parser.add_argument(
        "--metrics", nargs="+", default=["avg", "pass", "cons"],
        choices=["avg", "pass", "cons", "opts"],
        help="Metrics to compute in parquet mode.",
    )
    parser.add_argument(
        "--k", type=int, nargs="+", default=[32],
        help="k values for parquet-mode metrics.",
    )
    parser.add_argument(
        "--output_tag", default=None,
        help="Optional suffix for parquet-mode output JSON filename to avoid overwriting repeated evaluations.",
    )
    args = parser.parse_args()

    # Parquet mode: score a pre-generated parquet and exit.
    if args.pregenerated_parquet is not None:
        summary = evaluate_pregenerated_parquet(
            parquet_path=args.pregenerated_parquet,
            metrics=args.metrics,
            k_values=args.k,
        )
        os.makedirs(args.output_dir, exist_ok=True)
        tag = os.path.splitext(os.path.basename(args.pregenerated_parquet))[0]
        if args.output_tag:
            tag = f"{tag}__{args.output_tag}"
        out_path = os.path.join(args.output_dir, f"{tag}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "data_path": args.pregenerated_parquet,
                "metrics": args.metrics,
                "k": args.k,
                "output_tag": args.output_tag,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "results": summary,
            }, f, ensure_ascii=False, indent=2)
        print(f"Saved summary to {out_path}")
        return

    # Generation mode: keep the existing vLLM pipeline.
    if not args.model_path:
        parser.error("--model_path is required when --pregenerated_parquet is not set.")

    from vllm import LLM, SamplingParams  # lazy import

    # Greedy when n=1
    temperature = 0.0 if args.n == 1 else args.temperature
    top_p = 1.0 if args.n == 1 else args.top_p

    # 1. Load test data
    print("=" * 60)
    print("Loading test datasets...")
    test_data = load_test_data(args.datasets, args.data_root)
    if not test_data:
        print("No test data loaded. Exiting.")
        return

    # 2. Init vLLM
    print("=" * 60)
    print(f"Loading model: {args.model_path}")
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        n=args.n,
        temperature=temperature,
        top_p=top_p,
        max_tokens=args.max_tokens,
    )
    print(f"Sampling: n={args.n}, temperature={temperature}, top_p={top_p}, max_tokens={args.max_tokens}")

    # 3. Evaluate each dataset
    all_metrics = {}
    os.makedirs(args.output_dir, exist_ok=True)

    for ds_name, df in test_data.items():
        print("=" * 60)
        print(f"[{ds_name}] Generating {len(df)} x {args.n} responses...")

        # Build chat prompts
        prompts, ground_truths = [], []
        for _, row in df.iterrows():
            messages = row["prompt"]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=True,
            )
            prompts.append(text)
            ground_truths.append(str(row["reward_model"]["ground_truth"]))

        # Generate
        outputs = llm.generate(prompts, sampling_params)

        # Score
        print(f"[{ds_name}] Scoring...")
        details = []
        for i, output in enumerate(tqdm(outputs, desc="Scoring")):
            gt = ground_truths[i]
            scores = [compute_score(strip_thinking(c.text), gt) for c in output.outputs]
            details.append({
                "index": i,
                "ground_truth": gt,
                "scores": scores,
                "responses": [c.text for c in output.outputs],
            })

        # Compute metrics
        metrics = {"total": len(details)}
        correct_counts = [sum(1 for s in d["scores"] if s > 0) for d in details]

        if args.n == 1:
            correct = sum(correct_counts)
            metrics["accuracy"] = correct / len(details)
            metrics["correct"] = correct
        else:
            # pass@k for k = 1, 2, 4, ..., up to n
            k_values = [k for k in [1, 2, 4, 8, 16, 32, 64, 128, 256] if k <= args.n]
            for k in k_values:
                pk = np.mean([pass_at_k(args.n, c, k) for c in correct_counts])
                metrics[f"pass@{k}"] = float(pk)

        all_metrics[ds_name] = metrics

        # Print
        if args.n == 1:
            print(f"  {ds_name}: accuracy = {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['total']})")
        else:
            for k in sorted(k for k in metrics if isinstance(k, str) and k.startswith("pass@")):
                print(f"  {ds_name}: {k} = {metrics[k]:.4f}")

        # Save per-dataset detail
        detail_path = os.path.join(args.output_dir, f"{ds_name}_n{args.n}.json")
        with open(detail_path, "w", encoding="utf-8") as f:
            json.dump(details, f, ensure_ascii=False, indent=2)

    # 4. Summary
    print("=" * 60)
    print("Summary:")
    print(f"  Model: {args.model_path}")
    print(f"  n={args.n}, temperature={temperature}")
    for ds_name, m in all_metrics.items():
        if "accuracy" in m:
            print(f"  {ds_name}: accuracy = {m['accuracy']:.4f} ({m['correct']}/{m['total']})")
        else:
            keys = sorted([k for k in m if k.startswith("pass@")], key=lambda x: int(x.split("@")[1]))
            vals = "  ".join(f"{k}={m[k]:.4f}" for k in keys)
            print(f"  {ds_name}: {vals}")

    summary = {
        "model": args.model_path,
        "n": args.n,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": args.max_tokens,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": all_metrics,
    }
    summary_path = os.path.join(args.output_dir, f"summary_n{args.n}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {args.output_dir}/")


if __name__ == "__main__":
    main()
