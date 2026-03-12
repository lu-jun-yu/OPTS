"""
Offline evaluation for math reasoning benchmarks.

Generates responses with vLLM, scores with math_verify, computes pass@k.

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
"""

import argparse
import json
import math
import os
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm
from vllm import LLM, SamplingParams

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
    parser.add_argument("--model_path", required=True, help="Model path or HuggingFace ID.")
    parser.add_argument(
        "--datasets", nargs="+", default=list(DATASET_PATHS.keys()),
        choices=list(DATASET_PATHS.keys()),
        help="Test sets to evaluate on.",
    )
    parser.add_argument("--data_root", default=".", help="Root dir for data paths.")
    parser.add_argument("--n", type=int, default=1, help="Samples per problem (>1 for pass@k).")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature (forced 0 when n=1).")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling.")
    parser.add_argument("--max_tokens", type=int, default=4096, help="Max new tokens.")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="TP size for vLLM.")
    parser.add_argument("--output_dir", default="eval/results", help="Output directory.")
    args = parser.parse_args()

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
