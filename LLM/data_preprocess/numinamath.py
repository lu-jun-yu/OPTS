import argparse
import os
import random
import re

import datasets
import pandas as pd
from math_verify import parse

from prompts import SYSTEM_PROMPT

# Sources to keep (competition-level)
VALID_SOURCES = {"olympiads", "amc_aime"}

# Default per-source sampling quotas
DEFAULT_QUOTAS = {
    "amc_aime": 2000,
    "olympiads": 3000,
}


def is_answer_parsable(answer: str) -> bool:
    """Check if math-verify can parse this answer for reliable reward verification."""
    try:
        result = parse(str(answer).strip(), parsing_timeout=0)
        return len(result) > 0
    except Exception:
        return False


def normalize_text(text: str) -> str:
    """Normalize text for deduplication comparison."""
    text = str(text).strip().lower()
    text = re.sub(r'\s+', ' ', text)
    return text


def load_test_problems(test_data_dirs: list) -> set:
    """Load problem texts from test parquet files for deduplication."""
    test_problems = set()
    for test_dir in test_data_dirs:
        parquet_path = os.path.join(test_dir, "test.parquet")
        if not os.path.exists(parquet_path):
            print(f"  Warning: {parquet_path} not found, skipping.")
            continue
        df = pd.read_parquet(parquet_path)
        count = 0
        for _, row in df.iterrows():
            prompt = row["prompt"]
            if isinstance(prompt, list) and len(prompt) > 1:
                problem_text = prompt[1]["content"]
            else:
                problem_text = str(prompt)
            test_problems.add(normalize_text(problem_text))
            count += 1
        print(f"  Loaded {count} problems from {parquet_path}")
    return test_problems


def dedup_against_test(dataset, test_problems: set):
    """Remove training examples that overlap with test sets."""
    before = len(dataset)
    dataset = dataset.filter(lambda x: normalize_text(x["problem"]) not in test_problems)
    after = len(dataset)
    print(f"  Test set dedup: {before} -> {after} (removed {before - after})")
    return dataset


def filter_dataset(dataset, max_problem_len: int, max_solution_len: int):
    """Apply rule-based filters to the dataset."""
    before = len(dataset)

    # 1. Only keep valid sources
    dataset = dataset.filter(lambda x: x["source"] in VALID_SOURCES)
    after_source = len(dataset)
    print(f"  Source filter: {before} -> {after_source}")

    # 2. Remove overly long problems
    dataset = dataset.filter(lambda x: len(x["problem"]) <= max_problem_len)
    after_problem = len(dataset)
    print(f"  Problem length filter (<={max_problem_len}): {after_source} -> {after_problem}")

    # 3. Remove overly long solutions (proxy for difficulty)
    dataset = dataset.filter(lambda x: len(x["solution"]) <= max_solution_len)
    after_solution = len(dataset)
    print(f"  Solution length filter (<={max_solution_len}): {after_problem} -> {after_solution}")

    return dataset


def stratified_sample(dataset, quotas: dict, seed: int):
    """Stratified sampling by source."""
    rng = random.Random(seed)
    sampled_indices = []

    for source, quota in quotas.items():
        indices = [i for i, x in enumerate(dataset) if x["source"] == source]
        available = len(indices)
        n = min(quota, available)
        selected = rng.sample(indices, n)
        sampled_indices.extend(selected)
        print(f"  {source}: {available} available, sampled {n}")

    sampled_indices.sort()
    return dataset.select(sampled_indices)


def make_map_fn(split, data_source):
    def process_fn(example, idx):
        problem = example["problem"]
        answer = str(example["answer"]).strip()
        data = {
            "data_source": data_source,
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": problem},
            ],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {
                "split": split,
                "index": idx,
                "source": example.get("source", ""),
            },
        }
        return data

    return process_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess NuminaMath-1.5-RL-Verifiable for RLVR training.")
    parser.add_argument("--local_dataset_path", default=None, help="Local path to the raw dataset.")
    parser.add_argument("--local_save_dir", default="data/numinamath", help="Save directory for processed data.")
    parser.add_argument("--max_problem_len", type=int, default=2000, help="Max problem text length in characters.")
    parser.add_argument("--max_solution_len", type=int, default=3000, help="Max solution text length in characters.")
    parser.add_argument("--quota_amc_aime", type=int, default=4000, help="Sampling quota for amc_aime.")
    parser.add_argument("--quota_olympiads", type=int, default=6000, help="Sampling quota for olympiads.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument(
        "--test_data_dirs", nargs="*",
        default=["data/aime25", "data/amc23", "data/math12k", "data/minervamath"],
        help="Test data directories for deduplication (each should contain test.parquet).",
    )

    args = parser.parse_args()

    data_source = "nlile/NuminaMath-1.5-RL-Verifiable"

    # Load dataset
    print(f"Loading dataset: {data_source}")
    if args.local_dataset_path is not None:
        dataset = datasets.load_dataset(args.local_dataset_path, split="train")
    else:
        dataset = datasets.load_dataset(data_source, split="train")
    print(f"Total samples: {len(dataset)}")

    # Filter
    print("Filtering...")
    dataset = filter_dataset(dataset, args.max_problem_len, args.max_solution_len)
    print(f"After filtering: {len(dataset)}")

    # Dedup against test sets
    if args.test_data_dirs:
        print("Loading test problems for deduplication...")
        test_problems = load_test_problems(args.test_data_dirs)
        print(f"Total unique test problems: {len(test_problems)}")
        print("Deduplicating...")
        dataset = dedup_against_test(dataset, test_problems)

    # Stratified sampling
    quotas = {
        "amc_aime": args.quota_amc_aime,
        "olympiads": args.quota_olympiads,
    }
    print("Stratified sampling...")
    dataset = stratified_sample(dataset, quotas, args.seed)
    print(f"After sampling: {len(dataset)}")

    # Filter unparsable answers (run after sampling so only ~5k calls to math-verify)
    print("Answer parsability filter...")
    before = len(dataset)
    valid_indices = [i for i in range(before) if is_answer_parsable(dataset[i]["answer"])]
    dataset = dataset.select(valid_indices)
    print(f"  {before} -> {len(dataset)} (removed {before - len(dataset)})")

    # Format to training schema
    train_dataset = dataset.map(
        function=make_map_fn("train", data_source),
        with_indices=True,
        remove_columns=dataset.column_names,
    )

    # Save
    os.makedirs(args.local_save_dir, exist_ok=True)
    output_path = os.path.join(args.local_save_dir, "train.parquet")
    train_dataset.to_parquet(output_path)
    print(f"Saved {len(train_dataset)} samples to {output_path}")
