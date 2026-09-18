import argparse
import os
import re

import datasets
from transformers import AutoTokenizer


def norm_problem(row):
    problem = next(m["content"] for m in row["prompt"] if m["role"] == "user")
    return re.sub(r"\s+", "", problem).lower()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge Math12K and NuminaMath training datasets.")
    parser.add_argument(
        "--math12k_path",
        default="data/math12k/train.parquet",
        help="Path to Math12K train.parquet file.",
    )
    parser.add_argument(
        "--numinamath_path",
        default="data/numinamath/train.parquet",
        help="Path to NuminaMath train.parquet file.",
    )
    parser.add_argument(
        "--output_path",
        default="data/train.parquet",
        help="Output path for the merged dataset.",
    )
    parser.add_argument(
        "--test_path",
        default="data/test.parquet",
        help="Path to the test set used for leak removal.",
    )
    parser.add_argument(
        "--model_path",
        default="models/Qwen3-1.7B",
        help="Tokenizer path for the prompt length filter.",
    )
    parser.add_argument(
        "--max_prompt_length",
        type=int,
        default=1024,
        help="Drop prompts longer than this after chat templating.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        default=True,
        help="Whether to shuffle the merged dataset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling.",
    )

    args = parser.parse_args()

    # Expand paths
    math12k_path = os.path.expanduser(args.math12k_path)
    numinamath_path = os.path.expanduser(args.numinamath_path)
    output_path = os.path.expanduser(args.output_path)

    # Read datasets
    print(f"Reading Math12K from: {math12k_path}")
    math12k_ds = datasets.Dataset.from_parquet(math12k_path)
    print(f"  Math12K samples: {len(math12k_ds)}")

    print(f"Reading NuminaMath from: {numinamath_path}")
    numinamath_ds = datasets.Dataset.from_parquet(numinamath_path)
    print(f"  NuminaMath samples: {len(numinamath_ds)}")

    # Merge datasets
    merged_ds = datasets.concatenate_datasets([math12k_ds, numinamath_ds])
    print(f"Merged samples: {len(merged_ds)}")

    test_keys = {norm_problem(row) for row in datasets.Dataset.from_parquet(os.path.expanduser(args.test_path))}
    tokenizer = AutoTokenizer.from_pretrained(os.path.expanduser(args.model_path))
    seen = set()
    keep = []
    n_dup, n_leak, n_long = 0, 0, 0
    for i, row in enumerate(merged_ds):
        key = norm_problem(row)
        if key in test_keys:
            n_leak += 1
        elif key in seen:
            n_dup += 1
        elif len(tokenizer.apply_chat_template(row["prompt"], add_generation_prompt=True, tokenize=True)) > args.max_prompt_length:
            n_long += 1
        else:
            seen.add(key)
            keep.append(i)
    merged_ds = merged_ds.select(keep)
    print(f"Removed {n_dup} internal duplicates, {n_leak} test-overlap and {n_long} over-length samples -> {len(merged_ds)}")

    # Shuffle if requested
    if args.shuffle:
        print(f"Shuffling with seed={args.seed}...")
        merged_ds = merged_ds.shuffle(seed=args.seed)

    # Truncate to 16384 samples (1024 * 16)
    max_samples = 16384
    if len(merged_ds) > max_samples:
        print(f"Truncating: {len(merged_ds)} -> {max_samples}")
        merged_ds = merged_ds.select(range(max_samples))
    print(f"Final samples: {len(merged_ds)}")

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save merged dataset
    merged_ds.to_parquet(output_path)
    print(f"Saved merged dataset to: {output_path}")
