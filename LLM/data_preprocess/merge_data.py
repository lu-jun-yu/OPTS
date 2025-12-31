import argparse
import os

import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge GSM8K and Math12K training datasets.")
    parser.add_argument(
        "--gsm8k_path",
        default="data/gsm8k/train.parquet",
        help="Path to GSM8K train.parquet file.",
    )
    parser.add_argument(
        "--math12k_path",
        default="data/math12k/train.parquet",
        help="Path to Math12K train.parquet file.",
    )
    parser.add_argument(
        "--output_path",
        default="data/train.parquet",
        help="Output path for the merged dataset.",
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
    gsm8k_path = os.path.expanduser(args.gsm8k_path)
    math12k_path = os.path.expanduser(args.math12k_path)
    output_path = os.path.expanduser(args.output_path)

    # Read datasets
    print(f"Reading GSM8K from: {gsm8k_path}")
    gsm8k_df = pd.read_parquet(gsm8k_path)
    print(f"  GSM8K samples: {len(gsm8k_df)}")

    print(f"Reading Math12K from: {math12k_path}")
    math12k_df = pd.read_parquet(math12k_path)
    print(f"  Math12K samples: {len(math12k_df)}")

    # Merge datasets
    merged_df = pd.concat([gsm8k_df, math12k_df], ignore_index=True)
    print(f"Merged samples: {len(merged_df)}")

    # Shuffle if requested
    if args.shuffle:
        print(f"Shuffling with seed={args.seed}...")
        merged_df = merged_df.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save merged dataset
    merged_df.to_parquet(output_path)
    print(f"Saved merged dataset to: {output_path}")
