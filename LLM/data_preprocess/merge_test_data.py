import argparse
import os

import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge test datasets.")
    parser.add_argument(
        "--math500_path",
        default="data/math12k/test.parquet",
        help="Path to Math12K test.parquet file.",
    )
    parser.add_argument(
        "--minervamath_path",
        default="data/minervamath/test.parquet",
        help="Path to minervamath test.parquet file.",
    )
    parser.add_argument(
        "--aime25_path",
        default="data/aime25/test.parquet",
        help="Path to aime25 test.parquet file.",
    )
    parser.add_argument(
        "--amc23_path",
        default="data/amc23/test.parquet",
        help="Path to amc23 test.parquet file.",
    )
    parser.add_argument(
        "--output_path",
        default="data/test.parquet",
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
    math500_path = os.path.expanduser(args.math500_path)
    minervamath_path = os.path.expanduser(args.minervamath_path)
    aime25_path = os.path.expanduser(args.aime25_path)
    amc23_path = os.path.expanduser(args.amc23_path)
    output_path = os.path.expanduser(args.output_path)

    # Read datasets
    print(f"Reading Math500 from: {math500_path}")
    math500_df = pd.read_parquet(math500_path)
    print(f"  Math500 samples: {len(math500_df)}")

    print(f"Reading minervamath from: {minervamath_path}")
    minervamath_df = pd.read_parquet(minervamath_path)
    print(f"  minervamath samples: {len(minervamath_df)}")

    print(f"Reading aime25 from: {aime25_path}")
    aime25_df = pd.read_parquet(aime25_path)
    print(f"  aime25 samples: {len(aime25_df)}")

    print(f"Reading amc23 from: {amc23_path}")
    amc23_df = pd.read_parquet(amc23_path)
    print(f"  amc23 samples: {len(amc23_df)}")

    # Merge datasets
    merged_df = pd.concat([math500_df, minervamath_df, aime25_df, amc23_df], ignore_index=True)
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
