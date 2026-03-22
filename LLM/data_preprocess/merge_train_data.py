import argparse
import os

import datasets


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
