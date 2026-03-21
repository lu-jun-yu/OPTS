"""
Upload processed datasets to HuggingFace Hub.

Usage:
    # Login first (if not already logged in)
    huggingface-cli login

    # Upload merged train.parquet and test.parquet
    python upload_to_hf.py --repo_id YOUR_USERNAME/RLVR-Math-16k

    # Dry run
    python upload_to_hf.py --repo_id YOUR_USERNAME/RLVR-Math-16k --dry_run
"""

import argparse
import os
import tempfile

import pandas as pd
from huggingface_hub import HfApi, create_repo


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
SPLITS = ["train.parquet", "test.parquet"]


def generate_dataset_card(repo_id: str, data_dir: str) -> str:
    """Generate a HuggingFace dataset card (README.md) with dataset statistics."""
    dataset_name = repo_id.split("/")[-1]

    # Collect stats from parquet files
    split_stats = {}
    source_stats = {}
    for name in SPLITS:
        path = os.path.join(data_dir, name)
        if not os.path.exists(path):
            continue
        df = pd.read_parquet(path)
        split_name = name.replace(".parquet", "")
        split_stats[split_name] = len(df)

        # Count by data_source
        if "data_source" in df.columns:
            for src, cnt in df["data_source"].value_counts().items():
                source_stats.setdefault(split_name, {})[src] = cnt

    total = sum(split_stats.values())

    # Build split table
    split_rows = "\n".join(f"| {s} | {n:,} |" for s, n in split_stats.items())

    # Build source breakdown
    source_section = ""
    for split_name, sources in source_stats.items():
        rows = "\n".join(f"| {src} | {cnt:,} |" for src, cnt in sorted(sources.items(), key=lambda x: -x[1]))
        source_section += f"\n### {split_name}\n\n| Source | Samples |\n|--------|--------:|\n{rows}\n"

    # Build train source descriptions
    train_sources = source_stats.get("train", {})
    train_desc_parts = []
    if "hiyouga/math12k" in train_sources:
        train_desc_parts.append("- [hiyouga/math12k](https://huggingface.co/datasets/hiyouga/math12k): MATH competition problems (converted from OpenAI PRM800K)")
    if "nlile/NuminaMath-1.5-RL-Verifiable" in train_sources:
        train_desc_parts.append("- [nlile/NuminaMath-1.5-RL-Verifiable](https://huggingface.co/datasets/nlile/NuminaMath-1.5-RL-Verifiable): AMC/AIME and Olympiad competition problems")
    train_desc = "\n".join(train_desc_parts) if train_desc_parts else ""

    test_sources = source_stats.get("test", {})
    test_desc_parts = []
    if "hiyouga/math12k" in test_sources:
        test_desc_parts.append("- [hiyouga/math12k](https://huggingface.co/datasets/hiyouga/math12k): MATH500")
    if "math-ai/minervamath" in test_sources:
        test_desc_parts.append("- [math-ai/minervamath](https://huggingface.co/datasets/math-ai/minervamath): Minerva Math")
    if "math-ai/aime25" in test_sources:
        test_desc_parts.append("- [math-ai/aime25](https://huggingface.co/datasets/math-ai/aime25): AIME 2025")
    if "math-ai/amc23" in test_sources:
        test_desc_parts.append("- [math-ai/amc23](https://huggingface.co/datasets/math-ai/amc23): AMC 2023")
    test_desc = "\n".join(test_desc_parts) if test_desc_parts else ""

    card = f"""---
language:
- en
task_categories:
- text-generation
tags:
- math
- reasoning
- rlhf
- rlvr
- grpo
size_categories:
- 10K<n<100K
---

# {dataset_name}

A curated math reasoning dataset for **RLVR (Reinforcement Learning with Verifiable Rewards)** training.

## Dataset Summary

| Split | Samples |
|-------|--------:|
{split_rows}
| **Total** | **{total:,}** |

## Source Datasets
{source_section}
"""

    if train_desc:
        card += f"""### Training Sources

{train_desc}

"""

    if test_desc:
        card += f"""### Test Sources

{test_desc}

"""

    card += """## Data Format

Each sample follows the verl-compatible chat format:

```json
{
    "data_source": "source_dataset_id",
    "prompt": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "math problem text"}
    ],
    "ability": "math",
    "reward_model": {"style": "rule", "ground_truth": "answer"},
    "extra_info": {"split": "train/test", "index": 0}
}
```

## Preprocessing

**Training data filters:**
- Source filter: only competition-level problems (olympiads, amc_aime)
- Length filter: problem <= 2000 chars, solution <= 3000 chars
- Test set deduplication: removed overlapping problems with all test benchmarks
- Stratified sampling by source category
- Answer parsability: verified via [math-verify](https://github.com/huggingface/Math-Verify) to ensure reliable reward signals

**Test data:** standard benchmarks used as-is (no filtering applied).

## Intended Use

This dataset is designed for RLVR math reasoning training (e.g., DAPO, REINFORCE++) with rule-based reward verification.
"""

    return card


def upload_datasets(repo_id: str, data_dir: str, dry_run: bool):
    files = []
    for name in SPLITS:
        path = os.path.join(data_dir, name)
        if not os.path.exists(path):
            print(f"Warning: {path} not found, skipping.")
            continue
        files.append((name, path))

    if not files:
        print("No parquet files found to upload.")
        return

    print(f"Repository: {repo_id}")
    for name, path in files:
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  {name:<20s} {size_mb:.2f} MB")

    # Generate dataset card
    print("\nGenerating dataset card...")
    card_content = generate_dataset_card(repo_id, data_dir)

    if dry_run:
        print("\n--- Dataset Card Preview ---")
        print(card_content)
        print("--- End Preview ---")
        print("\n[Dry run] No files uploaded.")
        return

    api = HfApi()
    create_repo(repo_id, repo_type="dataset", private=False, exist_ok=True)
    print(f"\nRepository '{repo_id}' is ready.")

    # Upload dataset card
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
        f.write(card_content)
        card_path = f.name
    try:
        print("Uploading README.md ...", end=" ", flush=True)
        api.upload_file(
            path_or_fileobj=card_path,
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
        )
        print("done.")
    finally:
        os.unlink(card_path)

    # Upload data files
    for name, path in files:
        path_in_repo = f"data/{name}"
        print(f"Uploading {name} -> {path_in_repo} ...", end=" ", flush=True)
        api.upload_file(
            path_or_fileobj=path,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="dataset",
        )
        print("done.")

    print(f"\nAll files uploaded to https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload processed datasets to HuggingFace Hub.")
    parser.add_argument("--repo_id", required=True, help="HuggingFace repo ID, e.g. 'your-username/RLVR-Math-16k'.")
    parser.add_argument("--data_dir", default=DATA_DIR, help=f"Data directory (default: {DATA_DIR}).")
    parser.add_argument("--dry_run", action="store_true", help="Show what would be uploaded without uploading.")

    args = parser.parse_args()
    upload_datasets(repo_id=args.repo_id, data_dir=args.data_dir, dry_run=args.dry_run)
