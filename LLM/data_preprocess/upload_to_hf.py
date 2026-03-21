"""
Upload processed datasets to HuggingFace Hub.

Usage:
    # Login first (if not already logged in)
    huggingface-cli login

    # Upload merged train.parquet and test.parquet
    python upload_to_hf.py --repo_id YOUR_USERNAME/RLVR-Math-16K

    # Dry run
    python upload_to_hf.py --repo_id YOUR_USERNAME/RLVR-Math-16K --dry_run
"""

import argparse
import os

from huggingface_hub import HfApi, create_repo


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
SPLITS = ["train.parquet", "test.parquet"]


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

    if dry_run:
        print("\n[Dry run] No files uploaded.")
        return

    api = HfApi()
    create_repo(repo_id, repo_type="dataset", private=False, exist_ok=True)
    print(f"\nRepository '{repo_id}' is ready.")

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
    parser.add_argument("--repo_id", required=True, help="HuggingFace repo ID, e.g. 'your-username/RLVR-Math-16K'.")
    parser.add_argument("--data_dir", default=DATA_DIR, help=f"Data directory (default: {DATA_DIR}).")
    parser.add_argument("--dry_run", action="store_true", help="Show what would be uploaded without uploading.")

    args = parser.parse_args()
    upload_datasets(repo_id=args.repo_id, data_dir=args.data_dir, dry_run=args.dry_run)
