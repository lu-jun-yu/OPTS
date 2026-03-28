"""Upload OPTS_TTPO MuJoCo checkpoints to Hugging Face Hub.

Usage:
    python scripts/upload_checkpoints_to_hf.py --repo-id your_username/opts-ttpo-variance-reduction
    python scripts/upload_checkpoints_to_hf.py --repo-id your_username/opts-ttpo-variance-reduction --private
"""

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, CommitOperationAdd


TASKS = ["HalfCheetah-v4", "Walker2d-v4", "Hopper-v4", "Ant-v4", "Humanoid-v4"]

MODEL_CARD_TEMPLATE = """\
---
tags:
- reinforcement-learning
- deep-reinforcement-learning
- mujoco
- opts
- ttpo
- policy-gradient
- variance-reduction
library_name: cleanrl
---

# OPTS_TTPO MuJoCo Checkpoints

Trained checkpoints for **OPTS_TTPO** (Online Tree-Search with Trust-region Policy Optimization) on MuJoCo continuous control tasks.

## Tasks

{task_table}

## Checkpoint Format

Each `.pt` file is a PyTorch checkpoint dict containing:

```python
{{
    "model_state_dict": ...,   # Agent network weights
    "obs_rms_mean": ...,       # NormalizeObservation running mean
    "obs_rms_var": ...,        # NormalizeObservation running variance
    "obs_rms_count": ...,      # NormalizeObservation sample count
    "ret_rms_mean": ...,       # NormalizeReward running mean
    "ret_rms_var": ...,        # NormalizeReward running variance
    "ret_rms_count": ...,      # NormalizeReward sample count
}}
```

## Usage

```python
from huggingface_hub import hf_hub_download
import torch

path = hf_hub_download(repo_id="{repo_id}", filename="HalfCheetah-v4/seed1.pt")
checkpoint = torch.load(path, map_location="cpu")
agent.load_state_dict(checkpoint["model_state_dict"])
```

## Training

```bash
python cleanrl/cleanrl/opts_ttpo_continuous_action.py \\
    --env-id HalfCheetah-v4 --seed 1 \\
    --total-timesteps 1000000 --num-steps 4096 --save-model
```

## Hyperparameters

```json
{hyperparams}
```
"""


def main():
    parser = argparse.ArgumentParser(description="Upload OPTS_TTPO checkpoints to Hugging Face")
    parser.add_argument("--repo-id", type=str, required=True,
                        help="HuggingFace repo id, e.g. 'username/opts-ttpo-mujoco'")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Directory containing checkpoint files (default: checkpoints)")
    parser.add_argument("--private", action="store_true",
                        help="Create a private repository")
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    files = sorted(checkpoint_dir.rglob("*.pt"))
    if not files:
        raise FileNotFoundError(f"No .pt files found in {checkpoint_dir}")

    print(f"Found {len(files)} checkpoint files:")
    for f in files:
        print(f"  {f.relative_to(checkpoint_dir)}")

    api = HfApi()
    repo_url = api.create_repo(repo_id=args.repo_id, exist_ok=True, private=args.private)
    entity, repo = str(repo_url).rstrip("/").split("/")[-2:]
    repo_id = f"{entity}/{repo}"
    print(f"\nRepo: {repo_url}")

    # Upload checkpoint files
    operations = []
    for f in files:
        rel_path = str(f.relative_to(checkpoint_dir)).replace("\\", "/")
        operations.append(CommitOperationAdd(path_or_fileobj=str(f), path_in_repo=rel_path))

    # Build task table
    task_seeds: dict[str, list[str]] = {}
    for f in files:
        rel = f.relative_to(checkpoint_dir)
        task = rel.parts[0] if len(rel.parts) > 1 else "unknown"
        task_seeds.setdefault(task, []).append(rel.stem)

    table_lines = ["| Task | Seeds |", "|------|-------|"]
    for task in TASKS:
        if task in task_seeds:
            table_lines.append(f"| {task} | {', '.join(sorted(task_seeds[task]))} |")
    for task, seeds in task_seeds.items():
        if task not in TASKS:
            table_lines.append(f"| {task} | {', '.join(sorted(seeds))} |")

    hyperparams = json.dumps({
        "total_timesteps": 1000000, "num_steps": 4096, "learning_rate": 3e-4,
        "gamma": 0.99, "gae_lambda": 0.95, "num_minibatches": 32,
        "update_epochs": 10, "clip_coef": 0.2, "max_search_per_tree": 4,
    }, indent=2)

    readme = MODEL_CARD_TEMPLATE.format(
        repo_id=repo_id, task_table="\n".join(table_lines), hyperparams=hyperparams,
    )

    # Write README to temp file and upload
    readme_tmp = checkpoint_dir / "_README_tmp.md"
    readme_tmp.write_text(readme, encoding="utf-8")
    operations.append(CommitOperationAdd(path_or_fileobj=str(readme_tmp), path_in_repo="README.md"))

    print(f"\nUploading {len(operations)} files...")
    api.create_commit(
        repo_id=repo_id,
        operations=operations,
        commit_message="Upload OPTS_TTPO MuJoCo checkpoints",
    )
    readme_tmp.unlink(missing_ok=True)

    print(f"\nDone! https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
