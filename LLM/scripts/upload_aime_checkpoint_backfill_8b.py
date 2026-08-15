#!/usr/bin/env python3
"""Upload 8B AIME24/AIME26 metrics to dedicated W&B backfill runs."""

import argparse
import json
from pathlib import Path

import wandb


ENTITY = "julianlou2002-massachusetts-institute-of-technology"
PROJECT = "opts_ttpo_8B"
PARENT_RUN_IDS = {
    "dapo_0805_n8_8B": "yhtrevjj",
    "reinforce_pp_baseline_0805_n8_8B": "5ez537zd",
}
BACKFILL_RUN_IDS = {
    "dapo_0805_n8_8B": "dapo8baime",
    "reinforce_pp_baseline_0805_n8_8B": "rein8baime",
}
EXPECTED_KEYS = {
    f"val-core/math-ai/{dataset}/{metric}"
    for dataset in ("aime24", "aime26")
    for metric in ("acc/avg@32", "acc/pass@32", "acc/cons@32", "reward/mean@32")
}


def load_records(metrics_dir: Path) -> list[dict]:
    records = []
    for path in metrics_dir.glob("step_*.json"):
        with path.open(encoding="utf-8") as stream:
            record = json.load(stream)
        keys = set(record["metrics"])
        if keys != EXPECTED_KEYS:
            raise ValueError(f"{path}: metric keys differ: missing={EXPECTED_KEYS - keys}, extra={keys - EXPECTED_KEYS}")
        records.append(record)
    return sorted(records, key=lambda record: record["step"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("logs/backfill_aime24_aime26"))
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()

    all_records = {}
    for experiment_name, parent_run_id in PARENT_RUN_IDS.items():
        checkpoint_dir = Path("/share/lujunyu/ckpts/opts_ckpts") / PROJECT / experiment_name
        expected_steps = sorted(
            int(path.name.removeprefix("global_step_"))
            for path in checkpoint_dir.glob("global_step_*")
            if (path / "actor").is_dir()
        )
        records = load_records(args.results_root / experiment_name / "metrics")
        actual_steps = [record["step"] for record in records]
        if actual_steps != expected_steps:
            raise RuntimeError(
                f"{experiment_name}: incomplete results; "
                f"missing={sorted(set(expected_steps) - set(actual_steps))}, "
                f"extra={sorted(set(actual_steps) - set(expected_steps))}"
            )
        all_records[experiment_name] = records
        print(
            f"VALID {experiment_name}: {len(records)} checkpoints -> "
            f"W&B {BACKFILL_RUN_IDS[experiment_name]} (parent {parent_run_id})"
        )

    if not args.upload:
        print("Validation only; pass --upload to write to W&B.")
        return

    for experiment_name, parent_run_id in PARENT_RUN_IDS.items():
        backfill_run_id = BACKFILL_RUN_IDS[experiment_name]
        receipt_path = args.results_root / experiment_name / "wandb_upload.json"
        expected_steps = [record["step"] for record in all_records[experiment_name]]
        if receipt_path.exists():
            with receipt_path.open(encoding="utf-8") as stream:
                receipt = json.load(stream)
            if receipt.get("backfill_run_id") == backfill_run_id and receipt.get("steps") == expected_steps:
                print(f"ALREADY UPLOADED {experiment_name} -> W&B {backfill_run_id}")
                continue
        run = wandb.init(
            entity=ENTITY,
            project=PROJECT,
            id=backfill_run_id,
            resume="never",
            name=f"{experiment_name}_aime24_aime26_backfill",
            group=experiment_name,
            job_type="checkpoint_validation_backfill",
            tags=["aime24", "aime26", "backfill", f"parent-{parent_run_id}"],
            config={
                "parent_run_id": parent_run_id,
                "checkpoint_experiment": experiment_name,
                "datasets": ["math-ai/aime24", "math-ai/aime26"],
                "val_n": 32,
                "max_prompt_length": 1024,
                "max_response_length": 2048,
                "temperature": 1.0,
                "top_p": 0.95,
            },
            reinit="finish_previous",
        )
        for record in all_records[experiment_name]:
            run.log({"training/global_step": record["step"], **record["metrics"]}, step=record["step"])
        run.finish()
        receipt = {
            "entity": ENTITY,
            "project": PROJECT,
            "parent_run_id": parent_run_id,
            "backfill_run_id": backfill_run_id,
            "steps": expected_steps,
            "metric_keys": sorted(EXPECTED_KEYS),
        }
        temporary_path = receipt_path.with_suffix(".json.tmp")
        with temporary_path.open("w", encoding="utf-8") as stream:
            json.dump(receipt, stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.write("\n")
        temporary_path.replace(receipt_path)
        print(
            f"UPLOADED {experiment_name}: {len(all_records[experiment_name])} checkpoints "
            f"-> W&B {backfill_run_id}"
        )


if __name__ == "__main__":
    main()
