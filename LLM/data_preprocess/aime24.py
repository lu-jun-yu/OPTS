import argparse
import os
import re

import datasets
import pyarrow.parquet as pq

from prompts import SYSTEM_PROMPT


BOXED_ANSWER_RE = re.compile(r"\\boxed\{([^{}]+)\}")


def extract_answer(solution: str) -> str:
    r"""Extract the numeric answer from the source dataset's ``\boxed{...}`` value."""
    match = BOXED_ANSWER_RE.fullmatch(solution.strip())
    if match is None:
        raise ValueError(f"Unexpected AIME24 solution format: {solution!r}")
    return match.group(1)


def load_source_dataset(local_dataset_path: str | None):
    if local_dataset_path is None:
        return datasets.load_dataset("math-ai/aime24")
    if local_dataset_path.endswith(".parquet"):
        return datasets.DatasetDict({"test": datasets.Dataset.from_parquet(local_dataset_path)})
    return datasets.load_dataset(local_dataset_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess AIME 2024 for verl evaluation.")
    parser.add_argument("--local_dir", default=None, help="Deprecated alias for --local_save_dir.")
    parser.add_argument("--local_dataset_path", default=None, help="Local raw dataset directory or parquet file.")
    parser.add_argument(
        "--local_save_dir", default="data/aime24", help="The save directory for the preprocessed dataset."
    )
    args = parser.parse_args()

    data_source = "math-ai/aime24"
    test_dataset = load_source_dataset(args.local_dataset_path)["test"]

    def process_fn(example, idx):
        return {
            "data_source": data_source,
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["problem"]},
            ],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": extract_answer(example["solution"]),
            },
            "extra_info": {
                "split": "test",
                "index": idx,
            },
        }

    test_dataset = test_dataset.map(
        function=process_fn,
        with_indices=True,
        remove_columns=test_dataset.column_names,
    )

    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir

    os.makedirs(local_save_dir, exist_ok=True)
    output_path = os.path.join(local_save_dir, "test.parquet")
    pq.write_table(test_dataset.data.table, output_path, compression="snappy", use_dictionary=True)
    print(f"Saved {len(test_dataset)} samples to {output_path}")
