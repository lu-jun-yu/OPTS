"""Validate independent async fixed controls and g* before gradient computation."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.RQ3.gradients import _prompt_ids
from experiments.RQ2.search import request_sampling_seed
from transformers import AutoTokenizer


def c_root_map(path, prompt_offset, group_stride):
    frame = pd.read_parquet(
        path, columns=["tree_seqs", "tree_rounds", "response_ids", "tree_rewards"]
    )
    result = {}
    for local_prompt, row in frame.iterrows():
        prompt_idx = prompt_offset + int(local_prompt)
        for index, (tree_seq, round_idx) in enumerate(zip(row.tree_seqs, row.tree_rounds)):
            if int(round_idx) == 0:
                group = int(tree_seq) // group_stride
                result[(prompt_idx, group)] = (
                    np.asarray(row.response_ids[index], dtype=np.int32),
                    float(row.tree_rewards[index]),
                )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed", required=True)
    parser.add_argument("--gstar", required=True)
    parser.add_argument("--c", required=True)
    parser.add_argument("--prompts", default="data/train.parquet")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-offset", type=int, default=0)
    parser.add_argument("--group-stride", type=int, default=2048)
    parser.add_argument("--estimator-seed", type=int, default=20_260_915)
    parser.add_argument("--reference-seed", type=int, default=120_260_915)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    fixed = pd.read_parquet(args.fixed)
    gstar = pd.read_parquet(args.gstar)
    c_roots = c_root_map(args.c, args.prompt_offset, args.group_stride)
    if fixed.duplicated(["prompt_idx", "group"]).any():
        raise ValueError("duplicate fixed prompt/group")
    if gstar.duplicated(["prompt_idx", "group"]).any():
        raise ValueError("duplicate gstar prompt/group")

    fixed_map = {
        (int(row.prompt_idx), int(row.group)): row for row in fixed.itertuples(index=False)
    }
    if fixed_map.keys() != c_roots.keys():
        raise ValueError(f"fixed/C prompt-group grids differ: fixed={len(fixed_map)}, C={len(c_roots)}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompt_frame = pd.read_parquet(args.prompts)
    exact_c_root_matches = 0
    branch_count = 0
    fixed_seeds = set()
    for key, row in fixed_map.items():
        c_ids, _ = c_roots[key]
        if np.array_equal(np.asarray(row.backbone_ids, dtype=np.int32), c_ids):
            exact_c_root_matches += 1
        prompt_idx, group = key
        prompt_len = len(_prompt_ids(prompt_frame.iloc[prompt_idx], tokenizer))
        if prompt_len + 128 > 1152:
            raise ValueError(f"ineligible prompt leaked into fixed data: {prompt_idx}")
        expected_root_seed = request_sampling_seed(args.estimator_seed, prompt_idx, group, 0)
        if int(row.backbone_sampling_seed) != expected_root_seed:
            raise ValueError(f"fixed root seed mismatch at {key}")
        fixed_seeds.add(int(row.backbone_sampling_seed))
        if len(row.backbone_ids) > 128:
            if len(row.suffix_ids) != 7 or any(len(ids) == 0 for ids in row.suffix_ids):
                raise ValueError(f"missing fixed suffixes at {key}")
            for round_idx, seed in enumerate(row.suffix_sampling_seeds, start=1):
                expected = request_sampling_seed(
                    args.estimator_seed, prompt_idx, group, round_idx
                )
                if int(seed) != expected:
                    raise ValueError(f"fixed seed mismatch at {key} round={round_idx}")
                fixed_seeds.add(int(seed))
                branch_count += 1
    if exact_c_root_matches == len(fixed_map):
        raise ValueError("fixed roots improperly reuse all C trajectories; independence violated")

    expected_reference = {(pi, group) for pi, _ in c_roots for group in range(32)}
    if set(zip(gstar.prompt_idx, gstar.group)) != expected_reference:
        raise ValueError('Incomplete reference prompt/group grid')
    gstar_seeds = set()
    for row in gstar.itertuples(index=False):
        prompt_idx, group = int(row.prompt_idx), int(row.group)
        if not 0 <= group < 32:
            raise ValueError("invalid gstar prompt/group")
        expected = request_sampling_seed(args.reference_seed, prompt_idx, group, 0)
        if int(row.backbone_sampling_seed) != expected:
            raise ValueError(f"gstar seed mismatch at {(prompt_idx, group)}")
        gstar_seeds.add(int(row.backbone_sampling_seed))
    if len(gstar_seeds) != len(gstar) or not fixed_seeds.isdisjoint(gstar_seeds):
        raise ValueError("duplicate or overlapping control/reference seed spaces")

    summary = {
        "fixed_independent_roots": len(fixed),
        "fixed_roots_accidentally_equal_to_c": exact_c_root_matches,
        "fixed_branches": branch_count,
        "gstar_roots": len(gstar),
        "fixed_gstar_seed_overlap": 0,
    }
    Path(args.output).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
