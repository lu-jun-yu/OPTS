"""Allocate independent fixed-128 trees using OPTS per-prompt, per-round counts."""
import argparse
import glob
from pathlib import Path
import numpy as np
import pandas as pd

BRANCH_POS = 128
SNAP_ROUNDS = (1, 3, 7)
SEED = 20260917

def allocate(counts, eligible, seed=SEED):
    """Sample all groups uniformly; short chains do not trigger replacement draws."""
    groups, prompts = eligible.shape
    if counts.shape != (prompts, 7) or np.any(counts < 0) or np.any(counts > groups):
        raise ValueError("Expected seven per-prompt counts between zero and n_groups")
    chosen = np.zeros((groups, prompts, 7), dtype=bool)
    for pi in range(prompts):
        rng = np.random.default_rng(np.random.SeedSequence([seed, pi]))
        for r, n in enumerate(counts[pi]):
            chosen[rng.permutation(groups)[:int(n)], pi, r] = True
    realized = chosen & eligible[:, :, None]
    k = np.cumsum(realized, axis=2)[:, :, np.asarray(SNAP_ROUNDS)-1].astype(np.int8)
    assert np.array_equal(chosen.sum(axis=0), counts)
    return dict(k=k, nominal_selection=chosen, realized_selection=realized,
                prompt_round_counts=counts, counts=realized.sum(axis=1),
                snap_rounds=np.asarray(SNAP_ROUNDS), seed=seed,
                branch_pos=BRANCH_POS, allocation="prompt_round_v2")


def build_masks(events, eligible, seed=SEED):
    """Match OPTS branch counts for each prompt and round, with p=1."""
    if events.duplicated(["prompt_idx", "group", "round"]).any():
        raise ValueError("Duplicate OPTS prompt/group/round")
    counts = np.zeros((eligible.shape[1], 7), dtype=np.int64)
    np.add.at(counts, (events.prompt_idx.to_numpy(dtype=int),
                      events['round'].to_numpy(dtype=int)-1), 1)
    return allocate(counts, eligible, seed)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opts-dir", required=True)
    ap.add_argument("--pattern", default="opts_bias_s7_t8_full_shard")
    ap.add_argument("--shard-size", type=int, default=2048)
    ap.add_argument("--n-shards", type=int, default=8)
    ap.add_argument("--prompts", default="data/train.parquet")
    ap.add_argument("--tree-data", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    n_prompts = args.shard_size * args.n_shards
    expected_prompts = set(range(n_prompts))
    prompts = pd.read_parquet(args.prompts, columns=["extra_info"])
    counts = np.zeros((n_prompts, 7), dtype=np.int64)
    events = []
    for sh in range(args.n_shards):
        path = Path(args.opts_dir) / f"{args.pattern}{sh}.parquet"
        rows = pd.read_parquet(path, columns=["extra_info"])
        offset = sh * args.shard_size
        if len(rows) != args.shard_size or rows.extra_info.tolist() != prompts.iloc[
                offset:offset + args.shard_size].extra_info.tolist():
            raise ValueError(f"Prompt mapping mismatch: {path}")
        sel = pd.read_parquet(path.with_name(path.stem + "_selections.parquet"))
        if sel.duplicated(["tree_seq", "round"]).any():
            raise ValueError(f"Duplicate tree/round selection: {path}")
        for row in sel.itertuples(index=False):
            pi, r, ts = int(row.dataset_idx), int(row.round), int(row.tree_seq)
            if not (0 <= pi < args.shard_size and 1 <= r <= 7
                    and 0 <= ts < 8 * args.shard_size and ts % args.shard_size == pi):
                raise ValueError(f"Invalid selection mapping: {row}")
            counts[offset + pi, r - 1] += 1
            events.append(dict(prompt_idx=offset+pi, group=ts//args.shard_size,
                               round=r, score=float(row.score)))
    files = sorted({f for pattern in args.tree_data for f in glob.glob(pattern)})
    if not files:
        raise ValueError("No A-side tree data")
    df = pd.concat([pd.read_parquet(f, columns=["prompt_idx", "group", "prefix_len",
                                              "backbone_ids"]) for f in files])
    if (len(df) != 8 * len(expected_prompts) or df.duplicated(["group", "prompt_idx"]).any()
            or not df.group.between(0, 7).all()
            or not df.prompt_idx.between(0, n_prompts - 1).all()):
        raise ValueError("A-side data must cover all eight groups and all prompts exactly once")
    for group in range(8):
        actual = set(df.loc[df.group == group, "prompt_idx"].astype(int).tolist())
        if actual != expected_prompts:
            missing = sorted(expected_prompts - actual)[:10]
            extra = sorted(actual - expected_prompts)[:10]
            raise ValueError(
                f"A-side prompt grid mismatch for group {group}: missing={missing}, extra={extra}"
            )
    lengths = df.backbone_ids.str.len().to_numpy()
    if not np.array_equal(df.prefix_len.to_numpy(), np.minimum(lengths, BRANCH_POS)):
        raise ValueError("Expected fixed-128 data; do not reuse fixed-256 suffixes")
    eligible = np.zeros((8, n_prompts), dtype=bool)
    eligible[df.group.to_numpy(), df.prompt_idx.to_numpy()] = lengths > BRANCH_POS
    result = build_masks(pd.DataFrame(events, columns=['prompt_idx','group','round','score']),
                         eligible, args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **result)
    print("nominal per round:", counts.sum(axis=0).tolist())
    print("realized per round:", result["counts"].sum(axis=0).tolist())
    print("short-chain skipped:", int(result["nominal_selection"].sum()
                                       - result["realized_selection"].sum()))
    print("saved", args.out)

if __name__ == "__main__":
    main()
