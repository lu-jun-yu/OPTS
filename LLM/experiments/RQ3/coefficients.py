"""Direct V=0, gamma=lambda=1 return coefficients on complete OPTS snapshots."""
import os
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import torch

SNAP_ROUNDS = (0, 1, 3, 7)
RETURN_SCHEMA = "full_tree_zero_value_v1"

def full_tree_returns(rid_list, included, rewards, backup):
    """Back up returns over all suffixes in the snapshot.

    A child branches after its parent's bp token. Mean averages immediate
    successors, not all descendant leaves. Segments are in generation order.
    """
    if backup not in ("max", "mean"):
        raise ValueError(backup)
    children = defaultdict(lambda: defaultdict(list))
    inc = set(included)
    for j in included:
        parent, bp, ids = rid_list[j]
        if not len(ids) or not np.isfinite(rewards[j]):
            raise ValueError("empty segment or nonfinite reward")
        if parent >= 0:
            if parent not in inc or parent >= j or not 0 <= bp < len(rid_list[parent][2]):
                raise ValueError("invalid snapshot parent/branch position")
            children[parent][bp].append(j)
    result = {}
    for j in reversed(included):
        out = np.full(len(rid_list[j][2]), float(rewards[j]), dtype=np.float64)
        for bp, kids in sorted(children[j].items(), reverse=True):
            successors = [result[k][0] for k in kids]
            if bp + 1 < len(out):
                successors.append(out[bp + 1])
            value = max(successors) if backup == "max" else float(np.mean(successors))
            if bp == len(out) - 1:
                value += float(rewards[j])
            out[:bp + 1] = value
        result[j] = out
    return result

def build_tree_coefficients(row, shard_size, backup="max"):
    """Reconstruct return credit and tree weights without stored critic values."""
    uids = row["tree_uids"]
    seqs = row["tree_seqs"]
    rids = row["tree_rids"]
    pids = row["tree_pids"]
    bps = row["tree_branch_pos"]
    ids_col = row["response_ids"]
    by_uid = defaultdict(list)
    for i, u in enumerate(uids):
        by_uid[str(u)].append(i)

    trees = []
    for uid, idxs in by_uid.items():
        # full tree rids, in generation order
        idxs = sorted(idxs, key=lambda i: row["global_indices"][i])
        rid_list = []           # (pid_idx, bp, ids)
        rid2local = {}
        for i in idxs:
            rid = str(rids[i])
            pid = pids[i]
            pid = None if pid in (None, "None") else str(pid)
            ids = np.asarray(ids_col[i], dtype=np.int32)
            pid_idx = -1 if pid is None else rid2local.get(pid, -2)
            assert pid_idx != -2, f"parent {pid} of {rid} not seen before it (uid {uid})"
            rid2local[rid] = len(rid_list)
            rid_list.append((pid_idx, int(bps[i]), ids))

        group = int(seqs[idxs[0]]) // shard_size
        n = len(rid_list)

        slots = {}
        for s in SNAP_ROUNDS:
            included = [j for j, i in enumerate(idxs)
                        if int(row["tree_rounds"][i]) <= s]
            if not included:
                continue
            inc_set = set(included)
            pure_returns = full_tree_returns(rid_list, included,
                            [row["tree_rewards"][i] for i in idxs], backup)
            # state_branches: nb[j][t] = 1 + #included children of j with bp == t
            nb = [np.ones(len(rid_list[j][2]), dtype=np.float64) for j in range(n)]
            roots = 0
            for j in included:
                pid_idx, bp, _ = rid_list[j]
                if pid_idx < 0:
                    roots += 1
                    continue
                if pid_idx not in inc_set:
                    # parent trimmed away at this round cannot happen (parent
                    # is always generated before child), but guard anyway
                    raise AssertionError(f"parent trimmed at round {s}")
                if bp >= 0:
                    nb[pid_idx][bp] += 1.0

            # init weights: walk ancestor chain
            init_w = np.zeros(n)
            for j in included:
                w = 1.0
                cur, cur_bp = j, rid_list[j][1]
                while rid_list[cur][0] >= 0:
                    p = rid_list[cur][0]
                    w *= float(nb[p][: cur_bp + 1].prod()) if cur_bp >= 0 else 1.0
                    cur, cur_bp = p, rid_list[p][1]
                init_w[j] = w * roots

            # per-token weights within each rid's own segment; coefficients stay
            # unnormalized (raw w * credit); the tree's weight sum D_i is stored
            # alongside for the normalized-aggregation denominator.
            segs = []
            tree_sum = 0.0
            for j in included:
                _, _, ids = rid_list[j]
                L = len(ids)
                adv = pure_returns[j]
                assert len(adv) == L, (
                    f"adv/ids length mismatch uid {uid} rid {rids[idxs[j]]} s{s}: "
                    f"{len(adv)} vs {L}"
                )
                credit = adv
                padded = np.concatenate([[1.0], nb[j][:-1]])
                w_tok = 1.0 / (init_w[j] * np.cumprod(padded))
                tree_sum += float(w_tok.sum())
                segs.append((j, (w_tok * credit).astype(np.float32)))
            assert tree_sum > 0
            slots[s] = {"segs": segs, "weight_sum": np.float32(tree_sum),
                        "n_suffixes": len(included) - 1}

        trees.append({
            "prompt_row": int(row.name),
            "tree_seq": int(seqs[idxs[0]]),
            "group": group,
            "rids": rid_list,
            "slots": slots,
        })
    return trees

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--in-dir', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--pattern', default='opts_bias_s7_t8_full_shard')
    p.add_argument('--shard-size', type=int, default=2048)
    p.add_argument('--n-shards', type=int, default=8)
    args = p.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    columns = ['tree_uids','tree_seqs','tree_rids','tree_pids','tree_branch_pos',
               'response_ids','global_indices','tree_rewards','tree_rounds']
    for sh in range(args.n_shards):
        path = Path(args.in_dir)/f'{args.pattern}{sh}.parquet'
        target = out/f'coef_{args.pattern}{sh}_max.pt'
        if target.exists():
            raise FileExistsError(f'Refusing to overwrite coefficients: {target}')
        df = pd.read_parquet(path, columns=columns)
        if len(df) != args.shard_size:
            raise ValueError(f'Incomplete shard: {path}')
        trees = [t for _, row in df.iterrows()
                 for t in build_tree_coefficients(row, args.shard_size)]
        payload = dict(trees=trees, shard=sh, shard_size=args.shard_size,
                       schema_version='exp1_v2', gen_path=str(path.resolve()),
                       credit_mode='return', return_schema=RETURN_SCHEMA)
        torch.save(payload, str(target)+'.tmp')
        os.replace(str(target)+'.tmp', target)
        print(target, flush=True)

if __name__ == '__main__':
    main()
