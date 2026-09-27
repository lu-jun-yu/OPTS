"""Coverage and actual sampled response-token budgets for the matched experiment."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mask', required=True)
    p.add_argument('--bias-root', required=True)
    p.add_argument('--fixed-data', nargs='+', required=True)
    p.add_argument('--tag', default='full')
    p.add_argument('--shard-size', type=int, default=2048)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    mask = np.load(args.mask)
    points = ['s0','s1','s3','s7']
    n = mask['k'].shape[1]
    success = {(side,point): np.zeros(n, dtype=bool) for side in ['A_B','C1_C2'] for point in points}
    tokens = {key: 0 for key in success}
    searches = {key: 0 for key in success}
    fixed_paths = [Path(path) for path in args.fixed_data]
    seen_fixed = set()
    for path in fixed_paths:
        for row in pd.read_parquet(path).itertuples(index=False):
            g, pi = int(row.group), int(row.prompt_idx)
            key_id = (g, pi)
            if key_id in seen_fixed:
                raise ValueError(f'Duplicate fixed prompt/group: {key_id}')
            seen_fixed.add(key_id)
            ks = [0] + list(mask['k'][g,pi])
            for point,k in zip(points,ks):
                key = ('A_B',point)
                success[key][pi] |= row.backbone_reward > 0 or any(r > 0 for r in row.suffix_rewards[:k])
                tokens[key] += len(row.backbone_ids) + sum(len(ids) for ids in row.suffix_ids[:k])
                searches[key] += int(k)
    if len(seen_fixed) != 8 * n:
        raise ValueError(
            f'Incomplete fixed grid: {len(seen_fixed)} != {8 * n}'
        )

    for sh in range(8):
        path = Path(args.bias_root)/f'opts_bias_s7_t8_{args.tag}_shard{sh}.parquet'
        cols = ['tree_seqs','tree_rounds','tree_rewards','response_ids']
        for local,row in enumerate(pd.read_parquet(path, columns=cols).itertuples(index=False)):
            pi = sh*args.shard_size+local
            for ts,r,reward,ids in zip(row.tree_seqs,row.tree_rounds,row.tree_rewards,row.response_ids):
                g = int(ts)//args.shard_size
                limits = [0,1,3,7]
                for point,limit in zip(points,limits):
                    if r <= limit:
                        key = ('C1_C2',point)
                        success[key][pi] |= reward > 0
                        tokens[key] += len(ids)
                        searches[key] += int(r > 0)
    out = {side: {point: dict(prompt_success_rate=float(success[side,point].mean()),
            generated_response_tokens=int(tokens[side,point]), searches=int(searches[side,point]))
            for point in points} for side in ['A_B','C1_C2']}
    out['n_prompts'] = n
    out['notes'] = 'Coverage is any positive terminal reward across eight trees per retained prompt. Tokens exclude reused prefixes and prompts; budgets need not be token-equal. A short chain is not redrawn.'
    target = Path(args.out)
    target.with_suffix('.tmp').write_text(json.dumps(out, indent=2))
    target.with_suffix('.tmp').replace(target)
    print(target)


if __name__ == '__main__':
    main()
