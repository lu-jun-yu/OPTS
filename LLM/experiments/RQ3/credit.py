"""CPU-only prefix-credit analysis of paired max/mean TreeGAE snapshots.

Distance zero is the parent token at branch_pos (the retained branch state).
Only shared response prefixes with a downstream branch are included. Root-only
resampling has no shared response prefix and is reported, not assigned distance 0.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def tree_advantages(row, snapshot, backup, lam):
    """Rebuild zero-value TreeGAE from topology and terminal rewards."""
    included = [j for j, r in enumerate(row['tree_rounds']) if r <= snapshot]
    rid_to_j = {row['tree_rids'][j]: j for j in included}
    children = defaultdict(lambda: defaultdict(list))
    for j in included:
        parent = row['tree_pids'][j]
        if parent in (None, 'None'):
            continue
        if parent not in rid_to_j:
            raise ValueError('Missing snapshot parent')
        pj = rid_to_j[parent]
        bp = int(row['tree_branch_pos'][j])
        if row['tree_uids'][pj] != row['tree_uids'][j]:
            raise ValueError('Cross-tree parent')
        if not 0 <= bp < len(row['response_ids'][pj]):
            raise ValueError('Invalid branch position')
        children[pj][bp].append(j)
    adv = {}
    for j in reversed(included):
        length = len(row['response_ids'][j])
        out = np.zeros(length, dtype=float)
        reward = float(row['tree_rewards'][j])
        for t in range(length - 1, -1, -1):
            successors = []
            if t + 1 < length:
                successors.append(out[t + 1])
            for child in children[j].get(t, ()):
                successors.append(adv[child][0])
            if successors:
                continuation = max(successors) if backup == 'max' else float(np.mean(successors))
            else:
                continuation = 0.0
            out[t] = (reward if t == length - 1 else 0.0) + lam * continuation
        adv[j] = out
    return adv, rid_to_j


def prefix_samples(row, snapshot, rebuilt=None):
    rounds = list(row['tree_rounds'])
    rids = list(row['tree_rids'])
    if len(set(rids)) != len(rids):
        raise ValueError('Duplicate response IDs')
    included = {rid: j for j, rid in enumerate(rids) if rounds[j] <= snapshot}
    if rebuilt is None:
        rebuilt = {
            lam: (tree_advantages(row, snapshot, 'max', lam)[0],
                  tree_advantages(row, snapshot, 'mean', lam)[0])
            for lam in (.999, 1.)
        }
    points = defaultdict(set)
    roots = 0
    for rid, j in included.items():
        parent = row['tree_pids'][j]
        if parent is None or parent == 'None':
            roots += int(rounds[j] > 0)
            continue
        if parent not in included:
            raise ValueError('Missing snapshot parent')
        bp = int(row['tree_branch_pos'][j])
        pj = included[parent]
        if not 0 <= bp < len(row['response_ids'][pj]):
            raise ValueError('Invalid parent-local branch position')
        if row['tree_uids'][j] != row['tree_uids'][pj]:
            raise ValueError('Cross-tree parent')
        points[pj].add(bp)
    samples = []
    for j, branches in points.items():
        branches = np.asarray(sorted(branches))
        positions = np.arange(branches[-1]+1)
        anchors = branches[np.searchsorted(branches, positions)]
        distances = anchors-positions
        credits = []
        for lam in (.999, 1.):
            max_adv, mean_adv = rebuilt[lam]
            a = np.asarray(max_adv[j], dtype=float)
            b = np.asarray(mean_adv[j], dtype=float)
            if len(a) != len(row['response_ids'][j]) or a.shape != b.shape:
                raise ValueError('Snapshot length mismatch')
            delta = a-b  # same critic baseline cancels exactly
            if not np.isfinite(delta).all():
                raise ValueError('Nonfinite credit')
            credits.append((delta[positions], delta[anchors]))
        samples.append((distances, credits))
    return samples, roots


def check_credit_identity(row, snapshot, lam, atol=1e-5, rtol=1e-4, rebuilt=None):
    """Reconstruct sum_u W(u)*lambda**(distance+1)*b(u), gamma=1.

    Uses only max marks on the RHS; observed max-minus-mean is used solely
    for comparison. Reverse topological summation equals the descendant sum
    without materializing the quadratic ancestor/descendant matrix.
    """
    included = {rid:j for j,rid in enumerate(row['tree_rids']) if row['tree_rounds'][j] <= snapshot}
    if len(included) != sum(r <= snapshot for r in row['tree_rounds']):
        raise ValueError('Duplicate response IDs')
    starts, maxima, means, children, root_nodes = {}, [], [], [], defaultdict(list)
    if rebuilt is None:
        rebuilt = (tree_advantages(row, snapshot, 'max', lam)[0],
                   tree_advantages(row, snapshot, 'mean', lam)[0])
    max_adv, mean_adv = rebuilt
    for rid,j in included.items():
        a = list(max_adv[j]); b = list(mean_adv[j])
        if len(a) != len(row['response_ids'][j]) or len(a) != len(b) or not a:
            raise ValueError('Empty or mismatched snapshot response')
        start = len(maxima); starts[rid] = start
        maxima.extend(a); means.extend(b)
        children.extend([[start+t+1] if t+1<len(a) else [] for t in range(len(a))])
    for rid,j in included.items():
        parent = row['tree_pids'][j]
        if parent is None or parent == 'None':
            root_nodes[row['tree_uids'][j]].append(starts[rid])
        else:
            if parent not in included:
                raise ValueError('Missing parent')
            pj = included[parent]; pos = int(row['tree_branch_pos'][j])
            if row['tree_uids'][pj] != row['tree_uids'][j] or not 0 <= pos < len(row['response_ids'][pj]):
                raise ValueError('Invalid parent edge')
            children[starts[parent]+pos].append(starts[rid])
    maxima,means = np.asarray(maxima,float),np.asarray(means,float)
    if not np.isfinite(maxima).all() or not np.isfinite(means).all():
        raise ValueError('Nonfinite snapshot')
    weights = np.zeros(len(maxima)); order = []
    for roots in root_nodes.values():
        for root in roots:
            weights[root] = 1/len(roots); order.append(root)
    cursor = 0
    while cursor < len(order):
        node = order[cursor]; cursor += 1
        for child in children[node]:
            weights[child] = weights[node]/len(children[node]); order.append(child)
        if len(order)>len(maxima):
            raise ValueError('Cyclic or duplicate edges')
    if len(order)!=len(maxima):
        raise ValueError('Disconnected/cyclic tree')
    reconstructed = np.zeros(len(maxima)); branch_count = 0
    for node in reversed(order):
        cs = children[node]
        if not cs:
            continue
        gap = 0.
        if len(cs)>1:
            marks = maxima[cs]
            gap = float(marks.max()-marks.mean()); branch_count += 1
        reconstructed[node] = lam*(weights[node]*gap + sum(reconstructed[c] for c in cs))
    observed = weights*(maxima-means)
    error = reconstructed-observed
    threshold = atol + rtol*np.maximum(np.abs(observed),np.abs(reconstructed))
    return dict(n_tokens=len(error),n_branch_points=branch_count,
                max_abs_error=float(np.abs(error).max(initial=0)),
                squared_error=float(error@error),squared_observed=float(observed@observed),
                n_outside_tolerance=int((np.abs(error)>threshold).sum()))


def summarize(rows, snapshots=(1,3,7), bin_width=16):
    sums = defaultdict(lambda: np.zeros(6))
    checks = {}
    diagnostics = {str(s): dict(prompts=0, prefix_tokens=0, root_resamples=0) for s in snapshots}
    for row in rows:
        for s in snapshots:
            rebuilt = {
                lam: (tree_advantages(row, s, 'max', lam)[0],
                      tree_advantages(row, s, 'mean', lam)[0])
                for lam in (.999, 1.)
            }
            for lam in (.999,1.):
                key = f's{s}_lambda{lam:g}'
                check = check_credit_identity(row,s,lam,rebuilt=rebuilt[lam])
                total = checks.setdefault(key,dict.fromkeys(check,0))
                for field,value in check.items():
                    total[field] = max(total[field],value) if field=='max_abs_error' else total[field]+value
            samples, roots = prefix_samples(row,s,rebuilt=rebuilt)
            diag = diagnostics[str(s)]
            diag['prompts'] += 1
            diag['root_resamples'] += roots
            for distances, credits in samples:
                diag['prefix_tokens'] += len(distances)
                # Keep d=0 separate; subsequent bins are 1..width, width+1..2width.
                bins = np.where(distances == 0, 0, (distances-1)//bin_width+1)
                for lam,(delta,anchor) in zip(('0.999','1'),credits):
                    for bucket in np.unique(bins):
                        take = bins == bucket
                        valid = take & (np.abs(anchor) > 1e-8)
                        sums[s,lam,int(bucket)] += [take.sum(),delta[take].sum(),
                            distances[take].sum(),valid.sum(),(delta[valid]/anchor[valid]).sum(),
                            (float(lam)**distances[valid]).sum()]
    records = []
    for (s,lam,bucket),(n,total,ds,nratio,ratio,expected) in sorted(sums.items()):
        records.append(dict(snapshot=s,lam=lam,bin=bucket,n_tokens=int(n),distance=ds/n,
            mean_credit=total/n,n_ratio_tokens=int(nratio),
            mean_anchor_ratio=ratio/nratio if nratio else None,
            expected_decay=expected/nratio if nratio else None))
    for check in checks.values():
        check['rmse'] = float(np.sqrt(check['squared_error']/max(check['n_tokens'],1)))
        check['relative_l2_error'] = float(np.sqrt(check['squared_error']/check['squared_observed'])) if check['squared_observed'] else None
        check['passed'] = check['n_outside_tolerance']==0
    return dict(schema_version='exp2_v2',bin_width=bin_width,diagnostics=diagnostics,records=records,
        identity_check=dict(gamma=1,atol=1e-5,rtol=1e-4,results=checks,
                            definition='W*(Amax-Amean) versus descendant sum of W(u)*lambda^(distance+1)*b(u); equal child weights, roots split within uid'),
        notes='Token-weighted descriptive means, not independent observations. Raw credit bins have composition effects; anchor-normalized ratios isolate within-prefix decay. Zero anchors excluded only from ratios. Root resamples have no shared response prefix. Stored paired snapshots use identical values, so max-minus-mean advantages equal the extra return credit. No monotonicity or positivity is imposed.')




def main():
    p = argparse.ArgumentParser()
    p.add_argument('--in-dir',required=True)
    p.add_argument('--pattern',default='opts_bias_s7_t8_full_shard')
    p.add_argument('--n-shards',type=int,default=8)
    p.add_argument('--start-shard',type=int,default=0)
    p.add_argument('--end-shard',type=int,default=None)
    p.add_argument('--shard-size',type=int,default=2048)
    p.add_argument('--bin-width',type=int,default=16)
    p.add_argument('--out-dir',required=True)
    args = p.parse_args()
    if min(args.n_shards,args.shard_size,args.bin_width) <= 0:
        p.error('Counts and bin width must be positive')
    end_shard = args.n_shards if args.end_shard is None else args.end_shard
    if not (0 <= args.start_shard < end_shard <= args.n_shards):
        p.error('Invalid shard range')
    files = [Path(args.in_dir)/f'{args.pattern}{s}.parquet' for s in range(args.start_shard, end_shard)]
    columns = ['tree_rids','tree_pids','tree_uids','tree_rounds','tree_branch_pos',
               'response_ids','tree_rewards']
    for file in files:
        parquet = pq.ParquetFile(file)
        if parquet.metadata.num_rows != args.shard_size or not set(columns).issubset(parquet.schema_arrow.names):
            raise ValueError(f'Incomplete or incompatible shard: {file}')
    def rows():
        for file in files:
            print(f'Reading {file}',flush=True)
            for batch in pq.ParquetFile(file).iter_batches(batch_size=8,columns=columns):
                yield from batch.to_pylist()
    result = summarize(rows(),bin_width=args.bin_width)
    result['inputs'] = [dict(path=str(f.resolve()),size=f.stat().st_size,mtime_ns=f.stat().st_mtime_ns) for f in files]
    out = Path(args.out_dir)
    out.mkdir(parents=True,exist_ok=True)
    target = out/'credit_stats.json'
    target.with_suffix('.tmp').write_text(json.dumps(result,indent=2,allow_nan=False))
    target.with_suffix('.tmp').replace(target)
    if not all(c['passed'] for c in result['identity_check']['results'].values()):
        raise ValueError(f'Credit identity check failed; inspect {target}')
    print(f'E2 finished: {out}',flush=True)


if __name__ == '__main__':
    main()
