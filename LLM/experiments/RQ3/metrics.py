"""M=1 theory relative bias for the three methods plotted in RQ3."""
import argparse
import json
import math
from pathlib import Path

import torch


def load_groups(root, prefix, count):
    files = sorted(Path(root).glob(f'{prefix}_g*_rank0.pt'))
    if len(files) != count:
        raise ValueError(f'{root}: expected {count} complete groups, got {len(files)}')
    groups = [torch.load(p, map_location='cpu', mmap=True, weights_only=False) for p in files]
    for i, g in enumerate(groups):
        if (g.get('schema_version') != 'exp1_v2' or g['group'] != i
                or g['n_trees_shard'] != g['n_trees_total'] or g['n_trees_total'] <= 0):
            raise ValueError(f'Incomplete group: {files[i]}')
    return groups


def reduce(est, opts, refs, chunk=262144):
    """Compute theory/M=1 relative bias in FP64 chunks."""
    groups = est + opts + refs
    base = refs[0]
    size = base['accs']['s0'].numel()
    for g in groups:
        if (g['parameter_names'] != base['parameter_names']
                or g['n_trees_total'] != base['n_trees_total'] or g['model'] != base['model']):
            raise ValueError('Model, parameter order or prompt grid mismatch')
        if any(v.numel() != size for v in g['accs'].values()):
            raise ValueError('Gradient dimension mismatch')
    sources = {}
    for s in (0, 1, 3, 7):
        for method, gs, slot in (
            ('A', est, 's0' if s == 0 else f's{s}_ttpg'),
            ('B', est, 's0' if s == 0 else f's{s}_naive'),
            ('C2', opts, f'max_s{s}'),
        ):
            sources[f'{method}/s{s}'] = (gs, slot)
    squared_bias = dict.fromkeys(sources, 0.0)
    reference_norm2 = 0.0
    for start in range(0, size, chunk):
        end = min(start + chunk, size)
        star = sum(g['accs']['s0'][start:end].double() for g in refs) / sum(
            g['n_trees_total'] for g in refs)
        reference_norm2 += star.square().sum().item()
        for key, (gs, slot) in sources.items():
            mean = torch.stack([g['accs'][slot][start:end].double() / g['n_trees_total']
                                for g in gs]).mean(0)
            squared_bias[key] += (mean - star).square().sum().item()
    if reference_norm2 <= 0 or not math.isfinite(reference_norm2):
        raise ValueError('Invalid reference gradient norm')
    return dict(schema_version='rq3_v1', n_prompts=base['n_trees_total'],
                aggregations={'theory': {'metrics': {
                    key: {'1': {'bias': math.sqrt(value / reference_norm2)}}
                    for key, value in squared_bias.items()}}})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--grad', required=True)
    p.add_argument('--chunk', type=int, default=262144)
    args = p.parse_args()
    if args.chunk <= 0:
        p.error('--chunk must be positive')
    root = Path(args.grad)
    est = load_groups(root/'est', 'treegrad', 8)
    opts = load_groups(root/'opts', 'optsgrad', 8)
    refs = load_groups(root/'gstar', 'treegrad', 32)
    if any(g.get('credit_mode') != 'return' for g in opts):
        raise ValueError('RQ3 requires V=0 return gradients')
    result = reduce(est, opts, refs, args.chunk)
    target = root/'metrics.json'
    target.with_suffix('.tmp').write_text(json.dumps(result, indent=2, allow_nan=False))
    target.with_suffix('.tmp').replace(target)
    print(target)


if __name__ == '__main__':
    main()
