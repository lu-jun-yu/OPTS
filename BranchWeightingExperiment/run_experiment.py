#!/usr/bin/env python3
"""One command: verify exact gradients, run both initializations, and export figures."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('MPLBACKEND', 'Agg')
ROOT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache' / 'matplotlib'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs')
    parser.add_argument('--plot-only', action='store_true', help='Replot the existing JSON; no optimization or data rewrite')
    parser.add_argument('--verify-only', action='store_true', help='Run the exact arithmetic and finite-difference checks only')
    args = parser.parse_args()
    from plot_results import render
    import experiment
    import numpy as np
    import matplotlib
    output = args.output.resolve()
    stem = output / 'exact_branch_weighting_initializations'
    if args.plot_only:
        render(json.loads(stem.with_suffix('.json').read_text()), output)
        print(f'Figure: {stem.with_suffix(".pdf")}')
        return
    started = time.perf_counter()
    print('1/3: Checking exact chain/tree gradients and finite differences...', flush=True)
    checks = experiment.verify()
    if args.verify_only:
        print(json.dumps(checks, indent=2))
        return
    output.mkdir(parents=True, exist_ok=True)
    results = {}
    plotted_methods = ('weighted', 'naive', 'treerl_style_sqrt')
    for initial_p in (.2, .8):
        print(f'2/3: Running p0={initial_p}, K={experiment.KS.tolist()}, 50,000 updates...', flush=True)
        updates, curves, summary, error = experiment.run(50000, .5, initial_p)
        selected = {method: {str(k): curves[:, experiment.METHODS.index(method), i].tolist()
                            for i, k in enumerate(experiment.KS)} for method in plotted_methods}
        if not np.isfinite(curves).all():
            raise RuntimeError('Nonfinite optimization result')
        weighted_error = float(np.max(np.abs(curves[:, 0] - curves[:, 1])))
        if weighted_error > 1e-12:
            raise RuntimeError(f'Weighted/true policy trajectories disagree: {weighted_error}')
        results[str(initial_p)] = dict(curves=selected,
            summary=[row for row in summary if row['method'] in plotted_methods],
            max_weighted_identity_error=error,
            max_weighted_vs_true_return_error=weighted_error)
    payload = dict(protocol=dict(K=experiment.KS.tolist(), initial_probabilities=[.2, .8],
        audit_initial_probabilities=experiment.INITIAL_PS.tolist(),
        learning_rate=.5, updates_count=50000, dtype='float64',
        environment='Two-step binary MDP; reward is one only for actions (0,1)',
        policy='One Bernoulli logit shared across both steps; p=sigmoid(theta)',
        gradient='Exact expectation of each tree estimator; true gradient computed analytically',
        common_conditions='Same exact state advantages, original-root denominator, initialization, optimizer and step size; no whitening/clipping/momentum',
        weights=dict(prefix=1, weighted_suffix='1/(K+1)', naive_suffix=1, heuristic_suffix='1/sqrt(K+1)'),
        branching='K is fixed before sampling; one first transition and K+1 on-policy suffixes',
        treerl_style='Inverse-square-root aggregation rule only; NOT full TreeRL GA/LA or entropy-based branching',
        finite_run='Analytic limits are separate from measured finite-update results. Some biased runs have not converged by 50,000 updates.',
        scope='Constructed population-expectation mechanism example; not a finite-sample-budget benchmark or full algorithm comparison'),
        checks=checks, updates=updates, initializations=results)
    print('3/3: Rendering the 2 x 5 figure...', flush=True)
    render(payload, output)
    payload['elapsed_seconds'] = time.perf_counter()-started
    stem.with_suffix('.json').write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n')
    with (output / 'summary.csv').open('w', newline='') as f:
        rows=[row for item in results.values() for row in item['summary']]
        writer=csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (output/'checks.json').write_text(json.dumps(checks, indent=2)+'\n')
    environment = dict(python=sys.version, platform=platform.platform(), numpy=np.__version__,
        matplotlib=matplotlib.__version__, elapsed_seconds=payload['elapsed_seconds'],
        code_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                     for name in ['experiment.py','exact_oracle.py','plot_results.py','run_experiment.py']})
    (output/'environment.json').write_text(json.dumps(environment, indent=2)+'\n')
    print(f'Done in {payload["elapsed_seconds"]:.2f}s. Outputs: {output}', flush=True)
    for suffix in ('.png','.pdf','.json'):
        print(stem.with_suffix(suffix))


if __name__ == '__main__':
    main()
