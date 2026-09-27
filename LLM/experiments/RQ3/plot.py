"""The two paper RQ3 panels from saved E1 metrics/coverage and E2 credit bins."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


def plot(metrics, coverage, credit, output):
    if not all(c['passed'] for c in credit['identity_check']['results'].values()):
        raise ValueError('E2 credit identity checks must pass before plotting')
    if 'n_prompts' in metrics and metrics['n_prompts'] != coverage['n_prompts']:
        raise ValueError('Gradient and coverage prompt grids differ')
    values = metrics['aggregations']['theory']['metrics']
    rounds = (0, 1, 3, 7)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), layout='constrained')
    coordinates = {}
    for method, side, label, color in (
        ('A', 'A_B', 'Fixed + TTPG', '#B5475D'),
        ('B', 'A_B', 'Fixed + NaivePG', '#6F7F8C'),
        ('C2', 'C1_C2', 'OPTS + TTPG', '#4F7C8D'),
    ):
        baseline = values[f'{method}/s0']['1']['bias']
        x = np.array([values[f'{method}/s{s}']['1']['bias'] - baseline for s in rounds])
        y = 100*np.array([coverage[side][f's{s}']['prompt_success_rate']
                          - coverage[side]['s0']['prompt_success_rate'] for s in rounds])
        if not (np.isfinite(x).all() and np.isfinite(y).all()):
            raise ValueError('Nonfinite plot coordinates')
        axes[0].plot(x, y, '-o', label=label, color=color, markersize=4)
        for s, xx, yy in zip(rounds[1:], x[1:], y[1:]):
            axes[0].annotate(str(s), (xx, yy), xytext=(4, 4), textcoords='offset points', fontsize=8)
        coordinates[method] = {'rounds': list(rounds), 'relative_bias_difference': x.tolist(),
                               'coverage_gain_pp': y.tolist()}
    axes[0].set(title='(a) Coverage–bias trade-off', xlabel='Relative bias difference from s=0',
                ylabel='Prompt coverage gain (pp)')
    axes[0].legend(fontsize=8)
    colors = {1: '#6F7F8C', 3: '#4F7C8D', 7: '#B5475D'}
    for s, color in colors.items():
        for lam, style in [('1', '-'), ('0.999', '--')]:
            rows = sorted((r for r in credit['records'] if r['snapshot']==s and r['lam']==lam),
                          key=lambda r: r['distance'])
            if not rows:
                raise ValueError(f'Missing prefix credit: s={s}, lambda={lam}')
            axes[1].plot([r['distance'] for r in rows], [r['mean_credit'] for r in rows],
                         style, color=color)
    axes[1].set(title='(b) Max−mean prefix credit', xlabel='Tokens to first downstream branch',
                ylabel='Mean extra prefix credit')
    handles = [Line2D([0], [0], color=c, label=f's={s}') for s, c in colors.items()]
    handles += [Line2D([0], [0], color='black', linestyle=style, label=f'λ={lam}')
                for lam, style in [('1', '-'), ('0.999', '--')]]
    axes[1].legend(handles=handles, fontsize=8)
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
        ax.grid(alpha=.2)
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
    output.with_suffix('.json').write_text(json.dumps(coordinates, indent=2, allow_nan=False))
    return coordinates


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--metrics', required=True)
    p.add_argument('--coverage', required=True)
    p.add_argument('--credit', required=True)
    p.add_argument('--output', default='results/rq3/figures/mechanism.pdf')
    args = p.parse_args()
    plot(*(json.loads(Path(path).read_text()) for path in
           (args.metrics, args.coverage, args.credit)), args.output)


if __name__ == '__main__':
    main()
