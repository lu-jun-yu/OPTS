# Exact branch-weighting experiment

A standalone, CPU-only mechanism experiment with an exactly known policy gradient.
The runner verifies the estimators, performs the optimization, and produces the
2-row x 5-column figure. No GPU, PyTorch, VeRL, model checkpoint, dataset, API key,
or parent-project checkout is required.

## One-command reproduction

Requirements: Python 3.10+ with `venv` and `pip`. Internet access is needed only
for installing NumPy and Matplotlib on the first run.

```bash
python3 run.py
```

On Windows use `python run.py`. On Linux/macOS the equivalent executable entry
point is `bash run.sh`. Run these commands inside this folder; `run.sh` also works
from any other working directory. A private `.venv` is created in this folder;
the global Python environment is not modified. Subsequent runs reuse it.

With dependencies already installed, the environment-free alternative is:

```bash
python3 run_experiment.py
```

CPU computation takes roughly 10--20 seconds on the tested machine, excluding
the first dependency installation. Runtime varies with the machine.

Replot the saved data without changing or recomputing numerical results:

```bash
python3 run.py --plot-only
```

Run only the exact-arithmetic and finite-difference checks:

```bash
python3 run.py --verify-only
```

## Files and outputs

- `experiment.py`: the common environment, exact estimators, and gradient ascent.
- `exact_oracle.py`: independent rational-arithmetic finite-MDP value computation.
- `plot_results.py`: plotting only, using the stored numerical curves.
- `run_experiment.py`: validation, computation, output, and figure generation.
- `run.py` / `run.sh`: private-environment setup and one-command execution.
- `requirements.txt`: the only external dependencies (NumPy and Matplotlib).

Results are written to `outputs/` by default:

- `exact_branch_weighting_initializations.png`: preview figure.
- `exact_branch_weighting_initializations.pdf`: vector figure for a manuscript.
- `exact_branch_weighting_initializations.json`: all plotted numerical curves,
  experiment settings, validation checks, and finite-update summaries.
- `summary.csv`: final return versus the separately identified analytic limit.
- `checks.json`: independent numerical checks.
- `environment.json`: Python/library versions, runtime, and source-file hashes.

Use `--output /path/to/results` to select another output directory. The included
outputs can be inspected before running anything.

## Environment and exact target

There are two transitions and two actions, 0 and 1. Terminal reward is 1 only
for the sequence `(0,1)`; all other sequences receive 0. Both steps use the same
Bernoulli policy parameter, `p = sigmoid(theta)`. Therefore

    J(theta) = p(1-p),
    dJ/dtheta = p(1-p)(1-2p).

The maximum **within this shared-parameter policy class** is `J=1/4` at `p=1/2`.
An unrestricted policy with separate step parameters could achieve reward 1;
that is not the policy class studied here.

A tree has one first transition and `m=K+1` independent on-policy second
transitions. K is fixed before sampling any action. There is no selection based
on future tokens, final trajectory length, or observed reward. This satisfies
the prefix-measurability condition of the branch aggregation lemma.

Values and advantages are recomputed exactly under each method's current policy at every update. Their numerical values naturally change as policies diverge; the advantage definition is identical.

All methods use the same exact state advantage `A^pi`, first-transition weight
1, and fixed-original-tree-count normalization. Let

    g0 = -p^2(1-p),
    g1 =  p(1-p)^2.

The methods differ only in the weight of each suffix:

| Method | Each suffix weight | Expected gradient |
|---|---|---|
| Branch-weighted (TTPG) | `1/m` | `g0+g1` |
| Naive | `1` | `g0+m*g1` |
| Heuristic (TreeRL-style) | `1/sqrt(m)` | `g0+sqrt(m)*g1` |

**TreeRL-style denotes only the controlled inverse-square-root aggregation
rule. It is not a reproduction or performance evaluation of the complete
TreeRL algorithm, its empirical GA/LA advantages, or its entropy-based branch
selection.**

## What is actually computed

The learning curves use the **exact population expectation** of each estimator
at its current policy, rather than a finite Monte Carlo estimate. Four possible
complete chains are enumerated; linearity of expectation accounts for all
independent suffixes. The true gradient is computed separately from the analytic
derivative of J. This removes finite-reference noise and critic-estimation error.

For independent verification, the script also:

1. Computes values and gradient contributions using `Fraction` arithmetic.
2. Fully enumerates 27 small joint tree distributions and checks exact recovery.
3. Checks the analytic derivative with central finite differences.
4. Checks equality of true-gradient and weighted trajectories throughout learning.

These are population-gradient diagnostics, not a comparison under a fixed budget
of sampled rollouts. There are no sample seeds or confidence intervals to report.
The plots are deterministic; they do not contain simulated error bars.

## Fixed optimization protocol

- K: `0, 3, 15, 63, 255` (the five columns).
- Main initial policies: `p0=0.2` and `p0=0.8` (the two rows).
- Learning rate: `0.5` for every method and K.
- Updates: `50,000` plain gradient-ascent steps.
- Float64 arithmetic; gamma=lambda=1; exact advantage.
- No advantage whitening, PPO clipping, optimizer momentum, or fitted critic.
- Additional audit initial policies: `0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9`.
  They are all run and their final-return ranges are retained; none is selected
  according to whether it produces a preferred outcome.

The horizontal axis is the number of gradient updates, displayed with a
symmetric-logarithmic scale (linear near zero and logarithmic farther away).
Sampling points are dense over the first 200 updates and every 25 updates later.
The vertical axis is the exact expected return of the current policy.

## Interpreting the curves

With `m=K+1`, the interior stationary policies are

    weighted: p*=1/2,
    naive:    p*=m/(1+m),
    heuristic:p*=sqrt(m)/(1+sqrt(m)).

Weighted aggregation preserves the true objective's gradient. The two other
rules change its stationary point. Starting at p0=0.2 can produce an early
near-optimal peak followed by degradation. Starting at p0=0.8 shows that the
biased updates need not pass through the optimal policy at all.

Finite-update results are not automatically converged results. In particular,
with K=255 and p0=0.2, the raw naive update moves the logit far into saturation
under the common learning rate; it has **not** reached its analytic stationary
point after 50,000 updates. The JSON/CSV report the measured final return and
the analytic limit separately. Do not interpret that saturation alone as proof
of a wrong stationary point: the stationary-point difference follows directly
from the formulas above.

This constructed example establishes a possible failure mechanism. It does not
claim that unbiased aggregation universally improves finite-budget downstream
performance or that complete TreeRL/TreePO algorithms must behave like these
controlled aggregation rules.

## Suggested figure caption

Exact expected-return trajectories under prefix-measurable branching. Columns vary the number of additional suffixes K; rows use initial Bernoulli probabilities 0.2 and 0.8. All methods use exact current-policy advantages, a common learning rate of 0.5, and 50,000 gradient-ascent updates; only suffix aggregation weights differ. Branch weighting coincides with the analytic true policy gradient. The inverse-square-root heuristic is a controlled TreeRL-style weighting rule, not the full TreeRL algorithm. Curves are deterministic population-expectation computations, and finite-update endpoints need not equal analytic stationary limits.

## Reproduction validation

The complete one-command entry point was tested from a separate copy of this folder, with no parent-project imports and a fresh Python 3.14 virtual environment. It installed NumPy 2.5.5 and Matplotlib 3.11.2, passed all numerical checks, and generated the figure in about 8.3 seconds after installation. Its 30 plotted curves were compared with the original Python 3.10 / NumPy 2.2.6 / Matplotlib 3.10.9 run and agreed within 1e-12. See `outputs/reproduction_check.json` for the measured difference and environments.
