"""Exact, CPU-only population-gradient experiment; no external project imports.

Same validated numerical routines as the accompanying experiment. The only
methods retained are exact PG, weighted, naive, and inverse-sqrt weighting.
"""
from __future__ import annotations
from fractions import Fraction
from itertools import product
import math
import numpy as np
from exact_oracle import ExactBinaryTreeMDP

KS = np.array([0, 3, 15, 63, 255], dtype=int)
INITIAL_PS = np.array([.1, .2, .35, .5, .65, .8, .9])
METHODS = ('exact_pg', 'weighted', 'naive', 'treerl_style_sqrt')


def _process_signal(value, root_value, parent_value, descendant_count):
    """Native TreeRL formula, used ONLY to check its inverse-sqrt factor.

    No empirical TreeRL advantages or entropy-based sampler are used here.
    """
    if descendant_count < 1:
        raise ValueError('descendant_count must be positive')
    return ((value-root_value)+(value-parent_value))/math.sqrt(descendant_count)


def sigmoid(theta):
    return np.exp(-np.logaddexp(0., -np.asarray(theta, dtype=np.float64)))

def exact_quantities(theta):
    """Enumerate all four chain paths with exact state advantages (float64)."""
    p = sigmoid(theta)
    a0 = np.array([0., 0., 1., 1.])
    a1 = np.array([0., 1., 0., 1.])
    reward = np.array([0., 1., 0., 0.])
    prob0 = np.where(a0 == 1, p[..., None], 1-p[..., None])
    prob1 = np.where(a1 == 1, p[..., None], 1-p[..., None])
    path_prob = prob0 * prob1
    j = (path_prob * reward).sum(axis=-1)
    next_value = np.where(a0 == 0, p[..., None], 0.)
    first_adv = next_value - j[..., None]
    second_adv = reward - next_value
    g0 = (path_prob * first_adv * (a0-p[..., None])).sum(axis=-1)
    g1 = (path_prob * second_adv * (a1-p[..., None])).sum(axis=-1)
    # Independent closed-form derivative of J(theta)=sigmoid(theta)(1-sigmoid(theta)).
    true_gradient = p*(1-p)*(1-2*p)
    return p, j, g0, g1, true_gradient

def fraction_oracle(p):
    """Reuse the existing exact finite-MDP implementation as an independent oracle."""
    mdp = ExactBinaryTreeMDP(depth=2,
        terminal_rewards={(0, 0): Fraction(0), (0, 1): Fraction(1),
                          (1, 0): Fraction(0), (1, 1): Fraction(0)},
        p_right={(): p, (0,): p, (1,): p})
    g = [Fraction(0), Fraction(0)]
    for path in product((0, 1), repeat=2):
        pr = mdp.path_probability(path)
        for depth in range(2):
            g[depth] += pr * mdp.td_residual(path, depth) * (path[depth]-p)
    return mdp.value(()), g

def enumerate_forests(p, m):
    """Enumerate the full joint tree law for small m, including every suffix."""
    weighted = Fraction(0)
    naive = Fraction(0)
    v0 = p*(1-p)
    for a0 in (0, 1):
        q0 = p if a0 == 0 else Fraction(0)
        h0 = (q0-v0)*(a0-p)
        for children in product((0, 1), repeat=m):
            probability = p if a0 else 1-p
            suffix_sum = Fraction(0)
            for a1 in children:
                probability *= p if a1 else 1-p
                reward = Fraction(int((a0, a1) == (0, 1)))
                suffix_sum += (reward-q0)*(a1-p)
            weighted += probability*(h0+suffix_sum/m)
            naive += probability*(h0+suffix_sum)
    return weighted, naive

def verify():
    max_weighted_error = 0.
    max_finite_difference_error = 0.
    comparisons = 0
    for numerator in range(1, 10):
        rational_p = Fraction(numerator, 10)
        p = float(rational_p)
        theta = math.log(p/(1-p))
        _, j, g0, g1, true = exact_quantities(theta)
        value, parts = fraction_oracle(rational_p)
        assert abs(float(j)-float(value)) < 1e-14
        assert abs(float(g0)-float(parts[0])) < 1e-14
        assert abs(float(g1)-float(parts[1])) < 1e-14
        max_weighted_error = max(max_weighted_error, abs(float(g0+g1-true)))
        h = 1e-5
        fd = (exact_quantities(theta+h)[1]-exact_quantities(theta-h)[1])/(2*h)
        max_finite_difference_error = max(max_finite_difference_error, abs(float(fd-true)))
        for m in (1, 2, 4):
            w, n = enumerate_forests(rational_p, m)
            assert w == sum(parts)
            assert n == parts[0]+m*parts[1]
            comparisons += 1
    assert max_weighted_error < 1e-14
    assert max_finite_difference_error < 1e-9
    # Reuse the native TreeRL kernel to check its inverse-sqrt descendant factor.
    # Remove the common GA+LA factor of two, retaining only the aggregation rule.
    max_sqrt_error = max(abs(_process_signal(1., 0., 0., int(k+1))/2
                             - 1/math.sqrt(int(k+1))) for k in KS)
    assert max_sqrt_error < 1e-14
    return dict(treerl_kernel='local inverse-square-root factor check',
                max_treerl_inverse_sqrt_kernel_error=max_sqrt_error,
                reused_oracle='exact_oracle.ExactBinaryTreeMDP',
                exact_rational_forest_checks=comparisons,
                max_weighted_vs_true_gradient_error=max_weighted_error,
                max_finite_difference_error=max_finite_difference_error)

def run(steps, lr, initial_p=.2):
    matches = np.flatnonzero(np.isclose(INITIAL_PS, initial_p, atol=1e-12, rtol=0))
    if len(matches) != 1:
        raise ValueError(f"initial_p must be one of {INITIAL_PS.tolist()}")
    main_index = int(matches[0])
    # Every method/K combination is run at EVERY initialization; no seed selection.
    logits = np.log(INITIAL_PS/(1-INITIAL_PS))
    theta = np.broadcast_to(logits, (len(METHODS), len(KS), len(logits))).copy()
    snapshots = sorted(set(range(201)) | set(range(0, steps+1, 25)) | {steps})
    saved = []
    snapshot_set = set(snapshots)
    max_identity_error = 0.
    m = (KS+1)[:, None]
    for step in range(steps+1):
        p, j, g0, g1, true = exact_quantities(theta)
        max_identity_error = max(max_identity_error, float(np.max(np.abs(g0+g1-true))))
        if step in snapshot_set:
            saved.append(j[:, :, main_index].copy())
        if step == steps:
            break
        gradient = np.empty_like(theta)
        gradient[0] = true[0]
        gradient[1] = g0[1]+g1[1]
        gradient[2] = g0[2]+m*g1[2]
        # Controlled TreeRL-style weighting: same exact A, same prefix weight 1,
        # same original-root denominator; suffix weights are 1/sqrt(m).
        gradient[3] = g0[3]+np.sqrt(m)*g1[3]
        theta += lr*gradient
    assert np.isfinite(theta).all()
    assert np.max(np.abs(theta[0]-theta[1])) < 1e-10
    assert max_identity_error < 1e-14
    expected_p = (KS+1)/(KS+2)
    expected_return = expected_p*(1-expected_p)
    summary = []
    for k_index, k in enumerate(KS):
        for method_index, method in enumerate(METHODS):
            returns = j[method_index, k_index]
            if method in ('exact_pg', 'weighted'):
                target = .25
            elif method == 'treerl_style_sqrt':
                root_m = math.sqrt(int(k+1))
                target = root_m/(1+root_m)**2
            else:
                target = expected_return[k_index]
            summary.append(dict(K=int(k), suffixes=int(k+1), method=method,
                                initial_p=float(initial_p), final_p=float(p[method_index, k_index, main_index]),
                                final_return=float(returns[main_index]), analytic_limit=float(target),
                                all_initializations_min_return=float(returns.min()),
                                all_initializations_max_return=float(returns.max()),
                                maximum_limit_error=float(np.abs(returns-target).max())))
    return snapshots, np.stack(saved), summary, max_identity_error
