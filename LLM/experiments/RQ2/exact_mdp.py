#!/usr/bin/env python3
"""E5: exact OPTS monotonicity audit in a finite deterministic MDP.

The environment is a depth-four binary tree.  Transitions are deterministic,
rewards occur only on the terminal transition, the behavior policy has full
support, and V^pi is computed exactly with fractions.  We then enumerate every
possible initial rollout and every possible on-policy suffix sampled by OPTS,
so both J_lambda(pi_j^S) and J(pi_j^S) are expectations rather than Monte Carlo
estimates.  We evaluate both full outcome-reward advantages and the truncated
value-guidance signal used by the LLM implementation.

The search implementation mirrors the production trajectory-tree semantics.
Under max backup, only the current greedy path can be expanded and its backed-up
advantages cannot decrease.  A child that fails to strictly beat the incumbent
can therefore never affect a later decision.  We prune such dominated branches
exactly and represent each behaviorally equivalent tree by its greedy path.
Continuation wins a max-backup tie, and rebranching occurs only when the largest
unpenalized performance-difference estimate is strictly above the zero baseline.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from fractions import Fraction
from functools import lru_cache
from itertools import product
from pathlib import Path
from typing import Iterable, Optional


ActionPath = tuple[int, ...]
State = ActionPath


POLICY_PROBABILITIES = (
    Fraction(1, 4),
    Fraction(1, 3),
    Fraction(1, 2),
    Fraction(2, 3),
    Fraction(3, 4),
)
DEFAULT_LAMBDAS = (
    Fraction(0, 1),
    Fraction(3, 10),
    Fraction(3, 5),
    Fraction(19, 20),
)
GUIDANCE_MODES = ("full", "truncated")


def fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def lambda_label(value: Fraction) -> str:
    if value == 0:
        return "0"
    return f"{float(value):g}"


def all_action_paths(length: int) -> Iterable[ActionPath]:
    return product((0, 1), repeat=length)


Tree = ActionPath


@dataclass(frozen=True)
class ExactBinaryTreeMDP:
    depth: int
    terminal_rewards: dict[ActionPath, Fraction]
    p_right: dict[State, Fraction]
    gamma: Fraction = Fraction(1, 1)
    _value_cache: dict[State, Fraction] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def action_probability(self, state: State, action: int) -> Fraction:
        p = self.p_right[state]
        return p if action == 1 else 1 - p

    def transition_reward(self, state: State, action: int) -> Fraction:
        next_state = state + (action,)
        if len(next_state) == self.depth:
            return self.terminal_rewards[next_state]
        return Fraction(0, 1)

    def value(self, state: State) -> Fraction:
        if state in self._value_cache:
            return self._value_cache[state]
        if len(state) == self.depth:
            return Fraction(0, 1)
        total = Fraction(0, 1)
        for action in (0, 1):
            prob = self.action_probability(state, action)
            reward = self.transition_reward(state, action)
            total += prob * (reward + self.gamma * self.value(state + (action,)))
        self._value_cache[state] = total
        return total

    def td_residual(self, path: ActionPath, depth: int) -> Fraction:
        state = path[:depth]
        action = path[depth]
        reward = self.transition_reward(state, action)
        return reward + self.gamma * self.value(path[: depth + 1]) - self.value(state)

    def path_probability(self, path: ActionPath, start_depth: int = 0) -> Fraction:
        probability = Fraction(1, 1)
        for depth in range(start_depth, self.depth):
            probability *= self.action_probability(path[:depth], path[depth])
        return probability

    def path_return(self, path: ActionPath) -> Fraction:
        total = Fraction(0, 1)
        discount = Fraction(1, 1)
        for depth, action in enumerate(path):
            total += discount * self.transition_reward(path[:depth], action)
            discount *= self.gamma
        return total

    def lambda_return(self, path: ActionPath, lam: Fraction) -> Fraction:
        total = self.value(())
        discount = Fraction(1, 1)
        gamma_lam = self.gamma * lam
        for depth in range(self.depth):
            total += discount * self.td_residual(path, depth)
            discount *= gamma_lam
        return total

def make_configuration(seed: int, depth: int) -> ExactBinaryTreeMDP:
    """Create one reproducible environment-policy configuration.

    Reward permutations and state-wise policy probabilities use separate random
    streams.  This keeps the two design choices independent while retaining a
    compact public seed convention.
    """

    leaves = list(all_action_paths(depth))
    reward_values = [Fraction(i, len(leaves) - 1) for i in range(len(leaves))]
    reward_rng = random.Random(seed)
    reward_rng.shuffle(reward_values)
    terminal_rewards = dict(zip(leaves, reward_values))

    policy_rng = random.Random(1_000_000 + seed)
    states = [state for state_depth in range(depth) for state in all_action_paths(state_depth)]
    p_right = {state: policy_rng.choice(POLICY_PROBABILITIES) for state in states}
    return ExactBinaryTreeMDP(
        depth=depth,
        terminal_rewards=terminal_rewards,
        p_right=p_right,
    )


class ExactOPTSEnumerator:
    def __init__(self, mdp: ExactBinaryTreeMDP, lam: Fraction, guidance_mode: str) -> None:
        if guidance_mode not in GUIDANCE_MODES:
            raise ValueError(f"unknown guidance mode: {guidance_mode}")
        self.mdp = mdp
        self.lam = lam
        self.guidance_mode = guidance_mode
        self.pathwise_guidance_comparisons = 0
        self.pathwise_guidance_violations = 0
        self.pathwise_target_violations = 0

    def guidance_td_residual(self, path: ActionPath, depth: int) -> Fraction:
        if self.guidance_mode == "full":
            return self.mdp.td_residual(path, depth)

        # Match value-guided OPTS: all observed rewards are zero and the last
        # valid position's exact value is used as a pseudo terminal reward.
        # Consequently the last TD residual is exactly zero.
        if depth + 1 == self.mdp.depth:
            return Fraction(0, 1)
        state = path[:depth]
        return self.mdp.gamma * self.mdp.value(path[: depth + 1]) - self.mdp.value(state)

    @lru_cache(maxsize=None)
    def advantage(self, path: ActionPath, depth: int) -> Fraction:
        """TreeGAE along the retained greedy path under the search signal."""

        delta = self.guidance_td_residual(path, depth)
        successor = (
            self.advantage(path, depth + 1)
            if depth + 1 < self.mdp.depth
            else Fraction(0, 1)
        )
        return delta + self.mdp.gamma * self.lam * successor

    @lru_cache(maxsize=None)
    def target_objective(self, tree: Tree) -> Fraction:
        """The theorem's full-reward lambda-return objective."""

        return self.mdp.lambda_return(tree, self.lam)

    @lru_cache(maxsize=None)
    def guidance_objective(self, tree: Tree) -> Fraction:
        backed_up = self.mdp.value(()) + self.advantage(tree, 0)
        if self.guidance_mode == "full":
            direct = self.target_objective(tree)
            if direct != backed_up:
                raise AssertionError(f"max-backup mismatch: direct={direct}, backup={backed_up}")
        return backed_up

    @lru_cache(maxsize=None)
    def true_return(self, tree: Tree) -> Fraction:
        return self.mdp.path_return(tree)

    @lru_cache(maxsize=None)
    def selected_depth(self, tree: Tree) -> Optional[int]:
        performance_differences = [Fraction(0, 1)] * self.mdp.depth
        running = Fraction(0, 1)
        for depth in range(self.mdp.depth - 1, -1, -1):
            running = -self.advantage(tree, depth) + self.mdp.gamma * running
            performance_differences[depth] = running

        best_depth = max(range(self.mdp.depth), key=performance_differences.__getitem__)
        if performance_differences[best_depth] <= 0:
            return None
        return best_depth

    def suffix_outcomes(self, prefix: ActionPath) -> Iterable[tuple[ActionPath, Fraction]]:
        start_depth = len(prefix)
        for suffix in all_action_paths(self.mdp.depth - start_depth):
            path = prefix + tuple(suffix)
            yield path, self.mdp.path_probability(path, start_depth=start_depth)

    @lru_cache(maxsize=None)
    def transition_distribution(self, tree: Tree) -> tuple[tuple[Tree, Fraction], ...]:
        selected_depth = self.selected_depth(tree)
        if selected_depth is None:
            return ((tree, Fraction(1, 1)),)

        prefix = tree[:selected_depth]
        old_guidance = self.guidance_objective(tree)
        old_target = self.target_objective(tree)
        outcomes: dict[Tree, Fraction] = defaultdict(Fraction)
        for path, probability in self.suffix_outcomes(prefix):
            new_tree = (
                path
                if self.advantage(path, selected_depth) > self.advantage(tree, selected_depth)
                else tree
            )
            new_guidance = self.guidance_objective(new_tree)
            new_target = self.target_objective(new_tree)
            self.pathwise_guidance_comparisons += 1
            if new_guidance < old_guidance:
                self.pathwise_guidance_violations += 1
            if new_target < old_target:
                self.pathwise_target_violations += 1
            outcomes[new_tree] += probability

        if sum(outcomes.values(), Fraction(0, 1)) != 1:
            raise AssertionError("suffix probabilities do not sum exactly to one")
        return tuple(outcomes.items())

    def enumerate(self, max_search: int) -> dict:
        distribution: dict[Tree, Fraction] = defaultdict(Fraction)
        for path in all_action_paths(self.mdp.depth):
            path = tuple(path)
            distribution[path] += self.mdp.path_probability(path)
        if sum(distribution.values(), Fraction(0, 1)) != 1:
            raise AssertionError("initial trajectory probabilities do not sum exactly to one")

        objectives: list[Fraction] = []
        guidance_objectives: list[Fraction] = []
        true_returns: list[Fraction] = []
        state_counts: list[int] = []
        terminated_mass: list[Fraction] = []

        for budget in range(max_search + 1):
            mass = sum(distribution.values(), Fraction(0, 1))
            if mass != 1:
                raise AssertionError(f"budget {budget}: distribution mass is {mass}, expected 1")
            objectives.append(
                sum(
                    (prob * self.target_objective(tree) for tree, prob in distribution.items()),
                    Fraction(0, 1),
                )
            )
            guidance_objectives.append(
                sum(
                    (prob * self.guidance_objective(tree) for tree, prob in distribution.items()),
                    Fraction(0, 1),
                )
            )
            true_returns.append(
                sum((prob * self.true_return(tree) for tree, prob in distribution.items()), Fraction(0, 1))
            )
            state_counts.append(len(distribution))
            terminated_mass.append(
                sum(
                    (prob for tree, prob in distribution.items() if self.selected_depth(tree) is None),
                    Fraction(0, 1),
                )
            )

            if budget == max_search:
                break
            next_distribution: dict[Tree, Fraction] = defaultdict(Fraction)
            for tree, tree_probability in distribution.items():
                for new_tree, conditional_probability in self.transition_distribution(tree):
                    next_distribution[new_tree] += tree_probability * conditional_probability
            distribution = next_distribution

        baseline = self.mdp.value(())
        if objectives[0] != baseline or guidance_objectives[0] != baseline or true_returns[0] != baseline:
            raise AssertionError(
                "s=0 must equal J(pi): "
                f"target={objectives[0]}, guidance={guidance_objectives[0]}, "
                f"return={true_returns[0]}, baseline={baseline}"
            )
        objective_violations = sum(a > b for a, b in zip(objectives, objectives[1:]))
        guidance_violations = sum(
            a > b for a, b in zip(guidance_objectives, guidance_objectives[1:])
        )
        return_violations = sum(a > b for a, b in zip(true_returns, true_returns[1:]))
        if guidance_violations or self.pathwise_guidance_violations:
            raise AssertionError(
                "guidance monotonicity failure: "
                f"expected={guidance_violations}, pathwise={self.pathwise_guidance_violations}"
            )
        if self.guidance_mode == "full" and (
            objective_violations or return_violations or self.pathwise_target_violations
        ):
            raise AssertionError(
                "full-advantage theorem audit failed: "
                f"objective={objective_violations}, return={return_violations}, "
                f"pathwise_target={self.pathwise_target_violations}"
            )

        return {
            "baseline": baseline,
            "objective": objectives,
            "guidance_objective": guidance_objectives,
            "true_return": true_returns,
            "state_counts": state_counts,
            "terminated_mass": terminated_mass,
            "objective_violations": objective_violations,
            "guidance_violations": guidance_violations,
            "return_violations": return_violations,
            "objective_strict_steps": sum(a < b for a, b in zip(objectives, objectives[1:])),
            "guidance_strict_steps": sum(
                a < b for a, b in zip(guidance_objectives, guidance_objectives[1:])
            ),
            "return_strict_steps": sum(a < b for a, b in zip(true_returns, true_returns[1:])),
            "pathwise_guidance_comparisons": self.pathwise_guidance_comparisons,
            "pathwise_guidance_violations": self.pathwise_guidance_violations,
            "pathwise_target_violations": self.pathwise_target_violations,
        }


def configuration_json(seed: int, mdp: ExactBinaryTreeMDP) -> dict:
    def path_key(path: ActionPath) -> str:
        return "root" if not path else "".join(map(str, path))

    return {
        "seed": seed,
        "terminal_rewards": {
            path_key(path): fraction_text(reward) for path, reward in sorted(mdp.terminal_rewards.items())
        },
        "p_right": {path_key(state): fraction_text(prob) for state, prob in sorted(mdp.p_right.items())},
        "exact_value": fraction_text(mdp.value(())),
        "value": float(mdp.value(())),
    }


def mean_fraction(values: Iterable[Fraction]) -> Fraction:
    values = list(values)
    return sum(values, Fraction(0, 1)) / len(values)


def write_csv(output_path: Path, results: list[dict], max_search: int) -> None:
    fieldnames = [
        "seed",
        "guidance",
        "lambda",
        "search_budget",
        "baseline",
        "j_lambda",
        "guidance_objective",
        "true_return",
        "j_lambda_improvement",
        "guidance_objective_improvement",
        "true_return_improvement",
        "distribution_states",
        "terminated_mass",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            baseline = result["baseline"]
            for budget in range(max_search + 1):
                writer.writerow(
                    {
                        "seed": result["seed"],
                        "guidance": result["guidance"],
                        "lambda": lambda_label(result["lambda"]),
                        "search_budget": budget,
                        "baseline": f"{float(baseline):.12f}",
                        "j_lambda": f"{float(result['objective'][budget]):.12f}",
                        "guidance_objective": f"{float(result['guidance_objective'][budget]):.12f}",
                        "true_return": f"{float(result['true_return'][budget]):.12f}",
                        "j_lambda_improvement": f"{float(result['objective'][budget] - baseline):.12f}",
                        "guidance_objective_improvement": f"{float(result['guidance_objective'][budget] - baseline):.12f}",
                        "true_return_improvement": f"{float(result['true_return'][budget] - baseline):.12f}",
                        "distribution_states": result["state_counts"][budget],
                        "terminated_mass": f"{float(result['terminated_mass'][budget]):.12f}",
                    }
                )


def make_plot(output_dir: Path, results: list[dict], lambdas: tuple[Fraction, ...], max_search: int) -> None:
    import math
    import statistics

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter

    budgets = list(range(max_search + 1))
    display_budgets = [budget for budget in (0, 1, 3, 7, 15) if budget <= max_search]
    budget_positions = [math.log2(budget + 1) for budget in budgets]
    display_positions = [math.log2(budget + 1) for budget in display_budgets]
    colors = ("#4C78A8", "#55A868", "#8172B3", "#C44E52")
    styles = {
        lam: {
            "color": color,
            "linestyle": "-",
            "linewidth": 1.65,
            "solid_capstyle": "round",
            "zorder": 3,
        }
        for lam, color in zip(lambdas, colors)
    }
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "font.size": 10.0,
            "text.color": "#303030",
            "axes.labelcolor": "#303030",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 4, figsize=(10.8, 2.65), sharey=True)
    left = 0.060
    right = 0.995
    bottom = 0.205
    top = 0.665
    axis_gap = 0.030
    axis_width = (right - left - 3 * axis_gap) / 4
    axis_lefts = tuple(
        left + index * (axis_width + axis_gap)
        for index in range(4)
    )
    for axis, axis_left in zip(axes, axis_lefts):
        axis.set_position([axis_left, bottom, axis_width, top - bottom])
    panels = (
        ("full", "objective", "(a)", r"$J_\lambda(\pi_j^S)-J(\pi)$"),
        ("full", "true_return", "(b)", r"$J(\pi_j^S)-J(\pi)$"),
        ("truncated", "objective", "(c)", r"$J_\lambda(\pi_j^S)-J(\pi)$"),
        ("truncated", "true_return", "(d)", r"$J(\pi_j^S)-J(\pi)$"),
    )

    handles_by_lambda = {}
    plot_order = lambdas
    global_upper = 0.0
    for axis, (guidance, metric, heading, metric_title) in zip(axes, panels):
        panel_curves = {}
        for lam in lambdas:
            selected = [
                result
                for result in results
                if result["guidance"] == guidance and result["lambda"] == lam
            ]
            curves = [
                [result[metric][budget] - result["baseline"] for budget in budgets]
                for result in selected
            ]
            means = [
                float(mean_fraction(curve[position] for curve in curves))
                for position in budgets
            ]
            standard_deviations = []
            for position in budgets:
                values = [float(curve[position]) for curve in curves]
                standard_deviations.append(
                    statistics.stdev(values) if len(values) > 1 else 0.0
                )
            lower = [mean - std for mean, std in zip(means, standard_deviations)]
            upper = [mean + std for mean, std in zip(means, standard_deviations)]
            global_upper = max(global_upper, max(upper))
            panel_curves[lam] = (means, lower, upper)
            axis.fill_between(
                budget_positions,
                lower,
                upper,
                color=styles[lam]["color"],
                alpha=0.14,
                linewidth=0,
                zorder=1,
            )

        for lam in plot_order:
            means, _, _ = panel_curves[lam]
            handle, = axis.plot(
                budget_positions,
                means,
                label=rf"$\lambda={lambda_label(lam)}$",
                **styles[lam],
            )
            if axis is axes[0]:
                handles_by_lambda[lam] = handle

        axis.set_title(f"{heading}  {metric_title}", fontsize=10.0, fontweight="bold", pad=7)
        axis.set_xlim(-0.18, math.log2(max_search + 1) + 0.18)
        axis.set_xticks(display_positions, labels=[str(budget) for budget in display_budgets])
        axis.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        axis.set_xlabel(r"Search budget $j$", fontsize=10.0, labelpad=3)
        axis.axhline(0.0, color="#7E858A", linestyle="--", linewidth=0.8, zorder=2)
        axis.set_facecolor("white")
        axis.grid(
            axis="y",
            color="#C8D0D5",
            linestyle=(0, (3, 3)),
            linewidth=0.65,
            alpha=0.78,
        )
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_linewidth(0.7)
        axis.spines["bottom"].set_linewidth(0.7)
        axis.spines["left"].set_color("#A5ADB3")
        axis.spines["bottom"].set_color("#A5ADB3")
        axis.tick_params(
            axis="both",
            colors="#5D6870",
            labelsize=9.0,
            length=2.8,
            width=0.65,
            pad=2.3,
        )

    upper_limit = math.ceil((global_upper + 0.015) / 0.05) * 0.05
    for axis in axes:
        axis.set_ylim(-0.01, upper_limit)
    axes[0].set_ylabel("Improvement", fontsize=10.5, fontweight="bold", labelpad=5)
    legend_order = tuple(lam for lam in lambdas if lam in handles_by_lambda)
    fig.legend(
        [handles_by_lambda[lam] for lam in legend_order],
        [handles_by_lambda[lam].get_label() for lam in legend_order],
        frameon=False,
        fontsize=9.2,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        handlelength=2.5,
        columnspacing=1.6,
    )
    left_pair_center = (axes[0].get_position().x0 + axes[1].get_position().x1) / 2
    right_pair_center = (axes[2].get_position().x0 + axes[3].get_position().x1) / 2
    for x, title in (
        (left_pair_center, "Reward-guided OPTS"),
        (right_pair_center, "Value-guided OPTS"),
    ):
        fig.text(
            x,
            0.790,
            title,
            ha="center",
            va="center",
            fontsize=10.5,
            fontweight="bold",
        )
    for suffix in ("png", "pdf"):
        fig.savefig(
            output_dir / f"e5_exact_monotonicity.{suffix}",
            dpi=240,
            bbox_inches="tight",
            pad_inches=0.035,
        )
    plt.close(fig)


def serializable_result(result: dict) -> dict:
    return {
        "seed": result["seed"],
        "guidance": result["guidance"],
        "lambda": lambda_label(result["lambda"]),
        "lambda_exact": fraction_text(result["lambda"]),
        "baseline": float(result["baseline"]),
        "baseline_exact": fraction_text(result["baseline"]),
        "j_lambda": [float(value) for value in result["objective"]],
        "j_lambda_exact": [fraction_text(value) for value in result["objective"]],
        "guidance_objective": [float(value) for value in result["guidance_objective"]],
        "guidance_objective_exact": [
            fraction_text(value) for value in result["guidance_objective"]
        ],
        "true_return": [float(value) for value in result["true_return"]],
        "true_return_exact": [fraction_text(value) for value in result["true_return"]],
        "distribution_states": result["state_counts"],
        "terminated_mass": [float(value) for value in result["terminated_mass"]],
        "terminated_mass_exact": [fraction_text(value) for value in result["terminated_mass"]],
        "objective_violations": result["objective_violations"],
        "guidance_violations": result["guidance_violations"],
        "return_violations": result["return_violations"],
        "objective_strict_steps": result["objective_strict_steps"],
        "guidance_strict_steps": result["guidance_strict_steps"],
        "return_strict_steps": result["return_strict_steps"],
        "pathwise_guidance_comparisons": result["pathwise_guidance_comparisons"],
        "pathwise_guidance_violations": result["pathwise_guidance_violations"],
        "pathwise_target_violations": result["pathwise_target_violations"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/e5_exact_deterministic"))
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--num-configs", type=int, default=32)
    parser.add_argument("--max-search", type=int, default=15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.depth < 1:
        raise ValueError("depth must be positive")
    if args.num_configs < 1:
        raise ValueError("num-configs must be positive")
    if args.max_search < 0:
        raise ValueError("max-search must be nonnegative")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    configurations: list[dict] = []
    results: list[dict] = []

    for seed in range(args.num_configs):
        mdp = make_configuration(seed=seed, depth=args.depth)
        configurations.append(configuration_json(seed, mdp))
        for guidance in GUIDANCE_MODES:
            for lam in DEFAULT_LAMBDAS:
                enumerator = ExactOPTSEnumerator(
                    mdp=mdp,
                    lam=lam,
                    guidance_mode=guidance,
                )
                result = enumerator.enumerate(max_search=args.max_search)
                result["seed"] = seed
                result["guidance"] = guidance
                result["lambda"] = lam
                results.append(result)
                print(
                    f"seed={seed:02d} guidance={guidance:>9} lambda={lambda_label(lam):>5} "
                    f"states@{args.max_search}={result['state_counts'][-1]:3d} "
                    f"delta_Jlambda={float(result['objective'][-1] - result['baseline']):.8f} "
                    f"delta_J={float(result['true_return'][-1] - result['baseline']):.8f}"
                )

    comparisons_per_mode_metric = args.num_configs * len(DEFAULT_LAMBDAS) * args.max_search
    audit = {}
    for guidance in GUIDANCE_MODES:
        selected = [result for result in results if result["guidance"] == guidance]
        audit[guidance] = {
            "expected_monotonicity_comparisons_per_metric": comparisons_per_mode_metric,
            "j_lambda_violations": sum(result["objective_violations"] for result in selected),
            "guidance_objective_violations": sum(
                result["guidance_violations"] for result in selected
            ),
            "true_return_violations": sum(result["return_violations"] for result in selected),
            "j_lambda_strict_comparisons": sum(
                result["objective_strict_steps"] for result in selected
            ),
            "guidance_objective_strict_comparisons": sum(
                result["guidance_strict_steps"] for result in selected
            ),
            "true_return_strict_comparisons": sum(
                result["return_strict_steps"] for result in selected
            ),
            "pathwise_guidance_comparisons": sum(
                result["pathwise_guidance_comparisons"] for result in selected
            ),
            "pathwise_guidance_violations": sum(
                result["pathwise_guidance_violations"] for result in selected
            ),
            "pathwise_j_lambda_violations": sum(
                result["pathwise_target_violations"] for result in selected
            ),
        }

    aggregate = {}
    for guidance in GUIDANCE_MODES:
        aggregate[guidance] = {}
        for lam in DEFAULT_LAMBDAS:
            selected = [
                result
                for result in results
                if result["guidance"] == guidance and result["lambda"] == lam
            ]
            aggregate[guidance][lambda_label(lam)] = {
                "mean_j_lambda": [
                    float(mean_fraction(result["objective"][budget] for result in selected))
                    for budget in range(args.max_search + 1)
                ],
                "mean_guidance_objective": [
                    float(
                        mean_fraction(
                            result["guidance_objective"][budget] for result in selected
                        )
                    )
                    for budget in range(args.max_search + 1)
                ],
                "mean_true_return": [
                    float(mean_fraction(result["true_return"][budget] for result in selected))
                    for budget in range(args.max_search + 1)
                ],
                "mean_j_lambda_improvement": [
                    float(
                        mean_fraction(
                            result["objective"][budget] - result["baseline"]
                            for result in selected
                        )
                    )
                    for budget in range(args.max_search + 1)
                ],
                "mean_guidance_objective_improvement": [
                    float(
                        mean_fraction(
                            result["guidance_objective"][budget] - result["baseline"]
                            for result in selected
                        )
                    )
                    for budget in range(args.max_search + 1)
                ],
                "mean_true_return_improvement": [
                    float(
                        mean_fraction(
                            result["true_return"][budget] - result["baseline"]
                            for result in selected
                        )
                    )
                    for budget in range(args.max_search + 1)
                ],
            }

    summary = {
        "protocol": {
            "environment": "full deterministic binary tree",
            "depth": args.depth,
            "nonterminal_states": 2**args.depth - 1,
            "terminal_states": 2**args.depth,
            "reward_support": [f"{i}/{2**args.depth - 1}" for i in range(2**args.depth)],
            "reward_assignment": "independent fixed permutation for each seed",
            "p_right_support": [fraction_text(value) for value in POLICY_PROBABILITIES],
            "p_left": "1 - p_right",
            "policy_assignment": "independent uniform state-wise draws from p_right_support",
            "seeds": list(range(args.num_configs)),
            "gamma": "1/1",
            "lambdas": [fraction_text(value) for value in DEFAULT_LAMBDAS],
            "guidance_modes": {
                "full": "terminal outcome reward plus exact-value TD residuals",
                "truncated": (
                    "zero intermediate rewards; last-position exact value used as pseudo terminal "
                    "reward, making the final TD residual zero"
                ),
            },
            "search_budgets": list(range(args.max_search + 1)),
            "baseline": "0/1",
            "xi": "0/1",
            "backup": "max",
            "batch_size": 1,
            "tie_rule": "retain incumbent; earliest child among child ties",
            "expectation": "exact enumeration with fractions",
        },
        "audit": audit,
        "aggregate": aggregate,
        "configurations": configurations,
        "per_configuration": [serializable_result(result) for result in results],
    }

    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=False)
        handle.write("\n")
    write_csv(args.output_dir / "per_configuration.csv", results, args.max_search)
    make_plot(args.output_dir, results, DEFAULT_LAMBDAS, args.max_search)

    for guidance in GUIDANCE_MODES:
        mode_audit = audit[guidance]
        print(
            f"audit[{guidance}]: "
            f"J_lambda violations={mode_audit['j_lambda_violations']}/{comparisons_per_mode_metric}, "
            f"J violations={mode_audit['true_return_violations']}/{comparisons_per_mode_metric}, "
            f"guidance violations={mode_audit['guidance_objective_violations']}/{comparisons_per_mode_metric}"
        )
    print(f"outputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
