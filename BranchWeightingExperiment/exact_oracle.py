"""Self-contained rational-arithmetic finite-MDP oracle.

Adapted from the accompanying project's exact finite-MDP utility (Apache-2.0).
The oracle is independent of the vectorized optimization implementation.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from fractions import Fraction
ActionPath = tuple[int, ...]
State = ActionPath

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
