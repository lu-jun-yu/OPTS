"""Pure TreeRL tree-search and process-supervision utilities.

The implementation follows Algorithm 1 of TreeRL for the released training
configuration ``(M, N, L, T) = (6, 2, 1, 2)``.  The public TreeRL code calls
``-log p(y_t)`` entropy; we retain that sampled-token surprisal definition so
that the baseline matches the released implementation.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Hashable, Iterable, Sequence

import numpy as np
import torch


@dataclass(frozen=True)
class RolloutPlan:
    """Exact decomposition of a completed-rollout budget."""

    total_rollouts: int
    roots_per_prompt: int
    fork_points: int
    iterations: int
    branches_per_fork: int
    leaves_per_tree_prompt: int
    tree_prompt_count: int
    filler_prompt_count: int
    filler_root_count: int

    @property
    def base_prompt_count(self) -> int:
        return self.tree_prompt_count + self.filler_prompt_count

    @property
    def root_rollouts(self) -> int:
        return self.tree_prompt_count * self.roots_per_prompt + self.filler_root_count

    @property
    def branch_rollouts(self) -> int:
        return self.tree_prompt_count * self.roots_per_prompt * self.fork_points * self.iterations * self.branches_per_fork


def plan_rollout_budget(
    total_rollouts: int,
    roots_per_prompt: int = 6,
    fork_points: int = 2,
    iterations: int = 1,
    branches_per_fork: int = 2,
) -> RolloutPlan:
    """Match an arbitrary completed-rollout budget exactly.

    Full EPTree prompt groups use the requested ``(M, N, L, T)`` setting.  If
    the budget is not divisible by ``M(1+NLT)``, one final prompt receives the
    remaining number of independent roots.  This keeps the generated-rollout
    count exact without changing the TreeRL setting of any full tree group.
    """

    parameters = {
        "total_rollouts": total_rollouts,
        "roots_per_prompt": roots_per_prompt,
        "fork_points": fork_points,
        "iterations": iterations,
        "branches_per_fork": branches_per_fork,
    }
    for name, value in parameters.items():
        if int(value) != value or value < 1:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if iterations != 1:
        raise ValueError("The released TreeRL training setting uses L=1; this baseline currently enforces it")

    leaves_per_prompt = roots_per_prompt * (1 + fork_points * iterations * branches_per_fork)
    tree_prompt_count, remainder = divmod(total_rollouts, leaves_per_prompt)
    return RolloutPlan(
        total_rollouts=total_rollouts,
        roots_per_prompt=roots_per_prompt,
        fork_points=fork_points,
        iterations=iterations,
        branches_per_fork=branches_per_fork,
        leaves_per_tree_prompt=leaves_per_prompt,
        tree_prompt_count=tree_prompt_count,
        filler_prompt_count=int(remainder > 0),
        filler_root_count=remainder,
    )


@dataclass(frozen=True)
class BranchJob:
    """One suffix rollout forked from a sampled root response."""

    uid: Hashable
    parent_index: int
    parent_rid: Hashable
    selected_token_pos: int
    branch_pos: int
    surprisal: float
    child_rank: int


def select_entropy_branch_jobs(
    sampled_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    raw_prompt_lens: Sequence[int],
    uids: Sequence[Hashable],
    rids: Sequence[Hashable],
    eligible_indices: Iterable[int],
    fork_points: int,
    branches_per_fork: int,
    max_prompt_length: int,
    eligible_token_counts: Sequence[int] | None = None,
) -> list[BranchJob]:
    """Select TreeRL EPTree branch jobs using sampled-token surprisal.

    A selected token is regenerated, so token ``t`` maps to the inclusive
    prefix endpoint ``t-1``.  If a very short response has fewer than ``N``
    eligible positions, positions are repeated cyclically, matching the
    released implementation's fixed-size behavior.
    """

    if sampled_log_probs.ndim != 2 or response_mask.shape != sampled_log_probs.shape:
        raise ValueError("sampled_log_probs and response_mask must be same-shaped rank-2 tensors")
    if fork_points < 1 or branches_per_fork < 1:
        raise ValueError("fork_points and branches_per_fork must be positive")
    if eligible_token_counts is not None and len(eligible_token_counts) != sampled_log_probs.shape[0]:
        raise ValueError("eligible_token_counts must contain one entry per response")

    jobs: list[BranchJob] = []
    width = sampled_log_probs.shape[1]
    for parent_index in eligible_indices:
        valid: list[tuple[float, int]] = []
        raw_prompt_len = int(raw_prompt_lens[parent_index])
        eligible_token_count = (
            width if eligible_token_counts is None else int(eligible_token_counts[parent_index])
        )
        for token_pos in range(width):
            if not bool(response_mask[parent_index, token_pos]):
                continue
            if token_pos >= eligible_token_count:
                continue
            # The branch prompt contains the raw prompt plus tokens strictly
            # before the selected token.
            if raw_prompt_len + token_pos > max_prompt_length:
                continue
            log_prob = float(sampled_log_probs[parent_index, token_pos].item())
            if math.isfinite(log_prob):
                valid.append((-log_prob, token_pos))
        if not valid:
            raise ValueError(f"root row {parent_index} has no eligible EPTree branch position")

        valid.sort(key=lambda item: (-item[0], item[1]))
        selected = valid[:fork_points]
        while len(selected) < fork_points:
            selected.append(selected[len(selected) % len(valid)])

        for selected_rank, (surprisal, token_pos) in enumerate(selected):
            for child_rank in range(branches_per_fork):
                jobs.append(
                    BranchJob(
                        uid=uids[parent_index],
                        parent_index=int(parent_index),
                        parent_rid=rids[parent_index],
                        selected_token_pos=int(token_pos),
                        branch_pos=int(token_pos - 1),
                        surprisal=float(surprisal),
                        child_rank=selected_rank * branches_per_fork + child_rank,
                    )
                )
    return jobs


@dataclass(frozen=True)
class TreeLeaf:
    """One complete leaf path used by the TreeRL policy update."""

    uid: Hashable
    rid: Hashable
    parent_rid: Hashable | None
    branch_pos: int
    token_ids: tuple[int, ...]
    reward: float


def _process_signal(value: float, root_value: float, parent_value: float, descendant_count: int) -> float:
    if descendant_count < 1:
        raise ValueError("descendant_count must be positive")
    return ((value - root_value) + (value - parent_value)) / math.sqrt(descendant_count)


def compute_process_advantages(leaves: Sequence[TreeLeaf]) -> tuple[dict[Hashable, np.ndarray], dict[str, float]]:
    """Compute TreeRL ``(GA + LA) / sqrt(n)`` on complete leaf paths.

    The input uses the compact two-round representation: every root completion
    is a leaf, and every child row is a full logical leaf obtained by attaching
    its generated suffix to the parent's prefix.  The returned arrays align to
    each leaf's *complete* response tokens, which deliberately duplicates
    shared prefixes exactly as TreeRL does when extracting all root-to-leaf
    sequences for policy-gradient training.
    """

    if not leaves:
        raise ValueError("TreeRL requires at least one leaf")
    by_rid = {leaf.rid: leaf for leaf in leaves}
    if len(by_rid) != len(leaves):
        raise ValueError("TreeRL leaf rids must be unique")

    grouped: dict[Hashable, list[TreeLeaf]] = defaultdict(list)
    for leaf in leaves:
        grouped[leaf.uid].append(leaf)

    output: dict[Hashable, np.ndarray] = {}
    prompt_passes = []
    all_rewards = []

    for uid, prompt_leaves in grouped.items():
        root_leaves = [leaf for leaf in prompt_leaves if leaf.parent_rid is None]
        branch_leaves = [leaf for leaf in prompt_leaves if leaf.parent_rid is not None]
        if not root_leaves:
            raise ValueError(f"uid={uid!r} has no root leaves")
        for branch in branch_leaves:
            if branch.parent_rid not in by_rid:
                raise ValueError(f"branch {branch.rid!r} references missing parent {branch.parent_rid!r}")
            parent = by_rid[branch.parent_rid]
            if parent.uid != uid or parent.parent_rid is not None:
                raise ValueError("The released L=1 TreeRL setting requires every branch parent to be a root of the same uid")

        root_value = float(np.mean([leaf.reward for leaf in prompt_leaves]))
        prompt_passes.append(float(max(leaf.reward for leaf in prompt_leaves)))
        all_rewards.extend(float(leaf.reward) for leaf in prompt_leaves)

        branches_by_parent: dict[Hashable, list[TreeLeaf]] = defaultdict(list)
        for branch in branch_leaves:
            branches_by_parent[branch.parent_rid].append(branch)

        for root in root_leaves:
            root_tokens = root.token_ids
            children = branches_by_parent.get(root.rid, [])
            starts: dict[int, list[TreeLeaf]] = defaultdict(list)
            for child in children:
                start = child.branch_pos + 1
                if not 0 <= start <= len(root_tokens):
                    raise ValueError(
                        f"branch {child.rid!r} starts at {start}, outside parent length {len(root_tokens)}"
                    )
                if tuple(child.token_ids[:start]) != tuple(root_tokens[:start]):
                    raise ValueError(f"branch {child.rid!r} does not preserve its selected parent prefix")
                starts[start].append(child)

            split_positions = sorted(starts)
            root_path_adv = np.zeros(len(root_tokens), dtype=np.float32)
            parent_value_at_split: dict[int, float] = {}

            tree_descendants = [root, *children]
            current_start = 0
            current_value = float(np.mean([leaf.reward for leaf in tree_descendants]))
            current_count = len(tree_descendants)
            parent_value = root_value

            for split in split_positions:
                signal = _process_signal(current_value, root_value, parent_value, current_count)
                if split > current_start:
                    root_path_adv[current_start:split] = signal
                parent_value_at_split[split] = current_value

                future_children = [child for child in children if child.branch_pos + 1 > split]
                next_descendants = [root, *future_children]
                parent_value = current_value
                current_value = float(np.mean([leaf.reward for leaf in next_descendants]))
                current_count = len(next_descendants)
                current_start = split

            final_signal = _process_signal(current_value, root_value, parent_value, current_count)
            if current_start < len(root_tokens):
                root_path_adv[current_start:] = final_signal
            output[root.rid] = root_path_adv

            for child in children:
                split = child.branch_pos + 1
                child_signal = _process_signal(
                    float(child.reward),
                    root_value,
                    parent_value_at_split[split],
                    1,
                )
                child_adv = np.empty(len(child.token_ids), dtype=np.float32)
                if split:
                    child_adv[:split] = root_path_adv[:split]
                child_adv[split:] = child_signal
                output[child.rid] = child_adv

    if set(output) != set(by_rid):
        missing = set(by_rid) - set(output)
        raise AssertionError(f"failed to assign TreeRL advantages to leaves: {sorted(map(str, missing))}")

    metrics = {
        "treerl/prompt_count": float(len(grouped)),
        "treerl/leaf_count": float(len(leaves)),
        "treerl/root_count": float(sum(leaf.parent_rid is None for leaf in leaves)),
        "treerl/branch_count": float(sum(leaf.parent_rid is not None for leaf in leaves)),
        "treerl/mean_leaf_reward": float(np.mean(all_rewards)),
        "treerl/pass_rate": float(np.mean(prompt_passes)),
    }
    return output, metrics
