import sys
import unittest
from pathlib import Path

import torch


CLEANRL_DIR = Path(__file__).resolve().parents[1] / "cleanrl" / "cleanrl"
sys.path.insert(0, str(CLEANRL_DIR))

from opts_ttpo_core_unbiased_v3 import compute_tree_gae, select_next_states  # noqa: E402


def branch_metadata(parents):
    state_branches = torch.ones_like(parents)
    root_branch_counts = [{} for _ in range(parents.shape[1])]
    for env_idx in range(parents.shape[1]):
        for step in range(parents.shape[0]):
            parent = int(parents[step, env_idx].item())
            if parent < 0:
                root_branch_counts[env_idx][parent] = (
                    root_branch_counts[env_idx].get(parent, 0) + 1
                )
            else:
                state_branches[parent, env_idx] += 1
        state_branches[:, env_idx] -= 1
        state_branches[:, env_idx].clamp_(min=1)
    return state_branches, root_branch_counts


def select(
    rewards,
    values,
    parents,
    trees,
    advantages=None,
    estimates=None,
    counts=None,
    terminated_envs=None,
    affected_tree_ids=None,
    skip_init_search=None,
    max_search=1,
    gamma=0.9,
):
    num_envs = values.shape[1]
    advantages = torch.zeros_like(values) if advantages is None else advantages
    estimates = [{} for _ in range(num_envs)] if estimates is None else estimates
    counts = [{} for _ in range(num_envs)] if counts is None else counts
    terminated_envs = list(range(num_envs)) if terminated_envs is None else terminated_envs
    affected_tree_ids = [-1] * len(terminated_envs) if affected_tree_ids is None else affected_tree_ids
    skip_init_search = [False] * num_envs if skip_init_search is None else skip_init_search
    state_branches, root_branch_counts = branch_metadata(parents)
    selected = select_next_states(
        terminated_envs=terminated_envs,
        current_step=values.shape[0] - 1,
        rewards=rewards,
        search_advantages=advantages,
        values=values,
        parent_indices=parents,
        state_branches=state_branches,
        tree_indices=trees,
        search_count=counts,
        max_search=max_search,
        root_branch_counts=root_branch_counts,
        terminal_estimates=estimates,
        skip_init_search=skip_init_search,
        affected_tree_ids=affected_tree_ids,
        gamma=gamma,
    )
    return selected, estimates, counts


class UnbiasedV3TreeSearchTest(unittest.TestCase):
    def test_search_uses_separate_lambda_one_advantages(self):
        rewards = torch.tensor([[1.0], [2.0]])
        values = torch.zeros((2, 1))
        dones = torch.tensor([[False], [True]])
        parents = torch.tensor([[-1], [0]])
        trees = torch.full((2, 1), -1, dtype=torch.long)
        train_advantages = torch.zeros((2, 1))
        search_advantages = torch.zeros((2, 1))

        compute_tree_gae(
            1, 0, rewards, values, dones, parents, train_advantages, 0.9, 0.5
        )
        compute_tree_gae(
            1, 0, rewards, values, dones, parents, search_advantages, 0.9, 1.0
        )

        self.assertAlmostEqual(train_advantages[0, 0].item(), 1.9, places=6)
        self.assertAlmostEqual(search_advantages[0, 0].item(), 2.8, places=6)
        _, estimates, _ = select(
            rewards, values, parents, trees, advantages=search_advantages
        )
        self.assertAlmostEqual(estimates[0][-1], 2.8, places=6)

    def test_terminal_estimate_averages_root_advantages(self):
        rewards = torch.zeros((2, 1))
        values = torch.tensor([[10.0], [10.0]])
        parents = torch.tensor([[-1], [-1]])
        trees = torch.tensor([[-1], [-1]])
        advantages = torch.tensor([[2.0], [6.0]])

        _, estimates, _ = select(
            rewards, values, parents, trees, advantages=advantages
        )
        self.assertAlmostEqual(estimates[0][-1], 14.0, places=6)

    def test_node_m_is_discounted_prefix_reward_plus_bootstrap_value(self):
        rewards = torch.tensor([[2.0], [0.0]])
        values = torch.tensor([[10.0], [6.0]])
        parents = torch.tensor([[-1], [0]])
        trees = torch.full((2, 1), -1, dtype=torch.long)

        selected, _, _ = select(
            rewards,
            values,
            parents,
            trees,
            estimates=[{}, {-1: 5.5}],
            gamma=0.5,
        )
        self.assertEqual(selected, [1])

    def test_tree_prefers_minimum_candidate_depth(self):
        rewards = torch.zeros((3, 1))
        values = torch.tensor([[10.0], [8.0], [11.0]])
        parents = torch.tensor([[-1], [0], [1]])
        trees = torch.full((3, 1), -1, dtype=torch.long)

        selected, _, _ = select(
            rewards, values, parents, trees, estimates=[{}, {-1: 9.5}]
        )
        self.assertEqual(selected, [1])

    def test_tree_prefers_maximum_m_at_same_minimum_depth(self):
        rewards = torch.zeros((3, 1))
        values = torch.tensor([[10.0], [8.0], [9.0]])
        parents = torch.tensor([[-1], [0], [0]])
        trees = torch.full((3, 1), -1, dtype=torch.long)

        selected, _, _ = select(
            rewards, values, parents, trees, estimates=[{}, {-1: 9.5}]
        )
        self.assertEqual(selected, [2])

    def test_environment_prefers_maximum_m_across_trees(self):
        rewards = torch.zeros((4, 1))
        values = torch.tensor([[10.0], [8.0], [10.0], [9.0]])
        parents = torch.tensor([[-1], [0], [-2], [2]])
        trees = torch.tensor([[-1], [-1], [-2], [-2]])

        selected, _, counts = select(
            rewards,
            values,
            parents,
            trees,
            estimates=[{-1: 10.0}, {-1: 9.5}],
            affected_tree_ids=[-2],
        )
        self.assertEqual(selected, [3])
        self.assertEqual(counts[0], {-2: 1})

    def test_global_leave_one_out_is_input_order_independent(self):
        rewards = torch.zeros((1, 2))
        values = torch.tensor([[8.0, 10.0]])
        parents = torch.tensor([[-1, -1]])
        trees = torch.tensor([[-1, -1]])
        advantages = torch.tensor([[0.0, 4.0]])

        def run(order):
            selected, estimates, _ = select(
                rewards,
                values,
                parents,
                trees,
                advantages=advantages,
                terminated_envs=order,
                affected_tree_ids=[-1 for _ in order],
            )
            return dict(zip(order, selected)), estimates

        forward, estimates = run([0, 1])
        reverse, _ = run([1, 0])
        self.assertEqual(forward, reverse)
        self.assertEqual(forward, {0: 0, 1: -2})
        self.assertEqual(estimates, [{-1: 8.0}, {-1: 14.0}])

    def test_incomplete_initial_tree_is_excluded(self):
        rewards = torch.zeros((1, 2))
        values = torch.tensor([[10.0, 10.0]])
        parents = torch.tensor([[-1, -1]])
        trees = torch.tensor([[-1, -1]])
        estimates = [{-1: 99.0}, {}]

        selected, estimates, _ = select(
            rewards,
            values,
            parents,
            trees,
            advantages=torch.tensor([[0.0, 4.0]]),
            estimates=estimates,
            skip_init_search=[True, False],
        )
        self.assertNotIn(-1, estimates[0])
        self.assertEqual(selected, [-2, -2])

    def test_search_exhausted_tree_remains_in_baseline(self):
        rewards = torch.zeros((1, 2))
        values = torch.tensor([[10.0, 10.0]])
        parents = torch.tensor([[-1, -1]])
        trees = torch.tensor([[-1, -1]])

        selected, estimates, _ = select(
            rewards,
            values,
            parents,
            trees,
            advantages=torch.tensor([[0.0, 4.0]]),
            estimates=[{-1: 10.0}, {}],
            counts=[{-1: 1}, {}],
            terminated_envs=[1],
            affected_tree_ids=[-1],
        )
        self.assertEqual(selected, [-2])
        self.assertEqual(estimates, [{-1: 10.0}, {-1: 14.0}])

    def test_cached_estimate_changes_only_when_tree_completes(self):
        rewards = torch.zeros((1, 2))
        values = torch.tensor([[10.0, 10.0]])
        parents = torch.tensor([[-1, -1]])
        trees = torch.tensor([[-1, -1]])
        advantages = torch.tensor([[5.0, 0.0]])
        estimates = [{-1: 7.0}, {}]

        select(
            rewards,
            values,
            parents,
            trees,
            advantages=advantages,
            estimates=estimates,
            terminated_envs=[1],
            affected_tree_ids=[-1],
        )
        self.assertEqual(estimates[0][-1], 7.0)

        select(
            rewards,
            values,
            parents,
            trees,
            advantages=advantages,
            estimates=estimates,
            terminated_envs=[0],
            affected_tree_ids=[-1],
        )
        self.assertEqual(estimates[0][-1], 15.0)


if __name__ == "__main__":
    unittest.main()
