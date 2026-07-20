import sys
import unittest
from pathlib import Path

import torch


CLEANRL_DIR = Path(__file__).resolve().parents[1] / "cleanrl" / "cleanrl"
sys.path.insert(0, str(CLEANRL_DIR))

from opts_ttpo_core_unbiased import select_next_states  # noqa: E402


def branch_metadata(parents):
    state_branches = torch.ones_like(parents)
    root_branch_counts = [{} for _ in range(parents.shape[1])]
    for env_idx in range(parents.shape[1]):
        for step in range(parents.shape[0]):
            parent = int(parents[step, env_idx].item())
            if parent < 0:
                root_branch_counts[env_idx][parent] = root_branch_counts[env_idx].get(parent, 0) + 1
            else:
                state_branches[parent, env_idx] += 1
        state_branches[:, env_idx] -= 1
        state_branches[:, env_idx].clamp_(min=1)
    return state_branches, root_branch_counts


class UnbiasedTreeSearchTest(unittest.TestCase):
    def test_terminal_estimate_averages_root_advantages(self):
        advantages = torch.tensor([[2.0], [6.0]])
        values = torch.tensor([[10.0], [10.0]])
        parents = torch.tensor([[-1], [-1]])
        trees = torch.tensor([[-1], [-1]])
        state_branches, root_branch_counts = branch_metadata(parents)
        estimates = [{}]

        select_next_states(
            terminated_envs=[0],
            current_step=1,
            advantages=advantages,
            values=values,
            parent_indices=parents,
            state_branches=state_branches,
            tree_indices=trees,
            search_count=[{}],
            max_search=1,
            root_branch_counts=root_branch_counts,
            terminal_estimates=estimates,
            skip_init_search=[False],
            affected_tree_ids=[-1],
            gamma=0.9,
            gae_lambda=0.8,
        )

        self.assertAlmostEqual(estimates[0][-1], 13.2, places=6)

    def test_chain_uses_first_degraded_state(self):
        advantages = torch.tensor([[0.0], [2.0], [3.0]])
        values = torch.full((3, 1), 10.0)
        parents = torch.tensor([[-1], [0], [1]])
        trees = torch.full((3, 1), -1, dtype=torch.long)
        state_branches, root_branch_counts = branch_metadata(parents)

        selected = select_next_states(
            terminated_envs=[0],
            current_step=2,
            advantages=advantages,
            values=values,
            parent_indices=parents,
            state_branches=state_branches,
            tree_indices=trees,
            search_count=[{}],
            max_search=1,
            root_branch_counts=root_branch_counts,
            terminal_estimates=[{}, {-1: 9.5}],
            skip_init_search=[False],
            affected_tree_ids=[-1],
            gamma=0.9,
            gae_lambda=0.8,
        )

        self.assertEqual(selected, [1])

    def test_branch_follows_child_with_maximum_m(self):
        # Step 0 is the root edge. Steps 1 and 2 are two actions from the same
        # depth-1 state; their successor states are represented by steps 3 and 4.
        advantages = torch.tensor([[0.0], [2.0], [2.0], [10.0], [8.0]])
        values = torch.full((5, 1), 10.0)
        parents = torch.tensor([[-1], [0], [0], [1], [2]])
        trees = torch.full((5, 1), -1, dtype=torch.long)
        state_branches, root_branch_counts = branch_metadata(parents)

        selected = select_next_states(
            terminated_envs=[0],
            current_step=4,
            advantages=advantages,
            values=values,
            parent_indices=parents,
            state_branches=state_branches,
            tree_indices=trees,
            search_count=[{}],
            max_search=1,
            root_branch_counts=root_branch_counts,
            terminal_estimates=[{}, {-1: 8.5}],
            skip_init_search=[False],
            affected_tree_ids=[-1],
            gamma=0.9,
            gae_lambda=0.8,
        )

        self.assertEqual(selected, [4])

    def test_global_leave_one_out_and_simultaneous_order_independence(self):
        advantages = torch.tensor([[0.0, 4.0]])
        values = torch.tensor([[10.0, 10.0]])
        parents = torch.tensor([[-1, -1]])
        trees = torch.tensor([[-1, -1]])
        state_branches, root_branch_counts = branch_metadata(parents)

        def run(terminated_envs):
            estimates = [{}, {}]
            selected = select_next_states(
                terminated_envs=terminated_envs,
                current_step=0,
                advantages=advantages,
                values=values,
                parent_indices=parents,
                state_branches=state_branches,
                tree_indices=trees,
                search_count=[{}, {}],
                max_search=1,
                root_branch_counts=root_branch_counts,
                terminal_estimates=estimates,
                skip_init_search=[False, False],
                affected_tree_ids=[-1 for _ in terminated_envs],
                gamma=0.9,
                gae_lambda=0.8,
            )
            return dict(zip(terminated_envs, selected)), estimates

        forward, estimates = run([0, 1])
        reverse, _ = run([1, 0])

        self.assertEqual(forward, reverse)
        self.assertEqual(forward[0], 0)
        self.assertEqual(forward[1], -2)
        self.assertAlmostEqual(estimates[0][-1], 10.0)
        self.assertAlmostEqual(estimates[1][-1], 13.2, places=6)

    def test_one_env_selects_only_one_random_tree_when_m_ties(self):
        advantages = torch.zeros((2, 1))
        values = torch.full((2, 1), 10.0)
        parents = torch.tensor([[-1], [-2]])
        trees = torch.tensor([[-1], [-2]])
        state_branches, root_branch_counts = branch_metadata(parents)
        estimates = [{-1: 10.0}, {-1: 12.0}]
        counts = [{}, {}]

        selected = select_next_states(
            terminated_envs=[0],
            current_step=1,
            advantages=advantages,
            values=values,
            parent_indices=parents,
            state_branches=state_branches,
            tree_indices=trees,
            search_count=counts,
            max_search=2,
            root_branch_counts=root_branch_counts,
            terminal_estimates=estimates,
            skip_init_search=[False, False],
            affected_tree_ids=[-2],
            gamma=0.9,
            gae_lambda=0.8,
        )

        self.assertIn(selected[0], (0, 1))
        self.assertEqual(sum(counts[0].values()), 1)
        self.assertEqual(len(counts[0]), 1)

    def test_incomplete_tree_is_not_cached_or_used_as_baseline(self):
        advantages = torch.tensor([[0.0, 4.0]])
        values = torch.tensor([[10.0, 10.0]])
        parents = torch.tensor([[-1, -1]])
        trees = torch.tensor([[-1, -1]])
        state_branches, root_branch_counts = branch_metadata(parents)
        estimates = [{-1: 99.0}, {}]

        selected = select_next_states(
            terminated_envs=[0, 1],
            current_step=0,
            advantages=advantages,
            values=values,
            parent_indices=parents,
            state_branches=state_branches,
            tree_indices=trees,
            search_count=[{}, {}],
            max_search=1,
            root_branch_counts=root_branch_counts,
            terminal_estimates=estimates,
            skip_init_search=[True, False],
            affected_tree_ids=[-1, -1],
            gamma=0.9,
            gae_lambda=0.8,
        )

        self.assertNotIn(-1, estimates[0])
        self.assertEqual(selected, [-2, -2])

    def test_cached_tree_stays_in_baseline_when_search_limit_is_reached(self):
        advantages = torch.tensor([[0.0, 4.0]])
        values = torch.tensor([[10.0, 10.0]])
        parents = torch.tensor([[-1, -1]])
        trees = torch.tensor([[-1, -1]])
        state_branches, root_branch_counts = branch_metadata(parents)
        estimates = [{-1: 10.0}, {}]

        selected = select_next_states(
            terminated_envs=[1],
            current_step=0,
            advantages=advantages,
            values=values,
            parent_indices=parents,
            state_branches=state_branches,
            tree_indices=trees,
            search_count=[{-1: 1}, {}],
            max_search=1,
            root_branch_counts=root_branch_counts,
            terminal_estimates=estimates,
            skip_init_search=[False, False],
            affected_tree_ids=[-1],
            gamma=0.9,
            gae_lambda=0.8,
        )

        self.assertEqual(selected, [-2])
        self.assertEqual(estimates[0][-1], 10.0)
        self.assertAlmostEqual(estimates[1][-1], 13.2, places=6)

    def test_cached_terminal_estimate_is_replaced_only_after_completion(self):
        advantages = torch.tensor([[5.0, 0.0]])
        values = torch.tensor([[10.0, 10.0]])
        parents = torch.tensor([[-1, -1]])
        trees = torch.tensor([[-1, -1]])
        state_branches, root_branch_counts = branch_metadata(parents)
        estimates = [{-1: 7.0}, {}]

        select_next_states(
            terminated_envs=[1],
            current_step=0,
            advantages=advantages,
            values=values,
            parent_indices=parents,
            state_branches=state_branches,
            tree_indices=trees,
            search_count=[{}, {}],
            max_search=1,
            root_branch_counts=root_branch_counts,
            terminal_estimates=estimates,
            skip_init_search=[False, False],
            affected_tree_ids=[-1],
            gamma=0.9,
            gae_lambda=0.8,
        )
        self.assertEqual(estimates[0][-1], 7.0)

        select_next_states(
            terminated_envs=[0],
            current_step=0,
            advantages=advantages,
            values=values,
            parent_indices=parents,
            state_branches=state_branches,
            tree_indices=trees,
            search_count=[{}, {}],
            max_search=1,
            root_branch_counts=root_branch_counts,
            terminal_estimates=estimates,
            skip_init_search=[False, False],
            affected_tree_ids=[-1],
            gamma=0.9,
            gae_lambda=0.8,
        )
        self.assertAlmostEqual(estimates[0][-1], 14.0, places=6)


if __name__ == "__main__":
    unittest.main()
