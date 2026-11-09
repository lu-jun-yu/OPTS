"""Check TreeGAE timeout semantics and the MuJoCo experiment configuration."""
import unittest
import torch
from pathlib import Path
import sys
import argparse
import ast
import tempfile
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cleanrl" / "cleanrl"))
from opts_ttpo_core_wEqual_bMax import compute_tree_gae as max_gae
from opts_ttpo_core_wEqual_bMean import compute_tree_gae as mean_gae
from mujoco_config import ALGORITHMS, SETTINGS, DEFAULT_SETTINGS, ENVS, SCRIPTS, job_spec, job_dir, training_command
from run_queue import planned_jobs
import summarize

class TestBootstrap(unittest.TestCase):
    def tree(self,core,dones,bootstrap,expected):
        # Minimal chain: leaf at t=1, parent t=0, reward=1 at every step.
        rewards=torch.tensor([[1.],[1.]])
        values=torch.tensor([[0.],[0.]])
        parents=torch.tensor([[-1],[0]])
        adv=torch.zeros_like(rewards)
        core(terminal_step=1,env_idx=0,rewards=rewards,values=values,
             dones=torch.tensor([[0.],[float(dones)]]),parent_indices=parents,
             advantages=adv,gamma=0.99,gae_lambda=0.95,
             timeout_bootstrap_values=torch.tensor([[0.],[float(bootstrap)]]) if bootstrap is not None else None)
        self.assertAlmostEqual(float(adv[1,0]),expected,places=5)
        self.assertAlmostEqual(float(adv[0,0]),1+.99*.95*expected,places=5)
    def test_max(self):
        for core in [max_gae,mean_gae]:
            self.tree(core,1,None,1.0)
            self.tree(core,1,3.0,3.97)
            self.tree(core,0,3.0,1.0)  # nonterminal GAE tail requires next_value, not timeout
    def test_branch_max_mean(self):
        # Branch children t=1,t=2; bootstrap only changes truncated child's leaf.
        for fn in (max_gae,mean_gae):
            r=torch.tensor([[1.],[1.],[1.]])
            v=torch.zeros_like(r)
            done=torch.tensor([[0.],[1.],[1.]])
            p=torch.tensor([[-1],[0],[0]])
            a=torch.zeros_like(r)
            b=torch.tensor([[0.],[2.],[0.]])
            fn(terminal_step=1,env_idx=0,rewards=r,values=v,dones=done,parent_indices=p,
               advantages=a,gamma=.99,gae_lambda=.95,timeout_bootstrap_values=b)
            fn(terminal_step=2,env_idx=0,rewards=r,values=v,dones=done,parent_indices=p,
               advantages=a,gamma=.99,gae_lambda=.95,timeout_bootstrap_values=b)
            self.assertAlmostEqual(float(a[1,0]),2.98,places=5)
            self.assertAlmostEqual(float(a[2,0]),1.0,places=5)
            child=2.98 if fn is max_gae else 1.99
            self.assertAlmostEqual(float(a[0,0]),1+.99*.95*child,places=5)

class TestReproduction(unittest.TestCase):
    def test_matched_optimizer_steps_and_smoke_checkpoints(self):
        small = job_spec("2048_1m", "zero", "full", "Hopper-v4", 1)
        large = job_spec("4096_2m", "zero", "full", "Hopper-v4", 1)
        self.assertEqual(small["expected_iterations"], 488)
        self.assertEqual(small["optimizer_steps"], large["optimizer_steps"])
        self.assertEqual(large["minibatch_size"], 2 * small["minibatch_size"])
        for setting in SETTINGS:
            smoke = job_spec(setting, "zero", "ppo", "Hopper-v4", 1, True, 2)
            self.assertEqual(smoke["expected_iterations"], 2)
            self.assertEqual(smoke["checkpoint_every"], 1)

    def test_queue_scope_and_concurrency(self):
        args = argparse.Namespace(max_workers=80, seed_start=1, seed_end=50,
                                  settings=list(DEFAULT_SETTINGS), modes=["zero"], variants=list(SCRIPTS),
                                  envs=list(ENVS), smoke=False, eval_episodes=10)
        self.assertEqual(len(planned_jobs(args)), 2500)
        args.max_workers = 81
        with self.assertRaises(ValueError):
            planned_jobs(args)

    def test_8192_sweep_preserves_optimizer_schedule(self):
        args = argparse.Namespace(max_workers=80, seed_start=1, seed_end=50,
                                  settings=["8192_4m"], modes=["zero"], variants=list(SCRIPTS),
                                  envs=list(ENVS), smoke=False, eval_episodes=10)
        jobs = planned_jobs(args)
        self.assertEqual(len(jobs), 1250)
        self.assertEqual(len({(s["env"], s["variant"], s["seed"]) for s in jobs}), 1250)
        for spec in jobs:
            self.assertEqual(spec["num_steps"], 8192)
            self.assertEqual(spec["total_timesteps"], 4_000_000)
            self.assertEqual(spec["minibatch_size"], 256)
            self.assertEqual(spec["expected_iterations"], 488)
            self.assertEqual(spec["expected_final_step"], 3_997_696)
            self.assertEqual(spec["optimizer_steps"], 156_160)
            self.assertEqual(spec["checkpoint_every"], 50)
            self.assertEqual(spec["independent_eval_episodes"], 10)

    def test_command_variants_and_timeout(self):
        with tempfile.TemporaryDirectory() as folder:
            for setting in SETTINGS:
                for variant in SCRIPTS:
                    for mode in ("zero", "bootstrap"):
                        spec = job_spec(setting, mode, variant, "HalfCheetah-v4", 1)
                        command = training_command(spec, job_dir(folder, spec))
                        self.assertEqual("--timeout-bootstrap" in command, mode == "bootstrap")
                        self.assertEqual(command[1], str(ALGORITHMS / SCRIPTS[variant]))
                        self.assertEqual("--aggregation" in command, variant == "unweighted")

    def test_standalone_defaults_leave_instrumentation_disabled(self):
        for filename in set(SCRIPTS.values()):
            module = ast.parse((ALGORITHMS / filename).read_text(encoding="utf-8"))
            args = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "Args")
            values = {node.target.id: ast.literal_eval(node.value) for node in args.body
                      if isinstance(node, ast.AnnAssign) and node.target.id in ("timeout_bootstrap", "checkpoint_every")}
            self.assertIs(values["timeout_bootstrap"], False)
            self.assertEqual(values["checkpoint_every"], 0)
            self.assertFalse(any(isinstance(node, ast.ImportFrom) and node.module == "axap_checkpoint"
                                 for node in module.body))

    def test_nonterminal_uses_rollout_boundary_not_timeout(self):
        for core in (max_gae, mean_gae):
            rewards = torch.ones((1, 1))
            advantages = torch.zeros_like(rewards)
            core(0, 0, rewards, torch.zeros_like(rewards), torch.zeros_like(rewards),
                 torch.tensor([[-1]]), advantages, .99, .95, next_value=4.,
                 timeout_bootstrap_values=torch.tensor([[100.]]))
            self.assertAlmostEqual(float(advantages[0, 0]), 4.96, places=5)

    def test_statistics_pairing_and_denominators(self):
        records = [summarize.Run(setting, "zero", variant, env, seed, 10., 10.)
                   for setting in summarize.SETTINGS for variant in SCRIPTS for env in ENVS for seed in (1, 2)]
        self.assertEqual(len(summarize.validate_matching(records, "zero", [1, 2])), 100)
        with self.assertRaises(summarize.ValidationError):
            summarize.validate_matching(records[:-1], "zero", [1, 2])
        with self.assertRaises(summarize.ValidationError):
            summarize.validate_matching(records + records[:1], "zero", [1, 2])
        values = np.full((5, 5, 2), 10.)
        indices = [np.array([[0, 1], [1, 0]]) for _ in ENVS]
        report = summarize.summarize_metric(values, indices, [1, 2], "fixture")
        self.assertEqual(report["macro"]["full_vs_ppo"]["relative_percent"], 0.)
        values[:, 0, :] = 0.
        with self.assertRaises(summarize.ValidationError):
            summarize.summarize_metric(values, indices, [1, 2], "zero denominator")


if __name__ == '__main__':
    unittest.main()
