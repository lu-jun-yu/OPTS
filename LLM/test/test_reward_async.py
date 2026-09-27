"""CPU integration tests: PYTHONPATH=LLM:LLM/verl python -m unittest discover -s LLM/test."""

import asyncio
import inspect
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from utils.bounded_math import BoundedMathVerifier
from utils.reward_fn import _get_score_executor, _get_training_verifier, compute_score, compute_score_sync


def worker_probe():
    verifier = _get_training_verifier()
    if verifier.process is None:
        verifier._start()
    return verifier.process.pid, threading.current_thread() is threading.main_thread()


def delayed_probe():
    time.sleep(0.2)
    return worker_probe()


class AsyncRewardTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def tearDownClass(cls):
        if _get_score_executor.cache_info().currsize:
            _get_score_executor().shutdown(wait=True, cancel_futures=True)
            _get_score_executor.cache_clear()

    async def test_sync_async_results_match(self):
        cases = [
            (r"\boxed{\frac{1}{2}}", "0.5", None),
            (r"\boxed{\sqrt{\frac{1}{4}}}", "0.25", None),
            (r"\boxed{1+10\%}", "1.1", None),
            (r"\boxed{1+10\%}", "0.11", None),
            (r"\boxed{\frac{-1}{3}}", "-1./3", None),
            (r"\boxed{\arcsin(10/13)}", "np.arcsin(10/13)", None),
            (r"\boxed{1} then \boxed {42}", "42", None),
            (r"\boxed{}", "42", None),
            ("no box", "42", None),
            (r"\boxed{42}", "42", {"full_response_str": r"\boxed{42} then \boxed{9}",
                                      "unused_unpickleable": lambda: None}),
        ]
        expected = [compute_score_sync("test", text, gt, 2.5, extra) for text, gt, extra in cases]
        results = await asyncio.gather(*(compute_score("test", text, gt, 2.5, extra)
                                         for text, gt, extra in cases))
        self.assertEqual(results, expected)
        self.assertEqual(results[0]["reward_status"], "math_match")

    async def test_pool_reuse_and_event_loop_responsiveness(self):
        pool = _get_score_executor()
        self.assertIs(pool, _get_score_executor())
        self.assertEqual(pool._max_workers, 4)
        loop = asyncio.get_running_loop()
        ticks = []

        async def heartbeat():
            for _ in range(4):
                await asyncio.sleep(0.01)
                ticks.append(True)

        probes = [loop.run_in_executor(pool, delayed_probe) for _ in range(4)]
        await heartbeat()
        self.assertTrue(any(not task.done() for task in probes))
        for pid, is_main_thread in await asyncio.gather(*probes):
            self.assertNotEqual(pid, os.getpid())
            self.assertFalse(is_main_thread)  # The thread supervises a separate math process.
        self.assertEqual(len(ticks), 4)

    async def test_async_entry_from_thread_event_loop(self):
        def from_thread():
            return asyncio.run(compute_score("test", r"\boxed{\frac{1}{2}}", "0.5"))

        result = await asyncio.to_thread(from_thread)
        self.assertEqual(result["acc"], 1.0)
        self.assertEqual(result["reward_status"], "math_match")

    async def test_real_timeout_and_worker_recovery(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "utils").mkdir()
            (root / "utils/__init__.py").touch()
            shutil.copy2(Path(__file__).parent / "fixtures/staged_math_reward.py", root / "utils/reward_fn.py")
            with BoundedMathVerifier(root, memory_bytes=None, stage_timeouts=True) as verifier:
                verifier.stage_timeouts = dict.fromkeys(verifier.stage_timeouts, 0.15)
                with patch("utils.reward_fn._get_training_verifier", return_value=verifier):
                    timed_out = await compute_score("test", r"\boxed{hang_verify}", "truth")
                    recovered = await compute_score("test", r"\boxed{good}", "truth")
        self.assertEqual(timed_out["acc"], 0.0)
        self.assertEqual(timed_out["reward_status"], "verification_timeout")
        self.assertEqual(recovered["acc"], 1.0)
        self.assertEqual(recovered["reward_status"], "math_match")

    async def test_math_is_not_run_in_supervisor_threads(self):
        with patch("utils.reward_fn.validate_answer_with_status", side_effect=AssertionError("math ran in parent")):
            result = await compute_score("test", r"\boxed{\frac{7}{8}}", "0.875")
        self.assertEqual(result["acc"], 1.0)

    async def test_verl_dynamic_loader_and_async_manager(self):
        import numpy as np
        import torch
        from omegaconf import OmegaConf
        from verl import DataProto
        from verl.experimental.reward_loop.reward_manager.naive import NaiveRewardManager
        from verl.trainer.ppo.reward import get_custom_reward_fn

        config = OmegaConf.create({"custom_reward_function": {
            "path": str(Path(__file__).resolve().parents[1] / "utils/reward_fn.py"),
            "name": "compute_score", "reward_kwargs": {"correct_reward": 2.5},
        }})
        score_fn = get_custom_reward_fn(config)
        self.assertTrue(inspect.iscoroutinefunction(score_fn))
        self.assertTrue(inspect.iscoroutinefunction(compute_score))
        self.assertFalse(inspect.iscoroutinefunction(compute_score_sync))
        manager = NaiveRewardManager(config, SimpleNamespace(decode=lambda *a, **kw: r"\boxed{999}"), score_fn)
        self.assertTrue(manager.is_async_reward_score)
        batch = DataProto.from_dict(
            tensors={"responses": torch.tensor([[1, 2]]), "attention_mask": torch.ones((1, 2), dtype=torch.long)},
            non_tensors={"data_source": np.array(["test"], dtype=object),
                         "reward_model": np.array([{"ground_truth": "0.5"}], dtype=object),
                         "extra_info": np.array([{"full_response_str": r"\boxed{1} then \boxed {\frac{1}{2}}"}], dtype=object)},
        )
        result = await manager.run_single(batch)
        self.assertEqual(result["reward_score"], 2.5)
        self.assertEqual(result["reward_extra_info"]["acc"], 1.0)
        self.assertEqual(result["reward_extra_info"]["reward_status"], "math_match")


if __name__ == "__main__":
    unittest.main()
