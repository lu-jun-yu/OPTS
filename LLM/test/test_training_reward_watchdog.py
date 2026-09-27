"""CPU-only regression tests for the training reward's external deadlines."""

from inspect import signature
from math import comb
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import time
import unittest

from math_verify import parse, verify

from utils.bounded_math import BoundedMathVerifier


class TrainingRewardWatchdogTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / "utils").mkdir()
        (self.root / "utils/__init__.py").touch()
        shutil.copy2(Path(__file__).parent / "fixtures/staged_math_reward.py", self.root / "utils/reward_fn.py")

    def test_training_limits_match_library_and_do_not_add_memory_cap(self):
        with BoundedMathVerifier(self.root, memory_bytes=None, stage_timeouts=True) as verifier:
            self.assertEqual(verifier.stage_timeouts, {
                "answer_parse": signature(parse).parameters["parsing_timeout"].default,
                "ground_truth_parse": signature(parse).parameters["parsing_timeout"].default,
                "verification": signature(verify).parameters["timeout_seconds"].default,
            })
            self.assertIsNone(verifier.memory_bytes)
            verifier._start()
            limit = next(line for line in Path(f"/proc/{verifier.process.pid}/limits").read_text().splitlines()
                         if line.startswith("Max address space"))
            self.assertEqual(limit.split()[-3:], ["unlimited", "unlimited", "bytes"])

    def test_each_stage_can_expire_and_process_is_reaped_then_replaced(self):
        cases = [("hang_parse", "truth", "answer_parse_timeout"),
                 ("good", "hang_parse", "ground_truth_parse_timeout"),
                 ("hang_verify", "truth", "verification_timeout")]
        for answer, truth, status in cases:
            with self.subTest(stage=status), \
                 BoundedMathVerifier(self.root, memory_bytes=None, stage_timeouts=True) as verifier:
                verifier.stage_timeouts = dict.fromkeys(verifier.stage_timeouts, 0.15)
                verifier._start()
                old_process = verifier.process
                started = time.monotonic()
                self.assertEqual(verifier.compare(answer, truth), (False, status))
                self.assertLess(time.monotonic() - started, 2)
                self.assertFalse(Path(f"/proc/{old_process.pid}").exists())
                self.assertIsNotNone(old_process.returncode)
                self.assertNotIn((answer, truth), verifier.cache)
                self.assertTrue(verifier("good", "truth"))
                self.assertEqual(verifier.counts["worker_starts"], 2)
                self.assertEqual(verifier.counts["hard_timeouts"], 1)

    def test_phase_budgets_are_not_a_single_combined_deadline(self):
        with BoundedMathVerifier(self.root, memory_bytes=None, stage_timeouts=True) as verifier:
            verifier.stage_timeouts = dict.fromkeys(verifier.stage_timeouts, 0.5)
            verifier._start()
            started = time.monotonic()
            self.assertTrue(verifier("slow_answer", "slow_truth"))
            self.assertGreater(time.monotonic() - started, 0.7)
            self.assertEqual(verifier.counts["hard_timeouts"], 0)

    def test_actual_default_timeout_survives_ignored_sigalrm(self):
        with BoundedMathVerifier(self.root, memory_bytes=None, stage_timeouts=True) as verifier:
            verifier._start()
            started = time.monotonic()
            self.assertEqual(verifier.compare("hang_verify", "truth"), (False, "verification_timeout"))
            elapsed = time.monotonic() - started
            timeout = signature(verify).parameters["timeout_seconds"].default
            self.assertGreaterEqual(elapsed, timeout - 0.1)
            self.assertLess(elapsed, timeout + 2)
            self.assertTrue(verifier("good", "truth"))

    def test_b300_stalled_expression_is_bounded_and_next_answer_succeeds(self):
        # Exact extracted pair from the stuck B300 reward worker (step 97).
        answer = r"\sum_{S=0}^{\lfloor n/2 \rfloor} \binom{n}{S} \binom{S + 3}{3} \binom{(n - 2S) + 4}{4}"
        truth = r"2^{n}C_{2n}^{n}"
        # n=1 is already a counterexample; expensive simplification is unnecessary
        # mathematically, but preserve the original checker instead of changing it.
        n = 1
        self.assertNotEqual(sum(comb(n, s) * comb(s + 3, 3) * comb(n - 2*s + 4, 4)
                                for s in range(n // 2 + 1)), 2**n * comb(2*n, n))
        with BoundedMathVerifier(memory_bytes=None, stage_timeouts=True) as verifier:
            verifier._start()
            started = time.monotonic()
            correct, status = verifier.compare(answer, truth)
            elapsed = time.monotonic() - started
            self.assertFalse(correct)
            self.assertIn(status, ("verification_timeout", "not_equivalent"))
            self.assertLess(elapsed, sum(verifier.stage_timeouts.values()) + 2)
            self.assertTrue(verifier(r"\frac{7}{8}", "0.875"))
            print(f"B300 regression: {status}, {elapsed:.3f}s; next answer passed", flush=True)


if __name__ == "__main__":
    unittest.main()
