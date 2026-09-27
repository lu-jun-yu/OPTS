"""Real subprocess tests for offline math comparison resource isolation."""

import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import patch

from utils.bounded_math import BoundedMathVerifier


class BoundedMathTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / "utils").mkdir()
        (self.root / "utils/__init__.py").touch()
        shutil.copy2(Path(__file__).parent / "fixtures/bounded_math_reward.py", self.root / "utils/reward_fn.py")

    def test_exact_and_empty_answers_do_not_start_a_worker(self):
        with BoundedMathVerifier(self.root) as verifier, patch("utils.bounded_math.subprocess.Popen") as start:
            self.assertEqual(verifier.compare(" 42 ", "42"), (True, "exact_match"))
            self.assertEqual(verifier.compare("", ""), (False, "empty_answer"))
            self.assertFalse(verifier("42", ""))
            start.assert_not_called()

    def test_ignored_alarm_is_killed_reaped_cached_and_next_answer_succeeds(self):
        with BoundedMathVerifier(self.root, timeout=0.15, memory_bytes=128 * 1024**2) as verifier:
            verifier._start()
            process = verifier.process
            started = time.monotonic()
            self.assertEqual(verifier.compare("hang", "truth"), (False, "hard_timeout"))
            self.assertLess(time.monotonic() - started, 3)
            self.assertIsNotNone(process.returncode)
            self.assertFalse(Path(f"/proc/{process.pid}").exists())
            self.assertIsNone(verifier.process)
            self.assertFalse(verifier("hang", "truth"))
            self.assertEqual(verifier.counts["cache_hits"], 1)
            self.assertTrue(verifier("good", "truth"))
            self.assertEqual(verifier.counts["worker_starts"], 2)

    def test_memory_limit_is_not_swallowed_and_next_answer_succeeds(self):
        with BoundedMathVerifier(self.root, memory_bytes=128 * 1024**2) as verifier:
            verifier._start()
            limit = next(line for line in Path(f"/proc/{verifier.process.pid}/limits").read_text().splitlines()
                         if line.startswith("Max address space"))
            self.assertEqual(limit.split()[-3:], [str(128 * 1024**2), str(128 * 1024**2), "bytes"])
            self.assertEqual(verifier.compare("oom", "truth"), (False, "memory_limit"))
            self.assertIsNone(verifier.process)
            self.assertTrue(verifier("good", "truth"))

    def test_partial_output_cannot_bypass_the_deadline(self):
        with BoundedMathVerifier(self.root, timeout=0.15) as verifier:
            verifier._start()
            started = time.monotonic()
            self.assertEqual(verifier.compare("partial", "truth"), (False, "hard_timeout"))
            self.assertLess(time.monotonic() - started, 3)
            self.assertTrue(verifier("good", "truth"))

    def test_library_prints_do_not_corrupt_responses(self):
        with BoundedMathVerifier(self.root) as verifier:
            self.assertEqual(verifier.compare("noisy", "truth"), (False, "not_equivalent"))

    def test_crashed_worker_raises_instead_of_scoring_zero(self):
        with BoundedMathVerifier(self.root) as verifier:
            with self.assertRaisesRegex(RuntimeError, "exited unexpectedly"):
                verifier("crash", "truth")
            self.assertIsNone(verifier.process)
            self.assertNotIn(("crash", "truth"), verifier.cache)
            self.assertTrue(verifier("good", "truth"))

    def test_comparison_cache_preserves_direction(self):
        with BoundedMathVerifier(self.root) as verifier:
            self.assertTrue(verifier("good", "truth"))
            self.assertFalse(verifier("truth", "good"))
            self.assertEqual(verifier.counts["worker_calls"], 2)
            self.assertTrue(verifier("good", "truth"))
            self.assertEqual(verifier.counts["worker_calls"], 2)

    def test_real_math_rule_is_retained(self):
        with BoundedMathVerifier() as verifier:
            for answer, truth in [(r"\frac{1}{2}", "0.5"), (r"\sqrt{\frac{1}{4}}", "0.5"),
                                  (r"x^2+2*x+1", "(x+1)^2"), (r"\{1,2\}", r"\{2,1\}"),
                                  (r"1+10\%", "1.1"), ("-1./3", r"\frac{-1}{3}")]:
                with self.subTest(answer=answer):
                    self.assertTrue(verifier(answer, truth))
            self.assertFalse(verifier(r"\frac{1}{2}", "0.25"))
            self.assertFalse(verifier("(1,2)", "(2,1)"))
            self.assertLess(verifier.max_worker_rss_bytes, 1024**3)


if __name__ == "__main__":
    unittest.main()
