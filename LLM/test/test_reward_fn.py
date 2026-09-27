"""Run with: PYTHONPATH=LLM python -m unittest discover -s LLM/test -p test_reward_fn.py."""

import unittest
from unittest.mock import patch

from math_verify.errors import TimeoutException

from utils.reward_fn import (
    _cached_parse, compute_score_sync as compute_score, extract_answer, validate_answer, validate_answer_with_status,
)


class RewardTests(unittest.TestCase):
    def test_extraction(self):
        cases = [
            (r"\boxed{36} then \boxed{9}", "9"),
            (r"\boxed{9} corrected to \boxed{36}", "36"),
            (r"\boxed{36}</think>\boxed{9}", "9"),
            (r"<think>\boxed{36}</think>no more answers", "36"),
            (r"\boxed{36}</think>\boxed{9}</think>", "9"),
            (r"\boxed{4} then \boxed{\frac{5}{{\log_2 a}}}", r"\frac{5}{{\log_2 a}}"),
            (r"\boxed{\frac{1}{\sqrt{1+\frac{1}{2}}}}", r"\frac{1}{\sqrt{1+\frac{1}{2}}}"),
            (r"\boxed{\left\{1,2\right\}}", r"\left\{1,2\right\}"),
            ("\\boxed{\n  42 \n}", "42"),
            (r"\boxed {42}", "42"),
            ("\\boxed{1} then \\boxed \t\n {42}", "42"),
            (r"\boxed{1} then \boxed {\frac{1}{{2}}}", r"\frac{1}{{2}}"),
            (r"\boxed{1} unfinished \boxed {2", "1"),
            (r"\boxed{36} unfinished \boxed{\frac{1}{2}", "36"),
            (r"\boxed{36} then \boxed{}", ""),
            (r"unfinished \boxed{36", None),
            ("no boxed </think>", None),
            (r"\boxed{" + "{" * 50 + "42" + "}" * 51, "{" * 50 + "42" + "}" * 50),
        ]
        for text, expected in cases:
            with self.subTest(text=text):
                self.assertEqual(extract_answer(text), expected)

    def test_reward_uses_last_answer(self):
        for text, expected in [(r"\boxed{36}</think>\boxed{9}", 0.0),
                               (r"\boxed{9}</think>\boxed{36}", 1.0),
                               (r"\boxed{36} then \boxed{}", 0.0)]:
            with self.subTest(text=text):
                result = compute_score("test", text, "36")
                self.assertEqual(result["score"], expected)
                self.assertEqual(result["acc"], expected)

    def test_nested_answer_passed_to_validator(self):
        # Test extraction/dispatch independently of math-verify's LaTeX coverage.
        with patch("utils.reward_fn.validate_answer_with_status", return_value=(True, "math_match")) as validate:
            result = compute_score("test", r"\boxed{4}\boxed{\frac{1}{{2}}}", "0.5")
        validate.assert_called_once_with(r"\frac{1}{{2}}", "0.5")
        self.assertEqual(result, {"score": 1.0, "acc": 1.0, "pred": r"\frac{1}{{2}}",
                                  "reward_status": "math_match"})

    def test_equivalent_answers(self):
        cases = [
            ("42", "42.0"),
            ("1/2", "0.5"),
            (r"\frac{1}{2}", "0.5"),
            (r"\frac{1}{{2}}", "0.5"),
            (r"\frac{1}{\sqrt{4}}", "0.5"),
            (r"\sqrt{\frac{1}{4}}", "0.5"),
            (r"\frac{1}{\sqrt{1+\frac{1}{2}}}", r"\sqrt{\frac{2}{3}}"),
            (r"\frac{1}{{2}}", r"\frac{1}{{2}}"),
            (r"x^2+2*x+1", r"(x+1)^2"),
            (r"\log_2 8", "3"),
            (r"\{1,2\}", r"\{2,1\}"),
            (r"{1,3} \cup {2,4}", r"{1,2,3,4}"),
            (r"10\%", "0.1"),
            ("10%", "0.1"),
            ("10 percent", "0.1"),
            (r"$10\%$", "0.1"),
            (r"1+10\%", "1.1"),
            (r"50\%+25\%", "0.75"),
            (r"-10\%", "-0.1"),
            (r"0.5\%", "0.005"),
            (r".5\%", "0.005"),
            ("1+10 percent", "1.1"),
            ("50 PCT+25 Percentage", "0.75"),
            (r"10\%^2", "0.01"),
            (r"1/10\%", "10"),
            (r"\frac{-1}{3}", "-1./3"),
            (r"\frac{-3}{2}", "-3./2"),
            (r"\arcsin(10/13)", "np.arcsin(10/13)"),
            (r"\arccos(1/2)", "np.arccos(1/2)"),
            (r"\arctan(1)", "np.arctan(1)"),
            ("1,000", "1000"),
            ("A", "A"),
            (r"\text{No solution}", r"\text{No solution}"),
            ("\\frac{1}{\n2}", "0.5"),
        ]
        for answer, truth in cases:
            with self.subTest(answer=answer, truth=truth):
                self.assertTrue(validate_answer(answer, truth))
                self.assertTrue(validate_answer(truth, answer))

    def test_math_environments(self):
        for left, right in (("", ""), ("$", "$"), ("$$", "$$"),
                            (r"\(", r"\)"), (r"\[", r"\]")):
            answer = left + r"\frac{1}{\sqrt{4}}" + right
            with self.subTest(answer=answer):
                self.assertTrue(validate_answer(answer, "0.5"))
                self.assertTrue(validate_answer("0.5", answer))

    def test_inequivalent_answers(self):
        cases = [
            (r"\sqrt{\frac{1}{4}}", "0.25"),
            (r"\frac{1}{{2}}", "2"),
            (r"x^2+2*x+1", r"x^2+1"),
            (r"\{1,2\}", r"\{1,3\}"),
            ("(1,2)", "(2,1)"),
            ("1,2", "12"),
            ("1,2", "1.2"),
            (r"10\%", "10"),
            (r"1+10\%", "0.11"),
            (r"50\%+25\%", "75"),
            (r"\arcsin(10/13)", "np.arccos(10/13)"),
            (r"\frac{-1}{3}", "1./3"),
            ("A", "B"),
            ("", ""),
            (" ", "0"),
            (r"\unknown{\frac{1}{4}}", "0.25"),
            (r"\sqrt{\frac{1}{4}", "0.25"),
        ]
        for answer, truth in cases:
            with self.subTest(answer=answer, truth=truth):
                self.assertFalse(validate_answer(answer, truth))
                self.assertFalse(validate_answer(truth, answer))

    def test_exact_match_skips_math(self):
        with patch("utils.reward_fn._cached_parse") as parse, \
             patch("utils.reward_fn.verify") as verify:
            self.assertTrue(validate_answer("  same text \n", "same text"))
            self.assertTrue(validate_answer(r"\frac{1}{2}", r"\frac{1}{2}"))
            self.assertFalse(validate_answer("", ""))
            self.assertFalse(validate_answer(" \n", " "))
            self.assertFalse(validate_answer("", "42"))
            self.assertFalse(validate_answer("42", ""))
            parse.assert_not_called()
            verify.assert_not_called()

    def test_different_text_reaches_math(self):
        with patch("utils.reward_fn._cached_parse", side_effect=[["answer"], ["truth"]]) as parse, \
             patch("utils.reward_fn.verify", return_value=True) as verify:
            self.assertTrue(validate_answer(" 1/2 ", "0.5"))
            self.assertEqual(parse.call_count, 2)
            verify.assert_called_once_with(["truth"], ["answer"], raise_on_error=True)
        with patch("utils.reward_fn._cached_parse", return_value=["parsed"]), \
             patch("utils.reward_fn.verify", return_value=False) as verify:
            self.assertFalse(validate_answer("1/2", "0.5"))
            verify.assert_called_once()

    def test_parse_failure(self):
        for failure in ({"return_value": []}, {"side_effect": ValueError("parse failed")}):
            with patch("utils.reward_fn._cached_parse", **failure):
                self.assertTrue(validate_answer("  same text ", "same text"))
                self.assertFalse(validate_answer("1,2", "12"))
                self.assertFalse(validate_answer("A B", "AB"))
                self.assertFalse(validate_answer("", ""))

    def test_parse_cache(self):
        _cached_parse.cache_clear()
        self.assertTrue(validate_answer(r"\frac{1}{{2}}", "0.5"))
        self.assertTrue(validate_answer(r"\frac{1}{{2}}", "0.5"))
        self.assertEqual(_cached_parse.cache_info().misses, 2)
        self.assertEqual(_cached_parse.cache_info().hits, 2)
        _cached_parse.cache_clear()

    def test_diagnostic_status(self):
        self.assertEqual(validate_answer_with_status(" 42 ", "42"), (True, "exact_match"))
        self.assertEqual(validate_answer_with_status("", "42"), (False, "empty_answer"))
        self.assertEqual(validate_answer_with_status("42", ""), (False, "empty_ground_truth"))
        self.assertEqual(validate_answer_with_status("1/2", "0.5"), (True, "math_match"))
        self.assertEqual(validate_answer_with_status("1/2", "0.25"), (False, "not_equivalent"))
        self.assertEqual(compute_score("test", "no box", "42")["reward_status"], "no_complete_box")
        for failure, status in [(TimeoutException(), "parse_timeout"),
                                (ValueError(), "parse_failed"), ([], "parse_failed")]:
            for i, role in enumerate(("answer", "ground_truth")):
                with self.subTest(role=role, status=status), \
                     patch("utils.reward_fn._cached_parse", side_effect=[[1]] * i + [failure]):
                    result = compute_score("test", r"\boxed{1/2}", "0.5")
                    self.assertEqual(result["acc"], 0.0)
                    self.assertEqual(result["reward_status"], f"{role}_{status}")
        for failure, status in [(TimeoutException(), "verification_timeout"),
                                (ValueError(), "verification_error")]:
            with patch("utils.reward_fn.verify", side_effect=failure):
                result = compute_score("test", r"\boxed{1/2}", "0.5")
                self.assertEqual(result["acc"], 0.0)
                self.assertEqual(result["reward_status"], status)

    def test_parse_timeout_not_cached(self):
        _cached_parse.cache_clear()
        with patch("utils.reward_fn.parse", side_effect=[TimeoutException(), [1]]) as parse:
            self.assertEqual(validate_answer_with_status("1/2", "0.5"), (False, "answer_parse_timeout"))
            self.assertEqual(_cached_parse("1/2"), [1])
            self.assertEqual(parse.call_count, 2)
            self.assertTrue(parse.call_args.kwargs["raise_on_error"])
        _cached_parse.cache_clear()

    def test_nested_reward(self):
        for answer in (r"\frac{1}{{2}}", r"\frac{1}{\sqrt{4}}", r"\sqrt{\frac{1}{4}}"):
            with self.subTest(answer=answer):
                result = compute_score("test", r"\boxed{4}</think>\boxed{" + answer + "}", "0.5")
                self.assertEqual(result, {"score": 1.0, "acc": 1.0, "pred": answer,
                                          "reward_status": "math_match"})

    def test_tree_full_response(self):
        result = compute_score("test", r"\boxed{36}", "36", extra_info={
            "full_response_str": r"\boxed{36} unrelated \boxed{9}",
        })
        self.assertEqual(result, {"score": 0.0, "acc": 0.0, "pred": "9", "reward_status": "not_equivalent"})

    def test_custom_reward(self):
        result = compute_score("test", r"\boxed{9}\boxed{36}", "36", correct_reward=2.5)
        self.assertEqual(result, {"score": 2.5, "acc": 1.0, "pred": "36", "reward_status": "exact_match"})


if __name__ == "__main__":
    unittest.main()
