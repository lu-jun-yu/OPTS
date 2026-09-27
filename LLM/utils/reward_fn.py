# Copyright 2025 Anonymous authors. All rights reserved.

"""
Reward function for \\boxed{answer} format.

A response earns reward iff its last complete \\boxed{...} contains the correct answer.
"""

import asyncio
import atexit
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import re
import threading
from typing import Optional

from math_verify import LatexExtractionConfig, parse, verify
from math_verify.errors import TimeoutException

from utils.boxed import find_last_boxed_span


@lru_cache(maxsize=65536)
def _cached_parse(s: str):
    # The box was already extracted: parse the whole answer, not numbers inside it.
    s = s.strip()
    for left, right in (("$$", "$$"), ("$", "$"), (r"\(", r"\)"), (r"\[", r"\]")):
        if s.startswith(left) and s.endswith(right):
            s = s[len(left):-len(right)].strip()
            break
    # Normalize known dataset notation, without executing Python expressions.
    s = re.sub(r"(?<![\w.])(\d+)\.(?![\d.])", r"\1.0", s)
    s = re.sub(r"(?<![\w.])np\.(arcsin|arccos|arctan)\s*(?=\()",
               lambda m: "\\" + m[1], s)
    # Convert each numeric percentage, not its surrounding arithmetic expression.
    # Parentheses preserve precedence in 1/10%, 10%^2, and sums of percentages.
    s = re.sub(
        r"(?<![\w.])((?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+))\s*"
        r"(?:\\?%|(?i:percentage|percent|pct)\b)",
        lambda m: rf"(\frac{{{m[1]}}}{{100}})", s,
    )
    return parse(
        rf"\[{s}\]",
        extraction_config=[LatexExtractionConfig()],
        extraction_mode="first_match",
        fallback_mode="no_fallback",
        raise_on_error=True,
    )


def extract_answer(response_str: str) -> Optional[str]:
    """Extract the last complete \\boxed{...}, regardless of think tags."""
    span = find_last_boxed_span(response_str)
    return None if span is None else response_str[span[1]:span[2]].strip()


def validate_answer(answer: str, ground_truth: str) -> bool:
    """Validate if the extracted answer matches the ground truth.

    Accepts identical nonempty strings (ignoring surrounding whitespace) first,
    then uses math-verify on the complete answer for differing strings.

    Supported match types (via math-verify):
    - Plain numbers: 42 == 42.0
    - LaTeX fractions: \\frac{1}{2} == 0.5 == 1/2
    - LaTeX expressions: \\sqrt{2}, x^{2}+1, etc.
    - Sets: {1,3} \\cup {2,4} == {1,2,3,4}
    - Percentages: 10\\% == 0.1
    - Text/multiple choice: A, B, C, D
    """
    return validate_answer_with_status(answer, ground_truth)[0]


def validate_answer_with_status(answer: str, ground_truth: str) -> tuple[bool, str]:
    """Keep parse/comparison failures distinguishable from an ordinary mismatch."""
    answer, ground_truth = answer.strip(), ground_truth.strip()
    if not answer or not ground_truth:
        return False, "empty_answer" if not answer else "empty_ground_truth"
    if answer == ground_truth:
        return True, "exact_match"
    parsed = []
    for role, text in (("answer", answer), ("ground_truth", ground_truth)):
        try:
            value = _cached_parse(text)
        except TimeoutException:
            return False, f"{role}_parse_timeout"
        except Exception:
            return False, f"{role}_parse_failed"
        if not value:
            return False, f"{role}_parse_failed"
        parsed.append(value)
    try:
        correct = bool(verify(parsed[1], parsed[0], raise_on_error=True))
    except TimeoutException:
        return False, "verification_timeout"
    except Exception:
        return False, "verification_error"
    return correct, "math_match" if correct else "not_equivalent"


def compute_score_sync(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    correct_reward: float = 1.0,
    extra_info: Optional[dict] = None,
    **kwargs,
) -> dict:
    """Compute the score for a response.

    Reward = correct_reward iff the last complete \\boxed{...} is present and its
    answer matches ground truth; otherwise 0.

    Args:
        solution_str: The response string.
        ground_truth: The expected answer.
        correct_reward: Reward for a correct answer.
        extra_info: Optional extra info dict, may contain 'full_response_str' for tree search.

    Returns:
        A dictionary containing the training reward (`score`), correctness
        (`acc`), and the extracted answer (`pred`) used by validation metrics.
        `reward_status` distinguishes matching, missing answers, and checker failures.
    """
    # OPTS_TTPO: Use full response string if available (for tree search)
    if extra_info and "full_response_str" in extra_info:
        solution_str = extra_info["full_response_str"]

    return _score_answer(extract_answer(solution_str), ground_truth, correct_reward, validate_answer_with_status)


def _score_answer(answer_content, ground_truth, correct_reward, validate):
    acc = 0.0
    total_score = 0.0
    status = "no_complete_box"
    if answer_content is not None:
        correct, status = validate(answer_content, ground_truth)
        if correct:
            acc = 1.0
            total_score += correct_reward

    return {
        "score": total_score,
        "acc": acc,
        "pred": answer_content if answer_content is not None else "",
        "reward_status": status,
    }


@lru_cache(maxsize=1)
def _get_score_executor():
    # Threads only supervise isolated math processes; no math-verify runs in them.
    # Keep at most four concurrent math processes per reward worker, as before.
    executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="reward-watchdog")
    atexit.register(executor.shutdown, wait=True, cancel_futures=True)
    return executor


_score_worker = threading.local()


def _get_training_verifier():
    if not hasattr(_score_worker, "verifier"):
        from utils.bounded_math import BoundedMathVerifier

        _score_worker.verifier = BoundedMathVerifier(memory_bytes=None, stage_timeouts=True)
        atexit.register(_score_worker.verifier.close)
    return _score_worker.verifier


def _compute_score_guarded(answer_content, ground_truth, correct_reward):
    return _score_answer(answer_content, ground_truth, correct_reward, _get_training_verifier().compare)


async def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    correct_reward: float = 1.0,
    extra_info: Optional[dict] = None,
    **kwargs,
) -> dict:
    """VERL async entry: enforce the library's per-stage deadlines outside math processes."""
    if extra_info and "full_response_str" in extra_info:
        solution_str = extra_info["full_response_str"]
    # VERL loads this file under a dynamic name; share the canonical pool.
    from utils.reward_fn import _get_score_executor, _compute_score_guarded

    # Only these fields affect scoring; do not pickle unused rollout metadata.
    return await asyncio.get_running_loop().run_in_executor(
        _get_score_executor(), _compute_score_guarded,
        extract_answer(solution_str), ground_truth, correct_reward,
    )
