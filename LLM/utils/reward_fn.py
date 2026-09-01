# Copyright 2025 Junyu Lu (Julian Lou). All rights reserved.

"""
Reward function for \\boxed{answer} format.

A response earns reward iff it contains \\boxed{...} with the correct answer.
"""

import re
from functools import lru_cache
from typing import Optional

from math_verify import parse, verify


@lru_cache(maxsize=65536)
def _cached_parse(s: str):
    return parse(s)


def extract_answer(response_str: str) -> Optional[str]:
    """Extract the answer from \\boxed{...}.

    Searches the content after the last </think> tag when present (to skip
    intermediate \\boxed{} inside thinking), otherwise the whole response.
    """
    # Strip thinking block: only look after </think>
    think_end = response_str.rfind("</think>")
    if think_end != -1:
        answer_part = response_str[think_end + len("</think>"):]
    else:
        answer_part = response_str

    # Match \boxed{...}, handling nested braces
    matches = re.findall(r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', answer_part, re.DOTALL)
    if matches:
        return matches[0].strip()
    return None


def validate_answer(answer: str, ground_truth: str) -> bool:
    """Validate if the extracted answer matches the ground truth.

    Uses math-verify for robust mathematical equivalence checking,
    with a fallback to simple string comparison.

    Supported match types (via math-verify):
    - Plain numbers: 42 == 42.0
    - LaTeX fractions: \\frac{1}{2} == 0.5 == 1/2
    - LaTeX expressions: \\sqrt{2}, x^{2}+1, etc.
    - Sets: {1,3} \\cup {2,4} == {1,2,3,4}
    - Percentages: 10\\% == 0.1
    - Text/multiple choice: A, B, C, D
    """
    try:
        parsed_answer = _cached_parse(answer)
        parsed_gt = _cached_parse(ground_truth)
        return verify(parsed_gt, parsed_answer)
    except Exception:
        # Fallback: simple string comparison
        norm = lambda s: re.sub(r'[\$,\s]', '', s.strip())
        return norm(answer) == norm(ground_truth)


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    correct_reward: float = 1.0,
    extra_info: Optional[dict] = None,
    **kwargs,
) -> dict:
    """Compute the score for a response.

    Reward = correct_reward iff \\boxed{...} is present and the extracted
    answer matches ground truth; otherwise 0.

    Args:
        solution_str: The response string.
        ground_truth: The expected answer.
        correct_reward: Reward for a correct answer.
        extra_info: Optional extra info dict, may contain 'full_response_str' for tree search.

    Returns:
        A dictionary containing the training reward (`score`), correctness
        (`acc`), and the extracted answer (`pred`) used by validation metrics.
    """
    # OPTS_TTPO: Use full response string if available (for tree search)
    if extra_info and "full_response_str" in extra_info:
        solution_str = extra_info["full_response_str"]

    answer_content = extract_answer(solution_str)
    acc = 0.0
    total_score = 0.0
    if answer_content is not None and validate_answer(answer_content, ground_truth):
        acc = 1.0
        total_score += correct_reward

    return {
        "score": total_score,
        "acc": acc,
        "pred": answer_content if answer_content is not None else "",
    }
