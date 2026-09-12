# Copyright 2025 Junyu Lu (Julian Lou). All rights reserved.

"""
Reward function for \\boxed{answer} format.

A response earns reward iff its last complete \\boxed{...} contains the correct answer.
"""

import re
from functools import lru_cache
from typing import Optional

from math_verify import LatexExtractionConfig, parse, verify


@lru_cache(maxsize=65536)
def _cached_parse(s: str):
    # The box was already extracted: parse the whole answer, not numbers inside it.
    s = s.strip()
    for left, right in (("$$", "$$"), ("$", "$"), (r"\(", r"\)"), (r"\[", r"\]")):
        if s.startswith(left) and s.endswith(right):
            s = s[len(left):-len(right)].strip()
            break
    # Use division: math-verify's percentage comparison also accepts 10% == 10.
    percent = re.search(r"(\\?%|(?i:percentage|percent|pct))\s*$", s)
    if percent:
        s = rf"({s[:percent.start()]})/100"
    return parse(
        rf"\[{s}\]",
        extraction_config=[LatexExtractionConfig()],
        extraction_mode="first_match",
        fallback_mode="no_fallback",
    )


def extract_answer(response_str: str) -> Optional[str]:
    """Extract the last complete \\boxed{...}, regardless of think tags."""
    for match in reversed(list(re.finditer(r'\\boxed\{', response_str))):
        start = match.end()
        depth, escaped = 1, False
        for pos in range(start, len(response_str)):
            char = response_str[pos]
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return response_str[start:pos].strip()
    return None


def validate_answer(answer: str, ground_truth: str) -> bool:
    """Validate if the extracted answer matches the ground truth.

    Uses math-verify on the complete answer. If parsing fails, only identical
    nonempty strings (ignoring surrounding whitespace) are accepted.

    Supported match types (via math-verify):
    - Plain numbers: 42 == 42.0
    - LaTeX fractions: \\frac{1}{2} == 0.5 == 1/2
    - LaTeX expressions: \\sqrt{2}, x^{2}+1, etc.
    - Sets: {1,3} \\cup {2,4} == {1,2,3,4}
    - Percentages: 10\\% == 0.1
    - Text/multiple choice: A, B, C, D
    """
    answer, ground_truth = answer.strip(), ground_truth.strip()
    if not answer or not ground_truth:
        return False
    try:
        parsed_answer = _cached_parse(answer)
        parsed_gt = _cached_parse(ground_truth)
        if parsed_answer and parsed_gt:
            return verify(parsed_gt, parsed_answer)
    except Exception:
        pass
    return answer == ground_truth


def compute_score(
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
