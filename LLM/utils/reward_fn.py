# Copyright 2025 Junyu Lu (Julian Lou). All rights reserved.

"""
Reward function for <think>...</think>\\boxed{answer} format.

Note: The prompt already contains "<think>\n" in chat_template,
so the response should be: "...thinking...</think>\n...\\boxed{answer}..."
"""

import re
from typing import Optional, Tuple

from math_verify import parse, verify


def check_format(response_str: str) -> bool:
    """Check if the response follows the strict format: ...思考内容...</think>\n...\\boxed{答案}...

    The response should:
    1. End the thinking section with </think>
    2. Have \\boxed{...} after </think>

    Args:
        response_str: The response string to check (without the <think> prefix).

    Returns:
        True if the format is correct, False otherwise.
    """
    # Strict format: must have </think> followed by \boxed{...}
    pattern = r'^.*</think>\s*.*\\boxed\{.*\}.*$'
    return bool(re.match(pattern, response_str, re.DOTALL))


def extract_answer(response_str: str) -> Optional[str]:
    """Extract the answer from \\boxed{...} after </think>.

    Only searches in the content after the last </think> tag to avoid
    picking up intermediate \\boxed{} attempts inside the thinking block.

    Args:
        response_str: The response string.

    Returns:
        The answer content, or None if not found.
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
        parsed_answer = parse(answer)
        parsed_gt = parse(ground_truth)
        return verify(parsed_gt, parsed_answer)
    except Exception:
        # Fallback: simple string comparison
        norm = lambda s: re.sub(r'[\$,\s]', '', s.strip())
        return norm(answer) == norm(ground_truth)


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    format_reward: float = 0.1,
    correct_reward: float = 0.9,
    extra_info: Optional[dict] = None,
    **kwargs,
) -> float:
    """Compute the score for a response.

    Total reward = format_reward + correct_reward (if both conditions met)

    Args:
        solution_str: The response string (without <think> prefix, which is in prompt).
        ground_truth: The expected answer.
        format_reward: Reward for correct format (0.1).
        correct_reward: Reward for correct answer (0.9).
        extra_info: Optional extra info dict, may contain 'full_response_str' for tree search.

    Returns:
        The computed score (0, 0.1, 0.9, or 1.0).
    """
    # OPTS_TTPO: Use full response string if available (for tree search)
    if extra_info and "full_response_str" in extra_info:
        solution_str = extra_info["full_response_str"]

    total_score = 0.0

    # Check format: must have </think> followed by \boxed{...}
    if check_format(solution_str):
        total_score += format_reward

    # Check answer correctness (using math-verify for robust matching)
    answer_content = extract_answer(solution_str)
    if answer_content is not None:
        if validate_answer(answer_content, ground_truth):
            total_score += correct_reward

    return total_score
