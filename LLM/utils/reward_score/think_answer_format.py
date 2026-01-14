# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Reward function for <think>...</think>\\boxed{answer} format.

Note: The prompt already contains "<think>\n" in chat_template,
so the response should be: "...thinking...</think>\n...\\boxed{answer}..."
"""

import re
from typing import Optional, Tuple


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
    """Extract the answer from \\boxed{...}.

    If there are multiple \\boxed{} in the response, only the first one is taken.

    Args:
        response_str: The response string.

    Returns:
        The answer content, or None if not found.
    """
    # Match \boxed{...}, handling nested braces
    # Use findall to find all matches, then take the first one
    matches = re.findall(r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', response_str, re.DOTALL)
    if matches:
        return matches[0].strip()
    return None


def normalize_answer(answer: str) -> str:
    """Normalize the answer for comparison.

    Args:
        answer: The answer string to normalize.

    Returns:
        Normalized answer string.
    """
    answer = answer.strip()
    # Remove $, commas, and spaces
    answer = re.sub(r'[\$,\s]', '', answer)
    # Handle boxed answers: \boxed{...}
    boxed_match = re.search(r'\\boxed\{(.*?)\}', answer)
    if boxed_match:
        answer = boxed_match.group(1)
    return answer


def compute_score(
    solution_str: str,
    ground_truth: str,
    format_reward: float = 0.5,
    correct_reward: float = 0.5,
    extra_info: Optional[dict] = None,
) -> float:
    """Compute the score for a response.

    Total reward = format_reward + correct_reward (if both conditions met)

    Args:
        solution_str: The response string (without <think> prefix, which is in prompt).
        ground_truth: The expected answer.
        format_reward: Reward for correct format (0.5).
        correct_reward: Reward for correct answer (0.5).
        extra_info: Optional extra info dict, may contain 'full_response_str' for tree search.

    Returns:
        The computed score (0, 0.5, or 1.0).
    """
    # OPTS_TTPO: Use full response string if available (for tree search)
    if extra_info and "full_response_str" in extra_info:
        solution_str = extra_info["full_response_str"]

    total_score = 0.0

    # Check format: must have </think> followed by \boxed{...}
    if check_format(solution_str):
        total_score += format_reward

    # Check answer correctness
    answer_content = extract_answer(solution_str)
    if answer_content is not None:
        normalized_answer = normalize_answer(answer_content)
        normalized_ground_truth = normalize_answer(ground_truth)
        if normalized_answer == normalized_ground_truth:
            total_score += correct_reward

    return total_score
