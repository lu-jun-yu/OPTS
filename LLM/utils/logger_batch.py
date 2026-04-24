import time
import logging
from contextlib import contextmanager
from typing import List, Tuple
import numpy as np
import torch

from verl import DataProto


__all__ = [
    "logger_batch",
    "timed_block",
    "log_batch_state",
    "log_sample_generations",
]


# Setup logging for batch tracking
logger_batch = logging.getLogger("opts_ttpo")
logger_batch.setLevel(logging.DEBUG)
logger_batch.propagate = False  # Prevent duplicate logs from propagating to root logger
if not logger_batch.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] %(message)s')
    handler.setFormatter(formatter)
    logger_batch.addHandler(handler)


def _flatten_valid(tensor: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if mask is None:
        return tensor.reshape(-1)
    valid = mask.to(dtype=torch.bool)
    return tensor[valid]


def _log_tensor_stats(prefix: str, name: str, tensor: torch.Tensor, mask: torch.Tensor | None = None) -> None:
    valid_values = _flatten_valid(tensor, mask)
    if valid_values.numel() == 0:
        logger_batch.info(f"{prefix} {name}: no valid elements")
        return

    valid_values = valid_values.float()
    logger_batch.info(
        f"{prefix} {name}: "
        f"min={valid_values.min().item():.4f}, "
        f"max={valid_values.max().item():.4f}, "
        f"mean={valid_values.mean().item():.4f}, "
        f"std={valid_values.std(unbiased=False).item():.4f}"
    )


@contextmanager
def timed_block(name: str, step: int = -1, round_idx: int = -1):
    """Context manager to log the execution time of a code block.

    Args:
        name: Name of the code block being timed.
        step: Global training step number.
        round_idx: OPTS round index (for multi-round generation).
    """
    step_info = f"step={step}" if step >= 0 else ""
    round_info = f"round={round_idx}" if round_idx >= 0 else ""
    prefix = f"[{step_info}][{round_info}]" if step_info or round_info else ""

    logger_batch.info(f"{prefix}[{name}] >>> START")
    start_time = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start_time
        logger_batch.info(f"{prefix}[{name}] <<< END (elapsed: {elapsed:.3f}s)")


def log_batch_state(batch: DataProto, stage: str, step: int = -1, round_idx: int = -1) -> None:
    """Log the state of a batch for debugging purposes.

    Args:
        batch: The DataProto batch to log.
        stage: A string describing the current processing stage.
        step: Global training step number.
        round_idx: OPTS round index (for multi-round generation).
    """
    step_info = f"step={step}" if step >= 0 else ""
    round_info = f"round={round_idx}" if round_idx >= 0 else ""
    prefix = f"[{step_info}][{round_info}][{stage}]"

    # Basic batch info
    batch_size = batch.batch.batch_size[0] if hasattr(batch.batch, 'batch_size') else len(batch.batch.get('input_ids', []))

    logger_batch.info(f"{prefix} === Batch State ===")
    logger_batch.info(f"{prefix} batch_size: {batch_size}")

    # Key statistics for important tensors
    if 'attention_mask' in batch.batch:
        mask = batch.batch['attention_mask']
        seq_lens = mask.sum(dim=-1)
        logger_batch.info(f"{prefix} seq_lens: min={seq_lens.min().item()}, max={seq_lens.max().item()}, mean={seq_lens.float().mean().item():.1f}")

    response_mask = batch.batch.get('response_mask')
    if 'response_mask' in batch.batch:
        resp_lens = response_mask.sum(dim=-1)
        logger_batch.info(f"{prefix} response_lens: min={resp_lens.min().item()}, max={resp_lens.max().item()}, mean={resp_lens.float().mean().item():.1f}")

    if 'token_level_rewards' in batch.batch:
        rewards = batch.batch['token_level_rewards']
        if response_mask is not None:
            rewards = rewards * response_mask.to(rewards.dtype)
        total_rewards = rewards.sum(dim=-1)
        logger_batch.info(f"{prefix} total_rewards: min={total_rewards.min().item():.4f}, max={total_rewards.max().item():.4f}, mean={total_rewards.mean().item():.4f}")

    if 'advantages' in batch.batch:
        adv = batch.batch['advantages']
        _log_tensor_stats(prefix, "advantages", adv, response_mask)

    if 'values' in batch.batch:
        vals = batch.batch['values']
        _log_tensor_stats(prefix, "values", vals, response_mask)

    logger_batch.info(f"{prefix} === End Batch State ===")


def log_sample_generations(
    global_batch: DataProto,
    batch: DataProto,
    tokenizer,
    step: int = 1,
    round_idx: int = 1,
    sorted_states: List[Tuple[int, int]] = None,
) -> None:
    """Log decoded sample prompts and responses for debugging.

    This function finds the parent trajectory from global_batch and its children from batch,
    verifies that children's prompts match parent's prompt + partial response,
    and logs the parent's prompt/response and all children's responses.

    Args:
        global_batch: The global batch containing parent trajectories from previous rounds.
        batch: The current round's batch containing newly generated children trajectories.
        tokenizer: Tokenizer for decoding token ids to text.
        step: Global training step number.
        round_idx: OPTS round index.
        sorted_states: Sorted list of (parent_idx, branch_pos) from select_next_states,
                       sorted by descending branch_pos. If None or empty, function returns early.
    """
    global_rid = global_batch.non_tensor_batch.get("rid")
    batch_pid = batch.non_tensor_batch.get("pid")
    batch_rid = batch.non_tensor_batch.get("rid")

    pad_token_id = tokenizer.pad_token_id

    # 1. Find the parent trajectory corresponding to rid[sorted_states[0][0]] in global_batch
    parent_idx = sorted_states[0][0]
    branch_pos = sorted_states[0][1]
    parent_rid = global_rid[parent_idx]

    # Find children trajectories in batch where pid == parent_rid (exclude None values explicitly)
    child_indices = [i for i, p in enumerate(batch_pid) if p is not None and p == parent_rid]

    if len(child_indices) == 0:
        logger_batch.warning(f"[step={step}][round={round_idx}] No children found for parent rid={parent_rid}")
        return

    # Get parent prompt and response from global_batch
    parent_prompt_ids = global_batch.batch["prompts"][parent_idx]
    parent_response_ids = global_batch.batch["responses"][parent_idx]

    # Remove leading pads from parent prompt (left-padded)
    nonpad_mask = parent_prompt_ids != pad_token_id
    if nonpad_mask.any():
        first_nonpad = nonpad_mask.nonzero()[0].item()
        parent_prompt_ids_valid = parent_prompt_ids[first_nonpad:]
    else:
        parent_prompt_ids_valid = parent_prompt_ids[:0]

    # Parent's expected prompt for children = parent_prompt + partial response truncated by branch_pos
    # branch_pos is the position in response where branching happened (0-indexed)
    expected_prompt_ids = torch.cat([parent_prompt_ids_valid, parent_response_ids[: branch_pos + 1]], dim=0)

    # 2. Check if all children's prompts match the expected prompt
    for child_idx in child_indices:
        child_prompt_ids = batch.batch["prompts"][child_idx]

        # Remove leading pads from child prompt
        nonpad_mask = child_prompt_ids != pad_token_id
        if nonpad_mask.any():
            first_nonpad = nonpad_mask.nonzero()[0].item()
            child_prompt_ids_valid = child_prompt_ids[first_nonpad:]
        else:
            child_prompt_ids_valid = child_prompt_ids[:0]

        # Check if child's prompt matches expected prompt
        if not torch.equal(child_prompt_ids_valid, expected_prompt_ids):
            logger_batch.error(
                f"[step={step}][round={round_idx}] Prompt mismatch for child index {child_idx}!"
            )
            logger_batch.error(
                f"Expected prompt length: {len(expected_prompt_ids)}, actual: {len(child_prompt_ids_valid)}"
            )
            logger_batch.error(
                f"Expected prompt (last 50 tokens): {expected_prompt_ids[-50:].tolist()}"
            )
            logger_batch.error(
                f"Actual prompt (last 50 tokens): {child_prompt_ids_valid[-50:].tolist()}"
            )
            return

    # 3. Log: parent trajectory's prompt and response, and all children trajectories' responses
    logger_batch.info(f"[step={step}][round={round_idx}] === Sample Generations (Parent + Children) ===")

    # Parent prompt text
    parent_prompt_text = tokenizer.decode(parent_prompt_ids_valid, skip_special_tokens=False)

    # Remove trailing pads from parent response
    nonpad_mask = parent_response_ids != pad_token_id
    if nonpad_mask.any():
        last_nonpad = nonpad_mask.nonzero()[-1].item()
        parent_response_ids_valid = parent_response_ids[: last_nonpad + 1]
    else:
        parent_response_ids_valid = parent_response_ids[:0]
    parent_response_text = tokenizer.decode(parent_response_ids_valid, skip_special_tokens=False)

    # Truncate if too long
    if len(parent_prompt_text) > 500:
        parent_prompt_text = parent_prompt_text[:250] + "...[truncated]..." + parent_prompt_text[-250:]
    if len(parent_response_text) > 500:
        parent_response_text = parent_response_text[:250] + "...[truncated]..." + parent_response_text[-250:]

    logger_batch.info(
        f"[step={step}][round={round_idx}] Parent (idx={parent_idx}, rid={parent_rid}, branch_pos={branch_pos}) PROMPT:\n{parent_prompt_text}"
    )
    logger_batch.info(
        f"[step={step}][round={round_idx}] Parent (idx={parent_idx}, rid={parent_rid}) RESPONSE:\n{parent_response_text}"
    )

    # Log all children responses from batch
    for i, child_idx in enumerate(child_indices):
        child_response_ids = batch.batch["responses"][child_idx]

        # Remove trailing pads from child response
        nonpad_mask = child_response_ids != pad_token_id
        if nonpad_mask.any():
            last_nonpad = nonpad_mask.nonzero()[-1].item()
            child_response_ids_valid = child_response_ids[: last_nonpad + 1]
        else:
            child_response_ids_valid = child_response_ids[:0]

        child_response_text = tokenizer.decode(child_response_ids_valid, skip_special_tokens=False)
        if len(child_response_text) > 500:
            child_response_text = child_response_text[:250] + "...[truncated]..." + child_response_text[-250:]

        child_rid = batch_rid[child_idx] if batch_rid is not None else f"child_{child_idx}"
        logger_batch.info(
            f"[step={step}][round={round_idx}] Child {i} (idx={child_idx}, rid={child_rid}) RESPONSE:\n{child_response_text}"
        )

    logger_batch.info(f"[step={step}][round={round_idx}] === End Sample Generations ===")
