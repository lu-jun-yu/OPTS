"""Reuse OPTS answer text and locate boxed boundaries in original tokens."""

import json
from bisect import bisect_right
from functools import lru_cache
from itertools import accumulate

import numpy as np

from utils.boxed import find_last_boxed_span


@lru_cache(maxsize=8)
def _token_bytes(tokenizer):
    """Cache the byte-level vocabulary, without re-encoding generated text."""
    decoder = getattr(getattr(tokenizer, "backend_tokenizer", None), "decoder", None)
    if decoder is None or json.loads(decoder.__getstate__()).get("type") != "ByteLevel":
        raise ValueError("OPTS boxed-token alignment requires a ByteLevel tokenizer decoder")

    # Inverse of the reversible byte-to-Unicode alphabet used by ByteLevel BPE.
    byte_values = list(range(ord("!"), ord("~") + 1)) + list(range(161, 173)) + list(range(174, 256))
    characters = list(byte_values)
    extra_character = 256
    for value in range(256):
        if value not in byte_values:
            byte_values.append(value)
            characters.append(extra_character)
            extra_character += 1
    byte_decoder = {chr(character): value for character, value in zip(characters, byte_values)}
    special_ids = set(tokenizer.all_special_ids)
    added_ids = set(tokenizer.get_added_vocab().values())
    return {
        token_id: (
            b""
            if token_id in special_ids
            else token.encode("utf-8")
            if token_id in added_ids
            else bytes(byte_decoder[character] for character in token)
        )
        for token, token_id in tokenizer.get_vocab().items()
    }


def find_first_boxed_token(tokenizer, response_ids):
    """Return the original token containing the first ``\\boxed``, or -1.

    Positions count all original tokens, including skipped special tokens; -1
    means no marker. The caller supplies only the generated answer (including
    inherited answer prefixes), with padding removed. Locate the marker in raw
    bytes: decoded character offsets are not token offsets, and re-encoding the
    text may merge tokens differently from the actual generated sequence.
    This function never calls the tokenizer's decode or encode methods.
    """
    if hasattr(response_ids, "tolist"):
        response_ids = response_ids.tolist()
    token_bytes = _token_bytes(tokenizer)
    pieces = [token_bytes[token_id] for token_id in response_ids]
    byte_pos = b"".join(pieces).find(b"\\boxed")
    return -1 if byte_pos < 0 else bisect_right(list(accumulate(map(len, pieces))), byte_pos)


def find_last_boxed_token(tokenizer, response_ids):
    """Return the token containing the last complete box's opening marker.

    Match original token bytes, including inherited answer prefixes. No text
    decoding or re-encoding is needed, even when a marker spans several tokens.
    """
    if hasattr(response_ids, "tolist"):
        response_ids = response_ids.tolist()
    token_bytes = _token_bytes(tokenizer)
    pieces = [token_bytes[token_id] for token_id in response_ids]
    text = b"".join(pieces)
    span = find_last_boxed_span(text)
    return -1 if span is None else bisect_right(list(accumulate(map(len, pieces))), span[0])


def decode_response_strs(batch, tokenizer, max_prompt_length, response_length, skip_special_tokens=True):
    """Cache each full answer and its last complete boxed token, excluding the raw prompt.

    The boxed position is relative to the full generated answer, not the child
    suffix. These per-trajectory fields survive batch merging/reordering, but
    prepare_next_round_input does not copy them into a newly generated child.
    """
    if skip_special_tokens and "last_boxed_token_pos" in batch.non_tensor_batch:
        return batch.non_tensor_batch["full_response_str"].tolist()

    input_ids = batch.batch["input_ids"]
    attention_mask = batch.batch["attention_mask"]
    raw_prompt_lens = batch.non_tensor_batch["raw_prompt_len"]
    cached_texts = batch.non_tensor_batch.get("full_response_str")
    responses = []
    boxed_positions = []
    for i in range(input_ids.shape[0]):
        valid_prompt_len = int(attention_mask[i, :max_prompt_length].sum().item())
        start_pos = (max_prompt_length - valid_prompt_len) + int(raw_prompt_lens[i])
        response_ids = input_ids[i, start_pos : start_pos + response_length]
        valid_mask = attention_mask[i, start_pos : start_pos + response_length].bool()
        response_ids = response_ids[valid_mask]
        if skip_special_tokens:
            text = None if cached_texts is None else cached_texts[i]
            if text is None:  # Legacy/offline rollout without a decoded reward text.
                text = tokenizer.decode(response_ids, skip_special_tokens=True)
            responses.append(text)
            boxed_positions.append(find_last_boxed_token(tokenizer, response_ids) if "\\boxed" in text else -1)
        else:
            responses.append(tokenizer.decode(response_ids, skip_special_tokens=False))
    if skip_special_tokens:
        batch.non_tensor_batch["full_response_str"] = np.array(responses, dtype=object)
        batch.non_tensor_batch["last_boxed_token_pos"] = np.array(boxed_positions, dtype=np.int64)
    return responses
