"""Shared last-complete-box matching for decoded text and original token bytes."""

import re


def find_last_boxed_span(text):
    """Return (marker start, content start, content end), or None."""
    pattern = r"\\boxed[ \t\r\n\f\v]*\{"
    slash, left, right = "\\", "{", "}"
    if isinstance(text, bytes):
        pattern = pattern.encode("ascii")
        slash, left, right = map(ord, (slash, left, right))
    for match in reversed(list(re.finditer(pattern, text))):
        depth, escaped = 1, False
        for pos in range(match.end(), len(text)):
            char = text[pos]
            if escaped:
                escaped = False
            elif char == slash:
                escaped = True
            elif char == left:
                depth += 1
            elif char == right:
                depth -= 1
                if depth == 0:
                    return match.start(), match.end(), pos
    return None
