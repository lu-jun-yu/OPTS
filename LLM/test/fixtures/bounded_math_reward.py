"""Deliberately hostile fake comparator, used only by CPU isolation tests."""

import os
import signal
import time


def _cached_parse(answer):
    if answer == "oom":
        return bytearray(512 * 1024**2)
    return answer


def verify(*args, **kwargs):
    return True


def validate_answer_with_status(answer, truth):
    if answer in ("hang", "partial"):
        if answer == "partial":
            os.write(1, b"incomplete protocol response")
        signal.signal(signal.SIGALRM, signal.SIG_IGN)
        while True:
            time.sleep(0.01)
    if answer == "crash":
        os._exit(9)
    if answer == "noisy":
        print("library stdout must not corrupt the protocol")
    try:
        _cached_parse(answer)
    except Exception:
        return False, "parse_failed"
    return answer == "good", "math_match" if answer == "good" else "not_equivalent"
