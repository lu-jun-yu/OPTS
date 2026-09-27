"""Controlled parser/comparator failures for per-stage watchdog tests."""

import signal
import time


def _hang():
    signal.signal(signal.SIGALRM, signal.SIG_IGN)
    while True:
        time.sleep(0.01)


def _cached_parse(answer):
    if answer == "hang_parse":
        _hang()
    if answer.startswith("slow"):
        time.sleep(0.25)
    return [answer]


def verify(truth, answer, **kwargs):
    if answer == ["hang_verify"]:
        _hang()
    if answer[0].startswith("slow"):
        time.sleep(0.25)
    return answer[0] == "good" or answer[0].startswith("slow")


def validate_answer_with_status(answer, truth):
    answer, truth = _cached_parse(answer), _cached_parse(truth)
    correct = verify(truth, answer)
    return correct, "math_match" if correct else "not_equivalent"
