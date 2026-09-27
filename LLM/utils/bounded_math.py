"""Process-isolated mathematical comparison with externally enforced deadlines."""

import atexit
from collections import Counter, OrderedDict
from functools import lru_cache
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import threading
import time


class BoundedMathVerifier:
    """Keep the reward rule, but enforce its deadline outside the math process.

    Exact nonempty text matches are accepted before symbolic work. A timed-out
    or memory-limited comparison of differing strings is false, with a distinct
    diagnostic status. Worker startup/protocol failures abort scoring, not score 0.
    """

    def __init__(self, code_root=None, timeout=5.0, memory_bytes=1024**3, stage_timeouts=False):
        self.code_root = Path(code_root or Path(__file__).resolve().parents[1])
        self.timeout = timeout
        self.memory_bytes = memory_bytes
        self.process = None
        self.cache = OrderedDict()
        self.counts = Counter()
        self.statuses = Counter()
        self.lock = threading.RLock()
        self.worker_requests = 0
        self.max_worker_rss_bytes = 0
        self.read_buffer = b""
        self.stage_timeouts = None
        if stage_timeouts:
            from inspect import signature
            from math_verify import parse, verify

            parse_timeout = signature(parse).parameters["parsing_timeout"].default
            verify_timeout = signature(verify).parameters["timeout_seconds"].default
            self.stage_timeouts = {"answer_parse": parse_timeout, "ground_truth_parse": parse_timeout,
                                   "verification": verify_timeout}

    def _read(self, timeout):
        deadline = time.monotonic() + timeout
        while b"\n" not in self.read_buffer:
            if not select.select([self.process.stdout], [], [], max(0, deadline - time.monotonic()))[0]:
                raise TimeoutError("math worker exceeded wall-clock deadline")
            chunk = os.read(self.process.stdout.fileno(), 4096)
            if not chunk:
                raise RuntimeError("Math worker exited unexpectedly; refusing to emit a score")
            self.read_buffer += chunk
            if len(self.read_buffer) > 65536:
                raise RuntimeError("Oversized math worker response")
        line, _, self.read_buffer = self.read_buffer.partition(b"\n")
        return json.loads(line)

    def _start(self):
        self.process = subprocess.Popen(
            [sys.executable, "-u", str(Path(__file__).resolve()), "--worker",
             str(self.code_root), str(self.memory_bytes or 0)] + (["--staged"] if self.stage_timeouts else []),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OPENBLAS_NUM_THREADS": "1",
                 "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        )
        self.counts["worker_starts"] += 1
        self.worker_requests = 0
        try:
            if self._read(180) != {"ready": True}:  # cold sympy import from NFS can exceed 30 s under load
                raise RuntimeError("Invalid math worker startup response")
        except BaseException:
            self.close()
            raise

    def close(self):
        with self.lock:
            if self.process is not None:
                if self.process.poll() is None:
                    self.process.kill()
                self.process.wait(timeout=5)
                self.process.stdin.close()
                self.process.stdout.close()
                self.process = None
                self.read_buffer = b""

    def compare(self, answer, ground_truth):
        with self.lock:
            self.counts["calls"] += 1
            answer, ground_truth = answer.strip(), ground_truth.strip()
            key = (answer, ground_truth)  # math-verify is not necessarily symmetric.
            if not answer or not ground_truth:
                result = (False, "empty_answer" if not answer else "empty_ground_truth")
            elif answer == ground_truth:
                result = (True, "exact_match")
            elif key in self.cache:
                self.counts["cache_hits"] += 1
                result = self.cache[key]
                self.cache.move_to_end(key)
            else:
                if self.worker_requests >= 2048:
                    self.close()  # Bound retained parser/SymPy caches too.
                if self.process is None:
                    self._start()
                self.worker_requests += 1
                self.counts["worker_calls"] += 1
                stage = None
                try:
                    self.process.stdin.write((json.dumps(key) + "\n").encode())
                    self.process.stdin.flush()
                    response = self._read(30 if self.stage_timeouts else self.timeout)
                    while self.stage_timeouts and isinstance(response, dict):
                        next_stage = response.get("stage")
                        stages = list(self.stage_timeouts)
                        if (next_stage not in stages or
                                (stage is not None and stages.index(next_stage) <= stages.index(stage))):
                            raise RuntimeError("Invalid math worker stage transition")
                        stage = next_stage
                        response = self._read(self.stage_timeouts[stage])
                    if (not isinstance(response, list) or len(response) != 3
                            or type(response[0]) is not bool or not isinstance(response[1], str)):
                        raise RuntimeError("Invalid math worker comparison response")
                    result = (response[0], response[1])
                    self.max_worker_rss_bytes = max(self.max_worker_rss_bytes, response[2])
                    if result[1] == "memory_limit":
                        self.close()
                except TimeoutError:
                    self.close()  # Kill AND reap; Future.result(timeout) alone cannot do this.
                    if self.stage_timeouts and stage is None:
                        raise RuntimeError("Math worker did not start scoring; refusing to emit a score")
                    self.counts["hard_timeouts"] += 1
                    result = (False, f"{stage}_timeout" if stage else "hard_timeout")
                except BaseException:
                    self.close()
                    raise
                # A training timeout is not evidence of a wrong answer: allow later
                # occurrences to be judged again, as the unguarded reward did.
                if not self.stage_timeouts or not result[1].endswith("timeout"):
                    self.cache[key] = result
                    if len(self.cache) > 4096:
                        self.cache.popitem(last=False)
            self.statuses[result[1]] += 1
            return result

    def __call__(self, answer, ground_truth):
        return self.compare(answer, ground_truth)[0]

    def statistics(self):
        return {**self.counts, "statuses": dict(self.statuses), "timeout_seconds": self.timeout,
                "stage_timeouts": self.stage_timeouts,
                "worker_memory_limit_bytes": self.memory_bytes,
                "max_worker_rss_bytes": self.max_worker_rss_bytes}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


@lru_cache(maxsize=1)
def _default_verifier():
    verifier = BoundedMathVerifier()
    atexit.register(verifier.close)
    return verifier


def validate_answer(answer, ground_truth):
    return _default_verifier()(answer, ground_truth)


def _worker(code_root, memory_bytes, staged=False):
    from contextlib import redirect_stdout
    import ctypes
    import resource
    import signal

    # Die with the scoring parent even if its pool is forcibly terminated.
    if ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "Cannot set math worker parent-death signal")
    if os.getppid() == 1:
        return
    if memory_bytes:
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    sys.path.insert(0, code_root)
    with redirect_stdout(sys.stderr):
        from utils import reward_fn

    class MemoryLimitExceeded(BaseException):
        pass

    def propagate_memory_error(function):
        def wrapped(*args, **kwargs):
            try:
                return function(*args, **kwargs)
            except MemoryError:
                # reward_fn catches ordinary Exceptions; retain the resource-limit status.
                raise MemoryLimitExceeded from None
        return wrapped

    reward_fn._cached_parse = propagate_memory_error(reward_fn._cached_parse)
    reward_fn.verify = propagate_memory_error(reward_fn.verify)
    if staged:
        protocol_stdout = sys.stdout
        cached_parse, verify = reward_fn._cached_parse, reward_fn.verify
        parse_index = 0

        def staged_parse(*args, **kwargs):
            nonlocal parse_index
            stage = ("answer_parse", "ground_truth_parse")[parse_index]
            parse_index += 1
            print(json.dumps({"stage": stage}), file=protocol_stdout, flush=True)
            return cached_parse(*args, **kwargs)

        def staged_verify(*args, **kwargs):
            print(json.dumps({"stage": "verification"}), file=protocol_stdout, flush=True)
            return verify(*args, **kwargs)

        reward_fn._cached_parse = staged_parse
        reward_fn.verify = staged_verify
    print(json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        answer, truth = json.loads(line)
        if staged:
            parse_index = 0
        try:
            with redirect_stdout(sys.stderr):
                result = reward_fn.validate_answer_with_status(answer, truth)
        except (MemoryLimitExceeded, MemoryError):
            result = (False, "memory_limit")
        print(json.dumps([*result, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024]), flush=True)
        if result[1] == "memory_limit":
            return


if __name__ == "__main__":
    if len(sys.argv) not in (4, 5) or sys.argv[1] != "--worker" or (len(sys.argv) == 5 and sys.argv[4] != "--staged"):
        raise SystemExit("Internal worker: --worker CODE_ROOT MEMORY_BYTES [--staged]")
    _worker(sys.argv[2], int(sys.argv[3]), staged=len(sys.argv) == 5)
