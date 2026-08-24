import signal
from contextlib import contextmanager


class TimeoutError(Exception):
    pass


@contextmanager
def time_limit(seconds: int = 3):
    """Single-process timeout guard. Isticmaal multiprocessing.Process haddii
    aad u baahato worker-pool parallel verification."""
    def handler(signum, frame):
        raise TimeoutError(f"Timed out after {seconds}s")

    signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)