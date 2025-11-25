from collections import deque
import threading
from typing import List

_log_buffer = deque(maxlen=200)
_lock = threading.Lock()


def add(msg: str):
    """Add a line to the in-memory log buffer."""
    global _log_buffer
    with _lock:
        _log_buffer.append(msg)


def get_all() -> List[str]:
    """Get a copy of all log lines."""
    with _lock:
        return list(_log_buffer)
