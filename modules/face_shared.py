# modules/face_shared.py
"""
Shared storage for the latest face embedding so that
Navigation (voice commands) can register faces seen by Vision.
"""

import threading
import numpy as np

_lock = threading.Lock()
_latest_embedding = None


def set_latest_embedding(emb):
    """
    Store a copy of the latest face embedding.
    'emb' can be a list or numpy array.
    """
    if emb is None:
        return

    arr = np.array(emb, dtype="float32")
    global _latest_embedding
    with _lock:
        _latest_embedding = arr


def get_latest_embedding():
    """
    Return a copy of the last stored embedding, or None.
    """
    with _lock:
        if _latest_embedding is None:
            return None
        return _latest_embedding.copy()
