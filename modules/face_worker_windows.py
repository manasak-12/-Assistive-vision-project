# modules/face_worker_windows.py
"""
FaceWorker for Windows

- Uses InsightFace (buffalo_l) via ONNXRuntime (CPUExecutionProvider)
- Returns list of faces: {"bbox": [x1,y1,x2,y2], "embedding": np.array}
- Designed to be called only when 'person' is detected in the frame.
"""

import cv2
import numpy as np

try:
    from insightface.app import FaceAnalysis
except Exception as e:
    FaceAnalysis = None
    print("[FaceWorkerWin] insightface import failed:", e)


class FaceWorker:
    def __init__(self, det_size=(320, 320)):
        if FaceAnalysis is None:
            raise RuntimeError("InsightFace not available on Windows.")

        print("[FaceWorkerWin] Loading InsightFace models for Windows...")
        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"],  # CPU only to avoid GPU headache
        )
        # smaller det_size => faster, less lag
        self.app.prepare(ctx_id=0, det_size=det_size)
        print("[FaceWorkerWin] InsightFace model loaded successfully.")

    def process(self, frame_bgr):
        """
        frame_bgr: np.ndarray (H, W, 3) in BGR (OpenCV) format.
        Returns: list of dicts:
            { "bbox": [x1,y1,x2,y2], "embedding": np.array(512,) }
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return []

        # InsightFace expects BGR np.uint8
        faces = self.app.get(frame_bgr)
        results = []

        for f in faces:
            # bbox is [x1,y1,x2,y2] as float; convert + clamp
            x1, y1, x2, y2 = f.bbox.astype(int)
            h, w = frame_bgr.shape[:2]
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w - 1, x2))
            y2 = max(0, min(h - 1, y2))
            if x2 <= x1 or y2 <= y1:
                continue

            emb = getattr(f, "normed_embedding", None)
            if emb is None:
                continue
            emb = np.asarray(emb, dtype="float32")

            results.append(
                {
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "embedding": emb,
                }
            )

        return results
