# modules/face_worker_pi.py
"""
FaceWorker for Raspberry Pi

- Tries to use InsightFace (buffalo_l) via ONNXRuntime if available.
- If not available or too heavy, falls back to lightweight OpenCV Haar face detector.
- Returns list of:
    - If InsightFace: {"bbox": [x1,y1,x2,y2], "embedding": np.array}
    - If Haar fallback: {"bbox": [x1,y1,x2,y2]}  (no embedding)
"""

import cv2
import numpy as np

try:
    from insightface.app import FaceAnalysis
except Exception as e:
    FaceAnalysis = None
    print("[FaceWorkerPi] insightface import failed:", e)


class FaceWorker:
    def __init__(self, det_size=(256, 256)):
        self.use_insight = False
        self.app = None
        self.cascade = None

        # Try InsightFace first
        if FaceAnalysis is not None:
            try:
                print("[FaceWorkerPi] Attempting InsightFace on Pi...")
                self.app = FaceAnalysis(
                    name="buffalo_l",
                    providers=["CPUExecutionProvider"],
                )
                # smaller det_size to reduce lag
                self.app.prepare(ctx_id=0, det_size=det_size)
                self.use_insight = True
                print("[FaceWorkerPi] InsightFace initialised.")
            except Exception as e:
                print("[FaceWorkerPi] InsightFace init failed, will use Haar:", e)
                self.app = None
                self.use_insight = False

        # If InsightFace not available, use Haar cascade
        if not self.use_insight:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.cascade = cv2.CascadeClassifier(cascade_path)
            if self.cascade.empty():
                raise RuntimeError("[FaceWorkerPi] Failed to load Haar cascade.")
            print("[FaceWorkerPi] Using Haar cascade face detector (no embeddings).")

    def _process_insight(self, frame_bgr):
        faces = self.app.get(frame_bgr)
        results = []
        h, w = frame_bgr.shape[:2]

        for f in faces:
            x1, y1, x2, y2 = f.bbox.astype(int)
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

    def _process_haar(self, frame_bgr):
        # Convert to gray, downscale a bit for speed
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_LINEAR)
        scale_x = gray.shape[1] / small.shape[1]
        scale_y = gray.shape[0] / small.shape[0]

        faces = self.cascade.detectMultiScale(
            small,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(40, 40),
        )

        results = []
        for (sx, sy, sw, sh) in faces:
            x1 = int(sx * scale_x)
            y1 = int(sy * scale_y)
            x2 = int((sx + sw) * scale_x)
            y2 = int((sy + sh) * scale_y)
            results.append({"bbox": [x1, y1, x2, y2]})
        return results

    def process(self, frame_bgr):
        if frame_bgr is None or frame_bgr.size == 0:
            return []

        if self.use_insight and self.app is not None:
            try:
                return self._process_insight(frame_bgr)
            except Exception as e:
                print("[FaceWorkerPi] InsightFace error, falling back to Haar:", e)
                self.use_insight = False

        # Haar fallback
        return self._process_haar(frame_bgr)
