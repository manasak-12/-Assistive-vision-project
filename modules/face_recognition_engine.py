# modules/face_recognition_engine.py
"""
Simple face recognition engine.

- Stores each person as <name>.npy in face_db/
- Uses L2 distance on embeddings
- Same code for Windows and Raspberry Pi
"""

import os
import numpy as np
from numpy.linalg import norm

class FaceRecognitionEngine:
    def __init__(self, db_path="face_db", threshold=0.75):
        self.db_path = db_path
        self.threshold = threshold
        os.makedirs(db_path, exist_ok=True)
        self.database = self._load_db()

    def _load_db(self):
        db = {}
        for file in os.listdir(self.db_path):
            if file.endswith(".npy"):
                name = file[:-4]  # remove .npy
                path = os.path.join(self.db_path, file)
                try:
                    emb = np.load(path)
                    db[name] = emb.astype("float32")
                except Exception as e:
                    print(f"[FaceRec] Failed to load {path}:", e)
        print("[FaceRec] Loaded", len(db), "faces from", self.db_path)
        return db

    def match(self, emb):
        """
        Returns (name, distance).
        name = None if no match below threshold.
        """
        if emb is None:
            return None, float("inf")

        emb = np.asarray(emb, dtype="float32")
        best_name = None
        best_score = float("inf")

        for name, ref_emb in self.database.items():
            if ref_emb.shape != emb.shape:
                continue
            score = norm(ref_emb - emb)
            if score < best_score:
                best_score = score
                best_name = name

        if best_score < self.threshold:
            return best_name, best_score
        return None, best_score

    def register(self, name, emb):
        """
        Save new face embedding under given name.
        """
        emb = np.asarray(emb, dtype="float32")
        path = os.path.join(self.db_path, f"{name}.npy")
        np.save(path, emb)
        self.database[name] = emb
        print(f"[FaceRec] Registered new face: {name} -> {path}")
