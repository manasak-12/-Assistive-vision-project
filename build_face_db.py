# build_face_db.py
"""
Builds a simple face embedding database from images in face_db/ using InsightFace.

Usage:
    python build_face_db.py

Requirements:
    - folder face_db/ with images
    - insightface installed
"""

import os
import glob
import numpy as np
import cv2
from insightface.app import FaceAnalysis

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
FACE_DB_IMAGES_DIR = os.path.join(ROOT_DIR, "face_db")
FACE_DB_NPZ = os.path.join(ROOT_DIR, "face_db.npz")

def extract_name_from_filename(path):
    filename = os.path.basename(path)
    base, _ = os.path.splitext(filename)
    # name is before first underscore if present
    if "_" in base:
        return base.split("_", 1)[0]
    return base

def main():
    if not os.path.isdir(FACE_DB_IMAGES_DIR):
        print(f"❌ face_db folder not found at {FACE_DB_IMAGES_DIR}")
        return

    image_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        image_paths.extend(glob.glob(os.path.join(FACE_DB_IMAGES_DIR, ext)))

    if not image_paths:
        print(f"❌ No images found in {FACE_DB_IMAGES_DIR}")
        return

    print(f"[DB] Found {len(image_paths)} images in face_db/")

    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(320, 320))

    embeddings = []
    names = []

    for path in image_paths:
        name = extract_name_from_filename(path)
        img = cv2.imread(path)
        if img is None:
            print(f"[DB] Failed to read {path}, skipping")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = app.get(img_rgb)

        if not faces:
            print(f"[DB] No face found in {path}, skipping")
            continue

        # take the biggest face
        faces_sorted = sorted(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)
        face = faces_sorted[0]

        emb = face.normed_embedding  # L2-normalized embedding
        embeddings.append(emb)
        names.append(name)

        print(f"[DB] Added {name} from {os.path.basename(path)}")

    if not embeddings:
        print("❌ No faces were added. Check your images.")
        return

    embeddings = np.stack(embeddings, axis=0)
    names = np.array(names)

    np.savez(FACE_DB_NPZ, embeddings=embeddings, names=names)
    print(f"[DB] Saved face_db.npz with {len(names)} entries at {FACE_DB_NPZ}")

if __name__ == "__main__":
    main()
