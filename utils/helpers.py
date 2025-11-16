# utils/helpers.py
import cv2
import numpy as np
import os
import time
import onnxruntime as ort
from utils.logger import log

# ---------------------------------------------------------
# Load ONNX Model (Cross-platform: Windows + Linux + Pi)
# ---------------------------------------------------------
def load_onnx_model(model_path, providers=None):
    if not os.path.exists(model_path):
        log.error(f"ONNX model not found: {model_path}")
        raise FileNotFoundError(model_path)

    if providers is None:
        providers = ["CPUExecutionProvider"]

    try:
        session = ort.InferenceSession(model_path, providers=providers)
        log.success(f"Loaded ONNX model: {model_path}")
        return session
    except Exception as e:
        log.error(f"Failed to load ONNX model: {model_path}")
        raise e


# ---------------------------------------------------------
# Preprocess Image for YOLO / MiDaS
# ---------------------------------------------------------
def preprocess_image(img, size=(640, 640)):
    if img is None:
        raise ValueError("Received empty frame in preprocess_image()")

    img_resized = cv2.resize(img, size)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    img_chw = np.transpose(img_norm, (2, 0, 1))      # HWC → CHW
    img_batch = img_chw[np.newaxis, ...]             # Add batch dim

    return img_batch


# ---------------------------------------------------------
# Convert ONNX output (tensor) to numpy
# ---------------------------------------------------------
def to_numpy(tensor):
    if isinstance(tensor, np.ndarray):
        return tensor
    return np.array(tensor)


# ---------------------------------------------------------
# Haversine Distance (GPS) in meters
# ---------------------------------------------------------
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000  # earth radius in meters

    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*(np.sin(dlon/2)**2)
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c


# ---------------------------------------------------------
# Clamp utility
# ---------------------------------------------------------
def clamp(v, min_v, max_v):
    return max(min_v, min(v, max_v))


# ---------------------------------------------------------
# Timer utility (for FPS or delays)
# ---------------------------------------------------------
class Timer:
    def __init__(self):
        self.last = time.time()

    def elapsed(self):
        return time.time() - self.last

    def reset(self):
        self.last = time.time()
