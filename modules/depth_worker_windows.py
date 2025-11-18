# depth_worker_windows.py
# PyTorch MiDaS model for Windows/Laptop
# -----------------------------------------------------

import cv2
import torch
import numpy as np
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

class DepthWorker:
    def __init__(self, model_path="models/dpt_swin2_tiny_256.pt", device="cpu"):
        self.device = device

        print("[Depth] Loading PyTorch MiDaS model...")
        self.model = torch.hub.load(
            "intel-isl/MiDaS", 
            "DPT_Swin2_Tiny_256"          # Lightweight, fast CPU model
        )
        self.model.to(self.device)
        self.model.eval()

        self.transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform

    def estimate_depth_map(self, frame):
        """
        Returns depth prediction for an input frame.
        Output: normalized depth map (0-255)
        """
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(img).to(self.device)

        with torch.no_grad():
            prediction = self.model(input_batch)

        pred = prediction.squeeze().cpu().numpy()
        pred_norm = cv2.normalize(pred, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        return pred_norm
