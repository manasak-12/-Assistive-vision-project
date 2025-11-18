# depth_worker_pi.py
# ONNX MiDaS model for Raspberry Pi
# -----------------------------------------------------

import cv2
import numpy as np
import onnxruntime as ort

class DepthWorker:
    def __init__(self, model_path="models/dpt_swin2_tiny_256.onnx"):
        print("[Depth] Loading ONNX MiDaS model for Raspberry Pi...")

        self.session = ort.InferenceSession(
            model_path, 
            providers=["CPUExecutionProvider"]
        )
        
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        self.input_h = self.session.get_inputs()[0].shape[2]
        self.input_w = self.session.get_inputs()[0].shape[3]

    def preprocess(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.input_w, self.input_h))
        img = img.astype(np.float32) / 255.0

        # Normalize (ImageNet mean/std)
        img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]

        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img

    def estimate_depth_map(self, frame):
        """
        ONNX inference on Raspberry Pi
        Outputs: 0–255 normalized depth map
        """
        img = self.preprocess(frame)
        output = self.session.run([self.output_name], {self.input_name: img})[0]

        depth = output.squeeze()
        depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        return depth_norm
