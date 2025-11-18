# detector_worker_windows.py
# PyTorch YOLO version for Windows/Laptop
# -----------------------------------------------------

import cv2
from ultralytics import YOLO

class DetectorWorker:
    def __init__(self, model_path="models/yolov8n.pt", imgsz=320, conf=0.35):
        self.model = YOLO(model_path)         # PyTorch model (works on laptop)
        self.imgsz = imgsz
        self.conf = conf

    def detect(self, frame):
        """
        Returns YOLO detections from the PyTorch model.
        """
        results = self.model.predict(
            frame, 
            imgsz=self.imgsz, 
            conf=self.conf, 
            verbose=False
        )

        detections = []
        boxes = results[0].boxes

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf.cpu().numpy())
            cls = int(box.cls.cpu().numpy())
            label = self.model.names[cls]

            detections.append({
                "bbox": [x1, y1, x2, y2],
                "conf": conf,
                "label": label
            })

        return detections
