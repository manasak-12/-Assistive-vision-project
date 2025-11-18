# detector_worker_pi.py
# ONNX YOLO version optimized for Raspberry Pi
# -----------------------------------------------------

import cv2
import numpy as np
import onnxruntime as ort

class DetectorWorker:
    def __init__(self, model_path="models/yolov8n.onnx", imgsz=320, conf=0.35):
        self.imgsz = imgsz
        self.conf = conf
        
        # Load ONNX model (Pi supports IR 12)
        self.session = ort.InferenceSession(
            model_path, 
            providers=["CPUExecutionProvider"]
        )
        
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # class names for YOLOv8
        self.names = self.load_class_names()

    def load_class_names(self):
        # Default COCO labels for YOLOv8
        return [
            "person","bicycle","car","motorcycle","airplane","bus","train",
            "truck","boat","traffic light","fire hydrant","stop sign","parking meter",
            "bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra",
            "giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis",
            "snowboard","sports ball","kite","baseball bat","baseball glove","skateboard",
            "surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon",
            "bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza",
            "donut","cake","chair","couch","potted plant","bed","table","toilet","tv","laptop",
            "mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink",
            "refrigerator","book","clock","vase","scissors","teddy bear","hair drier",
            "toothbrush"
        ]

    def preprocess(self, frame):
        img = cv2.resize(frame, (self.imgsz, self.imgsz))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))       # HWC → CHW
        img = np.expand_dims(img, axis=0)        # add batch dimension
        return img

    def detect(self, frame):
        """
        Runs ONNX YOLO inference on Raspberry Pi.
        """
        img = self.preprocess(frame)
        outputs = self.session.run([self.output_name], {self.input_name: img})[0]

        detections = []
        for det in outputs[0]:
            conf = det[4]
            if conf < self.conf:
                continue

            cls = int(det[5])
            x1, y1, x2, y2 = map(int, det[:4])

            detections.append({
                "bbox": [x1, y1, x2, y2],
                "conf": float(conf),
                "label": self.names[cls]
            })

        return detections
