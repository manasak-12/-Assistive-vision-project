# emotion_worker_pi.py
# Raspberry Pi Emotion Recognition (Tiny ONNX CNN)
# -------------------------------------------------------

import cv2
import numpy as np
import onnxruntime as ort

class EmotionWorker:
    def __init__(self, model_path="models/emotion_fer.onnx"):
        print("[Emotion] Loading Tiny Emotion ONNX model for Pi...")

        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # 7-class FER
        self.classes = [
            "angry", "disgust", "fear", "happy",
            "sad", "surprise", "neutral"
        ]

    def preprocess(self, face_crop):
        img = cv2.resize(face_crop, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        img = np.expand_dims(img, axis=0)
        return img

    def process(self, face_crop):
        if face_crop is None or face_crop.size == 0:
            return None

        img = self.preprocess(face_crop)
        preds = self.session.run([self.output_name], {self.input_name: img})[0][0]

        idx = int(np.argmax(preds))
        emotion = self.classes[idx]
        score = float(preds[idx])

        return {
            "emotion": emotion,
            "score": score
        }
