# emotion_worker_windows.py
# Windows Emotion Recognition (FER)
# -------------------------------------------------------

from fer import FER
import cv2

class EmotionWorker:
    def __init__(self):
        print("[Emotion] Initializing FER model (Windows)...")
        self.detector = FER(mtcnn=True)   # uses fast MTCNN + CNN emotion model

    def process(self, face_crop):
        """
        Takes a cropped face image and returns the top emotion.
        """
        if face_crop is None or face_crop.size == 0:
            return None

        results = self.detector.detect_emotions(face_crop)

        if len(results) == 0:
            return None

        # FER gives emotion probabilities
        emotions = results[0]["emotions"]

        # Pick highest scoring emotion
        emotion = max(emotions, key=emotions.get)

        return {
            "emotion": emotion,
            "score": emotions[emotion]
        }
