# # app/vision.py

# from dataclasses import dataclass
# from typing import List, Tuple, Optional
# from collections import Counter
# import os

# import cv2
# import numpy as np
# from ultralytics import YOLO
# import easyocr
# import face_recognition
# from fer import FER


# @dataclass
# class Detection:
#     cls_name: str
#     conf: float
#     bbox: Tuple[int, int, int, int]  


# @dataclass
# class RecognizedPerson:
#     name: str              
#     emotion: Optional[str]  
#     bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)


# class VisionSystem:
#     def __init__(
#         self,
#         model_path: str = "yolov8m.pt",
#         conf_threshold: float = 0.55,
#         iou_threshold: float = 0.5,
#         imgsz: int = 640,
#         enable_ocr: bool = True,
#         enable_faces: bool = True,
#         faces_folder: str = "faces_db",
#     ):
#         """
#         Vision system using:
#         - YOLOv8 for object/person detection
#         - EasyOCR for text/board reading
#         - face_recognition + FER for face recognition and emotion
#         """
#         # YOLO
#         self.model = YOLO(model_path)
#         self.conf_threshold = conf_threshold
#         self.iou_threshold = iou_threshold
#         self.imgsz = imgsz

#         self.allowed_classes = {
#             "person",
#             "chair",
#             "couch",
#             "sofa",
#             "bed",
#             "dining table",
#             "tv",
#             "tvmonitor",
#             "cell phone",
#             "laptop",
#             "keyboard",
#             "mouse",
#             "bottle",
#             "cup",
#             "book",
#             "backpack",
#             "handbag",
#             "suitcase",
#             "refrigerator",
#             "microwave",
#             "remote",
#         }

#         # OCR
#         self.ocr_reader = easyocr.Reader(["en"], gpu=False) if enable_ocr else None

#         # Faces & emotion
#         self.enable_faces = enable_faces
#         self.known_face_encodings: List[np.ndarray] = []
#         self.known_face_names: List[str] = []
#         self.emotion_detector = FER(mtcnn=False) if enable_faces else None

#         if enable_faces:
#             self._load_known_faces(faces_folder)

#     # ------------- KNOWN FACES LOADING -------------

#     def _load_known_faces(self, folder: str):
#         """
#         Load known faces from faces_db/ folder.
#         Each image file name is treated as the person's name.
#         Clears old encodings first so deletions & updates are respected.
#         """
#         # Clear old faces so reloading reflects actual folder contents
#         self.known_face_encodings = []
#         self.known_face_names = []

#         if not os.path.isdir(folder):
#             print(f"[look] Faces folder '{folder}' not found. No known faces loaded.")
#             return

#         print(f"[look] Loading known faces from '{folder}'...")
#         for filename in os.listdir(folder):
#             if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
#                 continue

#             name = os.path.splitext(filename)[0]
#             path = os.path.join(folder, filename)

#             try:
#                 image = face_recognition.load_image_file(path)
#                 locations = face_recognition.face_locations(image)
#                 if not locations:
#                     print(f"[look] No face found in {filename}, skipping.")
#                     continue

#                 encodings = face_recognition.face_encodings(image, locations)
#                 if not encodings:
#                     print(f"[look] No encoding for {filename}, skipping.")
#                     continue

#                 self.known_face_encodings.append(encodings[0])
#                 self.known_face_names.append(name)
#                 print(f"[look] Loaded face for '{name}' from {filename}.")
#             except Exception as e:
#                 print(f"[look] Error loading face from {filename}: {e}")

#         if not self.known_face_encodings:
#             print("[look] No valid known faces loaded.")

#     # ------------- OBJECT / PERSON DETECTION -------------

#     def detect(self, frame) -> List[Detection]:
#         """
#         Run YOLO on a frame and return a list of filtered Detection objects.
#         """
#         results = self.model.predict(
#             frame,
#             conf=self.conf_threshold,
#             iou=self.iou_threshold,
#             imgsz=self.imgsz,
#             verbose=False,
#         )[0]

#         detections: List[Detection] = []
#         for box in results.boxes:
#             conf = float(box.conf[0])
#             cls_id = int(box.cls[0])
#             cls_name = results.names[cls_id]

#             if cls_name not in self.allowed_classes:
#                 continue

#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             detections.append(
#                 Detection(
#                     cls_name=cls_name,
#                     conf=conf,
#                     bbox=(x1, y1, x2, y2),
#                 )
#             )
#         return detections

#     def draw_detections(self, frame, detections: List[Detection]):
#         """
#         Draw bounding boxes and labels on a copy of the frame.
#         """
#         for det in detections:
#             x1, y1, x2, y2 = det.bbox
#             label = f"{det.cls_name} {det.conf:.2f}"
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(
#                 frame,
#                 label,
#                 (x1, max(y1 - 5, 0)),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.5,
#                 (0, 255, 0),
#                 1,
#             )
#         return frame

#     # ------------- FACES + EMOTION -------------

#     def recognize_faces(
#         self,
#         frame,
#         tolerance: float = 0.5,
#     ) -> List[RecognizedPerson]:
#         """
#         Detect and recognize faces in the frame.
#         Returns list of RecognizedPerson with name & emotion.
#         """
#         if not self.enable_faces:
#             return []

#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         face_locations = face_recognition.face_locations(rgb)
#         face_encodings = face_recognition.face_encodings(rgb, face_locations)

#         persons: List[RecognizedPerson] = []

#         for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
#             name = "Unknown"
#             emotion_label: Optional[str] = "neutral"

#             # ---- Match face to known faces ----
#             if self.known_face_encodings:
#                 matches = face_recognition.compare_faces(
#                     self.known_face_encodings,
#                     face_encoding,
#                     tolerance=tolerance,
#                 )
#                 face_distances = face_recognition.face_distance(
#                     self.known_face_encodings,
#                     face_encoding,
#                 )
#                 best_match_index = int(np.argmin(face_distances)) if len(face_distances) > 0 else -1

#                 if best_match_index >= 0 and matches[best_match_index]:
#                     name = self.known_face_names[best_match_index]

#             # ---- Emotion detection with FER ----
#             if self.emotion_detector is not None:
#                 face_img = frame[top:bottom, left:right]
#                 if face_img.size > 0:
#                     try:
#                         face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
#                         emo, score = self.emotion_detector.top_emotion(face_rgb) or (None, None)
#                         if emo is not None and score is not None and score >= 0.6:
#                             emotion_label = emo
#                         else:
#                             emotion_label = "neutral"
#                     except Exception:
#                         emotion_label = "neutral"

#             bbox = (left, top, right, bottom)
#             persons.append(
#                 RecognizedPerson(
#                     name=name,
#                     emotion=emotion_label,
#                     bbox=bbox,
#                 )
#             )

#         return persons

#     def draw_faces(self, frame, persons: List[RecognizedPerson]):
#         """
#         Draw face boxes and labels (name + emotion).
#         """
#         for p in persons:
#             x1, y1, x2, y2 = p.bbox
#             label = p.name
#             if p.emotion:
#                 label = f"{p.name} ({p.emotion})"
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
#             cv2.putText(
#                 frame,
#                 label,
#                 (x1, max(y1 - 5, 0)),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.5,
#                 (255, 0, 0),
#                 1,
#             )
#         return frame

#     # ------------- TEXT / BOARD READING (OCR) -------------

#     def read_text_from_frame(
#         self,
#         frame,
#         min_conf: float = 0.5,   # lowered from 0.7 to be less strict
#         auto: bool = False,
#     ) -> Optional[str]:
#         """
#         Use EasyOCR to read text from the current frame.

#         - If auto=False (manual mode), returns a full sentence for TTS.
#         - If auto=True, returns just the raw text (or None if nothing confident).
#         """
#         if self.ocr_reader is None:
#             return None if auto else "Text reading is not available."

#         # Basic preprocessing to help OCR:
#         # 1. Resize (bigger text)
#         # 2. Convert to grayscale
#         h, w = frame.shape[:2]
#         scale = 1.5
#         resized = cv2.resize(frame, (int(w * scale), int(h * scale)))

#         gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
#         gray = cv2.GaussianBlur(gray, (3, 3), 0)

#         results = self.ocr_reader.readtext(gray, detail=1)

#         texts = []
#         for (bbox, text, conf) in results:
#             text = text.strip()
#             if conf >= min_conf and text:
#                 texts.append(text)

#         if not texts:
#             return None if auto else "I could not read any clear text."

#         joined = ". ".join(texts)

#         if auto:
#             return joined

#         return f"The text I see says: {joined}."


# def build_scene_description(detections: List[Detection]) -> str:
#     """
#     Turn detections into a natural-language description.
#     Only uses filtered detections.
#     """
#     if not detections:
#         return "I do not see anything clearly."

#     counts = Counter(det.cls_name for det in detections)

#     parts = []
#     for cls, cnt in counts.items():
#         label = cls
#         if cnt > 1 and not label.endswith("s"):
#             label = label + "s"
#         parts.append(f"{cnt} {label}")

#     return "I see " + ", ".join(parts) + "."



# import os
# from dataclasses import dataclass
# from typing import List, Tuple, Optional

# import cv2
# import numpy as np
# import face_recognition
# from fer import FER
# from ultralytics import YOLO
# import easyocr


# @dataclass
# class Detection:
#     cls_name: str
#     conf: float
#     bbox: Tuple[int, int, int, int]


# @dataclass
# class RecognizedPerson:
#     name: str
#     emotion: Optional[str]
#     bbox: Tuple[int, int, int, int]


# def build_scene_description(detections: List[Detection]) -> str:
#     if not detections:
#         return "I do not see any important objects around you."

#     counts = {}
#     for d in detections:
#         counts[d.cls_name] = counts.get(d.cls_name, 0) + 1

#     parts = []
#     for cls_name, count in counts.items():
#         if count == 1:
#             parts.append(f"one {cls_name}")
#         else:
#             parts.append(f"{count} {cls_name}s")

#     if len(parts) == 1:
#         return f"I see {parts[0]} in front of you."
#     else:
#         joined = ", ".join(parts[:-1]) + " and " + parts[-1]
#         return f"I see {joined} in front of you."


# class VisionSystem:
#     def __init__(
#         self,
#         model_path: str,
#         conf_threshold: float = 0.5,
#         iou_threshold: float = 0.5,
#         imgsz: int = 640,
#         enable_ocr: bool = True,
#         enable_faces: bool = True,
#         faces_folder: str = "faces_db",
#     ):
#         self.model = YOLO(model_path)
#         self.conf_threshold = conf_threshold
#         self.iou_threshold = iou_threshold
#         self.imgsz = imgsz

#         self.allowed_classes = {
#             "person",
#             "chair",
#             "tv",
#             "cell phone",
#             "laptop",
#             "keyboard",
#             "mouse",
#             "bottle",
#             "cup",
#             "book",
#             "backpack",
#             "handbag",
#             "suitcase",
#             "refrigerator",
#             "microwave",
#             "remote",
#         }

#         self.ocr_reader = easyocr.Reader(["en"], gpu=False) if enable_ocr else None

#         self.enable_faces = enable_faces
#         self.known_face_encodings: List[np.ndarray] = []
#         self.known_face_names: List[str] = []
#         self.emotion_detector = FER(mtcnn=False) if enable_faces else None

#         if enable_faces:
#             self._load_known_faces(faces_folder)

#     # ------------- KNOWN FACES LOADING -------------

#     def _load_known_faces(self, folder: str):
#         self.known_face_encodings = []
#         self.known_face_names = []

#         if not os.path.isdir(folder):
#             print(f"[look] Faces folder '{folder}' not found. No known faces loaded.")
#             return

#         print(f"[look] Loading known faces from '{folder}'...")
#         for filename in os.listdir(folder):
#             if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
#                 continue

#             name = os.path.splitext(filename)[0]
#             path = os.path.join(folder, filename)

#             try:
#                 image = face_recognition.load_image_file(path)
#                 locations = face_recognition.face_locations(image)
#                 if not locations:
#                     print(f"[look] No face found in {filename}, skipping.")
#                     continue

#                 encodings = face_recognition.face_encodings(image, locations)
#                 if not encodings:
#                     print(f"[look] No encoding for {filename}, skipping.")
#                     continue

#                 self.known_face_encodings.append(encodings[0])
#                 self.known_face_names.append(name)
#                 print(f"[look] Loaded face for '{name}' from {filename}.")
#             except Exception as e:
#                 print(f"[look] Error loading face from {filename}: {e}")

#         if not self.known_face_encodings:
#             print("[look] No valid known faces loaded.")

#     # ------------- OBJECT / PERSON DETECTION -------------

#     def detect(self, frame) -> List[Detection]:
#         results = self.model.predict(
#             frame,
#             conf=self.conf_threshold,
#             iou=self.iou_threshold,
#             imgsz=self.imgsz,
#             verbose=False,
#         )[0]

#         detections: List[Detection] = []
#         for box in results.boxes:
#             conf = float(box.conf[0])
#             cls_id = int(box.cls[0])
#             cls_name = results.names[cls_id]

#             if cls_name not in self.allowed_classes:
#                 continue

#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             detections.append(
#                 Detection(
#                     cls_name=cls_name,
#                     conf=conf,
#                     bbox=(x1, y1, x2, y2),
#                 )
#             )
#         return detections

#     def draw_detections(self, frame, detections: List[Detection]):
#         for det in detections:
#             x1, y1, x2, y2 = det.bbox
#             label = f"{det.cls_name} {det.conf:.2f}"
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(
#                 frame,
#                 label,
#                 (x1, max(y1 - 5, 0)),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.5,
#                 (0, 255, 0),
#                 1,
#             )
#         return frame

#     # ------------- FACES + EMOTION -------------

#     def recognize_faces(
#         self,
#         frame,
#         tolerance: float = 0.5,
#     ) -> List[RecognizedPerson]:
#         if not self.enable_faces:
#             return []

#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         face_locations = face_recognition.face_locations(rgb)
#         face_encodings = face_recognition.face_encodings(rgb, face_locations)

#         persons: List[RecognizedPerson] = []

#         for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
#             name = "Unknown"
#             emotion_label: Optional[str] = "neutral"

#             if self.known_face_encodings:
#                 matches = face_recognition.compare_faces(
#                     self.known_face_encodings,
#                     face_encoding,
#                     tolerance=tolerance,
#                 )
#                 face_distances = face_recognition.face_distance(
#                     self.known_face_encodings,
#                     face_encoding,
#                 )
#                 best_match_index = int(np.argmin(face_distances)) if len(face_distances) > 0 else -1

#                 if best_match_index >= 0 and matches[best_match_index]:
#                     name = self.known_face_names[best_match_index]

#             if self.emotion_detector is not None:
#                 face_img = frame[top:bottom, left:right]
#                 if face_img.size > 0:
#                     try:
#                         face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
#                         emo, score = self.emotion_detector.top_emotion(face_rgb) or (None, None)
#                         if emo is not None and score is not None and score >= 0.6:
#                             emotion_label = emo
#                         else:
#                             emotion_label = "neutral"
#                     except Exception:
#                         emotion_label = "neutral"

#             bbox = (left, top, right, bottom)
#             persons.append(
#                 RecognizedPerson(
#                     name=name,
#                     emotion=emotion_label,
#                     bbox=bbox,
#                 )
#             )

#         return persons

#     def draw_faces(self, frame, persons: List[RecognizedPerson]):
#         for p in persons:
#             x1, y1, x2, y2 = p.bbox
#             label = p.name
#             if p.emotion:
#                 label = f"{p.name} ({p.emotion})"
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
#             cv2.putText(
#                 frame,
#                 label,
#                 (x1, max(y1 - 5, 0)),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.5,
#                 (255, 0, 0),
#                 1,
#             )
#         return frame

#     # ------------- OCR -------------

#     def read_text_from_frame(self, frame) -> str:
#         if self.ocr_reader is None:
#             return "Text reading is disabled."

#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         results = self.ocr_reader.readtext(gray, detail=0)
#         if not results:
#             return "I could not read any clear text."
#         return " ".join(results)


# import os
# from dataclasses import dataclass
# from typing import List, Tuple, Optional

# import cv2
# import numpy as np
# import face_recognition
# from fer import FER
# from ultralytics import YOLO
# import easyocr


# @dataclass
# class Detection:
#     cls_name: str
#     conf: float
#     bbox: Tuple[int, int, int, int]


# @dataclass
# class RecognizedPerson:
#     name: str
#     emotion: Optional[str]
#     bbox: Tuple[int, int, int, int]


# def build_scene_description(detections: List[Detection]) -> str:
#     if not detections:
#         return "I do not see any important objects around you."

#     counts = {}
#     for d in detections:
#         counts[d.cls_name] = counts.get(d.cls_name, 0) + 1

#     parts = []
#     for cls_name, count in counts.items():
#         if count == 1:
#             parts.append(f"one {cls_name}")
#         else:
#             parts.append(f"{count} {cls_name}s")

#     if len(parts) == 1:
#         return f"I see {parts[0]} in front of you."
#     else:
#         joined = ", ".join(parts[:-1]) + " and " + parts[-1]
#         return f"I see {joined} in front of you."


# class VisionSystem:
#     def __init__(
#         self,
#         model_path: str = "yolov8m.pt",
#         conf_threshold: float = 0.55,
#         iou_threshold: float = 0.5,
#         imgsz: int = 640,
#         enable_ocr: bool = True,
#         enable_faces: bool = True,
#         faces_folder: str = "faces_db",
#     ):
#         self.model = YOLO(model_path)
#         self.conf_threshold = conf_threshold
#         self.iou_threshold = iou_threshold
#         self.imgsz = imgsz

#         self.allowed_classes = {
#             "person",
#             "chair",
#             "couch",
#             "sofa",
#             "bed",
#             "dining table",
#             "tv",
#             "tvmonitor",
#             "cell phone",
#             "laptop",
#             "keyboard",
#             "mouse",
#             "bottle",
#             "cup",
#             "book",
#             "backpack",
#             "handbag",
#             "suitcase",
#             "refrigerator",
#             "microwave",
#             "remote",
#         }

#         self.ocr_reader = easyocr.Reader(["en"], gpu=False) if enable_ocr else None

#         self.enable_faces = enable_faces
#         self.known_face_encodings: List[np.ndarray] = []
#         self.known_face_names: List[str] = []
#         self.emotion_detector = FER(mtcnn=False) if enable_faces else None

#         if enable_faces:
#             self._load_known_faces(faces_folder)

#     # ------------- KNOWN FACES LOADING -------------

#     def _load_known_faces(self, folder: str):
#         self.known_face_encodings = []
#         self.known_face_names = []

#         if not os.path.isdir(folder):
#             print(f"[look] Faces folder '{folder}' not found. No known faces loaded.")
#             return

#         print(f"[look] Loading known faces from '{folder}'...")
#         for filename in os.listdir(folder):
#             if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
#                 continue

#             name = os.path.splitext(filename)[0]
#             path = os.path.join(folder, filename)

#             try:
#                 image = face_recognition.load_image_file(path)
#                 locations = face_recognition.face_locations(image)
#                 if not locations:
#                     print(f"[look] No face found in {filename}, skipping.")
#                     continue

#                 encodings = face_recognition.face_encodings(image, locations)
#                 if not encodings:
#                     print(f"[look] No encoding for {filename}, skipping.")
#                     continue

#                 self.known_face_encodings.append(encodings[0])
#                 self.known_face_names.append(name)
#                 print(f"[look] Loaded face for '{name}' from {filename}.")
#             except Exception as e:
#                 print(f"[look] Error loading face from {filename}: {e}")

#         if not self.known_face_encodings:
#             print("[look] No valid known faces loaded.")

#     # ------------- OBJECT / PERSON DETECTION -------------

#     def detect(self, frame) -> List[Detection]:
#         results = self.model.predict(
#             frame,
#             conf=self.conf_threshold,
#             iou=self.iou_threshold,
#             imgsz=self.imgsz,
#             verbose=False,
#         )[0]

#         detections: List[Detection] = []
#         for box in results.boxes:
#             conf = float(box.conf[0])
#             cls_id = int(box.cls[0])
#             cls_name = results.names[cls_id]

#             if cls_name not in self.allowed_classes:
#                 continue

#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             detections.append(
#                 Detection(
#                     cls_name=cls_name,
#                     conf=conf,
#                     bbox=(x1, y1, x2, y2),
#                 )
#             )
#         return detections

#     def draw_detections(self, frame, detections: List[Detection]):
#         for det in detections:
#             x1, y1, x2, y2 = det.bbox
#             label = f"{det.cls_name} {det.conf:.2f}"
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(
#                 frame,
#                 label,
#                 (x1, max(y1 - 5, 0)),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.5,
#                 (0, 255, 0),
#                 1,
#             )
#         return frame

#     # ------------- FACES + EMOTION -------------

#     def recognize_faces(
#         self,
#         frame,
#         tolerance: float = 0.5,
#     ) -> List[RecognizedPerson]:
#         if not self.enable_faces:
#             return []

#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         face_locations = face_recognition.face_locations(rgb)
#         face_encodings = face_recognition.face_encodings(rgb, face_locations)

#         persons: List[RecognizedPerson] = []

#         for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
#             name = "Unknown"
#             emotion_label: Optional[str] = "neutral"

#             if self.known_face_encodings:
#                 matches = face_recognition.compare_faces(
#                     self.known_face_encodings,
#                     face_encoding,
#                     tolerance=tolerance,
#                 )
#                 face_distances = face_recognition.face_distance(
#                     self.known_face_encodings,
#                     face_encoding,
#                 )
#                 best_match_index = int(np.argmin(face_distances)) if len(face_distances) > 0 else -1

#                 if best_match_index >= 0 and matches[best_match_index]:
#                     name = self.known_face_names[best_match_index]

#             if self.emotion_detector is not None:
#                 face_img = frame[top:bottom, left:right]
#                 if face_img.size > 0:
#                     try:
#                         face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
#                         emo, score = self.emotion_detector.top_emotion(face_rgb) or (None, None)
#                         if emo is not None and score is not None and score >= 0.6:
#                             emotion_label = emo
#                         else:
#                             emotion_label = "neutral"
#                     except Exception:
#                         emotion_label = "neutral"

#             bbox = (left, top, right, bottom)
#             persons.append(
#                 RecognizedPerson(
#                     name=name,
#                     emotion=emotion_label,
#                     bbox=bbox,
#                 )
#             )

#         return persons

#     def draw_faces(self, frame, persons: List[RecognizedPerson]):
#         for p in persons:
#             x1, y1, x2, y2 = p.bbox
#             label = p.name
#             if p.emotion:
#                 label = f"{p.name} ({p.emotion})"
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
#             cv2.putText(
#                 frame,
#                 label,
#                 (x1, max(y1 - 5, 0)),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.5,
#                 (255, 0, 0),
#                 1,
#             )
#         return frame

#     # ------------- OCR -------------

#     def read_text_from_frame(self, frame) -> str:
#         if self.ocr_reader is None:
#             return "Text reading is disabled."

#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         results = self.ocr_reader.readtext(gray, detail=0)
#         if not results:
#             return "I could not read any clear text."
#         return " ".join(results)

import os
from dataclasses import dataclass
from typing import List, Tuple, Optional

import cv2
import numpy as np
import face_recognition
from fer import FER
from ultralytics import YOLO
import easyocr


@dataclass
class Detection:
    cls_name: str
    conf: float
    bbox: Tuple[int, int, int, int]


@dataclass
class RecognizedPerson:
    name: str
    emotion: Optional[str]
    bbox: Tuple[int, int, int, int]


def build_scene_description(detections: List[Detection]) -> str:
    if not detections:
        return "I do not see any important objects around you."

    counts = {}
    for d in detections:
        counts[d.cls_name] = counts.get(d.cls_name, 0) + 1

    parts = []
    for cls_name, count in counts.items():
        if count == 1:
            parts.append(f"one {cls_name}")
        else:
            parts.append(f"{count} {cls_name}s")

    if len(parts) == 1:
        return f"I see {parts[0]} in front of you."
    else:
        joined = ", ".join(parts[:-1]) + " and " + parts[-1]
        return f"I see {joined} in front of you."


class VisionSystem:
    def __init__(
        self,
        model_path: str = "yolov8m.pt",
        conf_threshold: float = 0.55,
        iou_threshold: float = 0.5,
        imgsz: int = 640,
        enable_ocr: bool = True,
        enable_faces: bool = True,
        faces_folder: str = "faces_db",
    ):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz

        self.allowed_classes = {
            "person",
            "chair",
            "tv",
            "cell phone",
            "laptop",
            "keyboard",
            "mouse",
            "bottle",
            "cup",
            "book",
            "backpack",
            "handbag",
            "suitcase",
            "refrigerator",
            "microwave",
            "remote",
        }

        # OCR
        self.ocr_reader = easyocr.Reader(["en"], gpu=False) if enable_ocr else None

        # Faces + emotions
        self.enable_faces = enable_faces
        self.known_face_encodings: List[np.ndarray] = []
        self.known_face_names: List[str] = []
        self.emotion_detector = FER(mtcnn=False) if enable_faces else None

        if enable_faces:
            self._load_known_faces(faces_folder)

    # ------------- KNOWN FACES LOADING -------------

    def _load_known_faces(self, folder: str):
        self.known_face_encodings = []
        self.known_face_names = []

        if not os.path.isdir(folder):
            print(f"[look] Faces folder '{folder}' not found. No known faces loaded.")
            return

        print(f"[look] Loading known faces from '{folder}'...")
        for filename in os.listdir(folder):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            name = os.path.splitext(filename)[0]
            path = os.path.join(folder, filename)

            try:
                image = face_recognition.load_image_file(path)
                locations = face_recognition.face_locations(image)
                if not locations:
                    print(f"[look] No face found in {filename}, skipping.")
                    continue

                encodings = face_recognition.face_encodings(image, locations)
                if not encodings:
                    print(f"[look] No encoding for {filename}, skipping.")
                    continue

                self.known_face_encodings.append(encodings[0])
                self.known_face_names.append(name)
                print(f"[look] Loaded face for '{name}' from {filename}.")
            except Exception as e:
                print(f"[look] Error loading face from {filename}: {e}")

        if not self.known_face_encodings:
            print("[look] No valid known faces loaded.")

    # ------------- OBJECT / PERSON DETECTION -------------

    def detect(self, frame) -> List[Detection]:
        results = self.model.predict(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            verbose=False,
        )[0]

        detections: List[Detection] = []
        for box in results.boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = results.names[cls_id]

            if cls_name not in self.allowed_classes:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append(
                Detection(
                    cls_name=cls_name,
                    conf=conf,
                    bbox=(x1, y1, x2, y2),
                )
            )
        return detections

    def draw_detections(self, frame, detections: List[Detection]):
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            label = f"{det.cls_name} {det.conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 5, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )
        return frame

    # ------------- FACES + EMOTION -------------

    def recognize_faces(
        self,
        frame,
        tolerance: float = 0.5,
    ) -> List[RecognizedPerson]:
        """
        - Uses face_recognition for face location + identity
        - Uses FER for emotion
        - **No strict score threshold**: if FER gives any emotion, we use it.
        """
        if not self.enable_faces:
            return []

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb)
        face_encodings = face_recognition.face_encodings(rgb, face_locations)

        persons: List[RecognizedPerson] = []

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            name = "Unknown"
            emotion_label: Optional[str] = "neutral"

            # --- identity ---
            if self.known_face_encodings:
                matches = face_recognition.compare_faces(
                    self.known_face_encodings,
                    face_encoding,
                    tolerance=tolerance,
                )
                face_distances = face_recognition.face_distance(
                    self.known_face_encodings,
                    face_encoding,
                )
                best_match_index = int(np.argmin(face_distances)) if len(face_distances) > 0 else -1

                if best_match_index >= 0 and matches[best_match_index]:
                    name = self.known_face_names[best_match_index]

            # --- emotion ---
            if self.emotion_detector is not None:
                face_img = frame[top:bottom, left:right]  # BGR crop
                if face_img.size > 0:
                    try:
                        # FER can handle BGR; we just ask for top emotion
                        emo, score = self.emotion_detector.top_emotion(face_img) or (None, None)
                        # ✅ Main fix: use whatever emo we get, don't enforce score >= 0.6
                        if emo is not None:
                            emotion_label = emo
                        else:
                            emotion_label = "neutral"
                    except Exception:
                        emotion_label = "neutral"

            bbox = (left, top, right, bottom)
            persons.append(
                RecognizedPerson(
                    name=name,
                    emotion=emotion_label,
                    bbox=bbox,
                )
            )

        return persons

    def draw_faces(self, frame, persons: List[RecognizedPerson]):
        for p in persons:
            x1, y1, x2, y2 = p.bbox
            label = p.name
            if p.emotion:
                label = f"{p.name} ({p.emotion})"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 5, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1,
            )
        return frame

    # ------------- OCR -------------

    def read_text_from_frame(self, frame) -> str:
        if self.ocr_reader is None:
            return "Text reading is disabled."

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results = self.ocr_reader.readtext(gray, detail=0)
        if not results:
            return "I could not read any clear text."
        return " ".join(results)
