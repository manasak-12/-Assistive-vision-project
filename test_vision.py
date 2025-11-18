"""
test_vision.py
Vision-Assist main demo (no path nav, smart audio):

- YOLO detections every frame
- Depth (if available)
- Obstacle warnings ONLY when distance < 0.5 m (any object)
- Face recognition + emotion and short speech
- OCR: reads text aloud automatically
- Scene summary: spoken every few seconds
"""

import platform
import time
import multiprocessing as mp
import cv2
import numpy as np

IS_WINDOWS = platform.system() == "Windows"

# ---------------- CONDITIONAL IMPORTS ----------------
if IS_WINDOWS:
    from modules.detector_worker_windows import DetectorWorker
    from modules.depth_worker_windows import DepthWorker
    from modules.face_worker_windows import FaceWorker
    from modules.emotion_worker_windows import EmotionWorker
    from modules.ocr_worker_windows import OCRWorker
else:
    from modules.detector_worker_pi import DetectorWorker
    from modules.depth_worker_pi import DepthWorker
    from modules.face_worker_pi import FaceWorker
    from modules.emotion_worker_pi import EmotionWorker
    from modules.ocr_worker_pi import OCRWorker

from modules.event_manager import EventManager
from modules.tracker import CentroidTracker
from modules.distance_estimator import DistanceEstimator
from modules.navigation import navigation_advice
from modules.face_recognition_engine import FaceRecognitionEngine
from modules.face_shared import set_latest_embedding
# --------------------------------------------------------


# ---------------- FRAME GRABBER -------------------------
def frame_grabber(frame_q, camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("❌ ERROR: Camera not found / cannot open.")
        return

    print("[Grabber] Camera opened.")
    time.sleep(0.5)

    try:
        while True:
            ret, frame = cap.read()
            if ret:
                try:
                    frame_q.put(frame, block=False)
                except Exception:
                    try:
                        frame_q.get(block=False)  # drop oldest
                        frame_q.put(frame, block=False)
                    except Exception:
                        pass
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        print("[Grabber] Exiting.")
# ---------------------------------------------------------


# ---------------- DETECTION PROCESS ----------------------
def detection_process(frame_q, result_q):
    detector = DetectorWorker()

    # DepthWorker may fail if model not present
    try:
        depth = DepthWorker()
        print("[Detector] DepthWorker initialised.")
    except Exception as e:
        depth = None
        print("[Detector] DepthWorker failed:", e)

    frame_count = 0
    SKIP_DEPTH = 3
    print("[Detector] Started detection worker.")

    while True:
        frame = frame_q.get()
        if frame is None:
            continue

        detections = detector.detect(frame) or []

        depth_map = None
        if depth and frame_count % SKIP_DEPTH == 0:
            try:
                depth_map = depth.estimate_depth_map(frame)
            except Exception as e:
                depth_map = None
                print("[Detector] Depth error:", e)

        try:
            result_q.put((frame, detections, depth_map), block=False)
        except Exception:
            try:
                result_q.get(block=False)
                result_q.put((frame, detections, depth_map), block=False)
            except Exception:
                pass

        frame_count += 1
# ---------------------------------------------------------


# ---------------- MAIN PROGRAM ---------------------------
def main():
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    face_recognizer = FaceRecognitionEngine("face_db")
    frame_q = mp.Queue(maxsize=2)
    result_q = mp.Queue(maxsize=2)

    # Start grabber & detector
    grabber = mp.Process(target=frame_grabber, args=(frame_q,))
    grabber.daemon = True
    grabber.start()

    detector_proc = mp.Process(target=detection_process, args=(frame_q, result_q))
    detector_proc.daemon = True
    detector_proc.start()

    # Workers
    face_worker = FaceWorker()
    emotion_worker = EmotionWorker()
    ocr_worker = OCRWorker()
    events = EventManager()
    tracker = CentroidTracker()
    distance_est = DistanceEstimator(scale_to_meters=0.02)

    print("[Main] Started. Press 'q' to quit.")

    cv2.namedWindow("Vision Assist", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Vision Assist", 1280, 720)

    summary_timer = time.time()
    SUMMARY_INTERVAL = 30  # seconds
    frame_index = 0
    last_ocr_text = ""

    # Audio behaviour config
    NEAR_DISTANCE_M = 0.5           # Only warn if < 0.5 m (when depth works)
    PERSON_TTS_COOLDOWN = 2.5       # seconds
    OBSTACLE_TTS_COOLDOWN = 3.0     # seconds

    last_person_tts = 0.0
    last_obstacle_tts = 0.0
    last_obstacle_label = None

    last_main_time = time.time()
    fps = 0.0

    try:
        while True:
            try:
                frame, detections, depth_map = result_q.get(timeout=1)
            except Exception:
                continue

            frame_index += 1
            h, w = frame.shape[:2]

            # FPS
            now = time.time()
            dt = now - last_main_time if last_main_time else 0.033
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else (1.0 / dt)
            last_main_time = now

            person_present = False
            label_counts = {}

            # ---------------- DRAW DETECTIONS ----------------
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                label = det.get("label", "").lower()
                conf = det.get("conf", 0.0)

                label_counts[label] = label_counts.get(label, 0) + 1

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
                cv2.putText(
                    frame,
                    f"{label} {conf:.2f}",
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

                if label == "person":
                    person_present = True

            # Track person boxes (not used deeply yet)
            person_boxes = [
                det["bbox"]
                for det in detections
                if det.get("label", "").lower() == "person"
            ]
            _ = tracker.update(person_boxes)

            # ---------------- PERSON LOGIC ----------------
            if person_present:
                for det in [
                    d for d in detections if d.get("label", "").lower() == "person"
                ]:
                    x1, y1, x2, y2 = det["bbox"]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w - 1, x2), min(h - 1, y2)
                    if x2 <= x1 or y2 <= y1:
                        continue

                    person_crop = frame[y1:y2, x1:x2]

                    # ---- Depth / distance ----
                    dist_m = None
                    if depth_map is not None:
                        dist_m = distance_est.bbox_distance(
                            depth_map, [x1, y1, x2, y2]
                        )

                    # Draw distance
                    if dist_m is not None:
                        cv2.putText(
                            frame,
                            f"{dist_m:.2f}m",
                            (x2 - 80, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (200, 200, 50),
                            2,
                        )

                    # Obstacle warning ONLY if very near
                    if dist_m is not None and dist_m < NEAR_DISTANCE_M:
                        events.obstacle_warning("person", dist_m)

                    # Always send person_detected event for nav side
                    events.person_detected(
                        bbox=[x1, y1, x2, y2],
                        conf=det.get("conf", None),
                        distance_m=dist_m,
                    )

                    # ---- FACE + EMOTION ----
                    try:
                        face_results = []
                        try:
                            face_results = face_worker.process(person_crop)
                        except Exception:
                            face_results = face_worker.process(frame)

                        for face in face_results:
                            fb = face.get("bbox")
                            emb = face.get("embedding")
                            if fb is None or emb is None:
                                continue

                            # Share embedding for voice "register name"
                            set_latest_embedding(emb)

                            fx1, fy1, fx2, fy2 = fb
                            GX1, GY1 = x1 + fx1, y1 + fy1
                            GX2, GY2 = x1 + fx2, y1 + fy2

                            cv2.rectangle(
                                frame, (GX1, GY1), (GX2, GY2), (255, 165, 0), 2
                            )

                            # 1) FACE RECOGNITION
                            name, score = face_recognizer.match(emb)
                            if name:
                                label_text = name
                            else:
                                label_text = "Unknown"

                            cv2.putText(
                                frame,
                                label_text,
                                (GX1, GY1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (0, 255, 0),
                                2,
                            )

                            # 2) EMOTION
                            emo_label = None
                            face_crop = frame[GY1:GY2, GX1:GX2]
                            if face_crop.size > 0:
                                emo = emotion_worker.process(face_crop)
                                if emo:
                                    emo_label = emo["emotion"]
                                    cv2.putText(
                                        frame,
                                        emo_label,
                                        (GX1, GY2 + 25),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.7,
                                        (50, 255, 255),
                                        2,
                                    )
                                    events.emotion_event(emo_label)

                            # 3) SPEECH: ONLY WHEN PERSON IS CLOSE + COOLDOWN
                            if dist_m is not None and dist_m < NEAR_DISTANCE_M:
                                now_t = time.time()
                                if now_t - last_person_tts > PERSON_TTS_COOLDOWN:
                                    last_person_tts = now_t
                                    if name:
                                        phrase = f"{name} ahead"
                                    else:
                                        phrase = "Person ahead"
                                    if emo_label:
                                        phrase += f", {emo_label}"
                                    events.speak(phrase)

                    except Exception:
                        # Don't crash on face/emotion errors
                        pass

            # ---------------- GENERIC OBSTACLE WARNINGS ----------------
            # Any label (wall, chair, sofa, car, etc.) if depth is available
            if depth_map is not None:
                now_t = time.time()
                for det in detections:
                    label = det.get("label", "").lower()
                    if label == "person":
                        continue  # already handled

                    x1, y1, x2, y2 = det["bbox"]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w - 1, x2), min(h - 1, y2)
                    if x2 <= x1 or y2 <= y1:
                        continue

                    dist_m = distance_est.bbox_distance(
                        depth_map, [x1, y1, x2, y2]
                    )
                    if dist_m is None:
                        continue

                    if dist_m < NEAR_DISTANCE_M:
                        if (
                            now_t - last_obstacle_tts > OBSTACLE_TTS_COOLDOWN
                            or label != last_obstacle_label
                        ):
                            last_obstacle_tts = now_t
                            last_obstacle_label = label
                            events.obstacle_warning(label, dist_m)
                            events.speak(f"{label} ahead in {dist_m:.1f} meters")

            # ------------------- OCR --------------------
            if frame_index % 15 == 0:
                try:
                    text = ocr_worker.process(frame)
                except Exception:
                    text = None

                if text:
                    text = text.strip()
                    if text and text != last_ocr_text:
                        last_ocr_text = text
                        lines = text.splitlines()
                        display_text = lines[0] if lines else text
                        cv2.putText(
                            frame,
                            display_text[:80],
                            (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (255, 255, 0),
                            2,
                        )
                        # send to navigation & speak automatically
                        events.log_text(text)
                        events.speak(display_text[:80])

            # ------------------- SCENE SUMMARY --------------------
            if time.time() - summary_timer > SUMMARY_INTERVAL:
                summary_timer = time.time()
                if label_counts:
                    s = ", ".join(
                        [f"{v} {k}" for k, v in label_counts.items() if v > 0]
                    )
                    events.scene_summary(s)
                    # ALSO speak it, so you hear it even if STT is broken
                    events.speak(f"I see {s}")

            # ------------------- HUD --------------------
            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                2,
            )

            status = "Idle"
            if person_present:
                status = "Person"
            cv2.putText(
                frame,
                f"Status: {status}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (200, 200, 200),
                2,
            )

            # Optional depth visualization
            if depth_map is not None:
                try:
                    depth_norm = cv2.normalize(
                        depth_map, None, 0, 255, cv2.NORM_MINMAX
                    )
                    depth_norm = depth_norm.astype("uint8")
                    depth_vis = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)
                    cv2.imshow("Depth Map", depth_vis)
                except Exception:
                    pass

            cv2.imshow("Vision Assist", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("[Main] Interrupted by user.")
    finally:
        print("[Main] Shutting down.")
        try:
            grabber.terminate()
        except Exception:
            pass
        try:
            detector_proc.terminate()
        except Exception:
            pass
        cv2.destroyAllWindows()
        print("[Main] Cleanup done. Exiting.")


if __name__ == "__main__":
    main()
