"""
test_vision.py
Complete integrated demo for Vision-Assist:
- Multiprocess pipeline (frame grabber + detector)
- Conditional execution: Face/Emotion only when PERSON detected
- OCR only when TEXT-like detected
- Depth only when PERSON or VEHICLE detected
- Tracker (CentroidTracker), DistanceEstimator, Navigation advice
- EventManager for TTS, logging, obstacle warnings, scene summary
- Works on Windows (PyTorch) and Raspberry Pi (ONNX) via auto-switch
"""

import platform
import time
import multiprocessing as mp
import cv2
import numpy as np

# -------------------------------------------------------
# Auto-select workers based on platform
# -------------------------------------------------------
IS_WINDOWS = platform.system() == "Windows"

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

# -------------------------------------------------------
# Frame grabber process
# -------------------------------------------------------
def frame_grabber(frame_q, camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("❌ ERROR: Camera not found / cannot open.")
        return

    # Warm up camera
    time.sleep(0.5)
    print("[Grabber] Camera opened.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            # if queue is full drop the oldest to keep fresh frames
            try:
                frame_q.put(frame, block=False)
            except Exception:
                try:
                    _ = frame_q.get(block=False)
                    frame_q.put(frame, block=False)
                except Exception:
                    pass
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        print("[Grabber] Exiting.")


# -------------------------------------------------------
# Detection worker process (YOLO + Depth)
# -------------------------------------------------------
def detection_process(frame_q, result_q, detector_args=None, depth_args=None):
    # instantiate workers inside process
    detector = DetectorWorker() if detector_args is None else DetectorWorker(**detector_args)

    # depth is optional: if it fails, continue without depth
    depth = None
    try:
        depth = DepthWorker() if depth_args is None else DepthWorker(**depth_args)
    except Exception as e:
        print("[Detector] DepthWorker init failed, continuing without depth:", e)
        depth = None

    frame_count = 0
    SKIP_DEPTH = 3  # compute depth every N frames

    print("[Detector] Started detection worker.")
    try:
        while True:
            # get latest frame (block until available)
            frame = frame_q.get()
            if frame is None:
                continue

            detections = detector.detect(frame) or []

            depth_map = None
            if depth is not None and (frame_count % SKIP_DEPTH == 0):
                try:
                    depth_map = depth.estimate_depth_map(frame)
                except Exception as e:
                    depth_map = None
                    print("[Detector] Depth error:", e)

            # push results
            try:
                result_q.put((frame, detections, depth_map), block=False)
            except Exception:
                # if cannot put, drop one and re-put (keep queue fresh)
                try:
                    _ = result_q.get(block=False)
                    result_q.put((frame, detections, depth_map), block=False)
                except Exception:
                    pass

            frame_count += 1
    except KeyboardInterrupt:
        pass
    finally:
        print("[Detector] Exiting.")


# -------------------------------------------------------
# Main process: UI + face/emotion/ocr conditional pipeline
# -------------------------------------------------------
def main():
    # Use spawn (default on Windows, but explicit is safer)
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        # already set
        pass

    # Use a context so our queues/processes are from the same start method
    ctx = mp.get_context("spawn")

    frame_q = ctx.Queue(maxsize=2)
    result_q = ctx.Queue(maxsize=2)

    # Start background processes
    grabber = ctx.Process(target=frame_grabber, args=(frame_q,), name="GrabberProcess")
    detector_proc = ctx.Process(
        target=detection_process,
        args=(frame_q, result_q),
        name="DetectorProcess",
    )

    grabber.start()
    detector_proc.start()

    # Local workers in main process (face / emotion / ocr / utilities)
    face_worker = FaceWorker()
    emotion_worker = EmotionWorker()
    ocr_worker = OCRWorker()
    events = EventManager()
    tracker = CentroidTracker(max_disappeared=30, max_distance=80)
    distance_est = DistanceEstimator(scale_to_meters=0.02)  # tune later

    summary_timer = time.time()
    SUMMARY_INTERVAL = 12.0  # seconds
    last_main_time = time.time()
    fps = 0.0

    print("[Main] Started. Press 'q' in the window to quit.")

    try:
        while True:
            try:
                frame, detections, depth_map = result_q.get(timeout=1.0)
            except Exception:
                # no frame received, try again
                continue

            # compute FPS
            now = time.time()
            dt = now - last_main_time if last_main_time else 0.033
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else (1.0 / dt)
            last_main_time = now

            # prepare flags
            person_present = False
            vehicle_present = False
            text_present = False

            # Normalize detection list
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                label = det.get("label", "").lower()

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
                cv2.putText(
                    frame,
                    f"{label} {det.get('conf', 0):.2f}",
                    (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 200, 0),
                    1,
                )

                if label == "person":
                    person_present = True
                if label in ["car", "truck", "bus", "motorcycle", "bicycle", "vehicle"]:
                    vehicle_present = True
                if label in ["text", "sign", "board", "screen", "book", "label"]:
                    text_present = True

            # Update tracker using person boxes only
            person_boxes = [
                det["bbox"]
                for det in detections
                if det.get("label", "").lower() == "person"
            ]
            objects = tracker.update(person_boxes)  # dict: id -> centroid

            # Build labels summary for scene summary
            label_counts = {}
            for det in detections:
                lbl = det.get("label", "unknown")
                label_counts[lbl] = label_counts.get(lbl, 0) + 1

            # If person present, run face & emotion per-person crop
            if person_present:
                for det in [
                    d for d in detections if d.get("label", "").lower() == "person"
                ]:
                    x1, y1, x2, y2 = det["bbox"]
                    h, w = frame.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w - 1, x2), min(h - 1, y2)
                    if x2 <= x1 or y2 <= y1:
                        continue

                    person_crop = frame[y1:y2, x1:x2]

                    # estimate distance if depth_map available
                    dist_m = None
                    if depth_map is not None:
                        dist_m = distance_est.bbox_distance(
                            depth_map, [x1, y1, x2, y2]
                        )

                    # show distance on frame
                    if dist_m is not None:
                        cv2.putText(
                            frame,
                            f"{dist_m:.2f}m",
                            (x2 - 60, y2 + 18),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (200, 200, 50),
                            2,
                        )

                    # navigation advice
                    advice = navigation_advice(
                        [x1, y1, x2, y2], frame.shape[1], dist_m
                    )
                    if advice != "clear":
                        events.speak(advice)
                        cv2.putText(
                            frame,
                            advice,
                            (x1, y2 + 36),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            (0, 0, 255),
                            2,
                        )

                    # obstacle warning using events (named args for safety)
                    if dist_m is not None:
                        events.obstacle_warning(label="person", distance_m=dist_m)

                    # Run face detection on the person crop
                    try:
                        face_results = face_worker.process(person_crop)
                    except Exception:
                        # fallback: whole frame
                        try:
                            face_results = face_worker.process(frame)
                        except Exception:
                            face_results = []

                    for face in face_results:
                        fbbox = face.get("bbox")
                        if fbbox is None:
                            continue

                        fx1, fy1, fx2, fy2 = fbbox
                        if fx1 < 0 or fy1 < 0 or fx2 - fx1 < 0 or fy2 - fy1 < 0:
                            continue

                        # convert to full frame coords if relative to crop
                        if (
                            fx2 <= person_crop.shape[1]
                            and fy2 <= person_crop.shape[0]
                        ):
                            Gx1, Gy1, Gx2, Gy2 = (
                                fx1 + x1,
                                fy1 + y1,
                                fx2 + x1,
                                fy2 + y1,
                            )
                        else:
                            Gx1, Gy1, Gx2, Gy2 = fx1, fy1, fx2, fy2

                        cv2.rectangle(
                            frame,
                            (Gx1, Gy1),
                            (Gx2, Gy2),
                            (255, 165, 0),
                            2,
                        )

                        # Emotion on face crop
                        crop_for_emotion = frame[Gy1:Gy2, Gx1:Gx2]
                        if crop_for_emotion.size > 0:
                            emo = emotion_worker.process(crop_for_emotion)
                            if emo:
                                cv2.putText(
                                    frame,
                                    f"{emo['emotion']} {emo['score']:.2f}",
                                    (Gx1, Gy1 - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6,
                                    (50, 255, 255),
                                    2,
                                )
                                events.emotion_event(emo["emotion"])

                        events.face_detected()

            # OCR only if text-like detected
            if text_present:
                try:
                    text_output = ocr_worker.process(frame)
                except Exception:
                    text_output = None

                if text_output:
                    lines = text_output.splitlines()
                    display_text = lines[0] if len(lines) > 0 else text_output
                    cv2.putText(
                        frame,
                        display_text[:80],
                        (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 0),
                        2,
                    )
                    events.log_text(text_output)
                    events.speak("Text detected")

            # Periodic scene summary
            if time.time() - summary_timer > SUMMARY_INTERVAL:
                summary_timer = time.time()
                if label_counts:
                    s = ", ".join([f"{v} {k}" for k, v in label_counts.items()])
                    events.scene_summary(s)

            # HUD: FPS and small legend
            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                2,
            )
            status = "Person" if person_present else ("Text" if text_present else "Idle")
            cv2.putText(
                frame,
                f"Status: {status}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (200, 200, 200),
                2,
            )

            # show depth map if available
            if depth_map is not None:
                try:
                    depth_vis = cv2.applyColorMap(depth_map, cv2.COLORMAP_INFERNO)
                    cv2.imshow("Depth Map", depth_vis)
                except Exception:
                    pass

            # show main frame
            cv2.imshow("Vision Assist", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    except KeyboardInterrupt:
        print("[Main] Interrupted by user.")
    finally:
        print("[Main] Shutting down.")

        # Close queues so background feeder threads exit cleanly
        try:
            frame_q.close()
            frame_q.join_thread()
        except Exception:
            pass
        try:
            result_q.close()
            result_q.join_thread()
        except Exception:
            pass

        # Try to stop child processes cleanly
        for p in [grabber, detector_proc]:
            try:
                if p.is_alive():
                    p.join(timeout=3)
            except Exception:
                pass

        # Force terminate if still alive
        for p in [grabber, detector_proc]:
            try:
                if p.is_alive():
                    print(f"[Main] Terminating {p.name}...")
                    p.terminate()
            except Exception:
                pass

        cv2.destroyAllWindows()
        print("[Main] Cleanup done. Exiting.")


if __name__ == "__main__":
    main()
