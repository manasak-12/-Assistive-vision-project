"""
FULL VisionAssist Demo:
- Camera + YOLO + Depth + OCR + Face + Emotion
- Speech (Vosk + DroidCam mic)
- NavigationWorker (path + turn-by-turn)
- EventBus connects EVERYTHING
"""

import time
import multiprocessing as mp
import cv2
import platform

IS_WINDOWS = platform.system() == "Windows"

# ---------------------- Module Imports -----------------------
from modules.event_bus import EventBus
from modules.event_manager_nav import NavigationEventManager
from modules.navigation_worker import NavigationWorker
from modules.speech_worker import SpeechWorker

from modules.tracker import CentroidTracker
from modules.distance_estimator import DistanceEstimator
from modules.navigation import navigation_advice

from modules.face_recognition_engine import FaceRecognitionEngine

# Windows / Pi switch
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
# ---------------------------------------------------------------


# ---------------------- Frame Grabber -------------------------
def frame_grabber(frame_q, cam_index=0):
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print("❌ Camera not found.")
        return

    print("[Grabber] Camera opened.")

    while True:
        ret, frame = cap.read()
        if ret:
            try:
                frame_q.put(frame, block=False)
            except:
                try:
                    frame_q.get(block=False)
                    frame_q.put(frame, block=False)
                except:
                    pass
# ---------------------------------------------------------------


# ---------------------- Detection Worker ----------------------
def detection_process(frame_q, result_q):
    detector = DetectorWorker()
    try:
        depth = DepthWorker()
    except:
        depth = None

    SKIP_DEPTH = 3
    frame_count = 0

    print("[Detector] Ready.")

    while True:
        frame = frame_q.get()
        detections = detector.detect(frame)
        depth_map = None

        if depth and frame_count % SKIP_DEPTH == 0:
            try:
                depth_map = depth.estimate_depth_map(frame)
            except:
                depth_map = None

        try:
            result_q.put((frame, detections, depth_map), block=False)
        except:
            try:
                result_q.get(block=False)
                result_q.put((frame, detections, depth_map), block=False)
            except:
                pass

        frame_count += 1
# ---------------------------------------------------------------


# ---------------------- MAIN SYSTEM ---------------------------
def main():
    mp.set_start_method("spawn", force=True)

    # Event bus
    bus = EventBus()

    # Navigation + Speech Workers
    nav_manager = NavigationEventManager(event_bus=bus)
    nav_worker = NavigationWorker(event_bus=bus)
    speech_worker = SpeechWorker(event_bus=bus)

    nav_worker.start()
    speech_worker.start()

    # Vision modules
    face_rec = FaceRecognitionEngine("face_db")
    face_worker = FaceWorker()
    emotion_worker = EmotionWorker()
    ocr_worker = OCRWorker()
    tracker = CentroidTracker()
    dist_est = DistanceEstimator(scale_to_meters=0.02)

    frame_q = mp.Queue(maxsize=2)
    result_q = mp.Queue(maxsize=2)

    # Start camera + detection processes
    p1 = mp.Process(target=frame_grabber, args=(frame_q,))
    p2 = mp.Process(target=detection_process, args=(frame_q, result_q))
    p1.start()
    p2.start()

    # Demo route (Bangalore)
    DEMO_ROUTE = [
        {"lat": 12.9716, "lon": 77.5946, "instruction": "Start moving forward", "distance_m": 20},
        {"lat": 12.9718, "lon": 77.5949, "instruction": "Turn left in 10 meters", "distance_m": 10},
        {"lat": 12.9720, "lon": 77.5954, "instruction": "Continue straight for 50 meters", "distance_m": 50},
        {"lat": 12.9724, "lon": 77.5960, "instruction": "Destination ahead", "distance_m": 5},
    ]

    # Tell navigation to load this route
    bus.publish({
        "type": "navigation.set_route",
        "route": DEMO_ROUTE,
        "source": "vision_assist_full"
    })

    print("\n🎯 FULL SYSTEM RUNNING — Camera + Audio + Navigation + Vision\n")
    print("🎤 Speak into DroidCam mic: 'start navigation', 'stop', 'repeat', 'describe screen'\n")
    print("Press Q to quit.\n")

    cv2.namedWindow("Vision Assist", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Vision Assist", 1280, 720)

    frame_id = 0
    last_ocr = ""

    while True:
        try:
            frame, detections, depth_map = result_q.get(timeout=1)
        except:
            continue

        frame_id += 1
        h, w = frame.shape[:2]

        # -------------------- Drawing + Navigation Advice --------------------
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = det["label"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

            if label == "person":
                # Depth
                distance_m = None
                if depth_map is not None:
                    distance_m = dist_est.bbox_distance(depth_map, [x1,y1,x2,y2])

                # Advice for obstacles
                advice = navigation_advice([x1,y1,x2,y2], w, distance_m)
                if advice != "clear":
                    bus.publish({"type": "speech.say", "text": advice})

                # Face Recognition + Emotion
                crop = frame[y1:y2, x1:x2]
                try:
                    faces = face_worker.process(crop)
                except:
                    faces = []

                for face in faces:
                    fb = face["bbox"]
                    fx1, fy1, fx2, fy2 = [fb[0]+x1, fb[1]+y1, fb[2]+x1, fb[3]+y1]
                    emb = face["embedding"]

                    # Name
                    name, score = face_rec.match(emb)
                    if name:
                        bus.publish({"type": "speech.say", "text": f"{name} ahead"})

                    cv2.putText(frame, name or "Unknown",
                                (fx1, fy1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(0,255,0),2)

                    # Emotion
                    emo = emotion_worker.process(frame[fy1:fy2, fx1:fx2])
                    if emo:
                        cv2.putText(frame, emo["emotion"],
                                    (fx1, fy2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,255,255),2)
                        bus.publish({"type":"emotion.detected", "emotion": emo["emotion"]})

        # ------------------------- OCR -------------------------
        if frame_id % 15 == 0:
            text = ocr_worker.process(frame) or ""
            if text.strip() and text != last_ocr:
                last_ocr = text
                bus.publish({"type": "speech.say", "text": text.split("\n")[0][:80]})

        # ---------------------- Render -------------------------
        cv2.imshow("Vision Assist", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    print("Shutting down...")

    p1.terminate()
    p2.terminate()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
