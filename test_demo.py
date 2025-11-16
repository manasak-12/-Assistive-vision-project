import platform
import cv2
import time

# -------------------------------------------------------
# AUTO-SELECT WORKERS (Windows = PyTorch, Pi = ONNX)
# -------------------------------------------------------

if platform.system() == "Windows":
    from modules.detector_worker_windows import DetectorWorker
    from modules.depth_worker_windows import DepthWorker
    from modules.face_worker_windows import FaceWorker
    from modules.emotion_worker_windows import EmotionWorker
    print("[System] Running on Windows → Using PyTorch models")

else:
    from modules.detector_worker_pi import DetectorWorker
    from modules.depth_worker_pi import DepthWorker
    from modules.face_worker_pi import FaceWorker
    from modules.emotion_worker_pi import EmotionWorker
    print("[System] Running on Raspberry Pi → Using ONNX models")

# -------------------------------------------------------
# INITIALIZE WORKERS
# -------------------------------------------------------

detector = DetectorWorker(
    model_path="models/yolov8n.pt" if platform.system()=="Windows" else "models/yolov8n.onnx",
    imgsz=320,
    conf=0.35
)

depth = DepthWorker()
face_worker = FaceWorker()
emotion_worker = EmotionWorker()

# -------------------------------------------------------
# CAMERA SETUP
# -------------------------------------------------------

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ ERROR: Could not access webcam")
    exit()

print("✅ Camera ready — Press 'q' to exit")

FRAME_SKIP_DEPTH = 3
frame_count = 0

# -------------------------------------------------------
# MAIN LOOP
# -------------------------------------------------------

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to capture frame")
        break

    # ----------------------------------------
    # RUN YOLO DETECTION
    # ----------------------------------------
    detections = detector.detect(frame)

    # ----------------------------------------
    # FACE + EMOTION
    # ----------------------------------------
    face_results = face_worker.process(frame)

    for face in face_results:
        x1, y1, x2, y2 = face["bbox"]

        # Draw face bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 165, 0), 2)

        # Crop face for emotion
        face_crop = frame[y1:y2, x1:x2]

        emotion = emotion_worker.process(face_crop)

        if emotion:
            label = f"{emotion['emotion']} ({emotion['score']:.2f})"
            cv2.putText(
                frame, label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (50, 255, 255), 2
            )

    # ----------------------------------------
    # YOLO DRAW BOXES
    # ----------------------------------------
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = det["label"]
        conf  = det["conf"]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(
            frame, f"{label} {conf:.2f}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (0,255,0), 2
        )

    # ----------------------------------------
    # DEPTH MAP every N frames
    # ----------------------------------------
    if frame_count % FRAME_SKIP_DEPTH == 0:
        depth_map = depth.estimate_depth_map(frame)
        depth_small = cv2.applyColorMap(depth_map, cv2.COLORMAP_INFERNO)
        cv2.imshow("Depth Map", depth_small)

    # ----------------------------------------
    # SHOW VIDEO
    # ----------------------------------------
    cv2.imshow("Vision Assist Demo", frame)

    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -------------------------------------------------------
# CLEANUP
# -------------------------------------------------------

cap.release()
cv2.destroyAllWindows()
