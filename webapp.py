import os
import time
from typing import Dict, Any, List

import cv2
from flask import (
    Flask,
    render_template,
    Response,
    request,
    redirect,
    url_for,
    flash,
    jsonify,
)
from werkzeug.utils import secure_filename

from audio import TextToSpeech, SpeechToTextListener
from vision import VisionSystem, RecognizedPerson, Detection
from logic import (
    interpret_voice_command,
    decide_scene_speech,
    CommandType,
)
import logbuffer

# ================== CONFIG ==================

FACES_FOLDER = "faces_db"
CAMERA_INDEX = 0
AUTO_SCENE_INTERVAL = 25.0  # seconds between automatic scene descriptions

os.makedirs(FACES_FOLDER, exist_ok=True)

app = Flask(__name__)
app.secret_key = "super-secret-key"  # for flash() messages


# ================== GLOBAL OBJECTS ==================

tts = TextToSpeech(rate=0)
stt = SpeechToTextListener(model_path="models/vosk_en")
vision = VisionSystem(
    model_path="yolov8m.pt",
    conf_threshold=0.55,
    iou_threshold=0.5,
    imgsz=640,
    enable_ocr=True,
    enable_faces=True,
    faces_folder=FACES_FOLDER,
)

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    msg = "[web] WARNING: Could not open camera. Check CAMERA_INDEX."
    print(msg)
    logbuffer.add(msg)

assistant_running: bool = False
auto_talk: bool = True          # auto scene descriptions
listening_enabled: bool = True
detection_enabled: bool = True
voice_enabled: bool = True      # master audio mute: stop talking / start talking

last_auto_scene_time: float = 0.0
last_spoken: str = ""
last_command_text: str = ""

latest_detections: List[Detection] = []
latest_persons: List[RecognizedPerson] = []

face_state: Dict[str, Dict[str, Any]] = {}
frame_count: int = 0

# NEW: faces reload flag (so reload happens in camera thread, not Flask thread)
reload_faces_pending: bool = False


# ================== HELPER FUNCTIONS ==================

def describe_people(persons: List[RecognizedPerson]) -> str:
    if not persons:
        return "I do not see anyone clearly."

    parts = []
    for p in persons:
        who = p.name if p.name != "Unknown" else "an unknown person"
        if p.emotion:
            parts.append(f"{who} who looks {p.emotion}")
        else:
            parts.append(who)

    if len(parts) == 1:
        return f"I see {parts[0]}."
    else:
        joined = ", ".join(parts[:-1]) + " and " + parts[-1]
        return f"I see {joined}."


def describe_emotions(persons: List[RecognizedPerson]) -> str:
    if not persons:
        return "I cannot see anyone clearly to understand their emotion."

    parts = []
    for p in persons:
        who = p.name if p.name != "Unknown" else "an unknown person"
        emo = p.emotion or "neutral"
        parts.append(f"{who} seems {emo}")

    if len(parts) == 1:
        return parts[0] + "."
    else:
        joined = ", ".join(parts[:-1]) + " and " + parts[-1]
        return joined + "."


def analyze_face_events(
    persons: List[RecognizedPerson],
    state: Dict[str, Dict[str, Any]],
    auto_talk_flag: bool,
    now: float,
) -> str:
    """
    Detect approaching + smiling / emotion change events.
    Returns phrase to speak or "".
    """
    spoken = ""

    for p in persons:
        key = p.name if p.name != "Unknown" else "Unknown"

        x1, y1, x2, y2 = p.bbox
        area = max(1, (x2 - x1) * (y2 - y1))

        info = state.get(key, {})
        last_area = info.get("last_area")
        last_emotion = info.get("last_emotion")
        approach_cooldown_until = info.get("approach_cooldown_until", 0.0)
        emotion_cooldown_until = info.get("emotion_cooldown_until", 0.0)

        # approaching
        if auto_talk_flag and last_area is not None:
            if area > last_area * 1.6 and now > approach_cooldown_until:
                if p.name != "Unknown":
                    spoken = f"{p.name} is coming closer to you."
                else:
                    spoken = "I see an unknown person coming closer to you."
                info["approach_cooldown_until"] = now + 10.0

        # emotion change
        if auto_talk_flag and p.emotion and p.emotion != last_emotion and now > emotion_cooldown_until:
            emo = p.emotion.lower()
            if emo == "happy":
                if p.name != "Unknown":
                    spoken = f"{p.name} is smiling."
                else:
                    spoken = "I see an unknown person smiling."
            else:
                if p.name != "Unknown":
                    spoken = f"{p.name} looks {emo}."
                else:
                    spoken = f"I see an unknown person who looks {emo}."
            info["emotion_cooldown_until"] = now + 10.0

        info["last_area"] = area
        info["last_emotion"] = p.emotion
        info["last_seen"] = now
        state[key] = info

    return spoken


def handle_command(
    cmd: CommandType,
    recognized_text: str,
    frame,
    vision: VisionSystem,
    latest_dets: List[Detection],
    latest_persons_list: List[RecognizedPerson],
    last_spoken_text: str,
    auto_talk_flag: bool,
    listening_flag: bool,
    detection_flag: bool,
    voice_flag: bool,
):
    """
    Handle a parsed voice command and update flags.
    Returns:
      (response_text, auto_talk_flag, listening_flag, detection_flag, voice_flag, new_last_spoken)
    """
    response = ""

    if cmd == CommandType.NONE:
        return "", auto_talk_flag, listening_flag, detection_flag, voice_flag, last_spoken_text

    # Scene / objects
    if cmd in (CommandType.DESCRIBE_SCENE, CommandType.COUNT_OBJECTS):
        scene = decide_scene_speech(latest_dets)
        people = describe_people(latest_persons_list)
        response = f"{scene} {people}"

    # Read text
    elif cmd == CommandType.READ_TEXT:
        text = vision.read_text_from_frame(frame)
        response = text or "I could not read any clear text."

    # Who is here
    elif cmd == CommandType.WHO_IS_HERE:
        if not latest_persons_list:
            response = "I do not see anyone in front of you."
        else:
            parts = []
            for p in latest_persons_list:
                name = p.name if p.name != "Unknown" else "an unknown person"
                emo = p.emotion or "neutral"
                parts.append(f"{name} who looks {emo}")
            if len(parts) == 1:
                response = f"I see {parts[0]}."
            else:
                joined = ", ".join(parts[:-1]) + " and " + parts[-1]
                response = f"I see {joined}."

    # Repeat last
    elif cmd == CommandType.REPEAT_LAST:
        response = last_spoken_text or "I have not said anything important yet."

    # Quiet mode: FULL mute
    elif cmd == CommandType.QUIET_MODE:
        auto_talk_flag = False
        voice_flag = False
        # fully mute, no spoken response

    # Start talking again
    elif cmd == CommandType.START_TALKING:
        auto_talk_flag = True
        voice_flag = True
        response = "Got it. I will describe the scene and people for you."

    # Listening control
    elif cmd == CommandType.STOP_LISTENING:
        listening_flag = False
        if voice_flag:
            response = "Okay, I will stop listening until you say start listening."

    elif cmd == CommandType.START_LISTENING:
        listening_flag = True
        if voice_flag:
            response = "I am listening again."

    # Detection control
    elif cmd == CommandType.STOP_DETECTION:
        detection_flag = False
        if voice_flag:
            response = "I have turned off visual detection."

    elif cmd == CommandType.START_DETECTION:
        detection_flag = True
        if voice_flag:
            response = "I have turned visual detection back on."

    # Greetings / small talk
    elif cmd == CommandType.HELLO:
        if voice_flag:
            response = "Hello, I am your AI assistant. How can I help you?"

    elif cmd == CommandType.HOW_ARE_YOU:
        if voice_flag:
            response = "I am just code, but I am running fine. How are you?"

    # Fake weather
    elif cmd == CommandType.WEATHER:
        if voice_flag:
            response = "I am offline, but I hope the weather is nice where you are."

    # Emotion description
    elif cmd == CommandType.DESCRIBE_EMOTION:
        response = describe_emotions(latest_persons_list)

    new_last_spoken = response if response else last_spoken_text
    return response, auto_talk_flag, listening_flag, detection_flag, voice_flag, new_last_spoken


# ================== VIDEO STREAM GENERATOR ==================

def generate_frames():
    global cap, assistant_running, auto_talk, listening_enabled, detection_enabled, voice_enabled
    global last_auto_scene_time, last_spoken, latest_detections, latest_persons
    global face_state, frame_count, last_command_text, reload_faces_pending

    while True:
        # ---- Safe camera read ----
        try:
            success, frame = cap.read()
            if not success or frame is None:
                msg = "[web] Warning: failed to grab frame from camera."
                print(msg)
                logbuffer.add(msg)
                time.sleep(0.05)
                continue
        except Exception as e:
            msg = f"[web] Error reading frame from camera: {e}"
            print(msg)
            logbuffer.add(msg)
            time.sleep(0.05)
            continue

        frame_count += 1
        now = time.time()
        vis = frame.copy()

        # ---- Handle pending face reload (pause heavy work while reloading) ----
        if reload_faces_pending:
            reload_faces_pending = False

            # save current states
            saved_assistant_running = assistant_running
            saved_auto_talk = auto_talk
            saved_detection_enabled = detection_enabled
            saved_listening_enabled = listening_enabled

            # pause everything heavy (no detection / no command handling while we reload)
            assistant_running = False
            auto_talk = False
            detection_enabled = False
            listening_enabled = False

            try:
                logbuffer.add("[web] Reloading faces from folder (assistant paused)...")
                vision._load_known_faces(FACES_FOLDER)
                logbuffer.add("[web] Faces reloaded successfully.")
            except Exception as e:
                msg = f"[web] Error while reloading faces: {e}"
                print(msg)
                logbuffer.add(msg)
            finally:
                # restore previous states
                assistant_running = saved_assistant_running
                auto_talk = saved_auto_talk
                detection_enabled = saved_detection_enabled
                listening_enabled = saved_listening_enabled


        if assistant_running:
            # ----- Vision -----
            if detection_enabled:
                try:
                    if frame_count % 2 == 0:
                        detections = vision.detect(frame)
                        persons = vision.recognize_faces(frame)

                        latest_detections = detections
                        latest_persons = persons

                        phrase = analyze_face_events(
                            persons, face_state, auto_talk_flag=auto_talk, now=now
                        )
                        if phrase:
                            last_spoken = phrase
                            if voice_enabled:
                                tts.speak(phrase)
                            logbuffer.add(f"[look][TTS] {phrase}")

                    vis = vision.draw_detections(vis, latest_detections)
                    vis = vision.draw_faces(vis, latest_persons)
                except Exception as e:
                    msg = f"[web] Error in detection/face pipeline: {e}"
                    print(msg)
                    logbuffer.add(msg)
                    detection_enabled = False  # temporarily disable instead of crashing

            # ----- Auto scene description -----
            if (
                auto_talk
                and detection_enabled
                and (now - last_auto_scene_time) > AUTO_SCENE_INTERVAL
            ):
                try:
                    scene = decide_scene_speech(latest_detections)
                    people = describe_people(latest_persons)
                    combined = f"{scene} {people}"
                    last_spoken = combined
                    if voice_enabled:
                        tts.speak(combined)
                    last_auto_scene_time = now
                    logbuffer.add(f"[look][TTS] {combined}")
                except Exception as e:
                    msg = f"[web] Error in auto scene description: {e}"
                    print(msg)
                    logbuffer.add(msg)

            # ----- Voice commands -----
            if listening_enabled:
                recognized_text = stt.get_latest_command()
                if recognized_text:
                    last_command_text = recognized_text
                    msg = f"[web][MAIN] User said: {recognized_text!r}"
                    print(msg)
                    logbuffer.add(msg)
                    try:
                        cmd = interpret_voice_command(recognized_text)
                        (
                            response,
                            auto_talk,
                            listening_enabled,
                            detection_enabled,
                            voice_enabled,
                            last_spoken,
                        ) = handle_command(
                            cmd,
                            recognized_text,
                            frame,
                            vision,
                            latest_detections,
                            latest_persons,
                            last_spoken,
                            auto_talk,
                            listening_enabled,
                            detection_enabled,
                            voice_enabled,
                        )
                        if response:
                            msg2 = f"[web][MAIN] Responding: {response!r}"
                            print(msg2)
                            logbuffer.add(msg2)
                            if voice_enabled:
                                tts.speak(response)
                    except Exception as e:
                        msg = f"[web] Error while handling command: {e}"
                        print(msg)
                        logbuffer.add(msg)

        # ---- Encode frame for browser ----
        try:
            ret, buffer = cv2.imencode(".jpg", vis)
        except Exception as e:
            msg = f"[web] Error encoding frame: {e}"
            print(msg)
            logbuffer.add(msg)
            continue

        if not ret:
            msg = "[web] cv2.imencode returned False, skipping frame."
            print(msg)
            logbuffer.add(msg)
            continue

        frame_bytes = buffer.tobytes()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )


# ================== ROUTES ==================

@app.route("/")
def index():
    known_faces = getattr(vision, "known_face_names", [])
    return render_template(
        "index.html",
        assistant_running=assistant_running,
        known_faces=known_faces,
    )


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/status")
def status():
    return jsonify(
        {
            "assistant_running": assistant_running,
            "listening_enabled": listening_enabled,
            "detection_enabled": detection_enabled,
            "last_command": last_command_text,
            "last_spoken": last_spoken,
        }
    )


@app.route("/logs")
def logs():
    return jsonify({"lines": logbuffer.get_all()})


@app.route("/start_assistant", methods=["POST"])
def start_assistant():
    global assistant_running
    if not assistant_running:
        assistant_running = True
        logbuffer.add("[web] Assistant started.")
        if voice_enabled:
            tts.speak("Vision assistant started.")
    return redirect(url_for("index"))


@app.route("/stop_assistant", methods=["POST"])
def stop_assistant():
    global assistant_running
    if assistant_running:
        assistant_running = False
        logbuffer.add("[web] Assistant stopped.")
        if voice_enabled:
            tts.speak("Vision assistant stopped.")
    return redirect(url_for("index"))

@app.route("/upload_face", methods=["POST"])
def upload_face():
    """
    Save face image and ask camera thread to reload encodings.
    This route NEVER calls face_recognition directly (avoids native crash).
    """
    global reload_faces_pending

    try:
        file = request.files.get("face_image")
        name = request.form.get("person_name", "").strip()

        if not file or file.filename == "" or not name:
            flash("Please provide both a name and an image.")
            return redirect(url_for("index"))

        safe_name = name.strip().lower().replace(" ", "_")
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in [".jpg", ".jpeg", ".png"]:
            flash("Please upload a JPG or PNG image.")
            return redirect(url_for("index"))

        filename = secure_filename(safe_name + ext)
        path = os.path.join(FACES_FOLDER, filename)
        file.save(path)

        # just set flag; camera thread will reload safely
        reload_faces_pending = True
        logbuffer.add(f"[web] Saved face file for {name} as {filename}; reload pending.")
        flash(f"Saved face for {name}. Updating recognizer...")

        return redirect(url_for("index"))

    except Exception as e:
        msg = f"[web] Unexpected error in upload_face: {e}"
        print(msg)
        logbuffer.add(msg)
        flash("Something went wrong while adding the face.")
        return redirect(url_for("index"))



@app.route("/reload_faces", methods=["POST"])
def reload_faces():
    """
    Manual reload button -> just set the flag; camera thread does the heavy work.
    """
    global reload_faces_pending
    reload_faces_pending = True
    logbuffer.add("[web] Manual faces reload requested; will reload shortly.")
    flash("Reloading faces from folder...")
    return redirect(url_for("index"))


@app.route("/shutdown", methods=["POST"])
def shutdown():
    global cap
    stt.stop()
    if cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()
    logbuffer.add("[web] Shutting down vision assistant.")
    if voice_enabled:
        tts.speak("Shutting down vision assistant.")
    return "Shutting down."


# ================== MAIN ==================

if __name__ == "__main__":
    try:
        msg = "[web] Starting SpeechToTextListener..."
        print(msg)
        logbuffer.add(msg)
        stt.start()
        if voice_enabled:
            tts.speak("Web interface is ready.")
        logbuffer.add("[look][TTS] Web interface is ready.")
        app.run(host="127.0.0.1", port=5000, debug=False)
    finally:
        print("[web] Cleaning up...")
        logbuffer.add("[web] Cleaning up...")
        stt.stop()
        if cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
        tts.stop()
