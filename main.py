# import time
# from typing import Dict, Any

# import cv2

# from .audio import TextToSpeech, SpeechToTextListener
# from .vision import VisionSystem, RecognizedPerson
# from logic import (
#     interpret_voice_command,
#     decide_scene_speech,
#     CommandType,
# )

# # ---------- CONFIG ----------

# CAMERA_INDEX = 0            # 0 = default webcam
# AUTO_SCENE_INTERVAL = 25.0  # seconds between automatic scene summaries


# # ---------- HELPERS ----------

# def describe_people(persons):
#     """
#     Turn list[RecognizedPerson] into a nice sentence.
#     """
#     if not persons:
#         return "I do not see anyone clearly."

#     parts = []
#     for p in persons:
#         name = p.name if p.name != "Unknown" else "an unknown person"
#         if p.emotion:
#             parts.append(f"{name} who looks {p.emotion}")
#         else:
#             parts.append(name)

#     if len(parts) == 1:
#         return f"I see {parts[0]}."
#     else:
#         joined = ", ".join(parts[:-1]) + " and " + parts[-1]
#         return f"I see {joined}."


# def describe_emotions(persons):
#     """
#     Describe emotions of visible people.
#     """
#     if not persons:
#         return "I cannot see anyone clearly to understand their emotion."

#     parts = []
#     for p in persons:
#         name = p.name if p.name != "Unknown" else "an unknown person"
#         emo = p.emotion or "neutral"
#         parts.append(f"{name} seems {emo}")

#     if len(parts) == 1:
#         return parts[0] + "."
#     else:
#         joined = ", ".join(parts[:-1]) + " and " + parts[-1]
#         return joined + "."


# def analyze_face_events(persons, state: Dict[str, Dict[str, Any]], auto_talk: bool, now: float):
#     """
#     Detect 'approaching' and 'smiling' events.

#     state: per-person memory:
#       {
#         name_key: {
#           "last_area": float,
#           "last_emotion": str,
#           "last_seen": float,
#           "approach_cooldown_until": float,
#           "emotion_cooldown_until": float,
#         }
#       }

#     Returns: phrase to speak (or None)
#     """
#     spoken_phrase = None

#     for p in persons:
#         # key: known name or generic unknown
#         key = p.name if p.name != "Unknown" else "Unknown"

#         x1, y1, x2, y2 = p.bbox
#         area = max(1, (x2 - x1) * (y2 - y1))  # bbox area

#         info = state.get(key, {})
#         last_area = info.get("last_area")
#         last_emotion = info.get("last_emotion")
#         approach_cooldown_until = info.get("approach_cooldown_until", 0.0)
#         emotion_cooldown_until = info.get("emotion_cooldown_until", 0.0)

#         # ---- approaching detection (basic: area growth) ----
#         if auto_talk and last_area is not None:
#             if area > last_area * 1.6 and now > approach_cooldown_until:
#                 if p.name != "Unknown":
#                     phrase = f"{p.name} is coming closer to you."
#                 else:
#                     phrase = "I see an unknown person coming closer to you."
#                 spoken_phrase = phrase  # last event wins if multiple
#                 info["approach_cooldown_until"] = now + 10.0  # avoid spamming

#         # ---- emotion / smiling detection ----
#         if auto_talk and p.emotion and p.emotion != last_emotion and now > emotion_cooldown_until:
#             emo = p.emotion.lower()
#             if emo == "happy":
#                 if p.name != "Unknown":
#                     phrase = f"{p.name} is smiling."
#                 else:
#                     phrase = "I see an unknown person smiling."
#             else:
#                 if p.name != "Unknown":
#                     phrase = f"{p.name} looks {emo}."
#                 else:
#                     phrase = f"I see an unknown person who looks {emo}."

#             spoken_phrase = phrase
#             info["emotion_cooldown_until"] = now + 10.0

#         # update memory
#         info["last_area"] = area
#         info["last_emotion"] = p.emotion
#         info["last_seen"] = now
#         state[key] = info

#     return spoken_phrase


# def handle_command(
#     cmd: CommandType,
#     recognized_text: str,
#     frame,
#     vision: VisionSystem,
#     latest_detections,
#     latest_persons,
#     last_spoken: str,
#     auto_talk: bool,
#     listening_enabled: bool,
#     detection_enabled: bool,
# ):
#     """
#     Handle voice command and return:
#     (response_text, auto_talk, listening_enabled, detection_enabled, new_last_spoken)
#     """

#     response = ""

#     if cmd == CommandType.NONE:
#         # No known command -> stay silent
#         return "", auto_talk, listening_enabled, detection_enabled, last_spoken

#     # ----- Scene / objects -----
#     if cmd in (CommandType.DESCRIBE_SCENE, CommandType.COUNT_OBJECTS):
#         scene = decide_scene_speech(latest_detections)
#         people = describe_people(latest_persons)
#         response = f"{scene} {people}"

#     # ----- Read text from board/sign -----
#     elif cmd == CommandType.READ_TEXT:
#         text = vision.read_text_from_frame(frame)
#         response = text or "I could not read any clear text."

#     # ----- Who is here? -----
#     elif cmd == CommandType.WHO_IS_HERE:
#         if not latest_persons:
#             response = "I do not see anyone in front of you."
#         else:
#             parts = []
#             for p in latest_persons:
#                 name = p.name if p.name != "Unknown" else "an unknown person"
#                 emo = p.emotion or "neutral"
#                 parts.append(f"{name} who looks {emo}")
#             if len(parts) == 1:
#                 response = f"I see {parts[0]}."
#             else:
#                 joined = ", ".join(parts[:-1]) + " and " + parts[-1]
#                 response = f"I see {joined}."

#     # ----- Repeat last spoken -----
#     elif cmd == CommandType.REPEAT_LAST:
#         if last_spoken:
#             response = last_spoken
#         else:
#             response = "I have not said anything important yet."

#     # ----- Quiet mode -----
#     elif cmd == CommandType.QUIET_MODE:
#         auto_talk = False
#         response = "Okay, I will only speak when you ask me."

#     # ----- Start talking automatically -----
#     elif cmd == CommandType.START_TALKING:
#         auto_talk = True
#         response = "Got it. I will describe the scene and people for you."

#     # ----- Stop / start listening -----
#     elif cmd == CommandType.STOP_LISTENING:
#         listening_enabled = False
#         response = "Okay, I will stop listening until you say start listening."

#     elif cmd == CommandType.START_LISTENING:
#         listening_enabled = True
#         response = "I am listening again."

#     # ----- Stop / start detection -----
#     elif cmd == CommandType.STOP_DETECTION:
#         detection_enabled = False
#         response = "I have turned off visual detection."

#     elif cmd == CommandType.START_DETECTION:
#         detection_enabled = True
#         response = "I have turned visual detection back on."

#     # ----- Hello / small talk -----
#     elif cmd == CommandType.HELLO:
#         response = "Hello, I am your AI assistant. How can I help you?"

#     elif cmd == CommandType.HOW_ARE_YOU:
#         response = "I am just code, but I am running fine. How are you?"

#     # ----- Fake offline weather -----
#     elif cmd == CommandType.WEATHER:
#         response = "I am offline, but I hope the weather is nice where you are."

#     # ----- Explicit emotion description -----
#     elif cmd == CommandType.DESCRIBE_EMOTION:
#         response = describe_emotions(latest_persons)

#     # default: nothing

#     new_last_spoken = response if response else last_spoken
#     return response, auto_talk, listening_enabled, detection_enabled, new_last_spoken


# # ---------- MAIN LOOP ----------

# def main():
#     # --- Init subsystems ---
#     tts = TextToSpeech(rate=170)
#     stt = SpeechToTextListener(model_path="models/vosk_en")
#     vision = VisionSystem(
#         model_path="yolov8m.pt",
#         conf_threshold=0.55,
#         iou_threshold=0.5,
#         imgsz=640,
#         enable_ocr=True,
#         enable_faces=True,
#         faces_folder="faces_db",
#     )

#     cap = cv2.VideoCapture(CAMERA_INDEX)

#     if not cap.isOpened():
#         print("[look][MAIN] Could not open camera.")
#         return

#     stt.start()

#     tts.speak("Hello, I am your AI vision assistant. I am ready.")

#     auto_talk = True          # automatic scene announcements
#     listening_enabled = True  # respond to voice commands
#     detection_enabled = True  # run YOLO + face recognition

#     last_auto_scene_time = 0.0
#     last_spoken = ""

#     latest_detections = []
#     latest_persons = []

#     face_state: Dict[str, Dict[str, Any]] = {}

#     frame_count = 0

#     try:
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 print("[look][MAIN] Failed to grab frame.")
#                 break

#             now = time.time()
#             frame_count += 1

#             # ----- Vision: objects + people -----
#             if detection_enabled:
#                 # To reduce load, run full detection every 2nd frame
#                 if frame_count % 2 == 0:
#                     detections = vision.detect(frame)
#                     persons = vision.recognize_faces(frame)

#                     latest_detections = detections
#                     latest_persons = persons

#                     # 'wow' events: approaching / smiling
#                     phrase = analyze_face_events(persons, face_state, auto_talk, now)
#                     if phrase:
#                         tts.speak(phrase)
#                         last_spoken = phrase

#                 # Draw boxes for debugging
#                 vis = frame.copy()
#                 vis = vision.draw_detections(vis, latest_detections)
#                 vis = vision.draw_faces(vis, latest_persons)
#             else:
#                 vis = frame

#             # ----- Automatic scene description every N seconds -----
#             if (
#                 auto_talk
#                 and detection_enabled
#                 and (now - last_auto_scene_time) > AUTO_SCENE_INTERVAL
#             ):
#                 scene = decide_scene_speech(latest_detections)
#                 people = describe_people(latest_persons)
#                 combined = f"{scene} {people}"
#                 tts.speak(combined)
#                 last_spoken = combined
#                 last_auto_scene_time = now

#             # ----- Voice commands -----
#             if listening_enabled:
#                 recognized_text = stt.get_latest_command()
#                 if recognized_text:
#                     print(f"[look][MAIN] User said: {recognized_text!r}")
#                     cmd = interpret_voice_command(recognized_text)
#                     (
#                         response,
#                         auto_talk,
#                         listening_enabled,
#                         detection_enabled,
#                         last_spoken,
#                     ) = handle_command(
#                         cmd,
#                         recognized_text,
#                         frame,
#                         vision,
#                         latest_detections,
#                         latest_persons,
#                         last_spoken,
#                         auto_talk,
#                         listening_enabled,
#                         detection_enabled,
#                     )

#                     if response:
#                         print(f"[look][MAIN] Responding: {response!r}")
#                         tts.speak(response)

#             # ----- Show window for debugging (press q to quit) -----
#             cv2.imshow("Vision Assist", vis)
#             if cv2.waitKey(1) & 0xFF == ord("q"):
#                 break

#     finally:
#         print("[look][MAIN] Shutting down...")
#         stt.stop()
#         cap.release()
#         cv2.destroyAllWindows()


# if __name__ == "__main__":
#     main()


# import time
# from typing import Dict, Any

# import cv2

# from .audio import TextToSpeech, SpeechToTextListener
# from .vision import VisionSystem, RecognizedPerson
# from logic import (
#     interpret_voice_command,
#     decide_scene_speech,
#     CommandType,
# )

# # ---------- CONFIG ----------

# CAMERA_INDEX = 0            # 0 = default webcam
# AUTO_SCENE_INTERVAL = 25.0  # seconds between automatic scene summaries


# # ---------- HELPERS ----------

# def describe_people(persons):
#     """
#     Turn list[RecognizedPerson] into a nice sentence.
#     """
#     if not persons:
#         return "I do not see anyone clearly."

#     parts = []
#     for p in persons:
#         name = p.name if p.name != "Unknown" else "an unknown person"
#         if p.emotion:
#             parts.append(f"{name} who looks {p.emotion}")
#         else:
#             parts.append(name)

#     if len(parts) == 1:
#         return f"I see {parts[0]}."
#     else:
#         joined = ", ".join(parts[:-1]) + " and " + parts[-1]
#         return f"I see {joined}."


# def describe_emotions(persons):
#     """
#     Describe emotions of visible people.
#     """
#     if not persons:
#         return "I cannot see anyone clearly to understand their emotion."

#     parts = []
#     for p in persons:
#         name = p.name if p.name != "Unknown" else "an unknown person"
#         emo = p.emotion or "neutral"
#         parts.append(f"{name} seems {emo}")

#     if len(parts) == 1:
#         return parts[0] + "."
#     else:
#         joined = ", ".join(parts[:-1]) + " and " + parts[-1]
#         return joined + "."


# def analyze_face_events(persons, state: Dict[str, Dict[str, Any]], auto_talk: bool, now: float):
#     """
#     Detect 'approaching' and 'smiling' events.
#     """
#     spoken_phrase = None

#     for p in persons:
#         key = p.name if p.name != "Unknown" else "Unknown"

#         x1, y1, x2, y2 = p.bbox
#         area = max(1, (x2 - x1) * (y2 - y1))

#         info = state.get(key, {})
#         last_area = info.get("last_area")
#         last_emotion = info.get("last_emotion")
#         approach_cooldown_until = info.get("approach_cooldown_until", 0.0)
#         emotion_cooldown_until = info.get("emotion_cooldown_until", 0.0)

#         # approaching
#         if auto_talk and last_area is not None:
#             if area > last_area * 1.6 and now > approach_cooldown_until:
#                 if p.name != "Unknown":
#                     phrase = f"{p.name} is coming closer to you."
#                 else:
#                     phrase = "I see an unknown person coming closer to you."
#                 spoken_phrase = phrase
#                 info["approach_cooldown_until"] = now + 10.0

#         # emotion change
#         if auto_talk and p.emotion and p.emotion != last_emotion and now > emotion_cooldown_until:
#             emo = p.emotion.lower()
#             if emo == "happy":
#                 if p.name != "Unknown":
#                     phrase = f"{p.name} is smiling."
#                 else:
#                     phrase = "I see an unknown person smiling."
#             else:
#                 if p.name != "Unknown":
#                     phrase = f"{p.name} looks {emo}."
#                 else:
#                     phrase = f"I see an unknown person who looks {emo}."

#             spoken_phrase = phrase
#             info["emotion_cooldown_until"] = now + 10.0

#         info["last_area"] = area
#         info["last_emotion"] = p.emotion
#         info["last_seen"] = now
#         state[key] = info

#     return spoken_phrase


# def handle_command(
#     cmd: CommandType,
#     recognized_text: str,
#     frame,
#     vision: VisionSystem,
#     latest_detections,
#     latest_persons,
#     last_spoken: str,
#     auto_talk: bool,
#     listening_enabled: bool,
#     detection_enabled: bool,
# ):
#     """
#     Handle voice command and return:
#     (response_text, auto_talk, listening_enabled, detection_enabled, new_last_spoken)
#     """

#     response = ""

#     if cmd == CommandType.NONE:
#         return "", auto_talk, listening_enabled, detection_enabled, last_spoken

#     # ----- Scene / objects -----
#     if cmd in (CommandType.DESCRIBE_SCENE, CommandType.COUNT_OBJECTS):
#         scene = decide_scene_speech(latest_detections)
#         people = describe_people(latest_persons)
#         response = f"{scene} {people}"

#     # ----- Read text from board/sign -----
#     elif cmd == CommandType.READ_TEXT:
#         text = vision.read_text_from_frame(frame)
#         response = text or "I could not read any clear text."

#     # ----- Who is here? -----
#     elif cmd == CommandType.WHO_IS_HERE:
#         if not latest_persons:
#             response = "I do not see anyone in front of you."
#         else:
#             parts = []
#             for p in latest_persons:
#                 name = p.name if p.name != "Unknown" else "an unknown person"
#                 emo = p.emotion or "neutral"
#                 parts.append(f"{name} who looks {emo}")
#             if len(parts) == 1:
#                 response = f"I see {parts[0]}."
#             else:
#                 joined = ", ".join(parts[:-1]) + " and " + parts[-1]
#                 response = f"I see {joined}."

#     # ----- Repeat last spoken -----
#     elif cmd == CommandType.REPEAT_LAST:
#         if last_spoken:
#             response = last_spoken
#         else:
#             response = "I have not said anything important yet."

#     # ----- Quiet mode -----
#     elif cmd == CommandType.QUIET_MODE:
#         auto_talk = False
#         response = "Okay, I will only speak when you ask me."

#     # ----- Start talking automatically -----
#     elif cmd == CommandType.START_TALKING:
#         auto_talk = True
#         # also wake listening so "start talking" works as wake word
#         listening_enabled = True
#         response = "Got it. I will describe the scene and people for you."

#     # ----- Stop / start listening -----
#     elif cmd == CommandType.STOP_LISTENING:
#         listening_enabled = False
#         response = "Okay, I will stop listening until you say start listening."

#     elif cmd == CommandType.START_LISTENING:
#         listening_enabled = True
#         response = "I am listening again."

#     # ----- Stop / start detection -----
#     elif cmd == CommandType.STOP_DETECTION:
#         detection_enabled = False
#         response = "I have turned off visual detection."

#     elif cmd == CommandType.START_DETECTION:
#         detection_enabled = True
#         response = "I have turned visual detection back on."

#     # ----- Hello / small talk -----
#     elif cmd == CommandType.HELLO:
#         response = "Hello, I am your AI assistant. How can I help you?"

#     elif cmd == CommandType.HOW_ARE_YOU:
#         response = "I am just code, but I am running fine. How are you?"

#     # ----- Fake offline weather -----
#     elif cmd == CommandType.WEATHER:
#         response = "I am offline, but I hope the weather is nice where you are."

#     # ----- Explicit emotion description -----
#     elif cmd == CommandType.DESCRIBE_EMOTION:
#         response = describe_emotions(latest_persons)

#     new_last_spoken = response if response else last_spoken
#     return response, auto_talk, listening_enabled, detection_enabled, new_last_spoken


# # ---------- MAIN LOOP ----------

# def main():
#     # --- Init subsystems ---
#     tts = TextToSpeech(rate=170)
#     stt = SpeechToTextListener(model_path="models/vosk_en")
#     vision = VisionSystem(
#         model_path="yolov8m.pt",
#         conf_threshold=0.55,
#         iou_threshold=0.5,
#         imgsz=640,
#         enable_ocr=True,
#         enable_faces=True,
#         faces_folder="faces_db",
#     )

#     cap = cv2.VideoCapture(CAMERA_INDEX)

#     if not cap.isOpened():
#         print("[look][MAIN] Could not open camera.")
#         return

#     stt.start()

#     tts.speak("Hello, I am your AI vision assistant. I am ready.")

#     auto_talk = True          # automatic scene announcements
#     listening_enabled = True  # respond to voice commands
#     detection_enabled = True  # run YOLO + face recognition

#     last_auto_scene_time = 0.0
#     last_spoken = ""

#     latest_detections = []
#     latest_persons = []

#     face_state: Dict[str, Dict[str, Any]] = {}

#     frame_count = 0

#     try:
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 print("[look][MAIN] Failed to grab frame.")
#                 break

#             now = time.time()
#             frame_count += 1

#             # ----- Vision: objects + people -----
#             if detection_enabled:
#                 if frame_count % 2 == 0:
#                     detections = vision.detect(frame)
#                     persons = vision.recognize_faces(frame)

#                     latest_detections = detections
#                     latest_persons = persons

#                     phrase = analyze_face_events(persons, face_state, auto_talk, now)
#                     if phrase:
#                         tts.speak(phrase)
#                         last_spoken = phrase

#                 vis = frame.copy()
#                 vis = vision.draw_detections(vis, latest_detections)
#                 vis = vision.draw_faces(vis, latest_persons)
#             else:
#                 vis = frame

#             # ----- Automatic scene description every N seconds -----
#             if (
#                 auto_talk
#                 and detection_enabled
#                 and (now - last_auto_scene_time) > AUTO_SCENE_INTERVAL
#             ):
#                 scene = decide_scene_speech(latest_detections)
#                 people = describe_people(latest_persons)
#                 combined = f"{scene} {people}"
#                 tts.speak(combined)
#                 last_spoken = combined
#                 last_auto_scene_time = now

#             # ----- Voice commands (always listen, but gate actions) -----
#             recognized_text = stt.get_latest_command()
#             if recognized_text:
#                 print(f"[look][MAIN] User said: {recognized_text!r}")
#                 cmd = interpret_voice_command(recognized_text)

#                 # 1) If listening is OFF, only allow wake-up commands
#                 if not listening_enabled and cmd not in (
#                     CommandType.START_LISTENING,
#                     CommandType.START_TALKING,
#                 ):
#                     cmd = CommandType.NONE

#                 # 2) If detection is OFF, ignore visual commands
#                 if not detection_enabled and cmd in (
#                     CommandType.DESCRIBE_SCENE,
#                     CommandType.READ_TEXT,
#                     CommandType.WHO_IS_HERE,
#                     CommandType.COUNT_OBJECTS,
#                     CommandType.DESCRIBE_EMOTION,
#                 ):
#                     cmd = CommandType.NONE

#                 (
#                     response,
#                     auto_talk,
#                     listening_enabled,
#                     detection_enabled,
#                     last_spoken,
#                 ) = handle_command(
#                     cmd,
#                     recognized_text,
#                     frame,
#                     vision,
#                     latest_detections,
#                     latest_persons,
#                     last_spoken,
#                     auto_talk,
#                     listening_enabled,
#                     detection_enabled,
#                 )

#                 if response:
#                     print(f"[look][MAIN] Responding: {response!r}")
#                     tts.speak(response)

#             cv2.imshow("Vision Assist", vis)
#             if cv2.waitKey(1) & 0xFF == ord("q"):
#                 break

#     finally:
#         print("[look][MAIN] Shutting down.")
#         stt.stop()
#         cap.release()
#         cv2.destroyAllWindows()


# if __name__ == "__main__":
#     main()



import time
from typing import Dict, Any

import cv2

from .audio import TextToSpeech, SpeechToTextListener
from .vision import VisionSystem, RecognizedPerson
from logic import (
    interpret_voice_command,
    decide_scene_speech,
    CommandType,
)

CAMERA_INDEX = 0
AUTO_SCENE_INTERVAL = 25.0


def describe_people(persons):
    if not persons:
        return "I do not see anyone clearly."

    parts = []
    for p in persons:
        name = p.name if p.name != "Unknown" else "an unknown person"
        if p.emotion:
            parts.append(f"{name} who looks {p.emotion}")
        else:
            parts.append(name)

    if len(parts) == 1:
        return f"I see {parts[0]}."
    else:
        joined = ", ".join(parts[:-1]) + " and " + parts[-1]
        return f"I see {joined}."


def describe_emotions(persons):
    if not persons:
        return "I cannot see anyone clearly to understand their emotion."

    parts = []
    for p in persons:
        name = p.name if p.name != "Unknown" else "an unknown person"
        emo = p.emotion or "neutral"
        parts.append(f"{name} seems {emo}")

    if len(parts) == 1:
        return parts[0] + "."
    else:
        joined = ", ".join(parts[:-1]) + " and " + parts[-1]
        return joined + "."


def analyze_face_events(persons, state: Dict[str, Dict[str, Any]], auto_talk: bool, now: float):
    spoken_phrase = None

    for p in persons:
        key = p.name if p.name != "Unknown" else "Unknown"

        x1, y1, x2, y2 = p.bbox
        area = max(1, (x2 - x1) * (y2 - y1))

        info = state.get(key, {})
        last_area = info.get("last_area")
        last_emotion = info.get("last_emotion")
        approach_cooldown_until = info.get("approach_cooldown_until", 0.0)
        emotion_cooldown_until = info.get("emotion_cooldown_until", 0.0)

        if auto_talk and last_area is not None:
            if area > last_area * 1.6 and now > approach_cooldown_until:
                if p.name != "Unknown":
                    phrase = f"{p.name} is coming closer to you."
                else:
                    phrase = "I see an unknown person coming closer to you."
                spoken_phrase = phrase
                info["approach_cooldown_until"] = now + 10.0

        if auto_talk and p.emotion and p.emotion != last_emotion and now > emotion_cooldown_until:
            emo = p.emotion.lower()
            if emo == "happy":
                if p.name != "Unknown":
                    phrase = f"{p.name} is smiling."
                else:
                    phrase = "I see an unknown person smiling."
            else:
                if p.name != "Unknown":
                    phrase = f"{p.name} looks {emo}."
                else:
                    phrase = f"I see an unknown person who looks {emo}."

            spoken_phrase = phrase
            info["emotion_cooldown_until"] = now + 10.0

        info["last_area"] = area
        info["last_emotion"] = p.emotion
        info["last_seen"] = now
        state[key] = info

    return spoken_phrase


def handle_command(
    cmd: CommandType,
    recognized_text: str,
    frame,
    vision: VisionSystem,
    latest_detections,
    latest_persons,
    last_spoken: str,
    auto_talk: bool,
    listening_enabled: bool,
    detection_enabled: bool,
):
    response = ""

    if cmd == CommandType.NONE:
        return "", auto_talk, listening_enabled, detection_enabled, last_spoken

    if cmd in (CommandType.DESCRIBE_SCENE, CommandType.COUNT_OBJECTS):
        scene = decide_scene_speech(latest_detections)
        people = describe_people(latest_persons)
        response = f"{scene} {people}"

    elif cmd == CommandType.READ_TEXT:
        text = vision.read_text_from_frame(frame)
        response = text or "I could not read any clear text."

    elif cmd == CommandType.WHO_IS_HERE:
        if not latest_persons:
            response = "I do not see anyone in front of you."
        else:
            parts = []
            for p in latest_persons:
                name = p.name if p.name != "Unknown" else "an unknown person"
                emo = p.emotion or "neutral"
                parts.append(f"{name} who looks {emo}")
            if len(parts) == 1:
                response = f"I see {parts[0]}."
            else:
                joined = ", ".join(parts[:-1]) + " and " + parts[-1]
                response = f"I see {joined}."

    elif cmd == CommandType.REPEAT_LAST:
        response = last_spoken or "I have not said anything important yet."

    elif cmd == CommandType.QUIET_MODE:
        auto_talk = False
        response = "Okay, I will only speak when you ask me."

    elif cmd == CommandType.START_TALKING:
        auto_talk = True
        listening_enabled = True  # wake word
        response = "Got it. I will describe the scene and people for you."

    elif cmd == CommandType.STOP_LISTENING:
        listening_enabled = False
        response = "Okay, I will stop listening until you say start listening."

    elif cmd == CommandType.START_LISTENING:
        listening_enabled = True
        response = "I am listening again."

    elif cmd == CommandType.STOP_DETECTION:
        detection_enabled = False
        response = "I have turned off visual detection."

    elif cmd == CommandType.START_DETECTION:
        detection_enabled = True
        response = "I have turned visual detection back on."

    elif cmd == CommandType.HELLO:
        response = "Hello, I am your AI assistant. How can I help you?"

    elif cmd == CommandType.HOW_ARE_YOU:
        response = "I am just code, but I am running fine. How are you?"

    elif cmd == CommandType.WEATHER:
        response = "I am offline, but I hope the weather is nice where you are."

    elif cmd == CommandType.DESCRIBE_EMOTION:
        response = describe_emotions(latest_persons)

    new_last_spoken = response if response else last_spoken
    return response, auto_talk, listening_enabled, detection_enabled, new_last_spoken


def main():
    tts = TextToSpeech(rate=170)
    stt = SpeechToTextListener(model_path="models/vosk_en")
    vision = VisionSystem(
        model_path="yolov8m.pt",
        conf_threshold=0.55,
        iou_threshold=0.5,
        imgsz=640,
        enable_ocr=True,
        enable_faces=True,
        faces_folder="faces_db",
    )

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[look][MAIN] Could not open camera.")
        return

    stt.start()
    tts.speak("Hello, I am your AI vision assistant. I am ready.")

    auto_talk = True
    listening_enabled = True
    detection_enabled = True

    last_auto_scene_time = 0.0
    last_spoken = ""
    latest_detections = []
    latest_persons = []
    face_state: Dict[str, Dict[str, Any]] = {}
    frame_count = 0

    VISION_COMMANDS = {
        CommandType.DESCRIBE_SCENE,
        CommandType.READ_TEXT,
        CommandType.WHO_IS_HERE,
        CommandType.COUNT_OBJECTS,
        CommandType.DESCRIBE_EMOTION,
    }

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[look][MAIN] Failed to grab frame.")
                break

            now = time.time()
            frame_count += 1

            if detection_enabled:
                if frame_count % 2 == 0:
                    detections = vision.detect(frame)
                    persons = vision.recognize_faces(frame)
                    latest_detections = detections
                    latest_persons = persons
                    phrase = analyze_face_events(persons, face_state, auto_talk, now)
                    if phrase:
                        tts.speak(phrase)
                        last_spoken = phrase

                vis = frame.copy()
                vis = vision.draw_detections(vis, latest_detections)
                vis = vision.draw_faces(vis, latest_persons)
            else:
                vis = frame

            if (
                auto_talk
                and detection_enabled
                and (now - last_auto_scene_time) > AUTO_SCENE_INTERVAL
            ):
                scene = decide_scene_speech(latest_detections)
                people = describe_people(latest_persons)
                combined = f"{scene} {people}"
                tts.speak(combined)
                last_spoken = combined
                last_auto_scene_time = now

            recognized_text = stt.get_latest_command()
            if recognized_text:
                print(f"[look][MAIN] User said: {recognized_text!r}")
                cmd = interpret_voice_command(recognized_text)

                if not listening_enabled and cmd not in (
                    CommandType.START_LISTENING,
                    CommandType.START_TALKING,
                ):
                    cmd = CommandType.NONE

                if not detection_enabled and cmd in VISION_COMMANDS:
                    cmd = CommandType.NONE

                (
                    response,
                    auto_talk,
                    listening_enabled,
                    detection_enabled,
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
                )

                if response:
                    print(f"[look][MAIN] Responding: {response!r}")
                    tts.speak(response)

            cv2.imshow("Vision Assist", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        print("[look][MAIN] Shutting down.")
        stt.stop()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
