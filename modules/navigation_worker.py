# modules/navigation_worker.py
"""
Navigation + Audio manager

- Listens to microphone with Vosk + sounddevice (STT)
- Speaks messages with pyttsx3 (TTS)
- Handles events from Vision via handle_vision_event(event)
- Supports voice commands:
    * "describe scene" / "scene summary" / "what do you see"
    * "read the text"
    * "who is in front of me"
    * "what can you do" / "help"
    * "register name <NAME>" (face registration)
"""

import os
import json
import time
import threading

# ---------- Optional imports (fail gracefully) ----------
try:
    import sounddevice as sd
except Exception as e:  # pragma: no cover
    sd = None
    print("[NAV] sounddevice import failed:", e)

try:
    from vosk import Model, KaldiRecognizer
except Exception as e:  # pragma: no cover
    Model = None
    KaldiRecognizer = None
    print("[NAV] vosk import failed:", e)

try:
    import pyttsx3
except Exception as e:  # pragma: no cover
    pyttsx3 = None
    print("[NAV] pyttsx3 import failed:", e)

from modules.face_recognition_engine import FaceRecognitionEngine
from modules.face_shared import get_latest_embedding

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# project root: one level up from modules/
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))

# Adjust this if your model folder name is different
VOSK_MODEL_DIR = os.path.join(
    ROOT_DIR,
    "models",
    "vosk-model-small-en-in-0.4"
)


def _find_droidcam_input_index():
    """
    Try to find DroidCam microphone index from sounddevice.
    Falls back to None (default input).
    """
    if sd is None:
        return None
    try:
        devices = sd.query_devices()
        for idx, dev in enumerate(devices):
            name = str(dev.get("name", "")).lower()
            max_input = dev.get("max_input_channels", 0)
            if "droidcam" in name and max_input > 0:
                print(f"[NAV] Using DroidCam mic at index {idx}: {dev['name']}")
                return idx
    except Exception as e:
        print("[NAV] Could not query devices:", e)
    return None


class NavigationManager:
    def __init__(self, enable_stt: bool = True, enable_tts: bool = True):
        # Flags
        self.enable_tts = enable_tts and (pyttsx3 is not None)
        self.enable_stt = (
            enable_stt
            and (sd is not None)
            and (Model is not None)
            and (KaldiRecognizer is not None)
        )

        # ---------- TTS ----------
        self._tts = None
        if self.enable_tts:
            try:
                self._tts = pyttsx3.init()
                self._tts.setProperty("rate", 170)
                print("[NAV] TTS (pyttsx3) initialised")
            except Exception as e:
                print("[NAV] TTS init failed:", e)
                self.enable_tts = False

        # ---------- STT ----------
        self._stt_model = None
        self._stt_rec = None
        self._stt_thread = None
        self._stt_running = False
        self._stt_sr = 16000  # sample rate
        self._mic_device = None

        if self.enable_stt:
            if not os.path.isdir(VOSK_MODEL_DIR):
                print(f"[NAV] Vosk model folder not found at {VOSK_MODEL_DIR}")
                self.enable_stt = False
            else:
                try:
                    print(f"[NAV] Loading Vosk model from {VOSK_MODEL_DIR} .")
                    self._stt_model = Model(VOSK_MODEL_DIR)
                    self._stt_rec = KaldiRecognizer(self._stt_model, self._stt_sr)
                    self._stt_rec.SetWords(True)
                    print("[NAV] Vosk STT initialised")
                except Exception as e:
                    print("[NAV] Failed to init Vosk:", e)
                    self.enable_stt = False

        # Choose microphone (prefer DroidCam)
        if self.enable_stt and sd is not None:
            self._mic_device = _find_droidcam_input_index()
            if self._mic_device is None:
                print("[NAV] STT will use default input device.")
            else:
                print(f"[NAV] STT mic device index: {self._mic_device}")

        # ---------- Face recognition for voice registration ----------
        self.face_rec = FaceRecognitionEngine("face_db")

        # ---------- State from Vision ----------
        self.last_scene_summary = ""
        self.last_text = ""
        self.last_people_names = set()
        self.last_people_count = 0
        self.last_obstacle = None  # dict with label + distance

    # ----------------------------------------------------
    # TTS helper
    # ----------------------------------------------------
    def speak(self, text: str):
        print("[NAV][TTS]", text)
        if not self.enable_tts or self._tts is None:
            return
        try:
            self._tts.say(text)
            self._tts.runAndWait()
        except Exception as e:
            print("[NAV] TTS error:", e)

    # ----------------------------------------------------
    # STT loop
    # ----------------------------------------------------
    def _stt_loop(self):
        """Background microphone loop feeding audio into Vosk recogniser."""
        if not self.enable_stt or sd is None or self._stt_rec is None:
            print("[NAV] STT not enabled; _stt_loop will not run.")
            return

        print("[NAV] STT listening. (say 'what can you do' for help)")
        try:
            with sd.RawInputStream(
                samplerate=self._stt_sr,
                blocksize=8000,
                dtype="int16",
                channels=1,
                callback=self._stt_callback,
                device=self._mic_device,  # may be None -> default device
            ):
                while self._stt_running:
                    time.sleep(0.1)
        except Exception as e:
            print("[NAV] STT stream error:", e)
            self._stt_running = False

        print("[NAV] STT loop exited")

    def _stt_callback(self, indata, frames, time_info, status):
        if status:
            print("[NAV] STT status:", status)
        if not self._stt_rec:
            return
        try:
            # IMPORTANT: convert buffer to bytes for Vosk
            if self._stt_rec.AcceptWaveform(bytes(indata)):
                res = self._stt_rec.Result()
                self._handle_stt_result(res)
            else:
                # partial result ignored (could use PartialResult if needed)
                pass
        except Exception as e:
            print("[NAV] STT callback error:", e)

    def _handle_stt_result(self, result_json: str):
        """Process a final STT result from Vosk."""
        try:
            data = json.loads(result_json)
        except Exception:
            print("[NAV] Bad STT JSON:", result_json)
            return

        text = data.get("text", "").strip()
        if not text:
            return
        print("[NAV][STT]", text)
        low = text.lower()

        # ---- Voice command: "register name <NAME>" ----
        if "register" in low and "name" in low:
            try:
                name_part = low.split("name", 1)[1].strip()
                if not name_part:
                    self.speak("Please say the name after the word name.")
                    return
                name = name_part.replace(" ", "_")
            except Exception:
                self.speak("I did not catch the name. Please say: register name, then the name.")
                return

            emb = get_latest_embedding()
            if emb is None:
                self.speak("No face visible to register right now. Please face the camera and try again.")
                return

            self.face_rec.register(name, emb)
            self.speak(f"Registered {name}.")
            return

        # ---- Scene description command ----
        if (
            ("describe" in low and "scene" in low)
            or ("what do you see" in low)
            or ("scene summary" in low)
            or ("summarise scene" in low)
        ):
            if self.last_scene_summary:
                self.speak(f"I see {self.last_scene_summary}.")
            else:
                self.speak("I do not have a scene summary yet.")
            return

        # ---- Text reading command ----
        if ("read" in low and "text" in low) or ("what is written" in low):
            if self.last_text:
                short = self.last_text.replace("\n", " ")[:200]
                self.speak(f"The text says: {short}")
            else:
                self.speak("I do not see any readable text right now.")
            return

        # ---- Who is in front of me ----
        if ("who" in low and ("front" in low or "there" in low)) or ("who is here" in low):
            if self.last_people_names:
                names_list = ", ".join(sorted(self.last_people_names))
                self.speak(f"I see {names_list}.")
            elif self.last_people_count > 0:
                self.speak(f"I see {self.last_people_count} person.")
            else:
                self.speak("I do not see anyone right now.")
            return

        # ---- Help / commands ----
        if "help" in low or "what can you do" in low or "commands" in low:
            self.speak(
                "You can say: describe scene, scene summary, read the text, "
                "who is in front of me, or register name followed by a name."
            )
            return

        # Optional navigation phrases
        if "start" in low and "navigation" in low:
            self.speak("Starting navigation. This feature is not fully implemented.")
        elif "stop" in low:
            self.speak("Stopping navigation.")

    def start_listening(self):
        """Start STT background thread."""
        if not self.enable_stt:
            print("[NAV] STT is disabled; not starting listening.")
            return
        if self._stt_running:
            print("[NAV] STT already running.")
            return
        self._stt_running = True
        self._stt_thread = threading.Thread(target=self._stt_loop, daemon=True)
        self._stt_thread.start()
        print("[NAV] STT thread started")

    def stop_listening(self):
        """Stop STT background thread."""
        self._stt_running = False
        if self._stt_thread is not None and self._stt_thread.is_alive():
            self._stt_thread.join(timeout=2.0)
        print("[NAV] STT thread stopped")

    # ----------------------------------------------------
    # Vision event handler
    # ----------------------------------------------------
    def handle_vision_event(self, event: dict):
        """
        Handle events coming from the Vision side.
        """
        etype = event.get("type")
        payload = event.get("payload", {})

        print("[NAV] Vision event:", etype, payload)

        if etype == "person_detected":
            name = payload.get("name")
            if name:
                self.last_people_names.add(name)
            self.last_people_count += 1   # simple counter

        elif etype == "obstacle_warning":
            self.last_obstacle = payload

        elif etype == "text_detected":
            txt = payload.get("text", "")
            if txt:
                self.last_text = txt

        elif etype == "scene_summary":
            summary = payload.get("summary", "")
            if summary:
                self.last_scene_summary = summary

        elif etype == "emotion":
            # could store last emotion if needed
            pass

    def shutdown(self):
        """Clean up resources."""
        self.stop_listening()
        if self._tts is not None:
            try:
                self._tts.stop()
            except Exception:
                pass


# --------------------------------------------------------
# Module-level singleton + helper used by Vision EventManager
# --------------------------------------------------------
_nav_manager = None


def get_nav_manager() -> NavigationManager:
    global _nav_manager
    if _nav_manager is None:
        _nav_manager = NavigationManager(enable_stt=True, enable_tts=True)
        _nav_manager.start_listening()
    return _nav_manager


def handle_vision_event(event: dict):
    """
    Entry point called from Vision (modules/event_manager.py).
    """
    mgr = get_nav_manager()
    mgr.handle_vision_event(event)


# --------------------------------------------------------
# Standalone test runner
# --------------------------------------------------------
if __name__ == "__main__":
    nav = get_nav_manager()
    nav.speak("Navigation audio test online. Say: what can you do, to hear commands.")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[NAV] Keyboard interrupt; shutting down.")
    nav.shutdown()
