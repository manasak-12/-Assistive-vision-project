# modules/event_manager.py
"""
Vision Event Manager (Laptop A)

- Handles TTS using pyttsx3 (background thread -> less lag)
- Sends events to navigation_worker.handle_vision_event (if available)
- By default:
    * Speaks ONLY person/obstacle warnings
    * Text, scene summary, emotion are quiet unless requested by voice commands
"""

import time
import threading
import queue

# ---------- TTS ----------
try:
    import pyttsx3
except Exception as e:
    pyttsx3 = None
    print("[Vision] pyttsx3 import failed:", e)

# ---------- Navigation bridge ----------
try:
    # ASSUMPTION: navigation_worker.py is directly under modules/
    # D:\vision-assist\modules\navigation_worker.py
    from modules.navigation_worker import handle_vision_event
    NAV_AVAILABLE = True
    print("[Vision] Navigation handler loaded from modules.navigation_worker.")
except Exception as e:
    print("[Vision] Navigation import failed:", e)
    NAV_AVAILABLE = False


class EventManager:
    def __init__(self):
        # cooldowns per event key
        self._last_fired = {}
        self._cooldown_s = 1.0  # seconds

        # TTS settings: control how talkative Vision is
        self.announce_people = True           # say "<name> ahead" or "Person ahead"
        self.announce_obstacles = True        # say "Warning: obstacle..."
        self.announce_text_auto = False       # text is quiet by default
        self.announce_scene_auto = False      # scene summary quiet by default
        self.announce_emotion_auto = False    # emotion quiet by default

        # TTS engine + queue + thread
        self._tts = None
        self._tts_queue = queue.Queue()
        self._tts_thread = None

        if pyttsx3 is not None:
            try:
                self._tts = pyttsx3.init()
                self._tts.setProperty("rate", 175)
                print("[Vision] TTS initialised.")

                self._tts_thread = threading.Thread(
                    target=self._tts_loop, daemon=True
                )
                self._tts_thread.start()
            except Exception as e:
                print("[Vision] TTS init failed:", e)
                self._tts = None

    # ---------------- Cooldown ----------------
    def _can_fire(self, key: str) -> bool:
        now = time.time()
        last = self._last_fired.get(key, 0)
        if now - last >= self._cooldown_s:
            self._last_fired[key] = now
            return True
        return False

    # ---------------- TTS non-blocking ----------------
    def _tts_loop(self):
        """Background TTS loop so main thread doesn't freeze."""
        if not self._tts:
            return
        while True:
            text = self._tts_queue.get()
            if text is None:
                break
            try:
                self._tts.say(text)
                self._tts.runAndWait()
            except Exception as e:
                print("[Vision] TTS error:", e)

    def speak(self, text: str):
        print("[TTS]", text)
        if self._tts:
            try:
                self._tts_queue.put_nowait(text)
            except Exception:
                pass

    # ---------------- Build event + dispatch ----------------
    def _build_event(self, etype: str, payload: dict) -> dict:
        return {
            "type": etype,
            "timestamp": time.time(),
            "payload": payload,
        }

    def _dispatch(self, event: dict):
        """Send event to Navigation if available."""
        if NAV_AVAILABLE:
            try:
                handle_vision_event(event)
            except Exception as e:
                print("[Vision] Navigation event error:", e)

    # ---------------- Public API used from test_vision.py ----------------
    def person_detected(self, bbox=None, conf=None, distance_m=None, name=None):
        """
        Called when a person is detected.
        If name is provided, speak '<name> ahead' instead of generic 'Person ahead'.
        """
        if self._can_fire("person"):
            phrase = None
            if self.announce_people:
                if name:
                    phrase = f"{name} ahead"
                else:
                    phrase = "Person ahead"

            ev = self._build_event(
                "person_detected",
                {
                    "bbox": bbox,
                    "conf": float(conf) if conf is not None else None,
                    "distance_m": float(distance_m) if distance_m is not None else None,
                    "name": name,
                },
            )
            if phrase:
                self.speak(phrase)
            self._dispatch(ev)

    def vehicle_detected(self, bbox=None, conf=None, distance_m=None):
        if self._can_fire("vehicle"):
            phrase = None
            if self.announce_obstacles:
                phrase = "Vehicle approaching"

            ev = self._build_event(
                "vehicle_detected",
                {
                    "bbox": bbox,
                    "conf": float(conf) if conf is not None else None,
                    "distance_m": float(distance_m) if distance_m is not None else None,
                },
            )
            if phrase:
                self.speak(phrase)
            self._dispatch(ev)

    def emotion_event(self, emotion: str):
        key = f"emotion_{emotion}"
        if self._can_fire(key):
            ev = self._build_event("emotion", {"emotion": emotion})
            # emotion is NOT spoken automatically now
            if self.announce_emotion_auto:
                self.speak(f"Detected emotion {emotion}")
            self._dispatch(ev)

    def log_text(self, text: str):
        text = (text or "").strip()
        if not text:
            return
        ev = self._build_event("text_detected", {"text": text})
        short = text.replace("\n", " ")[:80]
        # text is quiet by default; only spoken if enabled OR via voice command
        if self._can_fire("text") and self.announce_text_auto:
            self.speak(f"Text detected: {short}")
        self._dispatch(ev)

    def obstacle_warning(self, label: str, distance_m: float):
        if self._can_fire("obstacle_warning"):
            phrase = None
            if self.announce_obstacles:
                phrase = f"Warning: {label} at {distance_m:.1f} meters"

            ev = self._build_event(
                "obstacle_warning",
                {"label": label, "distance_m": float(distance_m)},
            )
            if phrase:
                self.speak(phrase)
            self._dispatch(ev)

    def face_detected(self):
        if self._can_fire("face_detected"):
            ev = self._build_event("face_detected", {})
            self._dispatch(ev)

    def scene_summary(self, summary_text: str):
        summary_text = (summary_text or "").strip()
        if self._can_fire("scene_summary") and summary_text:
            ev = self._build_event("scene_summary", {"summary": summary_text})
            # scene summary is NOT spoken automatically now
            if self.announce_scene_auto:
                self.speak(f"Scene summary: {summary_text}")
            self._dispatch(ev)
