import threading
import queue
import json
from typing import Optional
import subprocess

import sounddevice as sd
from vosk import Model, KaldiRecognizer


# ================== TEXT TO SPEECH (WINDOWS POWERSHELL) ==================

class TextToSpeech:
    """
    Text-to-speech using Windows PowerShell + System.Speech.

    - Works offline.
    - Each speak() call spawns a small PowerShell process that says the text.
    - No issues with threads or "run loop already started".
    """

    def __init__(self, rate: int = 0):
        """
        rate: we map this to System.Speech Rate (-10 to +10).
        If something huge is passed (like 170), we just clamp to 0.
        """
        if rate < -10 or rate > 10:
            self.rate = 0
        else:
            self.rate = rate

    def speak(self, text: str):
        if not text:
            return

        # Escape single quotes for PowerShell single-quoted string
        safe_text = text.replace("'", "''")

        ps_command = (
            "Add-Type -AssemblyName System.Speech; "
            "$speak = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$speak.Rate = {self.rate}; "
            "$speak.Volume = 100; "
            f"$speak.Speak('{safe_text}');"
        )

        try:
            # Launch PowerShell TTS in a separate process (non-blocking for Python)
            subprocess.Popen(
                ["powershell", "-Command", ps_command],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            print("[look][TTS] Error while speaking:", e)


# ================== SPEECH TO TEXT (VOSK) ==================

class SpeechToTextListener:
    """
    Vosk STT using the **Windows default input device**.

    - Choose mic (WO Mic, etc.) in Windows Sound Settings as default.
    - Records audio at 16 kHz mono int16 (what Vosk expects).
    """

    def __init__(self, model_path: str = "models/vosk_en"):
        print("[look][STT] Initializing SpeechToTextListener (default input device)...")

        # default input device index from sounddevice
        default_in_index = sd.default.device[0]
        devices = sd.query_devices()
        dev_info = devices[default_in_index]

        self.device_index = default_in_index
        self.rec_rate = 16000  # Vosk model rate

        print(f"[look][STT] Default input device index: {self.device_index}")
        print(f"[look][STT] Device name: {dev_info['name']}")
        print(f"[look][STT] Using recognizer rate: {self.rec_rate} Hz")

        # Load Vosk model
        self.model = Model(model_path)
        self.recognizer = KaldiRecognizer(self.model, self.rec_rate)

        self._queue: "queue.Queue[str]" = queue.Queue()
        self._running: bool = False
        self._thread: Optional[threading.Thread] = None

    # -------- INTERNAL CALLBACK --------

    def _callback(self, indata, frames, time_info, status):
        if status:
            print("[look][STT] status:", status)
        if not self._running:
            raise sd.CallbackStop()

        data_bytes = bytes(indata)

        if self.recognizer.AcceptWaveform(data_bytes):
            try:
                result = json.loads(self.recognizer.Result())
                text = result.get("text", "").strip()
            except Exception:
                text = ""
            if text:
                print(f"[look][STT] FULL Heard: {text!r}")
                self._queue.put(text)
        else:
            try:
                partial = json.loads(self.recognizer.PartialResult()).get("partial", "").strip()
            except Exception:
                partial = ""
            if partial:
                print(f"[look][STT] partial: {partial!r}")

    def _run(self):
        try:
            print(
                f"[look][STT] Opening RawInputStream on default device "
                f"(index {self.device_index}) at {self.rec_rate} Hz."
            )
            with sd.RawInputStream(
                samplerate=self.rec_rate,
                blocksize=8000,
                dtype="int16",
                channels=1,
                device=self.device_index,
                callback=self._callback,
            ):
                while self._running:
                    sd.sleep(50)
        except Exception as e:
            print("[look][STT] Error in audio stream:", e)

    # -------- PUBLIC API --------

    def start(self):
        if self._running:
            return
        print("[look][STT] Starting microphone listener.")
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        print("[look][STT] Stopping microphone listener.")
        self._running = False

    def get_latest_command(self) -> str:
        """
        Returns the most recent recognized full sentence (if any),
        discarding older ones.
        """
        latest = ""
        try:
            while True:
                latest = self._queue.get_nowait()
        except queue.Empty:
            pass
        return latest
