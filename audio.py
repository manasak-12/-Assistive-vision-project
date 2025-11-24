# app/audio.py

import threading
import queue
import json
from typing import Optional

import sounddevice as sd
from vosk import Model, KaldiRecognizer
import pyttsx3


class TextToSpeech:
    def __init__(self, rate: int = 170):
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", rate)

    def speak(self, text: str):
        if not text:
            return
        self.engine.say(text)
        self.engine.runAndWait()


class SpeechToTextListener:
    """
    Vosk STT using the **Windows default input device**.

    - You choose the mic (like WO Mic) in Windows Sound Settings.
    - This class always records from the default input.
    - Captures audio at 16 kHz, mono, int16 (what Vosk expects).
    """

    def __init__(self, model_path: str = "models/vosk_en"):
        print("[look][STT] Initializing SpeechToTextListener (default input device)...")

        # Get default input device from sounddevice / PortAudio
        default_in_index = sd.default.device[0]
        devices = sd.query_devices()
        dev_info = devices[default_in_index]

        self.device_index = default_in_index
        self.rec_rate = 16000  # Vosk model rate (Hz)

        print(f"[look][STT] Default input device index: {self.device_index}")
        print(f"[look][STT] Device name: {dev_info['name']}")
        print(f"[look][STT] Using recognizer rate: {self.rec_rate} Hz")

        # Load Vosk model
        self.model = Model(model_path)
        self.recognizer = KaldiRecognizer(self.model, self.rec_rate)
        # Optional: recognize word timestamps
        # self.recognizer.SetWords(True)

        self._queue: "queue.Queue[str]" = queue.Queue()
        self._running: bool = False
        self._thread: Optional[threading.Thread] = None

    # ------------- INTERNAL AUDIO CALLBACK -------------

    def _callback(self, indata, frames, time_info, status):
        if status:
            print("[look][STT] status:", status)
        if not self._running:
            # Stop the stream
            raise sd.CallbackStop()

        # RawInputStream gives bytes; pass directly to Vosk
        data_bytes = bytes(indata)

        if self.recognizer.AcceptWaveform(data_bytes):
            # We got a completed utterance
            try:
                result = json.loads(self.recognizer.Result())
                text = result.get("text", "").strip()
            except Exception:
                text = ""
            if text:
                print(f"[look][STT] FULL Heard: {text!r}")
                self._queue.put(text)
        else:
            # Partial result (for debugging, optional)
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
                device=self.device_index,  # default input (set to WO Mic in Windows)
                callback=self._callback,
            ):
                while self._running:
                    sd.sleep(50)
        except Exception as e:
            print("[look][STT] Error in audio stream:", e)

    # ------------- PUBLIC API -------------

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
        Returns the most recent full recognized sentence (if any),
        and discards older ones.
        """
        latest = ""
        try:
            while True:
                latest = self._queue.get_nowait()
        except queue.Empty:
            pass
        return latest
