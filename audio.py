import threading
import queue
import json
from typing import Optional
import subprocess

import sounddevice as sd
from vosk import Model, KaldiRecognizer

import logbuffer  # for live console logs


# ================== TEXT TO SPEECH (QUEUE + POWERSHELL) ==================

class TextToSpeech:
    """
    Text-to-speech using Windows PowerShell + System.Speech.
    Uses a background worker thread + queue so audio never overlaps.
    """

    def __init__(self, rate: int = 0):
        # clamp to System.Speech rate range [-10, 10]
        if rate < -10 or rate > 10:
            self.rate = 0
        else:
            self.rate = rate

        self._queue: "queue.Queue[str | None]" = queue.Queue()
        self._running = True

        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _speak_once(self, text: str):
        """Blocking speak of a single text using PowerShell."""
        if not text:
            return

        safe_text = text.replace("'", "''")
        ps_command = (
            "Add-Type -AssemblyName System.Speech; "
            "$speak = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$speak.Rate = {self.rate}; "
            "$speak.Volume = 100; "
            f"$speak.Speak('{safe_text}');"
        )
        try:
            # run synchronously, so this one finishes before next starts
            subprocess.run(
                ["powershell", "-Command", ps_command],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            msg = f"[look][TTS] Error while speaking: {e}"
            print(msg)
            logbuffer.add(msg)

    def _worker(self):
        """Background worker that consumes the queue and speaks."""
        while self._running:
            try:
                text = self._queue.get()
                if text is None:
                    break  # sentinel to stop
                self._speak_once(text)
            except Exception as e:
                msg = f"[look][TTS] Error in TTS worker: {e}"
                print(msg)
                logbuffer.add(msg)

    def speak(self, text: str):
        """Public method: enqueue text to be spoken."""
        if not text:
            return
        self._queue.put(text)

    def stop(self):
        """Stop the TTS worker gracefully."""
        self._running = False
        self._queue.put(None)
        try:
            self._thread.join(timeout=1.0)
        except Exception:
            pass


# ================== SPEECH TO TEXT (VOSK) ==================

class SpeechToTextListener:
    """
    Vosk STT using the **Windows default input device**.
    Logs all [look][STT] lines to logbuffer for web console.
    """

    def __init__(self, model_path: str = "models/vosk_en"):
        msg = "[look][STT] Initializing SpeechToTextListener (default input device)..."
        print(msg)
        logbuffer.add(msg)

        default_in_index = sd.default.device[0]
        devices = sd.query_devices()
        dev_info = devices[default_in_index]

        self.device_index = default_in_index
        self.rec_rate = 16000  # Vosk model rate

        msg = (
            f"[look][STT] Default input device index: {self.device_index}\n"
            f"[look][STT] Device name: {dev_info['name']}\n"
            f"[look][STT] Using recognizer rate: {self.rec_rate} Hz"
        )
        print(msg)
        for line in msg.split("\n"):
            logbuffer.add(line)

        self.model = Model(model_path)
        self.recognizer = KaldiRecognizer(self.model, self.rec_rate)

        self._queue: "queue.Queue[str]" = queue.Queue()
        self._running: bool = False
        self._thread: Optional[threading.Thread] = None

    def _callback(self, indata, frames, time_info, status):
        if status:
            line = f"[look][STT] status: {status}"
            print(line)
            logbuffer.add(line)
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
                line = f"[look][STT] FULL Heard: {text!r}"
                print(line)
                logbuffer.add(line)
                self._queue.put(text)
        else:
            try:
                partial = json.loads(self.recognizer.PartialResult()).get("partial", "").strip()
            except Exception:
                partial = ""
            if partial:
                line = f"[look][STT] partial: {partial!r}"
                print(line)
                logbuffer.add(line)

    def _run(self):
        try:
            msg = (
                f"[look][STT] Opening RawInputStream on default device "
                f"(index {self.device_index}) at {self.rec_rate} Hz."
            )
            print(msg)
            logbuffer.add(msg)

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
            msg = f"[look][STT] Error in audio stream: {e}"
            print(msg)
            logbuffer.add(msg)

    def start(self):
        if self._running:
            return
        msg = "[look][STT] Starting microphone listener."
        print(msg)
        logbuffer.add(msg)
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        msg = "[look][STT] Stopping microphone listener."
        print(msg)
        logbuffer.add(msg)
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
