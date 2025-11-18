# modules/speech_worker.py

import queue
import threading
import logging
import time
import json
import os
import platform
from pathlib import Path

import sounddevice as sd
from vosk import Model, KaldiRecognizer

IS_WINDOWS = platform.system() == "Windows"

# ---------- CONFIG ----------

# Name of the vosk model folder inside /models
MODEL_FOLDER_NAME = "vosk-model-small-en-in-0.4"  # adjust if different

# Base dir = project root (vision-assist)
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models" / MODEL_FOLDER_NAME
VOSK_MODEL_PATH = str(MODEL_DIR)

# Mic device:
# - On Windows: use DroidCam (index 6 from your sd.query_devices())
# - On Pi / others: use default device (None)
if IS_WINDOWS:
    DEFAULT_MIC_DEVICE_INDEX = 6  # DroidCam DirectSound on your laptop
else:
    DEFAULT_MIC_DEVICE_INDEX = None  # let sounddevice choose default input

SAMPLE_RATE = 16000
BLOCK_SIZE = 8000


class SpeechWorker(threading.Thread):
    """
    Listens to microphone audio using sounddevice,
    runs Vosk STT, and publishes recognized text to the event bus.
    """

    def __init__(self, event_bus=None,
                 model_path: str = VOSK_MODEL_PATH,
                 mic_device_index: int | None = DEFAULT_MIC_DEVICE_INDEX):
        super().__init__(daemon=True)

        self.event_bus = event_bus
        self.model_path = model_path
        self.mic_device_index = mic_device_index

        self._running = threading.Event()
        self._running.set()

        self._queue: "queue.Queue[bytes]" = queue.Queue()
        self._recognizer = None

    # ---------- Audio setup ----------

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            logging.warning("Sounddevice status: %s", status)
        self._queue.put(bytes(indata))

    def _init_audio_defaults(self):
        """Configure sounddevice. On Pi: default mic; On Windows: DroidCam."""
        if self.mic_device_index is not None:
            sd.default.device = (self.mic_device_index, None)  # (input, output)
        sd.default.samplerate = SAMPLE_RATE
        sd.default.channels = 1

        logging.info("Using mic device index: %s", self.mic_device_index)
        logging.info("sounddevice default devices: %s", sd.default.device)

    # ---------- Thread main loop ----------

    def run(self):
        logging.info("Starting SpeechWorker...")
        self._init_audio_defaults()

        logging.info("Loading Vosk model from: %s", self.model_path)
        logging.info("Absolute path: %s", os.path.abspath(self.model_path))
        logging.info("Model dir exists: %s", os.path.isdir(self.model_path))
        if os.path.isdir(self.model_path):
            try:
                logging.info("Model dir contents: %s", os.listdir(self.model_path))
            except Exception as e:
                logging.warning("Could not list model dir contents: %s", e)

        try:
            model = Model(self.model_path)
        except Exception as e:
            logging.exception("Vosk failed to create model: %s", e)
            return

        self._recognizer = KaldiRecognizer(model, SAMPLE_RATE)

        # On Pi: don't pass device explicitly (use default)
        stream_kwargs = dict(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            dtype="int16",
            channels=1,
            callback=self._audio_callback,
        )
        if self.mic_device_index is not None:
            stream_kwargs["device"] = self.mic_device_index

        with sd.RawInputStream(**stream_kwargs):
            logging.info("🎤 SpeechWorker is now listening...")

            while self._running.is_set():
                try:
                    data = self._queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                if self._recognizer.AcceptWaveform(data):
                    result = json.loads(self._recognizer.Result())
                    text = result.get("text", "").strip()
                    if text:
                        logging.info("Recognized speech: %s", text)
                        self._publish_event(text)
                else:
                    _ = self._recognizer.PartialResult()

        logging.info("SpeechWorker stopped.")

    def stop(self):
        self._running.clear()

    # ---------- Event bus integration ----------

    def _publish_event(self, text: str):
        if not self.event_bus:
            print(f"[SPEECH] {text}")
            return

        event = {
            "type": "speech.recognized",
            "text": text,
            "timestamp": time.time(),
        }
        try:
            self.event_bus.publish(event)
        except Exception as e:
            logging.exception("Failed to publish speech event: %s", e)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    try:
        from modules.event_bus import EventBus
        bus = EventBus()

        def print_listener(event):
            print("EVENT:", event)

        bus.subscribe("speech.recognized", print_listener)
        event_bus = bus
    except Exception:
        logging.warning("Could not import EventBus; using print-only mode.")
        event_bus = None

    worker = SpeechWorker(event_bus=event_bus)
    worker.start()

    print("🎤 Speak into the mic. Ctrl+C to stop.")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Stopping SpeechWorker...")
        worker.stop()
        time.sleep(0.5)


if __name__ == "__main__":
    main()
