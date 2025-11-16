# import vosk
# import queue
# import json
# import sounddevice as sd
# import pyttsx3
# import threading

# class SpeechWorker:
#     def __init__(self, vosk_model_path="models/vosk_small_model"):
#         print("[INFO] Loading VOSK speech model...")
#         self.model = vosk.Model(vosk_model_path)
#         self.recognizer = vosk.KaldiRecognizer(self.model, 16000)

#         self.q = queue.Queue()

#         print("[INFO] Initializing TTS engine...")
#         self.tts = pyttsx3.init()
#         self.tts.setProperty("rate", 165)
#         self.tts.setProperty("volume", 1.0)

#         self.listening = False
#         self.listener_thread = None

#     # ------------------
#     # TEXT TO SPEECH
#     # ------------------
#     # def speak(self, text):
#     #     print("[TTS] >>", text)
#     #     self.tts.say(text)
#     #     self.tts.runAndWait()
#     def speak(self, text):
#         try:
#             self.tts.stop()   # <---- IMPORTANT FIX
#             self.tts.say(text)
#             self.tts.runAndWait()
#         except RuntimeError:
#             pass   # prevents crash if engine is already speaking

#     # ------------------
#     # MICROPHONE CALLBACK
#     # ------------------
#     def audio_callback(self, indata, frames, time, status):
#         self.q.put(bytes(indata))

#     # ------------------
#     # START LISTENING
#     # ------------------
#     def start_listening(self):
#         if self.listening:
#             return

#         self.listening = True

#         self.listener_thread = threading.Thread(target=self.listen_loop)
#         self.listener_thread.daemon = True
#         self.listener_thread.start()

#         sd.RawInputStream(
#             samplerate=16000,
#             blocksize=8000,
#             dtype="int16",
#             channels=1,
#             callback=self.audio_callback
#         ).start()

#         print("[STT] Microphone listening...")

#     # ------------------
#     # LISTEN LOOP
#     # ------------------
#     def listen_loop(self):
#         while self.listening:
#             data = self.q.get()
#             if self.recognizer.AcceptWaveform(data):
#                 result = json.loads(self.recognizer.Result())
#                 text = result.get("text", "").strip()

#                 if text:
#                     print("[STT] Heard:", text)
#                     self.process_command(text)

#     # ------------------
#     # VOICE COMMANDS
#     # ------------------
#     def process_command(self, text):
#         text = text.lower()

#         if "start" in text:
#             self.speak("Starting navigation.")

#         elif "repeat" in text:
#             self.speak("Repeating last instruction.")

#         elif "stop" in text:
#             self.speak("Stopping navigation.")
#             self.listening = False

#         else:
#             self.speak("I did not understand.")

import vosk
import queue
import json
import sounddevice as sd
import pyttsx3
import threading

class SpeechWorker:
    def __init__(self, vosk_model_path="models/vosk_small_model", event_bus=None):
        self.event_bus = event_bus

        print("[INFO] Loading VOSK speech model...")
        self.model = vosk.Model(vosk_model_path)
        self.recognizer = vosk.KaldiRecognizer(self.model, 16000)
        self.q = queue.Queue()

        print("[INFO] Initializing TTS engine...")
        self.tts = pyttsx3.init()
        self.tts.setProperty("rate", 165)
        self.tts.setProperty("volume", 1.0)

        self.listening = False
        self.listener_thread = None

    # TTS
    def speak(self, text):
        try:
            self.tts.stop()
            self.tts.say(text)
            self.tts.runAndWait()
        except RuntimeError:
            pass

    # MIC callback
    def audio_callback(self, indata, frames, time, status):
        self.q.put(bytes(indata))

    # START LISTENING
    def start_listening(self):
        if self.listening:
            return

        self.listening = True

        threading.Thread(target=self.listen_loop, daemon=True).start()

        sd.RawInputStream(
            samplerate=16000,
            blocksize=8000,
            dtype="int16",
            channels=1,
            callback=self.audio_callback
        ).start()

        print("[STT] Microphone listening...")

    # STT LOOP
    def listen_loop(self):
        while self.listening:
            data = self.q.get()
            if self.recognizer.AcceptWaveform(data):
                result = json.loads(self.recognizer.Result())
                text = result.get("text", "").strip()
                if text:
                    print("[STT] Heard:", text)
                    self.process_command(text)

    # PROCESS COMMAND
    def process_command(self, text):
        text = text.lower()

        # Publish voice command to EventBus
        if self.event_bus:
            self.event_bus.publish({
                "type": "voice_command",
                "data": {"text": text}
            })

        if "stop" in text:
            self.speak("Stopping navigation.")
            self.listening = False
        elif "start" in text:
            self.speak("Starting navigation.")
        elif "repeat" in text:
            self.speak("Repeating last instruction.")
        else:
            self.speak("Command received.")

