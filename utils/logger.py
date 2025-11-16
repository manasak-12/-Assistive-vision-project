# utils/logger.py
import datetime
import threading
import sys

class Logger:
    USE_COLORS = True

    COLORS = {
        "INFO": "\033[94m",
        "SUCCESS": "\033[92m",
        "WARNING": "\033[93m",
        "ERROR": "\033[91m",
        "ENDC": "\033[0m",
    }

    lock = threading.Lock()

    @staticmethod
    def _timestamp():
        return datetime.datetime.now().strftime("%H:%M:%S")

    @classmethod
    def _log(cls, level, message):
        with cls.lock:
            timestamp = cls._timestamp()

            if cls.USE_COLORS and sys.stdout.isatty():
                color = cls.COLORS.get(level, "")
                endc = cls.COLORS["ENDC"]
                print(f"{color}[{timestamp}] [{level}] {message}{endc}")
            else:
                print(f"[{timestamp}] [{level}] {message}")

    @classmethod
    def info(cls, msg):
        cls._log("INFO", msg)

    @classmethod
    def success(cls, msg):
        cls._log("SUCCESS", msg)

    @classmethod
    def warning(cls, msg):
        cls._log("WARNING", msg)

    @classmethod
    def error(cls, msg):
        cls._log("ERROR", msg)


# global instance
log = Logger
