# ocr_worker_windows.py
# Windows OCR using EasyOCR

import easyocr
import cv2

class OCRWorker:
    def __init__(self, lang_list=["en"]):
        print("[OCR] Loading EasyOCR model for Windows...")
        self.reader = easyocr.Reader(lang_list)  # loads model into RAM

    def process(self, frame):
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Run OCR
        results = self.reader.readtext(gray)

        if not results:
            return None

        # Combine text pieces
        final_text = " ".join([res[1] for res in results])
        return final_text.strip()
