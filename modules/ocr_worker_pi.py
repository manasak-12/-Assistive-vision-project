# ocr_worker_pi.py
# Lightweight OCR for Raspberry Pi using Tesseract

import cv2
import pytesseract

class OCRWorker:
    def __init__(self):
        print("[OCR] Tesseract OCR ready on Raspberry Pi")

    def process(self, frame):
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Threshold for clarity
        gray = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # Run Tesseract
        text = pytesseract.image_to_string(gray, config="--psm 6")

        text = text.strip()
        if text == "":
            return None

        return text
