FROM dustynv/piper-tts:r36.2.0 

# 1) Install system packages & build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    # For OCR
    tesseract-ocr \
    libtesseract-dev \
    # For audio playback/recording
    ffmpeg \
    alsa-utils \
    # (Optional) More dev packages if needed, e.g. libssl-dev, gfortran, etc.
 && rm -rf /var/lib/apt/lists/*


# 3) Install Python libraries
#    - opencv-python-headless for image processing
#    - pytesseract for Tesseract OCR integration
#    - pyttsx3 for a basic offline TTS fallback (optional)
RUN pip install --no-cache-dir \
    opencv-python-headless \
    pytesseract \
    pyttsx3 \
 && rm -rf /var/lib/apt/lists/*
# 4) Create a working directory
WORKDIR /app

# 5) Copy your Python script(s) and images
COPY app.py /app/
COPY demo.py /app/
COPY test_ocr.png /app/
COPY en_US-lessac-medium.onnx* /app/
COPY increase_processor_load.py /app/


RUN pip install awscli

# 6) Default command to run the Python script
# CMD ["python3", "increase_processor_load.py", "4", "30"]
# CMD ["python3", "app.py"]
CMD ["sh", "-c", "python3 increase_processor_load.py 3 30 & python3 app.py"]


