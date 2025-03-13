import cv2
import pytesseract
import subprocess
import os
import time
import json
import wave
import numpy as np
import psutil  # Import psutil to check CPU load

# Configurable CPU load threshold (in percent)
LOAD_THRESHOLD = 70  # Change this value as needed

# Local processing parameters
PIPER_MODEL = "en_US-lessac-medium.onnx"  # Ensure this model exists
OUTPUT_WAV = "tts_output.wav"

# Image filename prefix and AWS bucket name
IMAGE_PREFIX = "captured_image"
S3_BUCKET = "image-my-s3"

def print_sentences(data):
    print("Recognized Text:")
    sentences = {}
    for i in range(len(data["text"])):
        word = data["text"][i].strip()
        if word:
            key = (data["block_num"][i], data["line_num"][i])
            if key not in sentences:
                sentences[key] = {"words": [], "confs": []}
            sentences[key]["words"].append(word)
            try:
                conf_val = float(data["conf"][i])
            except:
                conf_val = -1
            if conf_val != -1:
                sentences[key]["confs"].append(conf_val)

    # Print each sentence with its average confidence
    for key in sorted(sentences.keys()):
        sentence = " ".join(sentences[key]["words"])
        print(f"{sentence}")

def ocr_and_speak_local(image_filename):
    # Load the image from file
    image = cv2.imread(image_filename)
    if image is None:
        print("Error: Could not load image:", image_filename)
        return

    # Convert captured image to RGB format for pytesseract
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Use image_to_data to extract text and confidence values
    data = pytesseract.image_to_data(rgb_image, output_type=pytesseract.Output.DICT)
    recognized_words = [word for word in data["text"] if word.strip() != ""]
    text = " ".join(recognized_words)
    print_sentences(data)

    confidences = [float(conf) for conf in data["conf"] if conf not in ("-1", "", None)]
    if confidences:
        avg_conf = sum(confidences) / len(confidences)
        print("Average OCR Confidence: {:.2f}".format(avg_conf))
    else:
        print("No valid confidence values found.")

    if text:
        try:
            piper_cmd = [
                "piper",
                "--model", PIPER_MODEL,
                "--output_file", OUTPUT_WAV
            ]
            subprocess.run(piper_cmd, input=text, text=True, check=True)
            subprocess.run(["aplay", "-D", "plughw:3,0", OUTPUT_WAV], check=True)
        except subprocess.CalledProcessError as e:
            print("Piper TTS processing failed:", e)
    else:
        print("No text recognized.")

def convert_pcm_to_wav(pcm_file, wav_file, sample_rate=16000, nchannels=1, sampwidth=2):
    try:
        with open(pcm_file, 'rb') as pcmf:
            pcm_data = pcmf.read()
        with wave.open(wav_file, 'wb') as wavf:
            wavf.setnchannels(nchannels)
            wavf.setsampwidth(sampwidth)
            wavf.setframerate(sample_rate)
            wavf.writeframes(pcm_data)
        print(f"Converted {pcm_file} to WAV file {wav_file}")
    except Exception as e:
        print("Failed to convert PCM to WAV:", e)

def process_image_aws(image_filename):
    base_filename = os.path.basename(image_filename)
    
    # 1. Upload image to S3
    upload_cmd = ["aws", "s3", "cp", image_filename, f"s3://{S3_BUCKET}/{base_filename}"]
    try:
        subprocess.run(upload_cmd, check=True)
        print(f"Uploaded image to s3 bucket")
    except subprocess.CalledProcessError as e:
        print("Failed to upload image to S3:", e)
        return

    # 2. Call AWS Textract to extract text
    textract_cmd = [
        "aws", "textract", "detect-document-text",
        "--document", f'{{"S3Object":{{"Bucket":"{S3_BUCKET}","Name":"{base_filename}"}}}}'
    ]
    try:
        textract_output = subprocess.check_output(textract_cmd, text=True)
    except subprocess.CalledProcessError as e:
        print("Textract command failed:", e)
        return

    try:
        textract_data = json.loads(textract_output)
    except Exception as e:
        print("Failed to parse Textract output:", e)
        return

    text_lines = []
    confidences = []
    for block in textract_data.get("Blocks", []):
        if block.get("BlockType") == "LINE":
            text_lines.append(block.get("Text", ""))
            confidences.append(block.get("Confidence", 0))
    extracted_text = "\n".join(text_lines)
    print("Recognized Text:")
    print(extracted_text)
    if confidences:
        avg_conf = sum(confidences) / len(confidences)
        print("Average Textract Confidence: {:.2f}".format(avg_conf))
    else:
        print("No confidence values found from Textract.")

    if not extracted_text:
        print("No text recognized by Textract.")
        return

    # 4. Use AWS Polly to synthesize speech as a PCM file
    pcm_file = "temp_output.pcm"
    polly_cmd = [
        "aws", "polly", "synthesize-speech",
        "--output-format", "pcm",
        "--voice-id", "Joanna",
        "--text", extracted_text,
        pcm_file
    ]
    try:
        subprocess.run(polly_cmd, check=True)
        print("Polly synthesis complete, PCM file generated.")
    except subprocess.CalledProcessError as e:
        print("Polly synthesis failed:", e)
        return

    # 5. Convert PCM file to WAV
    wav_output = "output.wav"
    convert_pcm_to_wav(pcm_file, wav_output)

    # 6. Play the WAV file
    try:
        subprocess.run(["aplay", "-D", "plughw:3,0", wav_output], check=True)
        print("Audio played successfully.")
    except subprocess.CalledProcessError as e:
        print("Failed to play audio:", e)
    
    if os.path.exists(pcm_file):
        os.remove(pcm_file)

if __name__ == "__main__":
    # Open the USB camera (device index may need adjustment)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open the camera.")
        exit(1)

    # Enable autofocus if supported
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    print("Auto focus enabled (if supported by your camera).")

    count = 0
    while True:
        user_input = input("Press Enter to capture an image (or type 'q' to quit): ")
        if user_input.lower() == 'q':
            break

        # Flush the camera buffer to get the latest frame.
        for _ in range(5):
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read from camera during flush.")
                break
            time.sleep(0.05)

        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image from the camera.")
            continue

        image_filename = f"{IMAGE_PREFIX}_{count}.png"
        cv2.imwrite(image_filename, frame)
        print(f"Image saved as {image_filename}")

        # Check current CPU load (this call blocks for 1 second to sample CPU usage)
        cpu_load = psutil.cpu_percent(interval=1)
        print(f"Current CPU load: {cpu_load}%")

        if cpu_load > LOAD_THRESHOLD:
            print(f"CPU load is above {LOAD_THRESHOLD}%, using AWS processing.")
            process_image_aws(image_filename)
        else:
            print(f"CPU load is below {LOAD_THRESHOLD}%, processing locally.")
            ocr_and_speak_local(image_filename)

        count += 1

    cap.release()
