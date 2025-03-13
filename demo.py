import cv2
import pytesseract
import subprocess
import os
import time
import json
import wave
import numpy as np

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
    # Join all non-empty text entries
    recognized_words = [word for word in data["text"] if word.strip() != ""]
    text = " ".join(recognized_words)
    #print("Recognized Text:", recognized_words)
    print_sentences(data)    
    # Calculate average confidence (ignore '-1' values)
    confidences = [float(conf) for conf in data["conf"] if conf not in ("-1", "", None)]
    if confidences:
        avg_conf = sum(confidences) / len(confidences)
        print("Average OCR Confidence: {:.2f}".format(avg_conf))
    else:
        print("No valid confidence values found.")

    # If text is detected, convert it to speech using Piper CLI
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
    """Converts raw PCM data to a WAV file using the wave module."""
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
    # Extract base filename for S3
    base_filename = os.path.basename(image_filename)
    
    # 1. Upload image to S3
    upload_cmd = ["aws", "s3", "cp", image_filename, f"s3://{S3_BUCKET}/{base_filename}"]
    try:
        subprocess.run(upload_cmd, check=True)
        #print(f"Uploaded {image_filename} to s3://{S3_BUCKET}/{base_filename}")
        print(f"Uploaded image to s3 bucket")
    except subprocess.CalledProcessError as e:
        print("Failed to upload image to S3:", e)
        return

    # 2. Call AWS Textract to extract text from the image in S3
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

    # 3. Extract text lines and compute average confidence from Textract output
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

    # 5. Convert PCM file to WAV using the wave module
    wav_output = "output.wav"
    convert_pcm_to_wav(pcm_file, wav_output)

    # 6. Play the WAV file using aplay
    try:
        subprocess.run(["aplay", "-D", "plughw:3,0", wav_output], check=True)
        print("Audio played successfully.")
    except subprocess.CalledProcessError as e:
        print("Failed to play audio:", e)
    
    # Cleanup temporary PCM file
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

        # Capture the current frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image from the camera.")
            continue

        # Save the captured image with a unique filename
        image_filename = f"{IMAGE_PREFIX}_{count}.png"
        cv2.imwrite(image_filename, frame)
        #print(f"Image saved as {image_filename}")
        print(f"Image saved")

        # Ask for processing mode each time
        mode = input("Select processing mode for this image ('local' or 'aws'): ").strip().lower()
        if mode not in ["local", "aws"]:
            print("Invalid mode selected, defaulting to local.")
            mode = "local"

        if mode == "aws":
            # process_image_aws(image_filename)
            process_image_aws("test_ocr.png")
        else:
            # Call local OCR/TTS with the image filename
            # ocr_and_speak_local(image_filename)
            ocr_and_speak_local("test_ocr.png")
        count += 1

    # Release the camera resource
    cap.release()

