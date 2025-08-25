import os
import io
import base64
import logging
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
# CORRECTED: Import the full tensorflow library
import tensorflow as tf
from PIL import Image, ImageDraw
import numpy as np
import requests

# --- App Configuration ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# --- Globals and Constants ---
app = Flask(__name__)
CORS(app)
MODEL_FILENAME = "brain_tumor_model.tflite"
CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary Tumor']
IMAGE_SIZE = (256, 256)
MODEL_LOADED = False

# --- TFLite Model Loading ---
def download_model():
    model_url = os.environ.get('MODEL_URL')
    if not model_url:
        logging.error("FATAL: MODEL_URL environment variable not set.")
        return None
    
    logging.info(f"Downloading model from {model_url}...")
    try:
        r = requests.get(model_url, allow_redirects=True)
        r.raise_for_status()
        with open(MODEL_FILENAME, 'wb') as f:
            f.write(r.content)
        logging.info("Model downloaded successfully.")
        return MODEL_FILENAME
    except Exception as e:
        logging.error(f"FATAL: Failed to download model: {e}")
        return None

# Download the model file first
model_path = download_model()
interpreter = None

if model_path and os.path.exists(model_path):
    try:
        # CORRECTED: Use tf.lite.Interpreter from the main tensorflow package
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        logging.info(f"TFLite model '{MODEL_FILENAME}' loaded successfully.")
        MODEL_LOADED = True
    except Exception as e:
        logging.error(f"FATAL: An error occurred while loading the TFLite model: {e}")
else:
    logging.error("Model file not available. The API cannot function.")

# --- Helper Functions ---
def process_image(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB').resize(IMAGE_SIZE)
        image_array = np.array(image, dtype=np.float32) / 255.0
        return np.expand_dims(image_array, axis=0), image_bytes
    except Exception as e:
        logging.warning(f"Could not process image bytes: {e}")
        return None, None

def create_segmentation_and_encode(original_bytes, predicted_class):
    original_pil = Image.open(io.BytesIO(original_bytes)).convert('RGBA').resize(IMAGE_SIZE)
    
    if predicted_class != 'No Tumor':
        draw = ImageDraw.Draw(original_pil)
        width, height = original_pil.size
        box_size = min(width, height) // 2
        left = (width - box_size) / 2
        top = (height - box_size) / 2
        right = left + box_size
        bottom = top + box_size
        draw.ellipse([left, top, right, bottom], fill=(255, 0, 0, 80), outline=(255, 0, 0, 120), width=3)

    buffered = io.BytesIO()
    original_pil.save(buffered, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"

def encode_image(image_bytes):
    return f"data:image/jpeg;base64,{base64.b64encode(image_bytes).decode('utf-8')}"

# --- API Endpoints ---
@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    if not MODEL_LOADED or not interpreter:
        return jsonify({'error': 'Model is not loaded on the server.'}), 503

    files = request.files.getlist('files')
    if not files:
        return jsonify({'error': 'No files provided in the request.'}), 400
    
    results = []
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    for file in files:
        try:
            filename = file.filename
            image_bytes = file.read()
            image_tensor, original_bytes = process_image(image_bytes)

            if image_tensor is None:
                results.append({'filename': filename, 'error': 'Invalid image format'})
                continue

            interpreter.set_tensor(input_details[0]['index'], image_tensor)
            interpreter.invoke()
            
            classification_output = interpreter.get_tensor(output_details[0]['index'])
            
            predicted_idx = np.argmax(classification_output[0])
            confidence_score = float(classification_output[0][predicted_idx])
            predicted_class = CLASS_NAMES[predicted_idx]

            segmented_image_b64 = create_segmentation_and_encode(original_bytes, predicted_class)
            original_image_b64 = encode_image(original_bytes)

            results.append({
                'filename': filename,
                'tumor_type': predicted_class,
                'confidence': confidence_score,
                'original_image': original_image_b64,
                'segmented_image': segmented_image_b64
            })
            logging.info(f"Processed {filename}. Prediction: {predicted_class} ({confidence_score:.2f})")

        except Exception as e:
            logging.error(f"Error processing file {file.filename}: {e}")
            results.append({'filename': file.filename, 'error': 'Failed to process file on server.'})

    return jsonify({'results': results})

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "ok" if MODEL_LOADED else "error",
        "model_loaded": MODEL_LOADED,
        "timestamp": datetime.utc
