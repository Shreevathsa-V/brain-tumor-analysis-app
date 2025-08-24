import os
import io
import base64
import logging
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from PIL import Image
import numpy as np

# --- App Configuration ---
# Configure logging to file and console
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()])

# --- Model Definition (TensorFlow/Keras U-Net) ---
# This function defines the robust U-Net architecture.
def build_unet_model(input_shape, n_classes):
    inputs = tf.keras.layers.Input(input_shape)

    # Encoder Path
    c1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = tf.keras.layers.Dropout(0.1)(c1)
    c1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

    c2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = tf.keras.layers.Dropout(0.1)(c2)
    c2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
    
    # Bottleneck for Classification Head
    c_class = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    
    # Classifier Head
    gap = tf.keras.layers.GlobalAveragePooling2D()(c_class)
    d1 = tf.keras.layers.Dense(128, activation='relu')(gap)
    d1 = tf.keras.layers.Dropout(0.5)(d1)
    classification_output = tf.keras.layers.Dense(n_classes, activation='softmax', name='classification_output')(d1)

    # Decoder Path (for Segmentation)
    u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c_class)
    u6 = tf.keras.layers.concatenate([u6, c2])
    c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = tf.keras.layers.Dropout(0.1)(c6)
    c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = tf.keras.layers.concatenate([u7, c1])
    c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = tf.keras.layers.Dropout(0.1)(c7)
    c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

    segmentation_output = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid', name='segmentation_output')(c7)

    # The model has two outputs
    model = tf.keras.Model(inputs=[inputs], outputs=[segmentation_output, classification_output])
    
    return model

# --- Globals and Constants ---
app = Flask(__name__)
CORS(app)
MODEL_FILENAME = "brain_tumor_model.h5"
# UPDATED: Added 'No Tumor' and changed N_CLASSES to 4
CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary Tumor']
IMAGE_SIZE = (256, 256)
N_CLASSES = 4

# --- Model Loading ---
try:
    # Build the model structure first
    model = build_unet_model((*IMAGE_SIZE, 3), N_CLASSES)
    # Then load the trained weights
    model.load_weights(MODEL_FILENAME)
    logging.info(f"Model '{MODEL_FILENAME}' loaded successfully.")
    MODEL_LOADED = True
except FileNotFoundError:
    logging.error(f"FATAL: Model file not found at '{MODEL_FILENAME}'. The API cannot function.")
    MODEL_LOADED = False
except Exception as e:
    logging.error(f"FATAL: An error occurred while loading the model: {e}")
    MODEL_LOADED = False

# --- Helper Functions ---
def process_image(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB').resize(IMAGE_SIZE)
        image_array = np.array(image) / 255.0 # Normalize to [0, 1]
        return np.expand_dims(image_array, axis=0), image_bytes
    except Exception as e:
        logging.warning(f"Could not process image bytes: {e}")
        return None, None

def apply_mask_and_encode(original_bytes, mask_array, predicted_class):
    original_pil = Image.open(io.BytesIO(original_bytes)).convert('RGBA').resize(IMAGE_SIZE)
    
    # Only apply mask if a tumor is detected
    if predicted_class != 'No Tumor':
        mask = np.squeeze(mask_array)
        mask = (mask > 0.5).astype('uint8') * 255 # Binarize the mask
        mask_pil = Image.fromarray(mask, mode='L')
        
        red_overlay = Image.new('RGBA', original_pil.size, (255, 0, 0, 0))
        red_overlay.paste((255, 0, 0, 128), mask=mask_pil)
        combined = Image.alpha_composite(original_pil, red_overlay)
    else:
        combined = original_pil

    buffered = io.BytesIO()
    combined.save(buffered, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"

def encode_image(image_bytes):
    return f"data:image/jpeg;base64,{base64.b64encode(image_bytes).decode('utf-8')}"

# --- API Endpoints ---
@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    if not MODEL_LOADED:
        return jsonify({'error': 'Model is not loaded on the server.'}), 503

    files = request.files.getlist('files')
    if not files:
        return jsonify({'error': 'No files provided in the request.'}), 400

    results = []
    for file in files:
        try:
            filename = file.filename
            image_bytes = file.read()
            image_tensor, original_bytes = process_image(image_bytes)

            if image_tensor is None:
                results.append({'filename': filename, 'error': 'Invalid image format'})
                continue

            # Get model prediction (returns a list of outputs)
            segmentation_output, classification_output = model.predict(image_tensor)
            
            # Process classification output
            predicted_idx = np.argmax(classification_output[0])
            confidence_score = float(classification_output[0][predicted_idx])
            predicted_class = CLASS_NAMES[predicted_idx]

            segmented_image_b64 = apply_mask_and_encode(original_bytes, segmentation_output, predicted_class)
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
        "timestamp": datetime.utcnow().isoformat()
    })

# --- Main Execution ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
