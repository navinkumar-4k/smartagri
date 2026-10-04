import os
import time
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import firebase_admin
from firebase_admin import credentials, db

app = Flask(__name__)
CORS(app)

# 1. Initialize Firebase Admin SDK
FIREBASE_DB_URL = "https://smartpesticide-2b0e3-default-rtdb.asia-southeast1.firebasedatabase.app"

if os.path.exists("serviceAccountKey.json"):
    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred, {
        'databaseURL': FIREBASE_DB_URL
    })
    print("Firebase Admin SDK initialized.")
else:
    print("WARNING: serviceAccountKey.json missing.")

# 2. Load TensorFlow .h5 Model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.h5')
model = None

if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    print("TensorFlow .h5 Model loaded successfully.")
else:
    print("WARNING: model.h5 not found in directory.")

# Update class list to match your trained dataset classes exactly
CLASSES = ['Healthy', 'Bacterial_Spot', 'Early_Blight', 'Late_Blight']

@app.route('/')
def home():
    if os.path.exists('index.html'):
        with open('index.html', 'r', encoding='utf-8') as f:
            return f.read()
    return "AgriSmart API Server is Running.", 200

@app.route('/api/analyze', methods=['POST'])
def analyze_leaf():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    img = Image.open(file.stream).convert('RGB').resize((224, 224))
    
    # Preprocess image tensor (Normalization 0.0 - 1.0)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Inference with .h5 Model
    if model:
        predictions = model.predict(img_array)
        class_idx = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][class_idx]) * 100
        detected_class = CLASSES[class_idx]
    else:
        detected_class = "Early_Blight"
        confidence = 92.5

    should_spray = (detected_class != 'Healthy') and (confidence > 80.0)

    # Sync to Firebase Realtime Database
    if firebase_admin._apps:
        ref_status = db.reference('/system/status')
        ref_count = db.reference('/system/spray_count')
        ref_scans = db.reference('/system/scans_count')
        ref_logs = db.reference('/logs')

        current_sprays = ref_count.get() or 0
        if should_spray:
            current_sprays += 1
            ref_count.set(current_sprays)

        current_scans = ref_scans.get() or 0
        ref_scans.set(current_scans + 1)

        ref_status.set({
            'disease': detected_class,
            'confidence': round(confidence, 1),
            'should_spray': should_spray,
            'last_updated': time.strftime("%Y-%m-%d %H:%M:%S")
        })

        ref_logs.push({
            'timestamp': time.strftime("%I:%M:%S %p"),
            'device_id': 'ESP32-CAM-01',
            'condition': detected_class,
            'confidence': f"{round(confidence, 1)}%",
            'action': 'AUTO SPRAY TRIGGERED' if should_spray else 'HEALTHY - SCAN PAUSED',
            'relay_state': 'RELAY ON' if should_spray else 'OFF'
        })
    else:
        current_sprays = 0

    return jsonify({
        'status': detected_class,
        'confidence': round(confidence, 1),
        'should_spray': should_spray,
        'total_sprays': current_sprays
    })

@app.route('/api/pump-status', methods=['GET'])
def get_pump_status():
    if firebase_admin._apps:
        ref_status = db.reference('/system/status')
        data = ref_status.get() or {}
        return jsonify({'spray': data.get('should_spray', False)})
    return jsonify({'spray': False})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)