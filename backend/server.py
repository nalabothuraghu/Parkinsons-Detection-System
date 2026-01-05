import os
import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from joblib import load
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

# ==========================================
# 1. CONFIGURATION & LOAD MODELS
# ==========================================
UPLOAD_FOLDER = 'uploads'
EXCEL_PATH = "Demographics_age_sex.xlsx"

# Create uploads folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

print("⏳ Loading Models...")
try:
    # Load Spiral Model (TensorFlow/Keras)
    spiral_model = tf.keras.models.load_model('spiral_mobilenet.keras')
    
    # Load Voice Model & Scaler (Scikit-Learn)
    speech_model = load("parkinsons_best_model.pkl")
    speech_scaler = load("speech_scaler.pkl")
    
    print("✅ All Models & Scaler loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    # We don't exit, so the server can still run (but prediction will fail)
    spiral_model = None
    speech_model = None

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def process_spiral_image(filepath):
    """ 
    Reads image, resizes to 128x128, and normalizes for MobileNetV2 
    """
    img = cv2.imread(filepath)
    if img is None:
        raise Exception("Could not read image file.")
        
    # MobileNet expects RGB images
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize to 128x128 (Must match your training size)
    img = cv2.resize(img, (128, 128))
    
    # Normalize pixel values (0 to 1)
    img = img / 255.0
    
    # Add batch dimension: (1, 128, 128, 3)
    img = np.expand_dims(img, axis=0)
    return img

def get_sample_id_from_audio(filename):
    """ Extracts ID from filename (removes extension) """
    return os.path.splitext(filename)[0]

def load_audio_features(audio_filename):
    """ Looks up features in Excel based on filename """
    sample_id = get_sample_id_from_audio(audio_filename)
    
    # Load Sheets
    sheets = ['Parselmouth', 'LPC_means', 'LAR_means', 'Cep_means', 
              'MFCC_means', 'LPC_vars', 'LAR_vars', 'Cep_vars', 'MFCC_vars']
    
    try:
        dfs = {s: pd.read_excel(EXCEL_PATH, sheet_name=s) for s in sheets}
    except FileNotFoundError:
        raise Exception("Excel file 'Demographics_age_sex.xlsx' not found in backend folder.")

    # Clean & Merge
    for df in dfs.values():
        df.columns = df.columns.str.strip()
        if 'Sample ID' in df.columns:
            df.rename(columns={'Sample ID': 'Sample'}, inplace=True)
            
    merged = dfs['Parselmouth']
    for s in sheets[1:]:
        df = dfs[s].drop(columns=['Label'], errors='ignore')
        merged = pd.merge(merged, df, on='Sample', how='inner')

    # Find Sample
    row = merged[merged['Sample'] == sample_id]
    if row.empty:
        raise ValueError(f"ID '{sample_id}' not found in Excel database.")

    # Encode Sex & Drop non-features
    if 'Sex' in row.columns:
        row['Sex'] = row['Sex'].map({'M': 1, 'F': 0})
        
    X = row.drop(columns=['Sample', 'Label'], errors='ignore')
    X = X.select_dtypes(include=np.number)
    X = X.fillna(X.mean())

    # Scale
    X_scaled = speech_scaler.transform(X)
    return X_scaled

# ==========================================
# 3. API ROUTES
# ==========================================

@app.route('/predict-spiral', methods=['POST'])
def predict_spiral():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
        
    file = request.files['file']
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        processed_img = process_spiral_image(filepath)
        prediction = spiral_model.predict(processed_img)
        prob = float(prediction[0][0])
        
        # Threshold > 0.5 means Detected
        detected = prob > 0.5  
        
        return jsonify({'detected': bool(detected), 'confidence': f"{prob:.2f}"})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict-voice', methods=['POST'])
def predict_voice():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
        
    file = request.files['file']
    # IMPORTANT: We must save with ORIGINAL name to match Excel ID
    filename = secure_filename(file.filename) 
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        features = load_audio_features(filename)
        # Predict
        prediction = speech_model.predict(features)
        prob = speech_model.predict_proba(features)[0, 1]
        
        detected = prediction[0] == 1
        
        return jsonify({'detected': bool(detected), 'confidence': f"{prob:.2f}"})
    except ValueError as e:
        return jsonify({'error': str(e)}), 404 # ID not found
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict-combined', methods=['POST'])
def predict_combined():
    if 'spiral_file' not in request.files or 'voice_file' not in request.files:
        return jsonify({'error': 'Both files required'}), 400

    # 1. Save Files
    spiral_file = request.files['spiral_file']
    voice_file = request.files['voice_file']
    
    spiral_path = os.path.join(UPLOAD_FOLDER, secure_filename(spiral_file.filename))
    voice_name = secure_filename(voice_file.filename)
    voice_path = os.path.join(UPLOAD_FOLDER, voice_name)
    
    spiral_file.save(spiral_path)
    voice_file.save(voice_path)

    try:
        # 2. Get Spiral Prob
        spiral_img = process_spiral_image(spiral_path)
        spiral_pred_raw = spiral_model.predict(spiral_img)
        spiral_prob = float(spiral_pred_raw[0][0])

        # 3. Get Voice Prob
        voice_feat = load_audio_features(voice_name)
        speech_prob = speech_model.predict_proba(voice_feat)[0, 1]

        # 4. FUSION LOGIC (Weighted Average)
        weight_speech = 0.65
        weight_spiral = 0.35
        threshold = 0.45
        
        fused_prob = (weight_speech * speech_prob) + (weight_spiral * spiral_prob)
        is_parkinsons = fused_prob > threshold

        return jsonify({
            'detected': bool(is_parkinsons),
            'confidence': f"{fused_prob:.2f}",
            'details': {
                'spiral_prob': f"{spiral_prob:.2f}",
                'voice_prob': f"{speech_prob:.2f}"
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)