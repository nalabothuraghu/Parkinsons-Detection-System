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
# 1. SETUP
# ==========================================
UPLOAD_FOLDER = 'uploads'
EXCEL_PATH = "Demographics_age_sex.xlsx"

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# ==========================================
# 2. LOAD MODELS
# ==========================================
print("⏳ Loading Models...")
spiral_model = None
speech_model = None
speech_scaler = None

try:
    if os.path.exists('spiral_mobilenet.keras'):
        spiral_model = tf.keras.models.load_model('spiral_mobilenet.keras')
        print("✅ Spiral Model loaded.")
    
    if os.path.exists('parkinsons_best_model.pkl'):
        speech_model = load("parkinsons_best_model.pkl")
        print("✅ Speech Model loaded.")
    
    if os.path.exists('speech_scaler.pkl'):
        speech_scaler = load("speech_scaler.pkl")
        print("✅ Speech Scaler loaded.")

except Exception as e:
    print(f"❌ Error loading models: {e}")

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================

def process_spiral_image(filepath):
    """ Resizes to 128x128 and normalizes for the Image Model. """
    img = cv2.imread(filepath)
    if img is None: raise ValueError("Could not read image.")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def get_features_from_excel(filename):
    """
    Replicates the EXACT merge logic from your speech_binary.py
    to find features for a specific file.
    """
    # 1. Get ID from filename (e.g., '101_AUDIO.wav' -> '101_AUDIO')
    sample_id = os.path.splitext(filename)[0]

    # 2. Load all sheets (Same list as your training script)
    sheets = ['Parselmouth', 'LPC_means', 'LAR_means', 'Cep_means', 
              'MFCC_means', 'LPC_vars', 'LAR_vars', 'Cep_vars', 'MFCC_vars']
    
    try:
        # Read Excel
        dfs = {name: pd.read_excel(EXCEL_PATH, sheet_name=name) for name in sheets}
    except Exception:
        raise Exception(f"Could not find or open '{EXCEL_PATH}'")

    # 3. Clean Columns (Exactly as in your script)
    for name, df in dfs.items():
        df.columns = df.columns.str.strip()
        if 'Sample ID' in df.columns:
            df.rename(columns={'Sample ID': 'Sample'}, inplace=True)
            
    # 4. Merge Sheets (Exactly as in your script)
    merged = dfs['Parselmouth']
    for name in sheets[1:]:
        # Drop 'Label' if it exists in subsequent sheets to avoid duplication
        df_to_merge = dfs[name].drop(columns=['Label'], errors='ignore')
        merged = pd.merge(merged, df_to_merge, on='Sample', how='inner')

    # 5. Find the row for this specific file
    row = merged[merged['Sample'] == sample_id]
    
    if row.empty:
        raise ValueError(f"ID '{sample_id}' not found in Excel. Use a dataset file.")

    # 6. Process Sex (Map 'M'->1, 'F'->0)
    if 'Sex' in row.columns:
        row['Sex'] = row['Sex'].map({'M': 1, 'F': 0})

    # 7. Drop Non-Features
    X_row = row.drop(columns=['Sample', 'Label'], errors='ignore')
    
    # 8. Scale (Crucial!)
    if speech_scaler:
        # Fill NaNs with 0 (safe fallback) or mean if possible
        X_row = X_row.fillna(0)
        X_scaled = speech_scaler.transform(X_row)
        return X_scaled
    else:
        return X_row.values

# ==========================================
# 4. API ROUTES
# ==========================================

@app.route('/predict-spiral', methods=['POST'])
def predict_spiral():
    if 'file' not in request.files: return jsonify({'error': 'No file'}), 400
    file = request.files['file']
    path = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
    file.save(path)

    try:
        if not spiral_model: return jsonify({'error': 'Model not loaded'}), 500
        img = process_spiral_image(path)
        pred = spiral_model.predict(img)
        prob = float(pred[0][0])
        return jsonify({'detected': prob > 0.5, 'confidence': f"{prob:.2f}"})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict-voice', methods=['POST'])
def predict_voice():
    if 'file' not in request.files: return jsonify({'error': 'No file'}), 400
    
    file = request.files['file']
    # KEEP ORIGINAL FILENAME so we can look it up in Excel
    filename = secure_filename(file.filename) 
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)

    try:
        if not speech_model: return jsonify({'error': 'Voice model not loaded'}), 500

        # Get features using Excel Lookup
        features = get_features_from_excel(filename)
        
        # Predict
        pred = speech_model.predict(features)
        prob = speech_model.predict_proba(features)[0, 1]
        
        return jsonify({'detected': bool(pred[0] == 1), 'confidence': f"{prob:.2f}"})
    
    except ValueError as e:
        return jsonify({'error': f"ID Not Found: {str(e)}"}), 404
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500

@app.route('/predict-combined', methods=['POST'])
def predict_combined():
    if 'spiral_file' not in request.files or 'voice_file' not in request.files:
        return jsonify({'error': 'Missing files'}), 400

    s_file = request.files['spiral_file']
    v_file = request.files['voice_file']
    
    s_path = os.path.join(UPLOAD_FOLDER, secure_filename(s_file.filename))
    
    # Keep original audio name for lookup
    v_name = secure_filename(v_file.filename) 
    v_path = os.path.join(UPLOAD_FOLDER, v_name)
    
    s_file.save(s_path)
    v_file.save(v_path)

    try:
        # Spiral
        s_img = process_spiral_image(s_path)
        s_prob = float(spiral_model.predict(s_img)[0][0])

        # Voice
        v_feat = get_features_from_excel(v_name)
        v_prob = speech_model.predict_proba(v_feat)[0, 1]

        # Fusion
        final_score = (0.6 * v_prob) + (0.4 * s_prob)
        
        return jsonify({
            'detected': final_score > 0.5, 
            'confidence': f"{final_score:.2f}",
            'details': {'spiral': f"{s_prob:.2f}", 'voice': f"{v_prob:.2f}"}
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(port=5000, debug=True)