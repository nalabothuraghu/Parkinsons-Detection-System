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
# 2. LOAD MODELS & DATA ONCE (AT STARTUP)
# ==========================================
print("⏳ Loading Models and Data into memory...")

# Global variables
spiral_interpreter = None
spiral_input_details = None
spiral_output_details = None
speech_model = None
speech_scaler = None
excel_data = None  # Stores the merged dataframe

try:
    # --- A. Load Fast TFLite Model ---
    if os.path.exists('spiral_mobilenet.tflite'):
        spiral_interpreter = tf.lite.Interpreter(model_path='spiral_mobilenet.tflite')
        spiral_interpreter.allocate_tensors()
        spiral_input_details = spiral_interpreter.get_input_details()
        spiral_output_details = spiral_interpreter.get_output_details()
        print("✅ Fast TFLite Spiral Model loaded.")
    else:
        print("⚠️ Warning: spiral_mobilenet.tflite not found. Run the conversion script!")

    # --- B. Load Voice Models ---
    if os.path.exists('parkinsons_best_model.pkl'):
        speech_model = load("parkinsons_best_model.pkl")
        print("✅ Speech Model loaded.")
    
    if os.path.exists('speech_scaler.pkl'):
        speech_scaler = load("speech_scaler.pkl")
        print("✅ Speech Scaler loaded.")

    # --- C. Pre-load Excel Data (MASSIVE SPEED BOOST) ---
    print("⏳ Reading Excel sheets (this only happens once)...")
    sheets = ['Parselmouth', 'LPC_means', 'LAR_means', 'Cep_means', 
              'MFCC_means', 'LPC_vars', 'LAR_vars', 'Cep_vars', 'MFCC_vars']
    
    dfs = {name: pd.read_excel(EXCEL_PATH, sheet_name=name) for name in sheets}
    
    for name, df in dfs.items():
        df.columns = df.columns.str.strip()
        if 'Sample ID' in df.columns:
            df.rename(columns={'Sample ID': 'Sample'}, inplace=True)
            
    excel_data = dfs['Parselmouth']
    for name in sheets[1:]:
        df_to_merge = dfs[name].drop(columns=['Label'], errors='ignore')
        excel_data = pd.merge(excel_data, df_to_merge, on='Sample', how='inner')
        
    print("✅ Excel Data loaded and merged into memory.")

except Exception as e:
    print(f"❌ Error during initialization: {e}")


# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================

def process_spiral_image(filepath):
    """ Resizes to 128x128, normalizes, and casts to float32 for TFLite. """
    img = cv2.imread(filepath)
    if img is None: raise ValueError("Could not read image.")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128))
    # TFLite requires float32
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict_spiral_tflite(img):
    """ Helper to run the fast TFLite prediction """
    spiral_interpreter.set_tensor(spiral_input_details[0]['index'], img)
    spiral_interpreter.invoke()
    pred = spiral_interpreter.get_tensor(spiral_output_details[0]['index'])
    return float(pred[0][0])

def get_features_from_memory(filename):
    """ Grabs features instantly from RAM instead of reading the file. """
    if excel_data is None:
        raise Exception("Excel data failed to load at startup.")

    sample_id = os.path.splitext(filename)[0]

    # Find row in pre-loaded data (using .copy() to prevent Pandas warnings)
    row = excel_data[excel_data['Sample'] == sample_id].copy()
    
    if row.empty:
        raise ValueError(f"ID '{sample_id}' not found in Excel. Use a dataset file.")

    if 'Sex' in row.columns:
        row['Sex'] = row['Sex'].map({'M': 1, 'F': 0})

    X_row = row.drop(columns=['Sample', 'Label'], errors='ignore')
    
    if speech_scaler:
        X_row = X_row.fillna(0)
        return speech_scaler.transform(X_row)
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
        if not spiral_interpreter: return jsonify({'error': 'TFLite Model not loaded'}), 500
        
        img = process_spiral_image(path)
        prob = predict_spiral_tflite(img)
        
        return jsonify({'detected': prob > 0.5, 'confidence': f"{prob:.2f}"})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict-voice', methods=['POST'])
def predict_voice():
    if 'file' not in request.files: return jsonify({'error': 'No file'}), 400
    
    file = request.files['file']
    filename = secure_filename(file.filename) 
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)

    try:
        if not speech_model: return jsonify({'error': 'Voice model not loaded'}), 500

        # Get features instantly from memory
        features = get_features_from_memory(filename)
        
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
    v_name = secure_filename(v_file.filename) 
    v_path = os.path.join(UPLOAD_FOLDER, v_name)
    
    s_file.save(s_path)
    v_file.save(v_path)

    try:
        # Fast Spiral Prediction
        s_img = process_spiral_image(s_path)
        s_prob = predict_spiral_tflite(s_img)

        # Fast Voice Prediction
        v_feat = get_features_from_memory(v_name)
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