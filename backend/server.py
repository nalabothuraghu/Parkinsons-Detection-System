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
# 1. SETUP & LOAD ONCE (AT STARTUP)
# ==========================================
print("⏳ Loading Models and Data into memory...")

spiral_interpreter = None
spiral_input_details = None
spiral_output_details = None
speech_model = None
speech_scaler = None
excel_data = None  

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

    # --- C. Pre-load Excel Data ---
    EXCEL_PATH = "Demographics_age_sex.xlsx"
    if os.path.exists(EXCEL_PATH):
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
    else:
        print(f"⚠️ Warning: {EXCEL_PATH} not found.")

except Exception as e:
    print(f"❌ Error during initialization: {e}")


# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def process_spiral_image_from_memory(file_stream):
    """ Reads image directly from RAM, skipping the hard drive entirely. """
    # Convert network stream directly to OpenCV image
    in_memory_file = file_stream.read()
    npimg = np.frombuffer(in_memory_file, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    
    if img is None: raise ValueError("Could not decode incoming image file.")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict_spiral_tflite(img):
    spiral_interpreter.set_tensor(spiral_input_details[0]['index'], img)
    spiral_interpreter.invoke()
    pred = spiral_interpreter.get_tensor(spiral_output_details[0]['index'])
    return float(pred[0][0])

def get_features_from_memory(filename):
    if excel_data is None: raise Exception("Excel data failed to load at startup.")
    sample_id = os.path.splitext(filename)[0]
    row = excel_data[excel_data['Sample'] == sample_id].copy()
    
    if row.empty: raise ValueError(f"ID '{sample_id}' not found in Excel.")
    if 'Sex' in row.columns: row['Sex'] = row['Sex'].map({'M': 1, 'F': 0})

    X_row = row.drop(columns=['Sample', 'Label'], errors='ignore')
    if speech_scaler:
        X_row = X_row.fillna(0)
        return speech_scaler.transform(X_row)
    return X_row.values


# ==========================================
# 3. API ROUTES
# ==========================================

@app.route('/predict-spiral', methods=['POST'])
def predict_spiral():
    if 'file' not in request.files: return jsonify({'error': 'No file'}), 400
    file = request.files['file']

    try:
        if not spiral_interpreter: return jsonify({'error': 'TFLite Model not loaded'}), 500
        
        # Process directly from memory (no file.save!)
        img = process_spiral_image_from_memory(file)
        prob = predict_spiral_tflite(img)
        
        return jsonify({'detected': prob > 0.5, 'confidence': f"{prob:.2f}"})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict-voice', methods=['POST'])
def predict_voice():
    if 'file' not in request.files: return jsonify({'error': 'No file'}), 400
    file = request.files['file']
    filename = secure_filename(file.filename) 
    
    # We do NOT save the audio file because your logic only needs the filename
    # to search the Excel sheet!

    try:
        if not speech_model: return jsonify({'error': 'Voice model not loaded'}), 500
        features = get_features_from_memory(filename)
        pred = speech_model.predict(features)
        prob = speech_model.predict_proba(features)[0, 1]
        return jsonify({'detected': bool(pred[0] == 1), 'confidence': f"{prob:.2f}"})
    
    except ValueError as e:
        return jsonify({'error': f"ID Not Found: {str(e)}"}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict-combined', methods=['POST'])
def predict_combined():
    if 'spiral_file' not in request.files or 'voice_file' not in request.files:
        return jsonify({'error': 'Missing files'}), 400

    s_file = request.files['spiral_file']
    v_file = request.files['voice_file']
    v_name = secure_filename(v_file.filename) 

    # We do NOT save ANY files to disk here either!

    try:
        # Fast Spiral Prediction (In-Memory)
        s_img = process_spiral_image_from_memory(s_file)
        s_prob = predict_spiral_tflite(s_img)

        # Fast Voice Prediction (Excel Lookup)
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