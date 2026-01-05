import os
import numpy as np
import cv2
import pandas as pd
from skimage import feature
from joblib import load

# =============================
# 1️⃣ Load Models & Scaler
# =============================
spiral_model = load("spiral_model.pkl")
speech_model = load("parkinsons_best_model.pkl")
speech_scaler = load("speech_scaler.pkl")

print("[INFO] Models and Scaler loaded successfully!")

# =============================
# 2️⃣ Spiral Feature Extraction (UNCHANGED)
# =============================
def quantify_image(image):
    return feature.hog(
        image,
        orientations=9,
        pixels_per_cell=(10, 10),
        cells_per_block=(2, 2),
        transform_sqrt=True,
        block_norm="L1"
    )

# =============================
# 3️⃣ Extract Sample ID from Audio Path
# =============================
def get_sample_id_from_audio(audio_path):
    """
    Example:
    AH_545622720-E1486AF6-8C95-47EB-829B-4D62698C987A.wav
    → AH_545622720-E1486AF6-8C95-47EB-829B-4D62698C987A
    """
    return os.path.splitext(os.path.basename(audio_path))[0]

# =============================
# 4️⃣ Load SINGLE Audio Features from Excel
# =============================
def load_audio_features_from_excel(excel_file, audio_path):
    sample_id = get_sample_id_from_audio(audio_path)

    sheets = [
        'Parselmouth', 'LPC_means', 'LAR_means', 'Cep_means',
        'MFCC_means', 'LPC_vars', 'LAR_vars', 'Cep_vars', 'MFCC_vars'
    ]

    dfs = {s: pd.read_excel(excel_file, sheet_name=s) for s in sheets}

    # Clean column names
    for df in dfs.values():
        df.columns = df.columns.str.strip()
        if 'Sample ID' in df.columns:
            df.rename(columns={'Sample ID': 'Sample'}, inplace=True)
        if 'Label' in df.columns:
            df['Label'] = df['Label'].astype(str).str.strip()

    # Merge exactly like training
    merged = dfs['Parselmouth']
    for s in sheets[1:]:
        df = dfs[s].drop(columns=['Label'], errors='ignore')
        merged = pd.merge(merged, df, on='Sample', how='inner')

    # Select sample
    row = merged[merged['Sample'] == sample_id]
    if row.empty:
        raise ValueError(f"[ERROR] Sample ID '{sample_id}' not found in Excel")

    # Encode sex
    row['Sex'] = row['Sex'].map({'M': 1, 'F': 0})

    X = row.drop(columns=['Sample', 'Label'], errors='ignore')
    X = X.select_dtypes(include=np.number)
    X = X.fillna(X.mean())

    # Apply SAME scaler
    X_scaled = speech_scaler.transform(X)

    return X_scaled, sample_id

# =============================
# 🧪 Single Test Function
# =============================
def test_single_input(
    spiral_file,
    audio_file,
    excel_file,
    weight_speech=0.65,
    weight_spiral=0.35,
    threshold=0.45
):
    print("\n[INFO] Running single-sample test...")

    # -------- Spiral --------
    img = cv2.imread(spiral_file)
    if img is None:
        raise FileNotFoundError(f"[ERROR] Spiral image not found: {spiral_file}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (200, 200))
    img = cv2.threshold(
        img, 0, 255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )[1]

    spiral_feat = quantify_image(img).reshape(1, -1)
    spiral_prob = spiral_model.predict_proba(spiral_feat)[0, 1]
    spiral_pred = spiral_model.predict(spiral_feat)[0]

    # -------- Audio (via Excel lookup) --------
    speech_feat, sample_id = load_audio_features_from_excel(excel_file, audio_file)
    speech_prob = speech_model.predict_proba(speech_feat)[0, 1]
    speech_pred = speech_model.predict(speech_feat)[0]

    # -------- Fusion --------
    fused_prob = (
        weight_speech * speech_prob +
        weight_spiral * spiral_prob
    )
    fused_label = int(fused_prob > threshold)

    # -------- Output --------
    print(f"\n🔹 Sample ID: {sample_id}")

    print("\n🔹 Spiral Model:")
    print(f"   Probability : {spiral_prob:.3f}")
    print(f"   Prediction  : {'Parkinson’s' if spiral_pred else 'Healthy'}")

    print("\n🔹 Speech Model:")
    print(f"   Probability : {speech_prob:.3f}")
    print(f"   Prediction  : {'Parkinson’s' if speech_pred else 'Healthy'}")

    print("\n🟢 Final Fused Decision:")
    print(f"   Weighted Probability : {fused_prob:.3f}")
    print(f"   Final Prediction     : {'Parkinson’s Detected' if fused_label else 'Healthy Control'}")

    print("\n--------------------------------------------")
    print(f"Fusion Weights → Speech: {weight_speech}, Spiral: {weight_spiral}, Threshold: {threshold}")
    print("--------------------------------------------")

    return fused_label, fused_prob

# =============================
# 🚀 RUN TEST
# =============================
if __name__ == "__main__":
    print("\n================= TESTING SECTION =================")

    spiral_path = r"C:\Users\DELL\OneDrive\Desktop\datasetnew\spiral\testing\healthy\V55HE14.png"
    audio_path = r"C:\Users\DELL\OneDrive\Desktop\datasetnew\PD_AH\AH_545841227-5C77713A-66F1-49D0-BC8A-702C152E668D.wav"
    excel_path = r"C:\Users\DELL\OneDrive\Desktop\datasetnew\Demographics_age_sex.xlsx"

    test_single_input(
        spiral_path,
        audio_path,
        excel_path,
        weight_speech=0.65,
        weight_spiral=0.35,
        threshold=0.45
    )

    print("===================================================")