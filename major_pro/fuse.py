import os
import numpy as np
import cv2
import pandas as pd
from skimage import feature
from joblib import load
import librosa
import parselmouth
from scipy.stats import skew, kurtosis

# =============================
# 1️⃣ Load Models & Scaler
# =============================
spiral_model = load("spiral_model.pkl")
speech_model = load("parkinsons_best_model.pkl")
speech_scaler = load("speech_scaler.pkl")
feature_order = load("feature_order.pkl")

print("[INFO] Models, Scaler, and Feature Order loaded successfully!")

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
# 3️⃣ Speech Feature Extraction (WAV-BASED, SAME AS TRAINING)
# =============================

SAMPLE_RATE = 16000
N_MFCC = 13
LPC_ORDER = 16

def extract_speech_features_from_wav(wav_path):
    features = {}

    # Load audio
    y, sr = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    for i in range(N_MFCC):
        features[f"MFCC{i+1}_mean"] = np.mean(mfcc[i])
        features[f"MFCC{i+1}_var"]  = np.var(mfcc[i])
        features[f"MFCC{i+1}_skew"] = skew(mfcc[i])
        features[f"MFCC{i+1}_kurt"] = kurtosis(mfcc[i])

    # LPC
    lpc = librosa.lpc(y, order=LPC_ORDER)
    for i, coef in enumerate(lpc):
        features[f"LPC{i+1}"] = coef

    # Cepstral
    spectrum = np.abs(np.fft.fft(y))
    cepstrum = np.fft.ifft(np.log(spectrum + 1e-10)).real
    features["Cep_mean"] = np.mean(cepstrum)
    features["Cep_var"]  = np.var(cepstrum)

    # Parselmouth (Praat-based)
    snd = parselmouth.Sound(wav_path)

    pitch = snd.to_pitch()
    features["Pitch_mean"] = np.nanmean(pitch.selected_array['frequency'])
    features["Pitch_std"]  = np.nanstd(pitch.selected_array['frequency'])

    point_process = parselmouth.praat.call(
        snd, "To PointProcess (periodic, cc)", 75, 500
    )

    features["Jitter_local"] = parselmouth.praat.call(
        point_process, "Get jitter (local)",
        0, 0, 0.0001, 0.02, 1.3
    )

    features["Shimmer_local"] = parselmouth.praat.call(
        [snd, point_process], "Get shimmer (local)",
        0, 0, 0.0001, 0.02, 1.3, 1.6
    )

    # Energy
    features["RMS_energy"] = np.mean(librosa.feature.rms(y=y))

    # Convert to DataFrame & align features
    df = pd.DataFrame([features])
    df = df.reindex(columns=feature_order, fill_value=0)

    # Scale
    df_scaled = speech_scaler.transform(df)

    return df_scaled

# =============================
# 🧪 Single Test Function (FUSION)
# =============================
def test_single_input(
    spiral_file,
    audio_file,
    weight_speech=0.65,
    weight_spiral=0.35,
    threshold=0.45
):
    print("\n[INFO] Running single-sample fusion test...")

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

    # -------- Speech (RAW WAV → FEATURES) --------
    speech_feat = extract_speech_features_from_wav(audio_file)
    speech_prob = speech_model.predict_proba(speech_feat)[0, 1]
    speech_pred = speech_model.predict(speech_feat)[0]

    # -------- Fusion --------
    fused_prob = (
        weight_speech * speech_prob +
        weight_spiral * spiral_prob
    )
    fused_label = int(fused_prob > threshold)

    # -------- Output --------
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

    spiral_path = r"C:\Users\DELL\OneDrive\Desktop\datasetnew\spiral\testing\parkinson\V01PE01.png"
    audio_path  = r"C:\Users\DELL\OneDrive\Desktop\datasetnew\HC_AH\AH_325A_3EB21DC7-C340-4D0E-AC9E-0EABF217BBEE.wav"

    test_single_input(
        spiral_path,
        audio_path,
        weight_speech=0.65,
        weight_spiral=0.35,
        threshold=0.45
    )

    print("===================================================")
