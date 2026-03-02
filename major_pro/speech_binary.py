# ============================================================
# Train & Evaluate Parkinson's Disease Classifier
# Input: speech_features_from_wav.xlsx
# ============================================================

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# ============================================================
# 1️⃣ LOAD DATA
# ============================================================

df = pd.read_excel("speech_features_from_wav.xlsx")

print("✅ Dataset loaded:", df.shape)

# ============================================================
# 2️⃣ PREPROCESSING
# ============================================================

# Encode labels
label_encoder = LabelEncoder()
df["Label"] = label_encoder.fit_transform(df["Label"])  
# Healthy = 0, PD = 1

# Separate features & labels
X = df.drop(columns=["Sample", "Label"])
y = df["Label"]

# Handle missing values
X = X.fillna(X.mean())

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler & feature order
joblib.dump(scaler, "speech_scaler.pkl")
joblib.dump(X.columns.tolist(), "feature_order.pkl")

# ============================================================
# 3️⃣ TRAIN–TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ============================================================
# 4️⃣ MODEL TRAINING
# ============================================================

models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
    "XGBoost": XGBClassifier(
        eval_metric="logloss",
        random_state=42
    )
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    results[name] = acc

    print(f"\n📌 Model: {name}")
    print("Accuracy:", round(acc * 100, 2), "%")
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))
    print("Classification Report:\n", classification_report(y_test, preds))

# ============================================================
# 5️⃣ CROSS-VALIDATION
# ============================================================

print("\n🔁 Cross-Validation Accuracy (5-fold):")
for name, model in models.items():
    cv = cross_val_score(model, X_scaled, y, cv=5, scoring="accuracy").mean()
    print(f"{name}: {round(cv * 100, 2)} %")

# ============================================================
# 6️⃣ SAVE BEST MODEL
# ============================================================

best_model_name = max(results, key=results.get)
best_model = models[best_model_name]

joblib.dump(best_model, "parkinsons_best_model.pkl")

print(f"\n🏆 Best Model Saved: {best_model_name}")
print("📁 Saved files:")
print("   - parkinsons_best_model.pkl")
print("   - speech_scaler.pkl")
print("   - feature_order.pkl")
# ============================================================
# Predict Parkinson's Disease from a RAW WAV file
# ============================================================

import numpy as np
import pandas as pd
import librosa
import parselmouth
import joblib
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# CONFIG (MUST MATCH TRAINING)
# ============================================================

SAMPLE_RATE = 16000
N_MFCC = 13
LPC_ORDER = 16

# ============================================================
# FEATURE EXTRACTION (IDENTICAL TO TRAINING)
# ============================================================

def extract_features_from_wav(wav_path):
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

    return pd.DataFrame([features])

# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_from_wav(wav_path):
    model = joblib.load("parkinsons_best_model.pkl")
    scaler = joblib.load("speech_scaler.pkl")
    feature_order = joblib.load("feature_order.pkl")

    # Extract features
    df = extract_features_from_wav(wav_path)

    # Ensure same feature order
    df = df.reindex(columns=feature_order, fill_value=0)

    # Scale
    df_scaled = scaler.transform(df)

    # Predict
    pred = model.predict(df_scaled)[0]
    prob = model.predict_proba(df_scaled)[0]

    label = "Parkinson’s Disease (PD)" if pred == 1 else "Healthy"

    return label, prob

# ============================================================
# RUN (CHANGE WAV FILE PATH ONLY)
# ============================================================

if __name__ == "__main__":

    wav_file = r"C:\Users\DELL\OneDrive\Desktop\datasetnew\PD_AH\AH_545622718-C052AD58-5E6B-4ADC-855C-F76B66BAFA6E.wav"

    label, probability = predict_from_wav(wav_file)

    print("\n🧠 Prediction Result:")
    print("File:", wav_file)
    print("Result:", label)
    print("Probability [Healthy, PD]:", probability)
