import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# =============================
# 1. Load all sheets
# =============================
file_path = "Demographics_age_sex.xlsx"
sheets = ['Parselmouth', 'LPC_means', 'LAR_means', 'Cep_means', 'MFCC_means',
          'LPC_vars', 'LAR_vars', 'Cep_vars', 'MFCC_vars']

dfs = {name: pd.read_excel(file_path, sheet_name=name) for name in sheets}

# =============================
# 2. Clean column names & standardize ID column
# =============================
for name, df in dfs.items():
    df.columns = df.columns.str.strip()
    if 'Sample ID' in df.columns:
        df.rename(columns={'Sample ID': 'Sample'}, inplace=True)
    # Remove extra spaces from labels
    if 'Label' in df.columns:
        df['Label'] = df['Label'].astype(str).str.strip()

# =============================
# 3. Drop duplicate 'Label' columns before merging
# =============================
for name in sheets[1:]:  # skip Parselmouth
    if 'Label' in dfs[name].columns:
        dfs[name] = dfs[name].drop(columns=['Label'])

# =============================
# 4. Merge all feature sheets
# =============================
merged = dfs['Parselmouth']
for name in sheets[1:]:
    merged = pd.merge(merged, dfs[name], on='Sample', how='inner')

print("✅ Data merged successfully! Final shape:", merged.shape)

# =============================
# 5. Encode categorical columns
# =============================
merged['Label'] = merged['Label'].str.strip()  # clean extra spaces
label_encoder = LabelEncoder()
merged['Label'] = label_encoder.fit_transform(merged['Label'])  # HC=0, PD=1

merged['Sex'] = merged['Sex'].map({'M': 1, 'F': 0})

# =============================
# 6. Features and Labels
# =============================
X = merged.drop(columns=['Sample', 'Label'])
y = merged['Label']

# Fill missing values if any
X = X.fillna(X.mean())

# =============================
# 7. Scale features
# =============================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =============================
# 8. Train-Test Split
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# =============================
# 9. Model Training and Evaluation
# =============================
models = {
    "Logistic Regression": LogisticRegression(max_iter=300),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    results[name] = acc
    print(f"\n📊 Model: {name}")
    print(f"Accuracy: {acc:.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))
    print("Classification Report:\n", classification_report(y_test, preds))

# =============================
# 10. Cross-validation
# =============================
print("\n==============================")
print("Cross-Validation Results")
print("==============================")
for name, model in models.items():
    cv_score = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy').mean()
    print(f"{name}: {cv_score:.4f}")

best_model_name = max(results, key=results.get)
print(f"\n🏆 Best Model: {best_model_name} with accuracy {results[best_model_name]:.4f}")
import joblib

# =============================
# 11. Save the best model
# =============================
best_model = models[best_model_name]
joblib.dump(best_model, "parkinsons_best_model.pkl")
print(f"💾 Best model saved as 'parkinsons_best_model.pkl'")
# Save scaler as well
joblib.dump(scaler, "speech_scaler.pkl")
print("💾 Scaler saved as 'speech_scaler.pkl'")