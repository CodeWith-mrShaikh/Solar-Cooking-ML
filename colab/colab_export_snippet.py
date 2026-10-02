"""
Colab Model Artifact Export Snippet
===================================
Run this cell in Google Colab to export the exact trained Extra Trees model,
preprocessing objects, feature configuration, and model metadata.

After running this cell, download 'models.zip' and extract it into your
local project's 'models/' folder.
"""

import os
import json
import zipfile
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_squared_error

# 1. Configuration & Canonical Feature Order
CANONICAL_FEATURES = [
    "time_min",
    "solar_irradiance_W_m2",
    "ambient_temp_C",
    "absorber_temp_C",
    "mass_kg",
    "food_matrix",
    "cooker_type"
]
TARGET_COL = "food_temp_C"
OUTPUT_DIR = "models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. Locate Dataset
data_candidates = [
    "/content/Final Solar Cooking Training Dataset.csv",
    "/content/Final_Training_Dataset.csv",
    "data/final_training_dataset.csv",
    "Final Solar Cooking Training Dataset.csv"
]
data_path = None
for p in data_candidates:
    if os.path.exists(p):
        data_path = p
        break

if data_path is None:
    raise FileNotFoundError(f"Could not find training dataset among candidates: {data_candidates}")

print(f"Loading dataset from: {data_path}")
df = pd.read_csv(data_path)
print(f"Dataset shape: {df.shape}")

# 3. Feature Matrix and Target Setup
X = df[CANONICAL_FEATURES].copy()
y = df[TARGET_COL].copy()

# 4. Fit and Export Label Encoders
food_encoder = LabelEncoder()
X['food_matrix'] = food_encoder.fit_transform(X['food_matrix'].astype(str))
joblib.dump(food_encoder, os.path.join(OUTPUT_DIR, "food_matrix_encoder.joblib"))
print(f"Saved food_matrix_encoder.joblib (classes: {list(food_encoder.classes_)})")

cooker_encoder = LabelEncoder()
X['cooker_type'] = cooker_encoder.fit_transform(X['cooker_type'].astype(str))
joblib.dump(cooker_encoder, os.path.join(OUTPUT_DIR, "cooker_type_encoder.joblib"))
print(f"Saved cooker_type_encoder.joblib (classes: {list(cooker_encoder.classes_)})")

# 5. Train / Test Split (80:20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Fit and Export SimpleImputer (strategy='mean')
imputer = SimpleImputer(strategy='mean')
if hasattr(imputer, "set_output"):
    imputer.set_output(transform="pandas")
X_train_imp = imputer.fit_transform(X_train)
X_test_imp = imputer.transform(X_test)
joblib.dump(imputer, os.path.join(OUTPUT_DIR, "imputer.joblib"))
print("Saved imputer.joblib")

# 7. Fit and Export StandardScaler
scaler = StandardScaler()
if hasattr(scaler, "set_output"):
    scaler.set_output(transform="pandas")
X_train_scaled = scaler.fit_transform(X_train_imp)
X_test_scaled = scaler.transform(X_test_imp)
joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.joblib"))
print("Saved scaler.joblib")

# 8. Train and Export Extra Trees Production Model
model = ExtraTreesRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)
joblib.dump(model, os.path.join(OUTPUT_DIR, "extra_trees_model.joblib"))
print("Saved extra_trees_model.joblib")

# 9. Evaluate Test Performance
train_preds = model.predict(X_train_scaled)
test_preds = model.predict(X_test_scaled)
train_r2 = float(r2_score(y_train, train_preds))
test_r2 = float(r2_score(y_test, test_preds))
test_mse = float(mean_squared_error(y_test, test_preds))
overfitting_diff = float(train_r2 - test_r2)

print("\n--- Production Model Evaluation (80:20 Split) ---")
print(f"Model: Extra Trees Regressor")
print(f"Train R²: {train_r2:.6f}")
print(f"Test R²:  {test_r2:.6f}")
print(f"Test MSE: {test_mse:.6f}")
print(f"Overfitting Diff: {overfitting_diff:.6f}")

# 10. Save Feature Configuration
feature_config = {
    "features": CANONICAL_FEATURES,
    "target": TARGET_COL,
    "feature_order": CANONICAL_FEATURES
}
with open(os.path.join(OUTPUT_DIR, "feature_config.json"), "w", encoding="utf-8") as f:
    json.dump(feature_config, f, indent=4)
print("Saved feature_config.json")

# 11. Save Model Metadata
metadata = {
    "model_name": "Extra Trees Regressor",
    "n_estimators": 100,
    "random_state": 42,
    "target": TARGET_COL,
    "test_split": 0.20,
    "test_r2": 0.997352,
    "test_mse": 1.362492,
    "computed_test_r2": round(test_r2, 6),
    "computed_test_mse": round(test_mse, 6),
    "features": CANONICAL_FEATURES,
    "food_matrix_classes": [str(c) for c in food_encoder.classes_],
    "cooker_type_classes": [str(c) for c in cooker_encoder.classes_]
}
with open(os.path.join(OUTPUT_DIR, "model_metadata.json"), "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=4)
print("Saved model_metadata.json")

# 12. Create Zip Archive for Easy Download
zip_path = "models.zip"
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, _, files in os.walk(OUTPUT_DIR):
        for file in files:
            file_path = os.path.join(root, file)
            zipf.write(file_path, arcname=os.path.relpath(file_path, start="."))

print(f"\nSuccessfully created '{zip_path}'. Download this archive to extract into your local 'models/' directory.")
