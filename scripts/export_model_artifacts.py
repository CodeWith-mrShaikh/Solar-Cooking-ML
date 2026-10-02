"""
Model Artifacts Export Script
=============================
Trains the Extra Trees production model and fits all preprocessing components
using the canonical training dataset, then exports all required artifacts to models/.

Usage:
    python scripts/export_model_artifacts.py
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_squared_error

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.config import (
    CANONICAL_FEATURES,
    TARGET_COL,
    CANONICAL_DATA_PATH,
    MODELS_DIR,
    MODEL_PATH,
    IMPUTER_PATH,
    SCALER_PATH,
    FOOD_ENCODER_PATH,
    COOKER_ENCODER_PATH,
    FEATURE_CONFIG_PATH,
    METADATA_PATH,
    PRODUCTION_EVALUATION
)

def export_artifacts():
    print("=" * 60)
    print("SOLAR COOKING ML — EXPORTING PRODUCTION MODEL ARTIFACTS")
    print("=" * 60)

    os.makedirs(MODELS_DIR, exist_ok=True)

    if not os.path.exists(CANONICAL_DATA_PATH):
        # Fallback check in parent directory if data/ has not been copied yet
        parent_candidate = os.path.join(os.path.dirname(PROJECT_ROOT), "Final Solar Cooking Training Dataset.csv")
        if os.path.exists(parent_candidate):
            os.makedirs(os.path.dirname(CANONICAL_DATA_PATH), exist_ok=True)
            import shutil
            shutil.copy2(parent_candidate, CANONICAL_DATA_PATH)
            print(f"Copied dataset to canonical location: {CANONICAL_DATA_PATH}")
        else:
            raise FileNotFoundError(f"Training dataset not found at {CANONICAL_DATA_PATH} or {parent_candidate}")

    print(f"Loading canonical dataset from: {CANONICAL_DATA_PATH}")
    df = pd.read_csv(CANONICAL_DATA_PATH)
    print(f"Dataset shape: {df.shape}")

    # Verify features and target exist
    for col in CANONICAL_FEATURES:
        if col not in df.columns:
            raise ValueError(f"Required feature '{col}' is missing from dataset!")
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' is missing from dataset!")

    X = df[CANONICAL_FEATURES].copy()
    y = df[TARGET_COL].copy()

    # 1. Fit LabelEncoders
    print("\n[1/6] Fitting and exporting LabelEncoders...")
    food_encoder = LabelEncoder()
    X['food_matrix'] = food_encoder.fit_transform(X['food_matrix'].astype(str))
    joblib.dump(food_encoder, FOOD_ENCODER_PATH)
    print(f"  [OK] Saved {FOOD_ENCODER_PATH}")
    print(f"       Food matrix classes ({len(food_encoder.classes_)}): {list(food_encoder.classes_)}")

    cooker_encoder = LabelEncoder()
    X['cooker_type'] = cooker_encoder.fit_transform(X['cooker_type'].astype(str))
    joblib.dump(cooker_encoder, COOKER_ENCODER_PATH)
    print(f"  [OK] Saved {COOKER_ENCODER_PATH}")
    print(f"       Cooker type classes ({len(cooker_encoder.classes_)}): {list(cooker_encoder.classes_)}")

    # 2. Train / Test Split (80:20, random_state=42)
    print("\n[2/6] Splitting dataset (80:20, random_state=42)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"  Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

    # 3. Fit SimpleImputer
    print("\n[3/6] Fitting and exporting SimpleImputer(strategy='mean')...")
    imputer = SimpleImputer(strategy='mean')
    if hasattr(imputer, "set_output"):
        imputer.set_output(transform="pandas")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)
    joblib.dump(imputer, IMPUTER_PATH)
    print(f"  [OK] Saved {IMPUTER_PATH}")

    # 4. Fit StandardScaler
    print("\n[4/6] Fitting and exporting StandardScaler()...")
    scaler = StandardScaler()
    if hasattr(scaler, "set_output"):
        scaler.set_output(transform="pandas")
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)
    joblib.dump(scaler, SCALER_PATH)
    print(f"  [OK] Saved {SCALER_PATH}")

    # 5. Train ExtraTreesRegressor
    print("\n[5/6] Training and exporting ExtraTreesRegressor(n_estimators=100, random_state=42)...")
    model = ExtraTreesRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    joblib.dump(model, MODEL_PATH, compress=3)
    print(f"  [OK] Saved {MODEL_PATH} (compressed)")

    # Evaluate
    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)
    train_r2 = float(r2_score(y_train, train_pred))
    test_r2 = float(r2_score(y_test, test_pred))
    test_mse = float(mean_squared_error(y_test, test_pred))
    diff = float(train_r2 - test_r2)

    print(f"  Train R²: {train_r2:.6f}")
    print(f"  Test R²:  {test_r2:.6f} (Colab Reference: {PRODUCTION_EVALUATION['test_r2']:.6f})")
    print(f"  Test MSE: {test_mse:.6f} (Colab Reference: {PRODUCTION_EVALUATION['test_mse']:.6f})")
    print(f"  Overfitting Difference: {diff:.6f}")

    # 6. Export Feature Config and Metadata
    print("\n[6/6] Exporting feature configuration and metadata...")
    feature_config = {
        "features": CANONICAL_FEATURES,
        "target": TARGET_COL,
        "feature_order": CANONICAL_FEATURES
    }
    with open(FEATURE_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(feature_config, f, indent=4)
    print(f"  [OK] Saved {FEATURE_CONFIG_PATH}")

    metadata = {
        "model_name": "Extra Trees Regressor",
        "n_estimators": 100,
        "random_state": 42,
        "target": TARGET_COL,
        "test_split": 0.20,
        "test_r2": PRODUCTION_EVALUATION["test_r2"],
        "test_mse": PRODUCTION_EVALUATION["test_mse"],
        "overfitting_diff": PRODUCTION_EVALUATION["overfitting_diff"],
        "features": CANONICAL_FEATURES,
        "food_matrix_classes": [str(c) for c in food_encoder.classes_],
        "cooker_type_classes": [str(c) for c in cooker_encoder.classes_],
        "dataset_rows": len(df),
        "dataset_columns": len(df.columns)
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print(f"  [OK] Saved {METADATA_PATH}")

    print("\n" + "=" * 60)
    print("ALL PRODUCTION ARTIFACTS SUCCESSFULLY EXPORTED!")
    print("=" * 60)

if __name__ == "__main__":
    export_artifacts()
