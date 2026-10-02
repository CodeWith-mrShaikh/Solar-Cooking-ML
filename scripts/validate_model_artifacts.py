"""
Model Artifact Validation Script
================================
Validates all required model and preprocessing artifacts, checks configurations,
and executes a sample inference test to confirm production readiness.

Usage:
    python scripts/validate_model_artifacts.py
"""

import os
import sys
import json

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.config import (
    MODEL_PATH,
    IMPUTER_PATH,
    SCALER_PATH,
    FOOD_ENCODER_PATH,
    COOKER_ENCODER_PATH,
    FEATURE_CONFIG_PATH,
    METADATA_PATH,
    CANONICAL_FEATURES,
    CANONICAL_DATA_PATH
)
from src.model_loader import load_model_artifacts, ArtifactLoadError
from src.predictor import predict_food_temperature
from src.data_loader import get_dataset_kpis

def validate_all():
    print("=" * 65)
    print("SOLAR COOKING ML — PRODUCTION ARTIFACT VERIFICATION")
    print("=" * 65)

    errors = []

    # 1. Check Canonical Dataset
    print("\n--- 1. Canonical Dataset ---")
    if os.path.exists(CANONICAL_DATA_PATH):
        try:
            kpis = get_dataset_kpis(CANONICAL_DATA_PATH)
            print(f"[OK] Canonical Dataset: {CANONICAL_DATA_PATH}")
            print(f"     Rows: {kpis['total_rows']}, Columns: {kpis['total_columns']}, Unique Runs: {kpis['unique_runs_count']}")
        except Exception as e:
            errors.append(f"Failed to inspect dataset: {e}")
            print(f"[FAIL] Canonical Dataset: {e}")
    else:
        errors.append(f"Dataset missing at {CANONICAL_DATA_PATH}")
        print(f"[FAIL] Canonical Dataset missing at {CANONICAL_DATA_PATH}")

    # 2. Check Artifact Files Existence
    print("\n--- 2. Artifact Files Existence ---")
    files_to_check = [
        ("Model", MODEL_PATH),
        ("Imputer", IMPUTER_PATH),
        ("Scaler", SCALER_PATH),
        ("Food Encoder", FOOD_ENCODER_PATH),
        ("Cooker Encoder", COOKER_ENCODER_PATH),
        ("Feature Configuration", FEATURE_CONFIG_PATH),
        ("Metadata", METADATA_PATH)
    ]

    for name, path in files_to_check:
        if os.path.exists(path):
            size_kb = os.path.getsize(path) / 1024.0
            print(f"[OK] File exists: {name:22} ({size_kb:.1f} KB) -> {os.path.basename(path)}")
        else:
            errors.append(f"Missing file: {name} ({path})")
            print(f"[FAIL] File missing: {name:22} -> {path}")

    if errors:
        print("\n" + "!" * 65)
        print("VERIFICATION FAILED: Missing artifact files.")
        print("Please run: python scripts/export_model_artifacts.py")
        print("!" * 65)
        sys.exit(1)

    # 3. Load Artifacts into Memory
    print("\n--- 3. Loading Artifacts into Memory ---")
    try:
        artifacts = load_model_artifacts()
        print("[OK] Model")
        print("[OK] Imputer")
        print("[OK] Scaler")
        print("[OK] Food Encoder")
        print("[OK] Cooker Encoder")
        print("[OK] Feature Configuration")
    except ArtifactLoadError as e:
        print(f"[FAIL] Error loading artifacts: {e}")
        sys.exit(1)

    # 4. Validate Feature Configuration
    print("\n--- 4. Feature Configuration & Order ---")
    feature_config = artifacts["feature_config"]
    features = feature_config.get("features", [])
    if features == CANONICAL_FEATURES:
        print(f"[OK] Feature Configuration: 7 features match canonical ordering exactly:")
        for idx, f in enumerate(features, 1):
            print(f"     {idx}. {f}")
    else:
        print(f"[FAIL] Feature mismatch! Found: {features}, Expected: {CANONICAL_FEATURES}")
        errors.append("Feature config order mismatch")

    # 5. Validate Classes
    food_encoder = artifacts["food_encoder"]
    cooker_encoder = artifacts["cooker_encoder"]
    print(f"\n[OK] Food Encoder Classes ({len(food_encoder.classes_)}): {list(food_encoder.classes_)}")
    print(f"[OK] Cooker Encoder Classes ({len(cooker_encoder.classes_)}): {list(cooker_encoder.classes_)}")

    # 6. Execute Test Prediction
    print("\n--- 5. End-to-End Test Prediction ---")
    sample_input = {
        "time_min": 30.0,
        "solar_irradiance_W_m2": 500.0,
        "ambient_temp_C": 28.0,
        "absorber_temp_C": 75.0,
        "mass_kg": 0.5,
        "food_matrix": food_encoder.classes_[0],
        "cooker_type": cooker_encoder.classes_[0]
    }
    try:
        res = predict_food_temperature(sample_input)
        print("[OK] Test Prediction")
        print(f"     Inputs: {sample_input}")
        print(f"     Predicted Temperature: {res['predicted_food_temp_C']:.2f} °C")
        print(f"     Model: {res['model_name']} (Test R² = {res['test_r2']:.6f} / {res['test_r2_display']})")
        print(f"     Inference Latency: {res['inference_time_ms']:.2f} ms")
        if res["warnings"]:
            print(f"     Warnings: {res['warnings']}")
    except Exception as e:
        print(f"[FAIL] Test prediction failed: {e}")
        import traceback
        traceback.print_exc()
        errors.append(f"Inference error: {e}")

    # Summary
    print("\n" + "=" * 65)
    if not errors:
        print("ALL ARTIFACT VALIDATIONS PASSED! SYSTEM IS PRODUCTION READY.")
        print("=" * 65)
        sys.exit(0)
    else:
        print(f"VERIFICATION FAILED WITH {len(errors)} ERROR(S).")
        print("=" * 65)
        sys.exit(1)

if __name__ == "__main__":
    validate_all()
