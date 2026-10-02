"""
Model loader module for Solar Cooking ML Prediction System.
Handles cached loading of the trained Extra Trees model and fitted preprocessing objects.
"""

import os
import json
import joblib
from typing import Dict, Any, Tuple

try:
    import streamlit as st
    cache_resource_decorator = st.cache_resource
except ImportError:
    def cache_resource_decorator(func):
        return func

from src.config import (
    MODEL_PATH,
    IMPUTER_PATH,
    SCALER_PATH,
    FOOD_ENCODER_PATH,
    COOKER_ENCODER_PATH,
    FEATURE_CONFIG_PATH,
    METADATA_PATH,
    CANONICAL_DATA_PATH
)

class ArtifactLoadError(Exception):
    """Custom exception raised when a required model artifact cannot be loaded."""
    pass

@cache_resource_decorator
def load_model_artifacts() -> Dict[str, Any]:
    """
    Loads all pre-fitted model and preprocessing artifacts into memory.
    Cached so loading occurs once across the application lifecycle.
    """
    artifacts = {}
    missing_files = []

    # 1. Feature configuration
    if os.path.exists(FEATURE_CONFIG_PATH):
        with open(FEATURE_CONFIG_PATH, "r", encoding="utf-8") as f:
            artifacts["feature_config"] = json.load(f)
    else:
        missing_files.append(FEATURE_CONFIG_PATH)

    # 2. Model metadata
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            artifacts["model_metadata"] = json.load(f)
    else:
        missing_files.append(METADATA_PATH)

    # 3. Label Encoders
    if os.path.exists(FOOD_ENCODER_PATH):
        artifacts["food_encoder"] = joblib.load(FOOD_ENCODER_PATH)
    else:
        missing_files.append(FOOD_ENCODER_PATH)

    if os.path.exists(COOKER_ENCODER_PATH):
        artifacts["cooker_encoder"] = joblib.load(COOKER_ENCODER_PATH)
    else:
        missing_files.append(COOKER_ENCODER_PATH)

    # 4. SimpleImputer
    if os.path.exists(IMPUTER_PATH):
        imputer = joblib.load(IMPUTER_PATH)
        # Cross-version compatibility patch for scikit-learn (handles _fill_dtype vs _fit_dtype)
        fit_dtype = getattr(imputer, "_fit_dtype", None) or getattr(imputer, "_fill_dtype", None)
        if fit_dtype is None and hasattr(imputer, "statistics_"):
            fit_dtype = imputer.statistics_.dtype
        if fit_dtype is None:
            import numpy as np
            fit_dtype = np.dtype("float64")
        imputer._fill_dtype = fit_dtype
        imputer._fit_dtype = fit_dtype
        artifacts["imputer"] = imputer
    else:
        missing_files.append(IMPUTER_PATH)

    # 5. StandardScaler
    if os.path.exists(SCALER_PATH):
        artifacts["scaler"] = joblib.load(SCALER_PATH)
    else:
        missing_files.append(SCALER_PATH)

    # 6. Production Model
    if os.path.exists(MODEL_PATH):
        artifacts["model"] = joblib.load(MODEL_PATH)
    else:
        missing_files.append(MODEL_PATH)

    if missing_files:
        raise ArtifactLoadError(
            f"Missing required model artifacts: {[os.path.basename(f) for f in missing_files]}. "
            f"Please run 'python scripts/export_model_artifacts.py' or download artifacts from Colab."
        )

    return artifacts

def get_system_status() -> Dict[str, Any]:
    """
    Checks availability of dataset, model, and preprocessing artifacts
    without raising exceptions. Used for sidebar system status badges.
    """
    status = {
        "dataset_loaded": os.path.exists(CANONICAL_DATA_PATH),
        "model_loaded": os.path.exists(MODEL_PATH),
        "preprocessing_loaded": (
            os.path.exists(IMPUTER_PATH) and
            os.path.exists(SCALER_PATH) and
            os.path.exists(FOOD_ENCODER_PATH) and
            os.path.exists(COOKER_ENCODER_PATH) and
            os.path.exists(FEATURE_CONFIG_PATH) and
            os.path.exists(METADATA_PATH)
        ),
        "missing_items": []
    }

    if not status["dataset_loaded"]:
        status["missing_items"].append("data/final_training_dataset.csv")
    if not status["model_loaded"]:
        status["missing_items"].append("models/extra_trees_model.joblib")
    if not os.path.exists(IMPUTER_PATH):
        status["missing_items"].append("models/imputer.joblib")
    if not os.path.exists(SCALER_PATH):
        status["missing_items"].append("models/scaler.joblib")
    if not os.path.exists(FOOD_ENCODER_PATH):
        status["missing_items"].append("models/food_matrix_encoder.joblib")
    if not os.path.exists(COOKER_ENCODER_PATH):
        status["missing_items"].append("models/cooker_type_encoder.joblib")
    if not os.path.exists(FEATURE_CONFIG_PATH):
        status["missing_items"].append("models/feature_config.json")
    if not os.path.exists(METADATA_PATH):
        status["missing_items"].append("models/model_metadata.json")

    return status
