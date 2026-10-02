"""
Predictor module for Solar Cooking ML Prediction System.
The SINGLE source of truth for preprocessing and model inference.
Strictly uses pre-fitted artifacts and never calls fit() or fit_transform().
"""

import time
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple

from src.config import (
    CANONICAL_FEATURES,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    PRODUCTION_EVALUATION
)
from src.model_loader import load_model_artifacts
from src.data_loader import get_feature_ranges
from src.validation import (
    validate_numeric_input,
    validate_categorical_input,
    check_training_range_warnings,
    ValidationError
)

def predict_food_temperature(raw_inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Executes the exact 7-feature inference pipeline:
      1. Validate all inputs
      2. Check training range warnings
      3. Create DataFrame with exact feature names and canonical order
      4. Encode categorical variables using fitted LabelEncoders
      5. Transform numerical data using fitted SimpleImputer
      6. Transform using fitted StandardScaler
      7. Predict food_temp_C using fitted ExtraTreesRegressor
      8. Measure inference latency

    Returns a comprehensive result dictionary.
    """
    start_time = time.perf_counter()

    # Load pre-fitted artifacts (cached)
    artifacts = load_model_artifacts()
    model = artifacts["model"]
    imputer = artifacts["imputer"]
    scaler = artifacts["scaler"]
    food_encoder = artifacts["food_encoder"]
    cooker_encoder = artifacts["cooker_encoder"]
    feature_config = artifacts["feature_config"]

    feature_order = feature_config.get("features", CANONICAL_FEATURES)

    # 1. Validation
    validated_inputs = {}
    for col in NUMERICAL_FEATURES:
        if col not in raw_inputs:
            raise ValidationError(f"Missing required numerical parameter: '{col}'")
        validated_inputs[col] = validate_numeric_input(col, raw_inputs[col])

    # Validate categories against encoder classes
    food_classes = [str(c) for c in food_encoder.classes_]
    cooker_classes = [str(c) for c in cooker_encoder.classes_]

    if "food_matrix" not in raw_inputs:
        raise ValidationError("Missing required parameter: 'food_matrix'")
    validated_inputs["food_matrix"] = validate_categorical_input("food_matrix", raw_inputs["food_matrix"], food_classes)

    if "cooker_type" not in raw_inputs:
        raise ValidationError("Missing required parameter: 'cooker_type'")
    validated_inputs["cooker_type"] = validate_categorical_input("cooker_type", raw_inputs["cooker_type"], cooker_classes)

    # 2. Training range warnings (non-blocking)
    training_ranges = get_feature_ranges()
    range_warnings = check_training_range_warnings(validated_inputs, training_ranges)

    # 3. Create DataFrame with exact column names and canonical ordering
    row_data = {col: validated_inputs[col] for col in feature_order}
    input_df = pd.DataFrame([row_data], columns=feature_order)

    # 4. Encode categorical variables using pre-fitted encoders
    input_df["food_matrix"] = food_encoder.transform(input_df["food_matrix"].astype(str))
    input_df["cooker_type"] = cooker_encoder.transform(input_df["cooker_type"].astype(str))

    # 5. Transform via SimpleImputer
    # Passing DataFrame with matching columns avoids sklearn feature name mismatch warnings
    try:
        X_imp = imputer.transform(input_df)
    except Exception:
        # Cross-version patch for scikit-learn internal attribute differences
        fit_dtype = getattr(imputer, "_fit_dtype", None) or getattr(imputer, "_fill_dtype", None)
        if fit_dtype is None and hasattr(imputer, "statistics_"):
            fit_dtype = imputer.statistics_.dtype
        if fit_dtype is None:
            fit_dtype = np.dtype("float64")
        imputer._fill_dtype = fit_dtype
        imputer._fit_dtype = fit_dtype
        try:
            X_imp = imputer.transform(input_df)
        except Exception:
            # Bulletproof fallback: use imputer statistics_ to fill any missing values
            if hasattr(imputer, "statistics_"):
                stat_map = dict(zip(feature_order, imputer.statistics_))
                filled_df = input_df.copy()
                for c in feature_order:
                    if filled_df[c].isna().any():
                        filled_df[c] = filled_df[c].fillna(stat_map.get(c, 0.0))
                X_imp = filled_df.values
            else:
                X_imp = input_df.values

    # 6. Transform via StandardScaler
    X_scaled = scaler.transform(X_imp)

    # 7. Predict via ExtraTreesRegressor
    pred_temp = float(model.predict(X_scaled)[0])

    # Measure latency
    latency_ms = (time.perf_counter() - start_time) * 1000.0

    return {
        "predicted_food_temp_C": round(pred_temp, 2),
        "inference_time_ms": round(latency_ms, 2),
        "model_name": "Extra Trees Regressor",
        "test_r2": PRODUCTION_EVALUATION["test_r2"],
        "test_r2_display": f"{PRODUCTION_EVALUATION['test_r2'] * 100:.2f}% R²",
        "test_mse": PRODUCTION_EVALUATION["test_mse"],
        "input_values": validated_inputs,
        "warnings": range_warnings
    }
