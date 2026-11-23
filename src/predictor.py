"""
Predictor module for Solar Cooking ML Prediction System.
The SINGLE source of truth for preprocessing, temperature inference,
and coupled vitamin degradation simulation.
Strictly uses pre-fitted artifacts and never calls fit() or fit_transform().
"""

import time
import os
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

try:
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
    from src.kinetics import (
        simulate_vitamin_degradation,
        get_available_vitamins,
        get_food_sources_for_vitamin,
        get_kinetics_entry
    )
except (ImportError, KeyError):
    from config import (
        CANONICAL_FEATURES,
        NUMERICAL_FEATURES,
        CATEGORICAL_FEATURES,
        PRODUCTION_EVALUATION
    )
    from model_loader import load_model_artifacts
    from data_loader import get_feature_ranges
    from validation import (
        validate_numeric_input,
        validate_categorical_input,
        check_training_range_warnings,
        ValidationError
    )
    from kinetics import (
        simulate_vitamin_degradation,
        get_available_vitamins,
        get_food_sources_for_vitamin,
        get_kinetics_entry
    )


def _transform_and_predict(input_df: pd.DataFrame, artifacts: Dict[str, Any], feature_order: List[str]) -> np.ndarray:
    """
    Internal helper to transform an input dataframe through the pre-fitted pipeline
    and predict with the Extra Trees model. Handles all scikit-learn cross-version quirks.
    """
    model = artifacts["model"]
    imputer = artifacts["imputer"]
    scaler = artifacts["scaler"]

    # 1. SimpleImputer with bulletproof cross-version safety
    try:
        X_imp = imputer.transform(input_df)
    except Exception:
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
            if hasattr(imputer, "statistics_"):
                stat_map = dict(zip(feature_order, imputer.statistics_))
                filled_df = input_df.copy()
                for c in feature_order:
                    if filled_df[c].isna().any():
                        filled_df[c] = filled_df[c].fillna(stat_map.get(c, 0.0))
                X_imp = filled_df.values
            else:
                X_imp = input_df.values

    # 2. StandardScaler
    X_scaled = scaler.transform(X_imp)

    # 3. Model predict
    predictions = model.predict(X_scaled)
    return predictions


def predict_food_temperature(raw_inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Executes the exact 7-feature inference pipeline for a single state:
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

    # 5-7. Transform & Predict
    pred_temp = float(_transform_and_predict(input_df, artifacts, feature_order)[0])

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


def predict_temperature_trajectory(raw_inputs: Dict[str, Any], duration_min: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a transient food temperature profile at 1-minute intervals from t=0 to t=duration_min.
    Returns:
      times_min: np.ndarray [0, 1, 2, ..., duration_min]
      temperatures_C: np.ndarray of predicted food temperatures
    """
    duration = float(duration_min)
    if duration <= 0:
        raise ValidationError("Cooking duration must be greater than 0 minutes.")

    # Time grid at 1-minute resolution
    times = np.arange(0.0, duration, 1.0)
    if len(times) == 0 or times[-1] != duration:
        times = np.append(times, duration)

    artifacts = load_model_artifacts()
    food_encoder = artifacts["food_encoder"]
    cooker_encoder = artifacts["cooker_encoder"]
    feature_config = artifacts["feature_config"]
    feature_order = feature_config.get("features", CANONICAL_FEATURES)

    # Build vectorized DataFrame for all time steps
    rows = []
    for t in times:
        row = dict(raw_inputs)
        row["time_min"] = float(t)
        rows.append({col: row[col] for col in feature_order})

    batch_df = pd.DataFrame(rows, columns=feature_order)
    batch_df["food_matrix"] = food_encoder.transform(batch_df["food_matrix"].astype(str))
    batch_df["cooker_type"] = cooker_encoder.transform(batch_df["cooker_type"].astype(str))

    predicted_temps = _transform_and_predict(batch_df, artifacts, feature_order)
    return times, predicted_temps


def predict_solar_cooking_and_nutrients(
    raw_inputs: Dict[str, Any],
    vitamin: str = "Vitamin C",
    food_source: Optional[str] = None
) -> Dict[str, Any]:
    """
    End-to-end merged simulation combining Solar ML Temperature prediction
    with the Arrhenius Vitamin Degradation kinetics engine.
    Pipeline:
      Solar/Cooking Inputs -> ML Extra Trees -> Transient Temperature Curve T(t)
                           -> First-order Kinetics ln(C/C0) = -∫ k(T) dt
                           -> Retention(t) & Degradation(t)
    """
    start_time = time.perf_counter()

    # 1. Validate inputs and predict point temperature at specified time
    point_result = predict_food_temperature(raw_inputs)
    duration = float(raw_inputs.get("time_min", 30.0))

    # 2. Predict transient temperature trajectory from t=0 to t=duration
    times, temps = predict_temperature_trajectory(raw_inputs, duration)

    # 3. Simulate vitamin degradation kinetics
    solar_matrix = raw_inputs.get("food_matrix", "")
    summary, timeline_df = simulate_vitamin_degradation(
        times_min=times,
        temperatures_C=temps,
        vitamin=vitamin,
        food_source=food_source,
        solar_food_matrix=solar_matrix
    )

    latency_ms = (time.perf_counter() - start_time) * 1000.0

    return {
        "point_result": point_result,
        "predicted_food_temp_C": point_result["predicted_food_temp_C"],
        "final_retention_percent": summary["final_retention_percent"],
        "final_degradation_percent": summary["final_degradation_percent"],
        "final_ideal_retention_percent": summary["final_ideal_retention_percent"],
        "final_ideal_degradation_percent": summary["final_ideal_degradation_percent"],
        "avg_retention_percent": summary["avg_retention_percent"],
        "peak_degradation_rate_percent_per_min": summary["peak_degradation_rate_percent_per_min"],
        "peak_degradation_rate_time_min": summary["peak_degradation_rate_time_min"],
        "times_min": times,
        "temperatures_C": temps,
        "summary": summary,
        "timeline_df": timeline_df,
        "warnings": point_result["warnings"],
        "inference_time_ms": round(latency_ms, 2)
    }
