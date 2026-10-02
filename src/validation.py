"""
Input validation module for Solar Cooking ML Prediction System.
Ensures numeric sanity, handles physical bounds, and checks against observed training ranges.
"""

import math
from typing import Dict, Any, List
try:
    from src.config import FEATURE_UI_CONFIG, NUMERICAL_FEATURES
except (ImportError, KeyError):
    from config import FEATURE_UI_CONFIG, NUMERICAL_FEATURES

class ValidationError(Exception):
    """Raised when an input fails critical validity requirements (non-numeric, NaN, inf)."""
    pass

def validate_numeric_input(feature_name: str, value: Any) -> float:
    """
    Validates that an input is a finite, non-NaN real number.
    Raises ValidationError with a clean message if invalid.
    """
    cfg = FEATURE_UI_CONFIG.get(feature_name, {})
    label = cfg.get("label", feature_name)

    if value is None:
        raise ValidationError(f"Missing input for '{label}'. Please provide a valid numerical value.")

    try:
        val = float(value)
    except (ValueError, TypeError):
        raise ValidationError(f"Invalid input for '{label}': '{value}'. Must be a valid numerical value.")

    if math.isnan(val) or not math.isfinite(val):
        raise ValidationError(f"Invalid input for '{label}': Value must be a finite, real number (not NaN or Infinity).")

    # Hard physical checks
    min_allowed = cfg.get("min_allowed", -float("inf"))
    max_allowed = cfg.get("max_allowed", float("inf"))
    if val < min_allowed:
        raise ValidationError(f"Constraint violation for '{label}': Value cannot be below {min_allowed} {cfg.get('unit', '')}.")
    if val > max_allowed:
        raise ValidationError(f"Constraint violation for '{label}': Value cannot exceed {max_allowed} {cfg.get('unit', '')}.")

    return val

def check_training_range_warnings(input_dict: Dict[str, Any], training_ranges: Dict[str, Dict[str, float]]) -> List[str]:
    """
    Checks if numerical inputs fall outside the minimum and maximum values
    observed in the training dataset. Returns non-blocking warning strings without emojis.
    """
    warnings = []
    for col in NUMERICAL_FEATURES:
        if col in input_dict and col in training_ranges:
            val = input_dict[col]
            obs_min = training_ranges[col]["min"]
            obs_max = training_ranges[col]["max"]
            cfg = FEATURE_UI_CONFIG.get(col, {})
            label = cfg.get("label", col)
            unit = cfg.get("unit", "")
            unit_str = f" {unit}" if unit else ""

            if val < obs_min:
                warnings.append(
                    f"Warning: {label} ({val}{unit_str}) is below the observed training minimum ({obs_min:.1f}{unit_str}). "
                    f"The prediction may be less reliable."
                )
            elif val > obs_max:
                warnings.append(
                    f"Warning: {label} ({val}{unit_str}) is above the observed training maximum ({obs_max:.1f}{unit_str}). "
                    f"The prediction may be less reliable."
                )
    return warnings

def validate_categorical_input(feature_name: str, value: Any, valid_classes: List[str]) -> str:
    """
    Validates that a categorical selection exists within the trained encoder's classes.
    """
    str_val = str(value)
    if str_val not in valid_classes:
        cfg = FEATURE_UI_CONFIG.get(feature_name, {})
        label = cfg.get("label", feature_name)
        raise ValidationError(
            f"Unknown category '{str_val}' for '{label}'. "
            f"Allowed categories are: {', '.join(valid_classes)}"
        )
    return str_val
