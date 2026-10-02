"""
Configuration module for Solar Cooking ML Prediction System.
Centralizes paths, canonical feature definitions, and verified benchmark metrics.
"""

import os

# ---------------------------------------------------------------------------
# Directories and File Paths
# ---------------------------------------------------------------------------
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Canonical dataset path
CANONICAL_DATA_PATH = os.path.join(DATA_DIR, "final_training_dataset.csv")

# Model and Preprocessing Artifact Paths
MODEL_PATH = os.path.join(MODELS_DIR, "extra_trees_model.joblib")
IMPUTER_PATH = os.path.join(MODELS_DIR, "imputer.joblib")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")
FOOD_ENCODER_PATH = os.path.join(MODELS_DIR, "food_matrix_encoder.joblib")
COOKER_ENCODER_PATH = os.path.join(MODELS_DIR, "cooker_type_encoder.joblib")
FEATURE_CONFIG_PATH = os.path.join(MODELS_DIR, "feature_config.json")
METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.json")

# ---------------------------------------------------------------------------
# Canonical Feature Order and Target
# ---------------------------------------------------------------------------
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

NUMERICAL_FEATURES = [
    "time_min",
    "solar_irradiance_W_m2",
    "ambient_temp_C",
    "absorber_temp_C",
    "mass_kg"
]

CATEGORICAL_FEATURES = [
    "food_matrix",
    "cooker_type"
]

# Metadata columns dropped during modeling
METADATA_DROPPED_COLS = ["source_file", "run_id", "Unnamed: 10"]

# UI Display labels with units
FEATURE_UI_CONFIG = {
    "time_min": {
        "label": "Time",
        "unit": "min",
        "default": 30.0,
        "step": 1.0,
        "min_allowed": 0.0,
        "max_allowed": 600.0,
        "description": "Elapsed cooking time in minutes"
    },
    "solar_irradiance_W_m2": {
        "label": "Solar Irradiance",
        "unit": "W/m²",
        "default": 500.0,
        "step": 10.0,
        "min_allowed": 0.0,
        "max_allowed": 2000.0,
        "description": "Solar radiation intensity incident on cooker plane"
    },
    "ambient_temp_C": {
        "label": "Ambient Temperature",
        "unit": "°C",
        "default": 28.0,
        "step": 0.5,
        "min_allowed": -10.0,
        "max_allowed": 60.0,
        "description": "Surrounding ambient air temperature"
    },
    "absorber_temp_C": {
        "label": "Absorber Plate Temperature",
        "unit": "°C",
        "default": 75.0,
        "step": 0.5,
        "min_allowed": 0.0,
        "max_allowed": 250.0,
        "description": "Temperature of the solar thermal absorber plate"
    },
    "mass_kg": {
        "label": "Food Mass",
        "unit": "kg",
        "default": 0.50,
        "step": 0.05,
        "min_allowed": 0.0,
        "max_allowed": 20.0,
        "description": "Mass of food / water cooking medium"
    },
    "food_matrix": {
        "label": "Food Matrix",
        "unit": "",
        "description": "Physical food or surrogate medium being cooked"
    },
    "cooker_type": {
        "label": "Cooker Type",
        "unit": "",
        "description": "Solar cooker design geometry"
    }
}

# ---------------------------------------------------------------------------
# Colab Benchmark Evaluation Results (Source of Truth)
# ---------------------------------------------------------------------------
PRODUCTION_EVALUATION = {
    "model_name": "Extra Trees Regressor",
    "n_estimators": 100,
    "random_state": 42,
    "split": "80:20",
    "test_r2": 0.997352,
    "test_mse": 1.362492,
    "overfitting_diff": 0.002648
}

BENCHMARK_RESULTS = [
    # 80:20 Split
    {
        "Model": "Extra Trees",
        "Split": "80:20",
        "Test R²": 0.997352,
        "Test MSE": 1.362492,
        "Overfitting (Diff)": 0.002648
    },
    {
        "Model": "XGBoost",
        "Split": "80:20",
        "Test R²": 0.996554,
        "Test MSE": 1.772795,
        "Overfitting (Diff)": 0.002767
    },
    {
        "Model": "Random Forest",
        "Split": "80:20",
        "Test R²": 0.995648,
        "Test MSE": 2.239038,
        "Overfitting (Diff)": 0.003773
    },
    {
        "Model": "Gradient Boosting",
        "Split": "80:20",
        "Test R²": 0.980221,
        "Test MSE": 10.176398,
        "Overfitting (Diff)": 0.008601
    },
    {
        "Model": "Linear Regression",
        "Split": "80:20",
        "Test R²": 0.847437,
        "Test MSE": 78.493719,
        "Overfitting (Diff)": -0.025582
    },
    {
        "Model": "Ridge",
        "Split": "80:20",
        "Test R²": 0.847434,
        "Test MSE": 78.495313,
        "Overfitting (Diff)": -0.025579
    },
    # 70:30 Split
    {
        "Model": "Extra Trees",
        "Split": "70:30",
        "Test R²": 0.997067,
        "Test MSE": 1.505324,
        "Overfitting (Diff)": 0.002933
    },
    {
        "Model": "Random Forest",
        "Split": "70:30",
        "Test R²": 0.996744,
        "Test MSE": 1.671054,
        "Overfitting (Diff)": 0.002467
    },
    {
        "Model": "XGBoost",
        "Split": "70:30",
        "Test R²": 0.996314,
        "Test MSE": 1.891805,
        "Overfitting (Diff)": 0.003473
    },
    {
        "Model": "Gradient Boosting",
        "Split": "70:30",
        "Test R²": 0.984489,
        "Test MSE": 7.961237,
        "Overfitting (Diff)": 0.004442
    },
    {
        "Model": "Linear Regression",
        "Split": "70:30",
        "Test R²": 0.852089,
        "Test MSE": 75.916210,
        "Overfitting (Diff)": -0.036210
    },
    {
        "Model": "Ridge",
        "Split": "70:30",
        "Test R²": 0.852075,
        "Test MSE": 75.923429,
        "Overfitting (Diff)": -0.036196
    }
]
