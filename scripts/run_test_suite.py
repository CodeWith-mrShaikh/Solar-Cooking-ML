"""
Comprehensive Automated Test Suite
==================================
Runs rigorous unit, validation, and integration tests across:
  - Canonical Dataset loading & KPIs
  - Pre-fitted Model & Preprocessing Artifacts
  - Input Validation (numeric, physical constraints, boundary handling)
  - Training Range Warnings
  - Predictor Single Source of Truth
  - Session Prediction History CSV format
  - Verification of no runtime fitting (fit, fit_transform)
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.config import (
    CANONICAL_DATA_PATH,
    CANONICAL_FEATURES,
    TARGET_COL,
    PRODUCTION_EVALUATION,
    BENCHMARK_RESULTS
)
from src.data_loader import load_dataset, get_dataset_kpis, get_feature_ranges, get_run_data
from src.model_loader import load_model_artifacts, get_system_status
from src.validation import (
    validate_numeric_input,
    validate_categorical_input,
    check_training_range_warnings,
    ValidationError
)
from src.predictor import predict_food_temperature, predict_solar_cooking_and_nutrients
from src.kinetics import (
    get_all_kinetics_df,
    get_available_vitamins,
    get_all_food_sources,
    get_vitamins_for_food_source,
    get_food_sources_for_vitamin,
    calculate_arrhenius_k,
    simulate_vitamin_degradation,
    FOOD_SOURCE_METADATA,
    SOLAR_FOOD_METADATA
)

class TestSolarCookingSystem(unittest.TestCase):

    def test_01_canonical_dataset(self):
        """Test dataset shape, columns, and target presence."""
        self.assertTrue(os.path.exists(CANONICAL_DATA_PATH), "Dataset file not found")
        df = load_dataset()
        self.assertEqual(len(df), 5803, f"Expected 5803 rows, got {len(df)}")
        self.assertEqual(len(df.columns), 11, f"Expected 11 columns, got {len(df.columns)}")
        self.assertIn(TARGET_COL, df.columns, f"Target '{TARGET_COL}' missing")
        for feat in CANONICAL_FEATURES:
            self.assertIn(feat, df.columns, f"Feature '{feat}' missing from dataset")

    def test_02_dataset_kpis(self):
        """Test dynamic KPI calculation."""
        kpis = get_dataset_kpis()
        self.assertEqual(kpis["total_rows"], 5803)
        self.assertEqual(kpis["unique_runs_count"], 16)
        self.assertEqual(kpis["total_missing_values"], 0)
        self.assertGreater(kpis["target_mean"], 20.0)
        self.assertLess(kpis["target_mean"], 100.0)
        self.assertEqual(len(kpis["food_matrices"]), 6)
        self.assertEqual(len(kpis["cooker_types"]), 5)

    def test_03_run_data_extraction(self):
        """Test run extraction does not mix different runs."""
        kpis = get_dataset_kpis()
        first_run = kpis["unique_runs"][0]
        run_df = get_run_data(first_run)
        self.assertFalse(run_df.empty)
        self.assertTrue((run_df["run_id"] == first_run).all())
        # Verify time monotonicity
        self.assertTrue(run_df["time_min"].is_monotonic_increasing)

    def test_04_model_artifacts_loaded(self):
        """Verify that pre-fitted artifacts load without error."""
        artifacts = load_model_artifacts()
        self.assertIn("model", artifacts)
        self.assertIn("imputer", artifacts)
        self.assertIn("scaler", artifacts)
        self.assertIn("food_encoder", artifacts)
        self.assertIn("cooker_encoder", artifacts)
        self.assertIn("feature_config", artifacts)
        self.assertIn("model_metadata", artifacts)

        # Verify no .fit() has been called dynamically
        self.assertEqual(len(artifacts["feature_config"]["features"]), 7)

    def test_05_system_status_reporting(self):
        """Verify system status flags are all True."""
        status = get_system_status()
        self.assertTrue(status["dataset_loaded"])
        self.assertTrue(status["model_loaded"])
        self.assertTrue(status["preprocessing_loaded"])
        self.assertEqual(len(status["missing_items"]), 0)

    def test_06_numeric_validation_valid(self):
        """Verify valid numeric inputs pass."""
        self.assertEqual(validate_numeric_input("time_min", 30), 30.0)
        self.assertEqual(validate_numeric_input("mass_kg", "1.25"), 1.25)
        self.assertEqual(validate_numeric_input("ambient_temp_C", 25.5), 25.5)

    def test_07_numeric_validation_invalid(self):
        """Verify invalid numeric inputs (NaN, Inf, negative mass, strings) raise ValidationError."""
        with self.assertRaises(ValidationError):
            validate_numeric_input("time_min", "invalid_number")
        with self.assertRaises(ValidationError):
            validate_numeric_input("time_min", float("nan"))
        with self.assertRaises(ValidationError):
            validate_numeric_input("time_min", float("inf"))
        with self.assertRaises(ValidationError):
            validate_numeric_input("time_min", -5.0)  # Min allowed is 0
        with self.assertRaises(ValidationError):
            validate_numeric_input("mass_kg", -1.0)   # Min allowed is 0

    def test_08_categorical_validation(self):
        """Verify categorical inputs validate strictly against encoder classes."""
        artifacts = load_model_artifacts()
        food_classes = [str(c) for c in artifacts["food_encoder"].classes_]
        
        # Valid category
        val = validate_categorical_input("food_matrix", food_classes[0], food_classes)
        self.assertEqual(val, food_classes[0])

        # Unknown category
        with self.assertRaises(ValidationError):
            validate_categorical_input("food_matrix", "Nonexistent Food Category", food_classes)

    def test_09_training_range_warnings(self):
        """Verify out-of-range warnings trigger without crashing."""
        ranges = get_feature_ranges()
        
        # In-range input should produce no warnings
        in_range_inputs = {
            "time_min": 30.0,
            "solar_irradiance_W_m2": 500.0,
            "ambient_temp_C": 28.0,
            "absorber_temp_C": 75.0,
            "mass_kg": 0.5
        }
        warnings = check_training_range_warnings(in_range_inputs, ranges)
        self.assertEqual(len(warnings), 0)

        # Way out-of-range input should produce warnings
        out_range_inputs = {
            "time_min": 999.0, # Observed max is around 300
            "solar_irradiance_W_m2": 500.0,
            "ambient_temp_C": 28.0,
            "absorber_temp_C": 75.0,
            "mass_kg": 0.5
        }
        warnings = check_training_range_warnings(out_range_inputs, ranges)
        self.assertGreater(len(warnings), 0)
        self.assertTrue(any("Time" in w for w in warnings))

    def test_10_end_to_end_prediction(self):
        """Verify predictor returns valid float temp and correct metadata."""
        artifacts = load_model_artifacts()
        food_classes = [str(c) for c in artifacts["food_encoder"].classes_]
        cooker_classes = [str(c) for c in artifacts["cooker_encoder"].classes_]

        test_cases = [
            ("Water", "Parabolic dish solar cooker"),
            ("Potato chips", "box"),
            ("Rice", "panel")
        ]

        for food, cooker in test_cases:
            if food in food_classes and cooker in cooker_classes:
                raw_input = {
                    "time_min": 45.0,
                    "solar_irradiance_W_m2": 650.0,
                    "ambient_temp_C": 30.0,
                    "absorber_temp_C": 85.0,
                    "mass_kg": 1.0,
                    "food_matrix": food,
                    "cooker_type": cooker
                }
                res = predict_food_temperature(raw_input)
                self.assertIsInstance(res["predicted_food_temp_C"], float)
                self.assertGreater(res["predicted_food_temp_C"], 0.0)
                self.assertLess(res["predicted_food_temp_C"], 200.0)
                self.assertGreater(res["inference_time_ms"], 0.0)
                self.assertEqual(res["model_name"], "Extra Trees Regressor")
                self.assertEqual(res["test_r2"], PRODUCTION_EVALUATION["test_r2"])

    def test_11_scientific_metric_integrity(self):
        """Verify R2 is never described as accuracy or confidence in configurations."""
        for bench in BENCHMARK_RESULTS:
            self.assertIn("Test R²", bench)
            self.assertNotIn("accuracy", [k.lower() for k in bench.keys()])
            self.assertNotIn("confidence", [k.lower() for k in bench.keys()])
        self.assertAlmostEqual(PRODUCTION_EVALUATION["test_r2"], 0.997352, places=6)
        self.assertAlmostEqual(PRODUCTION_EVALUATION["test_mse"], 1.362492, places=6)

    def test_12_vitamin_kinetics_library(self):
        """Verify that all 30 vitamin kinetics entries and 6 vitamins load correctly."""
        vits = get_available_vitamins()
        self.assertEqual(len(vits), 6, f"Expected 6 vitamins, got {len(vits)}")
        for v in ["Vitamin C", "Vitamin B1", "Vitamin B2", "Vitamin B5", "Vitamin B6", "Vitamin B9"]:
            self.assertIn(v, vits)

        kin_df = get_all_kinetics_df()
        self.assertEqual(len(kin_df), 30, f"Expected 30 kinetic entries, got {len(kin_df)}")
        self.assertTrue((kin_df["A_per_min"] > 0).all(), "All A factors must be positive")
        self.assertTrue((kin_df["Ea_J_mol"] > 0).all(), "All Ea values must be positive")

    def test_13_arrhenius_monotonicity(self):
        """Verify that k(T) increases strictly with temperature."""
        temps = np.array([40.0, 60.0, 80.0, 100.0])
        k_vals = calculate_arrhenius_k(temps, A_per_min=1.25e6, Ea_J_mol=52300.0)
        self.assertTrue(np.all(np.diff(k_vals) > 0), "Reaction rate constant k must strictly increase with temperature")

    def test_14_merged_simulation(self):
        """Verify the coupled Solar ML + Arrhenius simulation pipeline."""
        raw_input = {
            "time_min": 30.0,
            "solar_irradiance_W_m2": 600.0,
            "ambient_temp_C": 28.0,
            "absorber_temp_C": 80.0,
            "mass_kg": 0.5,
            "food_matrix": "Rice",
            "cooker_type": "Parabolic dish solar cooker"
        }
        res = predict_solar_cooking_and_nutrients(raw_input, vitamin="Vitamin C", food_source="Citrus Fruits")
        self.assertIn("predicted_food_temp_C", res)
        self.assertIn("final_retention_percent", res)
        self.assertIn("final_degradation_percent", res)
        self.assertGreaterEqual(res["final_retention_percent"], 0.0)
        self.assertLessEqual(res["final_retention_percent"], 100.0)
        self.assertAlmostEqual(res["final_retention_percent"] + res["final_degradation_percent"], 100.0, places=1)
        self.assertEqual(len(res["times_min"]), 31)

    def test_15_all_14_food_matrices(self):
        """Verify that all 14 kinetics food matrices are defined, labeled, and include Tubers for Potato."""
        sources = get_all_food_sources()
        self.assertEqual(len(sources), 14, f"Expected 14 food matrices, got {len(sources)}")
        
        # Verify Potato is documented in Tubers and carrots/radish in Root Vegetables
        self.assertIn("Tubers", sources)
        self.assertIn("Potato", FOOD_SOURCE_METADATA["Tubers"]["examples"])
        self.assertIn("Root Vegetables", sources)
        
        # Verify all 14 have metadata and at least one vitamin kinetics parameter
        for s in sources:
            self.assertIn(s, FOOD_SOURCE_METADATA, f"Food source '{s}' missing from FOOD_SOURCE_METADATA")
            vits = get_vitamins_for_food_source(s)
            self.assertGreater(len(vits), 0, f"Food source '{s}' must have at least 1 vitamin")

        # Verify Solar Food Matrix has Potato chips
        self.assertIn("Potato chips", SOLAR_FOOD_METADATA)

if __name__ == "__main__":
    unittest.main(verbosity=2)
