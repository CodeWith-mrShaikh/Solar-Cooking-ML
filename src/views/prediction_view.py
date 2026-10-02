"""
Prediction View — Real-time Food Temperature Inference
Provides 2-column input UI, dynamic dropdowns from fitted encoders,
training-range warnings, instant result card, and prediction history with CSV download.
Full Dark & Light mode compatible.
"""

import streamlit as st
import pandas as pd
import datetime
from typing import Dict, Any

from src.config import FEATURE_UI_CONFIG, PRODUCTION_EVALUATION
from src.model_loader import load_model_artifacts, ArtifactLoadError
from src.predictor import predict_food_temperature
from src.validation import ValidationError

def render_prediction():
    st.markdown("<h1 class='main-title'>Solar Cooking Food Temperature Predictor</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='subtitle-desc'>"
        "Enter environmental and experimental parameters below to predict food temperature in real time "
        "using the trained Extra Trees production model."
        "</p>",
        unsafe_allow_html=True
    )

    # ---------------------------------------------------------
    # Load Pre-fitted Artifacts
    # ---------------------------------------------------------
    try:
        artifacts = load_model_artifacts()
        food_encoder = artifacts["food_encoder"]
        cooker_encoder = artifacts["cooker_encoder"]
        food_options = sorted([str(c) for c in food_encoder.classes_])
        cooker_options = sorted([str(c) for c in cooker_encoder.classes_])
    except ArtifactLoadError as e:
        st.error(f"Model Artifacts Not Ready: {e}")
        st.info("To generate required artifacts, execute: python scripts/export_model_artifacts.py")
        return
    except Exception as e:
        st.error(f"Failed to load model artifacts: {e}")
        return

    # Initialize session state for prediction history and form inputs
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []

    defaults = {
        "time_min": FEATURE_UI_CONFIG["time_min"]["default"],
        "solar_irradiance_W_m2": FEATURE_UI_CONFIG["solar_irradiance_W_m2"]["default"],
        "ambient_temp_C": FEATURE_UI_CONFIG["ambient_temp_C"]["default"],
        "absorber_temp_C": FEATURE_UI_CONFIG["absorber_temp_C"]["default"],
        "mass_kg": FEATURE_UI_CONFIG["mass_kg"]["default"],
        "food_matrix": food_options[0] if food_options else "",
        "cooker_type": cooker_options[0] if cooker_options else ""
    }

    # Reset trigger handler
    if st.session_state.get("trigger_reset", False):
        for k, v in defaults.items():
            st.session_state[f"input_{k}"] = v
        st.session_state.trigger_reset = False

    # ---------------------------------------------------------
    # Two-Column Input Layout
    # ---------------------------------------------------------
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Solar and Environmental Conditions")
        
        time_min = st.number_input(
            label="Time (min)",
            min_value=FEATURE_UI_CONFIG["time_min"]["min_allowed"],
            max_value=FEATURE_UI_CONFIG["time_min"]["max_allowed"],
            value=st.session_state.get("input_time_min", defaults["time_min"]),
            step=FEATURE_UI_CONFIG["time_min"]["step"],
            help="Cooking duration elapsed since start of experiment (minutes)",
            key="input_time_min"
        )

        solar_irradiance = st.number_input(
            label="Solar Irradiance (W/m²)",
            min_value=FEATURE_UI_CONFIG["solar_irradiance_W_m2"]["min_allowed"],
            max_value=FEATURE_UI_CONFIG["solar_irradiance_W_m2"]["max_allowed"],
            value=st.session_state.get("input_solar_irradiance_W_m2", defaults["solar_irradiance_W_m2"]),
            step=FEATURE_UI_CONFIG["solar_irradiance_W_m2"]["step"],
            help="Solar radiation intensity incident on the aperture plane (W/m²)",
            key="input_solar_irradiance_W_m2"
        )

        ambient_temp = st.number_input(
            label="Ambient Temperature (°C)",
            min_value=FEATURE_UI_CONFIG["ambient_temp_C"]["min_allowed"],
            max_value=FEATURE_UI_CONFIG["ambient_temp_C"]["max_allowed"],
            value=st.session_state.get("input_ambient_temp_C", defaults["ambient_temp_C"]),
            step=FEATURE_UI_CONFIG["ambient_temp_C"]["step"],
            help="Surrounding dry-bulb atmospheric temperature (°C)",
            key="input_ambient_temp_C"
        )

    with col2:
        st.markdown("#### Thermal and Setup Parameters")

        absorber_temp = st.number_input(
            label="Absorber Plate Temperature (°C)",
            min_value=FEATURE_UI_CONFIG["absorber_temp_C"]["min_allowed"],
            max_value=FEATURE_UI_CONFIG["absorber_temp_C"]["max_allowed"],
            value=st.session_state.get("input_absorber_temp_C", defaults["absorber_temp_C"]),
            step=FEATURE_UI_CONFIG["absorber_temp_C"]["step"],
            help="Temperature of the heat collector/absorber plate (°C)",
            key="input_absorber_temp_C"
        )

        mass_kg = st.number_input(
            label="Food Mass (kg)",
            min_value=FEATURE_UI_CONFIG["mass_kg"]["min_allowed"],
            max_value=FEATURE_UI_CONFIG["mass_kg"]["max_allowed"],
            value=st.session_state.get("input_mass_kg", defaults["mass_kg"]),
            step=FEATURE_UI_CONFIG["mass_kg"]["step"],
            help="Net mass of the food or water cooking medium (kg)",
            key="input_mass_kg"
        )

        # Dropdowns derived exclusively from fitted LabelEncoders
        curr_food = st.session_state.get("input_food_matrix", defaults["food_matrix"])
        food_idx = food_options.index(curr_food) if curr_food in food_options else 0
        food_matrix = st.selectbox(
            label="Food Matrix",
            options=food_options,
            index=food_idx,
            help="Food substrate or surrogate medium derived from fitted encoder",
            key="input_food_matrix"
        )

        curr_cooker = st.session_state.get("input_cooker_type", defaults["cooker_type"])
        cooker_idx = cooker_options.index(curr_cooker) if curr_cooker in cooker_options else 0
        cooker_type = st.selectbox(
            label="Cooker Type",
            options=cooker_options,
            index=cooker_idx,
            help="Solar cooker design class derived from fitted encoder",
            key="input_cooker_type"
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # Action Buttons: Predict and Reset
    # ---------------------------------------------------------
    btn_col1, btn_col2, btn_spacer = st.columns([2, 1, 3])

    with btn_col1:
        predict_clicked = st.button("Predict Food Temperature", type="primary", use_container_width=True)
    with btn_col2:
        if st.button("Reset Defaults", use_container_width=True):
            st.session_state.trigger_reset = True
            st.rerun()

    # ---------------------------------------------------------
    # Prediction Execution & Results
    # ---------------------------------------------------------
    if predict_clicked:
        raw_inputs = {
            "time_min": time_min,
            "solar_irradiance_W_m2": solar_irradiance,
            "ambient_temp_C": ambient_temp,
            "absorber_temp_C": absorber_temp,
            "mass_kg": mass_kg,
            "food_matrix": food_matrix,
            "cooker_type": cooker_type
        }

        try:
            result = predict_food_temperature(raw_inputs)
            
            # Display any non-blocking training range warnings
            if result["warnings"]:
                for warn in result["warnings"]:
                    st.warning(warn)

            # Display Theme-Adaptive Result Card
            pred_temp = result["predicted_food_temp_C"]
            latency = result["inference_time_ms"]
            r2_display = result["test_r2_display"]

            st.markdown(
                f"""
                <div class='result-card'>
                    <div class='metric-label' style='font-size: 0.9rem; margin-bottom: 5px;'>
                        Predicted Food Temperature
                    </div>
                    <div class='temp-value'>
                        {pred_temp:.2f} °C
                    </div>
                    <div class='card-desc' style='margin-top: 15px; display: flex; justify-content: center; gap: 30px; font-size: 0.95rem; flex-wrap: wrap;'>
                        <div><b>Model:</b> {result['model_name']}</div>
                        <div><b>Validation R²:</b> <span class='metric-accent-green'>{r2_display}</span></div>
                        <div><b>Inference Latency:</b> <span class='metric-accent-blue'>{latency:.1f} ms</span></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Record in session-level history
            history_entry = {
                "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Time (min)": time_min,
                "Solar (W/m²)": solar_irradiance,
                "Ambient (°C)": ambient_temp,
                "Absorber (°C)": absorber_temp,
                "Mass (kg)": mass_kg,
                "Food Matrix": food_matrix,
                "Cooker Type": cooker_type,
                "Predicted Temp (°C)": pred_temp,
                "Latency (ms)": latency
            }
            st.session_state.prediction_history.insert(0, history_entry)

        except ValidationError as ve:
            st.error(f"Input Validation Error: {ve}")
        except Exception as ex:
            st.error(f"Prediction Engine Error: {ex}")

    st.markdown("<br><hr>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # Prediction History Section
    # ---------------------------------------------------------
    st.markdown("### Session Prediction History")
    
    if st.session_state.prediction_history:
        hist_df = pd.DataFrame(st.session_state.prediction_history)
        st.dataframe(hist_df, use_container_width=True)

        h_col1, h_col2, _ = st.columns([1, 1, 4])
        with h_col1:
            csv_data = hist_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name=f"solar_cooking_predictions_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        with h_col2:
            if st.button("Clear History", use_container_width=True):
                st.session_state.prediction_history = []
                st.rerun()
    else:
        st.info("No predictions recorded in this session yet. Run a prediction above to build history.")
