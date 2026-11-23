"""
Prediction View — Real-time Food Temperature & Vitamin Retention Inference
Couples the trained Extra Trees production model with the Arrhenius kinetics engine.
Simulates minute-by-minute transient food temperature curves and dynamic nutrient retention.
Renders all 3 required analytical Plotly graphs, summary KPI metrics, minute-wise timeline,
and session history with CSV export.
Full Dark & Light mode compatible.
"""

import streamlit as st
import pandas as pd
import numpy as np
import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any

try:
    from src.config import FEATURE_UI_CONFIG, PRODUCTION_EVALUATION
    from src.model_loader import load_model_artifacts, ArtifactLoadError
    from src.predictor import predict_solar_cooking_and_nutrients
    from src.kinetics import (
        get_available_vitamins,
        get_food_sources_for_vitamin,
        get_kinetics_entry
    )
    from src.validation import ValidationError
except (ImportError, KeyError):
    from config import FEATURE_UI_CONFIG, PRODUCTION_EVALUATION
    from model_loader import load_model_artifacts, ArtifactLoadError
    from predictor import predict_solar_cooking_and_nutrients
    from kinetics import (
        get_available_vitamins,
        get_food_sources_for_vitamin,
        get_kinetics_entry
    )
    from validation import ValidationError


def render_prediction():
    st.markdown("<h1 class='main-title'>Solar Cooking & Nutrient Retention Predictor</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='subtitle-desc'>"
        "Predicts transient food temperature and dynamic thermal degradation for heat-sensitive vitamins. "
        "Couples the Extra Trees ML model (99.74% R²) with first-order Arrhenius numerical kinetics."
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

    # Available vitamins and food sources
    vitamins_list = get_available_vitamins()

    # Initialize session state for prediction history and form inputs
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []

    defaults = {
        "time_min": FEATURE_UI_CONFIG["time_min"]["default"],
        "solar_irradiance_W_m2": FEATURE_UI_CONFIG["solar_irradiance_W_m2"]["default"],
        "ambient_temp_C": FEATURE_UI_CONFIG["ambient_temp_C"]["default"],
        "absorber_temp_C": FEATURE_UI_CONFIG["absorber_temp_C"]["default"],
        "mass_kg": FEATURE_UI_CONFIG["mass_kg"]["default"],
        "food_matrix": "Rice" if "Rice" in food_options else food_options[0],
        "cooker_type": "Thermal storage solar cooker" if "Thermal storage solar cooker" in cooker_options else cooker_options[0],
        "vitamin": "Vitamin C",
        "vitamin_source": "Citrus Fruits"
    }

    # Reset trigger handler
    if st.session_state.get("trigger_reset", False):
        for k, v in defaults.items():
            st.session_state[f"input_{k}"] = v
        st.session_state.trigger_reset = False

    # ---------------------------------------------------------
    # Three-Column Input Layout (Solar Conditions, Thermal Setup, Nutrient Target)
    # ---------------------------------------------------------
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.markdown("#### 1. Solar & Ambient Conditions")
        
        time_min = st.number_input(
            label="Cooking Duration / Time (min)",
            min_value=FEATURE_UI_CONFIG["time_min"]["min_allowed"],
            max_value=FEATURE_UI_CONFIG["time_min"]["max_allowed"],
            value=st.session_state.get("input_time_min", defaults["time_min"]),
            step=FEATURE_UI_CONFIG["time_min"]["step"],
            help="Total elapsed solar cooking duration (minutes)",
            key="input_time_min"
        )

        solar_irradiance = st.number_input(
            label="Solar Irradiance (W/m²)",
            min_value=FEATURE_UI_CONFIG["solar_irradiance_W_m2"]["min_allowed"],
            max_value=FEATURE_UI_CONFIG["solar_irradiance_W_m2"]["max_allowed"],
            value=st.session_state.get("input_solar_irradiance_W_m2", defaults["solar_irradiance_W_m2"]),
            step=FEATURE_UI_CONFIG["solar_irradiance_W_m2"]["step"],
            help="Incident solar radiation intensity on the collector aperture plane (W/m²)",
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
        st.markdown("#### 2. Thermal & Setup Parameters")

        absorber_temp = st.number_input(
            label="Absorber Plate Temperature (°C)",
            min_value=FEATURE_UI_CONFIG["absorber_temp_C"]["min_allowed"],
            max_value=FEATURE_UI_CONFIG["absorber_temp_C"]["max_allowed"],
            value=st.session_state.get("input_absorber_temp_C", defaults["absorber_temp_C"]),
            step=FEATURE_UI_CONFIG["absorber_temp_C"]["step"],
            help="Temperature of the heat collector or absorber plate surface (°C)",
            key="input_absorber_temp_C"
        )

        mass_kg = st.number_input(
            label="Food Mass (kg)",
            min_value=FEATURE_UI_CONFIG["mass_kg"]["min_allowed"],
            max_value=FEATURE_UI_CONFIG["mass_kg"]["max_allowed"],
            value=st.session_state.get("input_mass_kg", defaults["mass_kg"]),
            step=FEATURE_UI_CONFIG["mass_kg"]["step"],
            help="Net mass of the food substance or cooking medium (kg)",
            key="input_mass_kg"
        )

        curr_food = st.session_state.get("input_food_matrix", defaults["food_matrix"])
        food_idx = food_options.index(curr_food) if curr_food in food_options else 0
        food_matrix = st.selectbox(
            label="Solar Food Matrix",
            options=food_options,
            index=food_idx,
            help="Food substrate or surrogate medium tested in physical solar cooker runs",
            key="input_food_matrix"
        )

        curr_cooker = st.session_state.get("input_cooker_type", defaults["cooker_type"])
        cooker_idx = cooker_options.index(curr_cooker) if curr_cooker in cooker_options else 0
        cooker_type = st.selectbox(
            label="Solar Cooker Type",
            options=cooker_options,
            index=cooker_idx,
            help="Solar collector geometry and thermal storage design class",
            key="input_cooker_type"
        )

    with col3:
        st.markdown("#### 3. Nutrient Degradation Target")

        curr_vit = st.session_state.get("input_vitamin", defaults["vitamin"])
        vit_idx = vitamins_list.index(curr_vit) if curr_vit in vitamins_list else 0
        selected_vitamin = st.selectbox(
            label="Heat-Sensitive Vitamin",
            options=vitamins_list,
            index=vit_idx,
            help="Select the specific vitamin target for kinetic degradation simulation",
            key="input_vitamin"
        )

        available_sources = get_food_sources_for_vitamin(selected_vitamin)
        if "input_vitamin_source" in st.session_state and st.session_state["input_vitamin_source"] not in available_sources:
            st.session_state["input_vitamin_source"] = available_sources[0]

        curr_src = st.session_state.get("input_vitamin_source", available_sources[0])
        src_idx = available_sources.index(curr_src) if curr_src in available_sources else 0
        selected_source = st.selectbox(
            label="Kinetics Food Matrix / Substrate",
            options=available_sources,
            index=src_idx,
            help="Biological food matrix or buffer system from which Arrhenius parameters (A, Ea) are derived",
            key="input_vitamin_source"
        )

        # Show kinetic constants preview
        k_entry = get_kinetics_entry(selected_vitamin, selected_source)
        st.markdown(
            f"""
            <div style='background-color: rgba(128, 128, 128, 0.08); border-radius: 8px; padding: 12px; margin-top: 15px; font-size: 0.85rem;'>
                <b>Kinetic Parameters:</b><br>
                • Frequency Factor <i>A</i>: <code class='code-highlight'>{k_entry['A_per_min']:.2e} min⁻¹</code><br>
                • Activation Energy <i>Ea</i>: <code class='code-highlight'>{k_entry['Ea_J_mol']/1000.0:.2f} kJ/mol</code><br>
                • Literature Source: <span style='opacity: 0.85;'>{k_entry['origin']}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # Action Buttons: Predict and Reset
    # ---------------------------------------------------------
    btn_col1, btn_col2, btn_spacer = st.columns([2, 1, 3])

    with btn_col1:
        predict_clicked = st.button("Predict Temperature & Vitamin Retention", type="primary", use_container_width=True)
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
            with st.spinner("Executing Solar ML Temperature inference and numerical vitamin degradation integration..."):
                sim_result = predict_solar_cooking_and_nutrients(
                    raw_inputs=raw_inputs,
                    vitamin=selected_vitamin,
                    food_source=selected_source
                )

            summary = sim_result["summary"]
            timeline_df = sim_result["timeline_df"]
            pred_temp = sim_result["predicted_food_temp_C"]
            retention = sim_result["final_retention_percent"]
            degradation = sim_result["final_degradation_percent"]
            latency = sim_result["inference_time_ms"]

            # Display any non-blocking scientific warnings
            warnings_list = list(sim_result["warnings"])
            if summary["num_temps_outside_calibration"] > 0:
                warnings_list.append(
                    f"Notice: {summary['num_temps_outside_calibration']} predicted temperature points fell outside "
                    f"the literature calibration range ({summary['calib_temp_min_C']}°C – {summary['calib_temp_max_C']}°C). "
                    f"Kinetic Arrhenius rates were extrapolated."
                )
            if not summary["food_matrix_match"]:
                warnings_list.append(
                    f"Notice: Selected solar food medium ('{food_matrix}') differs from kinetics biological source "
                    f"('{selected_source}'). Retention estimates should be interpreted as indicative comparative benchmarks."
                )

            for warn in warnings_list:
                st.warning(warn)

            # ---------------------------------------------------------
            # Primary KPI Result Cards
            # ---------------------------------------------------------
            kpi_c1, kpi_c2, kpi_c3 = st.columns(3)

            with kpi_c1:
                st.markdown(
                    f"""
                    <div class='result-card' style='padding: 20px;'>
                        <div class='metric-label' style='font-size: 0.85rem;'>Predicted Food Temperature</div>
                        <div class='temp-value' style='font-size: 2.8rem;'>{pred_temp:.2f} °C</div>
                        <div style='font-size: 0.85rem; opacity: 0.85;'>
                            At t = {time_min:.0f} min (ML R²: 99.74%)
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with kpi_c2:
                ret_color = "#10b981" if retention >= 70.0 else ("#f59e0b" if retention >= 40.0 else "#ef4444")
                st.markdown(
                    f"""
                    <div class='result-card' style='padding: 20px; border-color: {ret_color} !important;'>
                        <div class='metric-label' style='font-size: 0.85rem;'>Vitamin Retention</div>
                        <div class='temp-value' style='font-size: 2.8rem; color: {ret_color} !important;'>{retention:.2f}%</div>
                        <div style='font-size: 0.85rem; opacity: 0.85;'>
                            {selected_vitamin} in {selected_source}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with kpi_c3:
                deg_color = "#ef4444" if degradation >= 50.0 else ("#f59e0b" if degradation >= 25.0 else "#10b981")
                st.markdown(
                    f"""
                    <div class='result-card' style='padding: 20px; border-color: {deg_color} !important;'>
                        <div class='metric-label' style='font-size: 0.85rem;'>Thermal Degradation</div>
                        <div class='temp-value' style='font-size: 2.8rem; color: {deg_color} !important;'>{degradation:.2f}%</div>
                        <div style='font-size: 0.85rem; opacity: 0.85;'>
                            Total Loss via Arrhenius Decay
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # Secondary KPI Ribbon
            st.markdown(
                f"""
                <div class='glass-card' style='padding: 14px 22px; display: flex; justify-content: space-around; flex-wrap: wrap; gap: 15px; margin-top: 15px;'>
                    <div><span class='metric-label'>Maximum Temp:</span> <b>{summary['max_food_temp_C']} °C</b></div>
                    <div><span class='metric-label'>Average Temp:</span> <b>{summary['avg_food_temp_C']} °C</b></div>
                    <div><span class='metric-label'>Peak Degradation Rate:</span> <b>{summary['peak_degradation_rate_percent_per_min']:.4f} %/min</b> (at t = {summary['peak_degradation_rate_time_min']:.0f} min)</div>
                    <div><span class='metric-label'>Ideal Isothermal Retention:</span> <b>{summary['final_ideal_retention_percent']:.2f}%</b></div>
                    <div><span class='metric-label'>Inference Latency:</span> <span class='metric-accent-blue'>{latency:.1f} ms</span></div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # ---------------------------------------------------------
            # The 3 Required Interactive Plotly Visualizations
            # ---------------------------------------------------------
            st.markdown("<br>### Dynamic Process Visualizations", unsafe_allow_html=True)

            plot_tab1, plot_tab2, plot_tab3 = st.tabs([
                "1. Temperature vs Time Profile",
                "2. Vitamin Retention Profile (Real vs Ideal)",
                "3. Combined Dual-Axis Result"
            ])

            # GRAPH 1: Temperature vs Time Profile
            with plot_tab1:
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(
                    x=timeline_df["time_min"],
                    y=timeline_df["predicted_food_temp_C"],
                    mode="lines+markers",
                    name="Predicted Food Temp (°C)",
                    line=dict(color="#3b82f6", width=3),
                    marker=dict(size=5),
                    hovertemplate="Time: %{x} min<br>Food Temp: %{y:.2f} °C"
                ))
                fig1.add_trace(go.Scatter(
                    x=[timeline_df["time_min"].iloc[0], timeline_df["time_min"].iloc[-1]],
                    y=[absorber_temp, absorber_temp],
                    mode="lines",
                    name="Absorber Temp Reference",
                    line=dict(color="#ef4444", width=1.5, dash="dash"),
                    hovertemplate="Absorber Temp: %{y:.1f} °C"
                ))
                fig1.add_trace(go.Scatter(
                    x=[timeline_df["time_min"].iloc[0], timeline_df["time_min"].iloc[-1]],
                    y=[ambient_temp, ambient_temp],
                    mode="lines",
                    name="Ambient Temp Reference",
                    line=dict(color="#10b981", width=1.5, dash="dot"),
                    hovertemplate="Ambient Temp: %{y:.1f} °C"
                ))
                fig1.update_layout(
                    title="Transient Food Temperature Curve During Solar Cooking",
                    xaxis_title="Cooking Time (minutes)",
                    yaxis_title="Food Temperature (°C)",
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    hovermode="x unified",
                    margin=dict(l=40, r=40, t=50, b=40)
                )
                st.plotly_chart(fig1, use_container_width=True)

            # GRAPH 2: Vitamin Retention Profile (Real vs Ideal)
            with plot_tab2:
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=timeline_df["time_min"],
                    y=timeline_df["real_retention_percent"],
                    mode="lines",
                    name="Real Process (Dynamic Changing Temperature)",
                    line=dict(color="#10b981", width=3),
                    hovertemplate="Time: %{x} min<br>Real Retention: %{y:.2f}%"
                ))
                fig2.add_trace(go.Scatter(
                    x=timeline_df["time_min"],
                    y=timeline_df["ideal_retention_percent"],
                    mode="lines",
                    name=f"Ideal Isothermal Benchmark (Constant {pred_temp:.1f} °C)",
                    line=dict(color="#f59e0b", width=2, dash="dash"),
                    hovertemplate="Time: %{x} min<br>Ideal Retention: %{y:.2f}%"
                ))
                fig2.add_trace(go.Scatter(
                    x=timeline_df["time_min"],
                    y=timeline_df["real_degradation_percent"],
                    mode="lines",
                    name="Real Thermal Degradation (%)",
                    line=dict(color="#ef4444", width=2, dash="dot"),
                    hovertemplate="Time: %{x} min<br>Degradation: %{y:.2f}%"
                ))
                fig2.update_layout(
                    title=f"{selected_vitamin} Retention & Degradation Profile ({selected_source})",
                    xaxis_title="Cooking Time (minutes)",
                    yaxis_title="Percentage (%)",
                    yaxis=dict(range=[0, 105]),
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    hovermode="x unified",
                    margin=dict(l=40, r=40, t=50, b=40)
                )
                st.plotly_chart(fig2, use_container_width=True)

            # GRAPH 3: Combined Dual-Axis Result
            with plot_tab3:
                fig3 = make_subplots(specs=[[{"secondary_y": True}]])
                fig3.add_trace(
                    go.Scatter(
                        x=timeline_df["time_min"],
                        y=timeline_df["predicted_food_temp_C"],
                        mode="lines",
                        name="Food Temp (°C)",
                        line=dict(color="#3b82f6", width=3)
                    ),
                    secondary_y=False
                )
                fig3.add_trace(
                    go.Scatter(
                        x=timeline_df["time_min"],
                        y=timeline_df["real_retention_percent"],
                        mode="lines",
                        name=f"{selected_vitamin} Retention (%)",
                        line=dict(color="#10b981", width=3)
                    ),
                    secondary_y=True
                )
                fig3.update_layout(
                    title=f"Coupled Thermal Heating & Nutrient Retention: {selected_vitamin} in {selected_source}",
                    xaxis_title="Cooking Time (minutes)",
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    hovermode="x unified",
                    margin=dict(l=40, r=40, t=50, b=40),
                    legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5)
                )
                fig3.update_yaxes(title_text="Food Temperature (°C)", secondary_y=False)
                fig3.update_yaxes(title_text="Vitamin Retention (%)", range=[0, 105], secondary_y=True)
                st.plotly_chart(fig3, use_container_width=True)

            # ---------------------------------------------------------
            # Minute-by-Minute Degradation Timeline Table
            # ---------------------------------------------------------
            st.markdown("<br>### Minute-by-Minute Degradation Timeline", unsafe_allow_html=True)
            with st.expander("View Full Simulation Data Table (t = 0 to t = duration)", expanded=False):
                st.dataframe(
                    timeline_df.rename(columns={
                        "time_min": "Time (min)",
                        "predicted_food_temp_C": "Food Temp (°C)",
                        "k_real_per_min": "k(T) [min⁻¹]",
                        "real_retention_percent": "Retention (%)",
                        "real_degradation_percent": "Degradation (%)",
                        "ideal_retention_percent": "Ideal Retention (%)",
                        "ideal_degradation_percent": "Ideal Degradation (%)",
                        "temp_outside_calibration": "Extrapolated Flag"
                    }),
                    use_container_width=True
                )
                csv_bytes = timeline_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Timeline CSV",
                    data=csv_bytes,
                    file_name=f"timeline_{selected_vitamin.replace(' ', '_')}_{time_min:.0f}min.csv",
                    mime="text/csv"
                )

            # Record in session-level history
            history_entry = {
                "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Duration (min)": time_min,
                "Solar (W/m²)": solar_irradiance,
                "Food Matrix": food_matrix,
                "Cooker": cooker_type,
                "Vitamin": selected_vitamin,
                "Kinetics Matrix": selected_source,
                "Food Temp (°C)": pred_temp,
                "Retention (%)": retention,
                "Degradation (%)": degradation,
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
                label="Download History CSV",
                data=csv_data,
                file_name=f"solar_nutrient_history_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        with h_col2:
            if st.button("Clear History", use_container_width=True):
                st.session_state.prediction_history = []
                st.rerun()
    else:
        st.info("No predictions recorded in this session yet. Run a prediction above to build history.")
