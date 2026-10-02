"""
Dashboard View — Overview of Solar Cooking ML Prediction System
Displays dynamic dataset metrics, production model highlights, and pipeline summary.
Full Dark & Light mode compatible.
"""

import streamlit as st
import pandas as pd
from src.data_loader import get_dataset_kpis
from src.config import PRODUCTION_EVALUATION

def render_dashboard():
    st.markdown("<h1 class='main-title'>Solar Cooking ML Dashboard</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='subtitle-desc'>"
        "Real-time operational dashboard for predicting food temperature from environmental, "
        "thermal, temporal, food matrix, and cooker parameters."
        "</p>",
        unsafe_allow_html=True
    )

    # ---------------------------------------------------------
    # Production Model Banner Card
    # ---------------------------------------------------------
    st.markdown(
        f"""
        <div class='highlight-card'>
            <div style='display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 15px;'>
                <div>
                    <span class='badge-pill badge-primary'>PRODUCTION MODEL</span>
                    <h2 class='card-title' style='font-size: 1.6rem;'>{PRODUCTION_EVALUATION['model_name']}</h2>
                    <p class='card-desc' style='margin: 0;'>
                        Optimized with <code class='code-highlight'>n_estimators=100</code>, <code class='code-highlight'>random_state=42</code> on 80:20 evaluation split.
                    </p>
                </div>
                <div style='display: flex; gap: 20px; align-items: center;'>
                    <div style='text-align: right;'>
                        <div class='metric-label'>Validation R²</div>
                        <div class='metric-accent-green' style='font-size: 1.9rem;'>{PRODUCTION_EVALUATION['test_r2'] * 100:.2f}%</div>
                        <div class='metric-label' style='font-size: 0.75rem; text-transform: none;'>R² Score: {PRODUCTION_EVALUATION['test_r2']:.6f}</div>
                    </div>
                    <div style='text-align: right; border-left: 1px solid rgba(128, 128, 128, 0.25); padding-left: 20px;'>
                        <div class='metric-label'>Test MSE</div>
                        <div class='metric-accent-blue' style='font-size: 1.9rem;'>{PRODUCTION_EVALUATION['test_mse']:.3f}</div>
                        <div class='metric-label' style='font-size: 0.75rem; text-transform: none;'>Mean Squared Error</div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # Dynamic Dataset KPI Cards (Computed from actual data)
    # ---------------------------------------------------------
    try:
        kpis = get_dataset_kpis()
    except Exception as e:
        st.error(f"Error reading canonical dataset statistics: {e}")
        return

    st.markdown("### Canonical Training Dataset Metrics")
    st.markdown(
        "<span class='card-desc' style='font-size: 0.9rem;'>"
        "Computed directly from <code class='code-highlight'>data/final_training_dataset.csv</code> (1-minute resampled experimental observations)."
        "</span>",
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(label="Dataset Observations", value=f"{kpis['total_rows']:,}", help="Total 1-minute resampled rows")
    with c2:
        st.metric(label="Feature Columns", value=f"{kpis['total_columns']}", help="Total tabular columns")
    with c3:
        st.metric(label="Experimental Runs", value=f"{kpis['unique_runs_count']}", help="Distinct solar cooking runs (run_id)")
    with c4:
        st.metric(label="Missing Values", value=f"{kpis['total_missing_values']}", help="Total NaN values in feature columns")

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        st.metric(label="Target Mean (Food Temp)", value=f"{kpis['target_mean']:.2f} °C")
    with c6:
        st.metric(label="Target Min (Food Temp)", value=f"{kpis['target_min']:.2f} °C")
    with c7:
        st.metric(label="Target Max (Food Temp)", value=f"{kpis['target_max']:.2f} °C")
    with c8:
        st.metric(label="Target Std Dev", value=f"{kpis['target_std']:.2f} °C")

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # System Architecture and Methodology Summary
    # ---------------------------------------------------------
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown(
            """
            <div class='glass-card'>
                <h4 class='card-title' style='margin-top: 0;'>Modeling Pipeline Architecture</h4>
                <ul class='card-desc' style='font-size: 0.92rem; line-height: 1.6; padding-left: 20px; margin: 0;'>
                    <li><b>Resampling:</b> Raw heterogeneous measurements resampled to synchronized 1-minute intervals per <code class='code-highlight'>run_id</code>.</li>
                    <li><b>Metadata Propagation:</b> Categorical and constant attributes forward- and backward-filled across each run.</li>
                    <li><b>Thermodynamic Predictors:</b> Incident solar irradiance, absorber plate temperature, and ambient temperature.</li>
                    <li><b>Categorical Factors:</b> Food matrix (physical substrate) and Cooker geometry type encoded via fitted LabelEncoders.</li>
                    <li><b>Inference Rigor:</b> Strict separation of training and inference; zero runtime re-fitting.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_right:
        st.markdown(
            f"""
            <div class='glass-card'>
                <h4 class='card-title' style='margin-top: 0;'>Model Feature Inputs (7 Canonical)</h4>
                <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 0.9rem;'>
                    <div><code class='code-highlight'>time_min</code> (min)</div>
                    <div><code class='code-highlight'>solar_irradiance_W_m2</code> (W/m²)</div>
                    <div><code class='code-highlight'>ambient_temp_C</code> (°C)</div>
                    <div><code class='code-highlight'>absorber_temp_C</code> (°C)</div>
                    <div><code class='code-highlight'>mass_kg</code> (kg)</div>
                    <div><code class='code-highlight'>food_matrix</code> (categorical)</div>
                    <div><code class='code-highlight'>cooker_type</code> (categorical)</div>
                    <div class='metric-accent-green'><code>food_temp_C</code> (Target)</div>
                </div>
                <div class='card-desc' style='margin-top: 15px; font-size: 0.85rem;'>
                    <b>Identified Classes:</b> {len(kpis['food_matrices'])} Food Matrices and {len(kpis['cooker_types'])} Cooker Types.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # Navigation Guide / Interactive Module Cards
    # ---------------------------------------------------------
    st.markdown("### Application Modules")
    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.markdown(
            """
            <div class='module-card'>
                <span class='badge-pill badge-primary' style='width: fit-content; margin-bottom: 8px;'>INFERENCE</span>
                <h4 class='card-title' style='font-size: 1.15rem; margin-top: 0;'>Prediction</h4>
                <p class='card-desc' style='font-size: 0.88rem; line-height: 1.5; margin-bottom: 12px;'>
                    Run real-time inference with training-range validation, custom food/cooker inputs, and history logging.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Open Prediction", key="btn_dash_pred", use_container_width=True):
            st.session_state.current_page = "Prediction"
            st.rerun()

    with m2:
        st.markdown(
            """
            <div class='module-card'>
                <span class='badge-pill badge-primary' style='width: fit-content; margin-bottom: 8px;'>ANALYTICS</span>
                <h4 class='card-title' style='font-size: 1.15rem; margin-top: 0;'>EDA Dashboard</h4>
                <p class='card-desc' style='font-size: 0.88rem; line-height: 1.5; margin-bottom: 12px;'>
                    Interactive Plotly charts of target distribution, correlation heatmap, food/cooker dynamics, and time-series.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Open EDA Dashboard", key="btn_dash_eda", use_container_width=True):
            st.session_state.current_page = "EDA Dashboard"
            st.rerun()

    with m3:
        st.markdown(
            """
            <div class='module-card'>
                <span class='badge-pill badge-primary' style='width: fit-content; margin-bottom: 8px;'>BENCHMARKS</span>
                <h4 class='card-title' style='font-size: 1.15rem; margin-top: 0;'>Model Performance</h4>
                <p class='card-desc' style='font-size: 0.88rem; line-height: 1.5; margin-bottom: 12px;'>
                    Side-by-side benchmark comparison of 6 regressors across 70:30 and 80:20 splits and Feature Importance.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Open Performance", key="btn_dash_perf", use_container_width=True):
            st.session_state.current_page = "Model Performance"
            st.rerun()

    with m4:
        st.markdown(
            """
            <div class='module-card'>
                <span class='badge-pill badge-primary' style='width: fit-content; margin-bottom: 8px;'>THEORY</span>
                <h4 class='card-title' style='font-size: 1.15rem; margin-top: 0;'>About and Science</h4>
                <p class='card-desc' style='font-size: 0.88rem; line-height: 1.5; margin-bottom: 12px;'>
                    Project methodology, thermodynamic formulations, data science split notes, and academic citations.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Open Documentation", key="btn_dash_about", use_container_width=True):
            st.session_state.current_page = "About and Science"
            st.rerun()
