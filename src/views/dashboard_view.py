"""
Dashboard View — Overview of Solar Cooking ML Prediction System
Displays dynamic dataset metrics, production model highlights, coupled kinetics pipeline,
and academic project research modules.
Full Dark & Light mode compatible.
"""

import streamlit as st
import pandas as pd
try:
    from src.data_loader import get_dataset_kpis
    from src.config import PRODUCTION_EVALUATION
    from src.kinetics import get_all_kinetics_df
except (ImportError, KeyError):
    from data_loader import get_dataset_kpis
    from config import PRODUCTION_EVALUATION
    from kinetics import get_all_kinetics_df

def render_dashboard():
    st.markdown("<h1 class='main-title'>Solar Cooking & Nutrient Prediction System</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='subtitle-desc'>"
        "Academic EPICS research platform integrating machine learning surrogate models for transient solar food heating "
        "with first-order Arrhenius thermal kinetics to predict nutritional retention in heat-sensitive vitamins."
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
                    <span class='badge-pill badge-primary'>PRODUCTION ARCHITECTURE</span>
                    <h2 class='card-title' style='font-size: 1.6rem;'>Extra Trees Regressor + Arrhenius Kinetics</h2>
                    <p class='card-desc' style='margin: 0;'>
                        Coupled temperature surrogate (<code class='code-highlight'>R² = 99.74%</code>) with minute-by-minute numerical nutrient decay integration.
                    </p>
                </div>
                <div style='display: flex; gap: 20px; align-items: center;'>
                    <div style='text-align: right;'>
                        <div class='metric-label'>Validation R²</div>
                        <div class='metric-accent-green' style='font-size: 1.9rem;'>{PRODUCTION_EVALUATION['test_r2'] * 100:.2f}%</div>
                        <div class='metric-label' style='font-size: 0.75rem; text-transform: none;'>Test R²: {PRODUCTION_EVALUATION['test_r2']:.6f}</div>
                    </div>
                    <div style='text-align: right; border-left: 1px solid rgba(128, 128, 128, 0.25); padding-left: 20px;'>
                        <div class='metric-label'>Test MSE</div>
                        <div class='metric-accent-blue' style='font-size: 1.9rem;'>{PRODUCTION_EVALUATION['test_mse']:.3f}</div>
                        <div class='metric-label' style='font-size: 0.75rem; text-transform: none;'>Mean Squared Error</div>
                    </div>
                    <div style='text-align: right; border-left: 1px solid rgba(128, 128, 128, 0.25); padding-left: 20px;'>
                        <div class='metric-label'>Vitamins</div>
                        <div class='metric-accent-green' style='font-size: 1.9rem;'>6 Types</div>
                        <div class='metric-label' style='font-size: 0.75rem; text-transform: none;'>30 Kinetics Matrices</div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # Dynamic Dataset KPI Cards
    # ---------------------------------------------------------
    try:
        kpis = get_dataset_kpis()
    except Exception as e:
        st.error(f"Error reading canonical dataset statistics: {e}")
        return

    st.markdown("### Experimental Datasets Overview")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(label="Solar Observations", value=f"{kpis['total_rows']:,}", help="Total 1-minute resampled rows")
    with c2:
        st.metric(label="Experimental Cooking Runs", value=f"{kpis['unique_runs_count']}", help="Distinct physical runs (run_id)")
    with c3:
        st.metric(label="Nutrient Kinetics Entries", value="30 entries", help="Zenodo and literature Arrhenius parameters")
    with c4:
        st.metric(label="Vitamins Tracked", value="6 Vitamins", help="C, B1, B2, B5, B6, B9")

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        st.metric(label="Food Temp Mean", value=f"{kpis['target_mean']:.2f} °C")
    with c6:
        st.metric(label="Food Temp Range", value=f"{kpis['target_min']:.1f} – {kpis['target_max']:.1f} °C")
    with c7:
        st.metric(label="Food Matrices", value=f"{len(kpis['food_matrices'])} types")
    with c8:
        st.metric(label="Solar Cookers", value=f"{len(kpis['cooker_types'])} designs")

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # Coupled Pipeline Architecture Summary
    # ---------------------------------------------------------
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown(
            """
            <div class='glass-card'>
                <h4 class='card-title' style='margin-top: 0;'>Coupled Scientific Modeling Workflow</h4>
                <div style='font-size: 0.92rem; line-height: 1.6;'>
                    <div style='background-color: rgba(59, 130, 246, 0.12); padding: 10px; border-radius: 8px; margin-bottom: 8px;'>
                        <b>Step 1: Environmental & Thermodynamic Input Vector</b><br>
                        <span style='font-size: 0.85rem;'>Time, Solar Irradiance, Ambient Temp, Absorber Temp, Mass, Food Matrix, Cooker</span>
                    </div>
                    <div style='text-align: center; color: #3b82f6; font-weight: bold;'>↓</div>
                    <div style='background-color: rgba(16, 185, 129, 0.12); padding: 10px; border-radius: 8px; margin-bottom: 8px;'>
                        <b>Step 2: Extra Trees Food Temperature Trajectory</b><br>
                        <span style='font-size: 0.85rem;'>Generates transient heating curve <i>T<sub>food</sub>(t)</i> from <i>t = 0</i> to cooking duration (R²: 99.74%)</span>
                    </div>
                    <div style='text-align: center; color: #10b981; font-weight: bold;'>↓</div>
                    <div style='background-color: rgba(245, 158, 11, 0.12); padding: 10px; border-radius: 8px; margin-bottom: 8px;'>
                        <b>Step 3: First-Order Arrhenius Kinetics Integration</b><br>
                        <span style='font-size: 0.85rem;'><i>k(T) = A · exp(-E<sub>a</sub> / RT)</i>; trapezoidal integral evaluates <i>ln(C/C₀) = -∫ k(T) dt</i></span>
                    </div>
                    <div style='text-align: center; color: #f59e0b; font-weight: bold;'>↓</div>
                    <div style='background-color: rgba(139, 92, 246, 0.12); padding: 10px; border-radius: 8px;'>
                        <b>Step 4: Output Retention & Thermal Degradation</b><br>
                        <span style='font-size: 0.85rem;'>Delivers final retention %, degradation %, dual-axis plots, and minute-by-minute timeline</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_right:
        st.markdown(
            """
            <div class='glass-card'>
                <h4 class='card-title' style='margin-top: 0;'>Project Team Contributions</h4>
                <div style='font-size: 0.88rem; line-height: 1.6;'>
                    <table style='width: 100%; border-collapse: collapse;'>
                        <tr style='border-bottom: 1px solid rgba(128, 128, 128, 0.2);'>
                            <td style='padding: 6px 0;'><b>Om</b></td>
                            <td style='padding: 6px 0;'>Basic ML Temperature Regressors (Extra Trees, RF, LR, XGBoost) & evaluation tables</td>
                        </tr>
                        <tr style='border-bottom: 1px solid rgba(128, 128, 128, 0.2);'>
                            <td style='padding: 6px 0;'><b>Urvi</b></td>
                            <td style='padding: 6px 0;'>Vitamin kinetics data mining (Zenodo datasets), rate conversion (s⁻¹ to min⁻¹), Arrhenius fitting</td>
                        </tr>
                        <tr style='border-bottom: 1px solid rgba(128, 128, 128, 0.2);'>
                            <td style='padding: 6px 0;'><b>Bhavini</b></td>
                            <td style='padding: 6px 0;'>Merged ML Temperature + Vitamin Kinetics pipeline, dynamic trapezoidal numerical integration</td>
                        </tr>
                        <tr style='border-bottom: 1px solid rgba(128, 128, 128, 0.2);'>
                            <td style='padding: 6px 0;'><b>Govind</b></td>
                            <td style='padding: 6px 0;'>Validation on held-out runs, physical sensor drop anomaly detection, leakage checks</td>
                        </tr>
                        <tr style='border-bottom: 1px solid rgba(128, 128, 128, 0.2);'>
                            <td style='padding: 6px 0;'><b>Yash</b></td>
                            <td style='padding: 6px 0;'>Comparative results, dual-axis and retention vs time graphs, procedure documentation</td>
                        </tr>
                        <tr>
                            <td style='padding: 6px 0;'><b>Tariq</b></td>
                            <td style='padding: 6px 0;'>Interactive Streamlit web platform architecture, GUI design, Plotly visualizations, deployment</td>
                        </tr>
                    </table>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # Application Modules
    # ---------------------------------------------------------
    st.markdown("### Application Modules")
    m1, m2, m3, m4, m5 = st.columns(5)

    with m1:
        st.markdown(
            """
            <div class='module-card'>
                <span class='badge-pill badge-primary' style='width: fit-content; margin-bottom: 8px;'>INFERENCE</span>
                <h4 class='card-title' style='font-size: 1.1rem; margin-top: 0;'>Prediction</h4>
                <p class='card-desc' style='font-size: 0.85rem; line-height: 1.4; margin-bottom: 10px;'>
                    Simulate temperature and nutrient degradation with 3 interactive graphs.
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
                <span class='badge-pill badge-primary' style='width: fit-content; margin-bottom: 8px;'>KINETICS</span>
                <h4 class='card-title' style='font-size: 1.1rem; margin-top: 0;'>Nutrients</h4>
                <p class='card-desc' style='font-size: 0.85rem; line-height: 1.4; margin-bottom: 10px;'>
                    Explore 30 Arrhenius parameter sets, ln(k) vs 1/T plots, and half-life tables.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Open Nutrients", key="btn_dash_nutr", use_container_width=True):
            st.session_state.current_page = "Nutrient Kinetics"
            st.rerun()

    with m3:
        st.markdown(
            """
            <div class='module-card'>
                <span class='badge-pill badge-primary' style='width: fit-content; margin-bottom: 8px;'>ANALYTICS</span>
                <h4 class='card-title' style='font-size: 1.1rem; margin-top: 0;'>EDA Data</h4>
                <p class='card-desc' style='font-size: 0.85rem; line-height: 1.4; margin-bottom: 10px;'>
                    Inspect distributions, correlation matrices, and physical sensor traces.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Open EDA", key="btn_dash_eda", use_container_width=True):
            st.session_state.current_page = "EDA Dashboard"
            st.rerun()

    with m4:
        st.markdown(
            """
            <div class='module-card'>
                <span class='badge-pill badge-primary' style='width: fit-content; margin-bottom: 8px;'>BENCHMARKS</span>
                <h4 class='card-title' style='font-size: 1.1rem; margin-top: 0;'>Models</h4>
                <p class='card-desc' style='font-size: 0.85rem; line-height: 1.4; margin-bottom: 10px;'>
                    Compare Extra Trees, Random Forest, Linear Regression, and XGBoost.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Open Models", key="btn_dash_models", use_container_width=True):
            st.session_state.current_page = "Model Performance"
            st.rerun()

    with m5:
        st.markdown(
            """
            <div class='module-card'>
                <span class='badge-pill badge-primary' style='width: fit-content; margin-bottom: 8px;'>SCIENCE</span>
                <h4 class='card-title' style='font-size: 1.1rem; margin-top: 0;'>About</h4>
                <p class='card-desc' style='font-size: 0.85rem; line-height: 1.4; margin-bottom: 10px;'>
                    Thermodynamic equations, resampling details, and future PIML roadmap.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Open About", key="btn_dash_about", use_container_width=True):
            st.session_state.current_page = "About and Science"
            st.rerun()
