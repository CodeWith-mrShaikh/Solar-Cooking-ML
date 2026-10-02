"""
About View — Project Documentation & Thermodynamic Background
Details the BTech/EPICS project origin, thermodynamic principles,
1-minute resampling methodology, and engineering architecture.
Full Dark & Light mode compatible.
"""

import streamlit as st

def render_about():
    st.markdown("<h1 class='main-title'>About Solar Cooking ML System</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='subtitle-desc'>"
        "Engineering background, thermodynamic modeling principles, data resampling methodology, "
        "and architectural documentation for the Solar Cooking Machine Learning Prediction System."
        "</p>",
        unsafe_allow_html=True
    )

    c1, c2 = st.columns([3, 2])

    with c1:
        st.markdown(
            """
            ### Project Motivation and Context
            Solar cooking represents an eco-friendly, carbon-neutral cooking modality utilizing concentrated or direct 
            solar thermal radiation to heat food and boil water. However, solar cooking dynamics are highly transient:
            * **Weather Dependency:** Solar irradiance ($G$) fluctuates due to cloud transit and atmospheric attenuation.
            * **Thermal Storage Lag:** Heat transfer across the glass cover, reflector geometry, absorber plate, and cooking vessel introduces non-linear thermal inertia.
            * **Food Matrix Heterogeneity:** Different food matrices (e.g., pure water, starchy potato chips, parboiled rice) exhibit disparate specific heat capacities ($C_p$) and phase-change behaviors.

            This academic project develops a **machine-learning surrogate model** to predict instantaneous food temperature 
            ($T_{food}$) accurately from easily observable environmental, temporal, and thermodynamic sensor inputs.
            """
        )

        st.markdown(
            r"""
            ### 1-Minute Resampling Methodology
            Experimental sensor data from various physical campaigns often feature irregular sample intervals, sensor drops, 
            or mismatched logging frequencies. The data preprocessing pipeline enforces standard 1-minute temporal resolution:
            1. **Grouping by Run:** Data is segmented into distinct physical experiments via `run_id`.
            2. **Temporal Grid:** Complete minute-level time indexes ($\Delta t = 1\text{ min}$) are constructed from $\min(t)$ to $\max(t)$.
            3. **Metadata Propagation:** Constant physical parameters (`run_id`, `food_matrix`, `cooker_type`, `mass_kg`, `source_file`) are forward- and backward-filled across each run.
            4. **Sensor Interpolation:** Numeric variables (`solar_irradiance_W_m2`, `ambient_temp_C`, `absorber_temp_C`, `food_temp_C`) are linearly interpolated.
            5. **Cleaned Training Matrix:** Rows with unresolved target values are pruned, resulting in the canonical 5,803-row dataset.
            """
        )

    with c2:
        st.markdown(
            """
            <div class='glass-card'>
                <h4 class='card-title' style='margin-top: 0; font-size: 1.15rem;'>Core Thermodynamic Variables</h4>
                <div class='card-desc' style='font-size: 0.9rem; line-height: 1.6;'>
                    <p style='margin-bottom: 8px;'><b>1. <i>t</i> (Elapsed Time, min):</b> Captures transient cumulative heat accumulation in the cooking vessel.</p>
                    <p style='margin-bottom: 8px;'><b>2. <i>G</i> (Solar Irradiance, W/m²):</b> Total incident solar flux providing primary energy input.</p>
                    <p style='margin-bottom: 8px;'><b>3. <i>T<sub>amb</sub></i> (Ambient Temperature, °C):</b> Drives convective and radiative thermal loss back to the atmosphere.</p>
                    <p style='margin-bottom: 8px;'><b>4. <i>T<sub>abs</sub></i> (Absorber Plate Temp, °C):</b> Intermediate thermal reservoir conducting heat into the vessel base.</p>
                    <p style='margin-bottom: 8px;'><b>5. <i>m</i> (Mass, kg):</b> Determines the net thermal capacitance (<i>C = m &middot; C<sub>p</sub></i>) required for temperature rise.</p>
                    <p style='margin-bottom: 8px;'><b>6. Food Matrix:</b> Dictates thermal conductivity (<i>k</i>) and effective heat capacity.</p>
                    <p style='margin-bottom: 0;'><b>7. Cooker Type:</b> Dictates optical concentration ratio (<i>C<sub>R</sub></i>) and overall heat loss coefficient (<i>U<sub>L</sub></i>).</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<hr>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # Engineering Stack & Integrity Principles
    # ---------------------------------------------------------
    st.markdown("### Production Engineering and Architectural Integrity")
    e1, e2, e3 = st.columns(3)

    with e1:
        st.markdown(
            """
            **Separation of Concerns**
            * Training, parameter tuning, and benchmark cross-validation take place offline.
            * The production GUI is purely an inference consumer.
            * Zero training or re-fitting occurs at runtime (`fit()` is strictly forbidden).
            """
        )

    with e2:
        st.markdown(
            """
            **Exact Feature Order**
            * Canonical order defined in `models/feature_config.json`.
            * Inputs transformed via pre-fitted `LabelEncoder`, `SimpleImputer`, and `StandardScaler`.
            * Column names preserved to avoid scikit-learn warnings.
            """
        )

    with e3:
        st.markdown(
            """
            **Honest Scientific Reporting**
            * $R^2$ is never mislabeled as "accuracy" or "confidence".
            * Out-of-range inputs trigger non-blocking reliability warnings.
            * Prediction history is logged for review and exportable as CSV.
            """
        )
