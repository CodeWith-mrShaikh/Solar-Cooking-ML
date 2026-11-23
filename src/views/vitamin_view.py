"""
Nutrient Kinetics View — Thermal Degradation & Arrhenius Analysis
Comprehensive interactive module exploring the 30 kinetic parameter entries across 6 heat-sensitive vitamins
(Vitamins C, B1, B2, B5, B6, B9) compiled from published peer-reviewed literature and experimental Zenodo datasets.
Full Dark & Light mode compatible.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

try:
    from src.kinetics import (
        get_all_kinetics_df,
        get_available_vitamins,
        get_food_sources_for_vitamin,
        calculate_arrhenius_k,
        R_GAS_CONSTANT,
        CELSIUS_TO_KELVIN
    )
except (ImportError, KeyError):
    from kinetics import (
        get_all_kinetics_df,
        get_available_vitamins,
        get_food_sources_for_vitamin,
        calculate_arrhenius_k,
        R_GAS_CONSTANT,
        CELSIUS_TO_KELVIN
    )


def render_vitamin_kinetics():
    st.markdown("<h1 class='main-title'>Nutrient Kinetics & Thermal Degradation</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='subtitle-desc'>"
        "Empirical first-order Arrhenius degradation kinetics across 6 heat-sensitive vitamins and 30 food matrices. "
        "Calculates thermal susceptibility, kinetic reaction rate constants k(T), and nutrient retention curves."
        "</p>",
        unsafe_allow_html=True
    )

    df_kinetics = get_all_kinetics_df()
    vitamins_list = get_available_vitamins()

    # ---------------------------------------------------------
    # Top KPI Metrics Cards
    # ---------------------------------------------------------
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

    with kpi_col1:
        st.markdown(
            """
            <div class='glass-card' style='text-align: center; padding: 18px;'>
                <div class='metric-label'>Vitamins Modeled</div>
                <div class='metric-accent-blue' style='font-size: 2.2rem;'>6</div>
                <div style='font-size: 0.8rem; opacity: 0.8;'>C, B1, B2, B5, B6, B9</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with kpi_col2:
        st.markdown(
            """
            <div class='glass-card' style='text-align: center; padding: 18px;'>
                <div class='metric-label'>Kinetics Entries</div>
                <div class='metric-accent-green' style='font-size: 2.2rem;'>30</div>
                <div style='font-size: 0.8rem; opacity: 0.8;'>Food & Buffer Matrices</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with kpi_col3:
        avg_ea = df_kinetics["Ea_kJ_mol"].mean()
        st.markdown(
            f"""
            <div class='glass-card' style='text-align: center; padding: 18px;'>
                <div class='metric-label'>Mean Activation Energy</div>
                <div class='metric-accent-blue' style='font-size: 2.2rem;'>{avg_ea:.1f} <span style='font-size: 1.1rem;'>kJ/mol</span></div>
                <div style='font-size: 0.8rem; opacity: 0.8;'>Arrhenius barrier (Ea)</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with kpi_col4:
        st.markdown(
            """
            <div class='glass-card' style='text-align: center; padding: 18px;'>
                <div class='metric-label'>Degradation Model</div>
                <div class='metric-accent-green' style='font-size: 1.8rem; line-height: 1.3;'>1st Order</div>
                <div style='font-size: 0.8rem; opacity: 0.8;'>ln(C/C₀) = -k·t</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # Interactive Tabs
    # ---------------------------------------------------------
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Kinetics Parameters Table",
        "Arrhenius Plots (ln k vs 1/T)",
        "Rate Constant Explorer k(T)",
        "Thermal Half-Life Comparison",
        "Scientific Formulations & References"
    ])

    # ---------------------------------------------------------
    # TAB 1: Kinetics Parameters Table
    # ---------------------------------------------------------
    with tab1:
        st.markdown("### Literature Kinetics Parameters Registry")
        st.markdown(
            "Filter and inspect pre-exponential frequency factors ($A$) and activation energies ($E_a$) "
            "compiled from literature (Om literature benchmarks) and experimental reported kinetics (Urvi Zenodo datasets)."
        )

        f_col1, f_col2 = st.columns([1, 2])
        with f_col1:
            selected_vit_filter = st.selectbox(
                "Filter by Vitamin:",
                options=["All Vitamins"] + vitamins_list,
                key="filter_vit_tab1"
            )

        filtered_df = df_kinetics.copy()
        if selected_vit_filter != "All Vitamins":
            filtered_df = filtered_df[filtered_df["vitamin"] == selected_vit_filter]

        display_cols = [
            "vitamin", "food_source", "A_per_min", "Ea_kJ_mol",
            "k_at_80C_per_min", "half_life_at_80C_min", "origin", "doi"
        ]
        col_rename = {
            "vitamin": "Vitamin",
            "food_source": "Food Matrix / Substrate",
            "A_per_min": "A (min⁻¹)",
            "Ea_kJ_mol": "Ea (kJ/mol)",
            "k_at_80C_per_min": "k at 80°C (min⁻¹)",
            "half_life_at_80C_min": "Half-Life at 80°C (min)",
            "origin": "Data Origin",
            "doi": "DOI / Identifier"
        }

        st.dataframe(
            filtered_df[display_cols].rename(columns=col_rename).style.format({
                "A (min⁻¹)": "{:.2e}",
                "Ea (kJ/mol)": "{:.2f}",
                "k at 80°C (min⁻¹)": "{:.4f}",
                "Half-Life at 80°C (min)": "{:.1f}"
            }),
            use_container_width=True
        )

        csv_bytes = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Kinetics Parameters (CSV)",
            data=csv_bytes,
            file_name="solar_cooking_vitamin_kinetics.csv",
            mime="text/csv"
        )

    # ---------------------------------------------------------
    # TAB 2: Arrhenius Plots (ln k vs 1000/T)
    # ---------------------------------------------------------
    with tab2:
        st.markdown("### Arrhenius Thermal Sensitivity Plots")
        st.markdown(
            r"The Arrhenius equation linearizes in logarithmic coordinates: "
            r"$$\ln(k) = \ln(A) - \frac{E_a}{R} \cdot \frac{1}{T_K}$$ "
            r"The slope of each curve is directly proportional to $-\frac{E_a}{R}$. Steeper slopes represent higher activation energies, "
            r"meaning degradation rates accelerate more drastically as food temperature increases."
        )

        p_col1, p_col2 = st.columns([1, 2])
        with p_col1:
            vit_choice = st.selectbox(
                "Select Vitamin to Analyze:",
                options=vitamins_list,
                key="arrh_vit_choice"
            )

        vit_df = df_kinetics[df_kinetics["vitamin"] == vit_choice]
        
        # Temperature range from 30 °C to 110 °C
        temp_C_range = np.linspace(30.0, 110.0, 81)
        temp_K_range = temp_C_range + CELSIUS_TO_KELVIN
        inv_T_1000 = 1000.0 / temp_K_range

        fig_arrh = go.Figure()
        custom_temps = [f"{t:.0f} °C" for t in temp_C_range]
        for _, row in vit_df.iterrows():
            k_vals = calculate_arrhenius_k(temp_C_range, row["A_per_min"], row["Ea_J_mol"])
            ln_k = np.log(np.maximum(k_vals, 1e-12))
            fig_arrh.add_trace(go.Scatter(
                x=inv_T_1000,
                y=ln_k,
                mode="lines",
                name=f"{row['food_source']} ({row['Ea_kJ_mol']:.1f} kJ/mol)",
                customdata=custom_temps,
                hovertemplate="1000/T: %{x:.2f} K⁻¹<br>ln(k): %{y:.2f}<br>T: %{customdata}<extra></extra>"
            ))

        fig_arrh.update_layout(
            title=f"Arrhenius Plot for {vit_choice} Across Food Matrices",
            xaxis_title="1000 / Temperature (1000 / K)",
            yaxis_title="ln(k) [k in min⁻¹]",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            hovermode="x unified",
            margin=dict(l=40, r=40, t=50, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=-0.35, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_arrh, use_container_width=True)

    # ---------------------------------------------------------
    # TAB 3: Rate Constant Explorer k(T)
    # ---------------------------------------------------------
    with tab3:
        st.markdown("### Reaction Rate Constant $k(T)$ vs Food Temperature")
        st.markdown(
            "Observe the non-linear exponential surge in reaction rate constants as solar cooking heating progresses. "
            "At ambient temperatures (25°C–40°C), $k(T)$ remains negligible, but expands by multiple orders of magnitude near water boiling."
        )

        r_col1, r_col2 = st.columns(2)
        with r_col1:
            sel_vit = st.selectbox("Select Vitamin:", options=vitamins_list, key="rate_vit")
        with r_col2:
            sources = get_food_sources_for_vitamin(sel_vit)
            sel_source = st.selectbox("Select Food Matrix:", options=sources, key="rate_source")

        sub_df = df_kinetics[(df_kinetics["vitamin"] == sel_vit) & (df_kinetics["food_source"] == sel_source)]
        entry = sub_df.iloc[0] if not sub_df.empty else df_kinetics.iloc[0]

        temps = np.linspace(25.0, 115.0, 91)
        k_rates = calculate_arrhenius_k(temps, entry["A_per_min"], entry["Ea_J_mol"])

        fig_k = go.Figure()
        fig_k.add_trace(go.Scatter(
            x=temps,
            y=k_rates,
            mode="lines+markers",
            name=f"k(T) [{entry['food_source']}]",
            line=dict(color="#3b82f6", width=3),
            hovertemplate="Food Temp: %{x:.1f} °C<br>k: %{y:.5f} min⁻¹"
        ))

        # Annotate 70°C and 90°C reference points
        k_70 = calculate_arrhenius_k(np.array([70.0]), entry["A_per_min"], entry["Ea_J_mol"])[0]
        k_90 = calculate_arrhenius_k(np.array([90.0]), entry["A_per_min"], entry["Ea_J_mol"])[0]

        fig_k.add_vline(x=70, line_dash="dash", line_color="orange", annotation_text=f"70°C (k={k_70:.4f})")
        fig_k.add_vline(x=90, line_dash="dash", line_color="red", annotation_text=f"90°C (k={k_90:.4f})")

        fig_k.update_layout(
            title=f"Reaction Rate Constant k(T) for {sel_vit} ({sel_source})",
            xaxis_title="Food Temperature (°C)",
            yaxis_title="Degradation Rate Constant k (min⁻¹)",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=40, r=40, t=50, b=40)
        )
        st.plotly_chart(fig_k, use_container_width=True)

    # ---------------------------------------------------------
    # TAB 4: Thermal Half-Life Comparison
    # ---------------------------------------------------------
    with tab4:
        st.markdown("### Thermal Half-Life Comparison at Typical Solar Temperatures")
        st.markdown(
            r"Under first-order kinetics, thermal half-life ($t_{1/2}$) represents the cooking duration required to degrade 50% of the initial vitamin: "
            r"$$t_{1/2} = \frac{\ln(2)}{k(T)}$$"
        )

        eval_temps = [60.0, 75.0, 85.0, 95.0]
        rows = []
        for v in vitamins_list:
            v_sub = df_kinetics[df_kinetics["vitamin"] == v]
            for _, r in v_sub.iterrows():
                row_dict = {
                    "Vitamin": v,
                    "Food Source": r["food_source"],
                    "Ea (kJ/mol)": r["Ea_kJ_mol"]
                }
                for temp in eval_temps:
                    k_val = calculate_arrhenius_k(np.array([temp]), r["A_per_min"], r["Ea_J_mol"])[0]
                    t_half = (np.log(2) / max(k_val, 1e-12))
                    row_dict[f"t½ at {int(temp)}°C (min)"] = round(t_half, 1) if t_half < 10000 else "> 10,000"
                rows.append(row_dict)

        hl_df = pd.DataFrame(rows)
        st.dataframe(hl_df, use_container_width=True)

    # ---------------------------------------------------------
    # TAB 5: Scientific Formulations & References
    # ---------------------------------------------------------
    with tab5:
        st.markdown("### Scientific Kinetics Formulation & Literature Citations")
        
        c_left, c_right = st.columns(2)
        with c_left:
            st.markdown(
                r"""
                #### Mathematical Formulation
                1. **First-Order Degradation Decay:**
                $$\frac{dC(t)}{dt} = -k(T(t)) \cdot C(t)$$

                2. **Transient Integral Form:**
                $$\ln\left(\frac{C(t)}{C_0}\right) = -\int_0^t k(T(\tau)) \, d\tau$$

                3. **Arrhenius Temperature Dependency:**
                $$k(T) = A \cdot \exp\left(-\frac{E_a}{R \cdot (T_C + 273.15)}\right)$$
                Where:
                * $C(t)$: Residual vitamin concentration at time $t$
                * $C_0$: Initial concentration ($100\%$ benchmark)
                * $A$: Pre-exponential collision frequency factor ($\text{min}^{-1}$)
                * $E_a$: Activation energy ($\text{J}\cdot\text{mol}^{-1}$)
                * $R = 8.314462618 \text{ J}/(\text{mol}\cdot\text{K})$: Universal gas constant
                * $T_C$: Food temperature in Celsius
                """
            )

        with c_right:
            st.markdown(
                """
                #### Literature Data Sources
                * **Zenodo Open Access Dataset 13938958 (2024):** High-precision kinetic degradation benchmarks for Vitamin C in strawberry nectar and McIlvaine buffer systems under thermal processing.
                * **Rosehip Nectar Study (DOI: 10.1590/1678-457X.11417):** Ascorbic acid thermal stability across pasteurization and heating ranges (70°C–95°C).
                * **Foods 2024 Chickpea Study (DOI: 10.3390/foods13121847):** Pyridoxine (Vitamin B6) kinetics in legume matrices during domestic thermal cooking.
                * **Hamm & Lund Food Science Benchmarks (1978):** Arrhenius activation parameters for B-complex vitamins (B1 thiamine, B2 riboflavin, B5 pantothenate, B9 folates) across whole grains, meats, dairy, and tubers.
                """
            )
