"""
EDA View — Exploratory Data Analysis Dashboard
Provides interactive Plotly visualizations for dataset distribution,
correlation analysis, categorical dynamics, and run-level time series.
Full Dark & Light mode compatible.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
try:
    from src.data_loader import load_dataset, get_dataset_kpis, get_run_data
    from src.config import NUMERICAL_FEATURES, TARGET_COL
except (ImportError, KeyError):
    from data_loader import load_dataset, get_dataset_kpis, get_run_data
    from config import NUMERICAL_FEATURES, TARGET_COL

def render_eda():
    st.markdown("<h1 class='main-title'>Exploratory Data Analysis (EDA)</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='subtitle-desc'>"
        "Interactive exploratory analysis computed from the canonical 1-minute resampled experimental dataset. "
        "All visual analytics are derived directly from actual measurements without fabrication."
        "</p>",
        unsafe_allow_html=True
    )

    try:
        df = load_dataset()
        kpis = get_dataset_kpis()
    except Exception as e:
        st.error(f"Failed to load dataset for EDA: {e}")
        return

    # Tabs for organized analytics
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Dataset Overview",
        "Target and Feature Distributions",
        "Correlation Heatmap",
        "Run-Level Time Series",
        "Food and Cooker Analysis"
    ])

    # ---------------------------------------------------------
    # TAB 1: Dataset Overview
    # ---------------------------------------------------------
    with tab1:
        st.markdown("### Canonical Dataset Overview")
        st.markdown(
            f"The training dataset contains **{len(df):,}** resampled minute observations across **{df['run_id'].nunique()}** "
            f"distinct experimental runs and **{len(df.columns)}** columns."
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Numerical Features:**")
            st.write(NUMERICAL_FEATURES)
        with c2:
            st.markdown("**Categorical Features:**")
            st.write(["food_matrix", "cooker_type"])
        with c3:
            st.markdown("**Target Variable:**")
            st.write([TARGET_COL])

        st.markdown("#### Sample Dataset Records")
        preview_cols = [c for c in df.columns if c != "Unnamed: 10"]
        st.dataframe(df[preview_cols].head(20), use_container_width=True)

        st.markdown("#### Column Data Types and Missing Values")
        overview_table = pd.DataFrame({
            "Column Name": df.columns,
            "Data Type": [str(t) for t in df.dtypes],
            "Missing Values": df.isnull().sum().values,
            "Missing (%)": (df.isnull().sum().values / len(df) * 100).round(2),
            "Unique Count": [df[c].nunique() for c in df.columns]
        })
        st.dataframe(overview_table, use_container_width=True)

    # ---------------------------------------------------------
    # TAB 2: Target & Feature Distributions
    # ---------------------------------------------------------
    with tab2:
        st.markdown("### Food Temperature (Target) Distribution")

        # Summary statistics card
        t_col1, t_col2, t_col3, t_col4, t_col5 = st.columns(5)
        with t_col1:
            st.metric("Mean Temp", f"{kpis['target_mean']:.2f} °C")
        with t_col2:
            st.metric("Median Temp", f"{kpis['target_median']:.2f} °C")
        with t_col3:
            st.metric("Min Temp", f"{kpis['target_min']:.2f} °C")
        with t_col4:
            st.metric("Max Temp", f"{kpis['target_max']:.2f} °C")
        with t_col5:
            st.metric("Std Dev", f"{kpis['target_std']:.2f} °C")

        # Target Histogram
        fig_target = px.histogram(
            df,
            x=TARGET_COL,
            nbins=40,
            marginal="box",
            title="Distribution of Food Temperature (°C)",
            color_discrete_sequence=["#3b82f6"],
            labels={TARGET_COL: "Food Temperature (°C)", "count": "Frequency"}
        )
        fig_target.add_vline(x=kpis['target_mean'], line_dash="dash", line_color="#ef4444", annotation_text=f"Mean: {kpis['target_mean']:.1f}°C")
        fig_target.add_vline(x=kpis['target_median'], line_dash="dot", line_color="#10b981", annotation_text=f"Median: {kpis['target_median']:.1f}°C")
        fig_target.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=40, r=40, t=50, b=40))
        st.plotly_chart(fig_target, use_container_width=True, theme="streamlit")

        st.markdown("---")
        st.markdown("### Predictor Feature Distributions")
        selected_feat = st.selectbox("Select numerical predictor to inspect:", NUMERICAL_FEATURES)
        
        feat_col1, feat_col2 = st.columns([2, 1])
        with feat_col1:
            fig_feat = px.histogram(
                df,
                x=selected_feat,
                nbins=35,
                marginal="violin",
                title=f"Distribution of {selected_feat}",
                color_discrete_sequence=["#06b6d4"]
            )
            fig_feat.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=40, r=40, t=50, b=40))
            st.plotly_chart(fig_feat, use_container_width=True, theme="streamlit")
        with feat_col2:
            st.markdown(f"**{selected_feat} Summary Statistics**")
            s = df[selected_feat].describe()
            st.dataframe(pd.DataFrame(s).round(2), use_container_width=True)

    # ---------------------------------------------------------
    # TAB 3: Correlation Heatmap
    # ---------------------------------------------------------
    with tab3:
        st.markdown("### Numerical Feature Correlation")
        st.markdown(
            "> [!NOTE]\n"
            "> This correlation matrix is computed strictly on continuous numerical measurements. "
            "Categorical attributes are intentionally excluded to prevent misleading linear correlation artifacts. "
            "Correlation does not imply causation."
        )

        corr_cols = [c for c in NUMERICAL_FEATURES + [TARGET_COL] if c in df.columns]
        corr_matrix = df[corr_cols].corr()

        fig_corr = px.imshow(
            corr_matrix,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="Blues",
            title="Numerical Feature Correlation Matrix (Pearson r)",
            labels=dict(color="Correlation")
        )
        fig_corr.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=40, r=40, t=50, b=40),
            xaxis_title="",
            yaxis_title=""
        )
        st.plotly_chart(fig_corr, use_container_width=True, theme="streamlit")

        if TARGET_COL in corr_matrix:
            st.markdown("#### Correlation with Target (`food_temp_C`):")
            corr_target = corr_matrix[TARGET_COL].drop(TARGET_COL).sort_values(ascending=False)
            st.dataframe(pd.DataFrame({"Pearson r": corr_target.round(4)}), use_container_width=True)

    # ---------------------------------------------------------
    # TAB 4: Time-Series EDA by Run
    # ---------------------------------------------------------
    with tab4:
        st.markdown("### Dynamic Run-Level Time-Series Analysis")
        st.markdown(
            "Select any individual experimental solar cooking run to inspect its time-course thermal trajectory. "
            "Separate physical runs are never merged together."
        )

        run_options = kpis["unique_runs"]
        selected_run = st.selectbox("Select Experimental Run (run_id):", run_options, index=0)

        run_df = get_run_data(selected_run)

        if not run_df.empty:
            r_meta_c1, r_meta_c2, r_meta_c3 = st.columns(3)
            with r_meta_c1:
                st.info(f"Food Matrix: {run_df['food_matrix'].iloc[0]}")
            with r_meta_c2:
                st.info(f"Cooker Type: {run_df['cooker_type'].iloc[0]}")
            with r_meta_c3:
                st.info(f"Duration: {int(run_df['time_min'].max() - run_df['time_min'].min())} min ({len(run_df)} records)")

            # Combined multi-line temperature trajectory
            fig_run = go.Figure()

            if "food_temp_C" in run_df.columns:
                fig_run.add_trace(go.Scatter(
                    x=run_df["time_min"],
                    y=run_df["food_temp_C"],
                    mode="lines+markers",
                    name="Food Temp (°C)",
                    line=dict(color="#ef4444", width=2.5),
                    marker=dict(size=4)
                ))

            if "absorber_temp_C" in run_df.columns:
                fig_run.add_trace(go.Scatter(
                    x=run_df["time_min"],
                    y=run_df["absorber_temp_C"],
                    mode="lines",
                    name="Absorber Temp (°C)",
                    line=dict(color="#f97316", width=2, dash="dash")
                ))

            if "ambient_temp_C" in run_df.columns:
                fig_run.add_trace(go.Scatter(
                    x=run_df["time_min"],
                    y=run_df["ambient_temp_C"],
                    mode="lines",
                    name="Ambient Temp (°C)",
                    line=dict(color="#3b82f6", width=1.5, dash="dot")
                ))

            fig_run.update_layout(
                title=f"Thermal Trajectory — Run: {selected_run}",
                xaxis_title="Time (min)",
                yaxis_title="Temperature (°C)",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=40, r=40, t=60, b=40)
            )
            st.plotly_chart(fig_run, use_container_width=True, theme="streamlit")

            # Solar irradiance curve for the run
            if "solar_irradiance_W_m2" in run_df.columns:
                fig_solar = px.area(
                    run_df,
                    x="time_min",
                    y="solar_irradiance_W_m2",
                    title=f"Solar Irradiance Profile — Run: {selected_run}",
                    labels={"time_min": "Time (min)", "solar_irradiance_W_m2": "Solar Irradiance (W/m²)"},
                    color_discrete_sequence=["#eab308"]
                )
                fig_solar.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=40, r=40, t=50, b=40))
                st.plotly_chart(fig_solar, use_container_width=True, theme="streamlit")
        else:
            st.warning("No data found for the selected run.")

    # ---------------------------------------------------------
    # TAB 5: Food & Cooker Analysis
    # ---------------------------------------------------------
    with tab5:
        st.markdown("### Categorical Analysis: Food Matrix and Cooker Geometry")

        cat_col1, cat_col2 = st.columns(2)

        with cat_col1:
            st.markdown("#### Food Matrix Distribution")
            food_counts = df['food_matrix'].value_counts().reset_index()
            food_counts.columns = ['Food Matrix', 'Count']
            fig_food = px.bar(
                food_counts,
                x="Food Matrix",
                y="Count",
                text="Count",
                color="Food Matrix",
                title="Observations by Food Matrix"
            )
            fig_food.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False, margin=dict(l=40, r=40, t=50, b=40))
            st.plotly_chart(fig_food, use_container_width=True, theme="streamlit")

            fig_food_temp = px.box(
                df,
                x="food_matrix",
                y="food_temp_C",
                color="food_matrix",
                title="Food Temperature by Food Matrix (°C)",
                labels={"food_matrix": "Food Matrix", "food_temp_C": "Food Temp (°C)"}
            )
            fig_food_temp.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False, margin=dict(l=40, r=40, t=50, b=40))
            st.plotly_chart(fig_food_temp, use_container_width=True, theme="streamlit")

        with cat_col2:
            st.markdown("#### Cooker Type Distribution")
            cooker_counts = df['cooker_type'].value_counts().reset_index()
            cooker_counts.columns = ['Cooker Type', 'Count']
            fig_cooker = px.bar(
                cooker_counts,
                x="Cooker Type",
                y="Count",
                text="Count",
                color="Cooker Type",
                title="Observations by Cooker Type"
            )
            fig_cooker.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False, margin=dict(l=40, r=40, t=50, b=40))
            st.plotly_chart(fig_cooker, use_container_width=True, theme="streamlit")

            fig_cooker_temp = px.box(
                df,
                x="cooker_type",
                y="food_temp_C",
                color="cooker_type",
                title="Food Temperature by Cooker Type (°C)",
                labels={"cooker_type": "Cooker Type", "food_temp_C": "Food Temp (°C)"}
            )
            fig_cooker_temp.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False, margin=dict(l=40, r=40, t=50, b=40))
            st.plotly_chart(fig_cooker_temp, use_container_width=True, theme="streamlit")
