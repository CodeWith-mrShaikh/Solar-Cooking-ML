"""
Model Performance View — Evaluation Benchmarks & Feature Importance
Compares the six evaluated regression models across 70:30 and 80:20 splits,
presents the production Extra Trees model architecture, and displays feature importance.
Guarantees pure white, high-contrast text across all elements in Dark Mode.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

try:
    from src.config import BENCHMARK_RESULTS, PRODUCTION_EVALUATION, CANONICAL_FEATURES, FEATURE_UI_CONFIG
    from src.model_loader import load_model_artifacts
except (ImportError, KeyError):
    from config import BENCHMARK_RESULTS, PRODUCTION_EVALUATION, CANONICAL_FEATURES, FEATURE_UI_CONFIG
    from model_loader import load_model_artifacts

def render_model_performance():
    st.markdown("<h1 class='main-title'>Model Performance and Evaluation</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='subtitle-desc'>"
        "Comparative evaluation of six regression models across 70:30 and 80:20 evaluation splits. "
        "All performance metrics represent exact experimental benchmarks established during Colab training."
        "</p>",
        unsafe_allow_html=True
    )

    # ---------------------------------------------------------
    # Production Model Highlight Card
    # ---------------------------------------------------------
    st.markdown(
        f"""
        <div class='highlight-card'>
            <div style='display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 15px;'>
                <div>
                    <span class='badge-pill badge-primary'>SELECTED PRODUCTION MODEL</span>
                    <h2 class='card-title' style='margin: 8px 0 4px 0; font-size: 1.6rem;'>{PRODUCTION_EVALUATION['model_name']}</h2>
                    <p class='card-desc' style='margin: 0; font-size: 0.95rem;'>
                        Hyperparameters: <code class='code-highlight'>n_estimators=100</code>, <code class='code-highlight'>random_state=42</code> | Train/Test Split: <code class='code-highlight'>80:20</code>
                    </p>
                </div>
                <div style='display: flex; gap: 20px; align-items: center;'>
                    <div style='text-align: right;'>
                        <div class='metric-label'>Validation R²</div>
                        <div class='metric-accent-green' style='font-size: 1.9rem;'>{PRODUCTION_EVALUATION['test_r2'] * 100:.2f}%</div>
                        <div class='metric-label' style='font-size: 0.75rem; text-transform: none;'>R² = {PRODUCTION_EVALUATION['test_r2']:.6f}</div>
                    </div>
                    <div style='text-align: right; border-left: 1px solid rgba(128, 128, 128, 0.3); padding-left: 20px;'>
                        <div class='metric-label'>Validation MSE</div>
                        <div class='metric-accent-blue' style='font-size: 1.9rem;'>{PRODUCTION_EVALUATION['test_mse']:.3f}</div>
                        <div class='metric-label' style='font-size: 0.75rem; text-transform: none;'>MSE = {PRODUCTION_EVALUATION['test_mse']:.6f}</div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # Evaluation Benchmark Table
    # ---------------------------------------------------------
    st.markdown("### Benchmark Evaluation Summary (All 6 Models)")
    st.markdown(
        "<p style='font-size: 0.9rem;'>"
        "Evaluated on 1-minute resampled experimental observations. "
        "<i>Note: R² measures proportion of variance explained and is not an accuracy percentage.</i>"
        "</p>",
        unsafe_allow_html=True
    )

    bench_df = pd.DataFrame(BENCHMARK_RESULTS)

    # Filter by split selector
    split_filter = st.radio(
        "Select Evaluation Split View:",
        ["All Splits", "80:20 Split Only", "70:30 Split Only"],
        horizontal=True
    )

    if split_filter == "80:20 Split Only":
        display_df = bench_df[bench_df["Split"] == "80:20"].sort_values("Test R²", ascending=False)
    elif split_filter == "70:30 Split Only":
        display_df = bench_df[bench_df["Split"] == "70:30"].sort_values("Test R²", ascending=False)
    else:
        display_df = bench_df.sort_values(["Split", "Test R²"], ascending=[True, False])

    # Format table for display
    formatted_table = display_df.copy()
    formatted_table["Test R² Score"] = formatted_table["Test R²"].apply(lambda x: f"{x:.6f} ({x*100:.2f}%)")
    formatted_table["Test MSE"] = formatted_table["Test MSE"].apply(lambda x: f"{x:.6f}")
    formatted_table["Overfitting (Diff)"] = formatted_table["Overfitting (Diff)"].apply(lambda x: f"{x:+.6f}")
    
    st.dataframe(
        formatted_table[["Model", "Split", "Test R² Score", "Test MSE", "Overfitting (Diff)"]],
        use_container_width=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # Interactive Benchmark Visualizations
    # ---------------------------------------------------------
    st.markdown("### Visual Benchmark Comparison")
    
    v_col1, v_col2 = st.columns(2)

    with v_col1:
        fig_r2 = px.bar(
            bench_df,
            x="Model",
            y="Test R²",
            color="Split",
            barmode="group",
            text_auto=".4f",
            title="Test R² Comparison Across Models and Splits",
            color_discrete_map={"80:20": "#3b82f6", "70:30": "#06b6d4"}
        )
        fig_r2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis_range=[0.80, 1.005],
            margin=dict(l=40, r=40, t=50, b=40)
        )
        st.plotly_chart(fig_r2, use_container_width=True, theme="streamlit")

    with v_col2:
        fig_mse = px.bar(
            bench_df,
            x="Model",
            y="Test MSE",
            color="Split",
            barmode="group",
            text_auto=".2f",
            title="Test MSE Comparison Across Models and Splits (Lower is Better)",
            color_discrete_map={"80:20": "#ef4444", "70:30": "#f97316"}
        )
        fig_mse.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=40, r=40, t=50, b=40)
        )
        st.plotly_chart(fig_mse, use_container_width=True, theme="streamlit")

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # Production Feature Importance
    # ---------------------------------------------------------
    st.markdown("### Production Model Feature Importance")
    
    try:
        artifacts = load_model_artifacts()
        model = artifacts["model"]
        feature_config = artifacts["feature_config"]
        features = feature_config.get("features", CANONICAL_FEATURES)

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            if len(importances) == len(features):
                fi_df = pd.DataFrame({
                    "Feature": [FEATURE_UI_CONFIG.get(f, {}).get("label", f) for f in features],
                    "Feature Key": features,
                    "Importance": importances
                }).sort_values("Importance", ascending=True)

                fig_fi = px.bar(
                    fi_df,
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    text_auto=".4f",
                    title="Gini Feature Importance (Extra Trees Regressor)",
                    color="Importance",
                    color_continuous_scale="Blues"
                )
                fig_fi.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=40, r=40, t=50, b=40),
                    coloraxis_showscale=False
                )
                st.plotly_chart(fig_fi, use_container_width=True, theme="streamlit")
            else:
                st.warning(f"Feature importance count ({len(importances)}) does not match canonical features ({len(features)}).")
        else:
            st.info("The loaded production model does not expose feature_importances_.")
    except Exception as e:
        st.info(f"Feature importance visualization unavailable until model artifacts are loaded: {e}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # Rigorous Data Science Notes
    # ---------------------------------------------------------
    st.markdown(
        """
        <div class='glass-card'>
            <h4 class='card-title' style='margin-top: 0; font-size: 1.15rem;'>Critical Data Science and Validation Note</h4>
            <p class='card-desc' style='font-size: 0.92rem; line-height: 1.6; margin-bottom: 10px;'>
                The reported evaluation metrics are based on the standard randomized train/test split (<code class='code-highlight'>train_test_split(..., random_state=42)</code>) 
                applied across individual 1-minute rows as implemented in the original Google Colab research pipeline.
            </p>
            <p class='card-desc' style='font-size: 0.92rem; line-height: 1.6; margin-bottom: 0;'>
                Because consecutive 1-minute rows belong to the same underlying physical experiment (<code class='code-highlight'>run_id</code>), 
                row-level random splitting may produce high autocorrelation between train and test sets. 
                Consequently, the high R² (99.74%) reflects exceptional interpolation across observed cooking trajectories 
                rather than proven zero-shot generalization to unobserved solar cooker designs or novel weather regimes.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
