"""
Solar Cooking ML Prediction System — Streamlit Web Application
Main entry point orchestrating sidebar navigation, system health status, and page views.
Equipped with real clickable navigation buttons and full Dark/Light mode contrast support.
"""

import streamlit as st
import os
import sys

# Ensure project root and src directory are on sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
for p in [PROJECT_ROOT, SRC_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from src.model_loader import get_system_status
    from src.views.dashboard_view import render_dashboard
    from src.views.prediction_view import render_prediction
    from src.views.eda_view import render_eda
    from src.views.model_performance_view import render_model_performance
    from src.views.about_view import render_about
except (ImportError, KeyError):
    from model_loader import get_system_status
    from views.dashboard_view import render_dashboard
    from views.prediction_view import render_prediction
    from views.eda_view import render_eda
    from views.model_performance_view import render_model_performance
    from views.about_view import render_about

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Solar Cooking ML Prediction System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------
# High-Contrast Adaptive Custom CSS (Full Dark & Light Mode Support)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Primary Page Titles and Subtitles (Theme-Adaptive) */
    h1.main-title, .main-title,
    div[data-testid="stMarkdownContainer"] h1,
    div[data-testid="stMarkdownContainer"] h2,
    div[data-testid="stMarkdownContainer"] h3,
    div[data-testid="stMarkdownContainer"] h4 {
        color: inherit !important;
    }

    h1.main-title, .main-title {
        font-size: 2.1rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
        letter-spacing: -0.02em;
    }

    p.subtitle-desc, .subtitle-desc {
        color: inherit !important;
        opacity: 0.88;
        font-size: 1.0rem;
        margin-bottom: 1.5rem;
        line-height: 1.5;
    }

    /* Cards with Adaptive Theme */
    .glass-card, .module-card {
        background-color: rgba(128, 128, 128, 0.08) !important;
        border: 1px solid rgba(128, 128, 128, 0.22) !important;
        border-radius: 12px;
        padding: 22px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
        color: inherit !important;
    }

    .module-card {
        padding: 20px;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        transition: transform 0.15s ease, border-color 0.15s ease;
    }

    .module-card:hover {
        border-color: #3b82f6 !important;
        transform: translateY(-2px);
    }

    .highlight-card {
        background-color: rgba(59, 130, 246, 0.08) !important;
        border: 1.5px solid rgba(59, 130, 246, 0.45) !important;
        border-radius: 14px;
        padding: 24px;
        box-shadow: 0 4px 8px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
        color: inherit !important;
    }

    .card-title,
    .glass-card h1, .glass-card h2, .glass-card h3, .glass-card h4,
    .highlight-card h1, .highlight-card h2, .highlight-card h3, .highlight-card h4,
    .module-card h1, .module-card h2, .module-card h3, .module-card h4 {
        font-size: 1.35rem;
        font-weight: 700;
        margin: 6px 0 4px 0;
        color: inherit !important;
    }

    .card-subtitle {
        font-size: 1.05rem;
        font-weight: 600;
        margin-top: 0;
        color: inherit !important;
    }

    .card-desc,
    .glass-card p, .glass-card span, .glass-card div, .glass-card li, .glass-card b, .glass-card ul,
    .highlight-card p, .highlight-card span, .highlight-card div, .highlight-card li, .highlight-card b,
    .module-card p, .module-card span, .module-card div, .module-card li, .module-card b {
        font-size: 0.93rem;
        line-height: 1.6;
        color: inherit !important;
    }

    /* Metric Accents */
    .metric-accent-green, span.metric-accent-green, div.metric-accent-green {
        color: #10b981 !important;
        font-weight: 700 !important;
    }

    .metric-accent-blue, span.metric-accent-blue, div.metric-accent-blue {
        color: #3b82f6 !important;
        font-weight: 700 !important;
    }

    .metric-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: inherit !important;
        opacity: 0.75;
        font-weight: 600;
    }

    .code-highlight, .glass-card code, .highlight-card code, .module-card code {
        color: #3b82f6 !important;
        background-color: rgba(59, 130, 246, 0.12) !important;
        border: 1px solid rgba(59, 130, 246, 0.25) !important;
        padding: 2px 6px !important;
        border-radius: 4px !important;
        font-size: 0.88em !important;
        font-family: monospace !important;
    }

    .result-card {
        background-color: rgba(59, 130, 246, 0.12) !important;
        border: 2px solid #3b82f6 !important;
        border-radius: 16px;
        padding: 28px;
        text-align: center;
        margin-top: 25px;
        box-shadow: 0 8px 16px -2px rgba(59, 130, 246, 0.15);
        color: inherit !important;
    }

    .temp-value {
        font-size: 3.5rem;
        font-weight: 800;
        color: #3b82f6 !important;
        line-height: 1.1;
        margin: 10px 0;
        letter-spacing: -0.02em;
    }

    /* Badges & Pills */
    .badge-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }

    .badge-primary {
        background-color: rgba(59, 130, 246, 0.18);
        color: #3b82f6 !important;
        border: 1px solid rgba(59, 130, 246, 0.35);
    }

    .status-badge {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 8px 12px;
        border-radius: 8px;
        margin-bottom: 6px;
        font-size: 0.85rem;
        font-weight: 500;
    }

    .status-ok {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }

    .status-missing {
        background-color: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }

    /* ======================================================== */
    /* Real Clickable Navigation Buttons in Sidebar             */
    /* ======================================================== */
    section[data-testid="stSidebar"] div.stButton {
        margin-bottom: 6px !important;
    }

    section[data-testid="stSidebar"] div.stButton > button {
        border-radius: 8px !important;
        padding: 10px 16px !important;
        font-size: 0.93rem !important;
        font-weight: 500 !important;
        text-align: left !important;
        justify-content: flex-start !important;
        display: flex !important;
        align-items: center !important;
        transition: all 0.15s ease-in-out !important;
        width: 100% !important;
        cursor: pointer !important;
    }

    /* Inactive navigation button */
    section[data-testid="stSidebar"] div.stButton > button[kind="secondary"] {
        background-color: rgba(128, 128, 128, 0.08) !important;
        border: 1px solid rgba(128, 128, 128, 0.22) !important;
        color: inherit !important;
    }

    section[data-testid="stSidebar"] div.stButton > button[kind="secondary"] *,
    section[data-testid="stSidebar"] div.stButton > button[kind="secondary"] p,
    section[data-testid="stSidebar"] div.stButton > button[kind="secondary"] span,
    section[data-testid="stSidebar"] div.stButton > button[kind="secondary"] div {
        color: inherit !important;
        font-weight: 500 !important;
    }

    section[data-testid="stSidebar"] div.stButton > button[kind="secondary"]:hover {
        border-color: #3b82f6 !important;
        background-color: rgba(59, 130, 246, 0.12) !important;
        transform: translateX(2px) !important;
    }

    section[data-testid="stSidebar"] div.stButton > button[kind="secondary"]:hover *,
    section[data-testid="stSidebar"] div.stButton > button[kind="secondary"]:hover p,
    section[data-testid="stSidebar"] div.stButton > button[kind="secondary"]:hover span,
    section[data-testid="stSidebar"] div.stButton > button[kind="secondary"]:hover div {
        color: #3b82f6 !important;
    }

    /* Active (selected) navigation button */
    section[data-testid="stSidebar"] div.stButton > button[kind="primary"] {
        background-color: #2563eb !important;
        border: 1px solid #1d4ed8 !important;
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.3) !important;
    }

    section[data-testid="stSidebar"] div.stButton > button[kind="primary"] *,
    section[data-testid="stSidebar"] div.stButton > button[kind="primary"] p,
    section[data-testid="stSidebar"] div.stButton > button[kind="primary"] span,
    section[data-testid="stSidebar"] div.stButton > button[kind="primary"] div {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# Sidebar Navigation and System Status
# ---------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style='text-align: left; padding: 10px 0 15px 0;'>
            <h2 style='margin: 0; font-size: 1.25rem; color: inherit; font-weight: 700;'>Solar Cooking ML</h2>
            <span style='font-size: 0.8rem; color: inherit; opacity: 0.7;'>Prediction and Analytics Platform</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.markdown("<div style='font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: inherit; opacity: 0.85; margin-bottom: 8px;'>Navigation</div>", unsafe_allow_html=True)

    # Initialize current page in session state
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Dashboard"

    NAV_ITEMS = [
        ("Dashboard", "dashboard"),
        ("Prediction", "prediction"),
        ("EDA Dashboard", "eda"),
        ("Model Performance", "model_performance"),
        ("About and Science", "about")
    ]

    for label, key_suffix in NAV_ITEMS:
        is_active = (st.session_state.current_page == label)
        btn_type = "primary" if is_active else "secondary"
        if st.button(label, key=f"nav_btn_{key_suffix}", use_container_width=True, type=btn_type):
            st.session_state.current_page = label
            st.rerun()

    st.markdown("---")
    st.markdown("<div style='font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: inherit; opacity: 0.85; margin-bottom: 8px;'>System Status</div>", unsafe_allow_html=True)

    status = get_system_status()

    # 1. Dataset Status
    if status["dataset_loaded"]:
        st.markdown(
            "<div class='status-badge status-ok'><span>Dataset</span><span>Loaded</span></div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div class='status-badge status-missing'><span>Dataset</span><span>Missing</span></div>",
            unsafe_allow_html=True
        )

    # 2. Model Status
    if status["model_loaded"]:
        st.markdown(
            "<div class='status-badge status-ok'><span>Model</span><span>Loaded</span></div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div class='status-badge status-missing'><span>Model</span><span>Missing</span></div>",
            unsafe_allow_html=True
        )

    # 3. Preprocessing Status
    if status["preprocessing_loaded"]:
        st.markdown(
            "<div class='status-badge status-ok'><span>Preprocessing</span><span>Loaded</span></div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div class='status-badge status-missing'><span>Preprocessing</span><span>Missing</span></div>",
            unsafe_allow_html=True
        )

    if status["missing_items"]:
        with st.expander("Missing Artifacts Detail", expanded=False):
            st.caption("Missing files:")
            for item in status["missing_items"]:
                st.code(item, language="bash")
            st.caption("Run python scripts/export_model_artifacts.py to generate.")

    st.markdown("---")
    st.markdown(
        """
        <div style='font-size: 0.75rem; color: inherit; opacity: 0.75; text-align: left; line-height: 1.5;'>
            <b>Academic EPICS Project</b><br>
            Production Extra Trees Regressor<br>
            Validation R²: 99.74% (80:20 Split)
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------------------------------------------------------
# Main Content Router
# ---------------------------------------------------------
selected_page = st.session_state.current_page

if selected_page == "Dashboard":
    render_dashboard()
elif selected_page == "Prediction":
    render_prediction()
elif selected_page == "EDA Dashboard":
    render_eda()
elif selected_page == "Model Performance":
    render_model_performance()
elif selected_page == "About and Science":
    render_about()
