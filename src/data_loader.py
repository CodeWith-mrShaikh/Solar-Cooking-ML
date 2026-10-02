"""
Data loader module for Solar Cooking ML Prediction System.
Handles cached dataset loading, statistical KPI calculations, and run extraction.
"""

import os
import pandas as pd
import numpy as np
try:
    import streamlit as st
    cache_data_decorator = st.cache_data
except ImportError:
    def cache_data_decorator(func):
        return func

try:
    from src.config import (
        CANONICAL_DATA_PATH,
        TARGET_COL,
        NUMERICAL_FEATURES,
        CATEGORICAL_FEATURES,
        CANONICAL_FEATURES
    )
except (ImportError, KeyError):
    from config import (
        CANONICAL_DATA_PATH,
        TARGET_COL,
        NUMERICAL_FEATURES,
        CATEGORICAL_FEATURES,
        CANONICAL_FEATURES
    )

@cache_data_decorator
def load_dataset(data_path: str = CANONICAL_DATA_PATH) -> pd.DataFrame:
    """
    Loads the canonical training dataset from disk with caching.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Canonical dataset not found at: {data_path}")
    df = pd.read_csv(data_path)
    return df

@cache_data_decorator
def get_dataset_kpis(data_path: str = CANONICAL_DATA_PATH) -> dict:
    """
    Computes dynamic summary statistics and KPIs from the actual dataset.
    Never hard-codes statistics.
    """
    df = load_dataset(data_path)
    
    target_series = df[TARGET_COL].dropna() if TARGET_COL in df.columns else pd.Series(dtype=float)
    clean_cols = [c for c in df.columns if not c.startswith("Unnamed")]

    kpis = {
        "total_rows": int(len(df)),
        "total_columns": int(len(df.columns)),
        "clean_columns": clean_cols,
        "columns_list": list(df.columns),
        "unique_runs_count": int(df['run_id'].nunique()) if 'run_id' in df.columns else 0,
        "unique_runs": sorted(df['run_id'].dropna().unique().tolist()) if 'run_id' in df.columns else [],
        "total_missing_values": int(df[clean_cols].isnull().sum().sum()),
        "raw_missing_values": int(df.isnull().sum().sum()),
        "missing_by_column": df.isnull().sum().to_dict(),
        "target_mean": float(target_series.mean()) if not target_series.empty else 0.0,
        "target_median": float(target_series.median()) if not target_series.empty else 0.0,
        "target_min": float(target_series.min()) if not target_series.empty else 0.0,
        "target_max": float(target_series.max()) if not target_series.empty else 0.0,
        "target_std": float(target_series.std()) if not target_series.empty else 0.0,
        "food_matrices": sorted(df['food_matrix'].dropna().unique().tolist()) if 'food_matrix' in df.columns else [],
        "cooker_types": sorted(df['cooker_type'].dropna().unique().tolist()) if 'cooker_type' in df.columns else [],
    }
    return kpis

@cache_data_decorator
def get_feature_ranges(data_path: str = CANONICAL_DATA_PATH) -> dict:
    """
    Computes observed min, max, mean, and median for all numerical features in the training dataset.
    Used for user input validation and out-of-range warnings.
    """
    df = load_dataset(data_path)
    ranges = {}
    for col in NUMERICAL_FEATURES:
        if col in df.columns:
            s = df[col].dropna()
            ranges[col] = {
                "min": float(s.min()),
                "max": float(s.max()),
                "mean": float(s.mean()),
                "median": float(s.median()),
                "std": float(s.std())
            }
    return ranges

@cache_data_decorator
def get_run_data(run_id: str, data_path: str = CANONICAL_DATA_PATH) -> pd.DataFrame:
    """
    Returns time-ordered subset for a specific run_id.
    """
    df = load_dataset(data_path)
    if 'run_id' not in df.columns:
        return pd.DataFrame()
    run_df = df[df['run_id'] == run_id].sort_values('time_min').copy()
    return run_df
