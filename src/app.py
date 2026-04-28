import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt

from features import build_full_features
from config import (
    TRAIN_PATH, TEST_PATH, OIL_PATH, HOLIDAYS_PATH,
    OUTPUT_DIR, MODEL_PATH, CAT_COLS, BASE_COLS, FOURIER_ORDER
)

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="Retail Sales Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------
# Custom Styling
# ---------------------------
st.markdown("""
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1f3b6e;
        margin-bottom: 0;
    }
    .sub-title {
        font-size: 1rem;
        color: #6c757d;
        margin-top: 0;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f4ff;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #1f3b6e;
        border-bottom: 2px solid #1f3b6e;
        padding-bottom: 0.3rem;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------
# Header
# ---------------------------
st.markdown('<p class="main-title">📈 Retail Sales Forecasting Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Machine Learning–Driven Time Series Forecasting · Corporación Favorita · Ecuador</p>', unsafe_allow_html=True)

# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/combo-chart.png", width=60)
    st.markdown("## ⚙️ Settings")
    use_oil = st.checkbox("Include Oil Price Feature", value=True)
    use_holidays = st.checkbox("Include Holiday Features", value=True)
    fourier_order = st.slider("Fourier Order (Monthly Seasonality)", 0, 10, FOURIER_ORDER)

    st.divider()
    st.markdown("## 📁 Project Info")
    st.markdown("**Model:** XGBoost Pipeline")
    st.markdown("**Target:** Daily Store × Family Sales")
    st.markdown("**Metric:** RMSLE")
    st.markdown("**Forecast Horizon:** 16 days")
    st.divider()
    st.markdown("Built by **Paul Emakpor**")
    st.markdown("[GitHub](https://github.com/Emakporpaul/FUTURE_ML_01)")

# ---------------------------
# Load Data + Model
# ---------------------------
@st.cache_data
def load_data():
    train = pd.read_csv(TRAIN_PATH, parse_dates=["date"])
    test  = pd.read_csv(TEST_PATH,  parse_dates=["date"])
    oil   = pd.read_csv(OIL_PATH,   parse_dates=["date"])
    hol   = pd.read_csv(HOLIDAYS_PATH, parse_dates=["date"])
    return train, test, oil, hol

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    st.error(f"Model not found at `{MODEL_PATH}`. Run `python src/train.py` first.")
    st.stop()

train, test, oil, hol = load_data()
model = load_model()

# ---------------------------
# Tabs
# ---------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Data Overview",
    "📉 Model Evaluation",
    "🔮 Generate Forecast",
    "📋 Project Summary",
])

# ===========================
# TAB 1 — Data Overview
# ===========================
with tab1:
    st.markdown('<p class="section-header">Dataset Overview</p>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Training Rows", f"{len(train):,}")
    c2.metric("Test Rows", f"{len(test):,}")
    c3.metric("Stores", train["store_nbr"].nunique())
    c4.metric("Product Families", train["family"].nunique())

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Train Data Preview**")
        st.dataframe(train.head(10), use_container_width=True)
    with col2:
        st.markdown("**Test Data Preview**")
        st.dataframe(test.head(10), use_container_width=True)

    st.divider()
    st.markdown("**Average Daily Sales Trend**")
    avg_daily = train.groupby("date")["sales"].mean().sort_index()
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(avg_daily.index, avg_daily.values, color="steelblue", linewidth=1.5)
    ax.set_title("Average Daily Sales (All Stores & Families)", fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Avg Sales")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

    st.divider()
    st.markdown("**Sales Distribution by Product Family (Top 10)**")
    top10 = (
        train.groupby("family")["sales"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    ax2.barh(top10.index[::-1], top10.values[::-1], color="steelblue")
    ax2.set_title("Top 10 Product Families by Average Sales", fontweight="bold")
    ax2.set_xlabel("Avg Sales")
    ax2.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    st.pyplot(fig2)

# ===========================
# TAB 2 — Model Evaluation
# ===========================
with tab2:
    st.markdown('<p class="section-header">Model Evaluation</p>', unsafe_allow_html=True)

    metrics_path = os.path.join(OUTPUT_DIR, "metrics.json")
    chart_path   = os.path.join(OUTPUT_DIR, "validation_chart.png")

    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)

        st.markdown(f"**Validation Cutoff Date:** `{metrics.get('cutoff_date', 'N/A')}`")
        st.markdown(f"**Train rows:** `{metrics.get('n_train_rows', 'N/A'):,}` &nbsp;|&nbsp; **Validation rows:** `{metrics.get('n_valid_rows', 'N/A'):,}`")

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🔵 XGBoost Pipeline")
            xgb = metrics.get("xgb_pipeline", {})
            st.metric("RMSLE", xgb.get("rmsle", "N/A"))
            st.metric("RMSE",  xgb.get("rmse",  "N/A"))
            st.metric("MAE",   xgb.get("mae",   "N/A"))

        with col2:
            st.markdown("#### 🟠 Baseline (Mean by Store-Family)")
            base = metrics.get("baseline_mean_by_store_family", {})
            st.metric("RMSLE", base.get("rmsle", "N/A"))
            st.metric("RMSE",  base.get("rmse",  "N/A"))
            st.metric("MAE",   base.get("mae",   "N/A"))

        st.divider()
        st.markdown("**Interpretation:** Lower RMSLE = better. XGBoost should outperform the baseline, confirming that feature engineering and ML modeling add real value over a naive approach.")

    else:
        st.warning("No metrics found. Run `python src/evaluate.py` first.")

    if os.path.exists(chart_path):
        st.divider()
        st.markdown("**Actual vs Predicted — Validation Period**")
        st.image(chart_path, use_container_width=True)
    else:
        st.info("Validation chart not found. Run `python src/evaluate.py` to generate it.")

# ===========================
# TAB 3 — Generate Forecast
# ===========================
with tab3:
    st.markdown('<p class="section-header">Generate Forecast & Download Submission</p>', unsafe_allow_html=True)

    st.info(
        "This rebuilds the full feature matrix for train + test consistently, "
        "runs the trained XGBoost pipeline, clips negative predictions to 0, "
        "and produces a Kaggle-ready `submission.csv` with columns: `id`, `sales`."
    )

    if st.button("🚀 Run Forecast + Build Submission", type="primary"):
        with st.spinner("Building feature matrix..."):
            full_train, full_test = build_full_features(
                train_df=train,
                test_df=test,
                oil_df=oil   if use_oil      else None,
                hol_df=hol   if use_holidays else None,
                use_oil=use_oil,
                use_holidays=use_holidays,
                fourier_order=fourier_order,
            )
            fourier_cols  = [c for c in full_train.columns if c.startswith("sin(") or c.startswith("cos(")]
            used_features = CAT_COLS + BASE_COLS + f