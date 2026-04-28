import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import io
from datetime import datetime
import matplotlib.pyplot as plt

from features import build_full_features
from config import CAT_COLS, BASE_COLS, FOURIER_ORDER, OUTPUT_DIR

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="Retail Sales Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1f3b6e;
    }
    .sub-title {
        font-size: 1rem;
        color: #6c757d;
        margin-bottom: 2rem;
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

st.markdown('<p class="main-title">📈 Retail Sales Forecasting Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Machine Learning–Driven Time Series Forecasting · Corporación Favorita · Ecuador</p>', unsafe_allow_html=True)

# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    use_oil = st.checkbox("Include Oil Price Feature", value=True)
    use_holidays = st.checkbox("Include Holiday Features", value=True)
    fourier_order = st.slider("Fourier Order", 0, 10, FOURIER_ORDER)

    st.divider()
    st.markdown("## 📂 Upload Data Files")
    st.caption("Upload your Kaggle dataset files below.")

    train_file   = st.file_uploader("train.csv",           type="csv")
    test_file    = st.file_uploader("test.csv",            type="csv")
    oil_file     = st.file_uploader("oil.csv",             type="csv")
    hol_file     = st.file_uploader("holidays_events.csv", type="csv")
    model_file   = st.file_uploader("Model (.pkl)",        type="pkl")

    st.divider()
    st.markdown("Built by **Paul Emakpor**")
    st.markdown("[GitHub](https://github.com/Emakporpaul/FUTURE_ML_01)")

# ---------------------------
# Check uploads
# ---------------------------
files_ready = all([train_file, test_file, model_file])

if not files_ready:
    st.info(
        "👈 Upload **train.csv**, **test.csv**, and your **model .pkl** file "
        "in the sidebar to get started. oil.csv and holidays_events.csv are optional but recommended."
    )
    st.markdown("""
    ### How to use this app
    1. Upload your Kaggle dataset files in the sidebar
    2. Upload the trained model `.pkl` file
    3. Explore the **Data Overview** tab
    4. Check **Model Evaluation** for metrics
    5. Go to **Generate Forecast** to create `submission.csv`

    ---
    ### About this Project
    This app forecasts daily sales for **54 stores × 33 product families**
    for Corporación Favorita, a large Ecuadorian retail chain, using:
    - XGBoost with engineered time series features
    - Monthly Fourier terms for seasonality
    - Holiday and oil price signals
    - Time-based validation (no data leakage)
    """)
    st.stop()

# ---------------------------
# Load uploaded files
# ---------------------------
@st.cache_data
def load_uploads(train_bytes, test_bytes, oil_bytes, hol_bytes):
    train = pd.read_csv(io.BytesIO(train_bytes), parse_dates=["date"])
    test  = pd.read_csv(io.BytesIO(test_bytes),  parse_dates=["date"])
    oil   = pd.read_csv(io.BytesIO(oil_bytes),   parse_dates=["date"]) if oil_bytes else None
    hol   = pd.read_csv(io.BytesIO(hol_bytes),   parse_dates=["date"]) if hol_bytes else None
    return train, test, oil, hol

@st.cache_resource
def load_model_bytes(model_bytes):
    import joblib
    import io
    return joblib.load(io.BytesIO(model_bytes))

train, test, oil, hol = load_uploads(
    train_file.read(),
    test_file.read(),
    oil_file.read()  if oil_file else None,
    hol_file.read()  if hol_file else None,
)

model = load_model_bytes(model_file.read())

st.success("✅ Files loaded successfully. Explore the tabs below.")

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
    c1.metric("Training Rows",     f"{len(train):,}")
    c2.metric("Test Rows",         f"{len(test):,}")
    c3.metric("Stores",            train["store_nbr"].nunique())
    c4.metric("Product Families",  train["family"].nunique())

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
    st.markdown("**Top 10 Product Families by Average Sales**")
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

    st.markdown("""
    #### Validation Strategy
    - **Method:** Time-based split (last 10% of training dates held out)
    - **No data leakage:** future dates never seen during training
    - **Metric:** RMSLE (Root Mean Squared Log Error) — standard for sales forecasting
    """)

    st.divider()
    st.markdown("""
    #### Model vs Baseline Comparison

    | Model | Approach |
    |-------|----------|
    | **Baseline** | Mean sales per store-family (naive) |
    | **XGBoost Pipeline** | Seasonal features + promotions + oil + holidays |

    Run `python src/evaluate.py` locally to generate full metrics and charts,
    then check `outputs/metrics.json` for exact scores.
    """)

    st.info("Upload `outputs/metrics.json` below to display live metrics here.")
    metrics_upload = st.file_uploader("metrics.json (optional)", type="json")

    if metrics_upload:
        metrics = json.load(metrics_upload)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🔵 XGBoost Pipeline")
            xgb = metrics.get("xgb_pipeline", {})
            st.metric("RMSLE", xgb.get("rmsle", "N/A"))
            st.metric("RMSE",  xgb.get("rmse",  "N/A"))
            st.metric("MAE",   xgb.get("mae",   "N/A"))
        with col2:
            st.markdown("#### 🟠 Baseline")
            base = metrics.get("baseline_mean_by_store_family", {})
            st.metric("RMSLE", base.get("rmsle", "N/A"))
            st.metric("RMSE",  base.get("rmse",  "N/A"))
            st.metric("MAE",   base.get("mae",   "N/A"))

# ===========================
# TAB 3 — Generate Forecast
# ===========================
with tab3:
    st.markdown('<p class="section-header">Generate Forecast & Download Submission</p>', unsafe_allow_html=True)

    st.info(
        "Builds the full feature matrix, runs the XGBoost pipeline, "
        "and produces a Kaggle-ready `submission.csv` with columns: `id`, `sales`."
    )

    if st.button("🚀 Run Forecast + Build Submission", type="primary"):
        with st.spinner("Building feature matrix..."):
            full_train, full_test = build_full_features(
                train_df=train,
                test_df=test,
                oil_df=oil if use_oil else None,
                hol_df=hol if use_holidays else None,
                use_oil=use_oil,
                use_holidays=use_holidays,
                fourier_order=fourier_order,
            )
            fourier_cols  = [c for c in full_train.columns if c.startswith("sin(") or c.startswith("cos(")]
            used_features = CAT_COLS + BASE_COLS + fourier_cols
            X_test = full_test[used_features]

        with st.spinner("Predicting sales..."):
            preds = model.predict(X_test)
            preds = np.clip(preds, 0, None)

        submission = pd.DataFrame({
            "id":    full_test["id"].values,
            "sales": preds,
        })

        st.success(f"✅ Forecast complete! {len(submission):,} predictions generated.")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Predictions", f"{len(submission):,}")
        col2.metric("Min Sales",         f"{preds.min():.2f}")
        col3.metric("Max Sales",         f"{preds.max():,.2f}")

        st.divider()
        st.markdown("**Forecast Preview**")
        st.dataframe(submission.head(20), use_container_width=True)

        st.divider()
        st.markdown("**Predicted Sales Distribution**")
        fig3, ax3 = plt.subplots(figsize=(10, 3))
        ax3.hist(preds, bins=80, color="steelblue", edgecolor="white")
        ax3.set_title("Distribution of Predicted Sales", fontweight="bold")
        ax3.set_xlabel("Predicted Sales")
        ax3.set_ylabel("Count")
        ax3.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig3)

        st.divider()
        csv_bytes = submission.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download submission.csv",
            data=csv_bytes,
            file_name="submission.csv",
            mime="text/csv",
            type="primary",
        )

# ===========================
# TAB 4 — Project Summary
# ===========================
with tab4:
    st.markdown('<p class="section-header">Project Summary</p>', unsafe_allow_html=True)
    st.markdown("""
    ### Business Problem
    Predict daily product sales across **54 stores** and **33 product families**
    for Corporación Favorita, a large Ecuadorian retail chain.

    ---
    ### Dataset
    | File | Description |
    |------|-------------|
    | `train.csv` | ~3M rows of historical daily sales (2013–2017) |
    | `test.csv` | 16-day forecast horizon (28,512 rows) |
    | `oil.csv` | Daily Brent crude oil prices |
    | `holidays_events.csv` | National, regional & local Ecuadorian holidays |

    ---
    ### Feature Engineering
    - Calendar features: day of week, week of year, month, day of year
    - Fourier terms: monthly sine/cosine for smooth seasonality
    - Promotion: onpromotion count per store-family
    - Oil price: forward-filled daily price
    - Holiday flags: national, regional, local indicators

    ---
    ### Modeling
    - **Algorithm:** XGBoost Regressor
    - **Validation:** Time-based split — no data leakage
    - **Pipeline:** scikit-learn Pipeline saved as .pkl

    ---
    ### Key Findings
    - Strong weekly seasonality in sales patterns
    - National holidays cause significant sales spikes
    - Promotions positively correlate with increased demand
    - Oil prices show indirect influence on consumer spending

    ---
    ### Next Steps
    - Add lag features (7-day, 14-day rolling means)
    - Apply log1p target transformation
    - Tune XGBoost hyperparameters
    - Add store metadata features (city, type, cluster)
    """)