import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

from features import build_full_features

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(
    page_title="Retail Sales Forecasting (Store × Family)",
    layout="wide"
)

# ---------------------------
# Constants / paths
# ---------------------------
DATA_DIR = "data/raw"
MODEL_PATH = "models/xgb_store_family_pipeline.pkl"
OUTPUT_DIR = "outputs"

TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
OIL_PATH = os.path.join(DATA_DIR, "oil.csv")
HOL_PATH = os.path.join(DATA_DIR, "holidays_events.csv")

# The exact features used in training (must match train.py)
CAT_COLS = ["store_nbr", "family"]
BASE_COLS = [
    "onpromotion", "dayofweek", "weekofyear", "month", "dayofyear",
    "dcoilwtico",
    "is_holiday_any", "is_holiday_national", "is_holiday_regional", "is_holiday_local",
]

# ---------------------------
# Cached loaders
# ---------------------------
@st.cache_data
def load_data():
    train = pd.read_csv(TRAIN_PATH, parse_dates=["date"])
    test = pd.read_csv(TEST_PATH, parse_dates=["date"])
    oil = pd.read_csv(OIL_PATH, parse_dates=["date"])
    hol = pd.read_csv(HOL_PATH, parse_dates=["date"])
    return train, test, oil, hol

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

# ---------------------------
# UI
# ---------------------------
st.title("Machine Learning–Driven Time Series Forecasting for Retail Sales Optimization")
st.caption("Store × Family daily sales forecasting (Kaggle-style: outputs id, sales).")

with st.sidebar:
    st.header("Settings")

    use_oil = st.checkbox("Use oil feature", value=True)
    use_holidays = st.checkbox("Use holidays feature", value=True)
    fourier_order = st.slider("Fourier order (monthly)", min_value=0, max_value=10, value=4)

    st.divider()
    st.subheader("Files")
    st.write("Model:", MODEL_PATH)
    st.write("Train:", TRAIN_PATH)
    st.write("Test:", TEST_PATH)

# ---------------------------
# Load data/model
# ---------------------------
if not os.path.exists(MODEL_PATH):
    st.error(f"Model not found at: {MODEL_PATH}. Run training first (python src/train.py).")
    st.stop()

train, test, oil, hol = load_data()
model = load_model()

# ---------------------------
# Data overview
# ---------------------------
st.subheader("Data Overview")
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Train rows", f"{len(train):,}")
with c2:
    st.metric("Test rows", f"{len(test):,}")
with c3:
    st.metric("Model file size (MB)", f"{os.path.getsize(MODEL_PATH)/1_000_000:.2f}")

tab1, tab2 = st.tabs(["Preview", "Generate Submission"])

with tab1:
    st.write("### Train preview")
    st.dataframe(train.head(10))

    st.write("### Test preview")
    st.dataframe(test.head(10))

with tab2:
    st.write("### Generate `submission.csv`")

    st.info(
        "This will rebuild features for train/test consistently, predict test sales, "
        "clip negatives to 0, and generate Kaggle-ready `id, sales`."
    )

    if st.button("Predict + Build Submission", type="primary"):
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

            fourier_cols = [c for c in full_train.columns if c.startswith("sin(") or c.startswith("cos(")]
            used_features = CAT_COLS + BASE_COLS + fourier_cols

            X_test = full_test[used_features]

        with st.spinner("Predicting..."):
            preds = model.predict(X_test)
            preds = np.clip(preds, 0, None)

        submission = pd.DataFrame({
            "id": full_test["id"].values,
            "sales": preds
        })

        st.success("Submission generated.")

        st.write("### Preview")
        st.dataframe(submission.head(20))

        # Optional: save to outputs folder
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(OUTPUT_DIR, f"submission_{timestamp}.csv")
        submission.to_csv(out_path, index=False)

        st.write("### Download")
        csv_bytes = submission.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download submission.csv",
            data=csv_bytes,
            file_name="submission.csv",
            mime="text/csv"
        )

        st.caption(f"Also saved locally to: {out_path}")