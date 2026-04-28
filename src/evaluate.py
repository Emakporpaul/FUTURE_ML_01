import os
import json
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from features import build_full_features
from config import (
    TRAIN_PATH, TEST_PATH, OIL_PATH, HOLIDAYS_PATH,
    OUTPUT_DIR, MODEL_PATH, CAT_COLS, BASE_COLS, FOURIER_ORDER
)
from metrics import rmsle, rmse, mae
from baseline import mean_by_store_family, predict_baseline


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading data...")
    train = pd.read_csv(TRAIN_PATH, parse_dates=["date"])
    test  = pd.read_csv(TEST_PATH, parse_dates=["date"])
    oil   = pd.read_csv(OIL_PATH, parse_dates=["date"])
    hol   = pd.read_csv(HOLIDAYS_PATH, parse_dates=["date"])

    print("Building features...")
    full_train, _ = build_full_features(
        train_df=train,
        test_df=test,
        oil_df=oil,
        hol_df=hol,
        use_oil=True,
        use_holidays=True,
        fourier_order=FOURIER_ORDER,
    )

    fourier_cols = [c for c in full_train.columns if c.startswith("sin(") or c.startswith("cos(")]
    features = CAT_COLS + BASE_COLS + fourier_cols

    # Time-based split (last 10% of dates = validation)
    df = full_train.copy()
    df["date_dt"] = pd.to_datetime(df["date"])
    df = df.sort_values("date_dt")

    unique_dates = df["date_dt"].unique()
    cutoff = unique_dates[int(len(unique_dates) * 0.9)]
    print(f"Validation cutoff date: {pd.to_datetime(cutoff).date()}")

    tr = df[df["date_dt"] <= cutoff]
    va = df[df["date_dt"] > cutoff]

    y_va = va["sales"].astype(float).values

    # --- Baseline predictions ---
    print("Evaluating baseline...")
    mapping = mean_by_store_family(tr[["store_nbr", "family", "sales"]])
    base_pred = predict_baseline(
        va[["store_nbr", "family"]],
        mapping
    ).values
    base_pred = np.clip(base_pred, 0, None)

    # --- Model predictions ---
    print("Evaluating XGBoost pipeline...")
    model = joblib.load(MODEL_PATH)
    model_pred = model.predict(va[features])
    model_pred = np.clip(model_pred, 0, None)

    # --- Metrics ---
    results = {
        "cutoff_date": str(pd.to_datetime(cutoff).date()),
        "n_train_rows": int(len(tr)),
        "n_valid_rows": int(len(va)),
        "baseline_mean_by_store_family": {
            "rmsle": round(rmsle(y_va, base_pred), 5),
            "rmse":  round(rmse(y_va, base_pred), 5),
            "mae":   round(mae(y_va, base_pred), 5),
        },
        "xgb_pipeline": {
            "rmsle": round(rmsle(y_va, model_pred), 5),
            "rmse":  round(rmse(y_va, model_pred), 5),
            "mae":   round(mae(y_va, model_pred), 5),
        },
    }

    # Print results
    print("\n========== EVALUATION RESULTS ==========")
    print(json.dumps(results, indent=2))

    # Save metrics JSON
    metrics_path = os.path.join(OUTPUT_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nMetrics saved to: {metrics_path}")

    # --- Chart: Actual vs Predicted (aggregate daily) ---
    print("Generating chart...")
    va_copy = va.copy()
    va_copy["model_pred"] = model_pred
    va_copy["base_pred"]  = base_pred
    va_copy["date_dt"]    = pd.to_datetime(va_copy["date"])

    daily = va_copy.groupby("date_dt").agg(
        actual=("sales", "mean"),
        model=("model_pred", "mean"),
        baseline=("base_pred", "mean"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(daily["date_dt"], daily["actual"],   label="Actual",          color="black",  linewidth=2)
    ax.plot(daily["date_dt"], daily["model"],    label="XGBoost",         color="steelblue", linewidth=2, linestyle="--")
    ax.plot(daily["date_dt"], daily["baseline"], label="Baseline (Mean)", color="orange", linewidth=1.5, linestyle=":")
    ax.set_title("Validation: Actual vs Predicted (Daily Average Sales)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Avg Sales")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    chart_path = os.path.join(OUTPUT_DIR, "validation_chart.png")
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"Chart saved to: {chart_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()