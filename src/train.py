import pandas as pd
import numpy as np
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_log_error
from xgboost import XGBRegressor

from features import build_full_features


def rmsle(y_true, y_pred):
    y_pred = np.clip(y_pred, 0, None)
    return float(np.sqrt(mean_squared_log_error(y_true, y_pred)))


def main():
    # 1) Load raw data
    train = pd.read_csv("data/raw/train.csv", parse_dates=["date"])
    test = pd.read_csv("data/raw/test.csv", parse_dates=["date"])
    oil = pd.read_csv("data/raw/oil.csv", parse_dates=["date"])
    holidays = pd.read_csv("data/raw/holidays_events.csv", parse_dates=["date"])

    # 2) Feature engineering (same for train and test)
    full_train, _ = build_full_features(
        train_df=train,
        test_df=test,
        oil_df=oil,
        hol_df=holidays,
        use_oil=True,
        use_holidays=True,
        fourier_order=4,
    )

    # 3) Define feature columns (must stay consistent for app + predict)
    cat_cols = ["store_nbr", "family"]
    base_cols = [
        "onpromotion",
        "dayofweek",
        "weekofyear",
        "month",
        "dayofyear",
        "dcoilwtico",
        "is_holiday_any",
        "is_holiday_national",
        "is_holiday_regional",
        "is_holiday_local",
    ]
    fourier_cols = [c for c in full_train.columns if c.startswith("sin(") or c.startswith("cos(")]
    features = cat_cols + base_cols + fourier_cols

    # 4) Time-based validation split by date (no leakage)
    df_tmp = full_train.copy()
    df_tmp["date_dt"] = pd.to_datetime(df_tmp["date"])
    df_tmp = df_tmp.sort_values("date_dt")

    unique_dates = df_tmp["date_dt"].unique()
    cutoff = unique_dates[int(len(unique_dates) * 0.9)]

    train_mask = df_tmp["date_dt"] <= cutoff
    valid_mask = df_tmp["date_dt"] > cutoff

    X_tr = df_tmp.loc[train_mask, features]
    y_tr = df_tmp.loc[train_mask, "sales"].astype(float)

    X_va = df_tmp.loc[valid_mask, features]
    y_va = df_tmp.loc[valid_mask, "sales"].astype(float)

    # 5) Preprocessing (one-hot for store_nbr/family)
    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", [c for c in features if c not in cat_cols]),
        ],
        remainder="drop",
    )

    # 6) XGBoost model (solid baseline)
    model = XGBRegressor(
        n_estimators=1200,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="reg:squarederror",
        n_jobs=-1,
    )

    pipe = Pipeline(steps=[("prep", preprocess), ("model", model)])

    print("Training...")
    pipe.fit(X_tr, y_tr)

    print("Validating...")
    preds = pipe.predict(X_va)
    score = rmsle(y_va.values, preds)
    print(f"Validation RMSLE: {score:.5f}")

    # 7) Save model pipeline (professional: includes preprocessing + model)
    out_path = "models/xgb_store_family_pipeline.pkl"
    joblib.dump(pipe, out_path)
    print(f"Saved model to: {out_path}")


if __name__ == "__main__":
    main()