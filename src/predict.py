import os
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

from features import build_full_features
from config import (
    TRAIN_PATH, TEST_PATH, OIL_PATH, HOLIDAYS_PATH,
    OUTPUT_DIR, MODEL_PATH, CAT_COLS, BASE_COLS, FOURIER_ORDER
)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading data...")
    train = pd.read_csv(TRAIN_PATH, parse_dates=["date"])
    test  = pd.read_csv(TEST_PATH,  parse_dates=["date"])
    oil   = pd.read_csv(OIL_PATH,   parse_dates=["date"])
    hol   = pd.read_csv(HOLIDAYS_PATH, parse_dates=["date"])

    print("Building features...")
    full_train, full_test = build_full_features(
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

    print("Loading model...")
    model = joblib.load(MODEL_PATH)

    print("Predicting...")
    preds = model.predict(full_test[features])
    preds = np.clip(preds, 0, None)

    submission = pd.DataFrame({
        "id":    full_test["id"].values,
        "sales": preds,
    })

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(OUTPUT_DIR, f"submission_{timestamp}.csv")
    submission.to_csv(out_path, index=False)

    print(f"Submission saved to: {out_path}")
    print(f"Shape: {submission.shape}")
    print(submission.head())


if __name__ == "__main__":
    main()