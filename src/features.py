import pandas as pd
import numpy as np
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess


def add_date_features(df, date_col="date"):
    df = df.copy()
    d = pd.to_datetime(df[date_col])
    df["dayofweek"] = d.dt.dayofweek.astype(int)
    df["weekofyear"] = d.dt.isocalendar().week.astype(int)
    df["month"] = d.dt.month.astype(int)
    df["dayofyear"] = d.dt.dayofyear.astype(int)
    return df


def add_fourier_features(df, date_col="date", order=4):
    df = df.copy()
    idx = pd.to_datetime(df[date_col]).dt.to_period("D")

    fourier = CalendarFourier(freq="ME", order=order)
    dp = DeterministicProcess(
        index=idx,
        constant=False,
        order=0,
        seasonal=False,
        additional_terms=[fourier],
        drop=True,
    )

    fourier_df = dp.in_sample().reset_index(drop=True)
    df = df.reset_index(drop=True)
    return pd.concat([df, fourier_df], axis=1)


def prep_oil(oil_df):
    oil = oil_df.copy()
    oil["date"] = pd.to_datetime(oil["date"])
    oil = oil.sort_values("date")
    oil["dcoilwtico"] = oil["dcoilwtico"].ffill()
    return oil[["date", "dcoilwtico"]]


def prep_holidays(hol_df):
    hol = hol_df.copy()
    hol["date"] = pd.to_datetime(hol["date"])

    hol["is_holiday_any"] = 1
    if "locale" in hol.columns:
        hol["is_holiday_national"] = (hol["locale"] == "National").astype(int)
        hol["is_holiday_regional"] = (hol["locale"] == "Regional").astype(int)
        hol["is_holiday_local"] = (hol["locale"] == "Local").astype(int)
    else:
        hol["is_holiday_national"] = 0
        hol["is_holiday_regional"] = 0
        hol["is_holiday_local"] = 0

    daily = (
        hol.groupby("date")[[
            "is_holiday_any",
            "is_holiday_national",
            "is_holiday_regional",
            "is_holiday_local",
        ]]
        .max()
        .reset_index()
    )
    return daily


def build_full_features(
    train_df,
    test_df,
    oil_df=None,
    hol_df=None,
    use_oil=True,
    use_holidays=True,
    fourier_order=4,
):
    train_df = train_df.copy()
    test_df = test_df.copy()

    train_df["is_train"] = 1
    test_df["is_train"] = 0
    if "sales" not in test_df.columns:
        test_df["sales"] = np.nan

    full = pd.concat([train_df, test_df], axis=0, ignore_index=True)

    full = add_date_features(full, "date")
    full = add_fourier_features(full, "date", order=fourier_order)

    if use_oil and oil_df is not None:
        oil_clean = prep_oil(oil_df)
        full["date_dt"] = pd.to_datetime(full["date"])
        full = full.merge(oil_clean, left_on="date_dt", right_on="date", how="left", suffixes=("", "_oil"))
        full.drop(columns=["date_oil"], inplace=True)
        full["dcoilwtico"] = full["dcoilwtico"].ffill()
        full.drop(columns=["date_dt"], inplace=True)
    else:
        full["dcoilwtico"] = np.nan

    if use_holidays and hol_df is not None:
        hol_daily = prep_holidays(hol_df)
        full = full.merge(hol_daily, on="date", how="left")
        for c in ["is_holiday_any", "is_holiday_national", "is_holiday_regional", "is_holiday_local"]:
            full[c] = full[c].fillna(0).astype(int)
    else:
        full["is_holiday_any"] = 0
        full["is_holiday_national"] = 0
        full["is_holiday_regional"] = 0
        full["is_holiday_local"] = 0

    full_train = full[full["is_train"] == 1].copy()
    full_test = full[full["is_train"] == 0].copy()
    return full_train, full_test