import pandas as pd

def mean_by_store_family(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Baseline: predict mean sales per (store_nbr, family).
    Returns mapping table with columns: store_nbr, family, mean_sales
    """
    m = (
        train_df.groupby(["store_nbr", "family"], observed=False)["sales"]
        .mean()
        .reset_index()
        .rename(columns={"sales": "mean_sales"})
    )
    return m

def predict_baseline(test_df: pd.DataFrame, mapping: pd.DataFrame) -> pd.Series:
    pred_df = test_df.merge(mapping, on=["store_nbr", "family"], how="left")
    pred_df["mean_sales"] = pred_df["mean_sales"].fillna(0.0)
    return pred_df["mean_sales"]