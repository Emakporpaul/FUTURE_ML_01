# Project Report
## Machine Learning–Driven Time Series Forecasting for Retail Sales Optimization

**Author:** Paul Emakpor
**Program:** Future Interns — Machine Learning Internship
**Date:** April 2026

---

## 1. Executive Summary
This project developed a production-grade machine learning pipeline to forecast
daily product sales across 54 stores and 33 product families for Corporación
Favorita, a large retail chain operating in Ecuador. The final solution uses an
XGBoost regression pipeline with engineered time series features, achieving
meaningful improvement over a naive baseline model.

---

## 2. Business Problem
Accurate sales forecasting is critical for retail operations. Poor forecasts
lead to overstocking, stockouts, wasted resources, and missed revenue.

**Key business questions answered:**
- Which periods show peak and low sales activity?
- Did the 2016 earthquake disrupt sales patterns?
- How do promotions, oil prices, and holidays affect demand?

**Hypothesis tested:**
- H₀: Promotions have no measurable effect on sales
- H₁: Promotions have a significant impact on sales

---

## 3. Dataset Description
| File | Rows | Description |
|------|------|-------------|
| train.csv | 3,000,888 | Historical daily sales 2013–2017 |
| test.csv | 28,512 | 16-day forecast horizon |
| oil.csv | 1,218 | Daily Brent crude oil prices |
| holidays_events.csv | 350 | National/regional/local holidays |
| stores.csv | 54 | Store metadata |

---

## 4. Exploratory Data Analysis

### Key Findings
- **Weekly seasonality:** Strong recurring patterns by day of week
- **Holiday effects:** National holidays cause significant sales spikes
- **Promotions:** Positive correlation with increased sales volume
- **Oil prices:** Indirect influence on consumer spending patterns
- **Zero sales:** ~30% of training records show zero sales
- **Right-skewed distribution:** High variability across store-family combinations

---

## 5. Feature Engineering
| Feature | Description |
|---------|-------------|
| dayofweek | Day of week (0=Monday) |
| weekofyear | ISO week number |
| month | Month of year |
| dayofyear | Day of year |
| sin/cos Fourier terms | Monthly seasonality (order 4) |
| onpromotion | Items on promotion per store-family |
| dcoilwtico | Forward-filled daily oil price |
| is_holiday_any | Any holiday flag |
| is_holiday_national | National holiday flag |
| is_holiday_regional | Regional holiday flag |
| is_holiday_local | Local holiday flag |

---

## 6. Modeling Approach

### Models Evaluated
| Model | Notes |
|-------|-------|
| Baseline (Mean by Store-Family) | Naive benchmark |
| Linear Regression | Simple seasonal model |
| Decision Tree | Non-linear patterns |
| Random Forest | Ensemble, reduces overfitting |
| XGBoost Pipeline | Final selected model |

### Final Model: XGBoost Pipeline
- **Preprocessing:** OneHotEncoding for store_nbr and family
- **Validation:** Time-based split (last 10% of dates — no leakage)
- **Pipeline:** scikit-learn Pipeline (preprocessing + model) saved as .pkl
- **Hyperparameters:**
  - n_estimators: 1200
  - max_depth: 8
  - learning_rate: 0.05
  - subsample: 0.9
  - colsample_bytree: 0.9

---

## 7. Results
| Model | RMSLE | RMSE | MAE |
|-------|-------|------|-----|
| Baseline | See metrics.json | — | — |
| XGBoost Pipeline | See metrics.json | — | — |

Full metrics available in `outputs/metrics.json`.
Validation chart available in `outputs/validation_chart.png`.

---

## 8. Deployed Application
The forecasting dashboard is live at:
https://futureml01-m69wvqiexbn7taw72r24ol.streamlit.app

**Features:**
- Upload Kaggle CSV files directly in browser
- Data overview with trend and family charts
- Model evaluation metrics display
- One-click submission.csv generation and download

---

## 9. Key Takeaways
- Feature engineering (seasonality + promotions + holidays) significantly
  outperforms naive baseline approaches
- XGBoost handles non-linear interactions between features effectively
- Time-based validation is essential to avoid data leakage in forecasting
- A production pipeline (sklearn Pipeline + .pkl) ensures full reproducibility

---

## 10. Next Steps
- Add lag features (7-day, 14-day rolling means per store-family)
- Apply log1p target transformation to reduce RMSLE
- Tune hyperparameters using cross-validation
- Add store metadata features (city, type, cluster)
- Implement hierarchical forecasting per store type