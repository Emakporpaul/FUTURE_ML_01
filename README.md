# Machine Learning–Driven Time Series Forecasting for Retail Sales Optimization

## Project Overview
This project builds a production-grade machine learning pipeline to forecast
daily product sales across multiple retail stores for Corporación Favorita,
a large retail chain operating in Ecuador.

Accurate sales forecasts enable:
- Better inventory control and reduced waste
- Smarter staffing and resource allocation
- More effective promotional planning
- Improved supply chain scheduling

---

## Business Problem
Predict daily sales for 54 stores × 33 product families (1,782 time series)
for a 16-day forecast horizon, using historical sales, promotions, oil prices,
and holiday events.

**Null Hypothesis (H₀):** Promotions have no measurable effect on sales  
**Alternative Hypothesis (H₁):** Promotions have a significant impact on sales

---

## Dataset
Source: [Kaggle — Store Sales Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)  
Provider: Corporación Favorita (Ecuador)

| File | Description |
|------|-------------|
| `train.csv` | Historical daily sales (2013–2017) |
| `test.csv` | 16-day forecast horizon |
| `oil.csv` | Daily oil prices (Ecuador is oil-dependent) |
| `holidays_events.csv` | National, regional, and local holidays |
| `stores.csv` | Store metadata (city, type, cluster) |

---

## Project Structure
FUTURE_ML_01/
src/
features.py       # Feature engineering pipeline
train.py          # Model training + saving
evaluate.py       # Validation metrics + charts
predict.py        # Generate submission.csv
app.py            # Streamlit forecasting app
config.py         # Central configuration
metrics.py        # RMSLE, RMSE, MAE utilities
baseline.py       # Baseline model
data/
raw/              # Raw Kaggle CSV files (not committed)
models/             # Saved model pipeline (.pkl)
outputs/            # Metrics, charts, submissions
reports/            # Project report
requirements.txt
README.md

---

## Approach

### Feature Engineering
- **Calendar features:** day of week, week of year, month, day of year
- **Fourier terms:** monthly seasonal sine/cosine terms (order 4)
- **Promotion feature:** `onpromotion` count per store-family
- **Oil price:** forward-filled daily Brent crude price
- **Holiday flags:** national, regional, and local holiday indicators

### Model
- **Algorithm:** XGBoost Regressor
- **Preprocessing:** OneHotEncoding for store and family; passthrough for numerics
- **Pipeline:** scikit-learn `Pipeline` (preprocessing + model) saved as `.pkl`
- **Validation:** Time-based split (last 10% of training dates held out)

### Baseline
Mean sales per store-family pair used as a naive baseline for comparison.

---

## Results

| Model | RMSLE | RMSE | MAE |
|-------|-------|------|-----|
| Baseline (Mean by Store-Family) | see outputs/metrics.json | — | — |
| XGBoost Pipeline | see outputs/metrics.json | — | — |

---

## How to Reproduce

### 1. Clone the repo
```bash
git clone https://github.com/Emakporpaul/retail-sales-forecasting.git
cd retail-sales-forecasting
```

### 2. Set up environment
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows
pip install -r requirements.txt
```

### 3. Add data files
Place Kaggle CSV files into `data/raw/`.

### 4. Train the model
```bash
python src/train.py
```

### 5. Evaluate
```bash
python src/evaluate.py
```

### 6. Generate submission
```bash
python src/predict.py
```

### 7. Run the app
```bash
streamlit run src/app.py
```

---

## Key Findings
- Strong weekly seasonality observed in sales patterns
- Holiday events (especially national holidays) cause significant sales spikes
- Promotions positively correlate with sales volume
- Oil price fluctuations show indirect influence on consumer spending

---

## Next Steps
- Add lag features (7-day, 14-day) for stronger time-series signals
- Apply log1p target transformation to reduce RMSLE
- Tune XGBoost hyperparameters with cross-validation
- Add store metadata features (city, type, cluster)
- Deploy app to Streamlit Community Cloud

---

## Author
**[EMAKPOR PAUL]**  
Machine Learning Intern — Future Interns Program  
April 2026