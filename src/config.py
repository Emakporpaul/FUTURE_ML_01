DATA_DIR = "data/raw"
MODEL_DIR = "models"
OUTPUT_DIR = "outputs"
LOG_DIR = "logs"

TRAIN_PATH = f"{DATA_DIR}/train.csv"
TEST_PATH = f"{DATA_DIR}/test.csv"
OIL_PATH = f"{DATA_DIR}/oil.csv"
HOLIDAYS_PATH = f"{DATA_DIR}/holidays_events.csv"

MODEL_PATH = f"{MODEL_DIR}/xgb_store_family_pipeline.pkl"

CAT_COLS = ["store_nbr", "family"]
BASE_COLS = [
    "onpromotion", "dayofweek", "weekofyear", "month", "dayofyear",
    "dcoilwtico",
    "is_holiday_any", "is_holiday_national", "is_holiday_regional", "is_holiday_local",
]
FOURIER_ORDER = 4