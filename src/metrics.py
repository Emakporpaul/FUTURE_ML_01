import numpy as np
from sklearn.metrics import mean_squared_log_error, mean_squared_error, mean_absolute_error

def clip_nonnegative(y):
    return np.clip(y, 0, None)

def rmsle(y_true, y_pred):
    y_pred = clip_nonnegative(y_pred)
    return float(np.sqrt(mean_squared_log_error(y_true, y_pred)))

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def mae(y_true, y_pred):
    return float(mean_absolute_error(y_true, y_pred))