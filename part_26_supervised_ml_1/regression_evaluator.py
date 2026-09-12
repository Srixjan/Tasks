from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import logging

class MissmatchedLengthError(Exception):
    """This is raised, when y_true and y_pred dont match samples"""
    pass

def evaluate_regression(y_true, y_pred, n_features: int) -> dict:
    if len(y_true) != len(y_pred):
        raise MissmatchedLengthError(f"y_true has {len(y_true)} samples, y_pred has {len(y_pred)} — cannot evaluate.")


    n_samples = len(y_true)
    p = n_features
    
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(mse ** 0.5)
    r2 = float(r2_score(y_true, y_pred))
    adj_r2 = float(1 - ((1 - r2) * (n_samples - 1) / (n_samples - p - 1)))

    result = {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "adj_r2": adj_r2
    }

    logging.info(f"METRICS: {result}")
    return result

def print_metrics_table(metrics: dict) -> None:
    print("=" * 40)
    print(f"{'METRIC':<15}{'VALUE':>15}")
    print("-" * 40)
    print(f"{'MAE':<15}{metrics['mae']:>14.2f}")
    print(f"{'MSE':<15}{metrics['mse']:>14.2f}")
    print(f"{'RMSE':<15}{metrics['rmse']:>14.2f}")
    print(f"{'R2':<15}{metrics['r2']*100:>13.2f}%")
    print(f"{'Adj R2':<15}{metrics['adj_r2']*100:>13.2f}%")
    print("=" * 40)