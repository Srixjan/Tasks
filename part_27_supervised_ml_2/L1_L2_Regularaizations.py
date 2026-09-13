import pandas as pd
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge, Lasso

X, y = make_regression(n_samples=100, n_features=10, n_informative=4, noise=15, random_state=42)

def compare_regularization(X: np.ndarray, y: np.ndarray, alphas: list[float]) -> pd.DataFrame:
    results = []
    for al in alphas:
        ridge = Ridge(alpha=al)
        ridge.fit(X, y)
        rid_non_zero = np.sum(np.abs(ridge.coef_) > 1e-4)

        lasso = Lasso(alpha=al)
        lasso.fit(X, y)
        las_non_zero = np.sum(np.abs(lasso.coef_) > 1e-4)

        results.append({
            "Alpha": al,
            "Model": "Ridge",
            "Non_Zero_Coeffs": rid_non_zero,
            "R2_Score": ridge.score(X, y)
        })
        results.append({
            "Alpha": al,
            "Model": "Lasso",
            "Non_Zero_Coeffs": las_non_zero,
            "R2_Score": lasso.score(X, y)
        })

    return pd.DataFrame(results)

compare_regularization(X, y, alphas=[0.01, 1, 10, 100])