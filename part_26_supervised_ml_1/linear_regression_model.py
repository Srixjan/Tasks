from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import logging
import joblib
import os

CONFIG = {
    "test_size": 0.2,
    "random_state": 42,
    "target_col": "MedHouseVal",
    "model_path": "models/model.joblib"
}

class ModelNotFitted(Exception):
    """Error caused by not model not fitted perfectly"""
    pass

class LinearRegressionModel:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ])

        self.is_fitted = False

    def train(self, X_train, y_train) -> None:
        self.pipeline.fit(X_train, y_train)
        
        trained_regression = self.pipeline.named_steps['model']
        
        coefficients = trained_regression.coef_
        intercept = trained_regression.intercept_
        logging.info(f"coefficients: {coefficients}")
        logging.info(f"Intercept: {intercept}")
        self.is_fitted = True

    def predict(self, X) -> np.ndarray:
        if not self.is_fitted:
            raise ModelNotFitted("Please train your model first!!")
        return self.pipeline.predict(X)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        artifact = {
            "pipeline": self.pipeline,
            "config": self.config
        }
        joblib.dump(artifact, path)
        logging.info(f"Model saved into {path}")


    @classmethod
    def load(cls, path: str) -> "LinearRegressionModel":
        artifact = joblib.load(path)
        instance = cls(config=artifact["config"])
        instance.pipeline = artifact["pipeline"]
        instance.is_fitted = True
        return instance


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s %(levelname)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    housing = fetch_california_housing(as_frame=True)
    df = housing.frame
    logging.info(f"Data loaded: {df.shape[0]} rows, {df.shape[1] - 1} features")

    X = df.drop(columns=[CONFIG["target_col"]])
    y = df[CONFIG["target_col"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=CONFIG["test_size"], 
        random_state=CONFIG["random_state"]
    )

    logging.info(f"Train/test split: {len(X_train)} train / {len(X_test)} test (test_size={CONFIG['test_size']}, seed={CONFIG['random_state']})")

    my_model = LinearRegressionModel(config=CONFIG)
    my_model.train(X_train, y_train)
    
    pred_before = my_model.predict(X_test)
    logging.info(f"Prediction before save, first 5 {pred_before[:5]}")

    my_model.save(CONFIG["model_path"])

    reloaded_model = LinearRegressionModel.load(CONFIG["model_path"])
    preds_after = reloaded_model.predict(X_test)
    logging.info(f"Predictions after reload (first 5): {preds_after[:5]}")

    match = np.allclose(pred_before, preds_after)
    logging.info(f"Predictions identical before/after reload: {match}")