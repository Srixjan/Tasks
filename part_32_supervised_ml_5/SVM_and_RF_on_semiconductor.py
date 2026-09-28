import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer

logging.basicConfig(
    level=logging.INFO,
    format="{asctime} [{levelname:<8}] {message}",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def load_and_preprocess(filepath):
    df = pd.read_csv(filepath)
    logger.info(f"Dataset loaded: shape {df.shape}")

    y = df.iloc[:, -1].replace({-1: 0}).astype(int).values
    X = df.iloc[:, :-1].copy()
    logger.info(f"Target extracted. Class distribution: {np.bincount(y)}")
    
    X = X.drop(columns="Time")
    logger.info(f"Irrelevant columns dropped. Features shape: {X.shape}")
    
    imputer = SimpleImputer(strategy="mean")
    X_imputed = pd.DataFrame(
        imputer.fit_transform(X),
        columns=X.columns,
        index=X.index
    )
    logger.info(f"Missing values imputed with mean strategy")

    selector = VarianceThreshold(threshold=0.0001)
    X_varianced = selector.fit_transform(X_imputed)
    logger.info(f"Low-variance features removed. Remaining: {X_varianced.shape[1]}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_varianced,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    logger.info(f"Train/test split: {X_train.shape} / {X_test.shape}")
    
    return X_train, X_test, y_train, y_test
