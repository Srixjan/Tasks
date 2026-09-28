import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

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


def tune_and_train_svm(X_train, y_train, X_test, y_test):
    
    svm_pipeline = Pipeline([
        ("safety_impute", SimpleImputer(strategy='mean')),
        ("scaler_features", StandardScaler()),
        ("pca", PCA(random_state=42)),
        ("SVC", SVC(class_weight='balanced', random_state=42))
    ])

    param_grid = {
        'pca__n_components': [10, 30, 50, 100, 0.90, 0.95],
        'SVC__C': [0.1, 1, 10, 100],
        'SVC__kernel': ['linear', 'sigmoid', 'rbf', 'poly']
    }

    grid_search_svc = GridSearchCV(
        estimator=svm_pipeline,
        param_grid=param_grid,
        cv=6,
        scoring='recall',
        n_jobs=-1,
        verbose=1
    )

    grid_search_svc.fit(X_train, y_train)

    best_model = grid_search_svc.best_estimator_
    y_pred = best_model.predict(X_test)

    logger.info("\n=== Model Performance ===")
    logger.info(classification_report(y_test, y_pred))
    logger.info("=== Confusion Matrix ===")
    logger.info(confusion_matrix(y_test, y_pred))
    logger.info(f"Best params: {grid_search_svc.best_params_}")
    logger.info(f"Best CV score: {grid_search_svc.best_score_:.3f}")
    
    return best_model, y_pred