import logging
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report
from xgboost import XGBClassifier

logging.basicConfig(
    level=logging.INFO,
    format="[+%(relativeCreated)dms] %(levelname)s: %(message)s"
)

logger = logging.getLogger(__name__)
log = logger.info

config = {
    "filepath": "predictive_maintenance.csv",
    "random_state": 42,
    "test_size": 0.2,
    "n_splits": 5,
    "cost_missed_failure": 10,
    "cost_false_alarm": 1,
    "xgb_params": {
        "n_estimators": 200,
        "max_depth": 5,
        "learning_rate": 0.05
    },
    "artifact_path": "machine_failure_artifact.joblib",
    "target_columns": "Target",
    "required_columns": ['UDI', 'Product ID', 'Type', 'air_temperature',
       'process_temperature', 'rotational_speed', 'torque',
       'tool_wear', 'Target', 'Failure Type'],
    "categorical_columns": ["Type"],
    "numerical_columns": ["air_temperature", "process_temperature", "rotational_speed", "torque", "tool_wear"],
    "drop_columns": ["UDI", "Product ID", "Failure Type"]
}

COLUMN_MAPPING = {
    "Air temperature [K]": "air_temperature",
    "Process temperature [K]": "process_temperature",
    "Rotational speed [rpm]": "rotational_speed",
    "Torque [Nm]": "torque",
    "Tool wear [min]": "tool_wear"
}

class DataValidationError(Exception):
    """Raise this when there is a mismatch of columns and stuff"""
    pass

def load_dataset(config: dict) -> pd.DataFrame:
    if "filepath" not in config:
        raise DataValidationError("Filepath doesnt contain dataset!! Please fix and rerun.")
    df = pd.read_csv(config["filepath"])

    df = df.rename(columns=COLUMN_MAPPING)

    for col in config["required_columns"]:
        if col not in df.columns:
            raise DataValidationError(f"Dataset doesnt contains {col}, make sure u have the correct dataset")

    target = config["target_columns"]
    unique_values = set(df[target].unique())
    if not unique_values <= {0, 1}:
        raise DataValidationError(
            f"Target '{target}' must be binary (0/1), found values: {sorted(unique_values)}"
        )

    positive_rate = df[config["target_columns"]].mean() * 100
    log(f"Dataset has been loaded \n Shape: {df.shape} \nPositive Rate:{positive_rate:.2f}%")
    return df

def cleaning_dataset(df: pd.DataFrame, config: dict):
    X = df.drop(columns=config["drop_columns"] + [config["target_columns"]])
    y = df[config["target_columns"]]
    return X, y

def split_dataset(X: pd.DataFrame, y: pd.Series, config: dict):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["test_size"],
        random_state=config["random_state"],
        stratify=y,
    )
    log(
        f"Split done | train: {X_train.shape}, positive rate {y_train.mean() * 100:.2f}% "
        f"| test: {X_test.shape}, positive rate {y_test.mean() * 100:.2f}%"
    )
    return X_train, X_test, y_train, y_test

def build_pipeline(config: dict, scale_pos_weight: float = 1.0) -> Pipeline:
    if "scale_pos_weight" in config["xgb_params"]:
        raise DataValidationError(
            "'scale_pos_weight' must not be set in config['xgb_params']; "
            "it is computed from y_train and passed as a function argument."
        )

    encoding_preprocessor = ColumnTransformer(
            transformers=[
                ("num", "passthrough", config["numerical_columns"]),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), config["categorical_columns"])
            ]
    )

    model = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=config["random_state"],
        **config["xgb_params"]
    )

    return Pipeline(
        steps=[
            ("encoding", encoding_preprocessor),
            ("classifier", model)
        ])    

def select_threshold(X_train: pd.DataFrame, y_train: pd.Series, config: dict):
    pos_samples = (y_train == 1).sum()
    neg_samples = (y_train == 0).sum()

    scale_post_weight = neg_samples / pos_samples

    oof_standard = np.zeros(len(X_train))
    oof_weighted = np.zeros(len(X_train))

    skf = StratifiedKFold(
        n_splits=config["n_splits"],
        shuffle=True,
        random_state=config["random_state"]
    )

    for train_idx, val_idx in skf.split(X_train, y_train):
        X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]

        pipe_std = build_pipeline(config, scale_pos_weight=1.0)
        pipe_std.fit(X_tr, y_tr)
        oof_standard[val_idx] = pipe_std.predict_proba(X_val)[:,1]
        
        pipe_wgt = build_pipeline(config, scale_pos_weight=scale_post_weight)
        pipe_wgt.fit(X_tr, y_tr)
        oof_weighted[val_idx] = pipe_wgt.predict_proba(X_val)[:,1]

    def calculate_cost(y_true, probabilities, threshold):
        preds = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        return (fn * config["cost_missed_failure"]) + (fp * config["cost_false_alarm"])

    thresholds = np.linspace(0.01, 0.99, 100)
    
    best_cost_std = float("inf")
    best_thresh_std = 0.5
    best_cost_wgt = float("inf")
    best_thresh_wgt = 0.5
    
    for t in thresholds:
        cost_std = calculate_cost(y_train, oof_standard, t)
        if cost_std < best_cost_std:
            best_cost_std = cost_std
            best_thresh_std = t
            
        cost_wgt = calculate_cost(y_train, oof_weighted, t)
        if cost_wgt < best_cost_wgt:
            best_cost_wgt = cost_wgt
            best_thresh_wgt = t

    if best_cost_wgt <= best_cost_std:
        log(f"Winner: Weighted Strategy | Cost: {best_cost_wgt} | Threshold: {best_thresh_wgt:.4f}")
        final_pipeline = build_pipeline(config, scale_pos_weight=scale_post_weight)
        best_threshold = best_thresh_wgt
    else:
        log(f"Winner: Standard Strategy | Cost: {best_cost_std} | Threshold: {best_thresh_std:.4f}")
        final_pipeline = build_pipeline(config, scale_pos_weight=1.0)
        best_threshold = best_thresh_std

    final_pipeline.fit(X_train, y_train)

    return final_pipeline, best_threshold

def evaluate_and_save(pipeline: Pipeline, threshold: float, X_test: pd.DataFrame, y_test: pd.Series, config: dict):
    test_probs = pipeline.predict_proba(X_test)[:, 1]
    
    test_preds = (test_probs >= threshold).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y_test, test_preds).ravel()
    final_cost = (fn * config["cost_missed_failure"]) + (fp * config["cost_false_alarm"])
    
    print("\n" + "="*50)
    print("FINAL HOLDOUT TEST REPORT")
    print("="*50)
    print(f"Operational Decision Threshold: {threshold:.4f}")
    print(f"Total Business Operating Cost: ${final_cost:,}")
    print("\nConfusion Matrix:")
    print(f"True Negatives: {tn} | False Positives (False Alarms): {fp}")
    print(f"False Negatives (Missed Failures): {fn} | True Positives: {tp}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, test_preds))
    print("="*50)
    
    artifact = {
        "pipeline": pipeline,
        "threshold": threshold,
        "config": config
    }
    
    joblib.dump(artifact, config["artifact_path"])
    log(f"Production artifact successfully saved to: {config['artifact_path']}")

if __name__ == "__main__":
    df_raw = load_dataset(config)
    X, y = cleaning_dataset(df_raw, config)
    
    X_train, X_test, y_train, y_test = split_dataset(X, y, config)
    
    best_pipeline, optimal_threshold = select_threshold(X_train, y_train, config)
    
    evaluate_and_save(best_pipeline, optimal_threshold, X_test, y_test, config)
