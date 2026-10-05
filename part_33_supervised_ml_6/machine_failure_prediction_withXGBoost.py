import pandas as pd
import logging

logging.basicConfig(
    level = logging.INFO,
    format = "[+%(relativeCreated)dms] %(levelname)s: %(message)s"
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
    "artificat_path" : "machine_failure_artifact.joblib",
    "target_columns": "Target",
    "required_columns": ['UDI', 'Product ID', 'Type', 'Air temperature [K]',
       'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]',
       'Tool wear [min]', 'Target', 'Failure Type'],
    "drop_columns": ["UDI", "Product ID", "Failure Type"],
    "categorical_columns": ["Type"],
    "numerical_columns": ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
}

class DataValidationError(Exception):
    pass

def load_dataset(config: dict) -> pd.DataFrame:
    if "filepath" not in config:
        raise DataValidationError("Filepath doesnt contain dataset!! Please fix and rerun.")
    df = pd.read_csv(config["filepath"])

    for col in config["required_columns"]:
        if col not in df.columns:
            raise DataValidationError(f"Dataset doesnt contains {col}, make sure u have the correct dataset")


    positive_rate = df[config["target_columns"]].mean() * 100
    log(f"Dataset has been loaded \n Shape: {df.shape} \nPositive Rate:{positive_rate:.2f}%")
    return df
    # shape (10,000, 10)
    # positive rate: 3.39%

def cleaning_dataset(df: pd.DataFrame):
    return