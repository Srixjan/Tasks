import logging
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline  
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

config = {
    "max_depth": 5,
    "random_stfate": 42,
    "cv_folds": 15,
    "max_depth_values": [3, 5, 8, 12, 15, 20]
}

logging.basicConfig(
    level = logging.INFO,
    format = "[+%(relativeCreated)dms] %(levelname)s: %(message)s"
)

logger = logging.getLogger(__name__)
log = logger.info

class BankMarketingClassifier:
    def __init__(self, config: dict):
        self.max_depth = config.get("max_depth", 5)
        self.random_state = config.get("random_state", 42)
        self.cv_folds = config.get("cv_folds", 10)
        self.max_depth_values = config.get("max_depth_values", [3, 5, 8, 12, 15, 20])


    def load_data(self) -> tuple:

        df = pd.read_csv(r"D:\development\aiml\tasks\part_31_supervised_ml_4\bank-full.csv", sep=";")
        
        log(f"Data shape: {df.shape}")
        log(f"Class Distribution:\n{df["y"].value_counts()}") 
        log(f"Class Distribtuion in percent{df["y"].value_counts(normalize=True) * 100}")
        numerical_features = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
        categorical_features = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome"]
        log(f"Numerical Features are {numerical_features}")
        log(f"Categorical Featuresa are{categorical_features}")
        log(f"Contains no null values")

        df['y'] = df['y'].map({'no': 0, 'yes': 1})
        
        X = df.drop(columns=["y"])
        y = df["y"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )

        log(f"Training Set size: {X_train.shape}, Testing Set size: {X_test.shape}")
        return X_train, X_test, y_train, y_test, numerical_features, categorical_features

    def preprocess(self, X_train, X_test, numerical_features, categorical_features) -> tuple:
        encoding_preprocessor = ColumnTransformer(
            transformers=[
                ("num_encode", "passthrough", numerical_features),
                ("cat_encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
            ]
        )

        X_train_trans = encoding_preprocessor.fit_transform(X_train)
        X_test_trans = encoding_preprocessor.transform(X_test)

        log(f"Training set preprocessed: {X_train_trans.shape}, Testing set processed: {X_test_trans.shape}")

        return X_train_trans, X_test_trans, encoding_preprocessor

    def cross_validator(self, X_train, y_train, max_depth_values):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        results = {}

        for depth in max_depth_values:
            train_acc = []
            val_acc = []

            for fold_idx, (train_idx, val_idx) in enumerate (skf.split(X_train, y_train)):
                X_fold_train = X_train[train_idx]
                y_fold_train = y_train.iloc[train_idx]
                X_fold_val = X_train[val_idx]
                y_fold_val = y_train.iloc[val_idx]

                clf = DecisionTreeClassifier(max_depth=depth, random_state=self.random_state)
                clf.fit(X_fold_train, y_fold_train)

                pred_train = clf.predict(X_fold_train)
                pred_val = clf.predict(X_fold_val)

                train_acc_fold = accuracy_score(y_fold_train, pred_train)
                val_acc_fold = accuracy_score(y_fold_val, pred_val)

                train_acc.append(train_acc_fold)
                val_acc.append(val_acc_fold)

                overfitting_gap = train_acc_fold - val_acc_fold
                log(f"Depth {depth}, Fold {fold_idx}: train_acc={train_acc_fold:.4f}, val_acc={val_acc_fold:.4f}, gap={overfitting_gap:.4f}")

            mean_train = np.mean(train_acc)
            std_train = np.std(train_acc)
            mean_val = np.mean(val_acc)
            std_val = np.std(val_acc)
            overfitting_gap = mean_train - mean_val

            results[depth] = {
                "train_acc": mean_train,
                "val_acc": mean_val,
                "std_train": std_train,
                "std_val": std_val,
                "overfitting_gap": overfitting_gap
            }

            log(f"\n=== DEPTH {depth} SUMMARY ===")
            log(f"Train accuracy: {mean_train:.4f} ± {std_train:.4f}")
            log(f"Val accuracy: {mean_val:.4f} ± {std_val:.4f}")
            log(f"Overfitting gap: {overfitting_gap:.4f}\n")
            
        return results

    def evaluate(self, X_train, y_train, X_test, y_test, best_depth, feature_names):
    
        clf = DecisionTreeClassifier(max_depth=best_depth, random_state=self.random_state)

        clf.fit(X_train, y_train)
        pred_train = clf.predict(X_train)
        pred_test = clf.predict(X_test)
    
        train_acc = accuracy_score(y_train, pred_train)
        test_acc = accuracy_score(y_test, pred_test)

        precision = precision_score(y_test, pred_test)
        recall = recall_score(y_test, pred_test)
        f1 = f1_score(y_test, pred_test)
        conf_matrix = confusion_matrix(y_test, pred_test)

        top_5_idx = np.argsort(clf.feature_importances_)[-5:][::-1]
        top_5_features = [(feature_names[i], clf.feature_importances_[i]) for i in top_5_idx]

        roc_auc = roc_auc_score(y_test, pred_test)

        log(f"Train Accuracy: {train_acc:.4f}")
        log(f"Test Accuracy: {test_acc:.4f}")
        log(f"Overfitting Gap: {train_acc - test_acc:.4f}")
        log(f"Precision: {precision:.4f}")
        log(f"Recall: {recall:.4f}")
        log(f"F1 Score: {f1:.4f}")
        log(f"ROC-AUC: {roc_auc:.4f}")
        log(f"Confusion Matrix:\n{conf_matrix}")
        log(f"Top 5 Features:")
        for feat, imp in top_5_features:
            log(f"  {feat}: {imp:.4f}")

        return {
        "train_acc": train_acc,
        "test_acc": test_acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": conf_matrix,
        "top_5_features": top_5_features
    }

if __name__ == "__main__":
    classifier = BankMarketingClassifier(config)
    
    X_train, X_test, y_train, y_test, numerical_features, categorical_features = classifier.load_data()
    X_train_proc, X_test_proc, preprocessor = classifier.preprocess(X_train, X_test, numerical_features, categorical_features)
    
    log("\n=== STARTING CROSS-VALIDATION ===")
    cv_results = classifier.cross_validator(X_train_proc, y_train, config["max_depth_values"])
    
    best_depth = max(cv_results, key=lambda d: cv_results[d]["val_acc"])
    log(f"\n=== BEST DEPTH: {best_depth} ===")
    
    log("\n=== FINAL EVALUATION ON TEST SET ===")
    feature_names = numerical_features + list(preprocessor.named_transformers_['cat_encode'].get_feature_names_out(categorical_features))
    metrics = classifier.evaluate(X_train_proc, y_train, X_test_proc, y_test, best_depth, feature_names)
    
    log("\n=== ASSIGNMENT COMPLETE ===")
