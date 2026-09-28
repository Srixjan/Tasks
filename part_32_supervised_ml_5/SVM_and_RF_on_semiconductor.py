import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    roc_curve, auc
)

logging.basicConfig(
    level=logging.INFO,
    format="{asctime} [{levelname:<8}] {message}",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# ============ LOAD & PREPROCESS ============
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
        X_varianced, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"Train/test split: {X_train.shape} / {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

# ============ SVM ============
def tune_and_train_svm(X_train, y_train, X_test, y_test) -> dict:
    logger.info("\n--- SVM WITH PCA ---")
    
    pipeline = Pipeline([
        ("impute", SimpleImputer(strategy='mean')),
        ("scaler", StandardScaler()),
        ("pca", PCA(random_state=42)),
        ("svm", SVC(class_weight='balanced', random_state=42, probability=True))
    ])

    param_grid = {
        'pca__n_components': [30, 50, 100],
        'svm__C': [0.1, 1, 10, 100],
        'svm__kernel': ['linear', 'rbf']
    }

    grid = GridSearchCV(
        pipeline, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1
    )
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    logger.info(f"Best params: {grid.best_params_}")
    logger.info(f"Best CV score: {grid.best_score_:.3f}")
    logger.info(f"\n{classification_report(y_test, y_pred)}")
    logger.info(f"ROC-AUC: {roc_auc:.3f}")
    
    return {
        'model': best_model,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'best_cv_score': grid.best_score_,
        'roc_auc': roc_auc,
        'report': report
    }

# ============ RANDOM FOREST ============
def tune_and_train_rf(X_train, y_train, X_test, y_test) -> dict:
    logger.info("\n--- RANDOM FOREST ---")
    
    pipeline = Pipeline([
        ('impute', SimpleImputer(strategy='mean')),
        ('rf', RandomForestClassifier(class_weight='balanced', random_state=42, oob_score=True, n_jobs=-1))
    ])
 
    param_grid = {
        'rf__n_estimators': [100, 200, 300],
        'rf__max_depth': [20, 30, 50],
        'rf__min_samples_split': [2, 5],
        'rf__min_samples_leaf': [1, 2],
        'rf__class_weight': ['balanced', 'balanced_subsample']
    }

    grid = GridSearchCV(
        pipeline, param_grid, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='roc_auc', n_jobs=-1, verbose=1
    )
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    
    report = classification_report(y_test, y_pred, output_dict=True)
    roc_auc = roc_auc_score(y_test, y_proba)
    oob_score = best_model.named_steps['rf'].oob_score_

    logger.info(f"Best params: {grid.best_params_}")
    logger.info(f"Best CV score: {grid.best_score_:.3f}")
    logger.info(f"OOB Score: {oob_score:.3f}")
    logger.info(f"\n{classification_report(y_test, y_pred)}")
    logger.info(f"ROC-AUC: {roc_auc:.3f}")
    
    return {
        'model': best_model,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'best_cv_score': grid.best_score_,
        'oob_score': oob_score,
        'roc_auc': roc_auc,
        'report': report
    }

# ============ COMPARE ============
def compare_models(svm_results, rf_results, y_test):
    logger.info("\n" + "="*60)
    logger.info("MODEL COMPARISON: SVM vs RANDOM FOREST")
    logger.info("="*60)
    
    comparison = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision (Failures)', 'Recall (Failures)', 'F1 (Failures)', 'ROC-AUC', 'CV Score'],
        'SVM': [
            svm_results['report']['accuracy'],
            svm_results['report']['1']['precision'],
            svm_results['report']['1']['recall'],
            svm_results['report']['1']['f1-score'],
            svm_results['roc_auc'],
            svm_results['best_cv_score']
        ],
        'Random Forest': [
            rf_results['report']['accuracy'],
            rf_results['report']['1']['precision'],
            rf_results['report']['1']['recall'],
            rf_results['report']['1']['f1-score'],
            rf_results['roc_auc'],
            rf_results['best_cv_score']
        ]
    })
    
    logger.info(f"\n{comparison.to_string(index=False)}")
    return comparison

def plot_results(svm_results, rf_results, y_test):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Confusion matrices
    svm_cm = confusion_matrix(y_test, svm_results['y_pred'])
    rf_cm = confusion_matrix(y_test, rf_results['y_pred'])
    
    axes[0, 0].imshow(svm_cm, cmap='Blues', aspect='auto')
    axes[0, 0].set_title('SVM Confusion Matrix')
    for i in range(2):
        for j in range(2):
            axes[0, 0].text(j, i, svm_cm[i, j], ha='center', va='center', color='white')
    
    axes[0, 1].imshow(rf_cm, cmap='Blues', aspect='auto')
    axes[0, 1].set_title('Random Forest Confusion Matrix')
    for i in range(2):
        for j in range(2):
            axes[0, 1].text(j, i, rf_cm[i, j], ha='center', va='center', color='white')
    
    # ROC curves
    svm_fpr, svm_tpr, _ = roc_curve(y_test, svm_results['y_proba'])
    rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_results['y_proba'])
    
    axes[1, 0].plot(svm_fpr, svm_tpr, lw=2, label=f"SVM (AUC={svm_results['roc_auc']:.3f})")
    axes[1, 0].plot([0, 1], [0, 1], 'k--', lw=1)
    axes[1, 0].set_title('SVM ROC Curve')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    axes[1, 1].plot(rf_fpr, rf_tpr, lw=2, color='orange', label=f"RF (AUC={rf_results['roc_auc']:.3f})")
    axes[1, 1].plot([0, 1], [0, 1], 'k--', lw=1)
    axes[1, 1].set_title('Random Forest ROC Curve')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('secom_comparison.png', dpi=100)
    logger.info("Plots saved to 'secom_comparison.png'")
    plt.close()

def deployment_recommendation(svm_results, rf_results):
    logger.info("\n" + "="*60)
    logger.info("DEPLOYMENT RECOMMENDATION")
    logger.info("="*60)
    
    svm_recall = svm_results['report']['1']['recall']
    rf_recall = rf_results['report']['1']['recall']
    
    if svm_recall > rf_recall:
        winner = "SVM"
        winner_recall = svm_recall
    else:
        winner = "Random Forest"
        winner_recall = rf_recall
    
    recommendation = f"""
RECOMMENDED: {winner}

The {winner} achieves {winner_recall:.1%} recall on failure detection.
In semiconductor manufacturing, catching defects is critical — missing failures costs far more than false alarms.
The {winner} catches {int(winner_recall * 21)} of 21 actual failures, providing stronger quality control.
Deploy with a lower probability threshold (~0.3) to catch more edge cases, and monitor quarterly as sensor patterns drift.
"""
    logger.info(recommendation)

# ============ MAIN ============
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_preprocess("uci_secom.csv")
    
    svm_results = tune_and_train_svm(X_train, y_train, X_test, y_test)
    rf_results = tune_and_train_rf(X_train, y_train, X_test, y_test)
    
    comparison_df = compare_models(svm_results, rf_results, y_test)
    plot_results(svm_results, rf_results, y_test)
    deployment_recommendation(svm_results, rf_results)
    
    logger.info("\n=== ASSIGNMENT 32.1 COMPLETE ===")