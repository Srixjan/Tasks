import logging
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from src.risk_target import defined_completion_risk

from src.feature_pipeline import build_feature_pipeline

class ModelNotTrainedError(Exception):
    """raise this if user forgets to trains, and goes directly to predict"""
    pass

class UnknownClassifierError(Exception):
    """raise this if user submits model out of model cabinet"""
    pass

logger = logging.getLogger(__name__)

class CompletionRiskModel:
    def __init__(self, *, model="logistic_regression", random_state=42):
        self.config = {
            "model_name": model,
            "random_state": random_state,
            "hyperparameters": {}
        }

        self.pipeline = None
        self.classifier = None
        self.is_trained = False

        self._initialize_classifier()

    def _initialize_classifier(self):
        name = self.config["model_name"]
        rand_stat = self.config["random_state"]

        model_cabinet = {
            "logistic_regression": LogisticRegression,
            "knn": KNeighborsClassifier,
            "random_forest": RandomForestClassifier
        }

        if name not in model_cabinet:
            raise UnknownClassifierError(f"Your requested model {name}, doesnt exist.")

        model_class = model_cabinet[name]

        if "random_state" in model_class.__init__.__code__.co_varnames:
            self.classifier = model_class(random_state=rand_stat)
        else:
            self.classifier = model_class()

        logger.info(f"Successfully initialized classifier, {name}")

        ## this shit works till now, please dont fuck it up!!

    def train(self, df):
        rand_stat = self.config["random_state"]
        
        target_df = defined_completion_risk(df)
        y = target_df["at_risk"]
        X = target_df.drop(columns=["at_risk"])

        pipeline = build_feature_pipeline()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=rand_stat, stratify=y
        )

        X_train_trans = pipeline.fit_transform(X_train)
        X_test_trans = pipeline.transform(X_test)

        self.pipeline = pipeline # backup to class.

        self.classifier.fit(X_train_trans, y_train)

        train_acc = self.classifier.score(X_train_trans, y_train)
        test_acc = self.classifier.score(X_test_trans ,y_test)

        logging.info(
            f"Training Complete. Model:{self.config['model_name']}."
            f"Train accuracy -> {train_acc:.2f}, Test accuracy -> {test_acc:.2f}"
        )

        self.is_trained = True

    def predict(self, X_fresh):
        if not self.is_trained or self.pipeline is None:
            raise ModelNotTrainedError(f"Model has not been trained yet....")

        X_fresh_trans = self.pipeline.transform(X_fresh)

        predictions = self.classifier.predict(X_fresh_trans)

        return list(predictions)

    def save(self, filepath):
        if not self.is_trained:
            raise ModelNotTrainedError(f"Train and Predict your model first!!!!")

        artifact = {
            "config": self.config,
            "pipeline": self.pipeline,
            "classifier": self.classifier
        }

        joblib.dump(artifact, filepath)
        logger.info(f"Model + pipeline + config saved successfully to {filepath}")

    @classmethod
    def load(cls, filepath):
        artifact = joblib.load(filepath)
        saved_config = artifact["config"]

        instance = cls(
            model=saved_config["model_name"],
            random_state=saved_config["random_state"]
        )

        instance.config = saved_config
        instance.pipeline = artifact["pipeline"]
        instance.classifier = artifact["classifier"]
        instance.is_trained = True

        logger.info(f"Model artifact successfully restored from {filepath}. Ready to predict.")
        return instance