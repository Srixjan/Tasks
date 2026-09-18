import logging
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

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
        