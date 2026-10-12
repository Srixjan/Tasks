import logging
import pandas as pd
import numpy as np
import sklearn
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from numpy.typing import NDArray
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="[+%(relativeCreated)dms] %(levelname)s: %(message)s"
)

logger = logging.getLogger(__name__)
log = logger.info

housing = fetch_california_housing(download_if_missing=True, as_frame=False)
df = pd.DataFrame(data=housing.data, columns=housing.feature_names)

class GradientCheckError(Exception):
    """If propagation math fucks up then,"""
    pass

class DivergenceError(Exception):
    """Learing Rate too high, or Nan values occured"""
    pass


def data_processing_pipeline(X_raw: NDArray[np.float64], y_raw: NDArray[np.float64], seed: int = 42) -> Tuple[NDArray[np.float64], NDArray[np.float64],  NDArray[np.float64], NDArray[np.float64],  NDArray[np.float64], NDArray[np.float64]]:
    # reshape, because dot multiplication needs a 2d array, y contians 1d
    y_reshaped = y_raw.reshape(-1, 1)
    # first split in 70 percent, rest we use for val and test.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_raw,
        y_reshaped, 
        random_state=seed,
        test_size=0.3
    )
    # 2nd split between val and test, so 50 - 50 [100 - 70 = 30/2 = 15 - 15]
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=seed
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    # WE DO NOT fit_transform on x_test, and x_val, just trasnform to prevent leakages.
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # log info of all the data
    log(f"Data Split executed succesfully using random seed{seed}")
    log(f"Training metrics shape X: {X_train.shape}, y: {y_train.shape}")
    log(f"Validation metrics shape X: {X_val.shape}, y: {y_val.shape}")
    log(f"Testing metrics shape X: {X_test.shape}, y: {y_test.shape}")

    return X_train, X_val, X_test, y_train, y_val, y_test

def relu(Z: NDArray[np.float64]):
    return np.maximum(0, Z)

def relu_derivative(Z: NDArray[np.float64]):
    return (Z > 0).astype(np.float64)

def sigmoid(Z: NDArray[np.float64]):
    Z_clipped = np.clip(Z, -1000, 1000)
    return 1.0 / (1.0 + np.exp(-Z_clipped))

def sigmoid_derivative(Z: NDArray[np.float64]):
    A = sigmoid(Z)
    return A * (1.0 - A)

def linear(Z: NDArray[np.float64]):
    return Z

def linear_derivative(Z: NDArray[np.float64]):
    return np.ones_like(Z)

def mse_loss(Y_pred: NDArray[np.float64], Y_true: NDArray[np.float64]) -> Tuple[float, NDArray[np.float64]]:
    N = Y_pred.shape[0]
    errors = Y_pred - Y_true
    
    loss_val = float(np.mean(errors ** 2))
    
    loss_grad = (2.0 / N) * errors
    
    return loss_val, loss_grad

ACTIVATIONS = {
    "relu": relu,
    "sigmoid": sigmoid,
    "linear": linear
}

DERIVATIVES = {
    "relu_derivative": relu_derivative,
    "sigmoid_derivative": sigmoid_derivative,
    "linear_derivative": linear_derivative
}

class DenseLayer:
    def __init__(self, in_features: int, out_features: int, activation_str: str, rng: np.random.Generator):
        self.activation_str = activation_str

        if self.activation_str == "relu":
            std = np.sqrt(2.0/in_features)
        else:
            std = np.sqrt(1.0/in_features)

        self.weight = rng.standard_normal(size=(in_features, out_features)) * std
        self.bias = np.zeros((1, out_features))

    def forward(self, A_prev: NDArray[np.float64]) -> NDArray[np.float64]:
        self.A_prev = A_prev
        self.Z =  A_prev @ self.weight  + self.bias
        return ACTIVATIONS[self.activation_str](self.Z)

    def backward(self, dA: NDArray[np.float64]) -> NDArray[np.float64]:
        dZ = dA * DERIVATIVES[self.activation_str](self.Z)
        self.dW = self.A_prev.T @ dZ
        self.db = np.sum(dZ, axis=0, keepdims=True)
        dA_prev = dZ @ self.weight.T
        return dA_prev

    