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
    "relu": relu_derivative,
    "sigmoid": sigmoid_derivative,
    "linear": linear_derivative
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

from typing import List, Dict, Any

class MLP:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        layer_sizes: List[int] = config["layer_sizes"]
        activations: List[str] = config["activations"]
        seed: int = config.get("seed", 42)
        
        self.rng = np.random.default_rng(seed)
        self.layers: List[DenseLayer] = []
        
        for i in range(len(layer_sizes) - 1):
            self.layers.append(
                DenseLayer(
                    in_features=layer_sizes[i],
                    out_features=layer_sizes[i+1],
                    activation_str=activations[i],
                    rng=self.rng
                )
            )

    def forward(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        current_activation = X
        for layer in self.layers:
            current_activation = layer.forward(current_activation)
        return current_activation

    def backward(self, dLoss_dPred: NDArray[np.float64]) -> None:
        current_gradient = dLoss_dPred
        for layer in reversed(self.layers):
            current_gradient = layer.backward(current_gradient)

    def update_parameters(self, learning_rate: float) -> None:
        for layer in self.layers:
            layer.weight -= learning_rate * layer.dW
            layer.bias -= learning_rate * layer.db

    def check_gradients(self, X_sample: NDArray[np.float64], y_sample: NDArray[np.float64], h: float = 1e-5, epsilon: float = 1e-8) -> None:
        y_pred = self.forward(X_sample)
        _, loss_grad = mse_loss(y_pred, y_sample)
        self.backward(loss_grad)
        
        total_checked = 0
        max_rel_error = 0.0
        
        for layer in self.layers:
            for param, grad in [(layer.weight, layer.dW), (layer.bias, layer.db)]:
                it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
                while not it.finished:
                    idx = it.multi_index
                    original_val = param[idx]
                    
                    param[idx] = original_val + h
                    loss_plus, _ = mse_loss(self.forward(X_sample), y_sample)
                    
                    param[idx] = original_val - h
                    loss_minus, _ = mse_loss(self.forward(X_sample), y_sample)
                    
                    param[idx] = original_val
                    
                    g_numeric = (loss_plus - loss_minus) / (2.0 * h)
                    g_analytic = grad[idx]
                    
                    rel_error = np.abs(g_analytic - g_numeric) / np.maximum(np.abs(g_analytic) + np.abs(g_numeric), epsilon)
                    
                    if rel_error > max_rel_error:
                        max_rel_error = rel_error
                        
                    total_checked += 1
                    it.iternext()
                    
        log(f"Gradient Check Report: Parameters verified: {total_checked}, Max Relative Error: {max_rel_error:.4e}")
        
        if max_rel_error > 1e-6:
            raise GradientCheckError(f"Calculus verification failed! Max relative error {max_rel_error} > 1e-6")
        log("🎉 GRADIENT PASSED: Network math is officially certified as error-free.")

def run_pipeline() -> None:
    from sklearn.linear_model import LinearRegression
    
    housing_data = fetch_california_housing(download_if_missing=True, as_frame=False)
    seed_value = 42
    
    X_train, X_val, X_test, y_train, y_val, y_test = data_processing_pipeline(
        X_raw=housing_data.data, 
        y_raw=housing_data.target, 
        seed=seed_value
    )
    
    config = {
        "layer_sizes":[8, 32, 16, 1],
        "activations": ["relu", "relu", "linear"],
        "seed": seed_value
    }
    
    mlp = MLP(config)
    
    mlp.check_gradients(X_train[:20], y_train[:20])
    
    epochs = 300
    learning_rate = 1e-5
    
    log(f"Beginning Full Batch Gradient Descent for {epochs} epochs...")
    
    for epoch in range(1, epochs + 1):
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
    "relu": relu_derivative,
    "sigmoid": sigmoid_derivative,
    "linear": linear_derivative
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

from typing import List, Dict, Any

class MLP:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        layer_sizes: List[int] = config["layer_sizes"]
        activations: List[str] = config["activations"]
        seed: int = config.get("seed", 42)
        
        self.rng = np.random.default_rng(seed)
        self.layers: List[DenseLayer] = []
        
        for i in range(len(layer_sizes) - 1):
            self.layers.append(
                DenseLayer(
                    in_features=layer_sizes[i],
                    out_features=layer_sizes[i+1],
                    activation_str=activations[i],
                    rng=self.rng
                )
            )

    def forward(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        current_activation = X
        for layer in self.layers:
            current_activation = layer.forward(current_activation)
        return current_activation

    def backward(self, dLoss_dPred: NDArray[np.float64]) -> None:
        current_gradient = dLoss_dPred
        for layer in reversed(self.layers):
            current_gradient = layer.backward(current_gradient)

    def update_parameters(self, learning_rate: float) -> None:
        for layer in self.layers:
            layer.weight -= learning_rate * layer.dW
            layer.bias -= learning_rate * layer.db

    def check_gradients(self, X_sample: NDArray[np.float64], y_sample: NDArray[np.float64], h: float = 1e-5, epsilon: float = 1e-8) -> None:
        y_pred = self.forward(X_sample)
        _, loss_grad = mse_loss(y_pred, y_sample)
        self.backward(loss_grad)
        
        total_checked = 0
        max_rel_error = 0.0
        
        for layer in self.layers:
            for param, grad in [(layer.weight, layer.dW), (layer.bias, layer.db)]:
                it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
                while not it.finished:
                    idx = it.multi_index
                    original_val = param[idx]
                    
                    param[idx] = original_val + h
                    loss_plus, _ = mse_loss(self.forward(X_sample), y_sample)
                    
                    param[idx] = original_val - h
                    loss_minus, _ = mse_loss(self.forward(X_sample), y_sample)
                    
                    param[idx] = original_val
                    
                    g_numeric = (loss_plus - loss_minus) / (2.0 * h)
                    g_analytic = grad[idx]
                    
                    rel_error = np.abs(g_analytic - g_numeric) / np.maximum(np.abs(g_analytic) + np.abs(g_numeric), epsilon)
                    
                    if rel_error > max_rel_error:
                        max_rel_error = rel_error
                        
                    total_checked += 1
                    it.iternext()
                    
        log(f"Gradient Check Report: Parameters verified: {total_checked}, Max Relative Error: {max_rel_error:.4e}")
        
        if max_rel_error > 1e-6:
            raise GradientCheckError(f"Calculus verification failed! Max relative error {max_rel_error} > 1e-6")
        log("🎉 GRADIENT PASSED: Network math is officially certified as error-free.")

def run_pipeline() -> None:
    from sklearn.linear_model import LinearRegression
    
    housing_data = fetch_california_housing(download_if_missing=True, as_frame=False)
    seed_value = 42
    
    X_train, X_val, X_test, y_train, y_val, y_test = data_processing_pipeline(
        X_raw=housing_data.data, 
        y_raw=housing_data.target, 
        seed=seed_value
    )
    
    config = {
        "layer_sizes":[8, 32, 16, 1],
        "activations": ["relu", "relu", "linear"],
        "seed": seed_value
    }
    
    mlp = MLP(config)
    
    mlp.check_gradients(X_train[:20], y_train[:20])
    
    epochs = 300
    learning_rate = 1e-5
    
    log(f"Beginning Full Batch Gradient Descent for {epochs} epochs...")
    
    for epoch in range(1, epochs + 1):
        y_pred_train = mlp.forward(X_train)
        y_pred_val = mlp.forward(X_val)
        
        if not (np.isfinite(y_pred_train).all() and np.isfinite(y_pred_val).all()):
            raise DivergenceError(f"Loss excccccploded into NaN/Inf limits at epoch {epoch}!")
            
        train_loss, train_grad = mse_loss(y_pred_train, y_train)
        val_loss, _ = mse_loss(y_pred_val, y_val)
        
        mlp.backward(train_grad)
        mlp.update_parameters(learning_rate)
        
        if epoch % 25 == 0 or epoch == 1:
            log(f"Epoch {epoch:03d} -> Training MSE: {train_loss:.5f} | Validation MSE: {val_loss:.5f}")
            
    mlp_test_predictions = mlp.forward(X_test)
    mlp_test_rmse = np.sqrt(np.mean((mlp_test_predictions - y_test) ** 2))
    
    baseline_model = LinearRegression()
    baseline_model.fit(X_train, y_train)
    baseline_test_predictions = baseline_model.predict(X_test)
    baseline_test_rmse = np.sqrt(np.mean((baseline_test_predictions - y_test) ** 2))
    
    print("\n" + "="*60)
    print("🏆 FINAL ASSESSMENT REPORT")
    print("="*60)
    print(f"Custom MLP Test RMSE                      : {mlp_test_rmse:.5f}")
    print(f"Sklearn Linear Regression Baseline Test RMSE: {baseline_test_rmse:.5f}")
    print("="*60)

if __name__ == "__main__":
    run_pipeline()