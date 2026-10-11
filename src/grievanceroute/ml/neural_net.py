"""
Neural Network and Optimizer Comparison for GrievanceRoute.
Compares SGD, Adam, and RMSprop optimizers on continuous regression,
records learning curves, loss convergence, and execution runtimes.
"""

from __future__ import annotations
import time
from typing import Dict, Any, List, Tuple

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error


def run_optimizer_benchmark(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    max_iter: int = 150,
    random_state: int = 42,
) -> Dict[str, Dict[str, Any]]:
    """
    Train small MLP Regressors using different optimizers (Adam, SGD, L-BFGS)
    and record training loss convergence curves and validation metrics.
    """
    solvers = ["adam", "sgd", "lbfgs"]
    results: Dict[str, Dict[str, Any]] = {}

    for solver in solvers:
        mlp = MLPRegressor(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            solver=solver,
            max_iter=max_iter,
            random_state=random_state,
            learning_rate_init=0.01 if solver != "lbfgs" else 0.001,
            early_stopping=False,
        )

        start_time = time.time()
        mlp.fit(X_train, y_train)
        duration = time.time() - start_time

        val_preds = mlp.predict(X_val)
        val_mse = mean_squared_error(y_val, val_preds)
        val_mae = mean_absolute_error(y_val, val_preds)

        loss_curve = mlp.loss_curve_ if hasattr(mlp, "loss_curve_") and mlp.loss_curve_ else []

        results[solver.upper()] = {
            "solver": solver,
            "train_duration_sec": round(float(duration), 4),
            "val_mse": round(float(val_mse), 4),
            "val_mae": round(float(val_mae), 4),
            "n_iter": int(mlp.n_iter_),
            "loss_curve": [round(float(l), 4) for l in loss_curve] if loss_curve else [],
        }

    return results


def run_gradient_descent_simulation(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.05,
    epochs: int = 80,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Custom educational optimizer comparison: Standard GD vs Momentum vs RMSprop vs Adam
    on linear regression surrogate loss L(w) = (1/2N) * ||Xw - y||^2.
    Returns epoch-by-epoch loss trajectories.
    """
    rng = np.random.default_rng(random_state)
    n_samples, n_features = X.shape

    # Standardize inputs for stable gradient descent
    X_std = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)
    X_bias = np.column_stack([np.ones(n_samples), X_std])
    n_params = X_bias.shape[1]

    initial_w = rng.normal(0, 0.1, size=n_params)

    optimizers = ["GD", "SGD_Momentum", "RMSprop", "Adam"]
    trajectories: Dict[str, List[float]] = {opt: [] for opt in optimizers}

    for opt in optimizers:
        w = initial_w.copy()
        v = np.zeros(n_params)
        m = np.zeros(n_params)

        for epoch in range(1, epochs + 1):
            y_pred = X_bias @ w
            error = y_pred - y
            grad = (X_bias.T @ error) / n_samples
            loss = float(np.mean(error ** 2)) / 2.0
            trajectories[opt].append(round(loss, 4))

            if opt == "GD":
                w -= lr * grad
            elif opt == "SGD_Momentum":
                beta = 0.9
                v = beta * v + lr * grad
                w -= v
            elif opt == "RMSprop":
                beta = 0.99
                eps = 1e-8
                v = beta * v + (1 - beta) * (grad ** 2)
                w -= (lr / (np.sqrt(v) + eps)) * grad
            elif opt == "Adam":
                beta1, beta2 = 0.9, 0.999
                eps = 1e-8
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * (grad ** 2)
                m_hat = m / (1 - beta1 ** epoch)
                v_hat = v / (1 - beta2 ** epoch)
                w -= (lr / (np.sqrt(v_hat) + eps)) * m_hat

    return {
        "epochs": epochs,
        "learning_rate": lr,
        "trajectories": trajectories,
    }
