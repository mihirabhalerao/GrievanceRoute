"""
Urgency and Resolution-Time Regression Models for GrievanceRoute.
Evaluates Random Forest Regressor and Linear Regression against a simple Median Baseline.
"""

from __future__ import annotations
import math
import time
from pathlib import Path
from typing import Dict, Any, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from grievanceroute.data.feature_engineering import build_tabular_preprocessor


def train_and_compare_regressors(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    target_name: str = "urgency_score",
) -> Tuple[Dict[str, Pipeline], Dict[str, Dict[str, Any]]]:
    """
    Train Linear Regression and Random Forest against a simple Median Baseline.
    Evaluates MAE, RMSE, R^2, and residual statistics.
    """
    models = {
        "Median Baseline": DummyRegressor(strategy="median"),
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=40, max_depth=8, random_state=42),
    }

    trained_pipelines: Dict[str, Pipeline] = {}
    comparison_metrics: Dict[str, Dict[str, Any]] = {}

    for name, reg in models.items():
        start_time = time.time()
        pipeline = Pipeline([
            ("preprocessor", build_tabular_preprocessor()),
            ("regressor", reg),
        ])
        pipeline.fit(X_train, y_train)
        train_time = time.time() - start_time

        preds = pipeline.predict(X_test)
        if target_name == "urgency_score":
            clipped_preds = np.clip(preds, 1.0, 5.0)
            mae = mean_absolute_error(y_test, clipped_preds)
            mse = mean_squared_error(y_test, clipped_preds)
            rmse = math.sqrt(mse)
            r2 = r2_score(y_test, clipped_preds)
            residuals = y_test.values - clipped_preds
        else:
            mae = mean_absolute_error(y_test, preds)
            mse = mean_squared_error(y_test, preds)
            rmse = math.sqrt(mse)
            r2 = r2_score(y_test, preds)
            residuals = y_test.values - preds

        trained_pipelines[name] = pipeline
        comparison_metrics[name] = {
            "target": target_name,
            "mae": round(float(mae), 4),
            "mse": round(float(mse), 4),
            "rmse": round(float(rmse), 4),
            "r2": round(float(r2), 4),
            "mean_residual": round(float(np.mean(residuals)), 4),
            "std_residual": round(float(np.std(residuals)), 4),
            "training_time_sec": round(float(train_time), 4),
        }

    return trained_pipelines, comparison_metrics


def save_regressor_pipeline(pipeline: Pipeline, path: str | Path) -> None:
    """Save fitted regressor pipeline to disk."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)


def load_regressor_pipeline(path: str | Path) -> Pipeline:
    """Load fitted regressor pipeline from disk."""
    return joblib.load(path)
