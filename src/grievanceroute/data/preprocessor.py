"""
Data Preprocessing Pipeline for GrievanceRoute.
Cleans raw complaints, deduplicates records, normalizes casing/whitespace,
imputes missing attributes, clips/quarantines coordinate anomalies, and enforces leak-free feature separation.
"""

from __future__ import annotations
from pathlib import Path
from typing import Tuple, Dict, Any

import pandas as pd
import numpy as np


PUNE_LAT_BOUNDS = (18.40, 18.65)
PUNE_LON_BOUNDS = (73.70, 74.00)


def clean_and_preprocess_complaints(
    df_raw: pd.DataFrame,
    save_path: str | Path | None = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Clean the raw complaints DataFrame and produce an analysis-ready dataset."""
    df = df_raw.copy()
    initial_rows = len(df)

    # 1. Deduplication
    df = df.drop_duplicates(subset=["complaint_id"]).copy()
    rows_after_dedup = len(df)
    duplicates_removed = initial_rows - rows_after_dedup

    # 2. Text cleaning
    if "complaint_text" in df.columns:
        df["complaint_text"] = df["complaint_text"].astype(str).str.strip()
        df = df[df["complaint_text"] != ""].copy()

    # 3. Categorical normalization
    if "zone" in df.columns:
        df["zone"] = df["zone"].astype(str).str.strip().str.title()
        # Fallback to nearest/central zone if unknown
        valid_zones = ["Kothrud", "Shivajinagar", "Hadapsar", "Baner", "Swargate", "Viman Nagar"]
        df["zone"] = df["zone"].apply(lambda z: z if z in valid_zones else "Shivajinagar")

    # 4. Impute missing weather
    null_weather_count = 0
    if "weather" in df.columns:
        null_weather_count = int(df["weather"].isnull().sum())
        mode_weather = df["weather"].dropna().mode()[0] if not df["weather"].dropna().empty else "Clear"
        df["weather"] = df["weather"].fillna(mode_weather)

    # 5. Coordinate anomaly clipping to valid Pune municipal bounds
    coords_repaired = 0
    if "latitude" in df.columns and "longitude" in df.columns:
        oob_mask = (
            (df["latitude"] < PUNE_LAT_BOUNDS[0]) | (df["latitude"] > PUNE_LAT_BOUNDS[1]) |
            (df["longitude"] < PUNE_LON_BOUNDS[0]) | (df["longitude"] > PUNE_LON_BOUNDS[1])
        )
        coords_repaired = int(oob_mask.sum())
        df["latitude"] = df["latitude"].clip(lower=PUNE_LAT_BOUNDS[0], upper=PUNE_LAT_BOUNDS[1])
        df["longitude"] = df["longitude"].clip(lower=PUNE_LON_BOUNDS[0], upper=PUNE_LON_BOUNDS[1])

    # 6. Target range validation
    if "urgency_score" in df.columns:
        df["urgency_score"] = df["urgency_score"].fillna(3).astype(int).clip(lower=1, upper=5)

    if "actual_resolution_hours" in df.columns:
        df["actual_resolution_hours"] = df["actual_resolution_hours"].astype(float).clip(lower=1.0, upper=168.0)

    # 7. Timestamp validation
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["hour"] = df["timestamp"].dt.hour.fillna(12).astype(int)
        df["day_of_week"] = df["timestamp"].dt.dayofweek.fillna(0).astype(int)

    cleaned_rows = len(df)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)

    summary = {
        "initial_rows": initial_rows,
        "duplicates_removed": duplicates_removed,
        "weather_nulls_imputed": null_weather_count,
        "coordinates_repaired": coords_repaired,
        "final_cleaned_rows": cleaned_rows,
        "class_distribution": df["complaint_type"].value_counts().to_dict() if "complaint_type" in df.columns else {},
    }

    return df, summary


def get_train_features_and_targets(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Extract strictly leak-free submission-time features and separate targets.
    Guarantees that targets and future operational details are excluded.
    """
    feature_cols = [
        "complaint_id",
        "complaint_text",
        "latitude",
        "longitude",
        "zone",
        "hour",
        "day_of_week",
        "weather",
        "monsoon",
        "low_lying_area",
        "repeat_complaint",
        "historical_resolution_hours",
    ]
    available_cols = [c for c in feature_cols if c in df.columns]
    X = df[available_cols].copy()

    y_class = df["complaint_type"] if "complaint_type" in df.columns else None
    y_urgency = df["urgency_score"] if "urgency_score" in df.columns else None
    y_resolution = df["actual_resolution_hours"] if "actual_resolution_hours" in df.columns else None

    return X, y_class, y_urgency, y_resolution
