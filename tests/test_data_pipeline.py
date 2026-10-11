"""
Tests for Data Pipeline, Generator, Validator, and Preprocessor.
"""

import pytest
import pandas as pd
import numpy as np

from grievanceroute.data.generator import generate_full_dataset, generate_crews, generate_road_network
from grievanceroute.data.validator import validate_complaint_dataframe, validate_road_network
from grievanceroute.data.preprocessor import clean_and_preprocess_complaints, get_train_features_and_targets
from grievanceroute.data.feature_engineering import project_coordinates_to_km


def test_generator_reproducibility():
    """Verify fixed-seed generation produces deterministic dataset."""
    df1, _, _ = generate_full_dataset(target_count=100, seed=42)
    df2, _, _ = generate_full_dataset(target_count=100, seed=42)
    assert len(df1) == len(df2)
    assert (df1["complaint_id"] == df2["complaint_id"]).all()
    assert (df1["complaint_type"] == df2["complaint_type"]).all()


def test_schema_and_value_ranges():
    """Verify validator checks categories, urgency, and null handling."""
    df, crews, road = generate_full_dataset(target_count=200, seed=42)
    report = validate_complaint_dataframe(df)
    assert report["is_valid"] is True
    assert report["invalid_urgency_count"] == 0
    assert report["duplicate_id_count"] > 0  # controlled duplicates present in raw


def test_preprocessing_and_deduplication():
    """Verify clean_and_preprocess_complaints eliminates duplicates and imputes nulls."""
    df_raw, _, _ = generate_full_dataset(target_count=200, seed=42)
    df_clean, summary = clean_and_preprocess_complaints(df_raw)
    assert summary["duplicates_removed"] > 0
    assert df_clean["weather"].isnull().sum() == 0
    assert len(df_clean) == 200


def test_target_leakage_prevention():
    """Verify training feature matrix does not contain targets or future columns."""
    df_raw, _, _ = generate_full_dataset(target_count=50, seed=42)
    X, y_cls, y_urg, y_res = get_train_features_and_targets(df_raw)
    forbidden_features = ["complaint_type", "urgency_score", "actual_resolution_hours", "status"]
    for col in forbidden_features:
        assert col not in X.columns


def test_coordinate_projection():
    """Verify planar projection transforms lat/lon to kilometers correctly."""
    lats = np.array([18.5204, 18.6204])
    lons = np.array([73.8567, 73.9567])
    coords_km = project_coordinates_to_km(lats, lons)
    assert coords_km.shape == (2, 2)
    # Center should be at (0, 0)
    assert abs(coords_km[0, 0]) < 1e-4
    assert abs(coords_km[0, 1]) < 1e-4
    # Offset point should have positive distance in km
    assert coords_km[1, 1] > 5.0
