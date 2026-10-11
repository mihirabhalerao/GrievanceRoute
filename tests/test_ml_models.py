"""
Tests for Machine Learning, Regression, Spatial Clustering, and Apriori.
"""

import pytest
import pandas as pd
import numpy as np

from grievanceroute.data.generator import generate_full_dataset
from grievanceroute.data.preprocessor import clean_and_preprocess_complaints, get_train_features_and_targets
from grievanceroute.ml.classifier import train_and_compare_classifiers
from grievanceroute.ml.regression import train_and_compare_regressors
from grievanceroute.ml.clustering import run_kmeans_hotspots, run_dbscan_hotspots
from grievanceroute.ml.association_rules import apriori_from_scratch, prepare_discretized_transactions


@pytest.fixture
def sample_data():
    df_raw, _, _ = generate_full_dataset(target_count=300, seed=42)
    df_clean, _ = clean_and_preprocess_complaints(df_raw)
    return df_clean


def test_classifier_benchmark(sample_data):
    """Verify text classification pipeline trains and evaluates without errors."""
    train_texts = sample_data["complaint_text"].iloc[:200]
    train_labels = sample_data["complaint_type"].iloc[:200]
    test_texts = sample_data["complaint_text"].iloc[200:]
    test_labels = sample_data["complaint_type"].iloc[200:]

    pipelines, metrics, errors = train_and_compare_classifiers(train_texts, train_labels, test_texts, test_labels)
    assert "Logistic Regression (Primary)" in pipelines
    assert "Majority Baseline" in pipelines
    assert "Decision Tree" in pipelines
    assert metrics["Logistic Regression (Primary)"]["accuracy"] > 0.80
    assert metrics["Logistic Regression (Primary)"]["macro_f1"] > 0.80


def test_regressor_benchmark(sample_data):
    """Verify urgency and resolution regression train with Median baseline and report all required metrics."""
    X, y_cls, y_urg, y_res = get_train_features_and_targets(sample_data)
    X_tr, y_tr = X.iloc[:200], y_urg.iloc[:200]
    X_te, y_te = X.iloc[200:], y_urg.iloc[200:]

    pipelines, metrics = train_and_compare_regressors(X_tr, y_tr, X_te, y_te, target_name="urgency_score")
    assert "Median Baseline" in metrics
    assert "Random Forest Regressor" in metrics
    for model_name in ["Median Baseline", "Random Forest Regressor"]:
        m = metrics[model_name]
        assert "mae" in m and m["mae"] > 0
        assert "rmse" in m and m["rmse"] > 0
        assert "r2" in m
        assert "mean_residual" in m
        assert "std_residual" in m
        assert np.isfinite(m["mae"]) and np.isfinite(m["rmse"])

    # Test resolution time target as well
    _, res_metrics = train_and_compare_regressors(X_tr, y_res.iloc[:200], X_te, y_res.iloc[200:], target_name="resolution_hours")
    assert "Median Baseline" in res_metrics
    assert res_metrics["Random Forest Regressor"]["mae"] > 0


def test_spatial_clustering(sample_data):
    """Verify K-Means and DBSCAN clustering on projected metric coordinates."""
    labels_k, summary_k = run_kmeans_hotspots(sample_data, n_clusters=4)
    assert len(labels_k) == len(sample_data)
    assert summary_k["silhouette_score"] > 0.2

    labels_db, summary_db = run_dbscan_hotspots(sample_data, eps_km=1.5, min_samples=10)
    assert len(labels_db) == len(sample_data)
    assert summary_db["n_clusters"] >= 1


def test_apriori_from_scratch(sample_data):
    """Verify Apriori finds valid frequent itemsets and association rules."""
    transactions = prepare_discretized_transactions(sample_data)
    itemsets, rules = apriori_from_scratch(transactions, min_support=0.05, min_confidence=0.2)
    assert len(itemsets) > 0
    if rules:
        first_rule = rules[0]
        assert "antecedents" in first_rule
        assert "consequents" in first_rule
        assert first_rule["confidence"] >= 0.2
        assert first_rule["lift"] >= 0.0
