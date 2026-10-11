"""
Tests for Machine Learning, Regression, Clustering, and Apriori.
"""

import pytest
import pandas as pd
import numpy as np

from grievanceroute.data.generator import generate_full_dataset
from grievanceroute.data.preprocessor import clean_and_preprocess_complaints, get_train_features_and_targets
from grievanceroute.ml.classifier import train_and_compare_classifiers
from grievanceroute.ml.clustering import run_kmeans_hotspots, run_dbscan_hotspots
from grievanceroute.ml.association_rules import apriori_from_scratch, prepare_discretized_transactions
from grievanceroute.ml.neural_net import run_gradient_descent_simulation


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

    pipelines, metrics = train_and_compare_classifiers(train_texts, train_labels, test_texts, test_labels)
    assert "Logistic Regression" in pipelines
    assert "Decision Tree" in pipelines
    assert metrics["Logistic Regression"]["accuracy"] > 0.80
    assert metrics["Logistic Regression"]["macro_f1"] > 0.80


def test_spatial_clustering(sample_data):
    """Verify K-Means and DBSCAN clustering on projected coordinates."""
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


def test_gradient_descent_simulation():
    """Verify gradient descent simulation optimizes surrogate loss."""
    X = np.random.randn(50, 4)
    y = X @ np.array([1.5, -2.0, 0.5, 1.0]) + np.random.randn(50) * 0.1
    sim = run_gradient_descent_simulation(X, y, epochs=30)
    assert "trajectories" in sim
    adam_traj = sim["trajectories"]["Adam"]
    # Final loss should be lower than initial loss
    assert adam_traj[-1] < adam_traj[0]
