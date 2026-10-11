"""
Complaint Classification Models for GrievanceRoute.
Implements Logistic Regression baseline, Decision Tree, KNN, and Multinomial Naive Bayes.
"""

from __future__ import annotations
import time
from pathlib import Path
from typing import Dict, Any, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

from grievanceroute.data.feature_engineering import build_text_vectorizer


def get_classification_models() -> Dict[str, Any]:
    """Return dictionary of classifier estimators for benchmark comparison."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=15, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Multinomial Naive Bayes": MultinomialNB(alpha=1.0),
    }


def train_and_compare_classifiers(
    train_texts: pd.Series,
    train_labels: pd.Series,
    test_texts: pd.Series,
    test_labels: pd.Series,
) -> Tuple[Dict[str, Pipeline], Dict[str, Dict[str, Any]]]:
    """Train all 4 classifiers on TF-IDF features and benchmark their test performance."""
    models = get_classification_models()
    trained_pipelines: Dict[str, Pipeline] = {}
    comparison_metrics: Dict[str, Dict[str, Any]] = {}

    for name, clf in models.items():
        start_time = time.time()
        pipeline = Pipeline([
            ("tfidf", build_text_vectorizer()),
            ("clf", clf)
        ])
        pipeline.fit(train_texts, train_labels)
        train_time = time.time() - start_time

        # Predict on test
        preds = pipeline.predict(test_texts)
        proba = pipeline.predict_proba(test_texts) if hasattr(pipeline, "predict_proba") else None

        acc = accuracy_score(test_labels, preds)
        macro_f1 = f1_score(test_labels, preds, average="macro", zero_division=0)
        macro_prec = precision_score(test_labels, preds, average="macro", zero_division=0)
        macro_rec = recall_score(test_labels, preds, average="macro", zero_division=0)
        cm = confusion_matrix(test_labels, preds, labels=sorted(test_labels.unique())).tolist()

        trained_pipelines[name] = pipeline
        comparison_metrics[name] = {
            "accuracy": round(float(acc), 4),
            "macro_f1": round(float(macro_f1), 4),
            "macro_precision": round(float(macro_prec), 4),
            "macro_recall": round(float(macro_rec), 4),
            "training_time_sec": round(float(train_time), 4),
            "confusion_matrix": cm,
            "classes": sorted(test_labels.unique()),
        }

    return trained_pipelines, comparison_metrics


def save_classifier_pipeline(pipeline: Pipeline, path: str | Path = "artifacts/models/complaint_classifier.pkl") -> None:
    """Save fitted classifier pipeline to disk."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)


def load_classifier_pipeline(path: str | Path = "artifacts/models/complaint_classifier.pkl") -> Pipeline:
    """Load fitted classifier pipeline from disk."""
    return joblib.load(path)
