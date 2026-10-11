"""
Complaint Classification Models for GrievanceRoute.
Implements primary Multinomial Logistic Regression alongside Naive Bayes and Decision Tree comparisons,
evaluated against a Majority Class Dummy baseline.
"""

from __future__ import annotations
import time
from pathlib import Path
from typing import Dict, Any, Tuple, List

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)

from grievanceroute.data.feature_engineering import build_text_vectorizer


def get_classification_models() -> Dict[str, Any]:
    """Return dictionary of classifier estimators (excluding redundant KNN)."""
    return {
        "Majority Baseline": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression (Primary)": LogisticRegression(max_iter=1000, random_state=42),
        "Multinomial Naive Bayes": MultinomialNB(alpha=1.0),
        "Decision Tree": DecisionTreeClassifier(max_depth=15, random_state=42),
    }


def train_and_compare_classifiers(
    train_texts: pd.Series,
    train_labels: pd.Series,
    test_texts: pd.Series,
    test_labels: pd.Series,
) -> Tuple[Dict[str, Pipeline], Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Train benchmark classifiers on TF-IDF text features and compute comprehensive test metrics,
    including macro and weighted scores, per-class metrics, latency, and sample misclassifications.
    """
    models = get_classification_models()
    trained_pipelines: Dict[str, Pipeline] = {}
    comparison_metrics: Dict[str, Dict[str, Any]] = {}
    sample_errors: List[Dict[str, Any]] = []

    classes = sorted(test_labels.unique())

    for name, clf in models.items():
        start_fit = time.time()
        pipeline = Pipeline([
            ("tfidf", build_text_vectorizer()),
            ("clf", clf),
        ])
        pipeline.fit(train_texts, train_labels)
        train_time = time.time() - start_fit

        # Evaluate inference
        start_pred = time.time()
        preds = pipeline.predict(test_texts)
        inf_time = time.time() - start_pred

        acc = accuracy_score(test_labels, preds)
        macro_f1 = f1_score(test_labels, preds, average="macro", zero_division=0)
        weighted_f1 = f1_score(test_labels, preds, average="weighted", zero_division=0)
        macro_prec = precision_score(test_labels, preds, average="macro", zero_division=0)
        macro_rec = recall_score(test_labels, preds, average="macro", zero_division=0)
        cm = confusion_matrix(test_labels, preds, labels=classes).tolist()

        # Per-class metrics
        report = classification_report(test_labels, preds, labels=classes, output_dict=True, zero_division=0)
        per_class = {
            cls: {
                "precision": round(report[cls]["precision"], 4),
                "recall": round(report[cls]["recall"], 4),
                "f1": round(report[cls]["f1-score"], 4),
                "support": int(report[cls]["support"]),
            }
            for cls in classes if cls in report
        }

        trained_pipelines[name] = pipeline
        comparison_metrics[name] = {
            "accuracy": round(float(acc), 4),
            "macro_f1": round(float(macro_f1), 4),
            "weighted_f1": round(float(weighted_f1), 4),
            "macro_precision": round(float(macro_prec), 4),
            "macro_recall": round(float(macro_rec), 4),
            "training_time_sec": round(float(train_time), 4),
            "inference_time_sec": round(float(inf_time), 4),
            "confusion_matrix": cm,
            "classes": classes,
            "per_class": per_class,
        }

        # Collect misclassified samples for Primary model
        if name == "Logistic Regression (Primary)":
            test_indices = test_texts.index.tolist()
            for idx, text, actual, pred in zip(test_indices, test_texts, test_labels, preds):
                if actual != pred and len(sample_errors) < 8:
                    sample_errors.append({
                        "complaint_text": text,
                        "actual_category": actual,
                        "predicted_category": pred,
                    })

    return trained_pipelines, comparison_metrics, sample_errors


def save_classifier_pipeline(pipeline: Pipeline, path: str | Path = "artifacts/models/complaint_classifier.pkl") -> None:
    """Save fitted classifier pipeline to disk."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)


def load_classifier_pipeline(path: str | Path = "artifacts/models/complaint_classifier.pkl") -> Pipeline:
    """Load fitted classifier pipeline from disk."""
    return joblib.load(path)
