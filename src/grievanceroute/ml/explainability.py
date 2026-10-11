"""
Explainability and Feature Importance for GrievanceRoute.
Extracts top positive n-grams per complaint category from Logistic Regression
and feature importances from tree-based models.
"""

from __future__ import annotations
from typing import Dict, List, Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def get_top_tfidf_features_per_class(
    pipeline: Pipeline,
    top_n: int = 8,
) -> Dict[str, List[Dict[str, Any]]]:
    """Extract top positive and negative vocabulary features for each category from Logistic Regression."""
    if "tfidf" not in pipeline.named_steps or "clf" not in pipeline.named_steps:
        return {}

    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["clf"]

    if not hasattr(classifier, "coef_"):
        return {}

    feature_names = np.array(vectorizer.get_feature_names_out())
    classes = classifier.classes_
    class_explanations: Dict[str, List[Dict[str, Any]]] = {}

    for i, cls_name in enumerate(classes):
        coefs = classifier.coef_[i]
        top_indices = np.argsort(coefs)[::-1][:top_n]
        tokens = [
            {"feature": feature_names[idx], "weight": round(float(coefs[idx]), 4)}
            for idx in top_indices
        ]
        class_explanations[str(cls_name)] = tokens

    return class_explanations


def get_tree_feature_importances(
    pipeline: Pipeline,
    feature_names: List[str],
    top_n: int = 10,
) -> List[Dict[str, Any]]:
    """Extract top feature importances from Random Forest Regressor."""
    if "regressor" not in pipeline.named_steps:
        return []
    reg = pipeline.named_steps["regressor"]
    if not hasattr(reg, "feature_importances_"):
        return []

    importances = reg.feature_importances_
    # In case feature_names length matches importances
    min_len = min(len(feature_names), len(importances))
    indices = np.argsort(importances[:min_len])[::-1][:top_n]

    results = []
    for idx in indices:
        results.append({
            "feature": feature_names[idx],
            "importance": round(float(importances[idx]), 4)
        })
    return results
