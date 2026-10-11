"""
Evaluation Utilities and Metric Plotting for GrievanceRoute.
Generates confusion matrix heatmaps, ROC curves, and metric comparison summaries.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, List

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless plot generation
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.pipeline import Pipeline


def plot_confusion_matrix(
    cm: List[List[int]] | np.ndarray,
    classes: List[str],
    out_path: str | Path = "artifacts/figures/confusion_matrix.png",
    title: str = "Grievance Classification Confusion Matrix",
) -> Path:
    """Render and save an annotated confusion matrix heatmap."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cm_arr = np.array(cm)

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm_arr,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        cbar=True,
    )
    plt.title(title, fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Predicted Category", fontsize=11)
    plt.ylabel("Actual Category", fontsize=11)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return Path(out_path)


def plot_roc_curve_multiclass(
    pipeline: Pipeline,
    test_texts: pd.Series,
    test_labels: pd.Series,
    out_path: str | Path = "artifacts/figures/roc_curve.png",
) -> Path:
    """Compute and save One-vs-Rest ROC curve and AUC for classification."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    classes = sorted(test_labels.unique())
    y_bin = label_binarize(test_labels, classes=classes)
    y_score = pipeline.predict_proba(test_texts)

    plt.figure(figsize=(8, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i % len(colors)], lw=2, label=f"{cls} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Chance")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=11)
    plt.ylabel("True Positive Rate", fontsize=11)
    plt.title("Receiver Operating Characteristic (One-vs-Rest)", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return Path(out_path)


def plot_model_comparison_bar(
    metrics_dict: Dict[str, Dict[str, Any]],
    out_path: str | Path = "artifacts/figures/model_comparison.png",
) -> Path:
    """Save model performance comparison bar chart."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    models = list(metrics_dict.keys())
    accuracies = [metrics_dict[m]["accuracy"] for m in models]
    f1_scores = [metrics_dict[m]["macro_f1"] for m in models]

    x = np.arange(len(models))
    width = 0.35

    plt.figure(figsize=(9, 5))
    plt.bar(x - width/2, accuracies, width, label="Accuracy", color="#3b82f6")
    plt.bar(x + width/2, f1_scores, width, label="Macro F1", color="#10b981")

    plt.ylabel("Score", fontsize=11)
    plt.title("Classification Model Benchmark Comparison", fontsize=13, fontweight="bold", pad=12)
    plt.xticks(x, models, fontsize=10)
    plt.ylim(0.0, 1.1)
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return Path(out_path)


def save_metrics_to_json(metrics: Dict[str, Any], out_path: str | Path) -> None:
    """Save metrics dictionary to JSON file."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
