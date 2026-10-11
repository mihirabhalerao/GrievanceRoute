"""
Model Training & Benchmark Evaluation Script for GrievanceRoute.
Trains classifiers, urgency regressors, resolution-time regressors,
evaluates metrics, plots figures, and persists pipelines.
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from grievanceroute.data.preprocessor import get_train_features_and_targets
from grievanceroute.data.feature_engineering import split_data_stratified
from grievanceroute.ml.classifier import (
    train_and_compare_classifiers,
    save_classifier_pipeline,
)
from grievanceroute.ml.regression import (
    train_and_compare_regressors,
    save_regressor_pipeline,
)
from grievanceroute.ml.evaluation import (
    plot_confusion_matrix,
    plot_roc_curve_multiclass,
    plot_model_comparison_bar,
    save_metrics_to_json,
)


def main():
    csv_path = Path("data/processed/cleaned_complaints.csv")
    if not csv_path.exists():
        print("Processed data not found. Run scripts/preprocess_data.py first.")
        sys.exit(1)

    print("Loading cleaned dataset...")
    df = pd.read_csv(csv_path)

    # 1. Stratified split
    train_df, test_df = split_data_stratified(df, target_col="complaint_type", test_size=0.2, random_state=42)
    print(f"Dataset split: Train={len(train_df)} rows, Test={len(test_df)} rows.")

    X_train, y_train_cls, y_train_urg, y_train_res = get_train_features_and_targets(train_df)
    X_test, y_test_cls, y_test_urg, y_test_res = get_train_features_and_targets(test_df)

    # 2. Train Classifiers
    print("Training and benchmarking complaint classifiers...")
    clf_pipelines, clf_metrics = train_and_compare_classifiers(
        X_train["complaint_text"], y_train_cls,
        X_test["complaint_text"], y_test_cls,
    )
    for model_name, m in clf_metrics.items():
        print(f"  [{model_name}] Accuracy: {m['accuracy']:.4f}, Macro-F1: {m['macro_f1']:.4f}, Time: {m['training_time_sec']:.2f}s")

    # Save primary classifier (Logistic Regression)
    primary_clf = clf_pipelines["Logistic Regression"]
    save_classifier_pipeline(primary_clf, "artifacts/models/complaint_classifier.pkl")
    print("Saved primary classifier pipeline to artifacts/models/complaint_classifier.pkl")

    # 3. Train Urgency Regressors
    print("Training urgency regression models...")
    urg_pipelines, urg_metrics = train_and_compare_regressors(
        X_train, y_train_urg, X_test, y_test_urg, target_name="urgency_score"
    )
    for model_name, m in urg_metrics.items():
        print(f"  [{model_name}] MAE: {m['mae']:.4f}, RMSE: {m['rmse']:.4f}, R2: {m['r2']:.4f}")
    save_regressor_pipeline(urg_pipelines["Random Forest Regressor"], "artifacts/models/urgency_regressor.pkl")

    # 4. Train Resolution Time Regressors
    print("Training resolution time regression models...")
    res_pipelines, res_metrics = train_and_compare_regressors(
        X_train, y_train_res, X_test, y_test_res, target_name="actual_resolution_hours"
    )
    for model_name, m in res_metrics.items():
        print(f"  [{model_name}] MAE: {m['mae']:.4f}, RMSE: {m['rmse']:.4f}, R2: {m['r2']:.4f}")
    save_regressor_pipeline(res_pipelines["Random Forest Regressor"], "artifacts/models/resolution_regressor.pkl")

    # 5. Save all comparison metrics
    all_metrics = {
        "classification": clf_metrics,
        "urgency_regression": urg_metrics,
        "resolution_regression": res_metrics,
    }
    save_metrics_to_json(all_metrics, "artifacts/metrics/model_comparison.json")
    print("Saved metrics to artifacts/metrics/model_comparison.json")

    # 6. Generate Figures
    print("Generating report figures...")
    lr_cm = clf_metrics["Logistic Regression"]["confusion_matrix"]
    classes = clf_metrics["Logistic Regression"]["classes"]
    plot_confusion_matrix(lr_cm, classes, "artifacts/figures/confusion_matrix.png")
    plot_roc_curve_multiclass(primary_clf, X_test["complaint_text"], y_test_cls, "artifacts/figures/roc_curve.png")
    plot_model_comparison_bar(clf_metrics, "artifacts/figures/model_comparison.png")
    print("Figures successfully generated in artifacts/figures/")


if __name__ == "__main__":
    main()
