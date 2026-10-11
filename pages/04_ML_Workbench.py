"""
ML Workbench for GrievanceRoute.
Presents classification and regression benchmark comparisons,
confusion matrices, ROC curves, and NLP feature explainability.
"""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import streamlit as st
import pandas as pd

from grievanceroute.services.ui_theme import apply_custom_theme, render_hero
from grievanceroute.ml.classifier import load_classifier_pipeline
from grievanceroute.ml.explainability import get_top_tfidf_features_per_class


st.set_page_config(page_title="ML Workbench — GrievanceRoute", page_icon="🧠", layout="wide")
apply_custom_theme()

render_hero(
    title="Machine Learning Workbench",
    subtitle="Supervised classification and regression benchmarks, evaluation curves, and feature token explainability.",
    badge="Unit III & IV • Supervised Learning"
)

# Load saved metrics
metrics_path = Path("artifacts/metrics/model_comparison.json")
if not metrics_path.exists():
    st.error("Model metrics not found. Run scripts/train_models.py first.")
    st.stop()

with open(metrics_path, "r", encoding="utf-8") as f:
    metrics = json.load(f)

tab_clf, tab_reg, tab_eval, tab_exp = st.tabs([
    "📊 Classification Models",
    "📈 Regression Models",
    "📉 Confusion Matrix & ROC",
    "🔍 Feature Explainability",
])

with tab_clf:
    st.subheader("Complaint Classification Benchmark (Untouched Test Set: 1,600 records)")
    clf_data = metrics.get("classification", {})
    
    rows = []
    for model_name, m in clf_data.items():
        rows.append({
            "Model": model_name,
            "Accuracy": f"{m['accuracy'] * 100:.2f}%",
            "Macro F1": f"{m['macro_f1']:.4f}",
            "Macro Precision": f"{m['macro_precision']:.4f}",
            "Macro Recall": f"{m['macro_recall']:.4f}",
            "Train Time (s)": f"{m['training_time_sec']:.2f}s",
        })
    df_clf_table = pd.DataFrame(rows)
    st.dataframe(df_clf_table, use_container_width=True)

    st.markdown("""
    **Architectural Notes:**
    - Primary pipeline: **TF-IDF (1-2 word n-grams, sublinear TF) + Multinomial Logistic Regression**.
    - Baseline models compared: **Decision Tree (max depth 15)**, **K-Nearest Neighbors (k=5)**, and **Multinomial Naive Bayes**.
    - Pure diagonal classification on synthetic templates demonstrates high pattern capture; real-world noisy text generalization caveats are documented in the project plan.
    """)

with tab_reg:
    st.subheader("Regression Models: Urgency & Resolution Time")
    
    c_urg, c_res = st.columns(2)
    with c_urg:
        st.markdown("#### Urgency Score Regression (1–5)")
        urg_data = metrics.get("urgency_regression", {})
        u_rows = []
        for m_name, m in urg_data.items():
            u_rows.append({
                "Model": m_name,
                "MAE": m["mae"],
                "RMSE": m["rmse"],
                "R² Score": m["r2"],
            })
        st.dataframe(pd.DataFrame(u_rows), use_container_width=True)

    with c_res:
        st.markdown("#### Actual Resolution Hours Regression")
        res_data = metrics.get("resolution_regression", {})
        r_rows = []
        for m_name, m in res_data.items():
            r_rows.append({
                "Model": m_name,
                "MAE (hrs)": m["mae"],
                "RMSE (hrs)": m["rmse"],
                "R² Score": m["r2"],
            })
        st.dataframe(pd.DataFrame(r_rows), use_container_width=True)

    st.info("ℹ️ **Leakage Prevention:** Regressors are trained strictly on intake context (weather, low-lying status, recurrence, submission hour, historical averages). Future completion timestamps are strictly excluded.")

with tab_eval:
    st.subheader("Evaluation Figures and Curves")
    cm_path = Path("artifacts/figures/confusion_matrix.png")
    roc_path = Path("artifacts/figures/roc_curve.png")
    bench_path = Path("artifacts/figures/model_comparison.png")

    c1, c2 = st.columns(2)
    with c1:
        if cm_path.exists():
            st.image(str(cm_path), caption="Annotated Confusion Matrix Heatmap (1,600 Test Complaints)")
    with c2:
        if roc_path.exists():
            st.image(str(roc_path), caption="One-vs-Rest ROC Curve & AUC per Category")

    if bench_path.exists():
        st.image(str(bench_path), caption="Comparative Classifier Performance Bar Chart")

with tab_exp:
    st.subheader("NLP Feature Explainability (Top Discriminative Tokens)")
    st.caption("Extracted from the fitted Logistic Regression weight vectors $w_k$ across the TF-IDF feature space.")

    model_path = Path("artifacts/models/complaint_classifier.pkl")
    if model_path.exists():
        clf_pipeline = load_classifier_pipeline(model_path)
        explanations = get_top_tfidf_features_per_class(clf_pipeline, top_n=8)

        cols = st.columns(len(explanations))
        for col, (cls_name, tokens) in zip(cols, explanations.items()):
            with col:
                st.markdown(f"**{cls_name.upper()}**")
                t_df = pd.DataFrame(tokens)
                st.dataframe(t_df, use_container_width=True)
    else:
        st.warning("Classifier model artifact not found.")
