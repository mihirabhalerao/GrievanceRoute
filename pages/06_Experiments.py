"""
Experiment Tracking & Artifacts Page for GrievanceRoute.
Logs MLflow runs, hyperparameter choices, benchmark summaries, and exported metrics.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import streamlit as st
import pandas as pd

from grievanceroute.services.ui_theme import apply_custom_theme, render_hero


st.set_page_config(page_title="Experiments — GrievanceRoute", page_icon="🧪", layout="wide")
apply_custom_theme()

render_hero(
    title="Experiment Tracking & Artifacts",
    subtitle="Consolidated metrics tracking, MLflow run histories, artifact registries, and downloadable evaluation data.",
    badge="Unit IV & VI • MLOps Tracking"
)

tab_runs, tab_rules, tab_routes, tab_artifacts = st.tabs([
    "📋 Experiment Runs Summary",
    "🛒 Association Rules (Apriori)",
    "🚚 VRP Optimization Results",
    "📦 Model Artifacts Registry",
])

with tab_runs:
    st.subheader("Consolidated AIML Experiment Runs")
    exp_summary_path = Path("artifacts/metrics/experiment_summary.csv")
    if exp_summary_path.exists():
        df_exp = pd.read_csv(exp_summary_path)
        st.dataframe(df_exp, use_container_width=True)
        st.download_button(
            "📥 Download Experiment Summary (CSV)",
            data=df_exp.to_csv(index=False).encode("utf-8"),
            file_name="experiment_summary.csv",
            mime="text/csv",
        )
    else:
        st.info("Run scripts/run_experiments.py to populate experiment summary metrics.")

    st.markdown("""
    **Tracked MLflow Parameters & Metrics:**
    - `seed`: 42
    - `classification_model`: LogisticRegression (C=1.0, sublinear_tf=True)
    - `urgency_model`: RandomForestRegressor (n_estimators=50, max_depth=10)
    - `clustering`: KMeans (K=2..8), DBSCAN (eps=0.8..1.8)
    - `vrp_optimizer`: SimulatedAnnealing (cooling_rate=0.92, initial_temp=80.0)
    """)

with tab_rules:
    st.subheader("Discovered Association Rules (Apriori Mining)")
    rules_path = Path("artifacts/metrics/association_rules.csv")
    if rules_path.exists():
        df_rules = pd.read_csv(rules_path)
        st.markdown(f"Total rules extracted: **{len(df_rules)}**")
        st.dataframe(df_rules.head(50), use_container_width=True)
        st.download_button(
            "📥 Download Association Rules (CSV)",
            data=df_rules.to_csv(index=False).encode("utf-8"),
            file_name="association_rules.csv",
            mime="text/csv",
        )
    else:
        st.info("No association rules found. Run scripts/run_experiments.py first.")

with tab_routes:
    st.subheader("Route Solver Comparison Results")
    route_path = Path("artifacts/metrics/route_comparison.csv")
    if route_path.exists():
        df_routes = pd.read_csv(route_path)
        st.dataframe(df_routes, use_container_width=True)
        st.download_button(
            "📥 Download Route Comparison (CSV)",
            data=df_routes.to_csv(index=False).encode("utf-8"),
            file_name="route_comparison.csv",
            mime="text/csv",
        )
    else:
        st.info("No route comparison records found.")

with tab_artifacts:
    st.subheader("Persisted Pipeline Artifacts Registry")
    models_dir = Path("artifacts/models")
    if models_dir.exists():
        artifacts_info = []
        for f in models_dir.glob("*.pkl"):
            size_kb = f.stat().st_size / 1024
            artifacts_info.append({
                "Artifact Name": f.name,
                "Size (KB)": f"{size_kb:.1f} KB",
                "Path": str(f),
                "Format": "Joblib Serialized Scikit-Learn Pipeline",
            })
        st.dataframe(pd.DataFrame(artifacts_info), use_container_width=True)
    else:
        st.info("No model artifacts directory found.")
