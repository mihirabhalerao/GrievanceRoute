"""
GrievanceRoute — Main Application Portal.
Intelligent Civic Complaint Triage and Crew Routing System.
"""

import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import streamlit as st
import pandas as pd

from grievanceroute.services.ui_theme import apply_custom_theme, render_hero
from grievanceroute.services.prediction_service import PredictionService
from grievanceroute.database.repository import GrievanceRepository


st.set_page_config(
    page_title="GrievanceRoute — Civic Triage & Routing",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_custom_theme()

render_hero(
    title="GrievanceRoute — Intelligent Civic Operations",
    subtitle="AI-driven complaint classification, multi-target regression, spatial hotspots, and multi-crew route optimization.",
    badge="Pune Municipal Operations • AIML Mini-Project"
)

# Sidebar info
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/traffic-jam.png", width=64)
    st.markdown("### **GrievanceRoute v2.0**")
    st.caption("Pune Municipal Corporation (Simulated Operations)")
    st.divider()
    st.markdown("**Navigation:**")
    st.markdown("- **01** Operations Dashboard")
    st.markdown("- **02** Complaint Management")
    st.markdown("- **03** Geographic Hotspots")
    st.markdown("- **04** ML Workbench")
    st.markdown("- **05** Crew Route Planner")
    st.markdown("- **06** Experiments")
    st.markdown("- **07** Data Quality & Insights")
    st.divider()
    st.info("Local-first architecture. All models and algorithms run offline on standard CPU.")

# KPI Status Bar
repo = GrievanceRepository()
pred_service = PredictionService()

total_complaints = 8000
try:
    df_db = repo.get_all_complaints(limit=5)
    db_status = "Online (SQLite)"
except Exception:
    db_status = "Standby"

model_status = "Loaded (Ready)" if pred_service.classifier is not None else "Models Pending"

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Trained Records", "8,000", "Synthetic Pune")
with c2:
    st.metric("ML Models", "Classifiers + Regressors", "TF-IDF, LR & RF")
with c3:
    st.metric("Active Crews", "8 Teams", "4 Specializations")
with c4:
    st.metric("System State", model_status, db_status)

st.divider()

# Interactive Instant Triage Card
st.subheader("⚡ Quick Civic Complaint Triage")
st.markdown("Test the trained NLP and regression models on free-text citizen grievances with automated safety overrides.")

SAMPLE_COMPLAINTS = [
    "Dangerous live electrical wire hanging over the road near FC Road, sparking violently!",
    "Large deep pothole on the main bus lane near Karve Road causing frequent two wheeler falls.",
    "Main drinking water pipeline burst in Shivajinagar colony, water flooding entire street for two days.",
    "Huge pile of uncollected garbage rotting and stinking near municipal school gate in Hadapsar.",
]

col_in, col_out = st.columns([1.2, 1])

with col_in:
    sample_choice = st.selectbox("Or choose a pre-configured sample:", ["Custom Text..."] + SAMPLE_COMPLAINTS)
    default_text = "" if sample_choice == "Custom Text..." else sample_choice
    user_input = st.text_area(
        "Enter Complaint Text (English / Hinglish):",
        value=default_text,
        height=140,
        placeholder="e.g. Sadak par bahut bada gaddha hai, vehicles are getting damaged near Balgandharva..."
    )
    run_btn = st.button("🚀 Analyze Complaint", type="primary", use_container_width=True)

with col_out:
    if run_btn:
        if len(user_input.strip()) < 8:
            st.warning("Please enter a complaint description (at least 8 characters).")
        else:
            with st.spinner("Classifying and evaluating urgency..."):
                result = pred_service.predict_complaint(user_input)

            # Display Triage Results
            st.success("Analysis Complete!")
            
            if result.get("safety_override"):
                st.error(f"⚠️ {result['safety_warning']}")

            cat = result["predicted_category"].title()
            conf = result["confidence"] * 100
            urgency = result["urgency_score"]
            res_hrs = result["estimated_resolution_hours"]
            crew = result["crew_required"]

            st.markdown(f"**Predicted Category:** `{cat}`")
            st.progress(result["confidence"], text=f"Model Confidence: {conf:.1f}%")
            
            c_u1, c_u2 = st.columns(2)
            with c_u1:
                st.markdown(f"**Urgency Score:** `{urgency} / 5`")
                if urgency >= 4:
                    st.markdown("<span class='badge-urgent'>HIGH PRIORITY</span>", unsafe_allow_html=True)
                else:
                    st.markdown("<span class='badge-pending'>STANDARD QUEUE</span>", unsafe_allow_html=True)
            with c_u2:
                st.markdown(f"**Est. Resolution:** `{res_hrs:.1f} hours`")
                st.markdown(f"**Assigned Crew:** `{crew}`")
    else:
        st.info("Enter or select a complaint on the left and click **Analyze Complaint** to test real-time classification, urgency regression, and safety overrides.")

st.divider()

# System Modules Grid
st.subheader("🛠️ System Modules & Features")
m1, m2, m3 = st.columns(3)

with m1:
    st.markdown("""
    #### 📊 Operations & Analytics
    - **Operations Dashboard**: Real-time KPI summaries, urgency breakdown, and department queues.
    - **Geographic Hotspots**: Interactive Pune map with K-Means and DBSCAN density clustering.
    - **Data Quality**: Cleaning, deduplication, equal-width binning, and Apriori association rules.
    """)

with m2:
    st.markdown("""
    #### 🧠 Machine Learning
    - **ML Workbench**: Benchmarking Logistic Regression against Majority Baseline, Naive Bayes, and Decision Tree.
    - **Regression Pipelines**: Urgency and resolution-time forecasting compared against Median Baselines.
    - **Explainability & Errors**: Vocabulary weights, predictive tokens, and misclassification error inspection.
    """)

with m3:
    st.markdown("""
    #### 🚚 Routing & Optimization
    - **A* Road Pathfinding**: Optimal shortest paths over the Pune road graph.
    - **Crew Route Planner**: Multi-crew VRP with Greedy baseline, 2-Opt Hill Climbing, and Simulated Annealing.
    - **Shift Constraints**: Crew skill matching, overtime penalties, and 8-hour shift limits.
    """)
