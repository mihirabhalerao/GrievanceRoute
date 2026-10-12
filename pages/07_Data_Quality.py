"""
Data Quality and Preprocessing Lab Page for GrievanceRoute.
Demonstrates missing value handling, deduplication, normalization vs standardization,
sampling techniques (SRSWOR, SRSWR, Stratified, Cluster), and equal-width binning.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from grievanceroute.services.ui_theme import apply_custom_theme, render_hero


st.set_page_config(page_title="Data Quality & Preprocessing — GrievanceRoute", page_icon="🧹", layout="wide")
apply_custom_theme()

render_hero(
    title="Data Quality & Preprocessing Lab",
    subtitle="Interactive demonstrations of imputation, normalization vs. standardization, sampling methods, and binning.",
    badge="Unit III • Laboratory Exercises"
)

# Load raw and cleaned datasets
raw_path = Path("data/raw/pune_civic_complaints.csv")
cleaned_path = Path("data/processed/cleaned_complaints.csv")

if not raw_path.exists() or not cleaned_path.exists():
    st.error("Dataset files missing. Run scripts/generate_dataset.py and scripts/preprocess_data.py first.")
    st.stop()

df_raw = pd.read_csv(raw_path)
df_clean = pd.read_csv(cleaned_path)

tab_clean, tab_scale, tab_sample, tab_bin = st.tabs([
    "🧽 Cleaning & Imputation",
    "⚖️ Normalization vs. Standardization",
    "🎲 Sampling Techniques",
    "📊 Equal-Width Binning",
])

with tab_clean:
    st.subheader("Data Cleaning & Defect Resolution")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Raw Dataset Rows", f"{len(df_raw):,}")
    with c2:
        st.metric("Cleaned Dataset Rows", f"{len(df_clean):,}")
    with c3:
        st.metric("Duplicates Deduplicated", f"{len(df_raw) - len(df_clean)}")

    st.markdown("#### Null Value Imputation Comparison")
    raw_nulls = df_raw.isnull().sum().to_dict()
    clean_nulls = df_clean.isnull().sum().to_dict()

    null_comp = []
    for col in df_raw.columns:
        if raw_nulls.get(col, 0) > 0 or clean_nulls.get(col, 0) > 0:
            null_comp.append({
                "Feature": col,
                "Raw Missing Count": raw_nulls.get(col, 0),
                "Cleaned Missing Count": clean_nulls.get(col, 0),
                "Imputation Strategy": "Mode Imputation ('Clear')" if col == "weather" else "Median / Dropped",
            })
    st.dataframe(pd.DataFrame(null_comp), use_container_width=True)

with tab_scale:
    st.subheader("Feature Transformation: Min-Max vs. Standard Normalization")
    st.caption("Demonstrates the mathematical effect of scaling on the `actual_resolution_hours` feature.")

    raw_vals = df_clean[["actual_resolution_hours"]].head(100)
    scaler_minmax = MinMaxScaler()
    scaler_std = StandardScaler()

    minmax_vals = scaler_minmax.fit_transform(raw_vals)
    std_vals = scaler_std.fit_transform(raw_vals)

    scaled_df = pd.DataFrame({
        "Raw Value (hours)": raw_vals["actual_resolution_hours"].values[:15],
        "Min-Max Normalized [0, 1]": minmax_vals.flatten()[:15],
        "Z-Score Standardized (μ=0, σ=1)": std_vals.flatten()[:15],
    })
    st.dataframe(scaled_df, use_container_width=True)

with tab_sample:
    st.subheader("Statistical Sampling Techniques")
    st.caption("Demonstrating SRSWOR, SRSWR, Stratified, and Cluster Sampling from the municipal complaint universe.")

    sample_size = st.slider("Sample Size (N):", min_value=50, max_value=500, value=150)
    method = st.selectbox("Sampling Method:", [
        "Simple Random Sampling Without Replacement (SRSWOR)",
        "Simple Random Sampling With Replacement (SRSWR)",
        "Stratified Sampling (Preserving Pune Zone Ratios)",
        "Cluster Sampling (Selecting Specific Zones as Clusters)",
    ])

    if method == "Simple Random Sampling Without Replacement (SRSWOR)":
        sampled_df = df_clean.sample(n=sample_size, replace=False, random_state=42).reset_index(drop=True)
    elif method == "Simple Random Sampling With Replacement (SRSWR)":
        sampled_df = df_clean.sample(n=sample_size, replace=True, random_state=42).reset_index(drop=True)
    elif method == "Stratified Sampling (Preserving Pune Zone Ratios)":
        sampled_df = df_clean.groupby("zone", group_keys=False).sample(
            frac=sample_size / len(df_clean),
            random_state=42
        ).reset_index(drop=True)
    else:  # Cluster Sampling
        selected_clusters = ["Kothrud", "Shivajinagar"]
        cluster_universe = df_clean[df_clean["zone"].isin(selected_clusters)]
        sampled_df = cluster_universe.sample(n=min(sample_size, len(cluster_universe)), random_state=42).reset_index(drop=True)

    st.write(f"Sampled **{len(sampled_df)}** records:")
    st.dataframe(sampled_df[["complaint_id", "complaint_type", "zone", "urgency_score", "weather"]].head(10), use_container_width=True)

    fig_sample = px.histogram(sampled_df, x="zone", color="complaint_type", title="Sample Zone & Category Distribution")
    st.plotly_chart(fig_sample, use_container_width=True)

with tab_bin:
    st.subheader("Equal-Width Discretization & Binning")
    st.caption("Discretizing continuous hours into fixed width bins (width = 10 hours).")

    df_clean["binned_resolution_hours"] = pd.cut(
        df_clean["actual_resolution_hours"],
        bins=list(range(0, 170, 10)),
        right=False
    )
    bin_counts = df_clean["binned_resolution_hours"].value_counts().sort_index().reset_index()
    bin_counts.columns = ["Resolution Hours Interval (Width 10)", "Complaint Frequency"]
    # Format intervals as clean strings so Plotly JSON serialization succeeds
    labels = [f"{int(i.left)}-{int(i.right)}h" for i in bin_counts["Resolution Hours Interval (Width 10)"]]
    bin_counts["Resolution Hours Interval (Width 10)"] = labels

    fig_bin = px.bar(
        bin_counts,
        x="Resolution Hours Interval (Width 10)",
        y="Complaint Frequency",
        color="Complaint Frequency",
        title="Equal-Width Binning Distribution (Width = 10 Hours)",
    )
    st.plotly_chart(fig_bin, use_container_width=True)
