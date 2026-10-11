"""
Geographic Hotspots Page for GrievanceRoute.
Spatial incident analysis, K-Means clustering on projected coordinates,
and DBSCAN density hotspot/noise detection across Pune.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from grievanceroute.services.ui_theme import apply_custom_theme, render_hero
from grievanceroute.database.repository import GrievanceRepository
from grievanceroute.ml.clustering import run_kmeans_hotspots, run_dbscan_hotspots
from grievanceroute.data.feature_engineering import PUNE_CENTER_LAT, PUNE_CENTER_LON


st.set_page_config(page_title="Geographic Hotspots — GrievanceRoute", page_icon="🗺️", layout="wide")
apply_custom_theme()

render_hero(
    title="Geographic Complaint Hotspots",
    subtitle="Unsupervised spatial clustering using projected metric coordinates (km) and density-based anomaly detection.",
    badge="Unit V • Spatial Analytics"
)

# Load data
csv_path = Path("data/processed/cleaned_complaints.csv")
if not csv_path.exists():
    st.error("Processed complaints dataset not found. Run scripts/preprocess_data.py first.")
    st.stop()

df = pd.read_csv(csv_path)

# Filters Row
f_col1, f_col2, f_col3 = st.columns(3)
with f_col1:
    cat_filter = st.multiselect("Filter Category:", ["All"] + list(df["complaint_type"].unique()), default=["All"])
with f_col2:
    zone_filter = st.multiselect("Filter Zone:", ["All"] + list(df["zone"].unique()), default=["All"])
with f_col3:
    urgency_min = st.slider("Minimum Urgency Score:", 1, 5, 1)

df_view = df.copy()
if "All" not in cat_filter and cat_filter:
    df_view = df_view[df_view["complaint_type"].isin(cat_filter)]
if "All" not in zone_filter and zone_filter:
    df_view = df_view[df_view["zone"].isin(zone_filter)]
df_view = df_view[df_view["urgency_score"] >= urgency_min]

st.markdown(f"**Analyzing {len(df_view):,} filtered incidents across Pune.**")

tab_kmeans, tab_dbscan, tab_raw_map = st.tabs(["🎯 K-Means Spatial Hotspots", "🔬 DBSCAN Density Clusters", "📍 Interactive Incident Map"])

with tab_kmeans:
    st.subheader("K-Means Hotspot Partitioning")
    st.caption("Clusters complaints using local planar projected coordinates (km) rather than raw latitude/longitude degrees.")

    c_k, c_metric = st.columns([1, 2])
    with c_k:
        k_val = st.slider("Select Number of Clusters (K):", min_value=2, max_value=8, value=5)
        run_k_btn = st.button("Run K-Means", type="primary")

    # Run K-Means
    labels, summary = run_kmeans_hotspots(df_view, n_clusters=k_val)
    df_view["kmeans_cluster"] = [f"Cluster {l+1}" for l in labels]

    with c_metric:
        m1, m2 = st.columns(2)
        with m1:
            st.metric("Silhouette Score", f"{summary['silhouette_score']:.4f}", "Cluster Quality")
        with m2:
            st.metric("Inertia", f"{summary['inertia']:,.1f}", "Compactness")

    # Interactive Map
    fig_k = px.scatter_mapbox(
        df_view.sample(min(1200, len(df_view)), random_state=42),
        lat="latitude",
        lon="longitude",
        color="kmeans_cluster",
        hover_name="complaint_type",
        hover_data=["zone", "urgency_score", "complaint_text"],
        zoom=11,
        center=dict(lat=PUNE_CENTER_LAT, lon=PUNE_CENTER_LON),
        mapbox_style="carto-positron",
        title=f"K-Means Spatial Hotspots (K={k_val})",
        height=520,
    )
    st.plotly_chart(fig_k, use_container_width=True)

    # Centroids Summary
    st.markdown("#### Hotspot Centroids & Workload Distribution")
    centers_df = pd.DataFrame(summary["cluster_centers"])
    st.dataframe(centers_df, use_container_width=True)

with tab_dbscan:
    st.subheader("DBSCAN Density-Based Hotspot Detection")
    st.caption("Identifies dense spatial clusters and flags isolated complaints as noise points (outliers).")

    col_e, col_m = st.columns(2)
    with col_e:
        eps_input = st.slider("Epsilon Neighborhood (km):", min_value=0.4, max_value=2.5, value=1.0, step=0.1)
    with col_m:
        min_samp_input = st.slider("Min Samples in Core:", min_value=10, max_value=50, value=25)

    db_labels, db_summary = run_dbscan_hotspots(df_view, eps_km=eps_input, min_samples=min_samp_input)
    df_view["dbscan_cluster"] = [f"Cluster {l}" if l != -1 else "Noise (Outlier)" for l in db_labels]

    col_stat1, col_stat2, col_stat3 = st.columns(3)
    with col_stat1:
        st.metric("Dense Clusters Found", db_summary["n_clusters"])
    with col_stat2:
        st.metric("Noise Points", db_summary["noise_count"])
    with col_stat3:
        st.metric("Noise Fraction", f"{db_summary['noise_fraction']*100:.2f}%")

    # Map for DBSCAN
    fig_db = px.scatter_mapbox(
        df_view.sample(min(1200, len(df_view)), random_state=42),
        lat="latitude",
        lon="longitude",
        color="dbscan_cluster",
        color_discrete_map={"Noise (Outlier)": "#94a3b8"},
        hover_name="complaint_type",
        hover_data=["zone", "urgency_score"],
        zoom=11,
        center=dict(lat=PUNE_CENTER_LAT, lon=PUNE_CENTER_LON),
        mapbox_style="carto-positron",
        title=f"DBSCAN Density Hotspots (eps={eps_input} km, min_samples={min_samp_input})",
        height=520,
    )
    st.plotly_chart(fig_db, use_container_width=True)

with tab_raw_map:
    st.subheader("Raw Incident Distribution by Complaint Category")
    fig_raw = px.scatter_mapbox(
        df_view.sample(min(1200, len(df_view)), random_state=42),
        lat="latitude",
        lon="longitude",
        color="complaint_type",
        size="urgency_score",
        hover_name="complaint_id",
        hover_data=["zone", "urgency_score", "weather"],
        zoom=11,
        center=dict(lat=PUNE_CENTER_LAT, lon=PUNE_CENTER_LON),
        mapbox_style="carto-positron",
        title="Civic Complaints by Type & Urgency Bubble Size",
        height=520,
    )
    st.plotly_chart(fig_raw, use_container_width=True)
