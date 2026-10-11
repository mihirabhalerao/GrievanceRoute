"""
Operations Dashboard for GrievanceRoute.
Presents complaint volume KPIs, priority queues, zone breakdowns, and resolution tracking.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import streamlit as st
import pandas as pd
import plotly.express as px

from grievanceroute.services.ui_theme import apply_custom_theme, render_hero
from grievanceroute.database.repository import GrievanceRepository


st.set_page_config(page_title="Operations Dashboard — GrievanceRoute", page_icon="📊", layout="wide")
apply_custom_theme()

render_hero(
    title="Municipal Operations Dashboard",
    subtitle="Real-time incident monitoring, urgent queue triage, and departmental workload distribution across Pune.",
    badge="Unit I & IV • System Overview"
)

repo = GrievanceRepository()

# Load complaints
try:
    df = repo.get_all_complaints(limit=3000)
except Exception as e:
    st.error(f"Failed to query database: {e}")
    df = pd.DataFrame()

if df.empty:
    st.info("No complaint records found in database. Initializing sample data...")
    csv_path = Path("data/processed/cleaned_complaints.csv")
    if csv_path.exists():
        repo.seed_initial_complaints(csv_path, max_rows=500)
        df = repo.get_all_complaints(limit=3000)

if not df.empty:
    # KPI Row
    total = len(df)
    pending = int((df["status"] == "Pending").sum())
    in_prog = int((df["status"] == "In Progress").sum())
    resolved = int((df["status"] == "Resolved").sum())
    urgent = int((df["urgency_score"] >= 4).sum())

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.metric("Total Incidents", f"{total:,}")
    with k2:
        st.metric("Pending Queue", f"{pending:,}", delta=f"{pending/total*100:.1f}%", delta_color="inverse")
    with k3:
        st.metric("In Progress", f"{in_prog:,}")
    with k4:
        st.metric("Resolved", f"{resolved:,}")
    with k5:
        st.metric("Urgent (Score 4-5)", f"{urgent:,}", delta="High Attention", delta_color="inverse")

    st.divider()

    # Visual Analytics Row
    c_left, c_right = st.columns(2)

    with c_left:
        st.subheader("📌 Complaint Category Distribution")
        cat_counts = df["complaint_type"].value_counts().reset_index()
        cat_counts.columns = ["Category", "Count"]
        fig_cat = px.pie(
            cat_counts,
            names="Category",
            values="Count",
            hole=0.45,
            color="Category",
            color_discrete_map={
                "pothole": "#f59e0b",
                "water leak": "#0284c7",
                "streetlight": "#eab308",
                "garbage": "#10b981",
            }
        )
        fig_cat.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=320)
        st.plotly_chart(fig_cat, use_container_width=True)

    with c_right:
        st.subheader("📍 Incidents by Pune Zone")
        zone_counts = df["zone"].value_counts().reset_index()
        zone_counts.columns = ["Zone", "Incidents"]
        fig_zone = px.bar(
            zone_counts,
            x="Zone",
            y="Incidents",
            color="Incidents",
            color_continuous_scale="Viridis"
        )
        fig_zone.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=320)
        st.plotly_chart(fig_zone, use_container_width=True)

    st.divider()

    # High Urgency Active Priority Queue
    st.subheader("🚨 High-Urgency Incident Queue (Score 4 & 5)")
    st.caption("Auto-prioritized queue for immediate dispatch and supervisor verification.")

    urgent_df = df[df["urgency_score"] >= 4].copy()
    if not urgent_df.empty:
        display_cols = ["complaint_id", "complaint_type", "zone", "urgency_score", "status", "crew_required", "complaint_text"]
        st.dataframe(
            urgent_df[display_cols].head(25),
            use_container_width=True,
            column_config={
                "complaint_id": st.column_config.TextColumn("Complaint ID", width="small"),
                "complaint_type": st.column_config.TextColumn("Type", width="small"),
                "urgency_score": st.column_config.NumberColumn("Urgency", width="small"),
                "status": st.column_config.TextColumn("Status", width="small"),
                "complaint_text": st.column_config.TextColumn("Complaint Details", width="large"),
            }
        )

        csv_bytes = urgent_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Export Urgent Queue to CSV", data=csv_bytes, file_name="urgent_complaints_queue.csv", mime="text/csv")
    else:
        st.success("No critical high-urgency incidents currently pending!")

    st.divider()

    # Quick Status Transition Section
    st.subheader("⚡ Quick Status Update")
    with st.expander("Update Complaint Status in Database"):
        col_id, col_stat, col_btn = st.columns([2, 2, 1])
        with col_id:
            complaint_options = df["complaint_id"].tolist()[:50]
            selected_cid = st.selectbox("Select Complaint ID:", complaint_options)
        with col_stat:
            new_status_val = st.selectbox("New Operational Status:", ["Pending", "Assigned", "In Progress", "Resolved"])
        with col_btn:
            st.write("")
            st.write("")
            if st.button("Update Status"):
                ok = repo.update_complaint_status(selected_cid, new_status_val, notes="Updated via Operations Dashboard.")
                if ok:
                    st.success(f"{selected_cid} updated to '{new_status_val}'!")
                    st.rerun()
                else:
                    st.error("Failed to update status.")
