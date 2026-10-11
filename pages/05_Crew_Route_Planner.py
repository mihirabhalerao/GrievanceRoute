"""
Crew Route Planner Page for GrievanceRoute.
Multi-Crew Vehicle Routing Problem (VRP) optimization,
comparing Greedy baseline, Hill Climbing, and Simulated Annealing.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from grievanceroute.services.ui_theme import apply_custom_theme, render_hero
from grievanceroute.services.planning_service import PlanningService
from grievanceroute.data.feature_engineering import PUNE_CENTER_LAT, PUNE_CENTER_LON


st.set_page_config(page_title="Crew Route Planner — GrievanceRoute", page_icon="🚚", layout="wide")
apply_custom_theme()

render_hero(
    title="Crew Dispatch & Route Optimization",
    subtitle="Vehicle routing under skill compatibility, job capacity, and 8-hour shift limits across Pune municipal zones.",
    badge="Unit II & VI • VRP Optimization"
)

planning_svc = PlanningService()
crews_df = planning_svc.load_crews()

# Load candidate complaints for scheduling
csv_path = Path("data/processed/cleaned_complaints.csv")
if not csv_path.exists():
    st.error("Processed complaints not found. Run scripts/preprocess_data.py first.")
    st.stop()

df_complaints = pd.read_csv(csv_path)

# Sidebar / top parameters
with st.expander("⚙️ Routing Objective Parameters (Cost Function Weights)", expanded=True):
    col_a, col_b, col_g, col_sample = st.columns(4)
    with col_a:
        alpha_val = st.number_input("α (Travel Distance Weight)", value=1.0, step=0.1)
    with col_b:
        beta_val = st.number_input("β (Missed Urgency Penalty)", value=10.0, step=1.0)
    with col_g:
        gamma_val = st.number_input("γ (Overtime Penalty)", value=5.0, step=0.5)
    with col_sample:
        sample_count = st.slider("Complaints to Schedule:", min_value=20, max_value=60, value=40)

# Filter high-priority pending sample for dispatch
candidate_complaints = df_complaints.head(sample_count).to_dict(orient="records")

tab_comp, tab_detail, tab_crews = st.tabs(["📊 Solver Benchmark Comparison", "🗺️ Tour Route Map", "👷 Crew Roster"])

with tab_comp:
    st.subheader("Benchmark: Greedy vs. Hill Climbing vs. Simulated Annealing")
    st.caption("Solvers evaluated on identical candidate complaints using global cost $J(R) = \\alpha D + \\beta U_{missed} + \\gamma T_{overtime}$.")

    if st.button("🚀 Re-calculate Route Schedules", type="primary"):
        with st.spinner("Executing solvers..."):
            comp_df = planning_svc.compare_all_solvers(
                candidate_complaints, alpha=alpha_val, beta=beta_val, gamma=gamma_val
            )
            st.session_state["route_comp_df"] = comp_df

    if "route_comp_df" in st.session_state:
        comp_df = st.session_state["route_comp_df"]
    else:
        # Load from saved metrics
        saved_route_csv = Path("artifacts/metrics/route_comparison.csv")
        if saved_route_csv.exists():
            comp_df = pd.read_csv(saved_route_csv)
        else:
            comp_df = planning_svc.compare_all_solvers(
                candidate_complaints, alpha=alpha_val, beta=beta_val, gamma=gamma_val
            )
            st.session_state["route_comp_df"] = comp_df

    st.dataframe(comp_df, use_container_width=True)

    # Plot comparisons
    c_p1, c_p2 = st.columns(2)
    with c_p1:
        fig_dist = px.bar(
            comp_df,
            x="Algorithm",
            y="Total Distance (km)",
            color="Algorithm",
            title="Total Travel Distance by Solver (Lower is Better)",
        )
        fig_dist.update_layout(showlegend=False, height=320)
        st.plotly_chart(fig_dist, use_container_width=True)
    with c_p2:
        fig_cost = px.bar(
            comp_df,
            x="Algorithm",
            y="Objective Cost J(R)",
            color="Algorithm",
            title="Global Objective Cost J(R) (Lower is Better)",
        )
        fig_cost.update_layout(showlegend=False, height=320)
        st.plotly_chart(fig_cost, use_container_width=True)

with tab_detail:
    st.subheader("Interactive Crew Tour Map")
    solver_choice = st.selectbox("Select Route Solution to Inspect:", ["Simulated Annealing", "Hill Climbing", "Greedy Baseline"])

    plan_result = planning_svc.generate_plan(
        candidate_complaints, algorithm=solver_choice, alpha=alpha_val, beta=beta_val, gamma=gamma_val
    )

    crew_reports = plan_result.get("crew_reports", [])
    selected_crew_id = st.selectbox("Select Crew to Inspect:", [r["crew_id"] + " — " + r["crew_name"] for r in crew_reports])
    target_cid = selected_crew_id.split(" — ")[0]

    crew_info = next((r for r in crew_reports if r["crew_id"] == target_cid), None)
    crew_meta = crews_df[crews_df["crew_id"] == target_cid].iloc[0]

    if crew_info:
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Route Distance", f"{crew_info['distance_km']} km")
        with m2:
            st.metric("Assigned Complaints", f"{crew_info['jobs_count']}")
        with m3:
            st.metric("Total Shift Hours", f"{crew_info['total_hours']} h", f"Max 8.0h")
        with m4:
            st.metric("Feasibility", "Feasible" if crew_info["is_feasible"] else "Violations")

        if crew_info["violations"]:
            st.warning(f"Constraint Violations: {crew_info['violations']}")

        # Build map for this crew's tour
        stops = plan_result["schedules"].get(target_cid, [])
        lats = [crew_meta["depot_lat"]] + [s["latitude"] for s in stops] + [crew_meta["depot_lat"]]
        lons = [crew_meta["depot_lon"]] + [s["longitude"] for s in stops] + [crew_meta["depot_lon"]]
        names = [f"Depot: {crew_meta['crew_name']}"] + [f"Stop {i+1}: {s.get('complaint_type')} ({s.get('complaint_id')})" for i, s in enumerate(stops)] + ["Return to Depot"]

        fig_map = go.Figure()
        ScatterClass = getattr(go, "Scattermap", getattr(go, "Scattermapbox", None))
        map_key = "map" if hasattr(go, "Scattermap") else "mapbox"

        # Tour path
        fig_map.add_trace(ScatterClass(
            mode="lines+markers",
            lat=lats,
            lon=lons,
            marker=dict(size=10, color="#3b82f6"),
            line=dict(width=3, color="#2563eb"),
            text=names,
            name="Route Tour",
        ))
        # Depot highlight
        fig_map.add_trace(ScatterClass(
            mode="markers",
            lat=[crew_meta["depot_lat"]],
            lon=[crew_meta["depot_lon"]],
            marker=dict(size=14, color="#ef4444"),
            text=[f"Base Depot: {crew_meta['crew_name']}"],
            name="Depot",
        ))

        fig_map.update_layout(
            **{map_key: dict(
                style="carto-positron",
                center=dict(lat=float(crew_meta["depot_lat"]), lon=float(crew_meta["depot_lon"])),
                zoom=12,
            )},
            margin=dict(t=10, b=10, l=10, r=10),
            height=480,
        )
        st.plotly_chart(fig_map, use_container_width=True)

with tab_crews:
    st.subheader("Municipal Repair Crew Roster")
    st.dataframe(crews_df, use_container_width=True)
