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
import folium
from streamlit_folium import st_folium

from grievanceroute.services.ui_theme import apply_custom_theme, render_hero
from grievanceroute.services.planning_service import PlanningService
from grievanceroute.data.feature_engineering import PUNE_CENTER_LAT, PUNE_CENTER_LON
from grievanceroute.routing.osm_router import OSMRoadGraphService


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

tab_comp, tab_detail, tab_crews, tab_osm = st.tabs([
    "📊 Solver Benchmark Comparison",
    "🗺️ Tour Route Map",
    "👷 Crew Roster",
    "🛣️ Real Road Routing (OSM A*)",
])

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

with tab_osm:
    st.subheader("OpenStreetMap-Backed A* Road Pathfinding")
    st.markdown(
        "Computes point-to-point road-following routes over a real Pune street-network graph "
        "extracted from OpenStreetMap. Evaluates $f(n) = g(n) + h(n)$ using accumulated road distance $g(n)$ "
        "and an admissible straight-line Euclidean lower-bound heuristic $h(n)$ in matching kilometer units."
    )

    # Initialize OSM service with caching
    @st.cache_resource
    def get_osm_service():
        return OSMRoadGraphService()

    osm_svc = get_osm_service()

    # Source status banner
    if osm_svc.is_osm:
        st.success(f"🗺️ **Road Graph Active:** {osm_svc.source_label} ({osm_svc.graph.number_of_nodes()} road intersections, {osm_svc.graph.number_of_edges()} street segments)")
    else:
        st.warning(f"⚠️ **Operating in Fallback Mode:** {osm_svc.source_label}. Real-world OSM cache not found or offline.")

    st.markdown("#### Select Route Origin and Destination")
    c_orig, c_dest = st.columns(2)

    # Predefined known points in Pune
    preset_origins = {
        "Shivajinagar Depot": (18.5314, 73.8446),
        "Deccan Gymkhana": (18.5170, 73.8410),
        "FC Road Chowk": (18.5236, 73.8418),
        "Kothrud Depot": (18.5074, 73.8077),
        "Swargate Depot": (18.5018, 73.8586),
    }

    with c_orig:
        orig_mode = st.radio("Origin Selection Mode:", ["Preset Depot / Junction", "Custom Coordinates"], horizontal=True, key="orig_mode_rad")
        if orig_mode == "Preset Depot / Junction":
            orig_choice = st.selectbox("Select Origin Junction:", list(preset_origins.keys()), index=0)
            orig_lat, orig_lon = preset_origins[orig_choice]
        else:
            col_ol, col_on = st.columns(2)
            with col_ol:
                orig_lat = st.number_input("Origin Latitude:", value=18.5314, format="%.6f")
            with col_on:
                orig_lon = st.number_input("Origin Longitude:", value=73.8446, format="%.6f")
            orig_choice = f"Custom ({orig_lat:.4f}, {orig_lon:.4f})"

    # Destinations: high-priority complaints in dataset or custom
    preset_destinations = {}
    for idx, c in enumerate(candidate_complaints[:10]):
        label = f"{c.get('complaint_id')} — {c.get('complaint_type', 'Grievance').title()} ({c.get('zone', 'Pune')})"
        preset_destinations[label] = (float(c["latitude"]), float(c["longitude"]))

    # Add fallback preset destinations if list empty
    if not preset_destinations:
        preset_destinations = {
            "FC Road Pothole": (18.5236, 73.8418),
            "Deccan Gymkhana Water Leak": (18.5170, 73.8410),
        }

    with c_dest:
        dest_mode = st.radio("Destination Selection Mode:", ["High-Priority Complaint", "Custom Coordinates"], horizontal=True, key="dest_mode_rad")
        if dest_mode == "High-Priority Complaint":
            dest_choice = st.selectbox("Select Target Complaint:", list(preset_destinations.keys()), index=min(1, len(preset_destinations) - 1))
            dest_lat, dest_lon = preset_destinations[dest_choice]
        else:
            col_dl, col_dn = st.columns(2)
            with col_dl:
                dest_lat = st.number_input("Destination Latitude:", value=18.5200, format="%.6f")
            with col_dn:
                dest_lon = st.number_input("Destination Longitude:", value=73.8420, format="%.6f")
            dest_choice = f"Custom ({dest_lat:.4f}, {dest_lon:.4f})"

    route_btn = st.button("🚀 Calculate Road-Following A* Route", type="primary", use_container_width=True)

    if route_btn or "last_osm_route" in st.session_state:
        if route_btn:
            with st.spinner("Executing A* search over road network..."):
                route_res = osm_svc.compute_route(orig_lat, orig_lon, dest_lat, dest_lon)
                st.session_state["last_osm_route"] = route_res
                st.session_state["last_osm_meta"] = {
                    "origin": orig_choice,
                    "dest": dest_choice,
                    "orig_coords": (orig_lat, orig_lon),
                    "dest_coords": (dest_lat, dest_lon),
                }

        route_res = st.session_state.get("last_osm_route")
        meta_info = st.session_state.get("last_osm_meta", {})

        if not route_res or not route_res.get("success"):
            st.error(f"Routing Failed: {route_res.get('error', 'Destination unreachable') if route_res else 'Unknown error'}")
        else:
            # Metrics Row
            m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
            with m_col1:
                st.metric("A* Road Distance", f"{route_res['road_distance_km']} km", "Actual street path")
            with m_col2:
                st.metric("Straight-Line Distance", f"{route_res['straight_line_km']} km", "Euclidean lower-bound")
            with m_col3:
                st.metric("Detour Factor", f"{route_res['detour_ratio']}x", "Road / Straight ratio")
            with m_col4:
                st.metric("Nodes Expanded", f"{route_res['expanded_count']}", f"Total path: {len(route_res['path_nodes'])} nodes")
            with m_col5:
                st.metric("A* Runtime", f"{route_res['runtime_ms']:.2f} ms")

            # Route Map using Folium
            st.markdown("##### 🗺️ Street-Level Road Route Map")
            coords = route_res["path_coords"]

            # Compute map center
            if coords:
                center_lat = sum(c[0] for c in coords) / len(coords)
                center_lon = sum(c[1] for c in coords) / len(coords)
            else:
                center_lat, center_lon = orig_lat, orig_lon

            folium_map = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=14,
                tiles="OpenStreetMap",
            )

            # Draw straight line as dashed comparison
            folium.PolyLine(
                locations=[[orig_lat, orig_lon], [dest_lat, dest_lon]],
                color="#94a3b8",
                weight=2,
                dash_array="6, 6",
                tooltip=f"Straight-Line Euclidean Lower Bound: {route_res['straight_line_km']} km",
            ).add_to(folium_map)

            # Draw road route polyline
            if len(coords) > 1:
                folium.PolyLine(
                    locations=coords,
                    color="#2563eb",
                    weight=5,
                    opacity=0.85,
                    tooltip=f"A* Real Road Route: {route_res['road_distance_km']} km ({len(coords)} nodes)",
                ).add_to(folium_map)

            # Origin Marker
            folium.Marker(
                location=[orig_lat, orig_lon],
                popup=f"<b>Origin:</b> {meta_info.get('origin')}<br>Mapped Node: {route_res.get('start_node')}",
                tooltip="Route Origin",
                icon=folium.Icon(color="green", icon="play", prefix="fa"),
            ).add_to(folium_map)

            # Destination Marker
            folium.Marker(
                location=[dest_lat, dest_lon],
                popup=f"<b>Destination:</b> {meta_info.get('dest')}<br>Mapped Node: {route_res.get('goal_node')}",
                tooltip="Route Destination",
                icon=folium.Icon(color="red", icon="flag", prefix="fa"),
            ).add_to(folium_map)

            st_folium(folium_map, width="100%", height=460)

            # Algorithmic Optimality Verification Accordion
            with st.expander("🔬 Algorithmic Validation: Compare A* vs. Dijkstra's Algorithm", expanded=False):
                st.caption(
                    "Because straight-line Euclidean distance in kilometers never exceeds road distance, "
                    "$h(n)$ is admissible ($h(n) \\le g^*(n)$) and consistent ($h(u) \\le c(u, v) + h(v)$). "
                    "Thus A* is mathematically guaranteed to find the identical shortest path as Dijkstra while expanding fewer or equal nodes."
                )
                if st.button("Run Dijkstra Comparison on Active Route"):
                    comp = osm_svc.verify_optimality_against_dijkstra(route_res["start_node"], route_res["goal_node"])
                    if comp.get("optimal"):
                        st.success(f"✅ **A\* Optimality Verified:** A\* cost = {comp['astar_cost']} km, Dijkstra cost = {comp['dijkstra_cost']} km (Difference = {comp['cost_difference']} km).")
                        c_a, c_b = st.columns(2)
                        with c_a:
                            st.metric("A* Expanded Nodes", comp["astar_expanded"])
                        with c_b:
                            st.metric("Cost Discrepancy", f"{comp['cost_difference']} km", "Optimal within tolerance")
                    else:
                        st.warning(f"Comparison note: {comp}")

            # Turn-by-Turn Itinerary
            with st.expander("📋 Turn-by-Turn Road Node Sequence", expanded=False):
                if "node_details" in route_res and route_res["node_details"]:
                    df_itinerary = pd.DataFrame(route_res["node_details"])
                    st.dataframe(df_itinerary, use_container_width=True, height=260)

