"""
Complaint Management Page for GrievanceRoute.
Handles validated citizen submissions, live AI triage prediction,
SQLite persistence, and full lifecycle status tracking.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium

from grievanceroute.services.ui_theme import apply_custom_theme, render_hero
from grievanceroute.services.complaint_service import ComplaintService
from grievanceroute.database.repository import GrievanceRepository
from grievanceroute.data.generator import PUNE_ZONES
from grievanceroute.components.voice_input import voice_input_widget
from grievanceroute.services.voice_service import (
    SUPPORTED_VOICE_LANGUAGES,
    DEFAULT_VOICE_LANGUAGE,
    format_voice_submission_metadata,
)


st.set_page_config(page_title="Complaint Management — GrievanceRoute", page_icon="📝", layout="wide")
apply_custom_theme()

render_hero(
    title="Civic Complaint Intake & Management",
    subtitle="Submit grievances, trigger automatic NLP triage & regression models, and manage operational lifecycle records.",
    badge="Unit I & III • Form & Persistence"
)

repo = GrievanceRepository()
complaint_svc = ComplaintService(repo=repo)

tab_submit, tab_explore = st.tabs(["📝 File New Complaint", "🔍 Search & Audit Trail"])

with tab_submit:
    st.subheader("Citizen Complaint Submission Form")
    st.markdown("Submit civic grievances via **multilingual voice dictation** or **manual text entry**. Pin the grievance on the interactive map to automatically capture exact coordinates.")

    # -------------------------------------------------------------
    # P0: Multilingual Voice Dictation Integration
    # -------------------------------------------------------------
    with st.expander("🎙️ Multilingual Voice-to-Text Input (English / Hindi / Marathi)", expanded=True):
        col_v1, col_v2 = st.columns([1, 2])
        with col_v1:
            lang_options = list(SUPPORTED_VOICE_LANGUAGES.keys())
            selected_lang_code = st.selectbox(
                "Speech Recognition Language",
                options=lang_options,
                format_func=lambda code: f"{SUPPORTED_VOICE_LANGUAGES[code]['label']}",
                index=0,
                help="Select your spoken language. The browser Web Speech API converts your voice to text."
            )
        with col_v2:
            st.caption(f"**Dialect Support:** {SUPPORTED_VOICE_LANGUAGES[selected_lang_code]['description']}")
            st.caption("Click **Start speaking**, dictate your complaint, and click **Stop recording**. The text will transfer to the description box below.")

        col_mode, col_clear = st.columns([3, 1])
        with col_mode:
            transfer_mode = st.radio(
                "When voice transcription arrives:",
                ["Replace description", "Append to existing description"],
                horizontal=True,
                key="voice_transfer_mode_radio"
            )
        with col_clear:
            if st.button("🗑️ Clear Description", use_container_width=True):
                st.session_state["complaint_form_text"] = ""
                st.session_state["complaint_desc_input"] = ""
                st.rerun()

        # Render custom Web Speech API component
        voice_res = voice_input_widget(language=selected_lang_code, key="voice_rec_widget")

        if voice_res and isinstance(voice_res, dict):
            v_status = voice_res.get("status")
            v_text = voice_res.get("transcript", "").strip()
            v_ts = voice_res.get("timestamp")

            if v_status == "unsupported":
                st.info("ℹ️ Native Web Speech API is not supported in this browser. Please use Chrome/Edge or type your grievance manually below.")
            elif v_status == "error":
                st.warning(f"Voice Recognition Notice: {voice_res.get('message', 'Audio recognition interrupted.')}")
            elif v_text and v_ts != st.session_state.get("_last_transcribed_ts"):
                st.session_state["_last_transcribed_ts"] = v_ts
                existing = st.session_state.get("complaint_desc_input", st.session_state.get("complaint_form_text", ""))

                if transfer_mode == "Append to existing description" and existing.strip():
                    combined = f"{existing.strip()} {v_text}".strip()
                else:
                    combined = v_text

                st.session_state["complaint_form_text"] = combined
                st.session_state["complaint_desc_input"] = combined
                st.session_state["_voice_meta"] = format_voice_submission_metadata(
                    selected_lang_code, v_text, combined
                )
                st.success("✅ Voice transcription captured! Review or edit the text in the description box below.")

    # Initialize text state
    if "complaint_form_text" not in st.session_state:
        st.session_state["complaint_form_text"] = ""

    # -------------------------------------------------------------
    # P1: Interactive Map-Based Complaint Location Selection
    # -------------------------------------------------------------
    st.markdown("### 📍 Complaint Location")
    st.caption("Select a service zone, then **click directly on the interactive map** to pin the exact street location of the civic grievance.")

    col_z_top, col_z_info = st.columns([1, 2])
    with col_z_top:
        zone_choice = st.selectbox("Pune Service Zone *", list(PUNE_ZONES.keys()), key="zone_selector_key")
    with col_z_info:
        z_meta = PUNE_ZONES[zone_choice]
        st.caption(f"**Zone Center:** Lat `{z_meta['lat']:.4f}`, Lon `{z_meta['lon']:.4f}` • Flood Risk: `{'High' if z_meta['low_lying'] else 'Standard'}`")

    # Coordinate state management
    if "selected_lat" not in st.session_state or "selected_lon" not in st.session_state:
        st.session_state["selected_lat"] = z_meta["lat"]
        st.session_state["selected_lon"] = z_meta["lon"]
        st.session_state["_active_zone"] = zone_choice
        st.session_state["location_pinned"] = False

    # If zone changes and user hasn't explicitly clicked a pin, sync to zone center
    if st.session_state.get("_active_zone") != zone_choice:
        st.session_state["_active_zone"] = zone_choice
        if not st.session_state.get("location_pinned"):
            st.session_state["selected_lat"] = z_meta["lat"]
            st.session_state["selected_lon"] = z_meta["lon"]

    # Build Folium map centered on active coordinates
    map_lat = st.session_state["selected_lat"]
    map_lon = st.session_state["selected_lon"]

    complaint_map = folium.Map(
        location=[map_lat, map_lon],
        zoom_start=14,
        tiles="OpenStreetMap",
    )

    # Add interactive draggable / visible location marker
    folium.Marker(
        location=[map_lat, map_lon],
        popup=f"Selected Complaint Location<br>Lat: {map_lat:.6f}, Lon: {map_lon:.6f}",
        tooltip="Selected Complaint Location (Click anywhere on map to reposition)",
        icon=folium.Icon(color="red", icon="exclamation-circle", prefix="fa"),
    ).add_to(complaint_map)

    # Render Folium map component
    map_output = st_folium(
        complaint_map,
        width="100%",
        height=320,
        key="complaint_location_folium_map",
        returned_objects=["last_clicked"],
    )

    # Intercept click on map
    if map_output and map_output.get("last_clicked"):
        clicked_pt = map_output["last_clicked"]
        c_lat = round(float(clicked_pt["lat"]), 6)
        c_lon = round(float(clicked_pt["lng"]), 6)

        # If clicked point differs from stored coordinates, update state
        if abs(c_lat - st.session_state["selected_lat"]) > 1e-5 or abs(c_lon - st.session_state["selected_lon"]) > 1e-5:
            st.session_state["selected_lat"] = c_lat
            st.session_state["selected_lon"] = c_lon
            st.session_state["location_pinned"] = True
            st.rerun()

    # Location feedback and manual coordinate adjustment fallback
    col_stat, col_man_lat, col_man_lon, col_reset = st.columns([2, 1, 1, 1])
    with col_stat:
        if st.session_state.get("location_pinned"):
            st.success(f"📍 **Pin Placed via Map:** `{st.session_state['selected_lat']:.6f}, {st.session_state['selected_lon']:.6f}`")
        else:
            st.info(f"📍 **Zone Default Location:** `{st.session_state['selected_lat']:.6f}, {st.session_state['selected_lon']:.6f}` (Click map to pin exact street)")

    with col_man_lat:
        manual_lat = st.number_input(
            "Latitude (Manual):",
            value=st.session_state["selected_lat"],
            format="%.6f",
            key="input_lat_manual",
            help="Fine-tune latitude or enter manually if preferred."
        )
        if abs(manual_lat - st.session_state["selected_lat"]) > 1e-5:
            st.session_state["selected_lat"] = manual_lat
            st.session_state["location_pinned"] = True

    with col_man_lon:
        manual_lon = st.number_input(
            "Longitude (Manual):",
            value=st.session_state["selected_lon"],
            format="%.6f",
            key="input_lon_manual",
            help="Fine-tune longitude or enter manually if preferred."
        )
        if abs(manual_lon - st.session_state["selected_lon"]) > 1e-5:
            st.session_state["selected_lon"] = manual_lon
            st.session_state["location_pinned"] = True

    with col_reset:
        st.write("")
        st.write("")
        if st.button("🔄 Reset Pin", help="Reset location pin back to zone center"):
            st.session_state["selected_lat"] = z_meta["lat"]
            st.session_state["selected_lon"] = z_meta["lon"]
            st.session_state["location_pinned"] = False
            st.rerun()

    # -------------------------------------------------------------
    # Complaint Submission Form
    # -------------------------------------------------------------
    st.markdown("### 📝 Complaint Details & Context")

    with st.form("complaint_form", clear_on_submit=False):
        c_text = st.text_area(
            "Complaint Description *",
            value=st.session_state.get("complaint_form_text", ""),
            placeholder="Describe the civic issue in detail (e.g. Deep pothole near Balgandharva chowk causing bike accidents...)",
            height=120,
            help="Review and edit the voice transcription or type manually.",
            key="complaint_desc_input",
        )

        col_w, col_ch1, col_ch2, col_ch3 = st.columns(4)
        with col_w:
            weather_choice = st.selectbox("Weather Condition", ["Clear", "Rainy", "Cloudy", "Stormy"])
        with col_ch1:
            is_monsoon = st.checkbox("Monsoon Season", value=False)
        with col_ch2:
            is_low_lying = st.checkbox("Low-Lying / Flood Risk", value=z_meta["low_lying"])
        with col_ch3:
            is_repeat = st.checkbox("Repeat Complaint", value=False)

        submit_btn = st.form_submit_button("📨 Submit & Triage Complaint", type="primary", use_container_width=True)

    if submit_btn:
        active_lat = float(st.session_state["selected_lat"])
        active_lon = float(st.session_state["selected_lon"])

        success, message, record = complaint_svc.submit_complaint(
            complaint_text=c_text,
            latitude=active_lat,
            longitude=active_lon,
            zone=zone_choice,
            weather=weather_choice,
            monsoon=is_monsoon,
            low_lying_area=is_low_lying,
            repeat_complaint=is_repeat,
        )

        if not success:
            st.error(f"Validation Error: {message}")
        else:
            st.success(f"Success! {message}")

            # Show Triage Decision Card
            st.markdown("### 🤖 Automatic AI Triage Decision")
            if st.session_state.get("_voice_meta"):
                vm = st.session_state["_voice_meta"]
                st.info(f"🎙️ **Input Source:** Dictated via Voice ({vm.get('language_name')}) • Reviewed and validated before intake.")
            if record.get("safety_override"):
                st.error(f"🚨 **Safety Rule Triggered:** {record['safety_warning']}")

            col_res1, col_res2, col_res3, col_res4 = st.columns(4)
            with col_res1:
                st.metric("Predicted Category", record["complaint_type"].title())
            with col_res2:
                conf_pct = record.get("confidence", 1.0) * 100
                st.metric("Model Confidence", f"{conf_pct:.1f}%")
            with col_res3:
                urg = record["urgency_score"]
                st.metric("Urgency Level", f"{urg} / 5", "Priority Score")
            with col_res4:
                st.metric("Assigned Crew", record["crew_required"])

            st.caption(f"Estimated Resolution Time: **{record.get('actual_resolution_hours', 24.0):.1f} hours**. Record stored in SQLite `complaints` table.")

with tab_explore:
    st.subheader("Historical Complaints Database Explorer")
    all_df = repo.get_all_complaints(limit=1000)

    if all_df.empty:
        st.info("No records in database yet.")
    else:
        # Search & Filter controls
        f1, f2, f3, f4 = st.columns(4)
        with f1:
            search_query = st.text_input("Filter by text keyword:", placeholder="e.g. pothole, wire, pipe...")
        with f2:
            status_filter = st.multiselect("Status:", ["All"] + list(all_df["status"].unique()), default=["All"])
        with f3:
            cat_filter = st.multiselect("Category:", ["All"] + list(all_df["complaint_type"].unique()), default=["All"])
        with f4:
            zone_filter = st.multiselect("Zone:", ["All"] + list(all_df["zone"].unique()), default=["All"])

        filtered_df = all_df.copy()
        if search_query:
            filtered_df = filtered_df[filtered_df["complaint_text"].str.contains(search_query, case=False, na=False)]
        if "All" not in status_filter and status_filter:
            filtered_df = filtered_df[filtered_df["status"].isin(status_filter)]
        if "All" not in cat_filter and cat_filter:
            filtered_df = filtered_df[filtered_df["complaint_type"].isin(cat_filter)]
        if "All" not in zone_filter and zone_filter:
            filtered_df = filtered_df[filtered_df["zone"].isin(zone_filter)]

        st.markdown(f"Showing **{len(filtered_df)}** of **{len(all_df)}** complaints:")
        st.dataframe(
            filtered_df[["complaint_id", "complaint_text", "complaint_type", "zone", "urgency_score", "status", "crew_required", "timestamp"]],
            use_container_width=True,
            height=380
        )

        st.divider()
        st.subheader("📜 Complaint Lifecycle Audit Trail")
        selected_audit_id = st.selectbox("Select Complaint ID to view status history:", all_df["complaint_id"].tolist()[:100])
        with repo.get_connection() as conn:
            hist_df = pd.read_sql_query("SELECT * FROM status_history WHERE complaint_id = ? ORDER BY changed_at DESC", conn, params=(selected_audit_id,))

        if not hist_df.empty:
            st.dataframe(hist_df[["id", "complaint_id", "old_status", "new_status", "changed_at", "notes"]], use_container_width=True)
        else:
            st.info("No status change log for this complaint.")
