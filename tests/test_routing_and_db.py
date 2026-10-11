"""
Tests for Routing Constraints, VRP Solvers, Database Persistence, and Services.
"""

import tempfile
from pathlib import Path
import pytest
import pandas as pd

from grievanceroute.routing.crew_constraints import check_skill_compatibility, evaluate_crew_shift
from grievanceroute.routing.vrp_solver import solve_greedy_vrp, calculate_route_distance
from grievanceroute.database.repository import GrievanceRepository
from grievanceroute.services.prediction_service import PredictionService
from grievanceroute.services.complaint_service import ComplaintService


def test_crew_skill_and_shift_constraints():
    """Verify skill compatibility check and shift overtime calculation."""
    assert check_skill_compatibility("Road Maintenance", "pothole") is True
    assert check_skill_compatibility("Road Maintenance", "water leak") is False

    crew = {"crew_id": "CRW-01", "skill": "Road Maintenance", "capacity": 5, "speed_kmh": 25.0}
    complaints = [{"complaint_id": f"P-{i}", "complaint_type": "pothole"} for i in range(4)]
    eval_res = evaluate_crew_shift(crew, complaints, total_travel_distance_km=20.0, max_shift_hours=8.0)
    # Travel: 20/25 = 0.8h. Service: 4 * 0.75 = 3.0h. Total: 3.8h <= 8.0h.
    assert eval_res["is_feasible"] is True
    assert eval_res["overtime_hours"] == 0.0


def test_safety_escalation_in_prediction_service():
    """Verify live wire / electrical hazard triggers category override and maximum urgency."""
    pred_svc = PredictionService()
    # Complaint with electrical hazard keywords
    res = pred_svc.predict_complaint("There is a live electrical wire sparking violently on the street!")
    assert res["predicted_category"] == "streetlight"
    assert res["safety_override"] is True
    assert res["urgency_score"] == 5
    assert res["crew_required"] == "Electrical Crew"


def test_database_crud_and_status_tracking():
    """Verify SQLite CRUD, parameterized insert, and status audit log in a temp database."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        db_path = Path(tmpdir) / "test_grievance.db"
        repo = GrievanceRepository(db_path)
        complaint_data = {
            "complaint_id": "PUN-TEST-001",
            "complaint_text": "Pothole on Karve road opposite bus stop.",
            "complaint_type": "pothole",
            "latitude": 18.5074,
            "longitude": 73.8077,
            "zone": "Kothrud",
            "timestamp": "2026-10-11 12:00:00",
            "hour": 12,
            "day_of_week": 6,
            "weather": "Clear",
            "monsoon": False,
            "low_lying_area": False,
            "repeat_complaint": False,
            "historical_resolution_hours": 24.0,
            "urgency_score": 3,
            "actual_resolution_hours": 24.0,
            "status": "Pending",
            "crew_required": "Road Maintenance",
        }
        ok = repo.insert_complaint(complaint_data)
        assert ok is True

        # Fetch
        record = repo.get_complaint_by_id("PUN-TEST-001")
        assert record is not None
        assert record["complaint_type"] == "pothole"

        # Update status
        updated = repo.update_complaint_status("PUN-TEST-001", "In Progress", notes="Crew dispatched.")
        assert updated is True

        # Check status history
        conn = repo.get_connection()
        try:
            hist = pd.read_sql_query("SELECT * FROM status_history WHERE complaint_id = 'PUN-TEST-001'", conn)
            assert len(hist) == 2  # initial + updated
            assert hist.iloc[-1]["new_status"] == "In Progress"
        finally:
            conn.close()
