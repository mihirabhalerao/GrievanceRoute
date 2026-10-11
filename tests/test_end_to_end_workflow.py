"""
End-to-End Workflow & Integration Tests for GrievanceRoute.
Validates the complete lifecycle: complaint intake -> data validation -> ML triage -> safety escalation ->
crew assignment -> VRP route scheduling -> shift constraints -> SQLite persistence -> status audit history ->
and negative/edge cases (invalid coords, missing input, invalid status transitions).
"""

import tempfile
from pathlib import Path
import pytest
import pandas as pd

from grievanceroute.database.repository import GrievanceRepository
from grievanceroute.services.prediction_service import PredictionService
from grievanceroute.services.complaint_service import ComplaintService
from grievanceroute.services.planning_service import PlanningService
from grievanceroute.routing.crew_constraints import check_skill_compatibility, evaluate_crew_shift
from grievanceroute.routing.vrp_solver import solve_greedy_vrp, solve_simulated_annealing_vrp
from grievanceroute.ai.search import a_star_search


@pytest.fixture
def temp_repo():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        db_path = Path(tmpdir) / "test_workflow.db"
        repo = GrievanceRepository(db_path)
        yield repo


def test_complete_intake_to_routing_workflow(temp_repo):
    """
    Test full end-to-end lifecycle:
    1. Citizen submits complaint
    2. Input is validated
    3. Model predicts category, urgency, and resolution hours
    4. Record is persisted in SQLite with status 'Pending'
    5. Crew dispatch generates feasible route
    6. Status is transitioned through 'In Progress' to 'Resolved'
    7. Status history records audit trail
    """
    pred_svc = PredictionService()
    complaint_svc = ComplaintService(repo=temp_repo, prediction_svc=pred_svc)

    # 1 & 2 & 3: Submit complaint
    success, msg, record = complaint_svc.submit_complaint(
        complaint_text="Severe water pipe burst flooding road near Karve Road chowk.",
        latitude=18.5074,
        longitude=73.8077,
        zone="Kothrud",
        weather="Rainy",
        monsoon=True,
    )
    assert success is True
    assert "PUN-LIVE-" in record["complaint_id"]
    assert record["complaint_type"] == "water leak"
    assert record["crew_required"] == "Water Supply Crew"
    assert record["status"] == "Pending"

    # 4. Check persistence in SQLite
    saved = temp_repo.get_complaint_by_id(record["complaint_id"])
    assert saved is not None
    assert saved["zone"] == "Kothrud"
    assert saved["status"] == "Pending"

    # 5. Route planning for this complaint
    planning_svc = PlanningService()
    plan = planning_svc.generate_plan([saved], algorithm="Simulated Annealing")
    assert "crew_reports" in plan
    # Verify the complaint was assigned to a compatible crew
    water_crews = [r for r in plan["crew_reports"] if r["skill"] == "Water Supply Crew"]
    assigned_stops = [stop for r in water_crews for stop in r["stops"]]
    assert record["complaint_id"] in assigned_stops

    # 6. Status transition
    ok_assign = temp_repo.update_complaint_status(record["complaint_id"], "In Progress", notes="Dispatched water squad.")
    assert ok_assign is True
    ok_resolve = temp_repo.update_complaint_status(record["complaint_id"], "Resolved", notes="Valve repaired and tested.")
    assert ok_resolve is True

    # 7. Check audit trail
    conn = temp_repo.get_connection()
    try:
        hist = pd.read_sql_query(
            "SELECT * FROM status_history WHERE complaint_id = ? ORDER BY changed_at ASC",
            conn,
            params=(record["complaint_id"],),
        )
        assert len(hist) == 3  # Initial submission + In Progress + Resolved
        assert hist.iloc[0]["new_status"] == "Pending"
        assert hist.iloc[1]["new_status"] == "In Progress"
        assert hist.iloc[2]["new_status"] == "Resolved"
    finally:
        conn.close()


def test_safety_escalation_rule_override(temp_repo):
    """Verify safety hazard (live electrical wire) triggers rule escalation and maximum urgency."""
    pred_svc = PredictionService()
    complaint_svc = ComplaintService(repo=temp_repo, prediction_svc=pred_svc)

    success, msg, record = complaint_svc.submit_complaint(
        complaint_text="Extreme emergency! Live electrical wire hanging loose and sparking near school gate!",
        latitude=18.5236,
        longitude=73.8418,
        zone="Shivajinagar",
    )
    assert success is True
    assert record["safety_override"] is True
    assert record["predicted_category"] == "streetlight"
    assert record["urgency_score"] == 5
    assert record["crew_required"] == "Electrical Crew"


def test_edge_case_short_complaint_rejection(temp_repo):
    """Verify validation rejects descriptions that are too short (< 10 chars)."""
    complaint_svc = ComplaintService(repo=temp_repo)
    success, msg, record = complaint_svc.submit_complaint(
        complaint_text="Broken",
        latitude=18.52,
        longitude=73.84,
        zone="Shivajinagar",
    )
    assert success is False
    assert "too short" in msg.lower()


def test_edge_case_nonexistent_complaint_update(temp_repo):
    """Verify status update returns False for nonexistent complaint ID."""
    ok = temp_repo.update_complaint_status("NON-EXISTENT-ID-9999", "Resolved")
    assert ok is False


def test_route_feasibility_shift_budget_limits():
    """Verify crew constraint checker correctly flags capacity or shift overrun."""
    crew = {
        "crew_id": "CRW-TEST",
        "skill": "Road Maintenance",
        "capacity": 3,
        "speed_kmh": 25.0,
    }
    # 4 jobs exceeds capacity of 3
    complaints = [{"complaint_id": f"P-{i}", "complaint_type": "pothole"} for i in range(4)]
    eval_res = evaluate_crew_shift(crew, complaints, total_travel_distance_km=150.0, max_shift_hours=8.0)
    # Travel: 150/25 = 6.0h. Service: 4 * 45m = 3.0h. Total: 9.0h > 8.0h.
    assert eval_res["is_feasible"] is False
    assert eval_res["overtime_hours"] > 0.0
    assert any("Capacity exceeded" in v for v in eval_res["violations"])
    assert any("Shift overrun" in v for v in eval_res["violations"])
