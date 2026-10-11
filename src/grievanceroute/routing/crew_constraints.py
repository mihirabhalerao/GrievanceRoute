"""
Crew Constraints and Schedule Feasibility for GrievanceRoute.
Checks skill compatibility, crew capacity, shift time budgets, and overtime penalties.
"""

from __future__ import annotations
from typing import Dict, List, Any, Tuple


SERVICE_MINUTES_PER_COMPLAINT = 45.0  # standard on-site repair duration


def check_skill_compatibility(crew_skill: str, complaint_type: str) -> bool:
    """Check if crew skill matches required complaint category."""
    mapping = {
        "Road Maintenance": "pothole",
        "Water Supply Crew": "water leak",
        "Electrical Crew": "streetlight",
        "Sanitation Crew": "garbage",
    }
    return mapping.get(crew_skill) == complaint_type


def evaluate_crew_shift(
    crew_info: Dict[str, Any],
    assigned_complaints: List[Dict[str, Any]],
    total_travel_distance_km: float,
    speed_kmh: float = 25.0,
    max_shift_hours: float = 8.0,
) -> Dict[str, Any]:
    """
    Evaluate crew shift constraints:
    Total time = travel_time + total_service_time.
    Calculates overtime, violations, and completed count.
    """
    n_complaints = len(assigned_complaints)
    capacity = int(crew_info.get("capacity", 8))
    skill = crew_info.get("skill", "")

    travel_hours = total_travel_distance_km / max(1.0, speed_kmh)
    service_hours = (n_complaints * SERVICE_MINUTES_PER_COMPLAINT) / 60.0
    total_hours = travel_hours + service_hours

    overtime_hours = max(0.0, total_hours - max_shift_hours)

    violations = []
    if n_complaints > capacity:
        violations.append(f"Capacity exceeded ({n_complaints} > {capacity} allowed).")
    if overtime_hours > 0:
        violations.append(f"Shift overrun by {overtime_hours:.1f} hours.")

    # Check skills
    for c in assigned_complaints:
        c_type = c.get("complaint_type", "")
        if not check_skill_compatibility(skill, c_type):
            violations.append(f"Skill mismatch: {skill} cannot service {c_type} ({c.get('complaint_id')}).")

    return {
        "crew_id": crew_info.get("crew_id"),
        "assigned_count": n_complaints,
        "capacity": capacity,
        "travel_hours": round(travel_hours, 2),
        "service_hours": round(service_hours, 2),
        "total_hours": round(total_hours, 2),
        "overtime_hours": round(overtime_hours, 2),
        "is_feasible": len(violations) == 0,
        "violations": violations,
    }
