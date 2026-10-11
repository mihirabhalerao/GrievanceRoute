"""
Complaint Service for GrievanceRoute.
Validates incoming citizen submissions, integrates prediction pipeline,
and persists records in SQLite.
"""

from __future__ import annotations
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import pandas as pd

from grievanceroute.database.repository import GrievanceRepository
from grievanceroute.services.prediction_service import PredictionService


class ComplaintService:
    """Manages complaint intake, triage orchestration, and storage."""

    def __init__(
        self,
        repo: Optional[GrievanceRepository] = None,
        prediction_svc: Optional[PredictionService] = None,
    ):
        self.repo = repo or GrievanceRepository()
        self.prediction_svc = prediction_svc or PredictionService()

    def submit_complaint(
        self,
        complaint_text: str,
        latitude: float,
        longitude: float,
        zone: str,
        weather: str = "Clear",
        monsoon: bool = False,
        low_lying_area: bool = False,
        repeat_complaint: bool = False,
        historical_resolution_hours: float = 24.0,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Validate input, generate predictions, and persist complaint."""
        clean_text = complaint_text.strip()
        if len(clean_text) < 10:
            return False, "Complaint description is too short (minimum 10 characters required).", {}

        now = datetime.now()
        context = {
            "latitude": float(latitude),
            "longitude": float(longitude),
            "zone": zone,
            "hour": now.hour,
            "day_of_week": now.weekday(),
            "weather": weather,
            "monsoon": bool(monsoon),
            "low_lying_area": bool(low_lying_area),
            "repeat_complaint": bool(repeat_complaint),
            "historical_resolution_hours": float(historical_resolution_hours),
        }

        # Predict triage metadata
        triage = self.prediction_svc.predict_complaint(clean_text, context)

        complaint_id = f"PUN-LIVE-{uuid.uuid4().hex[:6].upper()}"
        record = {
            "complaint_id": complaint_id,
            "complaint_text": clean_text,
            "complaint_type": triage["predicted_category"],
            "latitude": float(latitude),
            "longitude": float(longitude),
            "zone": zone,
            "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
            "hour": now.hour,
            "day_of_week": now.weekday(),
            "weather": weather,
            "monsoon": int(bool(monsoon)),
            "low_lying_area": int(bool(low_lying_area)),
            "repeat_complaint": int(bool(repeat_complaint)),
            "historical_resolution_hours": float(historical_resolution_hours),
            "urgency_score": triage["urgency_score"],
            "actual_resolution_hours": triage["estimated_resolution_hours"],
            "status": "Pending",
            "crew_required": triage["crew_required"],
        }

        success = self.repo.insert_complaint(record)
        record.update(triage)
        return success, f"Complaint {complaint_id} triaged and persisted successfully.", record
