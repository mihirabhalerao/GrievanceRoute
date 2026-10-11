"""
Database Repository for GrievanceRoute.
Handles parameterized SQLite transactions, CRUD operations, and state transitions.
Ensures connections are properly closed to avoid file locks on Windows.
"""

from __future__ import annotations
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd

from grievanceroute.database.models import (
    CREATE_COMPLAINTS_TABLE,
    CREATE_STATUS_HISTORY_TABLE,
    CREATE_CREWS_TABLE,
    CREATE_ROUTES_TABLE,
)


class GrievanceRepository:
    """SQLite data access repository with parameterized statements."""

    def __init__(self, db_path: str | Path = "data/grievance_route.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize_schema()

    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def initialize_schema(self) -> None:
        """Create tables if they do not exist."""
        conn = self.get_connection()
        try:
            with conn:
                cursor = conn.cursor()
                cursor.execute(CREATE_COMPLAINTS_TABLE)
                cursor.execute(CREATE_STATUS_HISTORY_TABLE)
                cursor.execute(CREATE_CREWS_TABLE)
                cursor.execute(CREATE_ROUTES_TABLE)
        finally:
            conn.close()

    def insert_complaint(self, data: Dict[str, Any]) -> bool:
        """Insert a newly submitted civic complaint."""
        sql = """
        INSERT INTO complaints (
            complaint_id, complaint_text, complaint_type, latitude, longitude, zone,
            timestamp, hour, day_of_week, weather, monsoon, low_lying_area,
            repeat_complaint, historical_resolution_hours, urgency_score,
            actual_resolution_hours, status, crew_required
        ) VALUES (
            :complaint_id, :complaint_text, :complaint_type, :latitude, :longitude, :zone,
            :timestamp, :hour, :day_of_week, :weather, :monsoon, :low_lying_area,
            :repeat_complaint, :historical_resolution_hours, :urgency_score,
            :actual_resolution_hours, :status, :crew_required
        )
        """
        conn = self.get_connection()
        try:
            with conn:
                cursor = conn.cursor()
                cursor.execute(sql, {
                    "complaint_id": data["complaint_id"],
                    "complaint_text": data["complaint_text"],
                    "complaint_type": data["complaint_type"],
                    "latitude": float(data["latitude"]),
                    "longitude": float(data["longitude"]),
                    "zone": data["zone"],
                    "timestamp": str(data["timestamp"]),
                    "hour": int(data.get("hour", 12)),
                    "day_of_week": int(data.get("day_of_week", 0)),
                    "weather": data.get("weather", "Clear"),
                    "monsoon": int(bool(data.get("monsoon", False))),
                    "low_lying_area": int(bool(data.get("low_lying_area", False))),
                    "repeat_complaint": int(bool(data.get("repeat_complaint", False))),
                    "historical_resolution_hours": float(data.get("historical_resolution_hours", 24.0)),
                    "urgency_score": int(data.get("urgency_score", 3)),
                    "actual_resolution_hours": float(data.get("actual_resolution_hours", 24.0)),
                    "status": data.get("status", "Pending"),
                    "crew_required": data.get("crew_required", "Road Maintenance"),
                })
                # Insert initial history record
                cursor.execute("""
                    INSERT INTO status_history (complaint_id, old_status, new_status, notes)
                    VALUES (?, ?, ?, ?)
                """, (data["complaint_id"], "None", data.get("status", "Pending"), "Complaint submitted."))
            return True
        finally:
            conn.close()

    def get_all_complaints(self, limit: int = 1000) -> pd.DataFrame:
        """Fetch all complaints up to limit into DataFrame."""
        conn = self.get_connection()
        try:
            df = pd.read_sql_query(f"SELECT * FROM complaints ORDER BY timestamp DESC LIMIT {limit}", conn)
            return df
        finally:
            conn.close()

    def get_complaint_by_id(self, complaint_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve single complaint record by ID."""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM complaints WHERE complaint_id = ?", (complaint_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def update_complaint_status(self, complaint_id: str, new_status: str, notes: str = "") -> bool:
        """Update complaint operational status and record history log."""
        conn = self.get_connection()
        try:
            with conn:
                cursor = conn.cursor()
                cursor.execute("SELECT status FROM complaints WHERE complaint_id = ?", (complaint_id,))
                row = cursor.fetchone()
                if not row:
                    return False
                old_status = row["status"]
                cursor.execute("UPDATE complaints SET status = ? WHERE complaint_id = ?", (new_status, complaint_id))
                cursor.execute("""
                    INSERT INTO status_history (complaint_id, old_status, new_status, notes)
                    VALUES (?, ?, ?, ?)
                """, (complaint_id, old_status, new_status, notes))
            return True
        finally:
            conn.close()

    def seed_initial_complaints(self, csv_path: str | Path, max_rows: int = 500) -> int:
        """Seed SQLite with complaints from processed CSV if database is empty."""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM complaints")
            count = cursor.fetchone()[0]
            if count > 0:
                return count
        finally:
            conn.close()

        df = pd.read_csv(csv_path).head(max_rows)
        for _, row in df.iterrows():
            self.insert_complaint(row.to_dict())
        return len(df)
