"""
Tests for Interactive Map-Based Complaint Location Selection.
Verifies coordinate validation, pin updates, SQLite persistence, and downstream data integrity.
"""

import tempfile
from pathlib import Path
import pytest

from grievanceroute.data.generator import PUNE_ZONES
from grievanceroute.database.repository import GrievanceRepository
from grievanceroute.services.complaint_service import ComplaintService


def test_pune_zone_defaults_and_coordinates():
    """Verify all defined service zones contain valid latitude and longitude within Pune metro area."""
    assert len(PUNE_ZONES) >= 6
    for zone_name, meta in PUNE_ZONES.items():
        assert "lat" in meta
        assert "lon" in meta
        assert 18.40 <= meta["lat"] <= 18.65, f"Latitude {meta['lat']} outside Pune metro range"
        assert 73.70 <= meta["lon"] <= 74.05, f"Longitude {meta['lon']} outside Pune metro range"


def test_map_selected_coordinates_persisted_in_database():
    """Verify that coordinates selected from an interactive map are stored and retrievable."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        test_db = Path(tmpdir) / "test_map_pin.db"
        repo = GrievanceRepository(test_db)
        svc = ComplaintService(repo=repo)

        # Coordinates clicked by user on FC Road junction
        clicked_lat = 18.523612
        clicked_lon = 73.841845

        ok, msg, record = svc.submit_complaint(
            complaint_text="Large pothole near FC Road Starbucks chowk causing severe traffic.",
            latitude=clicked_lat,
            longitude=clicked_lon,
            zone="Shivajinagar",
            weather="Clear",
        )

        assert ok is True
        assert record["latitude"] == clicked_lat
        assert record["longitude"] == clicked_lon

        # Query database to confirm exact float persistence
        stored = repo.get_complaint_by_id(record["complaint_id"])
        assert stored is not None
        assert abs(stored["latitude"] - clicked_lat) < 1e-5
        assert abs(stored["longitude"] - clicked_lon) < 1e-5
        assert stored["zone"] == "Shivajinagar"


def test_custom_pin_outside_zone_center():
    """Verify user can pin anywhere in Pune regardless of chosen zone default."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        test_db = Path(tmpdir) / "test_custom_pin.db"
        repo = GrievanceRepository(test_db)
        svc = ComplaintService(repo=repo)

        # User selects Kothrud zone but pins on Karve statue
        pinned_lat = 18.501234
        pinned_lon = 73.815678

        ok, msg, record = svc.submit_complaint(
            complaint_text="Streetlight pole knocked down near Karve Statue signal.",
            latitude=pinned_lat,
            longitude=pinned_lon,
            zone="Kothrud",
        )

        assert ok is True
        stored = repo.get_complaint_by_id(record["complaint_id"])
        assert stored["latitude"] == pinned_lat
        assert stored["longitude"] == pinned_lon
