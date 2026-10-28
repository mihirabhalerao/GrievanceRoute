"""
Tests for Voice-to-Text Complaint Submission Integration.
Verifies language configuration, transcription validation, empty input handling,
review/edit behavior, and end-to-end triage integration.
"""

import tempfile
from pathlib import Path
import pytest

from grievanceroute.services.voice_service import (
    SUPPORTED_VOICE_LANGUAGES,
    get_supported_languages,
    validate_transcription,
    format_voice_submission_metadata,
)
from grievanceroute.services.complaint_service import ComplaintService
from grievanceroute.database.repository import GrievanceRepository


def test_supported_languages_configuration():
    """Verify configured languages include English, Hindi, and Marathi."""
    langs = get_supported_languages()
    assert "en-IN" in langs
    assert "hi-IN" in langs
    assert "mr-IN" in langs

    assert "English" in langs["en-IN"]["label"]
    assert "Hindi" in langs["hi-IN"]["label"]
    assert "Marathi" in langs["mr-IN"]["label"]


def test_transcription_validation():
    """Verify transcription text validation rules."""
    # None or empty
    ok, cleaned, err = validate_transcription(None)
    assert ok is False
    assert "No transcription" in err

    ok, cleaned, err = validate_transcription("   ")
    assert ok is False
    assert "empty" in err

    # Below minimum length (10 chars)
    ok, cleaned, err = validate_transcription("Short", min_length=10)
    assert ok is False
    assert "too short" in err
    assert cleaned == "Short"

    # Valid transcription
    sample = "  Large pothole near Deccan Gymkhana chowk  "
    ok, cleaned, err = validate_transcription(sample, min_length=10)
    assert ok is True
    assert cleaned == "Large pothole near Deccan Gymkhana chowk"
    assert err == ""


def test_voice_submission_metadata_formatting():
    """Verify metadata records language code and edit history without storing audio."""
    raw = "Pothole on road"
    edited = "Dangerous deep pothole on Karve road near signal"
    meta = format_voice_submission_metadata("mr-IN", raw, edited)

    assert meta["input_channel"] == "voice"
    assert meta["recognition_language"] == "mr-IN"
    assert meta["was_edited_by_user"] is True
    assert meta["final_length"] > meta["raw_length"]
    # Ensure audio payloads are never preserved
    assert "audio" not in meta


def test_end_to_end_voice_complaint_intake():
    """
    Verify simulated voice transcription flows into ComplaintService,
    triggers AI classification, safety overrides, and SQLite persistence.
    """
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        test_db = Path(tmpdir) / "test_voice.db"
        repo = GrievanceRepository(test_db)
        svc = ComplaintService(repo=repo)

        # 1. Citizen dictates in Marathi (mocked transcription)
        voice_transcript = "रस्त्यावर मोठा खड्डा पडला आहे आणि दुचाकी घसरत आहेत"
        ok, valid_text, _ = validate_transcription(voice_transcript)
        assert ok is True

        # 2. Submit through existing complaint pipeline
        success, msg, record = svc.submit_complaint(
            complaint_text=valid_text,
            latitude=18.5074,
            longitude=73.8077,
            zone="Kothrud",
            weather="Rainy",
            monsoon=True,
        )

        assert success is True
        assert record["complaint_id"].startswith("PUN-LIVE-")
        assert record["complaint_text"] == valid_text
        assert record["urgency_score"] >= 1
        assert "predicted_category" in record

        # 3. Verify SQLite persistence
        db_rec = repo.get_complaint_by_id(record["complaint_id"])
        assert db_rec is not None
        assert db_rec["complaint_text"] == valid_text


def test_safety_override_triggered_via_voice_dictation():
    """
    Verify voice-dictated emergency (e.g. sparking live electrical wire)
    correctly triggers safety overrides.
    """
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        test_db = Path(tmpdir) / "test_voice_safety.db"
        repo = GrievanceRepository(test_db)
        svc = ComplaintService(repo=repo)

        voice_dictation = "There is a live electrical wire sparking violently on the street near school!"
        success, msg, record = svc.submit_complaint(
            complaint_text=voice_dictation,
            latitude=18.5314,
            longitude=73.8446,
            zone="Shivajinagar",
        )

        assert success is True
        assert record["safety_override"] is True
        assert record["urgency_score"] == 5
        assert record["crew_required"] == "Electrical Crew"
