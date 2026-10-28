"""
Voice Input Service for GrievanceRoute.
Provides language configuration, transcription validation, and workflow integration
for multilingual speech-to-text complaint intake.
"""

from __future__ import annotations
from typing import Dict, Tuple, Any, Optional

SUPPORTED_VOICE_LANGUAGES: Dict[str, Dict[str, str]] = {
    "en-IN": {
        "label": "English (India)",
        "native": "English",
        "description": "Standard Indian English accents and civic vocabulary.",
    },
    "hi-IN": {
        "label": "Hindi (हिन्दी)",
        "native": "हिन्दी",
        "description": "Devanagari Hindi civic complaints (सड़क, कचरा, पानी).",
    },
    "mr-IN": {
        "label": "Marathi (मराठी)",
        "native": "मराठी",
        "description": "Regional Marathi complaints (खड्डा, दिवाबत्ती, पाणी गळती).",
    },
}

DEFAULT_VOICE_LANGUAGE = "en-IN"


def get_supported_languages() -> Dict[str, Dict[str, str]]:
    """Return dictionary of supported speech recognition language codes and metadata."""
    return SUPPORTED_VOICE_LANGUAGES.copy()


def validate_transcription(transcript: Optional[str], min_length: int = 10) -> Tuple[bool, str, str]:
    """
    Validate transcribed text prior to submission.
    Returns (is_valid, cleaned_text, error_message).
    """
    if transcript is None:
        return False, "", "No transcription provided."

    cleaned = transcript.strip()
    if not cleaned:
        return False, "", "Voice transcription is empty. Please speak clearly into your microphone."

    if len(cleaned) < min_length:
        return False, cleaned, f"Transcription too short ({len(cleaned)} characters). Minimum {min_length} required."

    return True, cleaned, ""


def format_voice_submission_metadata(
    language_code: str,
    raw_transcript: str,
    edited_transcript: str,
) -> Dict[str, Any]:
    """
    Produce audit metadata tracking voice origin, language code, and user edits.
    Audio itself is never stored or persisted.
    """
    lang_info = SUPPORTED_VOICE_LANGUAGES.get(language_code, {"label": language_code})
    was_edited = (raw_transcript.strip() != edited_transcript.strip())
    return {
        "input_channel": "voice",
        "recognition_language": language_code,
        "language_name": lang_info.get("label", language_code),
        "was_edited_by_user": was_edited,
        "raw_length": len(raw_transcript.strip()),
        "final_length": len(edited_transcript.strip()),
    }
