"""
Streamlit Custom Component for Multilingual Voice-to-Text Input.
Wraps the browser's native Web Speech API with fallback support.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional

import streamlit.components.v1 as components

_FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"
_voice_input_component = components.declare_component(
    "voice_input_component",
    path=str(_FRONTEND_DIR),
)


def voice_input_widget(
    language: str = "en-IN",
    key: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Render browser voice input component.

    Parameters
    ----------
    language : str
        BCP-47 language tag (e.g. 'en-IN', 'hi-IN', 'mr-IN').
    key : Optional[str]
        Streamlit widget key.

    Returns
    -------
    Optional[Dict[str, Any]]
        Dictionary containing transcription, status, error (if any), and timestamp.
    """
    return _voice_input_component(language=language, key=key, default=None)
