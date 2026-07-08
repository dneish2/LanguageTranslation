"""Shared, module-level constants and helpers used across passage.ui pages.

Lives outside TranslationUI.py so page modules (voice_page.py, ...) can import
these without a circular import back to the main entry-point module.
"""
import json
import logging
from typing import Any

LOGGER = logging.getLogger("translation.ui")

# Static language list for type-ahead fields. Free text stays allowed —
# the backend takes any language name; this is autofill, not a whitelist.
LANGUAGES = [
    "English", "Spanish", "French", "German", "Italian", "Portuguese",
    "Dutch", "Russian", "Ukrainian", "Polish", "Turkish", "Arabic",
    "Hebrew", "Hindi", "Chinese", "Japanese", "Korean", "Vietnamese",
    "Thai", "Indonesian", "Swedish", "Norwegian", "Danish", "Finnish",
    "Greek", "Czech", "Romanian", "Hungarian",
]


def log_event(event: str, correlation_id: str | None = None, **fields: Any) -> None:
    payload: dict[str, Any] = {"event": event, **fields}
    if correlation_id:
        payload["correlation_id"] = correlation_id
    LOGGER.info(json.dumps(payload, default=str))
