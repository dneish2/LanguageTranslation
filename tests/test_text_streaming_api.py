import asyncio
import json
import os
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TranslationUI import TranslationUI


def _build_ui() -> TranslationUI:
    return TranslationUI()


def _app_request(ui_app: TranslationUI):
    """A request carrying the token the app's own pages embed."""
    return types.SimpleNamespace(
        headers={"x-passage-token": ui_app.api_guard.issue_token()},
        client=types.SimpleNamespace(host="127.0.0.1"),
    )


def test_stream_endpoint_fallbacks_to_non_streaming(monkeypatch):
    ui_app = _build_ui()
    monkeypatch.setenv("LIVE_TEXT_STREAMING", "false")
    monkeypatch.setenv("LIVE_TEXT_STREAMING_CHAR_THRESHOLD", "999")
    monkeypatch.setattr(ui_app.backend, "translate_text", lambda text, language: f"{language}:{text}")

    resp = asyncio.run(ui_app.api_text_translate_stream(_app_request(ui_app), text="short", language="es"))

    assert resp.media_type == "application/json"
    payload = json.loads(resp.body.decode())
    assert payload["fallback"] is True
    assert payload["translated_text"] == "es:short"


# The streaming cases live in test_live_streaming.py, against the model's
# real stream rather than a pre-sliced answer.
