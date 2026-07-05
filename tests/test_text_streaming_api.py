import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TranslationUI import TranslationUI


def _build_ui() -> TranslationUI:
    return TranslationUI()


def test_stream_endpoint_fallbacks_to_non_streaming(monkeypatch):
    ui_app = _build_ui()
    monkeypatch.setenv("LIVE_TEXT_STREAMING", "false")
    monkeypatch.setenv("LIVE_TEXT_STREAMING_CHAR_THRESHOLD", "999")
    monkeypatch.setattr(ui_app.backend, "translate_text", lambda text, language: f"{language}:{text}")

    resp = asyncio.run(ui_app.api_text_translate_stream(text="short", language="es"))

    assert resp.media_type == "application/json"
    payload = json.loads(resp.body.decode())
    assert payload["fallback"] is True
    assert payload["translated_text"] == "es:short"


def test_stream_endpoint_emits_start_and_complete(monkeypatch):
    ui_app = _build_ui()
    monkeypatch.setenv("LIVE_TEXT_STREAMING", "true")
    monkeypatch.setattr(
        ui_app.backend,
        "stream_translate_text",
        lambda text, language: ("hola mundo", ["hola", "hola mundo"]),
    )

    resp = asyncio.run(ui_app.api_text_translate_stream(text="hello world", language="es"))

    async def _collect_events():
        chunks = []
        async for part in resp.body_iterator:
            chunks.append(part.decode() if isinstance(part, bytes) else part)
        return "".join(chunks)

    body = asyncio.run(_collect_events())
    assert "event: start" in body
    assert "event: partial" in body
    assert "event: complete" in body
    assert '"canonical": true' in body


def test_stream_endpoint_emits_error_event(monkeypatch):
    ui_app = _build_ui()
    monkeypatch.setenv("LIVE_TEXT_STREAMING", "true")

    def _raise(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(ui_app.backend, "stream_translate_text", _raise)

    resp = asyncio.run(ui_app.api_text_translate_stream(text="hello world", language="es"))

    async def _collect_events():
        chunks = []
        async for part in resp.body_iterator:
            chunks.append(part.decode() if isinstance(part, bytes) else part)
        return "".join(chunks)

    body = asyncio.run(_collect_events())
    assert "event: error" in body
    assert "boom" in body
