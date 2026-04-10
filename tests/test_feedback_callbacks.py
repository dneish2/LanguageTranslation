import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TranslationBackend import TranslationBackend
from TranslationUI import TranslationUI


def _mute_ui_notifications(monkeypatch):
    monkeypatch.setattr("TranslationUI.ui.notify", lambda *args, **kwargs: None)


def test_approve_decline_callbacks_do_not_raise_typeerror(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("FEEDBACK_DIR", str(tmp_path))
    _mute_ui_notifications(monkeypatch)

    ui_app = TranslationUI()
    ui_app.original_segments_map = {"seg-1": "Hello"}
    ui_app.translated_segments_map = {"seg-1": "Hola"}

    ui_app.approve_segment_callback("seg-1")
    ui_app.decline_segment_callback("seg-1")

    feedback_file = tmp_path / "trl_finetune_data.jsonl"
    assert feedback_file.exists()

    rows = feedback_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 1

    record = json.loads(rows[0])
    assert "Translate to" in record["prompt"]
    assert record["completion"].strip() == "Hola"


def test_record_feedback_requires_keywords(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    backend = TranslationBackend()

    with pytest.raises(TypeError):
        backend.record_feedback(True, "Hello", "Hola")


def test_record_feedback_validation(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    backend = TranslationBackend()

    with pytest.raises(TypeError):
        backend.record_feedback(approved="yes", original="Hello", translated="Hola")

    with pytest.raises(ValueError):
        backend.record_feedback(approved=True, original="   ", translated="Hola")
