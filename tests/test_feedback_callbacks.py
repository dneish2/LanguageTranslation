"""Approve / decline in segment review become judgement rows in the one trace
store, not a separate trl_finetune_data.jsonl in the working directory (which on
Cloud Run was wiped at every restart, and stamped the language from state shared
across sessions)."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from passage import traces
from TranslationUI import TranslationUI


def _ui(monkeypatch, tmp_path):
    monkeypatch.setattr("TranslationUI.ui.notify", lambda *args, **kwargs: None)
    monkeypatch.setattr(traces, "TRACE_DIR", tmp_path)
    monkeypatch.setattr(TranslationUI, "session_cache_scope", property(lambda self: "browser:test"))
    ui_app = TranslationUI()
    ui_app.document_trace_id = "trace-1"
    ui_app.original_segments_map = {"seg-1": "Hello"}
    ui_app.translated_segments_map = {"seg-1": "Hola"}
    return ui_app


@pytest.mark.local_deployment
def test_approve_and_decline_are_recorded_as_judgements_on_disk_locally(tmp_path, monkeypatch):
    ui_app = _ui(monkeypatch, tmp_path)

    ui_app.approve_segment_callback("seg-1")
    ui_app.decline_segment_callback("seg-1")

    rows = [r for r in traces.read_all() if r["type"] == "judgement"]
    assert [r["approved"] for r in rows] == [True, False]
    assert rows[0]["source"] == "Hello" and rows[0]["output"] == "Hola"
    assert rows[0]["session"] and "browser:test" not in rows[0]["session"]  # hashed
    assert not list(Path.cwd().glob("trl_finetune_data.jsonl"))


def test_on_cloud_nothing_is_written_to_disk_but_the_page_keeps_its_rows(tmp_path, monkeypatch):
    ui_app = _ui(monkeypatch, tmp_path)

    ui_app.approve_segment_callback("seg-1")

    assert traces.read_all() == []
    assert [r["type"] for r in ui_app.session_trace_rows] == ["judgement"]
