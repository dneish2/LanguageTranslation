"""Traces: Langfuse-shaped, policy-gated, and never able to break a translation."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from passage import policy, traces


def _isolate(monkeypatch, tmp_path):
    monkeypatch.setattr(traces, "TRACE_DIR", tmp_path)


def test_a_document_run_writes_trace_generations_and_edits(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    trace_id = traces.record_trace(traces.Trace(
        name="report.pdf", target_language="Spanish", engine="local:tg", segment_count=2))
    assert trace_id
    traces.record_generation(trace_id=trace_id, segment_id="s1", source="Hello",
                             output="Hola", engine="local:tg")
    traces.record_edit(trace_id=trace_id, segment_id="s1", before="Hola", after="Buenas")

    rows = traces.read_all()
    kinds = [r["type"] for r in rows]
    assert kinds == ["trace", "generation", "score"]
    assert rows[2]["before"] == "Hola" and rows[2]["after"] == "Buenas"


def test_surfaces_that_may_not_keep_traces_write_nothing(monkeypatch, tmp_path):
    """Retention is decided in policy, not at the call site — a caller
    shouldn't have to know the rules to be safe."""
    _isolate(monkeypatch, tmp_path)

    for surface in (policy.Surface.LIVE_TEXT, policy.Surface.IMAGE, policy.Surface.VOICE):
        assert traces.record_trace(traces.Trace(name="x", surface=surface.value)) is None
        traces.record_generation(trace_id="forced", segment_id="s", source="a",
                                 output="b", engine="e", surface=surface)
        traces.record_edit(trace_id="forced", segment_id="s", before="a", after="b",
                           surface=surface)

    assert traces.read_all() == []


def test_a_non_edit_is_not_recorded_as_one(monkeypatch, tmp_path):
    """Saving a segment unchanged is not a correction; recording it would
    inflate the only ground-truth dataset this app gets."""
    _isolate(monkeypatch, tmp_path)
    traces.record_edit(trace_id="t1", segment_id="s1", before="same", after="same")

    assert traces.read_all() == []


def test_edits_record_how_much_changed(monkeypatch, tmp_path):
    """"Which segments were rewritten rather than tweaked" should be
    answerable without recomputing over every row."""
    _isolate(monkeypatch, tmp_path)
    traces.record_edit(trace_id="t", segment_id="tweak",
                       before="La junta rechazó la recompra.",
                       after="La junta rechazó la recompra hoy.")
    traces.record_edit(trace_id="t", segment_id="rewrite",
                       before="La junta rechazó la recompra.",
                       after="Completamente distinto en todos los sentidos.")

    rows = {r["segment_id"]: r for r in traces.read_all()}
    assert rows["tweak"]["rewritten"] is False
    assert rows["rewrite"]["rewritten"] is True
    assert rows["tweak"]["edit_ratio"] > rows["rewrite"]["edit_ratio"]


def test_summary_reports_edit_rate_per_engine(monkeypatch, tmp_path):
    """"Is the local model good enough for my documents" should be answerable
    from evidence rather than vibes."""
    _isolate(monkeypatch, tmp_path)
    for i in range(4):
        traces.record_generation(trace_id="t", segment_id=f"local{i}", source="s",
                                 output="o", engine="local:tg")
    for i in range(4):
        traces.record_generation(trace_id="t", segment_id=f"hosted{i}", source="s",
                                 output="o", engine="hosted:gpt")
    traces.record_edit(trace_id="t", segment_id="local0", before="a", after="b")
    traces.record_edit(trace_id="t", segment_id="local1", before="a", after="b")
    traces.record_edit(trace_id="t", segment_id="hosted0", before="a", after="b")

    summary = traces.summarise(traces.read_all())
    assert summary["generations"] == 8 and summary["edits"] == 3
    assert summary["per_engine"]["local:tg"]["edit_rate"] == 0.5
    assert summary["per_engine"]["hosted:gpt"]["edit_rate"] == 0.25


def test_long_text_is_truncated(monkeypatch, tmp_path):
    """A trace studies the decision; storing whole documents by accident is
    the liability the retention policy declines to take on."""
    _isolate(monkeypatch, tmp_path)
    traces.record_generation(trace_id="t", segment_id="s", source="x" * 9000,
                             output="y" * 9000, engine="e")

    row = traces.read_all()[0]
    assert len(row["source"]) == traces.MAX_TEXT_CHARS
    assert len(row["output"]) == traces.MAX_TEXT_CHARS


def test_an_unwritable_trace_directory_is_survivable(monkeypatch, tmp_path):
    """The record is a by-product, not the product — it must never surface to
    the user or fail the translation it describes."""
    monkeypatch.setattr(traces, "TRACE_DIR", tmp_path / "file-not-dir")
    (tmp_path / "file-not-dir").write_text("in the way", encoding="utf-8")

    trace_id = traces.record_trace(traces.Trace(name="x"))
    traces.record_generation(trace_id=trace_id, segment_id="s", source="a",
                             output="b", engine="e")
    assert traces.read_all() == []


def test_missing_trace_id_writes_nothing(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    traces.record_generation(trace_id=None, segment_id="s", source="a", output="b", engine="e")
    traces.record_edit(trace_id=None, segment_id="s", before="a", after="b")

    assert traces.read_all() == []
