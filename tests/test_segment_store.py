"""R3: where corrections go (DECISIONS.md §10) and how they come back out.

Cloud keeps nothing unless the visitor opts in; opted-in rows reach the private
bucket and never include an original file; the export turns rows into SFT, DPO
and TMX. Uploads are faked by a recorder, so the tests assert what WOULD have
been written, byte for byte.
"""
import json

import pytest

from passage import contribute, export, traces
from TranslationUI import TranslationUI


@pytest.fixture
def bucket(monkeypatch):
    uploaded = []
    monkeypatch.setattr(contribute, "BUCKET", "passage-contrib-test")
    monkeypatch.setattr(contribute, "uploader", lambda name, body: uploaded.append((name, body)))
    queued = []
    monkeypatch.setattr(contribute, "enqueue", lambda row: queued.append(row))
    return queued, uploaded


def _page(monkeypatch, tmp_path, *, opted_in):
    monkeypatch.setattr("TranslationUI.ui.notify", lambda *a, **k: None)
    monkeypatch.setattr(traces, "TRACE_DIR", tmp_path)
    monkeypatch.setattr(TranslationUI, "session_cache_scope", property(lambda self: "browser:abc"))
    monkeypatch.setattr(TranslationUI, "contribute_enabled", property(lambda self: opted_in))
    page = TranslationUI()
    page.document_trace_id = "t1"
    page.original_segments_map = {"s1": "Good morning"}
    page.translated_segments_map = {"s1": "Buenos dias"}
    return page


def test_cloud_without_opt_in_keeps_nothing_anywhere(bucket, tmp_path, monkeypatch):
    queued, _ = bucket
    page = _page(monkeypatch, tmp_path, opted_in=False)

    page.approve_segment_callback("s1")

    assert queued == [] and traces.read_all() == []


def test_cloud_with_opt_in_sends_the_pair_to_the_bucket_and_not_to_disk(bucket, tmp_path, monkeypatch):
    queued, _ = bucket
    page = _page(monkeypatch, tmp_path, opted_in=True)

    page.approve_segment_callback("s1")

    assert [r["type"] for r in queued] == ["judgement"]
    assert queued[0]["source"] == "Good morning" and queued[0]["output"] == "Buenos dias"
    assert len(queued[0]["session"]) == 16 and "abc" not in queued[0]["session"]
    assert traces.read_all() == []


def test_opt_in_is_ignored_where_no_bucket_is_configured(monkeypatch, tmp_path):
    monkeypatch.setattr(contribute, "BUCKET", "")
    page = _page(monkeypatch, tmp_path, opted_in=True)

    assert page._trace_destination().contribute is False


@pytest.mark.local_deployment
def test_local_deployment_keeps_corrections_on_disk_and_never_contributes(bucket, tmp_path, monkeypatch):
    queued, _ = bucket
    page = _page(monkeypatch, tmp_path, opted_in=True)

    page.approve_segment_callback("s1")

    assert queued == []
    assert [r["type"] for r in traces.read_all()] == ["judgement"]


def test_flush_writes_one_object_per_session_under_the_documented_layout(monkeypatch):
    uploaded = []
    monkeypatch.setattr(contribute, "uploader", lambda name, body: uploaded.append((name, body)))

    names = contribute.flush([{"session": "aaa", "n": 1}, {"session": "bbb", "n": 2},
                              {"session": "aaa", "n": 3}])

    assert len(names) == 2
    by_session = {name.split("/")[2]: body for name, body in uploaded}
    assert all(name.startswith("contributions/") and name.endswith(".jsonl") for name in names)
    assert [json.loads(l)["n"] for l in by_session["aaa"].decode().splitlines()] == [1, 3]


def test_a_failed_upload_is_dropped_not_raised(monkeypatch):
    def boom(name, body):
        raise OSError("network down")

    monkeypatch.setattr(contribute, "uploader", boom)
    assert contribute.flush([{"session": "a"}]) == []


ROWS = [
    {"type": "trace", "trace_id": "t", "target_language": "Spanish", "engine": "local:tg4b"},
    {"type": "generation", "trace_id": "t", "segment_id": "a", "source": "Hello",
     "output": "Hola", "engine": "local:tg4b"},
    {"type": "generation", "trace_id": "t", "segment_id": "b", "source": "The meeting",
     "output": "El meeting", "engine": "local:tg4b"},
    {"type": "generation", "trace_id": "t", "segment_id": "c", "source": "Untouched",
     "output": "Intacto", "engine": "local:tg4b"},
    {"type": "judgement", "trace_id": "t", "segment_id": "a", "source": "Hello",
     "output": "Hola", "approved": True, "at": 1},
    {"type": "score", "trace_id": "t", "segment_id": "b", "before": "El meeting",
     "after": "La reunion", "at": 2},
]


def test_export_sft_has_every_segment_a_human_settled_on_and_nothing_else():
    records = export.to_sft(export.segments(ROWS))

    finals = {r["messages"][1]["content"]: r["messages"][2]["content"] for r in records}
    assert finals == {"Hello": "Hola", "The meeting": "La reunion"}  # "Untouched" excluded
    assert "Spanish" in records[0]["messages"][0]["content"]


def test_export_dpo_pairs_the_model_output_against_the_human_edit():
    records = export.to_dpo(export.segments(ROWS))

    assert len(records) == 1
    assert records[0]["rejected"] == "El meeting" and records[0]["chosen"] == "La reunion"


def test_export_tmx_is_well_formed_and_escapes_text():
    rows = ROWS + [{"type": "judgement", "trace_id": "t", "segment_id": "d",
                    "source": "A & B <c>", "output": "A y B", "approved": True}]
    import xml.dom.minidom

    doc = xml.dom.minidom.parseString(export.to_tmx(export.segments(rows)).encode())
    assert len(doc.getElementsByTagName("tu")) == 3


def test_export_cli_reads_a_directory_tree(tmp_path, capsys):
    nested = tmp_path / "contributions" / "2026-09-24" / "abc"
    nested.mkdir(parents=True)
    (nested / "1.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ROWS), encoding="utf-8")
    out = tmp_path / "dpo.jsonl"

    assert export.main(["--format", "dpo", "--in", str(tmp_path), "--out", str(out)]) == 0
    assert len(out.read_text(encoding="utf-8").splitlines()) == 1


def test_corrections_contributed_after_translating_still_export():
    """Opting in after the document was translated means the bucket never
    received the generation rows. The correction rows must stand alone."""
    rows = [
        {"type": "score", "trace_id": "t", "segment_id": "b", "source": "The meeting",
         "target_language": "Spanish", "before": "El meeting", "after": "La reunion", "at": 2},
        {"type": "judgement", "trace_id": "t", "segment_id": "a", "source": "Hello",
         "output": "Hola", "target_language": "Spanish", "approved": True},
    ]
    segs = export.segments(rows)

    assert len(export.to_dpo(segs)) == 1
    assert len(export.to_sft(segs)) == 2
    assert all("Spanish" in r["messages"][0]["content"] for r in export.to_sft(segs))
