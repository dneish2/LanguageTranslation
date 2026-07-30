"""Document-fidelity regressions: D4 (Save All Edits discarded edits), D5 (DOCX
inline formatting and hyperlinks destroyed), D6 (malformed uploads leaked one
identical python-docx message), D8 (token counts reported as zero / one-sided).

Every assertion here opens the produced artefact and looks inside it. A test
that only proved "the call returned" would pass even if the edit never reached
the document, which is exactly the bug D4 describes.
"""
import json
import sys
import zipfile
from io import BytesIO
from pathlib import Path

import fitz
import pytest
from docx import Document
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.oxml.ns import qn
from docx.oxml.shared import OxmlElement
from pptx import Presentation

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TranslationBackend import TranslationBackend  # noqa: E402
from TranslationUI import TranslationUI  # noqa: E402
from passage import traces  # noqa: E402

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


# ─────────────────────────── fixtures / helpers ───────────────────────────

def _fake_translation(text: str) -> str:
    return f"ES::{text}"


def _install_fake_translator(backend, monkeypatch, calls=None):
    """Replace the network call, NOT the code under test.

    Returns the list that records every source string handed to the translator,
    so a test can prove the document path actually ran and how many units it
    sent.
    """
    seen = [] if calls is None else calls

    def fake_translate_text(text, target_language, correlation_id=None, file_metrics=None):
        seen.append(text)
        return _fake_translation(text)

    monkeypatch.setattr(backend, "translate_text", fake_translate_text)
    return seen


def _docx_bytes(build) -> bytes:
    doc = Document()
    build(doc)
    buf = BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf.getvalue()


def _add_hyperlink(paragraph, url: str, text: str) -> None:
    """python-docx has no hyperlink API; build the element the way Word does."""
    r_id = paragraph.part.relate_to(url, RT.HYPERLINK, is_external=True)
    link = OxmlElement("w:hyperlink")
    link.set(qn("r:id"), r_id)
    run = OxmlElement("w:r")
    t = OxmlElement("w:t")
    t.text = text
    run.append(t)
    link.append(run)
    paragraph._p.append(link)


def _bold_and_plain_paragraph(doc):
    para = doc.add_paragraph()
    bold_run = para.add_run("Important notice. ")
    bold_run.bold = True
    return para


def _output_xml(out_stream) -> str:
    out_stream.seek(0)
    with zipfile.ZipFile(BytesIO(out_stream.read())) as zf:
        return zf.read("word/document.xml").decode("utf-8")


def _output_rels(out_stream) -> str:
    out_stream.seek(0)
    with zipfile.ZipFile(BytesIO(out_stream.read())) as zf:
        return zf.read("word/_rels/document.xml.rels").decode("utf-8")


def _reopen(out_stream) -> Document:
    out_stream.seek(0)
    return Document(BytesIO(out_stream.read()))


@pytest.fixture
def backend(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    return TranslationBackend()


# ───────────────────────────── D5: DOCX fidelity ─────────────────────────────

def test_single_run_bold_survives_translation(backend, monkeypatch):
    """The common case: one formatted run per paragraph. Bold must survive AND
    the text must actually be the translation (not the untouched source)."""
    seen = _install_fake_translator(backend, monkeypatch)

    def build(doc):
        para = doc.add_paragraph()
        run = para.add_run("Important notice")
        run.bold = True
        run.italic = True

    out_stream, count, _tokens, _text, _segs = backend.process_docx(
        BytesIO(_docx_bytes(build)), target_language="Spanish", do_translate=True
    )

    assert seen == ["Important notice"], "the document path must have called the translator"
    result = _reopen(out_stream)
    para = result.paragraphs[0]
    assert para.text == _fake_translation("Important notice")
    runs = [r for r in para.runs if r.text]
    assert len(runs) == 1
    assert runs[0].bold is True, "bold was dropped by the rewrite"
    assert runs[0].italic is True, "italic was dropped by the rewrite"
    assert count == 1


def test_hyperlink_element_and_target_survive_translation(backend, monkeypatch):
    """w:hyperlink went 1 -> 0 before this fix: the link became plain text."""
    seen = _install_fake_translator(backend, monkeypatch)
    url = "https://example.com/pricing"

    def build(doc):
        para = doc.add_paragraph()
        para.add_run("See ")
        _add_hyperlink(para, url, "our pricing page")
        para.add_run(" for details.")

    source_bytes = _docx_bytes(build)
    assert _output_xml(BytesIO(source_bytes)).count("<w:hyperlink") == 1

    out_stream, _count, _tokens, _text, _segs = backend.process_docx(
        BytesIO(source_bytes), target_language="Spanish", do_translate=True
    )

    xml = _output_xml(out_stream)
    assert xml.count("<w:hyperlink") == 1, "the hyperlink element was destroyed"
    assert url in _output_rels(out_stream), "the hyperlink relationship target was lost"
    # The link text itself must be translated, not left in the source language.
    assert _fake_translation("our pricing page") in xml
    assert "our pricing page" in seen, "the link text was never sent for translation"


def test_hyperlink_paragraph_reports_that_it_was_split(backend, monkeypatch):
    """Honesty requirement: piecewise translation around a link is a real
    fidelity trade-off, so it is stated rather than hidden."""
    _install_fake_translator(backend, monkeypatch)
    from TranslationBackend import TranslationRunState

    state = TranslationRunState()

    def build(doc):
        para = doc.add_paragraph()
        para.add_run("See ")
        _add_hyperlink(para, "https://example.com", "the docs")
        para.add_run(" first.")

    backend.process_docx(
        BytesIO(_docx_bytes(build)),
        target_language="Spanish",
        do_translate=True,
        run_state=state,
    )

    messages = " ".join(n["message"] for n in state.fidelity_notes)
    assert state.fidelity_notes, "a link-split paragraph must produce a fidelity note"
    assert "hyperlink" in messages


def test_mixed_format_paragraph_keeps_first_run_and_says_what_was_lost(backend, monkeypatch):
    """Two differently formatted runs cannot map onto one translated string. We
    keep the first run's style and TELL the user, instead of silently flattening."""
    _install_fake_translator(backend, monkeypatch)
    from TranslationBackend import TranslationRunState

    state = TranslationRunState()

    def build(doc):
        para = doc.add_paragraph()
        bold = para.add_run("Warning: ")
        bold.bold = True
        para.add_run("do not unplug the device.")

    out_stream, _c, _t, _x, _s = backend.process_docx(
        BytesIO(_docx_bytes(build)),
        target_language="Spanish",
        do_translate=True,
        run_state=state,
    )

    para = _reopen(out_stream).paragraphs[0]
    assert para.text == _fake_translation("Warning: do not unplug the device.")
    assert [r for r in para.runs if r.text][0].bold is True
    assert any("collapsed" in n["message"] for n in state.fidelity_notes), (
        "lost inline formatting must be reported, not silently dropped"
    )


def test_table_cell_formatting_is_preserved_too(backend, monkeypatch):
    _install_fake_translator(backend, monkeypatch)

    def build(doc):
        table = doc.add_table(rows=1, cols=1)
        para = table.cell(0, 0).paragraphs[0]
        run = para.add_run("Total due")
        run.bold = True

    out_stream, _c, _t, _x, _s = backend.process_docx(
        BytesIO(_docx_bytes(build)), target_language="Spanish", do_translate=True
    )

    cell_para = _reopen(out_stream).tables[0].cell(0, 0).paragraphs[0]
    assert cell_para.text == _fake_translation("Total due")
    assert [r for r in cell_para.runs if r.text][0].bold is True


def test_editing_a_segment_also_preserves_formatting(backend, monkeypatch):
    """update_segment used to assign para.text as well, undoing the fix above."""
    _install_fake_translator(backend, monkeypatch)
    from TranslationBackend import TranslationRunState

    state = TranslationRunState()

    def build(doc):
        run = doc.add_paragraph().add_run("Quarterly results")
        run.bold = True

    backend.process_docx(
        BytesIO(_docx_bytes(build)),
        target_language="Spanish",
        do_translate=True,
        run_state=state,
    )
    seg_id = next(iter(state.segment_map))
    backend.update_segment(seg_id, "Resultados trimestrales", "Spanish", run_state=state)

    para = _reopen(state.output_stream).paragraphs[0]
    assert para.text == "Resultados trimestrales"
    assert [r for r in para.runs if r.text][0].bold is True


# ───────────────────────── D6: malformed upload messages ─────────────────────

def _message(backend, data: bytes, ext: str, name: str) -> str:
    with pytest.raises(ValueError) as excinfo:
        backend.translate_file(
            input_stream=BytesIO(data),
            file_extension=ext,
            target_language="Spanish",
            file_name=name,
        )
    return str(excinfo.value)


def test_empty_file_and_mislabelled_pdf_get_different_specific_messages(backend):
    empty_msg = _message(backend, b"", "docx", "empty.docx")
    pdf_bytes = BytesIO()
    pdf = fitz.open()
    pdf.new_page()
    pdf.save(pdf_bytes)
    renamed_msg = _message(backend, pdf_bytes.getvalue(), "docx", "wrong.docx")

    assert empty_msg != renamed_msg
    # Each names the file and the actual cause.
    assert "empty.docx" in empty_msg and "0 bytes" in empty_msg
    assert "wrong.docx" in renamed_msg and "PDF" in renamed_msg
    # And neither leaks the library's internals.
    for msg in (empty_msg, renamed_msg):
        assert "not a zip file" not in msg.lower()


def test_pptx_renamed_as_docx_names_the_real_format(backend):
    prs = Presentation()
    prs.slides.add_slide(prs.slide_layouts[6])
    buf = BytesIO()
    prs.save(buf)

    msg = _message(backend, buf.getvalue(), "docx", "deck.docx")
    assert "deck.docx" in msg
    assert ".pptx" in msg


def test_truncated_zip_is_reported_as_damaged_not_as_a_zip_error(backend):
    doc_bytes = _docx_bytes(lambda d: d.add_paragraph("Hello"))
    msg = _message(backend, doc_bytes[: len(doc_bytes) // 2], "docx", "half.docx")
    assert "half.docx" in msg
    assert "damaged" in msg.lower()


def test_valid_docx_still_passes_validation(backend, monkeypatch):
    """The guard must not reject good files — the sniff runs on the real path."""
    _install_fake_translator(backend, monkeypatch)
    doc_bytes = _docx_bytes(lambda d: d.add_paragraph("Hello"))
    out_stream, count, _tokens, _text, _segs = backend.translate_file(
        input_stream=BytesIO(doc_bytes),
        file_extension="docx",
        target_language="Spanish",
        file_name="good.docx",
    )
    assert count == 1
    assert _reopen(out_stream).paragraphs[0].text == _fake_translation("Hello")


# ──────────────────────────── D8: token accounting ───────────────────────────

def test_docx_tokens_cover_both_source_and_translation(backend, monkeypatch):
    _install_fake_translator(backend, monkeypatch)
    source = "The quarterly report is attached for your review."
    _out, _count, tokens, _text, _segs = backend.process_docx(
        BytesIO(_docx_bytes(lambda d: d.add_paragraph(source))),
        target_language="Spanish",
        do_translate=True,
    )

    source_only = backend.calculate_tokens(source)
    assert tokens > 0
    assert tokens > source_only, "the translated side must be counted too"
    assert tokens >= source_only + backend.calculate_tokens(_fake_translation(source)) - 2


def test_pptx_tokens_cover_both_sides(backend, monkeypatch):
    _install_fake_translator(backend, monkeypatch)
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    box = slide.shapes.add_textbox(left=1_000_000, top=1_000_000, width=4_000_000, height=1_000_000)
    box.text_frame.text = "Revenue grew across every region this quarter."
    buf = BytesIO()
    prs.save(buf)
    buf.seek(0)

    _out, count, tokens, _text, _segs = backend.process_pptx(
        buf, target_language="Spanish", do_translate=True
    )
    assert count == 1
    assert tokens > backend.calculate_tokens("Revenue grew across every region this quarter.")


def test_pdf_tokens_are_not_zero(backend, monkeypatch):
    """PDFs reported 0 tokens unconditionally — a confidently wrong number."""
    _install_fake_translator(backend, monkeypatch)
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "Annual maintenance is scheduled for October.")
    buf = BytesIO()
    pdf.save(buf)
    buf.seek(0)

    _out, count, tokens, _text, _segs = backend.process_pdf(
        buf, target_language="Spanish", do_translate=True
    )
    assert count >= 1
    assert tokens > 0, "PDF token count is still hardcoded to zero"
    assert tokens > backend.calculate_tokens("Annual maintenance is scheduled for October.")


# ───────────────────────── D4: Save All Edits actually saves ─────────────────

class _FakeTextarea:
    """Stands in for the NiceGUI textarea element only; the code under test
    (TranslationUI.save_all_edits) is the real one, on a real TranslationUI."""

    def __init__(self, value: str):
        self.value = value


def _prepared_ui(monkeypatch, tmp_path, notifications, *, text="Original sentence."):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        "TranslationUI.ui.notify",
        lambda message, **kwargs: notifications.append((message, kwargs.get("type"))),
    )
    monkeypatch.setattr(traces, "TRACE_DIR", tmp_path)

    ui_app = TranslationUI()
    _install_fake_translator(ui_app.backend, monkeypatch)
    ui_app.current_target_language = "Spanish"
    ui_app.uploaded_file_name = "report.docx"
    ui_app.uploaded_file_extension = "docx"
    ui_app.document_trace_id = "trace-under-test"

    ui_app.backend.translate_file(
        input_stream=BytesIO(_docx_bytes(lambda d: d.add_paragraph(text))),
        file_extension="docx",
        target_language="Spanish",
        run_state=ui_app.document_run_state,
        file_name="report.docx",
    )
    seg_map = ui_app.document_run_state.segment_map
    ui_app.original_segments_map = {k: v["original"] for k, v in seg_map.items()}
    ui_app.translated_segments_map = {k: v["translated"] for k, v in seg_map.items()}
    return ui_app, next(iter(seg_map))


def test_save_all_edits_puts_the_edit_in_the_downloaded_document(monkeypatch, tmp_path):
    notifications = []
    ui_app, seg_id = _prepared_ui(monkeypatch, tmp_path, notifications)
    machine_output = ui_app.translated_segments_map[seg_id]

    ui_app.segment_editors = {seg_id: _FakeTextarea("Frase corregida a mano.")}
    ui_app.save_all_edits()

    downloaded = ui_app.get_fresh_download_stream()
    text = Document(downloaded).paragraphs[0].text
    assert text == "Frase corregida a mano.", (
        f"the download still contains {text!r} instead of the edit"
    )
    assert machine_output not in text
    assert any(kind == "positive" for _msg, kind in notifications)


def test_save_all_edits_records_a_trace_score_row(monkeypatch, tmp_path):
    notifications = []
    ui_app, seg_id = _prepared_ui(monkeypatch, tmp_path, notifications)
    machine_output = ui_app.translated_segments_map[seg_id]

    ui_app.segment_editors = {seg_id: _FakeTextarea("Frase corregida a mano.")}
    ui_app.save_all_edits()

    rows = [
        json.loads(line)
        for path in sorted(tmp_path.glob("traces-*.jsonl"))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    scores = [r for r in rows if r.get("type") == "score"]
    assert len(scores) == 1, "the human edit never reached the ground-truth dataset"
    assert scores[0]["segment_id"] == seg_id
    assert scores[0]["before"] == machine_output
    assert scores[0]["after"] == "Frase corregida a mano."


def test_save_all_edits_does_not_claim_success_when_there_is_nothing_to_save(monkeypatch, tmp_path):
    """The dishonest part of D4 was the green toast for a no-op."""
    notifications = []
    ui_app, _seg_id = _prepared_ui(monkeypatch, tmp_path, notifications)
    ui_app.segment_editors = {}

    ui_app.save_all_edits()

    assert notifications, "the user must be told something happened"
    message, kind = notifications[-1]
    assert kind != "positive"
    assert "nothing to save" in message.lower()


def test_save_all_edits_leaves_untouched_segments_alone(monkeypatch, tmp_path):
    notifications = []
    ui_app, seg_id = _prepared_ui(monkeypatch, tmp_path, notifications)
    unchanged = ui_app.translated_segments_map[seg_id]

    ui_app.segment_editors = {seg_id: _FakeTextarea(unchanged)}
    ui_app.save_all_edits()

    assert Document(ui_app.get_fresh_download_stream()).paragraphs[0].text == unchanged
    assert not list(tmp_path.glob("traces-*.jsonl")), "a non-edit must not inflate the dataset"
    assert notifications[-1][1] != "positive"


def test_show_result_registers_every_editor_it_renders(monkeypatch, tmp_path):
    """save_all_edits is only correct if the registry is actually populated by
    the renderer; assert the wiring exists rather than assuming it."""
    import inspect
    import TranslationUI as ui_module

    source = inspect.getsource(ui_module.TranslationUI.show_result)
    assert "self.segment_editors[seg_id] = textarea" in source
    assert "self.segment_editors.clear()" in source
