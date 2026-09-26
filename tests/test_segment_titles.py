"""Segment review titles name a place in the user's document, not a machine id.

The segment ids (``docx:paragraph:0``, ``pptx:slide:0:shape:2``) are good keys
and bad labels, and two surfaces printed them verbatim: the review expansion
titles and the formatting-notes list. The tests here render those two surfaces
and read what a person would see, rather than asserting that a helper returned a
string — the helper was never the part that was wrong.

``test_every_docx_location_written_gets_a_human_label`` is the anti-drift half:
it walks the segment map a real DOCX produces, so a fifth location grammar added
to TranslationBackend with no label fails here instead of shipping an id.
"""
import sys
import time
from io import BytesIO
from pathlib import Path

import pytest
from docx import Document
from pptx import Presentation
from pptx.util import Inches

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TranslationBackend import TranslationBackend, describe_location  # noqa: E402
from TranslationUI import TranslationUI  # noqa: E402


# ──────────────────────────── fake NiceGUI surface ────────────────────────────
# NiceGUI elements need a client context, which pytest has no way to provide, so
# the element factories are replaced and every string handed to one is recorded.
# The code under test (show_result, _render_fidelity_notes) is the real one.

class _FakeElement:
    def __init__(self, text: str = ""):
        self.text = text
        self.value = text

    def __getattr__(self, _name):
        # .classes(...).props(...).tooltip(...) all chain and all return self.
        return lambda *args, **kwargs: self

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


class _FakeUI:
    """Records (factory name, first string argument) for everything rendered."""

    def __init__(self):
        self.rendered: list[tuple[str, str]] = []

    def __getattr__(self, name):
        def factory(*args, **kwargs):
            if args and isinstance(args[0], str):
                self.rendered.append((name, args[0]))
            elif "label" in kwargs and isinstance(kwargs["label"], str):
                self.rendered.append((name, kwargs["label"]))
            return _FakeElement(args[0] if args and isinstance(args[0], str) else "")

        return factory

    def notify(self, *_args, **_kwargs):
        return None

    # What the surface renders, by kind and as one blob of text.
    def titles(self) -> list[str]:
        return [text for kind, text in self.rendered if kind == "expansion"]

    def text(self) -> str:
        return "\n".join(text for _kind, text in self.rendered)


class _FakeContainer:
    def clear(self):
        return None

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


def _fake_translation(text: str) -> str:
    return f"ES::{text}"


def _docx_bytes(build) -> bytes:
    doc = Document()
    build(doc)
    buf = BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf.getvalue()


def _pptx_bytes(build) -> bytes:
    prs = Presentation()
    build(prs)
    buf = BytesIO()
    prs.save(buf)
    buf.seek(0)
    return buf.getvalue()


def _rendered_ui(
    monkeypatch, *, file_bytes: bytes, extension: str, render_as: str | None = None
) -> _FakeUI:
    """Translate a real file, then render the real document result surface."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    ui_app = TranslationUI()
    monkeypatch.setattr(
        ui_app.backend,
        "translate_text",
        lambda text, target_language, correlation_id=None, file_metrics=None: _fake_translation(text),
    )
    ui_app.current_target_language = "Spanish"
    ui_app.uploaded_file_name = f"notes.{extension}"
    ui_app.uploaded_file_extension = extension

    ui_app.backend.translate_file(
        input_stream=BytesIO(file_bytes),
        file_extension=extension,
        target_language="Spanish",
        run_state=ui_app.document_run_state,
        file_name=f"notes.{extension}",
    )
    seg_map = ui_app.document_run_state.segment_map
    assert seg_map, "nothing to review means this test proves nothing"
    ui_app.original_segments_map = {k: v["original"] for k, v in seg_map.items()}
    ui_app.translated_segments_map = {k: v["translated"] for k, v in seg_map.items()}

    if render_as is not None:
        ui_app.uploaded_file_extension = render_as

    fake = _FakeUI()
    monkeypatch.setattr("TranslationUI.ui", fake)
    ui_app.progress_container = _FakeContainer()
    ui_app.result_container = _FakeContainer()
    ui_app.stats_container = _FakeContainer()
    ui_app.show_result()
    return fake


# ──────────────────────────── the id -> label grammar ────────────────────────

@pytest.mark.parametrize("location,expected", [
    ("docx:paragraph:0", "Paragraph 1"),
    ("docx:paragraph:11", "Paragraph 12"),
    ("docx:table:0:row:1:col:2:para:0", "Table 1, row 2, cell 3"),
    # A second paragraph inside the same cell has to be distinguishable.
    ("docx:table:0:row:1:col:2:para:1", "Table 1, row 2, cell 3, paragraph 2"),
    ("pptx:slide:0:shape:2", "Slide 1, text box 3"),
    ("image:region:4", "Image region 5"),
])
def test_describe_location_renders_each_id_shape(location, expected):
    assert describe_location(location) == expected


@pytest.mark.parametrize("location", ["", "docx:paragraph:", "xlsx:sheet:0:cell:A1", "whatever"])
def test_unrecognised_locations_come_back_unchanged(location):
    """An ugly true label beats a friendly wrong one — and beats a crash."""
    assert describe_location(location) == location


# ──────────────────────────── the review surface ─────────────────────────────

def test_docx_review_titles_name_paragraphs_and_table_cells(monkeypatch):
    def build(doc):
        doc.add_paragraph("Attendees: the whole team.")
        doc.add_paragraph("Notes live at https://example.com/notes")
        table = doc.add_table(rows=2, cols=2)
        table.cell(0, 0).text = "Owner"
        table.cell(1, 1).text = "Ship the release"

    fake = _rendered_ui(monkeypatch, file_bytes=_docx_bytes(build), extension="docx")

    assert fake.titles() == [
        "1. Paragraph 1",
        "2. Paragraph 2",
        "3. Table 1, row 1, cell 1",
        "4. Table 1, row 2, cell 2",
    ]


def test_every_docx_location_written_gets_a_human_label(monkeypatch):
    """Anti-drift: every location the backend writes must have a label here."""
    def build(doc):
        doc.add_paragraph("One.")
        table = doc.add_table(rows=1, cols=1)
        table.cell(0, 0).text = "Two."

    fake = _rendered_ui(monkeypatch, file_bytes=_docx_bytes(build), extension="docx")

    for title in fake.titles():
        assert ":" not in title, f"raw segment id leaked into a review title: {title!r}"
        assert "docx" not in title.lower()


def test_pptx_review_titles_name_the_slide_and_text_box(monkeypatch):
    def build(prs):
        blank = prs.slide_layouts[6]
        for text in ("Quarterly review", "Two open risks"):
            slide = prs.slides.add_slide(blank)
            box = slide.shapes.add_textbox(Inches(1), Inches(1), Inches(4), Inches(1))
            box.text_frame.text = text

    fake = _rendered_ui(monkeypatch, file_bytes=_pptx_bytes(build), extension="pptx")

    assert fake.titles() == ["1. Slide 1, text box 1", "2. Slide 2, text box 1"]


@pytest.mark.parametrize("render_as", [None, "png"])
def test_document_results_carry_no_image_only_font_controls(monkeypatch, render_as):
    """show_result() is the DOCUMENT surface. It used to render Font size and
    Font family inputs behind ``extension in {png, jpg, jpeg, webp}`` — a
    condition no document upload can meet, since the uploader only accepts what
    SUPPORTED_DOCUMENT_EXTENSIONS lists. The ``png`` case forces that branch's
    condition true anyway, so this fails on the old code rather than passing for
    the incidental reason that documents never reached it.
    """
    fake = _rendered_ui(
        monkeypatch,
        file_bytes=_docx_bytes(lambda doc: doc.add_paragraph("Body text.")),
        extension="docx",
        render_as=render_as,
    )

    surface = fake.text()
    assert "Font size" not in surface
    assert "Font family" not in surface
    assert "Image overlay controls" not in surface
    assert "Download Translated File" in surface, "the surface itself must still render"


def test_pdf_blocks_are_described_by_page(monkeypatch):
    """PDF segments carry no location at all, so the page is the only label."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    ui_app = TranslationUI()
    title = ui_app._describe_segment_for_editor(
        1, {"type": "pdf_block", "page_idx": 2, "original": "x"}
    )
    assert title == "1. PDF page 3"


def test_a_segment_with_no_location_or_page_still_gets_a_title(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    ui_app = TranslationUI()
    assert ui_app._describe_segment_for_editor(7, {"type": "table_cell"}) == "7. Table Cell"


# ──────────────────────────── the progress ring ──────────────────────────────

def test_progress_percent_is_a_whole_number(monkeypatch):
    """ui.circular_progress(show_value=True) prints what it is handed, and two
    of eleven segments put "18.18181818181818183" inside the ring."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    backend = TranslationBackend()
    seen = []
    for current in range(0, 12):
        backend.update_progress(
            current, 11, time.time(),
            progress_callback=lambda value, _text: seen.append(value),
        )
    assert seen == [0, 9, 18, 27, 36, 45, 55, 64, 73, 82, 91, 100]
    assert all(isinstance(v, int) for v in seen)


def test_progress_does_not_divide_by_zero_on_an_empty_document(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    seen = []
    TranslationBackend().update_progress(
        0, 0, time.time(), progress_callback=lambda value, _text: seen.append(value)
    )
    assert seen == [0]


# ──────────────────────────── the formatting notes ───────────────────────────

def test_formatting_notes_show_labels_not_ids(monkeypatch):
    """The notes list sits inches under the review titles; a raw id there undoes
    the fix. The note itself is produced by a real hyperlink paragraph."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    from docx.opc.constants import RELATIONSHIP_TYPE as RT
    from docx.oxml.ns import qn
    from docx.oxml.shared import OxmlElement

    def build(doc):
        para = doc.add_paragraph()
        para.add_run("Notes are at ")
        r_id = para.part.relate_to("https://example.com/notes", RT.HYPERLINK, is_external=True)
        link = OxmlElement("w:hyperlink")
        link.set(qn("r:id"), r_id)
        run = OxmlElement("w:r")
        text = OxmlElement("w:t")
        text.text = "this link"
        run.append(text)
        link.append(run)
        para._p.append(link)

    ui_app = TranslationUI()
    monkeypatch.setattr(
        ui_app.backend,
        "translate_text",
        lambda text, target_language, correlation_id=None, file_metrics=None: _fake_translation(text),
    )
    ui_app.backend.translate_file(
        input_stream=BytesIO(_docx_bytes(build)),
        file_extension="docx",
        target_language="Spanish",
        run_state=ui_app.document_run_state,
        file_name="notes.docx",
    )
    notes = ui_app.document_run_state.fidelity_notes
    assert notes, "a hyperlink paragraph must report that it was split"
    assert notes[0]["location"] == "docx:paragraph:0", "the stored id stays a stable key"

    fake = _FakeUI()
    monkeypatch.setattr("TranslationUI.ui", fake)
    ui_app._render_fidelity_notes()

    surface = fake.text()
    assert "docx:paragraph:0" not in surface
    assert "Paragraph 1" in surface
