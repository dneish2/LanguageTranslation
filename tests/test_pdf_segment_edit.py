import os
import sys
from io import BytesIO
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TranslationBackend import TranslationBackend


class DummyProgress:
    def __init__(self) -> None:
        self.value = 0

    def set_value(self, value):
        self.value = value


class DummyLabel:
    def __init__(self) -> None:
        self.text = ""


def _create_pdf_bytes(text: str) -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer.getvalue()


def test_update_pdf_segment_renders_new_text(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    backend = TranslationBackend()
    monkeypatch.setattr(backend, "calculate_tokens", lambda _text: 0)

    pdf_bytes = _create_pdf_bytes("Original block text")
    progress = DummyProgress()
    label = DummyLabel()

    backend.process_pdf(pdf_bytes, target_language="Spanish", progress_ui=progress, label_ui=label, do_translate=False)

    assert backend.segment_map, "Expected at least one PDF segment to be registered."

    seg_id, seg_data = next(iter(backend.segment_map.items()))
    assert seg_data["type"] == "pdf_block"

    updated_text = "Texto actualizado"
    backend.update_segment(seg_id, updated_text, target_language="Spanish")

    output_stream = backend.output_stream
    assert output_stream is not None

    doc = fitz.open(stream=output_stream.getvalue(), filetype="pdf")
    page_text = doc[0].get_text()
    assert updated_text in page_text


def test_batch_updates_skip_regeneration_until_requested(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    backend = TranslationBackend()
    monkeypatch.setattr(backend, "calculate_tokens", lambda _text: 0)

    pdf_bytes = _create_pdf_bytes("Original block text")
    progress = DummyProgress()
    label = DummyLabel()

    backend.process_pdf(pdf_bytes, target_language="Spanish", progress_ui=progress, label_ui=label, do_translate=False)

    seg_id, _ = next(iter(backend.segment_map.items()))

    original_stream_bytes = backend.output_stream.getvalue()

    backend.update_segment(seg_id, "Texto actualizado", target_language="Spanish", regenerate=False)

    assert backend.output_stream.getvalue() == original_stream_bytes

    backend.regenerate_output_stream()
    new_stream_bytes = backend.output_stream.getvalue()

    assert new_stream_bytes != original_stream_bytes

    doc = fitz.open(stream=new_stream_bytes, filetype="pdf")
    page_text = doc[0].get_text()
    assert "Texto actualizado" in page_text
