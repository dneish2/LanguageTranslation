import os
import sys
from io import BytesIO
from pathlib import Path

import pytest
from docx import Document

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


def _docx_bytes(*paragraphs: str) -> bytes:
    doc = Document()
    for text in paragraphs:
        doc.add_paragraph(text)

    out = BytesIO()
    doc.save(out)
    out.seek(0)
    return out.getvalue()


def test_cancel_halts_processing_progression(monkeypatch):
    """User cancel should stop further segment processing."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    backend = TranslationBackend()
    progress = DummyProgress()
    label = DummyLabel()

    calls = {"count": 0}

    def translate_and_cancel(text, target_language):
        calls["count"] += 1
        backend.request_cancel()
        return f"translated:{text}"

    monkeypatch.setattr(backend, "translate_text", translate_and_cancel)
    monkeypatch.setattr(backend, "calculate_tokens", lambda _text: 0)

    docx_stream = BytesIO(_docx_bytes("First", "Second", "Third"))

    _out, processed, _tokens, _text, segment_map = backend.process_docx(
        docx_stream,
        target_language="Spanish",
        progress_ui=progress,
        label_ui=label,
        do_translate=True,
    )

    assert calls["count"] == 1
    assert processed == 1
    assert len(segment_map) == 1
    assert progress.value < 100


def test_unsupported_extension_fails_clearly(monkeypatch):
    """User gets an explicit error when file format is unsupported."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    backend = TranslationBackend()

    with pytest.raises(ValueError, match=r"Unsupported file extension: xlsx"):
        backend.translate_file(
            input_stream=BytesIO(b"unused"),
            file_extension="xlsx",
            target_language="French",
            progress_ui=DummyProgress(),
            label_ui=DummyLabel(),
        )


def test_segment_update_modifies_regenerated_output(monkeypatch):
    """User edits should appear in the regenerated downloadable file."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    backend = TranslationBackend()
    monkeypatch.setattr(backend, "calculate_tokens", lambda _text: 0)

    source_stream = BytesIO(_docx_bytes("Original paragraph"))
    backend.process_docx(
        source_stream,
        target_language="Spanish",
        progress_ui=DummyProgress(),
        label_ui=DummyLabel(),
        do_translate=False,
    )

    seg_id, segment = next(iter(backend.segment_map.items()))
    assert segment["type"] == "paragraph"

    backend.update_segment(seg_id, "Contenido actualizado", target_language="Spanish")

    regenerated = Document(BytesIO(backend.output_stream.getvalue()))
    assert regenerated.paragraphs[0].text == "Contenido actualizado"


def test_translate_fallback_returns_original_on_openai_exception(monkeypatch):
    """If OpenAI fails, user still receives their original text."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    backend = TranslationBackend()

    def raise_openai_error(**_kwargs):
        raise RuntimeError("simulated OpenAI outage")

    monkeypatch.setattr(backend.client.chat.completions, "create", raise_openai_error)

    original = "\tKeep me as-is\t"
    translated = backend.translate_text(original, target_language="German")

    assert translated == "Keep me as-is"
