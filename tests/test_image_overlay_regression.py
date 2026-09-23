"""process_image and translate_file on photos.

process_image used to OCR through a stub that returned one hardcoded region
with EMPTY text, so every region was skipped and the "translated" overlay was
the original photo re-encoded. The old test here monkeypatched that stub with
real-looking text, which is exactly why it never noticed: it tested the stub's
replacement, not the stub. These tests fake only the vision ENDPOINT and assert
that it was called and that what it read is what ended up translated.
"""
import json
import sys
import types
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TranslationBackend import TranslationBackend, TranslationRunState


def _sample_image(kind: str, fmt: str = "PNG") -> bytes:
    img = Image.new("RGB", (500, 300), "white")
    d = ImageDraw.Draw(img)
    text = {
        "menu": "Soup 10$\nSalad 8$",
        "photo": "Street sign",
        "doc": "Quarterly Report 2026",
    }[kind]
    d.text((40, 40), text, fill="black")
    out = BytesIO(); img.save(out, format=fmt); out.seek(0)
    return out.getvalue()


class _VisionEndpoint:
    """Counts its calls and stamps its identity into the OCR text it returns."""

    def __init__(self) -> None:
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        content = json.dumps({"recognized_blocks": [
            {"text": f"VISION-READ-{self.calls}", "confidence": 0.9, "bbox": [0.1, 0.1, 0.6, 0.3]},
        ]})
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=content))]
        )


@pytest.fixture
def backend_and_endpoint(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    backend = TranslationBackend()
    endpoint = _VisionEndpoint()
    backend.provider = types.SimpleNamespace(
        client=types.SimpleNamespace(chat=types.SimpleNamespace(completions=endpoint))
    )

    def stamp(blocks, target_language):
        for block in blocks:
            block["translated_text"] = f"{target_language}:{block['source_text']}"

    monkeypatch.setattr(backend, "_translate_blocks_together", stamp)
    monkeypatch.setattr(backend, "calculate_tokens", lambda _text: 0)
    return backend, endpoint


def test_process_image_translates_what_the_vision_endpoint_read(backend_and_endpoint):
    backend, endpoint = backend_and_endpoint
    for n, (kind, fmt) in enumerate([("menu", "PNG"), ("photo", "JPEG"), ("doc", "WEBP")], start=1):
        state = TranslationRunState()
        source = _sample_image(kind, fmt)
        out_stream, count, *_ , seg_map = backend.process_image(BytesIO(source), "Spanish", run_state=state)

        assert endpoint.calls == n          # the real OCR path ran, once per photo
        assert count == 1
        (segment,) = seg_map.values()
        assert segment["original"] == f"VISION-READ-{n}"
        assert segment["translated"] == f"Spanish:VISION-READ-{n}"
        assert segment["bbox"] is not None

        out = out_stream.getvalue()
        assert out[:8] == b"\x89PNG\r\n\x1a\n"
        assert state.output_stream is out_stream
        assert state.current_file_type == "image"


def test_process_image_refuses_non_image_bytes_before_calling_vision(backend_and_endpoint):
    backend, endpoint = backend_and_endpoint
    with pytest.raises(ValueError, match="Unsupported image format"):
        backend.process_image(BytesIO(b"%PDF-1.7 not a photo"), "Spanish", run_state=TranslationRunState())
    assert endpoint.calls == 0


@pytest.mark.parametrize("ext", ["png", "jpg", "jpeg", "webp"])
def test_translate_file_refuses_images_and_names_the_photo_path(backend_and_endpoint, ext):
    backend, endpoint = backend_and_endpoint
    with pytest.raises(ValueError, match="translate_image_text_blocks"):
        backend.translate_file(BytesIO(_sample_image("menu")), ext, "Spanish")
    assert endpoint.calls == 0
