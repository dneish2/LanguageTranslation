import sys
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TranslationBackend import TranslationBackend


def _sample_image(kind: str) -> bytes:
    img = Image.new("RGB", (500, 300), "white")
    d = ImageDraw.Draw(img)
    text = {
        "menu": "Soup 10$\nSalad 8$",
        "photo": "Street sign",
        "doc": "Quarterly Report 2026",
    }[kind]
    d.text((40, 40), text, fill="black")
    out = BytesIO(); img.save(out, format="PNG"); out.seek(0)
    return out.getvalue()


def test_image_overlay_visual_regression_common_cases(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    backend = TranslationBackend()

    monkeypatch.setattr(
        backend,
        "extract_image_text_regions",
        lambda b: [{"bbox": [35, 35, 300, 140], "text": "sample text", "direction": "ltr"}],
    )
    monkeypatch.setattr(
        backend,
        "translate_text",
        lambda text, target_language, correlation_id=None, file_metrics=None: f"{target_language}:{text}",
    )
    monkeypatch.setattr(backend, "calculate_tokens", lambda _text: 0)

    for kind in ["menu", "photo", "doc"]:
        out_stream, count, *_ = backend.process_image(BytesIO(_sample_image(kind)), "Spanish")
        out = out_stream.getvalue()
        assert count == 1
        assert len(out) > 1000
        assert out[:8] == b"\x89PNG\r\n\x1a\n"
