import json
from io import BytesIO
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TranslationUI import TranslationUI


class _Msg:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Msg(content)


class _Resp:
    def __init__(self, content):
        self.choices = [_Choice(content)]


class _Completions:
    def __init__(self, content):
        self._content = content

    def create(self, **_kwargs):
        return _Resp(self._content)


class _Chat:
    def __init__(self, content):
        self.completions = _Completions(content)


class _Client:
    def __init__(self, content):
        self.chat = _Chat(content)


class _Provider:
    def __init__(self, content):
        self.client = _Client(content)


class _Upload:
    def __init__(self, payload: bytes, name: str):
        self._payload = payload
        self.filename = name

    async def read(self):
        return self._payload


def test_backend_image_ocr_translation_success(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    ui_app = TranslationUI()
    backend = ui_app.backend
    backend.provider = _Provider(json.dumps({"recognized_blocks": [{"text": "hello", "confidence": 0.91}]}))
    monkeypatch.setattr(backend, "translate_text", lambda text, lang: f"{lang}:{text}")

    result = backend.translate_image_text_blocks(b"abc", "sample.png", "Spanish")

    assert result["translated_blocks"][0]["translated_text"] == "Spanish:hello"
    assert result["confidence_metadata"]["block_count"] == 1


def test_backend_image_ocr_unsupported_format_error():
    import os
    os.environ["OPENAI_API_KEY"] = "test-key"
    ui_app = TranslationUI()
    backend = ui_app.backend

    try:
        backend.translate_image_text_blocks(b"abc", "sample.gif", "Spanish")
    except ValueError as err:
        assert "Unsupported image format" in str(err)
    else:
        raise AssertionError("Expected ValueError for unsupported format")


async def _call_api_image_translate(ui_app):
    import types
    request = types.SimpleNamespace(
        headers={"x-passage-token": ui_app.api_guard.issue_token()},
        client=types.SimpleNamespace(host="127.0.0.1"),
    )
    upload = _Upload(b"abc", "sample.gif")
    return await ui_app.api_image_translate(request, upload, "Spanish")


def test_api_image_translate_error_path_for_unsupported_format():
    import asyncio
    import os
    os.environ["OPENAI_API_KEY"] = "test-key"

    ui_app = TranslationUI()
    response = asyncio.run(_call_api_image_translate(ui_app))

    assert response.status_code == 400
    assert b"Unsupported image format" in response.body
