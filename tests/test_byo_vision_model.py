"""The image (OCR) call names the model of the endpoint it is sent to.

translate_image_text_blocks resolved its provider through the profile override
but then called ``client.chat.completions.create(model=VISION_MODEL)``
directly, so a BYO session's photo went to the user's own base_url asking for
Passage's hosted model name ("gpt-5.4-mini"). These tests build providers
through the production path (using_profile -> provider_for_profile ->
ChatCompletionsProvider -> openai.OpenAI) with only ``openai.OpenAI`` faked.
Each fake endpoint records the model it was asked for and stamps its identity
into what it returns, so a test fails if the wrong endpoint served the photo
and fails if nothing ran.
"""
import json
import sys
import types
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import openai

import TranslationBackend as tb
from TranslationBackend import VISION_MODEL, TranslationBackend
from passage import provider_profiles as pp

BYO_URL = "http://byo.example/v1"
BYO_MODEL = "llava:13b"


class _Endpoints:
    def __init__(self) -> None:
        # base_url -> list of (kind, model) for every request it received.
        self.requests: dict[object, list[tuple[str, str]]] = {}

    def vision_models(self, base_url) -> list[str]:
        return [m for kind, m in self.requests.get(base_url, []) if kind == "vision"]


def _is_vision(messages) -> bool:
    return any(
        isinstance(m.get("content"), list)
        and any(part.get("type") == "image_url" for part in m["content"])
        for m in messages
    )


class _FakeCompletions:
    def __init__(self, endpoints: _Endpoints, base_url) -> None:
        self._endpoints = endpoints
        self._base_url = base_url

    def create(self, *, model, messages, **_kwargs):
        who = "HOSTED" if self._base_url is None else "BYO"
        kind = "vision" if _is_vision(messages) else "text"
        self._endpoints.requests.setdefault(self._base_url, []).append((kind, model))
        if kind == "vision":
            content = json.dumps({"recognized_blocks": [
                {"text": f"read-by-{who}", "confidence": 0.9, "bbox": [0.1, 0.1, 0.9, 0.5]},
            ]})
        else:
            content = json.dumps({"translations": [{"i": 0, "text": f"translated-by-{who}"}]})
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=content))]
        )


class _FakeOpenAI:
    endpoints: _Endpoints

    def __init__(self, **kwargs):
        self.api_key = kwargs.get("api_key")
        self.base_url = kwargs.get("base_url")
        self.chat = types.SimpleNamespace(
            completions=_FakeCompletions(type(self).endpoints, self.base_url)
        )


@pytest.fixture
def endpoints(monkeypatch):
    registry = _Endpoints()
    monkeypatch.setattr(_FakeOpenAI, "endpoints", registry, raising=False)
    monkeypatch.setattr(openai, "OpenAI", _FakeOpenAI)
    return registry


@pytest.fixture
def backend(monkeypatch, endpoints):
    monkeypatch.setenv("OPENAI_API_KEY", "hosted-key")
    monkeypatch.setenv("TRANSLATION_PROVIDER", "openai")
    return TranslationBackend()


def _png() -> bytes:
    buf = BytesIO()
    Image.new("RGB", (200, 100), (255, 255, 255)).save(buf, format="PNG")
    return buf.getvalue()


def test_byo_photo_asks_the_byo_endpoint_for_the_profile_model(backend, endpoints):
    profile = pp.ProviderProfile(
        label="BYO", kind=pp.KIND_BYO, base_url=BYO_URL, api_key="secret", model=BYO_MODEL,
    )
    with backend.using_profile(profile):
        result = backend.translate_image_text_blocks(_png(), "menu.png", "Spanish")

    assert endpoints.vision_models(BYO_URL) == [BYO_MODEL]
    assert VISION_MODEL not in [m for _k, m in endpoints.requests[BYO_URL]]
    assert None not in endpoints.requests            # hosted never saw the photo
    # Which path ran: both the OCR and the translation came from the BYO endpoint.
    block = result["translated_blocks"][0]
    assert block["source_text"] == "read-by-BYO"
    assert block["translated_text"] == "translated-by-BYO"


def test_hosted_photo_still_asks_for_vision_model(backend, endpoints):
    result = backend.translate_image_text_blocks(_png(), "menu.png", "Spanish")

    assert endpoints.vision_models(None) == [VISION_MODEL]
    assert BYO_URL not in endpoints.requests
    assert result["translated_blocks"][0]["source_text"] == "read-by-HOSTED"


def test_endpoint_without_an_explicit_vision_model_uses_its_own_model():
    """A local/Ollama provider built without a profile must not inherit the
    hosted constant either; it names the model it was configured with."""
    provider = tb.ChatCompletionsProvider(
        api_key="ollama", base_url="http://localhost:11434/v1", text_model="gemma3:4b",
    )
    assert provider.vision_model == "gemma3:4b"
    hosted = tb.ChatCompletionsProvider(api_key="k")
    assert hosted.vision_model == VISION_MODEL
