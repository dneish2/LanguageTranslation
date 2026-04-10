import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TranslationBackend import TranslationBackend


class _StubChatCreate:
    def __init__(self, responses=None, errors=None):
        self.calls = 0
        self._responses = list(responses or [])
        self._errors = list(errors or [])

    def __call__(self, **_kwargs):
        self.calls += 1
        if self._errors:
            err = self._errors.pop(0)
            if err is not None:
                raise err
        content = self._responses.pop(0) if self._responses else ""
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=content))]
        )


def _build_backend(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    return TranslationBackend()


def test_translate_text_cache_hit_uses_normalized_key(monkeypatch):
    backend = _build_backend(monkeypatch)
    stub = _StubChatCreate(responses=["Hola mundo"])
    monkeypatch.setattr(backend.client.chat.completions, "create", stub)

    first = backend.translate_text("\t Hello   world  ", " Spanish ")
    second = backend.translate_text("Hello world", "spanish")

    assert first == "Hola mundo"
    assert second == "Hola mundo"
    assert stub.calls == 1


def test_instruction_translation_cache_hit_uses_mode_and_instructions(monkeypatch):
    backend = _build_backend(monkeypatch)
    stub = _StubChatCreate(responses=["Veuillez reformuler"])
    monkeypatch.setattr(backend.client.chat.completions, "create", stub)

    first = backend.translate_text_with_instructions(
        "  Please rephrase this ", "French", " Use formal tone "
    )
    second = backend.translate_text_with_instructions(
        "Please rephrase this", " french ", "use   formal tone"
    )

    assert first == "Veuillez reformuler"
    assert second == "Veuillez reformuler"
    assert stub.calls == 1


def test_translate_text_retries_on_transient_openai_error(monkeypatch):
    backend = _build_backend(monkeypatch)
    backend.retry_base_delay = 0.01
    backend.retry_max_delay = 0.02

    transient = Exception("temporary outage")
    transient.status_code = 503
    stub = _StubChatCreate(errors=[transient, None], responses=["Recovered translation"])
    monkeypatch.setattr(backend.client.chat.completions, "create", stub)

    slept = []
    monkeypatch.setattr("TranslationBackend.time.sleep", lambda seconds: slept.append(seconds))

    translated = backend.translate_text("Resilient text", "German")

    assert translated == "Recovered translation"
    assert stub.calls == 2
    assert len(slept) == 1
    assert slept[0] >= backend.retry_base_delay


def test_translate_text_stops_after_max_attempts(monkeypatch):
    backend = _build_backend(monkeypatch)
    backend.max_openai_attempts = 3
    backend.retry_base_delay = 0
    backend.retry_max_delay = 0

    transient = Exception("still failing")
    transient.status_code = 503
    stub = _StubChatCreate(errors=[transient, transient, transient])
    monkeypatch.setattr(backend.client.chat.completions, "create", stub)
    monkeypatch.setattr("TranslationBackend.time.sleep", lambda _seconds: None)

    original = "Fallback text"
    translated = backend.translate_text(original, "Italian")

    assert translated == original
    assert stub.calls == backend.max_openai_attempts
