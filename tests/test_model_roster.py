"""The model roster: GPT-5-family models need max_completion_tokens (not
max_tokens) and get reasoning_effort pinned to "none" for latency; legacy
models keep max_tokens so PASSAGE_TEXT_MODEL can roll back without code."""
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import TranslationBackend as tb


def test_gpt5_family_uses_max_completion_tokens_and_no_reasoning():
    assert tb._completion_limit_kwargs("gpt-5.4-nano", 4000) == {
        "max_completion_tokens": 4000,
        "reasoning_effort": "none",
    }


def test_legacy_models_keep_max_tokens():
    assert tb._completion_limit_kwargs("gpt-4.1-nano", 4000) == {"max_tokens": 4000}


def _capturing_provider():
    provider = tb.OpenAITranslationProvider(api_key="test-key")
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])

    provider.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
    )
    return provider, captured


def test_provider_sends_configured_text_model_with_per_family_kwargs():
    provider, captured = _capturing_provider()

    provider.create_chat_completion(messages=[{"role": "user", "content": "hi"}], max_tokens=4000)

    assert captured["model"] == tb.TEXT_MODEL
    expected = tb._completion_limit_kwargs(tb.TEXT_MODEL, 4000)
    for key, value in expected.items():
        assert captured[key] == value
    assert not ("max_tokens" in captured and "max_completion_tokens" in captured)


def test_calculate_tokens_survives_model_names_unknown_to_tiktoken(monkeypatch):
    monkeypatch.setattr(tb, "TEXT_MODEL", "gpt-999-experimental")
    backend = tb.TranslationBackend()

    assert backend.calculate_tokens("hello world") > 0
