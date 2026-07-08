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


def _wav_bytes(rate=24000, channels=1, frames=b"\x01\x00\x02\x00\x03\x00\x04\x00"):
    import wave
    from io import BytesIO

    buf = BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(frames)
    return buf.getvalue()


def test_read_pcm16_wav_parses_mono_and_downmixes_stereo():
    pcm, rate = tb._read_pcm16_wav(_wav_bytes())
    assert (pcm, rate) == (b"\x01\x00\x02\x00\x03\x00\x04\x00", 24000)

    stereo_pcm, _ = tb._read_pcm16_wav(_wav_bytes(channels=2))
    assert stereo_pcm == b"\x01\x00\x03\x00"  # left channel only

    assert tb._read_pcm16_wav(b"\x1aE\xdf\xa3 not a wav") is None


def test_guess_audio_filename_by_magic_bytes():
    assert tb._guess_audio_filename(_wav_bytes()) == "speech.wav"
    assert tb._guess_audio_filename(b"\x1aE\xdf\xa3...") == "speech.webm"
    assert tb._guess_audio_filename(b"ID3\x04...") == "speech.mp3"
    assert tb._guess_audio_filename(b"\x00\x00\x00 ftypisom") == "speech.mp4"


def test_non_wav_audio_transcribes_via_rest_fallback(monkeypatch):
    """gpt-realtime-* models take PCM only; other payloads must hit the REST
    endpoint with the fallback model, never the websocket."""
    from io import BytesIO

    monkeypatch.setattr(tb, "TRANSCRIBE_MODEL", "gpt-realtime-whisper")
    provider = tb.OpenAITranslationProvider(api_key="test-key")
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(text="hello")

    provider.client = SimpleNamespace(
        audio=SimpleNamespace(transcriptions=SimpleNamespace(create=create)),
        realtime=None,  # touching the websocket would blow up
    )
    clip = BytesIO(b"\x1aE\xdf\xa3 fake webm")
    clip.name = "speech.webm"

    assert provider.transcribe_audio(audio_file=clip) == "hello"
    assert captured["model"] == tb.TRANSCRIBE_REST_MODEL


def test_build_translation_provider_ollama_targets_local_base_url_and_model():
    provider = tb.build_translation_provider("ollama", api_key="")

    assert provider.is_openai_hosted is False
    assert provider.base_url == tb.OLLAMA_BASE_URL
    assert provider.text_model == tb.OLLAMA_MODEL


def test_build_translation_provider_openai_is_hosted_with_configured_text_model():
    provider = tb.build_translation_provider("openai", api_key="test-key")

    assert provider.is_openai_hosted is True
    assert provider.text_model == tb.TEXT_MODEL


def test_build_translation_provider_rejects_unknown_name():
    try:
        tb.build_translation_provider("made-up-provider", api_key="x")
        assert False, "expected ValueError"
    except ValueError as e:
        assert "made-up-provider" in str(e)


def test_non_openai_provider_uses_plain_max_tokens_not_gpt5_kwargs():
    provider, captured = _capturing_provider()
    provider.base_url = "http://localhost:11434/v1"
    provider.is_openai_hosted = False
    provider.text_model = "gemma3:1b"

    provider.create_chat_completion(messages=[{"role": "user", "content": "hi"}], max_tokens=4000)

    assert captured == {"model": "gemma3:1b", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4000}


def test_non_openai_provider_refuses_voice_capabilities_with_a_clear_error():
    provider = tb.ChatCompletionsProvider(api_key="unused", base_url="http://localhost:11434/v1")

    for capability_call in (
        lambda: provider.transcribe_audio(audio_file=None),
        lambda: provider.synthesize_speech(text="hi"),
    ):
        try:
            capability_call()
            assert False, "expected NotImplementedError"
        except NotImplementedError as e:
            assert "localhost:11434" in str(e)


def test_backend_boots_ollama_provider_without_an_openai_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("TRANSLATION_PROVIDER", "ollama")

    backend = tb.TranslationBackend()

    assert backend.provider is not None
    assert backend.provider.is_openai_hosted is False


def test_backend_stays_providerless_without_key_when_provider_is_openai(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("TRANSLATION_PROVIDER", "openai")

    backend = tb.TranslationBackend()

    assert backend.provider is None


def test_ollama_provider_carries_the_input_char_cap():
    provider = tb.build_translation_provider("ollama", api_key="")
    assert provider.max_input_chars == tb.OLLAMA_MAX_INPUT_CHARS


def test_openai_provider_has_no_input_cap():
    provider = tb.build_translation_provider("openai", api_key="test-key")
    assert provider.max_input_chars is None


def test_split_into_chunks_respects_the_cap_and_preserves_all_text():
    text = (
        "The library opens at nine in the morning. It closes at six on weekdays. "
        "On Sundays it closes early at three. Please bring your card to check out books."
    )
    chunks = tb._split_into_chunks(text, 40)
    assert all(len(c) <= 40 for c in chunks)
    assert len(chunks) > 1
    # every sentence's distinctive words survive somewhere in some chunk
    for word in ("library", "Sundays", "card"):
        assert any(word in c for c in chunks)


def test_split_into_chunks_never_splits_mid_sentence_when_it_fits():
    text = "Short one. Another short one."
    chunks = tb._split_into_chunks(text, 100)
    assert chunks == ["Short one. Another short one."]


def test_split_into_chunks_hard_splits_a_single_oversized_sentence():
    long_sentence = "word " * 30  # no punctuation, one giant "sentence"
    chunks = tb._split_into_chunks(long_sentence.strip(), 20)
    assert len(chunks) > 1
    assert all(len(c) <= 20 for c in chunks)


def test_translate_text_splits_oversized_text_for_a_capped_provider(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    backend = tb.TranslationBackend()
    backend.provider.max_input_chars = 30
    calls = []

    def fake_chunk(text, target_language):
        calls.append(text)
        return f"[{target_language}:{text}]"

    monkeypatch.setattr(backend, "_translate_chunk", fake_chunk)

    long_text = "First sentence here. Second sentence here. Third sentence here."
    result = backend.translate_text(long_text, "French")

    assert len(calls) > 1, "expected the oversized text to be split into multiple model calls"
    assert all(len(c) <= 30 for c in calls)
    assert result == " ".join(f"[French:{c}]" for c in calls)


def test_translate_text_does_not_split_under_the_cap(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    backend = tb.TranslationBackend()
    backend.provider.max_input_chars = 1000
    calls = []
    monkeypatch.setattr(backend, "_translate_chunk", lambda t, lang: calls.append(t) or "translated")

    backend.translate_text("A short sentence.", "German")

    assert calls == ["A short sentence."]


def test_translate_text_uncapped_provider_never_splits(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    backend = tb.TranslationBackend()
    assert backend.provider.max_input_chars is None
    calls = []
    monkeypatch.setattr(backend, "_translate_chunk", lambda t, lang: calls.append(t) or "translated")

    long_text = "Sentence. " * 500  # far past any reasonable local-model cap
    backend.translate_text(long_text, "German")

    assert calls == [long_text.replace("\t", " ").strip()]
