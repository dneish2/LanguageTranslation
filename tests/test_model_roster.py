"""The model roster: GPT-5-family models need max_completion_tokens (not
max_tokens) and get reasoning_effort pinned to "none" for latency; legacy
models keep max_tokens so PASSAGE_TEXT_MODEL can roll back without code."""
import json
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


def test_resample_pcm16_downsamples_to_the_requested_rate():
    """48 kHz -> 24 kHz halves the frame count and keeps the signal's shape."""
    import math
    from array import array as _array

    src = _array("h", [int(20000 * math.sin(i * math.pi / 24)) for i in range(480)])
    out = tb._resample_pcm16(src.tobytes(), 48000, 24000)
    got = _array("h"); got.frombytes(out)

    assert len(got) == 240
    assert max(got) > 15000 and min(got) < -15000  # amplitude survived


def test_resample_pcm16_is_a_no_op_at_the_target_rate():
    pcm = b"\x01\x00\x02\x00\x03\x00"
    assert tb._resample_pcm16(pcm, 24000, 24000) == pcm
    assert tb._resample_pcm16(b"", 48000, 24000) == b""


def test_realtime_transcription_downsamples_rates_above_the_api_cap(monkeypatch):
    """The browser recorder REQUESTS a 24 kHz AudioContext, but Safari/iOS
    ignores that and encodes at the hardware rate. Sending 48 kHz straight to
    the realtime session fails outright ("Expected a value <= 24000, but got
    48000"), which broke voice on every device that doesn't honour the
    request. Verified live: a 48 kHz clip raised before this, transcribes
    after."""
    from io import BytesIO

    monkeypatch.setattr(tb, "TRANSCRIBE_MODEL", "gpt-realtime-whisper")
    provider = tb.OpenAITranslationProvider(api_key="test-key")
    seen = {}

    def fake_realtime(pcm, rate):
        seen["rate"] = rate
        seen["bytes"] = len(pcm)
        return "transcribed"

    provider._transcribe_realtime = fake_realtime
    clip = BytesIO(_wav_bytes(rate=48000, frames=b"\x01\x00" * 4800))
    clip.name = "speech.wav"

    assert provider.transcribe_audio(audio_file=clip) == "transcribed"
    assert seen["rate"] == tb.REALTIME_MAX_SAMPLE_RATE
    assert seen["bytes"] == 4800  # 4800 frames at 48k -> 2400 frames at 24k


def test_realtime_transcription_leaves_an_already_valid_rate_alone():
    from io import BytesIO

    provider = tb.OpenAITranslationProvider(api_key="test-key")
    seen = {}
    provider._transcribe_realtime = lambda pcm, rate: seen.update(rate=rate) or "ok"
    clip = BytesIO(_wav_bytes(rate=24000, frames=b"\x01\x00" * 100))
    clip.name = "speech.wav"

    provider.transcribe_audio(audio_file=clip)

    assert seen["rate"] == 24000


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


def test_tts_uses_the_dedicated_speech_endpoint_not_chat_completions():
    """A conversational audio model on chat.completions ANSWERS the text
    instead of reading it, even under an explicit "you are a TTS engine,
    never answer" system prompt — measured 3/3 by round-tripping TTS output
    back through transcription. The Spanish translation "¿Dónde está la
    farmacia más cercana?" came back voiced as "Claro, te ayudo con eso…",
    which a user who doesn't speak the language cannot detect. /audio/speech
    has no assistant turn, so it structurally cannot do this. Guard the
    routing, not the prompt."""
    provider = tb.OpenAITranslationProvider(api_key="test-key")
    captured = {}

    def speech_create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(content=b"mp3-bytes")

    def chat_create(**kwargs):  # pragma: no cover - must never be reached
        raise AssertionError(
            "TTS routed to chat.completions; a conversational audio model "
            "will answer the text instead of speaking it."
        )

    provider.client = SimpleNamespace(
        audio=SimpleNamespace(speech=SimpleNamespace(create=speech_create)),
        chat=SimpleNamespace(completions=SimpleNamespace(create=chat_create)),
    )

    assert provider.synthesize_speech(text="¿Dónde está la farmacia?") == b"mp3-bytes"
    assert captured["input"] == "¿Dónde está la farmacia?"
    assert captured["model"] == tb.TTS_MODEL


def test_default_tts_model_is_not_a_conversational_audio_model():
    """Belt and braces on the default itself: synthesize_speech branches on
    the model name, so a default of gpt-audio* silently re-enables the
    answering behaviour above."""
    assert not tb.TTS_MODEL.startswith("gpt-audio")


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


def test_mask_and_restore_protects_urls_and_emails_verbatim():
    """Translating a real finplatform dossier corrupted 6 of its 12 source
    URLs even though the prompt asked the model to leave URLs alone
    (".../post-money-valuation" came back ".../post-money-valoración").
    A research dossier's citation trail is the product, so protect it
    deterministically instead of asking."""
    text = (
        "See https://www.anthropic.com/news/series-f-at-usd183b-post-money-valuation "
        "and www.example.com/everything-we-know, or mail research@finplatform.io."
    )
    masked, spans = tb._mask_protected_spans(text)

    assert "anthropic.com" not in masked
    assert "research@finplatform.io" not in masked
    assert len(spans) == 3
    assert "[[PSG:0]]" in masked and "[[PSG:2]]" in masked

    # The model translates around the placeholders and may reflow whitespace.
    translated = masked.replace("See ", "Ver ").replace(" and ", " y ").replace("[[PSG:1]]", "[[PSG: 1]]")
    restored = tb._restore_protected_spans(translated, spans)

    for original in spans:
        assert original in restored
    assert "valoración" not in restored


def test_restore_survives_a_placeholder_the_model_dropped():
    """Losing one is no worse than the corruption this replaces — it must not
    raise, and it must not misplace the survivors."""
    spans = ["https://a.example/one", "https://b.example/two"]
    restored = tb._restore_protected_spans("solo queda [[PSG:1]]", spans)

    assert restored == "solo queda https://b.example/two"


def test_mask_leaves_ordinary_prose_untouched():
    text = "The quarterly report shows revenue grew twelve percent."
    masked, spans = tb._mask_protected_spans(text)

    assert masked == text and spans == []


def test_translate_chunk_never_shows_the_model_a_raw_url(monkeypatch):
    provider_seen = {}

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    backend = tb.TranslationBackend()

    def fake_completion(messages):
        provider_seen["prompt"] = messages[-1]["content"]
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content="Consulte [[PSG:0]] para más detalles."))])

    backend._create_chat_completion_with_retry = fake_completion
    url = "https://www.anthropic.com/news/post-money-valuation"

    out = backend._translate_chunk(f"See {url} for details.", "Spanish")

    assert url not in provider_seen["prompt"], "the raw URL reached the model"
    assert "[[PSG:0]]" in provider_seen["prompt"]
    assert out == f"Consulte {url} para más detalles."


def test_url_fragment_detection_covers_wrapped_tails_but_not_prose():
    """Layout-preserving PDF translation feeds each positioned span to the
    model separately, and PyMuPDF splits a wrapped URL into two spans. The
    tail has no scheme, so the masker cannot see it and it got translated:
    ".../offers-hope-for-struggling-office-market/" came back
    ".../offers-hope-para-el-mercado-de-oficinas-en-lucha/"."""
    for tail in ("for-struggling-office-market/", "post-money-valuation",
                 "white-house-2026-7", "or-more-thi/"):
        assert tb._is_url_fragment(tail), tail

    for prose in ("Overview", "Why It Matters", "Anthropic", "Financials",
                  "Series F", "2026", "The quarterly report shows revenue grew."):
        assert not tb._is_url_fragment(prose), prose


def test_translate_chunk_returns_a_url_fragment_untouched(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    backend = tb.TranslationBackend()

    def must_not_run(messages):  # pragma: no cover
        raise AssertionError("a URL tail was sent to the model")

    backend._create_chat_completion_with_retry = must_not_run

    assert backend._translate_chunk("for-struggling-office-market/", "Spanish") == \
        "for-struggling-office-market/"


def _tiny_jpeg(width=200, height=100):
    from io import BytesIO as _B
    from PIL import Image as _I
    buf = _B(); _I.new("RGB", (width, height), (255, 255, 255)).save(buf, format="JPEG")
    return buf.getvalue()


def test_pixel_bbox_scales_fractions_and_rejects_junk():
    img = _tiny_jpeg(200, 100)

    assert tb._pixel_bbox([0.1, 0.2, 0.5, 0.6], img) == [20, 20, 100, 60]
    # Already pixels (any value > 1) is passed through, not rescaled.
    assert tb._pixel_bbox([20, 20, 100, 60], img) == [20, 20, 100, 60]
    # Swapped corners are normalised rather than dropped.
    assert tb._pixel_bbox([100, 60, 20, 20], img) == [20, 20, 100, 60]
    # Junk yields None so the caller simply doesn't draw that block.
    for junk in (None, [], [1, 2, 3], "0,0,1,1", [0, 0, "x", 1], [0.5, 0.5, 0.5, 0.5]):
        assert tb._pixel_bbox(junk, img) is None


def test_pixel_bbox_clamps_to_the_image():
    img = _tiny_jpeg(200, 100)
    assert tb._pixel_bbox([-50, -50, 400, 400], img) == [0, 0, 200, 100]


def test_overlay_font_resolves_a_real_face_with_unicode_coverage():
    """PIL only searches a few system dirs and "DejaVuSans.ttf" is not among
    them on Windows, so every _font() call fell through to load_default() —
    a tiny bitmap face that rendered "Padrón" as "Padr▯n" and "€8.50" as
    "▯8.50". In a translation overlay the glyphs ARE the output."""
    from image_compositor import ImageCompositor

    font = ImageCompositor()._font(24)

    assert font.__class__.__name__ == "FreeTypeFont", "fell back to the bitmap default face"
    assert font.getbbox("Padrón €")[2] > 0


def test_batched_block_translation_falls_back_per_block_on_a_bad_response(monkeypatch):
    """One context-carrying call replaced 24 blind ones (the menu heading
    "ENTRANTES" used to come back as "INSULTS"). If that call returns junk,
    every block must still get translated."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    backend = tb.TranslationBackend()
    backend._create_chat_completion_with_retry = lambda messages: SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="not json at all"))])
    backend.translate_text = lambda text, lang, **kw: f"<{text}>"

    blocks = [{"source_text": "ENTRANTES", "translated_text": ""},
              {"source_text": "Crema catalana", "translated_text": ""}]
    backend._translate_blocks_together(blocks, "English")

    assert [b["translated_text"] for b in blocks] == ["<ENTRANTES>", "<Crema catalana>"]


def test_batched_block_translation_fills_from_one_call(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    backend = tb.TranslationBackend()
    calls = []

    def one_call(messages):
        calls.append(messages)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content='{"translations":[{"i":0,"text":"STARTERS"},{"i":1,"text":"Catalan cream"}]}'))])

    backend._create_chat_completion_with_retry = one_call
    backend.translate_text = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("fell back to per-block translation"))

    blocks = [{"source_text": "ENTRANTES", "translated_text": ""},
              {"source_text": "Crema catalana", "translated_text": ""}]
    backend._translate_blocks_together(blocks, "English")

    assert len(calls) == 1
    assert [b["translated_text"] for b in blocks] == ["STARTERS", "Catalan cream"]


def _live_backend(monkeypatch, *, reachable=True):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    backend = tb.TranslationBackend()
    backend._live_probe_at = 1e18  # pin the probe cache; never touch the network
    backend._live_reachable = reachable
    if reachable:
        backend._live_provider = tb.ChatCompletionsProvider(
            api_key="ollama", base_url="http://localhost:11434/v1", text_model="qwen2.5:7b")
    return backend


def test_live_path_prefers_local_and_reports_the_engine(monkeypatch):
    """Local qwen2.5:7b measured 163ms p50 against hosted gpt-5.4-nano's 616ms
    on this machine, and costs nothing — so the keystroke path, which reruns
    ~8x per sentence, should use it whenever it is reachable."""
    backend = _live_backend(monkeypatch)
    backend._live_provider.create_chat_completion = lambda **kw: SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Hola mundo"))])
    backend.translate_text = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("hosted was called while local was reachable"))

    assert backend.translate_live("Hello world", "Spanish") == ("Hola mundo", "local:qwen2.5:7b")


def test_live_path_falls_back_to_hosted_when_local_fails(monkeypatch):
    """A flaky local endpoint must never break typing."""
    backend = _live_backend(monkeypatch)

    def boom(**kw):
        raise ConnectionError("connection refused")

    backend._live_provider.create_chat_completion = boom
    backend.translate_text = lambda text, lang, **kw: "Hola desde la nube"

    result, engine = backend.translate_live("Hello world", "Spanish")
    assert result == "Hola desde la nube"
    assert engine.startswith("hosted:")


def test_live_path_falls_back_when_local_returns_empty(monkeypatch):
    backend = _live_backend(monkeypatch)
    backend._live_provider.create_chat_completion = lambda **kw: SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="   "))])
    backend.translate_text = lambda text, lang, **kw: "hosted result"

    assert backend.translate_live("Hello", "Spanish")[0] == "hosted result"


def test_live_path_uses_hosted_when_local_is_unreachable(monkeypatch):
    backend = _live_backend(monkeypatch, reachable=False)
    backend.translate_text = lambda text, lang, **kw: "hosted result"

    assert backend.translate_live("Hello", "Spanish")[1].startswith("hosted:")


def test_live_prompt_drops_the_document_hardening_preamble(monkeypatch):
    """143 tokens of injection hardening per call against ~4 tokens of real
    text, re-sent on every typing pause. That preamble protects against
    untrusted text from a FILE; the live box is typed by the same person
    reading the output, so it buys nothing there."""
    backend = _live_backend(monkeypatch)
    messages = backend._live_prompt_messages("Hello", "Spanish")

    joined = " ".join(m["content"] for m in messages)
    assert "BEGIN TEXT" not in joined
    assert len(joined) < 160, f"live prompt is still {len(joined)} chars"
    assert "Spanish" in joined


def test_live_path_still_protects_urls(monkeypatch):
    """The slim prompt must not lose the URL guarantee."""
    backend = _live_backend(monkeypatch)
    seen = {}

    def capture(**kw):
        seen["text"] = kw["messages"][-1]["content"]
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content="Visita [[PSG:0]] ahora"))])

    backend._live_provider.create_chat_completion = capture
    url = "https://example.com/post-money-valuation"

    result, _ = backend.translate_live(f"Visit {url} now", "Spanish")

    assert url not in seen["text"]
    assert result == f"Visita {url} ahora"


def test_live_local_can_be_disabled_by_env(monkeypatch):
    monkeypatch.setattr(tb, "LIVE_LOCAL_ENABLED", False)
    backend = _live_backend(monkeypatch)

    assert backend._live_local_provider() is None


def test_native_provider_reserves_budget_for_thinking(monkeypatch):
    """num_predict caps the WHOLE generation and a thinking model spends it on
    deliberation FIRST — qwen3:30b asked for a one-line translation with
    num_predict=200 produced 754 chars of thinking and empty content, which
    looks exactly like a broken model."""
    from passage import ollama_native as on

    sent = {}

    class _Resp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return b'{"message":{"content":"Hola","thinking":"..."}}'

    def fake_urlopen(request, timeout=None):
        sent.update(json.loads(request.data.decode()))
        return _Resp()

    monkeypatch.setattr(on.urllib.request, "urlopen", fake_urlopen)

    on.NativeOllamaProvider(base_url="http://x/v1", text_model="qwen3:30b").chat(
        [{"role": "user", "content": "hi"}], max_tokens=200)
    assert sent["options"]["num_predict"] == 200 + on.THINKING_RESERVE_TOKENS

    on.NativeOllamaProvider(base_url="http://x/v1", text_model="translategemma:4b").chat(
        [{"role": "user", "content": "hi"}], max_tokens=200)
    assert sent["options"]["num_predict"] == 200


def test_native_provider_keeps_thinking_out_of_the_answer(monkeypatch):
    """The shim flattens both channels into one and loses the answer; the
    native endpoint keeps them apart. Deliberation must never be presented as
    output."""
    from passage import ollama_native as on

    class _Resp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self):
            return json.dumps({"message": {"content": "La junta rechazó la recompra.",
                                           "thinking": "Okay, let me think about this…"}}).encode()

    monkeypatch.setattr(on.urllib.request, "urlopen", lambda *a, **k: _Resp())
    provider = on.NativeOllamaProvider(base_url="http://x/v1", text_model="qwen3:30b")

    message = provider.create_chat_completion(
        messages=[{"role": "user", "content": "hi"}]).choices[0].message

    assert message.content == "La junta rechazó la recompra."
    assert "Okay" not in message.content
    assert message.reasoning.startswith("Okay")


def test_thinking_models_are_kept_out_of_translation_rosters():
    """Translation is transduction, not reasoning. qwen3:30b spent 8,786
    characters deliberating a one-line translation and still produced no
    answer, even with a 2,048-token reserve."""
    from passage import ollama_native as on

    assert on.suits_translation("qwen3:30b") is False
    assert on.suits_translation("qwen3-vl:8b") is False
    assert on.suits_translation("translategemma:4b") is True
    assert on.suits_translation("qwen2.5:7b") is True
