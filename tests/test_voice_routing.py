"""Voice/transcript routing: WHICH engine ran, and on WHOSE credentials.

Every test here asserts the path, not the success. The bugs these cover all
shared one shape — the call succeeded, and the page then described a path that
had not run:

  * the /voice engine line computed "this machine" from the two steps that can
    be local while the MIDDLE step, the one carrying the actual words, went to
    a hosted model;
  * /api/text_translate_stream translated a BYO user's transcript on PASSAGE'S
    key because it passed no profile;
  * X-Text-Engine had a reader in the page and no producer in the server, so
    the transcript engine line was permanently stuck on "the server did not
    name the engine" (a test produced the header itself, which is why it
    passed);
  * tts_available("Spanish") was True and tts_available("es") False with the
    same voice on disk — and "es" is BOTH the JS default and the server
    default for an empty language box;
  * a mangled placeholder shipped "[[PSG:0]]" to the user's screen.

No test here builds an object with __new__ and hand-assigned attributes: each
goes through the production entry point.
"""
import asyncio
import json
import re
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import TranslationBackend as tb
from TranslationBackend import TEXT_MODEL
from TranslationUI import TranslationUI
from passage import local_voice, provider_profiles
from passage.ui.voice_page import format_engine_line


# ───────────────────── V1: the engine line covers 3 steps ───────────────── #

def _voice_backend(monkeypatch, *, translated="hola"):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    backend = tb.TranslationBackend()
    monkeypatch.setattr(backend, "_transcribe_hosted", lambda data: "from hosted")
    monkeypatch.setattr(backend, "translate_text", lambda text, lang, **kw: translated)
    backend._require_provider().synthesize_speech = lambda text: b"hosted-mp3"
    return backend


def test_meta_names_the_engine_for_each_of_the_three_steps(monkeypatch):
    """meta must report the TRANSLATION step, not just heard/spoken."""
    backend = _voice_backend(monkeypatch)
    monkeypatch.setattr(tb.local_voice, "stt_available", lambda: True)
    monkeypatch.setattr(tb.local_voice, "tts_available", lambda lang: True)
    monkeypatch.setattr(tb.local_voice, "transcribe", lambda data, **kw: "hello there")
    monkeypatch.setattr(tb.local_voice, "synthesize", lambda text, language: b"RIFFwav")

    _source, _translated, _audio, meta = backend.translate_audio(b"audio", "Spanish")

    assert meta["stt"].startswith("local")
    assert meta["tts"] == "local:piper"
    # The step that carries the words. It ran hosted, and it says hosted.
    assert meta["translation"] == f"hosted:{TEXT_MODEL}"


def test_line_does_not_claim_this_machine_while_the_words_are_sent_out(monkeypatch):
    """The exact false claim: local mic + local voice, hosted transcript.

    Would this fail if the middle step were ignored again? Yes — ignoring it
    is precisely what produced the bare "this machine" this asserts against.
    """
    backend = _voice_backend(monkeypatch)
    monkeypatch.setattr(tb.local_voice, "stt_available", lambda: True)
    monkeypatch.setattr(tb.local_voice, "tts_available", lambda lang: True)
    monkeypatch.setattr(tb.local_voice, "transcribe", lambda data, **kw: "hello there")
    monkeypatch.setattr(tb.local_voice, "synthesize", lambda text, language: b"RIFFwav")

    line = format_engine_line(backend.translate_audio(b"audio", "Spanish")[3])

    assert "partly on this machine" in line
    assert not re.search(r"—\s*this machine", line)
    assert TEXT_MODEL in line and "translated by" in line


def test_full_local_is_the_only_thing_that_earns_this_machine():
    """All three local → "this machine"; drop any one → it is withdrawn."""
    all_local = {"stt": "local:base", "translation": "local:qwen2.5", "tts": "local:piper"}
    assert format_engine_line(all_local).endswith("this machine")

    for step in ("stt", "translation", "tts"):
        partial = dict(all_local, **{step: "hosted"})
        assert "partly on this machine" in format_engine_line(partial), step


def test_a_meta_without_a_translation_label_never_over_claims():
    """An older/other producer omitting the key must degrade to honest."""
    assert "partly on this machine" in format_engine_line(
        {"stt": "local:base", "tts": "local:piper"})


# ────────────── V2/V3: the stream endpoint honours the profile ──────────── #

def _app_request(ui_app):
    return types.SimpleNamespace(
        headers={"x-passage-token": ui_app.api_guard.issue_token()},
        client=types.SimpleNamespace(host="127.0.0.1"),
    )


class _RecordingProvider:
    """Stands in for the endpoint a BYO profile points at."""

    def __init__(self, label):
        self.label = label
        self.calls = []

    def create_chat_completion(self, messages, **kwargs):
        self.calls.append(messages)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(
            message=types.SimpleNamespace(content=f"{self.label}-translation"))])


def _byo_ui(monkeypatch, ui_app):
    """A visitor with their OWN endpoint configured, plumbed the production
    way: active_profile is session storage, so it is patched at the property
    and the provider is resolved through backend.provider_for_profile."""
    profile = provider_profiles.ProviderProfile(
        label="My endpoint", kind="openai_compatible",
        model="my-model", id="byo-1", base_url="http://byo.invalid/v1",
        api_key="byo-key",
    )
    monkeypatch.setattr(type(ui_app), "active_profile",
                        property(lambda self: profile))
    theirs = _RecordingProvider("byo")
    passages = _RecordingProvider("passage")
    monkeypatch.setattr(ui_app.backend, "provider_for_profile", lambda p: theirs)
    ui_app.backend.provider = passages
    return profile, theirs, passages


def test_stream_fallback_translates_on_the_users_own_endpoint(monkeypatch):
    """The credential boundary: THEIR endpoint gets the request, ours does not."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("LIVE_TEXT_STREAMING", "false")
    monkeypatch.setenv("LIVE_TEXT_STREAMING_CHAR_THRESHOLD", "999")
    ui_app = TranslationUI()
    profile, theirs, passages = _byo_ui(monkeypatch, ui_app)

    resp = asyncio.run(ui_app.api_text_translate_stream(
        _app_request(ui_app), text="hola mundo", language="es"))

    payload = json.loads(resp.body.decode())
    assert payload["translated_text"] == "byo-translation"
    assert len(theirs.calls) == 1, "the user's own endpoint never saw the transcript"
    assert passages.calls == [], "the transcript went out on Passage's key"
    # V3: the header the page reads is now produced by the server.
    assert resp.headers["X-Text-Engine"] == profile.describe()
    assert payload["engine"] == profile.describe()


def test_streaming_path_also_honours_the_profile(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("LIVE_TEXT_STREAMING", "true")
    ui_app = TranslationUI()
    profile, theirs, passages = _byo_ui(monkeypatch, ui_app)

    resp = asyncio.run(ui_app.api_text_translate_stream(
        _app_request(ui_app), text="hola mundo", language="es"))

    async def _collect():
        chunks = []
        async for part in resp.body_iterator:
            chunks.append(part.decode() if isinstance(part, bytes) else part)
        return "".join(chunks)

    body = asyncio.run(_collect())
    assert "byo-translation" in body
    assert len(theirs.calls) >= 1
    assert passages.calls == []
    assert resp.headers["X-Text-Engine"] == profile.describe()
    assert f'"engine": "{profile.describe()}"' in body


def test_stream_records_a_ledger_entry_naming_the_engine(monkeypatch):
    """One transcript request produced no engine-run record at all, so the
    transparency surface could not show what had just run."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("LIVE_TEXT_STREAMING", "false")
    monkeypatch.setenv("LIVE_TEXT_STREAMING_CHAR_THRESHOLD", "999")
    ui_app = TranslationUI()
    profile, _theirs, _passages = _byo_ui(monkeypatch, ui_app)
    recorded = []
    monkeypatch.setattr(ui_app, "_record_engine_run",
                        lambda **kw: recorded.append(kw))

    asyncio.run(ui_app.api_text_translate_stream(
        _app_request(ui_app), text="hola mundo", language="es"))

    assert [r["engine"] for r in recorded] == [profile.describe()]


def test_default_hosted_engine_is_named_not_left_blank(monkeypatch):
    """With no profile the header must still name the hosted model — the page
    falls back to "the server did not name the engine" otherwise."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("LIVE_TEXT_STREAMING", "false")
    monkeypatch.setenv("LIVE_TEXT_STREAMING_CHAR_THRESHOLD", "999")
    ui_app = TranslationUI()
    monkeypatch.setattr(ui_app.backend, "translate_text",
                        lambda text, language: "hello world")

    resp = asyncio.run(ui_app.api_text_translate_stream(
        _app_request(ui_app), text="hola mundo", language="es"))

    assert resp.headers["X-Text-Engine"] == f"hosted:{TEXT_MODEL}"


def test_the_page_reader_and_the_server_producer_agree():
    """The header had a reader and no producer, and only a TEST produced it.
    Both sides of the contract must live in shipped code."""
    page = (ROOT / "passage" / "ui" / "voice_page.py").read_text(encoding="utf-8")
    server = (ROOT / "TranslationUI.py").read_text(encoding="utf-8")
    assert "X-Text-Engine" in page
    assert '"X-Text-Engine": engine' in server


# ────────────────── V6: codes and names resolve identically ─────────────── #

@pytest.fixture
def installed_spanish_voice(monkeypatch, tmp_path):
    monkeypatch.setenv("PASSAGE_LOCAL_VOICE", "1")
    monkeypatch.setattr(local_voice, "VOICE_DIR", tmp_path)
    monkeypatch.setattr(local_voice, "piper_installed", lambda: True)
    (tmp_path / "es_ES-davefx-medium.onnx").write_bytes(b"stub")
    (tmp_path / "es_ES-davefx-medium.onnx.json").write_text("{}", encoding="utf-8")
    return tmp_path


@pytest.mark.parametrize("spelling", ["Spanish", "spanish", "es", "ES", "es-ES", "es_ES"])
def test_both_spellings_find_the_installed_local_voice(installed_spanish_voice, spelling):
    """'es' is the JS default AND the server default for an empty language
    box, so the DEFAULT spelling was the one that failed to see the voice."""
    assert local_voice.voice_file_for(spelling) is not None, spelling
    assert local_voice.tts_available(spelling) is True, spelling


@pytest.mark.parametrize("code, name", [("es", "Spanish"), ("fr", "French"), ("de", "German")])
def test_code_and_name_resolve_to_the_same_voice_prefix(code, name):
    assert local_voice.iso_code(code) == local_voice.iso_code(name) == code


def test_an_unknown_code_still_has_no_local_voice(installed_spanish_voice):
    """Accepting codes must not start answering for languages we lack."""
    assert local_voice.tts_available("ja") is False
    assert local_voice.tts_available("Japanese") is False


def test_the_default_language_request_uses_the_local_voice(monkeypatch, installed_spanish_voice):
    """End to end through translate_audio with the DEFAULT spelling: the
    assertion is that local synthesis RAN, not that a call succeeded."""
    backend = _voice_backend(monkeypatch)
    monkeypatch.setattr(tb.local_voice, "stt_available", lambda: False)
    spoke_locally = []

    def _synth(text, language):
        spoke_locally.append(language)
        return b"RIFFwav"

    monkeypatch.setattr(tb.local_voice, "synthesize", _synth)

    def _must_not_run(text):
        raise AssertionError("translated text was sent out for hosted synthesis")

    backend._require_provider().synthesize_speech = _must_not_run

    _s, _t, audio, meta = backend.translate_audio(b"audio", "es")

    assert spoke_locally == ["es"]
    assert audio == b"RIFFwav"
    assert meta["tts"] == "local:piper" and meta["media_type"] == "audio/wav"


# ─────────────── V8: a failed placeholder restore is detected ───────────── #

@pytest.mark.parametrize("mangled", [
    "Visita [[PSG 0]] hoy",         # colon eaten: past the strict regex
    "Visita [PSG:0] hoy",           # one bracket eaten
    "Visita [[PSG:7]] hoy",         # index the mask never issued
    "Visita [[psg:0]] hoy",         # case-mangled
])
def test_a_mangled_placeholder_is_detected_not_shipped(mangled, caplog):
    spans = ["https://example.com/a"]
    with caplog.at_level("ERROR"):
        restored = tb._restore_protected_spans(mangled, spans)

    assert tb.detect_placeholder_debris(restored) == []
    assert "PSG" not in restored
    assert any("placeholder restore failed" in r.message for r in caplog.records)


def test_a_clean_restore_is_untouched_and_flags_nothing(caplog):
    masked, spans = tb._mask_protected_spans("Visita https://example.com/a hoy")
    with caplog.at_level("ERROR"):
        restored = tb._restore_protected_spans(masked, spans)

    assert restored == "Visita https://example.com/a hoy"
    assert tb.detect_placeholder_debris(restored) == []
    assert not [r for r in caplog.records if "placeholder restore failed" in r.message]


def test_detector_would_catch_the_leak_that_reached_a_user():
    """The observed symptom: [[PSG:0]] visible in output."""
    assert tb.detect_placeholder_debris("Hola [[PSG:0]] mundo") == ["[[PSG:0]]"]
    assert tb.detect_placeholder_debris("Hola mundo") == []


# ───────────── V4/V7: the page clears stale panes and audio ─────────────── #
# UNVERIFIED BY BROWSER: driving these needs a real microphone, which this
# machine does not have. These assert the shipped JS contains the clearing
# calls on every failure/text-only path — a source-level check, not a
# behavioural one, and reported as such.

def test_voice_js_clears_stale_output_on_every_path():
    from passage.ui.voice_page import VOICE_PAGE_JS as js

    assert "function clearAudioPlayer(" in js
    assert "function clearResultPanes(" in js
    # transcript path: text only, so the previous recording's audio must go
    transcript = js.split("async function translateTranscriptFallback")[1]
    assert "clearAudioPlayer();" in transcript
    assert "clearResultPanes();" in transcript.split("} catch (e) {")[1]
    # recording path: cleared before the request and again on failure
    stop = js.split("async function stopRecording")[1].split("async function translateTranscriptFallback")[0]
    assert "clearResultPanes();" in stop
    assert "clearResultPanes();" in stop.split("} catch(e) {")[1]
