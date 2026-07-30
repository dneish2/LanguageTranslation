"""Local speech: availability gating, language codes, and hosted fallback.

These use the PASSAGE_LOCAL_VOICE env var rather than pinning the old
module-level ENABLED constant: the flag is tri-state and resolved per call now
(see test_local_voice_default.py), so a frozen boolean is no longer the thing
under test.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from passage import local_voice


def test_iso_codes_come_from_the_voice_table_not_a_name_slice():
    """Slicing the first two letters of the English name looks right and
    mostly isn't: "Spanish" -> "sp" (it is "es"), "German" -> "ge" ("de"),
    "Dutch" -> "du" ("nl"). Whisper rejected "sp" outright, which was the good
    case — a slice that happens to be valid for a DIFFERENT language would
    have quietly transcribed as the wrong one."""
    assert local_voice.iso_code("Spanish") == "es"
    assert local_voice.iso_code("German") == "de"
    assert local_voice.iso_code("Dutch") == "nl"
    assert local_voice.iso_code("English") == "en"
    assert local_voice.iso_code("Portuguese") == "pt"
    # Unknown languages get None so Whisper auto-detects, which beats a guess.
    assert local_voice.iso_code("Klingon") is None
    assert local_voice.iso_code(None) is None


def test_everything_is_off_when_the_feature_is_disabled(monkeypatch):
    """Local speech pulls in a model stack; it must be opt-in, and disabled
    must mean disabled even when the libraries happen to be installed."""
    monkeypatch.setenv("PASSAGE_LOCAL_VOICE", "0")

    assert local_voice.stt_available() is False
    assert local_voice.tts_available("Spanish") is False
    assert "off" in local_voice.describe()


def test_transcribe_raises_rather_than_silently_doing_nothing(monkeypatch):
    """Callers fall back to hosted on the exception. Returning "" would look
    like a successful transcription of silence."""
    monkeypatch.setenv("PASSAGE_LOCAL_VOICE", "0")
    try:
        local_voice.transcribe(b"\x00\x01")
        assert False, "expected RuntimeError"
    except RuntimeError as error:
        assert "not available" in str(error)


def test_synthesis_needs_a_voice_for_that_language(monkeypatch, tmp_path):
    """A missing voice falls back to hosted rather than speaking one language
    with another's phonemes."""
    monkeypatch.setenv("PASSAGE_LOCAL_VOICE", "1")
    monkeypatch.setattr(local_voice, "VOICE_DIR", tmp_path)
    (tmp_path / "es_ES-davefx-medium.onnx").write_bytes(b"stub")
    # Weights alone are not a voice: discovery requires the .onnx.json Piper
    # loads alongside them.
    (tmp_path / "es_ES-davefx-medium.onnx.json").write_text("{}", encoding="utf-8")

    assert local_voice.voice_file_for("Spanish") is not None
    assert local_voice.voice_file_for("Japanese") is None
    try:
        local_voice.synthesize("hola", language="Japanese")
        assert False, "expected RuntimeError"
    except RuntimeError as error:
        assert "Japanese" in str(error)


def test_missing_voice_directory_is_not_an_error(monkeypatch, tmp_path):
    monkeypatch.setenv("PASSAGE_LOCAL_VOICE", "1")
    monkeypatch.setattr(local_voice, "VOICE_DIR", tmp_path / "nope")

    assert local_voice.voice_file_for("Spanish") is None
    assert local_voice.tts_available("Spanish") is False


def test_voice_steps_are_chosen_independently(monkeypatch):
    """Someone may have a local recogniser but no voice for their target
    language. Transcribing locally is still worth doing then — the recording
    never leaves even if the reply is synthesised elsewhere."""
    import TranslationBackend as tb

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    backend = tb.TranslationBackend()
    monkeypatch.setattr(tb.local_voice, "stt_available", lambda: True)
    monkeypatch.setattr(tb.local_voice, "tts_available", lambda lang: False)
    monkeypatch.setattr(tb.local_voice, "transcribe", lambda data, **kw: "hello there")
    monkeypatch.setattr(backend, "translate_text", lambda text, lang, **kw: "hola")
    backend._require_provider().synthesize_speech = lambda text: b"hosted-mp3"

    source, translated, audio, meta = backend.translate_audio(b"audio", "Japanese")

    assert (source, translated, audio) == ("hello there", "hola", b"hosted-mp3")
    assert meta["stt"].startswith("local") and meta["tts"] == "hosted"
    assert meta["media_type"] == "audio/mpeg"


def test_local_stt_failure_falls_back_to_hosted(monkeypatch):
    """A broken local model must never break voice."""
    import TranslationBackend as tb

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    backend = tb.TranslationBackend()
    monkeypatch.setattr(tb.local_voice, "stt_available", lambda: True)
    monkeypatch.setattr(tb.local_voice, "tts_available", lambda lang: False)

    def boom(data, **kw):
        raise RuntimeError("model file corrupt")

    monkeypatch.setattr(tb.local_voice, "transcribe", boom)
    monkeypatch.setattr(backend, "_transcribe_hosted", lambda data: "from hosted")
    monkeypatch.setattr(backend, "translate_text", lambda text, lang, **kw: "hola")
    backend._require_provider().synthesize_speech = lambda text: b"mp3"

    source, _, _, meta = backend.translate_audio(b"audio", "Spanish")

    assert source == "from hosted"
    assert meta["stt"] == "hosted"


def test_local_tts_reports_wav_not_mp3(monkeypatch):
    """Piper returns WAV and the hosted path MP3; sending the wrong media type
    leaves the browser guessing at the container."""
    import TranslationBackend as tb

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    backend = tb.TranslationBackend()
    monkeypatch.setattr(tb.local_voice, "stt_available", lambda: False)
    monkeypatch.setattr(tb.local_voice, "tts_available", lambda lang: True)
    monkeypatch.setattr(tb.local_voice, "synthesize", lambda text, language: b"RIFFwav")
    monkeypatch.setattr(backend, "_transcribe_hosted", lambda data: "hello")
    monkeypatch.setattr(backend, "translate_text", lambda text, lang, **kw: "hola")

    _, _, audio, meta = backend.translate_audio(b"audio", "Spanish")

    assert audio == b"RIFFwav"
    assert meta["media_type"] == "audio/wav" and meta["tts"] == "local:piper"
