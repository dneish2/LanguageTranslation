"""Local voice defaults ON when the models are actually installed.

DECISIONS.md §2: `PASSAGE_LOCAL_VOICE` is tri-state — unset means *auto* (on if
the speech stack and a voice are really present), "1" forces on, "0" forces
off. Local voice is faster (1.51s vs 7.62s) and more private, so defaulting off
penalised the better option.

Every assertion here is on the ENGINE LABEL, never on "it worked". A voice
pipeline that silently falls back to hosted still returns audio, so success
tells you nothing: broken local and absent local are indistinguishable unless
the label is checked. `translate_audio` returns meta["stt"]/meta["tts"], and
those are what these tests read.
"""
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from passage import engine_ledger, local_voice
from passage.ui.voice_page import format_engine_line


LOCAL_STT = f"local:{local_voice.WHISPER_MODEL}"


def _install_fake_models(monkeypatch, tmp_path, *, whisper=True, piper=True, voice=True):
    """Pretend the optional stack is (or isn't) installed.

    Fake modules rather than a patched `stt_available`: the thing under test IS
    the detection, so stubbing the detector would test nothing.
    """
    for name, present in (("faster_whisper", whisper), ("piper", piper)):
        if present:
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
        else:
            monkeypatch.setitem(sys.modules, name, None)  # import -> ImportError
    monkeypatch.setattr(local_voice, "VOICE_DIR", tmp_path)
    if voice:
        (tmp_path / "es_ES-davefx-medium.onnx").write_bytes(b"stub")


def _backend(monkeypatch):
    import TranslationBackend as tb
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    backend = tb.TranslationBackend()
    monkeypatch.setattr(backend, "translate_text", lambda text, lang, **kw: "hola")
    monkeypatch.setattr(backend, "_transcribe_hosted", lambda data: "from hosted")
    backend._require_provider().synthesize_speech = lambda text: b"hosted-mp3"
    return tb, backend


# ───────────────────────── (a) unset + models present ───────────────────── #

def test_unset_env_with_models_present_runs_local_and_says_so(monkeypatch, tmp_path):
    monkeypatch.delenv("PASSAGE_LOCAL_VOICE", raising=False)
    _install_fake_models(monkeypatch, tmp_path)
    tb, backend = _backend(monkeypatch)
    monkeypatch.setattr(tb.local_voice, "transcribe", lambda data, **kw: "hello there")
    monkeypatch.setattr(tb.local_voice, "synthesize", lambda text, language: b"RIFFwav")

    _, _, audio, meta = backend.translate_audio(b"audio", "Spanish")

    # THE LABELS, not the success: local:base heard it, piper spoke it.
    assert meta["stt"] == LOCAL_STT
    assert meta["tts"] == "local:piper"
    assert meta["media_type"] == "audio/wav" and audio == b"RIFFwav"


def test_auto_mode_is_reported_as_auto_not_as_a_flag(monkeypatch, tmp_path):
    monkeypatch.delenv("PASSAGE_LOCAL_VOICE", raising=False)
    _install_fake_models(monkeypatch, tmp_path)

    state = local_voice.status()
    assert state["mode"] == "auto" and state["enabled"] is True
    assert state["stt_engine"] == LOCAL_STT
    assert "detected automatically" in local_voice.describe()


# ─────────────────── (b) unset + faster_whisper missing ─────────────────── #

def test_unset_env_without_whisper_runs_hosted_and_says_hosted(monkeypatch, tmp_path):
    monkeypatch.delenv("PASSAGE_LOCAL_VOICE", raising=False)
    _install_fake_models(monkeypatch, tmp_path, whisper=False)
    tb, backend = _backend(monkeypatch)
    monkeypatch.setattr(tb.local_voice, "synthesize", lambda text, language: b"RIFFwav")

    source, _, _, meta = backend.translate_audio(b"audio", "Spanish")

    assert source == "from hosted"
    assert meta["stt"] == "hosted"          # the label is the point
    assert local_voice.status()["stt_engine"] == "hosted"


def test_auto_stays_off_entirely_when_nothing_is_installed(monkeypatch, tmp_path):
    monkeypatch.delenv("PASSAGE_LOCAL_VOICE", raising=False)
    _install_fake_models(monkeypatch, tmp_path, whisper=False, piper=False, voice=False)

    assert local_voice.enabled() is False
    assert local_voice.stt_available() is False
    assert local_voice.tts_available("Spanish") is False
    assert "off" in local_voice.describe()


# ───────────────────── (c) "0" forces hosted regardless ─────────────────── #

def test_zero_forces_hosted_even_with_every_model_present(monkeypatch, tmp_path):
    monkeypatch.setenv("PASSAGE_LOCAL_VOICE", "0")
    _install_fake_models(monkeypatch, tmp_path)
    tb, backend = _backend(monkeypatch)

    def must_not_run(*a, **kw):
        raise AssertionError("local voice ran despite PASSAGE_LOCAL_VOICE=0")

    monkeypatch.setattr(tb.local_voice, "transcribe", must_not_run)
    monkeypatch.setattr(tb.local_voice, "synthesize", must_not_run)

    _, _, _, meta = backend.translate_audio(b"audio", "Spanish")

    assert meta["stt"] == "hosted" and meta["tts"] == "hosted"
    assert meta["media_type"] == "audio/mpeg"


# ──────────────── (d) "1" with models missing degrades honestly ─────────── #

def test_forced_on_with_nothing_installed_admits_it(monkeypatch, tmp_path):
    """A flag cannot conjure a model. Forced-on must not read as success."""
    monkeypatch.setenv("PASSAGE_LOCAL_VOICE", "1")
    _install_fake_models(monkeypatch, tmp_path, whisper=False, piper=False, voice=False)

    state = local_voice.status()
    assert state["mode"] == "on" and state["forced_but_missing"] is True
    assert state["stt_engine"] == "hosted"
    text = local_voice.describe()
    assert "PASSAGE_LOCAL_VOICE=1" in text and "not installed" in text

    tb, backend = _backend(monkeypatch)
    _, _, _, meta = backend.translate_audio(b"audio", "Spanish")
    assert meta["stt"] == "hosted" and meta["tts"] == "hosted"


def test_forced_on_uses_whichever_half_exists(monkeypatch, tmp_path):
    """Whisper installed, no Piper voice: transcribe locally, speak hosted.
    The recording still never leaves, and each label says which happened."""
    monkeypatch.setenv("PASSAGE_LOCAL_VOICE", "1")
    _install_fake_models(monkeypatch, tmp_path, piper=False, voice=False)
    tb, backend = _backend(monkeypatch)
    monkeypatch.setattr(tb.local_voice, "transcribe", lambda data, **kw: "hello")

    _, _, _, meta = backend.translate_audio(b"audio", "Spanish")

    assert meta["stt"] == LOCAL_STT and meta["tts"] == "hosted"


# ─────────────────────────── fallback honesty ───────────────────────────── #

def test_broken_local_model_reports_hosted_not_local(monkeypatch, tmp_path):
    """The dangerous case: local was chosen, local failed, hosted served it.
    The label must name hosted — otherwise a broken local model looks private."""
    monkeypatch.delenv("PASSAGE_LOCAL_VOICE", raising=False)
    _install_fake_models(monkeypatch, tmp_path)
    tb, backend = _backend(monkeypatch)

    def boom(*a, **kw):
        raise RuntimeError("model file corrupt")

    monkeypatch.setattr(tb.local_voice, "transcribe", boom)
    monkeypatch.setattr(tb.local_voice, "synthesize", boom)

    _, _, audio, meta = backend.translate_audio(b"audio", "Spanish")

    assert meta["stt"] == "hosted" and meta["tts"] == "hosted"
    assert meta["media_type"] == "audio/mpeg" and audio == b"hosted-mp3"
    assert "corrupt" in meta["stt_fallback"] and "corrupt" in meta["tts_fallback"]
    assert "local was tried and failed" in format_engine_line(meta)


# ───────────────────────── /voice names the engine ──────────────────────── #

@pytest.mark.parametrize("meta, expected", [
    ({"stt": LOCAL_STT, "tts": "local:piper"}, "this machine"),
    ({"stt": "hosted", "tts": "hosted"}, "sent out"),
    ({"stt": LOCAL_STT, "tts": "hosted"}, "partly on this machine"),
])
def test_voice_page_line_names_the_engines_that_ran(meta, expected):
    line = format_engine_line(meta)
    assert meta["stt"] in line and meta["tts"] in line
    assert expected in line


def test_voice_page_renders_a_per_request_engine_element():
    """The line needs somewhere to land, and the JS has to fill it from the
    response header — a helper nothing calls would be transparency on paper."""
    source = (ROOT / "passage" / "ui" / "voice_page.py").read_text(encoding="utf-8")
    assert "id={scope}_engines" in source
    assert "X-Engine-Summary" in source
    assert "updateEngines(" in source


# ─────────────────── (e) /engines reflects RESOLVED state ───────────────── #

def test_engines_page_reports_resolved_state_not_the_env_var(monkeypatch, tmp_path):
    from passage import policy

    monkeypatch.delenv("PASSAGE_LOCAL_VOICE", raising=False)
    _install_fake_models(monkeypatch, tmp_path)
    row = engine_ledger.voice_state()
    assert row["mode"] == "auto" and row["is_local"] is True
    assert row["engine"] == LOCAL_STT
    # The privacy sentence beside it must agree — /engines once claimed
    # "hosted — metered" directly above "100% stayed on this machine".
    assert policy.voice_stays_local() is True
    assert "this machine" in policy.describe_voice_privacy("Spanish")

    monkeypatch.setenv("PASSAGE_LOCAL_VOICE", "0")
    row = engine_ledger.voice_state()
    assert row["mode"] == "off" and row["is_local"] is False and row["engine"] == "hosted"
    assert policy.voice_stays_local() is False

    monkeypatch.setenv("PASSAGE_LOCAL_VOICE", "1")
    _install_fake_models(monkeypatch, tmp_path, whisper=False, piper=False, voice=False)
    row = engine_ledger.voice_state()
    assert row["mode"] == "on" and row["is_local"] is False and row["engine"] == "hosted"
    assert row["forced_but_missing"] is True
    assert policy.voice_stays_local() is False
