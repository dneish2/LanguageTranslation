"""Local speech recognition must not be called "ready" because an import worked.

The bug these cover was a PRIVACY OVERCLAIM, which in this app is the worst
kind. `whisper_installed()` asked only "does `import faster_whisper` succeed",
`stt_available()` added nothing, and the ~140MB of Whisper weights are fetched
lazily inside `_whisper()` on the FIRST RECORDING. So the documented install
path (pip install -r requirements-local-voice.txt, then open /engines before
ever recording) reported `stt_ready: True`, `stt_engine: local:base`,
"speech recognition: ready", and the flat promise "Recordings are transcribed
on this machine." The first recording then downloaded on the request path or
fell back to hosted: the user's voice left the machine after being told it
would not.

Every test here therefore asserts on the REPORTED STATE and the PRIVACY TEXT,
never on a call succeeding — a test that only checked "transcribe returned"
would have passed on the broken code.

Absence is SIMULATED, never inherited: this machine really does have
models--Systran--faster-whisper-base in its HuggingFace cache, which is exactly
why the bug was invisible here. HF_HUB_CACHE/HF_HOME are pointed at an empty
tmp_path so the fresh-machine case is the one under test, and `faster_whisper`
is made importable with a stub module so the "importable but not downloaded"
state is reachable on a machine that has never installed it.
"""
from __future__ import annotations

import sys
import types

import pytest

from passage import local_voice, policy


def _empty_hf_cache(monkeypatch, tmp_path):
    """Point every cache root this probe knows at an empty directory."""
    cache = tmp_path / "hf"
    cache.mkdir()
    monkeypatch.setenv("HF_HUB_CACHE", str(cache))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf_home"))
    return cache


def _install_weights(cache, model="base"):
    """Lay down the real on-disk shape faster-whisper loads from."""
    snapshot = (cache / f"models--Systran--faster-whisper-{model}"
                / "snapshots" / "deadbeef")
    snapshot.mkdir(parents=True)
    (snapshot / "model.bin").write_bytes(b"not really a model")
    return snapshot


def _importable(monkeypatch, present=True):
    """Make `import faster_whisper` succeed (or fail) for the production path.

    A stub module in sys.modules, not a patched `whisper_installed`: the thing
    under test is what the real `import` in `whisper_installed()` sees.
    """
    if present:
        monkeypatch.setitem(sys.modules, "faster_whisper",
                            types.ModuleType("faster_whisper"))
    else:
        monkeypatch.setitem(sys.modules, "faster_whisper", None)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path):
    monkeypatch.delenv("PASSAGE_LOCAL_VOICE", raising=False)
    # No Piper voices either, so nothing here leans on this repo's real ones.
    monkeypatch.setattr(local_voice, "VOICE_DIR", tmp_path / "no_voices")
    monkeypatch.setattr(local_voice, "WHISPER_MODEL", "base")


# ----------------------------- state 1: absent -----------------------------

def test_not_installed_reports_hosted(monkeypatch, tmp_path):
    _empty_hf_cache(monkeypatch, tmp_path)
    _importable(monkeypatch, present=False)

    assert local_voice.stt_state() == local_voice.STT_NOT_INSTALLED
    state = local_voice.status()
    assert state["stt_ready"] is False
    assert state["stt_engine"] == "hosted"
    assert "speech recognition: not installed" in local_voice.describe()


# --------------- state 2: importable, weights NOT on disk ------------------

def test_importable_without_weights_is_not_ready(monkeypatch, tmp_path):
    """The exact measured scenario: pip installed, never recorded."""
    _empty_hf_cache(monkeypatch, tmp_path)
    _importable(monkeypatch)

    assert local_voice.whisper_installed() is True      # the import DOES work
    assert local_voice.whisper_weights_present() is False
    assert local_voice.stt_state() == local_voice.STT_NOT_DOWNLOADED

    state = local_voice.status()
    assert state["stt_ready"] is False, "import-only must never read as ready"
    assert state["stt_engine"] == "hosted"
    assert state["stt_state"] == local_voice.STT_NOT_DOWNLOADED


def test_not_downloaded_is_a_distinct_third_state_in_the_text(monkeypatch, tmp_path):
    """The page must not collapse "not downloaded" into "not installed"."""
    _empty_hf_cache(monkeypatch, tmp_path)
    _importable(monkeypatch)

    line = local_voice.describe()
    assert "speech recognition: ready" not in line
    assert "not downloaded" in line
    # It is a different sentence from the never-installed one.
    assert local_voice.STT_LABELS[local_voice.STT_NOT_DOWNLOADED] != \
        local_voice.STT_LABELS[local_voice.STT_NOT_INSTALLED]


def test_privacy_line_does_not_promise_local_transcription(monkeypatch, tmp_path):
    """The sentence the user reads, in the state that used to lie."""
    _empty_hf_cache(monkeypatch, tmp_path)
    _importable(monkeypatch)

    assert policy.voice_stays_local() is False
    for line in (policy.describe_voice_privacy(),
                 policy.describe_voice_privacy("Spanish")):
        assert "transcribed on this machine" not in line
        assert "on this machine" not in line
        assert "hosted" in line


def test_forced_on_without_weights_still_not_ready(monkeypatch, tmp_path):
    """A flag cannot conjure 140MB of weights."""
    _empty_hf_cache(monkeypatch, tmp_path)
    _importable(monkeypatch)
    monkeypatch.setenv("PASSAGE_LOCAL_VOICE", "1")

    assert local_voice.status()["stt_ready"] is False
    assert local_voice.stt_state() == local_voice.STT_NOT_DOWNLOADED
    assert "transcribed on this machine" not in policy.describe_voice_privacy()


def test_transcribe_refuses_rather_than_downloading_on_the_request_path(
        monkeypatch, tmp_path):
    """The gate is real: no weights means no lazy fetch inside a request."""
    _empty_hf_cache(monkeypatch, tmp_path)
    _importable(monkeypatch)

    def _must_not_run():
        raise AssertionError("model construction reached the request path")

    monkeypatch.setattr(local_voice, "_whisper", _must_not_run)
    with pytest.raises(RuntimeError):
        local_voice.transcribe(b"RIFF....", language="Spanish")


# --------------------- state 3: weights actually present -------------------

def test_weights_present_reports_the_local_engine(monkeypatch, tmp_path):
    cache = _empty_hf_cache(monkeypatch, tmp_path)
    _install_weights(cache)
    _importable(monkeypatch)

    assert local_voice.stt_state() == local_voice.STT_READY
    state = local_voice.status()
    assert state["stt_ready"] is True
    assert state["stt_engine"] == "local:base"
    assert "speech recognition: ready" in local_voice.describe()
    assert policy.voice_stays_local() is True
    assert "transcribed on this machine" in policy.describe_voice_privacy()


def test_engine_label_follows_the_configured_size(monkeypatch, tmp_path):
    """Readiness probes the model that will actually be loaded, not "base"."""
    cache = _empty_hf_cache(monkeypatch, tmp_path)
    _install_weights(cache, "base")
    _importable(monkeypatch)
    monkeypatch.setattr(local_voice, "WHISPER_MODEL", "small")

    # base on disk says nothing about small.
    assert local_voice.stt_state() == local_voice.STT_NOT_DOWNLOADED

    _install_weights(cache, "small")
    assert local_voice.stt_state() == local_voice.STT_READY
    assert local_voice.status()["stt_engine"] == "local:small"


def test_incomplete_snapshot_is_not_ready(monkeypatch, tmp_path):
    """A snapshot dir with metadata but no model.bin is an aborted download."""
    cache = _empty_hf_cache(monkeypatch, tmp_path)
    snapshot = cache / "models--Systran--faster-whisper-base" / "snapshots" / "abc"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    _importable(monkeypatch)

    assert local_voice.stt_state() == local_voice.STT_NOT_DOWNLOADED


def test_off_beats_present_weights(monkeypatch, tmp_path):
    cache = _empty_hf_cache(monkeypatch, tmp_path)
    _install_weights(cache)
    _importable(monkeypatch)
    monkeypatch.setenv("PASSAGE_LOCAL_VOICE", "0")

    assert local_voice.stt_state() == local_voice.STT_OFF
    assert local_voice.status()["stt_engine"] == "hosted"
    assert "transcribed on this machine" not in policy.describe_voice_privacy()


# ------------------------------- prefetch ----------------------------------

@pytest.fixture
def _fresh_prefetch(monkeypatch):
    monkeypatch.setattr(local_voice, "_whisper_prefetch_started", False)
    monkeypatch.setattr(local_voice, "_whisper_prefetch_lock", None)


def test_prefetch_pulls_the_weights_off_the_request_path(
        monkeypatch, tmp_path, _fresh_prefetch):
    _empty_hf_cache(monkeypatch, tmp_path)
    _importable(monkeypatch)
    import threading
    loaded = threading.Event()
    monkeypatch.setattr(local_voice, "_whisper", loaded.set)

    local_voice.prefetch_whisper()
    assert loaded.wait(timeout=5), "prefetch never fetched the weights"


def test_prefetch_is_a_noop_once_the_weights_are_there(
        monkeypatch, tmp_path, _fresh_prefetch):
    cache = _empty_hf_cache(monkeypatch, tmp_path)
    _install_weights(cache)
    _importable(monkeypatch)

    def _must_not_run():
        raise AssertionError("re-downloaded weights that were already present")

    monkeypatch.setattr(local_voice, "_whisper", _must_not_run)
    local_voice.prefetch_whisper()


def test_prefetch_never_raises_when_the_download_fails(
        monkeypatch, tmp_path, _fresh_prefetch):
    """Failure is silent and non-blocking, like the voice prefetch."""
    _empty_hf_cache(monkeypatch, tmp_path)
    _importable(monkeypatch)

    def _boom():
        raise RuntimeError("no network")

    monkeypatch.setattr(local_voice, "_whisper", _boom)
    local_voice.prefetch_whisper()   # must not raise
    assert local_voice.stt_state() == local_voice.STT_NOT_DOWNLOADED
