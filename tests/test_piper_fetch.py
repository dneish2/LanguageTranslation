"""Piper voices: fetch on demand, pre-fetch in the background, never block.

A Piper voice is ~63MB. DECISIONS.md §3 resolves that we fetch on demand into
models/piper/ and pre-fetch the current target language at workspace load. The
load-bearing property is not that downloads work - it is that a MISSING voice
degrades to hosted TTS and *says so*, instead of stalling a translation behind
a 63MB transfer.

So every test here asserts on the engine label the pipeline reports
(meta["tts"]) or on an observable effect (what is on disk, which thread ran),
never merely that a call returned. The network is mocked throughout: no test
may pull a real voice.
"""
import sys
import threading
import time
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from passage import local_voice
from TranslationBackend import TranslationBackend


class _StubProvider:
    """Stands in for the hosted provider, recording that it was the one used."""

    def __init__(self):
        self.transcribe_calls = 0
        self.synthesize_calls = 0

    def transcribe_audio(self, audio_file=None, **_kwargs):
        self.transcribe_calls += 1
        return "hello there"

    def synthesize_speech(self, text=None, **_kwargs):
        self.synthesize_calls += 1
        return b"HOSTED-MP3"


def _backend_with_hosted_stub(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    backend = TranslationBackend()
    provider = _StubProvider()
    monkeypatch.setattr(backend, "_require_provider", lambda: provider)
    monkeypatch.setattr(backend, "translate_text", lambda text, language, **_k: "hola")
    return backend, provider


def _install_voice(directory: Path, stem: str = "es_ES-davefx-medium") -> Path:
    """A complete voice pair on disk, minus the 63MB."""
    directory.mkdir(parents=True, exist_ok=True)
    onnx = directory / f"{stem}.onnx"
    onnx.write_bytes(b"onnx-weights")
    (directory / f"{stem}.onnx.json").write_text('{"sample_rate": 22050}', encoding="utf-8")
    return onnx


def _wire_local_tts_to_disk(monkeypatch):
    """Make the local-TTS gate depend on what is actually on disk.

    `piper` is not installed in CI, so its import check would veto everything
    and the disk-presence mechanism - the thing under test - would never run.
    These stubs keep the mechanism (is the voice file there?) and drop only the
    library import.
    """
    monkeypatch.setattr(
        local_voice, "tts_available",
        lambda language: local_voice.voice_installed(language),
    )
    def _synthesize(text, *, language):
        if not local_voice.voice_installed(language):
            raise RuntimeError(f"No local voice installed for {language}.")
        return b"RIFF-LOCAL-WAV"
    monkeypatch.setattr(local_voice, "synthesize", _synthesize)


# ─────────────── (a) missing voice: hosted TTS, named, and no wait ───────────

def test_missing_voice_falls_back_to_hosted_tts_and_says_so(monkeypatch, tmp_path):
    """The whole point. No voice on disk -> the hosted provider synthesises,
    and meta["tts"] says "hosted" rather than quietly implying local."""
    monkeypatch.setenv("PASSAGE_LOCAL_VOICE", "1")
    monkeypatch.setattr(local_voice, "VOICE_DIR", tmp_path / "piper")
    monkeypatch.setattr(local_voice, "stt_available", lambda: False)
    _wire_local_tts_to_disk(monkeypatch)
    backend, provider = _backend_with_hosted_stub(monkeypatch)

    _source, _translated, audio, meta = backend.translate_audio(b"\x00\x01", "Spanish")

    assert meta["tts"] == "hosted"
    assert meta["stt"] == "hosted"
    assert meta["media_type"] == "audio/mpeg"
    assert audio == b"HOSTED-MP3"
    assert provider.synthesize_calls == 1


def test_translation_does_not_block_on_a_voice_download(monkeypatch, tmp_path):
    """A translation must never wait on a 63MB fetch. The download helper is
    wired to a transfer that would take a minute; the call still returns
    promptly, on the hosted path, and the fetch is never invoked at all."""
    monkeypatch.setenv("PASSAGE_LOCAL_VOICE", "1")
    monkeypatch.setattr(local_voice, "VOICE_DIR", tmp_path / "piper")
    monkeypatch.setattr(local_voice, "stt_available", lambda: False)
    _wire_local_tts_to_disk(monkeypatch)

    fetched = []

    def _slow_fetch(url, destination, *, timeout):
        fetched.append(url)
        time.sleep(60)  # would blow the assertion below if it ever ran inline

    monkeypatch.setattr(local_voice, "_fetch_url", _slow_fetch)
    backend, provider = _backend_with_hosted_stub(monkeypatch)

    started = time.monotonic()
    _s, _t, _audio, meta = backend.translate_audio(b"\x00\x01", "Spanish")
    elapsed = time.monotonic() - started

    assert meta["tts"] == "hosted"       # named, not silently degraded
    assert provider.synthesize_calls == 1
    assert fetched == []                 # the request path never downloads
    assert elapsed < 5


# ───────────────── (b) pre-fetch is off the render path, silent ──────────────

def test_prefetch_returns_immediately_and_downloads_on_another_thread(monkeypatch, tmp_path):
    """Modelled on prewarm_live: the caller (page render) must get control back
    at once, with the work on a different thread."""
    voice_dir = tmp_path / "piper"
    monkeypatch.setenv("PASSAGE_LOCAL_VOICE", "1")
    monkeypatch.setattr(local_voice, "VOICE_DIR", voice_dir)

    caller = threading.get_ident()
    ran_on = []
    release = threading.Event()

    def _fetch(url, destination, *, timeout):
        ran_on.append(threading.get_ident())
        destination.write_bytes(b"x")
        release.set()

    monkeypatch.setattr(local_voice, "_fetch_url", _fetch)

    started = time.monotonic()
    local_voice.prefetch_voice("Spanish")
    elapsed = time.monotonic() - started

    assert elapsed < 1                       # never awaited by the UI
    assert release.wait(timeout=5)
    for _ in range(50):
        if local_voice.voice_installed("Spanish"):
            break
        time.sleep(0.02)
    assert ran_on and caller not in ran_on   # off the render path
    assert local_voice.voice_installed("Spanish")


def test_a_failing_prefetch_never_raises_into_the_ui(monkeypatch, tmp_path):
    """The network is the thing most likely to fail, and a page render is the
    worst place to learn about it. The failure is logged and swallowed, and
    leaves the language still un-installed so hosted keeps serving."""
    monkeypatch.setenv("PASSAGE_LOCAL_VOICE", "1")
    monkeypatch.setattr(local_voice, "VOICE_DIR", tmp_path / "piper")
    done = threading.Event()

    def _boom(url, destination, *, timeout):
        done.set()
        raise OSError("network unreachable")

    monkeypatch.setattr(local_voice, "_fetch_url", _boom)

    local_voice.prefetch_voice("Spanish")   # must not raise
    assert done.wait(timeout=5)
    time.sleep(0.1)
    assert local_voice.voice_installed("Spanish") is False
    # …and calling the helper directly is equally non-raising.
    assert local_voice.ensure_voice("Spanish") is None


def test_ui_workspace_load_prefetches_the_current_target_language(monkeypatch):
    """The hook exists on the UI object, passes the CURRENT target language,
    and survives a prefetch that explodes synchronously."""
    import TranslationUI

    ui_obj = TranslationUI.TranslationUI.__new__(TranslationUI.TranslationUI)
    ui_obj.current_target_language = "French"

    asked = []
    monkeypatch.setattr(TranslationUI.local_voice, "prefetch_voice", asked.append)
    ui_obj._prefetch_local_voice()
    assert asked == ["French"]

    def _explode(_language):
        raise RuntimeError("thread pool exhausted")

    monkeypatch.setattr(TranslationUI.local_voice, "prefetch_voice", _explode)
    ui_obj._prefetch_local_voice()   # must not raise into page render


# ──────────── (c) an interrupted download leaves nothing "installed" ─────────

def test_interrupted_download_leaves_nothing_a_probe_calls_installed(monkeypatch, tmp_path):
    """Killed mid-transfer, the partial weights must not be visible as a voice.
    Otherwise the next translation reports local, then fails to load a truncated
    .onnx - brokenness made indistinguishable from absence."""
    voice_dir = tmp_path / "piper"
    monkeypatch.setenv("PASSAGE_LOCAL_VOICE", "1")
    monkeypatch.setattr(local_voice, "VOICE_DIR", voice_dir)

    def _die_midway(url, destination, *, timeout):
        destination.write_bytes(b"half-a-voice")     # bytes really do land…
        if url.endswith(".onnx"):
            raise OSError("connection reset")        # …then the transfer dies

    monkeypatch.setattr(local_voice, "_fetch_url", _die_midway)

    assert local_voice.ensure_voice("Spanish") is None
    assert local_voice.voice_installed("Spanish") is False
    assert local_voice.voice_file_for("Spanish") is None      # the glob probe too
    assert list(voice_dir.glob("*.onnx")) == []
    assert list(voice_dir.glob("*.part")) == []               # temporaries cleaned up


def test_a_partial_pair_without_its_json_is_not_installed(monkeypatch, tmp_path):
    """Weights without the companion .onnx.json cannot be loaded by Piper, so
    the stricter probe must not call that installed even though the glob finds
    a file."""
    voice_dir = tmp_path / "piper"
    voice_dir.mkdir()
    (voice_dir / "es_ES-davefx-medium.onnx").write_bytes(b"weights-only")
    monkeypatch.setenv("PASSAGE_LOCAL_VOICE", "1")
    monkeypatch.setattr(local_voice, "VOICE_DIR", voice_dir)

    assert local_voice.voice_file_for("Spanish") is not None
    assert local_voice.voice_installed("Spanish") is False


def test_a_good_voice_is_never_shadowed_by_a_refetch(monkeypatch, tmp_path):
    """Re-fetching an installed language must not touch the working file."""
    voice_dir = tmp_path / "piper"
    onnx = _install_voice(voice_dir)
    monkeypatch.setenv("PASSAGE_LOCAL_VOICE", "1")
    monkeypatch.setattr(local_voice, "VOICE_DIR", voice_dir)

    def _must_not_run(*_args, **_kwargs):
        raise AssertionError("re-downloaded an already-installed voice")

    monkeypatch.setattr(local_voice, "_fetch_url", _must_not_run)

    assert local_voice.ensure_voice("Spanish") == onnx
    assert onnx.read_bytes() == b"onnx-weights"


# ───────────── (d) once the voice exists, the label names the local engine ───

def test_once_the_voice_is_present_meta_names_the_local_engine(monkeypatch, tmp_path):
    """The pay-off of the pre-fetch: same call, same language, but now
    meta["tts"] says local:piper and the hosted provider is not asked."""
    voice_dir = tmp_path / "piper"
    _install_voice(voice_dir)
    monkeypatch.setenv("PASSAGE_LOCAL_VOICE", "1")
    monkeypatch.setattr(local_voice, "VOICE_DIR", voice_dir)
    monkeypatch.setattr(local_voice, "stt_available", lambda: False)
    _wire_local_tts_to_disk(monkeypatch)
    backend, provider = _backend_with_hosted_stub(monkeypatch)

    _s, _t, audio, meta = backend.translate_audio(b"\x00\x01", "Spanish")

    assert meta["tts"] == "local:piper"
    assert meta["media_type"] == "audio/wav"
    assert audio == b"RIFF-LOCAL-WAV"
    assert provider.synthesize_calls == 0


def test_the_download_flips_the_label_from_hosted_to_local(monkeypatch, tmp_path):
    """End to end, with the network mocked: hosted before the fetch, local
    after it, from the identical call. The label tracks reality rather than
    configuration."""
    voice_dir = tmp_path / "piper"
    monkeypatch.setenv("PASSAGE_LOCAL_VOICE", "1")
    monkeypatch.setattr(local_voice, "VOICE_DIR", voice_dir)
    monkeypatch.setattr(local_voice, "stt_available", lambda: False)
    _wire_local_tts_to_disk(monkeypatch)
    backend, provider = _backend_with_hosted_stub(monkeypatch)

    before = backend.translate_audio(b"\x00\x01", "Spanish")[3]
    assert before["tts"] == "hosted"

    def _fetch(url, destination, *, timeout):
        destination.write_bytes(b"downloaded-bytes")

    monkeypatch.setattr(local_voice, "_fetch_url", _fetch)
    assert local_voice.ensure_voice("Spanish") is not None

    after = backend.translate_audio(b"\x00\x02", "Spanish")[3]
    assert after["tts"] == "local:piper"
    assert provider.synthesize_calls == 1   # only the first, hosted, call


def test_a_language_with_no_piper_voice_is_not_downloadable(monkeypatch, tmp_path):
    """Japanese has no entry, so there is nothing to fetch and nothing to
    pretend about: no request is made and hosted stays the answer."""
    monkeypatch.setenv("PASSAGE_LOCAL_VOICE", "1")
    monkeypatch.setattr(local_voice, "VOICE_DIR", tmp_path / "piper")

    def _must_not_run(*_args, **_kwargs):
        raise AssertionError("tried to download a voice that does not exist")

    monkeypatch.setattr(local_voice, "_fetch_url", _must_not_run)

    assert local_voice.ensure_voice("Japanese") is None
    local_voice.prefetch_voice("Japanese")   # no thread, no request, no raise


def test_every_downloadable_voice_matches_the_prefix_table(monkeypatch):
    """A voice you can select but cannot fetch would be a permanent silent
    fallback to hosted, so the two tables must agree."""
    assert set(local_voice.VOICE_ASSETS) == set(local_voice.VOICE_PREFIXES.values())
    for prefix, (repo_path, stem) in local_voice.VOICE_ASSETS.items():
        assert stem.startswith(prefix + "-")
        assert repo_path.split("/")[1] == prefix
