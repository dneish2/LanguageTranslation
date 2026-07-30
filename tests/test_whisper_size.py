"""The Whisper size is a ONE-LINE change, and the change is real.

DECISIONS.md §1 keeps `base` as the default on evidence that is admittedly weak
(a synthetic TTS fixture; see RESEARCH.md §4a). The mitigation for a decision
made on thin evidence is that reversing it must be cheap and unambiguous: one
env var, one constant, no size smeared across call sites, and an engine label
that moves with it so a reader of `/voice` can see WHICH model actually ran.

These tests assert that mechanism, not the prose around it. They check the size
the STT path REQUESTS and the engine label the pipeline REPORTS — never merely
that transcription returned something.
"""
import importlib
import re
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from passage import local_voice

SIZES = ("tiny", "base", "small", "medium", "large", "large-v2", "large-v3")


def _app_sources():
    """Every app .py file — tests and vendored envs excluded."""
    for path in ROOT.rglob("*.py"):
        parts = set(path.parts)
        if parts & {".venv", "venv", "tests", "__pycache__", "build", "site-packages"}:
            continue
        yield path


def test_the_whisper_size_is_single_sourced():
    """A size named in two places is a two-line change that looks like a
    one-line change — the second site is the one someone forgets, and a stale
    engine label is worse than no label because it lies with confidence."""
    env_sites = [p for p in _app_sources()
                 if "PASSAGE_WHISPER_MODEL" in p.read_text(encoding="utf-8")]
    assert [p.name for p in env_sites] == ["local_voice.py"], (
        f"the whisper size env var is read in more than one place: {env_sites}"
    )

    # And no other module hard-codes a size string near a whisper/model call.
    literal = re.compile(r"""["'](%s)["']""" % "|".join(map(re.escape, SIZES)))
    offenders = []
    for path in _app_sources():
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.lstrip().startswith("#") or line.lstrip().startswith("#:"):
                continue
            if literal.search(line) and re.search(r"whisper|stt", line, re.I):
                if path.name == "local_voice.py" and "PASSAGE_WHISPER_MODEL" in line:
                    continue  # the one line
                offenders.append(f"{path.name}:{number}: {line.strip()}")
    assert offenders == [], f"whisper size named outside the one line: {offenders}"


@pytest.fixture
def whisper_spy(monkeypatch):
    """A fake faster-whisper that records the size it was constructed with."""
    requested = {}

    class FakeSegment:
        text = "hello there"

    class FakeWhisperModel:
        def __init__(self, size, device=None, compute_type=None):
            requested["size"] = size
            requested["device"] = device

        def transcribe(self, *args, **kwargs):
            return [FakeSegment()], None

    module = types.ModuleType("faster_whisper")
    module.WhisperModel = FakeWhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    return requested


def _reload_with(monkeypatch, size):
    """Reload local_voice with the documented env var set, as a fresh process
    would see it. Import-time constants are only honestly testable by re-import."""
    if size is None:
        monkeypatch.delenv("PASSAGE_WHISPER_MODEL", raising=False)
    else:
        monkeypatch.setenv("PASSAGE_WHISPER_MODEL", size)
    module = importlib.reload(local_voice)
    module._whisper.cache_clear()
    return module


@pytest.fixture
def restore_local_voice(monkeypatch):
    """Always put the module back the way the rest of the suite expects."""
    yield
    monkeypatch.delenv("PASSAGE_WHISPER_MODEL", raising=False)
    importlib.reload(local_voice)._whisper.cache_clear()


def test_default_size_is_base(monkeypatch, restore_local_voice):
    module = _reload_with(monkeypatch, None)
    assert module.WHISPER_MODEL == "base"


def test_env_override_changes_the_size_the_stt_path_requests(
    monkeypatch, whisper_spy, restore_local_voice
):
    """The documented one-line change must reach the model constructor. Asserting
    that transcription succeeded would pass even if the override were ignored."""
    module = _reload_with(monkeypatch, "small")
    monkeypatch.setattr(module, "stt_available", lambda: True)

    assert module.transcribe(b"audio") == "hello there"
    assert whisper_spy["size"] == "small", "override never reached WhisperModel"

    module = _reload_with(monkeypatch, None)
    monkeypatch.setattr(module, "stt_available", lambda: True)
    assert module.transcribe(b"audio") == "hello there"
    assert whisper_spy["size"] == "base"


def test_engine_label_reports_the_overridden_size_not_a_hardcoded_one(
    monkeypatch, whisper_spy, restore_local_voice
):
    """Engine label under override: meta["stt"] == "local:small" (and
    "local:base" by default). The label is the user's only evidence of which
    model heard them; if it were hard-coded, a size change would silently make
    the app misreport itself."""
    import TranslationBackend as tb

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    backend = tb.TranslationBackend()

    for size, expected in (("small", "local:small"), (None, "local:base")):
        module = _reload_with(monkeypatch, size)
        monkeypatch.setattr(tb, "local_voice", module)
        monkeypatch.setattr(module, "stt_available", lambda: True)
        monkeypatch.setattr(module, "tts_available", lambda lang: False)
        monkeypatch.setattr(backend, "translate_text", lambda text, lang, **kw: "hola")
        backend._require_provider().synthesize_speech = lambda text: b"hosted-mp3"

        _, _, _, meta = backend.translate_audio(b"audio", "Spanish")

        assert meta["stt"] == expected
        assert meta["tts"] == "hosted"
        assert whisper_spy["size"] == expected.split(":", 1)[1]
