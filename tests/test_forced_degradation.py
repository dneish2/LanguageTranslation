"""PORTABILITY_PLAN.md Phase D: break each capability on purpose and check the
app degrades HONESTLY — and says so.

    §2.5 "Fallbacks that have never executed are decorative."

The trap this file exists for (§2.2): on a machine with no Ollama and no voice
models, every local path silently goes hosted and the app looks perfect. A test
that only checks "the call returned" passes in a world where the local engine
never ran once. So every assertion below is on WHICH PATH RAN — the engine
label, the recorded fallback reason, the resolved font report, the raised
error — never on success.

Two disciplines make that real rather than aspirational:

* **Paired positive controls.** Each degradation test has a sibling that drives
  the SAME harness with the capability present and asserts the local label
  appears. Without it, "meta['stt'] == 'hosted'" is satisfied by a harness that
  could never have produced anything else, which is exactly the class of test
  the audit caught.
* **No bypasses.** Detection is stubbed at the boundary the app actually reads
  — the socket (`urllib.request.urlopen`), the import (`sys.modules`), the font
  file, the environment variable — never by patching the detector whose
  behaviour is under test, and never by assigning the attribute being verified.

Unverified, and labelled as such: nothing here says anything about Apple
Silicon. These run on Windows/CI x86-64; the arm64 question is answered by
.github/workflows/ci.yml's output, not by this file.
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
import types
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import image_compositor
from image_compositor import ImageCompositor, OverlayStyle, resolve_font
from passage import local_voice
from passage.ui.voice_page import VOICE_PAGE_JS, format_engine_line


# ═══════════════════════ 1. NO OLLAMA REACHABLE ═══════════════════════════ #
#
# The live path probes Ollama over HTTP. Breaking it at the socket means the
# app's own reachability code runs for real; patching `_live_local_provider`
# would test nothing but the patch.

class _FakeCompletion:
    def __init__(self, text: str) -> None:
        self.choices = [types.SimpleNamespace(message=types.SimpleNamespace(content=text))]


class _FakeLocalProvider:
    """Stands in for NativeOllamaProvider once the probe has SUCCEEDED."""

    def __init__(self, *, base_url=None, text_model="gemma3:1b", max_input_chars=None, **_kw):
        self.text_model = text_model
        self.base_url = base_url
        self.max_input_chars = max_input_chars
        self.calls = 0

    def create_chat_completion(self, messages, **_kw):
        self.calls += 1
        return _FakeCompletion("hola desde la máquina")


def _hosted_backend(monkeypatch):
    """A backend with a hosted provider configured, hosted text stubbed."""
    import TranslationBackend as tb

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    backend = tb.TranslationBackend()
    monkeypatch.setattr(
        backend, "translate_text",
        lambda text, target_language, **kw: "hola desde el servidor",
    )
    return tb, backend


def _ollama_unreachable(monkeypatch):
    """Every HTTP call the local probes make refuses, as with no Ollama running."""
    import urllib.request

    def refuse(url, *a, **kw):
        raise ConnectionRefusedError(f"[Errno 111] Connection refused: {url}")

    monkeypatch.setattr(urllib.request, "urlopen", refuse)


def _ollama_reachable(monkeypatch, tb, tag="gemma3:1b"):
    """The probe answers, a model is installed, and a provider can be built."""
    import urllib.request

    class _Resp:
        def read(self):
            return json.dumps({"models": [{"name": tag}]}).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(urllib.request, "urlopen", lambda url, *a, **kw: _Resp())
    monkeypatch.setattr(tb.TranslationBackend, "choose_local_model", lambda self: tag)
    monkeypatch.setattr(tb, "NativeOllamaProvider", _FakeLocalProvider)
    # The warm-up runs on a daemon thread and must never affect the answer.
    monkeypatch.setattr(tb.TranslationBackend, "_warm_live_provider_async", lambda self: None)


def test_no_ollama_live_translate_is_served_by_hosted_and_the_label_says_hosted(monkeypatch):
    tb, backend = _hosted_backend(monkeypatch)
    _ollama_unreachable(monkeypatch)

    translation, engine = backend.translate_live("hello there", "Spanish")

    assert translation == "hola desde el servidor"
    # THE LABEL is the assertion. A silent local→hosted fallback that lied here
    # would be indistinguishable from local never being installed.
    assert engine == f"hosted:{tb.TEXT_MODEL}"
    assert not engine.startswith("local")


def test_positive_control_reachable_ollama_produces_a_local_label(monkeypatch):
    """Proves the test above could have failed.

    Same harness, same call — the only difference is that the socket answers.
    If this said "hosted" too, the degradation assertion above would be worth
    nothing, because no configuration of this harness could ever run local.
    """
    tb, backend = _hosted_backend(monkeypatch)
    _ollama_reachable(monkeypatch, tb)

    translation, engine = backend.translate_live("hello there", "Spanish")

    assert engine == "local:gemma3:1b"
    assert translation == "hola desde la máquina"


def test_no_ollama_reports_an_empty_local_model_list_rather_than_guessing(monkeypatch):
    _tb, backend = _hosted_backend(monkeypatch)
    _ollama_unreachable(monkeypatch)

    assert backend.available_local_models() == []
    # And the comparison page offers no local engine it cannot deliver.
    labels = [c["engine"] for c in backend.comparison_candidates()]
    assert not any(label.startswith("local:") for label in labels)


def test_local_ollama_that_answers_with_nothing_falls_back_and_says_hosted(monkeypatch):
    """Reachable but useless is a different failure from absent, and must not
    be reported as a local translation."""
    tb, backend = _hosted_backend(monkeypatch)
    _ollama_reachable(monkeypatch, tb)
    monkeypatch.setattr(
        _FakeLocalProvider, "create_chat_completion",
        lambda self, messages, **kw: _FakeCompletion("   "),
    )

    translation, engine = backend.translate_live("hello there", "Spanish")

    assert engine == f"hosted:{tb.TEXT_MODEL}"
    assert translation == "hola desde el servidor"


# ═══════════════════════ 2. NO VOICE MODELS ═══════════════════════════════ #
#
# Detection is faked at the import and the voice directory — the two facts
# `local_voice` reads. Stubbing stt_available()/tts_available() would stub the
# mechanism under test.

def _voice_stack(monkeypatch, tmp_path, *, whisper=True, piper=True, voice=True):
    for name, present in (("faster_whisper", whisper), ("piper", piper)):
        if present:
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
        else:
            monkeypatch.setitem(sys.modules, name, None)  # import -> ImportError

    # Whisper readiness is weights-on-disk, not importability. The HF cache is
    # redirected at tmp_path in BOTH directions: pointing it away only when
    # simulating absence would leave "installed" inheriting the dev machine's
    # real weights, which is precisely why this file passed here and failed on
    # every CI runner.
    cache = tmp_path / "hf"
    monkeypatch.setenv("HF_HUB_CACHE", str(cache))
    monkeypatch.delenv("HF_HOME", raising=False)
    if whisper:
        folder = "models--" + local_voice.whisper_repo_id().replace("/", "--")
        snapshot = cache / folder / "snapshots" / "deadbeef"
        snapshot.mkdir(parents=True, exist_ok=True)
        (snapshot / "model.bin").write_bytes(b"stub")

    monkeypatch.setattr(local_voice, "VOICE_DIR", tmp_path)
    if voice:
        (tmp_path / "es_ES-davefx-medium.onnx").write_bytes(b"stub")
        (tmp_path / "es_ES-davefx-medium.onnx.json").write_text("{}", encoding="utf-8")
    local_voice._warned_settings.clear()


def _voice_backend(monkeypatch):
    import TranslationBackend as tb

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    backend = tb.TranslationBackend()
    monkeypatch.setattr(backend, "translate_text", lambda text, lang, **kw: "hola")
    monkeypatch.setattr(backend, "_transcribe_hosted", lambda data: "heard by hosted")
    backend._require_provider().synthesize_speech = lambda text: b"hosted-mp3"
    return tb, backend


def test_no_voice_models_uses_hosted_stt_and_tts_and_meta_says_hosted(monkeypatch, tmp_path):
    monkeypatch.delenv("PASSAGE_LOCAL_VOICE", raising=False)
    _voice_stack(monkeypatch, tmp_path, whisper=False, piper=False, voice=False)
    _tb, backend = _voice_backend(monkeypatch)

    source, _translated, audio, meta = backend.translate_audio(b"audio", "Spanish")

    assert source == "heard by hosted"
    assert meta["stt"] == "hosted" and meta["tts"] == "hosted"
    assert meta["media_type"] == "audio/mpeg" and audio == b"hosted-mp3"
    # And the user is TOLD, in the line the page renders.
    assert format_engine_line(meta) == "heard by hosted · spoken by hosted — sent out"


def test_positive_control_installed_voice_models_produce_local_labels(monkeypatch, tmp_path):
    """Proves the test above could have failed: same harness, models present."""
    monkeypatch.delenv("PASSAGE_LOCAL_VOICE", raising=False)
    _voice_stack(monkeypatch, tmp_path)
    tb, backend = _voice_backend(monkeypatch)
    monkeypatch.setattr(tb.local_voice, "transcribe", lambda data, **kw: "hello there")
    monkeypatch.setattr(tb.local_voice, "synthesize", lambda text, language: b"RIFFwav")

    _source, _translated, audio, meta = backend.translate_audio(b"audio", "Spanish")

    assert meta["stt"] == f"local:{local_voice.WHISPER_MODEL}"
    assert meta["tts"] == "local:piper"
    assert meta["media_type"] == "audio/wav" and audio == b"RIFFwav"
    assert "this machine" in format_engine_line(meta)


def test_missing_voice_for_the_target_language_keeps_stt_local_and_says_partly(monkeypatch, tmp_path):
    """Half-degradation must be reported as half, not rounded either way: the
    recording stayed on the machine even though the reply was synthesised
    elsewhere."""
    monkeypatch.delenv("PASSAGE_LOCAL_VOICE", raising=False)
    _voice_stack(monkeypatch, tmp_path)  # only the es_ES voice exists
    tb, backend = _voice_backend(monkeypatch)
    monkeypatch.setattr(tb.local_voice, "transcribe", lambda data, **kw: "hello there")

    def must_not_run(*a, **kw):
        raise AssertionError("piper synthesised German with no German voice installed")

    monkeypatch.setattr(tb.local_voice, "synthesize", must_not_run)

    _s, _t, _a, meta = backend.translate_audio(b"audio", "German")

    assert meta["stt"] == f"local:{local_voice.WHISPER_MODEL}"
    assert meta["tts"] == "hosted"
    assert "partly on this machine" in format_engine_line(meta)


def test_broken_local_voice_records_the_REASON_not_just_the_hosted_label(monkeypatch, tmp_path):
    """Installed-but-broken must be distinguishable from never-installed.

    Both end up on hosted; only the recorded reason tells them apart, which is
    the difference between "this user has no models" and "this user's models
    are failing and nobody noticed".
    """
    monkeypatch.delenv("PASSAGE_LOCAL_VOICE", raising=False)
    _voice_stack(monkeypatch, tmp_path)
    tb, backend = _voice_backend(monkeypatch)

    def boom(*a, **kw):
        raise RuntimeError("ctranslate2: cublas64_12.dll not found")

    monkeypatch.setattr(tb.local_voice, "transcribe", boom)
    monkeypatch.setattr(tb.local_voice, "synthesize", boom)

    _s, _t, _a, meta = backend.translate_audio(b"audio", "Spanish")

    assert meta["stt"] == "hosted" and meta["tts"] == "hosted"
    assert "cublas64_12.dll" in meta["stt_fallback"]
    assert "cublas64_12.dll" in meta["tts_fallback"]
    assert "local was tried and failed" in format_engine_line(meta)


def test_forced_local_voice_with_nothing_installed_admits_it_instead_of_pretending(monkeypatch, tmp_path):
    monkeypatch.setenv("PASSAGE_LOCAL_VOICE", "1")
    _voice_stack(monkeypatch, tmp_path, whisper=False, piper=False, voice=False)

    state = local_voice.status()

    assert state["forced_but_missing"] is True
    assert state["stt_engine"] == "hosted"
    assert state["stt_ready"] is False


# ═══════════════════════ 3. NO FONT AVAILABLE ═════════════════════════════ #
#
# The overlay IS the output, so a bitmap-default fallback (missing glyphs, tiny
# text) is a product failure, not a cosmetic one. It must still render — and it
# must be REPORTED.

_NO_FONTS = ("/definitely/not/a/font.ttf", "PassageNoSuchFace.ttf")


def _sample_image() -> bytes:
    img = Image.new("RGB", (400, 200), "white")
    ImageDraw.Draw(img).text((20, 20), "Padron 8.50", fill="black")
    out = BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


def test_no_font_available_still_renders_the_overlay(monkeypatch):
    compositor = ImageCompositor(OverlayStyle(font_family="PassageNoSuchFace.ttf"))
    monkeypatch.setattr(compositor, "_FONT_FALLBACKS", _NO_FONTS, raising=False)

    png = compositor.compose(
        _sample_image(),
        [{"bbox": [15, 15, 300, 90], "translated": "Padrón €8.50", "direction": "ltr"}],
    )

    assert png[:8] == b"\x89PNG\r\n\x1a\n" and len(png) > 500


def test_no_font_available_is_REPORTED_not_silent(monkeypatch, caplog):
    """The whole point of Phase D: the degradation announces itself."""
    compositor = ImageCompositor(OverlayStyle(font_family="PassageNoSuchFace.ttf"))
    monkeypatch.setattr(compositor, "_FONT_FALLBACKS", _NO_FONTS, raising=False)

    with caplog.at_level(logging.WARNING):
        compositor.compose(
            _sample_image(),
            [{"bbox": [15, 15, 300, 90], "translated": "Padrón €8.50", "direction": "ltr"}],
        )

    report = compositor.font_report()
    assert report["fallback"] is True
    assert report["resolved"] is None
    # Deliberately not pinned to the exact spelling of the fallback kind — only
    # that it is NOT the real face anyone asked for.
    assert report["kind"] != "truetype"
    assert set(_NO_FONTS).issubset(set(report["tried"]))
    # Reported to the operator too, not only to a caller who thinks to ask.
    warnings = [
        record.getMessage() for record in caplog.records
        if record.levelno >= logging.WARNING and "font" in record.getMessage().lower()
    ]
    assert warnings, "a silent font fallback is exactly the bug this test exists for"
    assert any("default" in message.lower() for message in warnings), warnings


def test_positive_control_a_resolvable_face_is_reported_as_truetype():
    """Proves the fallback report above could have said something else."""
    report = resolve_font(24)
    if report["fallback"]:
        pytest.skip(
            "no vector font resolved on this machine — that is itself the "
            f"finding; tried {report['tried']}"
        )
    assert report["kind"] == "truetype"
    assert report["resolved"] in report["tried"]


def test_font_report_is_internally_consistent_on_whatever_machine_runs_it():
    """Always runs, on every CI OS, and prints what this machine resolved.

    Deliberately asserts CONSISTENCY rather than a particular font: which faces
    exist is a fact about the runner, and hard-coding one would be the platform
    assumption §1 forbids. The printed line is the evidence.
    """
    report = image_compositor.probe_font(24)
    print(f"[font-probe] {report}")

    assert "font" not in report  # safe to hand to a diagnostics endpoint
    assert (report["resolved"] is None) is report["fallback"]
    assert (report["kind"] == "truetype") is not report["fallback"]
    assert report["tried"], "a report that tried nothing is not a probe"


# ═══════════════════════ 4. NO HOSTED API KEY ═════════════════════════════ #
#
# Regression guard for 9345d82 "Raise on translation failure instead of echoing
# source": the app used to hand back the SOURCE text as if it were translated,
# which is the worst possible failure mode — undetectable to anyone who doesn't
# read the target language.

def _keyless_backend(monkeypatch):
    import TranslationBackend as tb

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    backend = tb.TranslationBackend()
    assert backend.provider is None, "harness failed: a provider was configured anyway"
    return tb, backend


SOURCE = "the delivery arrives on Tuesday"


def test_no_api_key_fails_with_a_clear_message_naming_what_to_set(monkeypatch):
    _tb, backend = _keyless_backend(monkeypatch)

    with pytest.raises(RuntimeError) as raised:
        backend.translate_text(SOURCE, "Spanish")

    message = str(raised.value)
    assert "OPENAI_API_KEY" in message
    assert "provider" in message.lower()


def test_no_api_key_does_NOT_echo_the_source_text_as_a_translation(monkeypatch):
    """The 9345d82 regression, stated as the behaviour a user would see.

    Ollama is refused as well, deliberately: this machine HAS a local model
    installed, and without that line `translate_live` really does answer from
    it — correct behaviour, and it would have made this test pass for a reason
    that has nothing to do with the echo bug. (That is itself the §2.2 trap,
    caught here in reverse.)
    """
    _tb, backend = _keyless_backend(monkeypatch)
    _ollama_unreachable(monkeypatch)

    for call in (
        lambda: backend.translate_text(SOURCE, "Spanish"),
        lambda: backend.translate_live(SOURCE, "Spanish"),
        lambda: backend.translate_text_with_instructions(SOURCE, "Spanish", "be formal"),
    ):
        with pytest.raises(Exception) as raised:
            result = call()
            pytest.fail(f"returned {result!r} instead of failing without a key")
        assert SOURCE not in str(raised.value), "the source text leaked out as an answer"


def test_provider_failure_raises_rather_than_echoing_source(monkeypatch):
    """Same guarantee when a key EXISTS but the call fails — the shape the
    original bug had."""
    import TranslationBackend as tb

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    backend = tb.TranslationBackend()

    def boom(*a, **kw):
        raise RuntimeError("upstream 503")

    monkeypatch.setattr(backend, "_create_chat_completion_with_retry", boom)
    monkeypatch.setattr(backend._require_provider(), "create_chat_completion", boom)

    with pytest.raises(Exception):
        result = backend.translate_text(SOURCE, "Spanish")
        pytest.fail(f"returned {result!r} instead of raising")

    assert backend.translation_cache == {}, "a failed translation was cached"


def test_no_api_key_and_no_ollama_still_never_returns_the_source(monkeypatch):
    """Both engines gone. The floor: honest failure, not a fake translation."""
    _tb, backend = _keyless_backend(monkeypatch)
    _ollama_unreachable(monkeypatch)

    with pytest.raises(Exception):
        translation, engine = backend.translate_live(SOURCE, "Spanish")
        pytest.fail(f"returned {translation!r} labelled {engine!r} with no engine available")


# ═══════════════════════ 5. NON-SECURE CONTEXT ════════════════════════════ #
#
# Voice capture is browser-side, so the mechanism is JS. These tests EXECUTE
# the shipped `VOICE_PAGE_JS` string in node with a stubbed DOM, rather than
# grepping it for reassuring words — a string containing the right message is
# not evidence that the guard fires.
#
# What this does NOT verify (§4): real iOS Safari, real Android Chrome, OS
# permission dialogs, or actual microphone hardware. Node is not a browser.

_NODE = shutil.which("node")
_needs_node = pytest.mark.skipif(_NODE is None, reason="node is required to execute the page JS")

_HARNESS = r"""
const fs = require('fs'), vm = require('vm');
const cfg = JSON.parse(process.argv[3]);
const src = fs.readFileSync(process.argv[2], 'utf8');
const seen = { status: null, debug: null };
const el = {
  desktop_voice_start_recording: { disabled: false, style: {} },
  desktop_voice_stop_recording: { disabled: false, style: {} },
};
const listeners = {};
const add = (type, fn) => { (listeners[type] = listeners[type] || []).push(fn); };
const win = {
  isSecureContext: cfg.secure,
  location: { hostname: cfg.hostname, search: '' },
  addEventListener: add,
  voiceUx: {
    setStatus: (_s, m) => { seen.status = m; },
    setDebug: (_s, m) => { seen.debug = m; },
    setEngines: () => {},
    setRecordingButtons: () => {},
  },
};
if (cfg.webaudio) win.AudioContext = function () {};
const doc = { getElementById: id => el[id] || null, addEventListener: add, querySelectorAll: () => [] };
const nav = cfg.getusermedia ? { mediaDevices: { getUserMedia: async () => ({}) } } : {};
const sandbox = { window: win, document: doc, navigator: nav, console };
sandbox.window.window = win;
vm.createContext(sandbox);
vm.runInContext(src, sandbox);
(listeners['load'] || []).forEach(fn => fn());
console.log(JSON.stringify({
  status: seen.status,
  debug: seen.debug,
  startDisabled: el.desktop_voice_start_recording.disabled,
  stopDisabled: el.desktop_voice_stop_recording.disabled,
}));
"""


def _run_voice_page_js(tmp_path, **cfg) -> dict:
    """Load the REAL shipped page JS and fire its window load handler."""
    page = tmp_path / "voice_page.js"
    page.write_text(VOICE_PAGE_JS.replace("<script>", "").replace("</script>", ""), encoding="utf-8")
    harness = tmp_path / "harness.js"
    harness.write_text(_HARNESS, encoding="utf-8")
    settings = {"secure": True, "hostname": "passage.example.com",
                "webaudio": True, "getusermedia": True, **cfg}
    done = subprocess.run(
        [_NODE, str(harness), str(page), json.dumps(settings)],
        capture_output=True, text=True, timeout=60,
    )
    assert done.returncode == 0, f"page JS threw: {done.stderr[:800]}"
    return json.loads(done.stdout)


@_needs_node
def test_non_secure_context_disables_capture_and_explains_why(tmp_path):
    result = _run_voice_page_js(tmp_path, secure=False)

    assert result["startDisabled"] is True and result["stopDisabled"] is True
    # Honest AND actionable: names the condition and the remedy.
    assert "unavailable" in result["status"].lower()
    assert "HTTPS" in result["status"] and "localhost" in result["status"]
    # The readout names WHICH capability failed, so this is distinguishable
    # from a missing microphone or an unsupported browser.
    assert "secure=false" in result["debug"]
    assert "getUserMedia=true" in result["debug"]


@_needs_node
def test_positive_control_a_secure_context_enables_capture(tmp_path):
    """Proves the guard above is the thing that fired."""
    result = _run_voice_page_js(tmp_path, secure=True)

    assert result["startDisabled"] is False
    assert result["status"] == "Ready to record"
    assert "secure=true" in result["debug"]


@_needs_node
def test_plain_http_on_localhost_is_treated_as_secure(tmp_path):
    """Development over http://localhost must keep working — the guard is
    about the browser's secure-context rule, not about the scheme."""
    result = _run_voice_page_js(tmp_path, secure=False, hostname="localhost")

    assert result["startDisabled"] is False
    assert result["status"] == "Ready to record"


@_needs_node
def test_missing_getusermedia_degrades_with_its_own_explanation(tmp_path):
    result = _run_voice_page_js(tmp_path, getusermedia=False)

    assert result["startDisabled"] is True
    assert "unavailable" in result["status"].lower()
    assert "getUserMedia=false" in result["debug"]
    assert "secure=true" in result["debug"]  # not blamed on the wrong thing


@_needs_node
def test_missing_web_audio_degrades_with_its_own_explanation(tmp_path):
    result = _run_voice_page_js(tmp_path, webaudio=False)

    assert result["startDisabled"] is True
    assert "webAudio=false" in result["debug"]
