"""Tests for the Phase B diagnostic surface (PORTABILITY_PLAN.md §5).

The governing question for every test here, from §2.2 of the plan: *would this
fail if the local engine silently never ran?* A diagnostic is only worth having
if it distinguishes "the local path served this" from "the local path was absent
and hosted quietly covered for it", so each probe is exercised twice — once with
the capability present, once with it forced absent — and the absent case asserts
the field is PRESENT AND FALSE, never missing.

A missing field and a false field are different failures. A missing field makes a
broken machine look like a machine that was never asked; that is precisely the
bug this page exists to prevent, so it gets its own assertions.

These call the real construction path (``TranslationUI(backend=...)`` and the
real ``api_health`` coroutine). No ``__new__``, no assigning the attribute the
test is meant to be verifying.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from passage import diagnostics


# ───────────────────────────── platform ─────────────────────────────── #

def test_platform_report_states_os_arch_and_python():
    report = diagnostics.platform_report()
    for key in ("system", "release", "machine", "processor_arch",
                "python_version", "python_implementation", "in_docker"):
        assert key in report, f"{key} missing — a diagnostic must not be silent"
    assert report["python_version"].split(".")[0].isdigit()
    assert report["system"]  # non-empty on every supported host


# ───────────────────────────── local LLM ────────────────────────────── #

def test_local_llm_reports_models_and_the_chosen_one():
    report = diagnostics.local_llm_report(
        list_models=lambda: ["qwen2.5:7b", "llama3.2:3b"],
        choose_model=lambda: "qwen2.5:7b",
    )
    assert report["reachable"] is True
    assert report["models"] == ["qwen2.5:7b", "llama3.2:3b"]
    assert report["model_count"] == 2
    assert report["chosen_model"] == "qwen2.5:7b"
    assert report["error"] is None


def test_local_llm_absent_says_absent_rather_than_omitting():
    """No Ollama. The app would silently serve every request from hosted and
    look perfect; the diagnostic has to say so in words."""
    report = diagnostics.local_llm_report(
        list_models=lambda: [], choose_model=lambda: None)
    assert "reachable" in report and "models" in report and "chosen_model" in report
    assert report["reachable"] is False
    assert report["models"] == []
    assert report["model_count"] == 0
    assert report["chosen_model"] is None


def test_local_llm_probe_failure_is_reported_not_swallowed():
    def boom():
        raise OSError("connection refused")

    report = diagnostics.local_llm_report(list_models=boom)
    assert report["reachable"] is False
    assert "refused" in (report["error"] or "")


def test_local_llm_distinguishes_a_timeout_from_a_refusal():
    """Both give an empty model list. §2.3: on a loaded machine the 0.6 s probe
    produces a false negative indistinguishable from 'nothing installed' — so
    the outcome and the elapsed time have to reach the page."""
    timed_out = diagnostics.local_llm_report(probe=lambda: {
        "reachable": False, "outcome": "timeout", "models": [],
        "endpoint": "http://localhost:11434/api/tags",
        "timeout_seconds": 0.6, "elapsed_ms": 601, "detail": "timed out",
    })
    refused = diagnostics.local_llm_report(probe=lambda: {
        "reachable": False, "outcome": "refused", "models": [],
        "endpoint": "http://localhost:11434/api/tags",
        "timeout_seconds": 0.6, "elapsed_ms": 2, "detail": "connection refused",
    })
    assert timed_out["outcome"] == "timeout"
    assert timed_out["timeout_seconds"] == 0.6
    assert timed_out["elapsed_ms"] == 601
    assert refused["outcome"] == "refused"
    assert timed_out["outcome"] != refused["outcome"]


def test_local_llm_endpoint_never_leaks_embedded_credentials():
    report = diagnostics.local_llm_report(probe=lambda: {
        "reachable": True, "outcome": "ok", "models": ["x"],
        "endpoint": "http://user:supersecret@ollama.internal:11434/api/tags",
    })
    assert "supersecret" not in report["endpoint"]
    assert "ollama.internal" in report["endpoint"]


def test_local_llm_auto_wires_the_backends_shipped_probe():
    """No injection: the real probe runs and reports its outcome either way."""
    report = diagnostics.local_llm_report()
    assert report["outcome"] is not None
    assert report["endpoint"]
    assert isinstance(report["reachable"], bool)


def test_a_probe_still_in_flight_is_not_reported_as_unreachable():
    """"We have not asked yet" and "we asked and nothing answered" are
    different facts. bool() collapsed the first into the second, so a page
    rendered "reachable: False" for the whole probe budget against an endpoint
    nothing had yet tried to reach."""
    report = diagnostics.local_llm_report(probe=lambda: {
        "reachable": None, "outcome": "probing", "models": [],
        "endpoint": None, "timeout_seconds": None, "elapsed_ms": None,
        "detail": "probe in flight",
    })
    assert report["reachable"] is None, "an unasked probe was called unreachable"
    assert report["outcome"] == "probing"
    text = diagnostics.format_text({"local_llm": report})
    assert "ollama reachable: unknown (not asked yet)" in text, text
    assert "ollama reachable: False" not in text


def test_a_landed_negative_probe_is_still_a_hard_False():
    """The other side of the same invariant: tri-state must not soften a real
    negative into a shrug."""
    report = diagnostics.local_llm_report(probe=lambda: {
        "reachable": False, "outcome": "refused", "models": [],
    })
    assert report["reachable"] is False
    assert "ollama reachable: False" in diagnostics.format_text(
        {"local_llm": report})


def test_format_text_carries_the_probe_age_and_cached_flag():
    """The COPY block is the artifact designed to leave the machine. The JSON
    and /engines carried age_seconds/cached; the pasteable block did not, and
    an undated cache turns a diagnostic into a confident guess."""
    text = diagnostics.format_text({"local_llm": {
        "reachable": True, "outcome": "ok", "elapsed_ms": 12,
        "timeout_seconds": 2.5, "age_seconds": 55.177, "cached": True,
        "ttl_seconds": 60.0,
    }})
    assert "55.2s old" in text, text
    assert "cached: True" in text, text
    assert "60.0 s" in text, text


def test_format_text_marks_a_fresh_probe_as_not_cached():
    text = diagnostics.format_text({"local_llm": {
        "reachable": True, "age_seconds": 0.004, "cached": False,
        "ttl_seconds": 60.0,
    }})
    assert "0.0s old" in text and "cached: False" in text, text


def test_format_text_dates_an_unstamped_report_as_taken_inline():
    """collect() with no snapshot really did probe just now — say that rather
    than printing nothing, which reads as 'age unknown'."""
    text = diagnostics.format_text(diagnostics.collect(None))
    assert "probe freshness: taken inline for this report" in text, text


def test_local_llm_with_no_probe_at_all_still_reports_the_field(monkeypatch):
    """A degenerate probe returning nothing must still yield every key."""
    report = diagnostics.local_llm_report(probe=lambda: {})
    assert report["reachable"] is False
    assert report["models"] == []
    assert report["chosen_model"] is None
    assert "outcome" in report


# ────────────────────────────── speech ──────────────────────────────── #

def _voice_status(**overrides):
    base = {
        "mode": "auto", "setting": "", "unrecognised_setting": None,
        "enabled": True, "stt_ready": True, "stt_engine": "local:base",
        "piper_ready": True, "voices": ["en_US-amy-medium"],
        "forced_but_missing": False,
    }
    base.update(overrides)
    return base


def test_speech_report_names_the_engine_that_would_serve():
    report = diagnostics.speech_report(status_fn=lambda: _voice_status())
    assert report["available"] is True
    assert report["stt_ready"] is True
    assert report["stt_engine"] == "local:base"
    assert report["voices"] == ["en_US-amy-medium"]
    assert report["voice_count"] == 1


def test_speech_absent_reports_hosted_engine_and_empty_voice_list():
    """The trap in full: with no voice models, hosted serves and everything
    works. The one visible difference is the engine label — assert on it."""
    report = diagnostics.speech_report(status_fn=lambda: _voice_status(
        stt_ready=False, stt_engine="hosted", piper_ready=False,
        voices=[], enabled=False,
    ))
    for key in ("available", "stt_ready", "tts_ready", "voices",
                "voice_count", "stt_engine", "forced_but_missing"):
        assert key in report, f"{key} missing — absence must be stated"
    assert report["available"] is False
    assert report["stt_ready"] is False
    assert report["tts_ready"] is False
    assert report["stt_engine"] == "hosted"
    assert report["voices"] == []
    assert report["voice_count"] == 0


def test_speech_forced_on_but_missing_surfaces_as_a_flag():
    report = diagnostics.speech_report(status_fn=lambda: _voice_status(
        mode="on", stt_ready=False, piper_ready=False, voices=[],
        forced_but_missing=True,
    ))
    assert report["forced_but_missing"] is True
    assert report["available"] is False


def test_speech_probe_failure_is_reported():
    def boom():
        raise RuntimeError("voice dir exploded")

    report = diagnostics.speech_report(status_fn=boom)
    assert report["available"] is False
    assert "exploded" in (report["error"] or "")


def test_speech_report_matches_the_real_local_voice_probe():
    """Not a stub: run the shipped probe and assert the two agree, so this page
    cannot drift from /engines the way the two voice lines once did."""
    from passage import local_voice

    real = diagnostics.speech_report()
    assert real["stt_installed"] == local_voice.whisper_installed()
    assert real["tts_installed"] == local_voice.piper_installed()
    assert real["voices"] == local_voice.status()["voices"]


# ─────────────────────────────── font ───────────────────────────────── #

def test_font_report_states_requested_resolved_and_fallback_flags():
    report = diagnostics.font_report()
    for key in ("requested", "resolved", "fell_back", "bitmap_fallback"):
        assert key in report
    assert report["error"] is None
    assert isinstance(report["fell_back"], bool)
    assert isinstance(report["bitmap_fallback"], bool)
    assert report["resolved"]


def test_font_report_flags_the_bitmap_fallback_when_no_face_loads(monkeypatch):
    """The exact shape of the shipped bug: PIL finds nothing, silently uses
    PIL's own default face, and the overlay still 'succeeds'. Forced through
    the REAL image_compositor.probe_font — the renderer's own resolution —
    because a second resolution here could report a face it would never load.
    """
    import image_compositor as ic
    from PIL import ImageFont

    real_truetype = ImageFont.truetype

    def no_named_face(name, size=None, **kwargs):
        # Simulate a host with none of the candidate faces installed. PIL's own
        # bundled default face still loads through this same function on
        # Pillow >= 10, so only the candidates are refused.
        if name in ic.FONT_CANDIDATES:
            raise OSError("cannot open resource")
        return real_truetype(name, size=size, **kwargs)

    monkeypatch.setattr(ImageFont, "truetype", no_named_face)

    report = diagnostics.font_report()
    assert report["error"] is None
    assert report["probe_source"] == "image_compositor.probe_font"
    assert report["bitmap_fallback"] is True
    assert report["fell_back"] is True
    assert report["resolved"] == "PIL.ImageFont.load_default"


def test_font_report_says_it_did_not_fall_back_when_the_request_resolves(monkeypatch):
    import image_compositor as ic
    from PIL import ImageFont

    monkeypatch.setattr(ImageFont, "truetype", lambda *a, **k: object())

    report = diagnostics.font_report()
    assert report["probe_source"] == "image_compositor.probe_font"
    assert report["fell_back"] is False
    assert report["bitmap_fallback"] is False
    assert report["resolved"] == ic.OverlayStyle().font_family


def test_font_report_flags_a_fallback_face_as_a_fallback(monkeypatch):
    """The requested family fails and a later candidate wins. It works — and
    the page must still say the request was not honoured."""
    import image_compositor as ic
    from PIL import ImageFont

    requested = ic.OverlayStyle().font_family

    def only_the_second(name, size=None, **kwargs):
        if name == requested:
            raise OSError("cannot open resource")
        return object()

    monkeypatch.setattr(ImageFont, "truetype", only_the_second)
    report = diagnostics.font_report()
    assert report["fell_back"] is True
    assert report["bitmap_fallback"] is False
    assert report["resolved"] != requested


def test_font_report_refuses_to_resolve_the_font_a_second_way(monkeypatch):
    """If the compositor's probe disappears, the diagnostic must say so rather
    than fall back to a private resolution order. Two font resolutions that can
    disagree is the bug class this page exists to remove."""
    import image_compositor as ic

    monkeypatch.delattr(ic, "probe_font", raising=False)
    monkeypatch.delattr(ic, "resolve_overlay_font", raising=False)

    report = diagnostics.font_report()
    assert report["probe_source"] == "unavailable"
    assert "probe_font" in (report["error"] or "")
    assert report["resolved"] is None


def test_font_report_uses_the_compositors_own_probe_not_a_second_copy():
    """image_compositor.probe_font is the resolution the RENDERER performs.
    Resolving fonts a second way here could report a face the overlay would
    never use, so the shipped probe must be the source when it exists."""
    import image_compositor as ic

    assert callable(getattr(ic, "probe_font", None)), (
        "image_compositor.probe_font vanished; /diagnostics would silently fall "
        "back to its own copy of the resolution order")
    report = diagnostics.font_report()
    assert report["probe_source"] == "image_compositor.probe_font"
    truth = ic.probe_font()
    assert report["bitmap_fallback"] is bool(truth["fallback"])
    assert report["candidates_tried"] == len(truth["tried"])


def test_font_report_translates_a_bitmap_probe_result_into_fallback_flags(monkeypatch):
    import image_compositor as ic

    monkeypatch.setattr(ic, "probe_font", lambda size=24: {
        "resolved": None, "fallback": True, "kind": "bitmap_default",
        "tried": ["DejaVuSans.ttf", "arial.ttf"], "size": size,
    })
    report = diagnostics.font_report()
    assert report["bitmap_fallback"] is True
    assert report["fell_back"] is True
    assert report["resolved"] == "PIL.ImageFont.load_default"
    assert report["candidates_tried"] == 2


def test_font_report_flags_a_non_default_face_from_the_probe_as_a_fallback(monkeypatch):
    import image_compositor as ic

    monkeypatch.setattr(ic, "probe_font", lambda size=24: {
        "resolved": "C:\\Windows\\Fonts\\arial.ttf", "fallback": False,
        "kind": "truetype", "tried": ["DejaVuSans.ttf", "C:\\Windows\\Fonts\\arial.ttf"],
        "size": size,
    })
    report = diagnostics.font_report()
    assert report["resolved"] == "arial.ttf"      # redacted to a basename
    assert report["requested"] == "DejaVuSans.ttf"
    assert report["fell_back"] is True            # the request was not honoured
    assert report["bitmap_fallback"] is False


# ──────────────────────────── path hygiene ──────────────────────────── #

def test_paths_never_expose_the_home_directory():
    home = Path.home()
    redacted = diagnostics.redact_path(home / "secretuser_docs" / "font.ttf")
    assert str(home) not in redacted
    assert redacted.startswith("~/")
    assert diagnostics.redact_path(r"C:\\Windows\\Fonts\\arial.ttf") == "arial.ttf"
    assert diagnostics.redact_path(None) is None


# The CI matrix's first run turned this from theory into a recorded fact:
# windows-latest passed and both ubuntu-latest and macos-14 failed, because
# redaction was reducing a path to its basename with HOST separator semantics.
# On POSIX a Windows path contains no recognised separator, so nothing was
# stripped and the full path leaked into a report meant to be pasted in public.
# These tests pin the property that killed it: the answer is a function of the
# INPUT ONLY, never of the machine running the test.

WINDOWS_STYLE = [
    (r"C:\Windows\Fonts\arial.ttf", "arial.ttf"),
    (r"\\fileserver\share\fonts\DejaVuSans.ttf", "DejaVuSans.ttf"),
    (r"C:\Users\david.neish\AppData\Local\x.ttf", "x.ttf"),
    (r"fonts\arial.ttf", "arial.ttf"),
]
POSIX_STYLE = [
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "DejaVuSans.ttf"),
    ("/System/Library/Fonts/Supplemental/Arial.ttf", "Arial.ttf"),
    ("/home/david/.fonts/secret_project.ttf", "secret_project.ttf"),
    ("fonts/arial.ttf", "arial.ttf"),
]
MIXED_STYLE = [
    (r"C:/Windows\Fonts/arial.ttf", "arial.ttf"),
    (r"/usr/share\fonts/arial.ttf", "arial.ttf"),
    (r"C:\\Windows\\Fonts\\arial.ttf", "arial.ttf"),
]


@pytest.mark.parametrize("raw,expected", WINDOWS_STYLE + POSIX_STYLE + MIXED_STYLE)
def test_redaction_is_identical_whichever_os_style_it_is_handed(raw, expected):
    """Every one of these must hold on Windows, Linux and macOS alike. The
    ubuntu/macos CI failure was precisely a Windows-style input surviving
    unredacted on a POSIX host."""
    assert diagnostics.redact_path(raw) == expected


@pytest.mark.parametrize("raw,expected", WINDOWS_STYLE + POSIX_STYLE + MIXED_STYLE)
def test_redaction_never_returns_a_path_shaped_value(raw, expected):
    """The failure mode was a leak, not a wrong basename: assert the ESCAPE,
    not just the happy value. Nothing separator-shaped may survive."""
    out = diagnostics.redact_path(raw)
    assert "/" not in out and "\\" not in out
    assert ":" not in out            # no drive letter, no UNC remnant
    assert "david" not in out.casefold()


def test_redaction_does_not_consult_the_host_separator(monkeypatch):
    """A prompt is not a mechanism, and neither is a comment: prove the code is
    separator-agnostic by lying about the host separator and demanding the same
    answers. If anything still branched on os.sep / os.path semantics, one of
    the two styles would change its answer here."""
    monkeypatch.setattr(os, "sep", "/")
    monkeypatch.setattr(os, "altsep", None)
    assert diagnostics.redact_path(r"C:\Windows\Fonts\arial.ttf") == "arial.ttf"
    assert diagnostics.redact_path("/usr/share/fonts/arial.ttf") == "arial.ttf"
    monkeypatch.setattr(os, "sep", "\\")
    monkeypatch.setattr(os, "altsep", "/")
    assert diagnostics.redact_path(r"C:\Windows\Fonts\arial.ttf") == "arial.ttf"
    assert diagnostics.redact_path("/usr/share/fonts/arial.ttf") == "arial.ttf"


def test_a_bare_basename_is_left_alone():
    assert diagnostics.redact_path("DejaVuSans.ttf") == "DejaVuSans.ttf"
    assert diagnostics.redact_path("Arial Black.ttf") == "Arial Black.ttf"


def test_empty_and_rootlike_inputs_are_none_not_a_crash():
    for raw in (None, "", "/", "\\", "//", "."):
        assert diagnostics.redact_path(raw) is None


@pytest.mark.parametrize("style", ["native", "posix", "windows"])
def test_a_home_path_becomes_tilde_in_either_path_style(style, monkeypatch):
    """The home directory is the secret being protected, so it must collapse to
    '~' whether the path arrives with '/' or '\\' — a Windows app can hand the
    diagnostic either."""
    fake_home = Path("C:\\Users\\testuser") if style == "windows" else Path("/home/testuser")
    monkeypatch.setattr(diagnostics.Path, "home", classmethod(lambda cls: fake_home))
    raws = {
        "native": str(fake_home / "fonts" / "x.ttf"),
        "posix": "/home/testuser/fonts/x.ttf",
        "windows": r"C:\Users\testuser\fonts\x.ttf",
    }
    out = diagnostics.redact_path(raws[style])
    assert out == "~/fonts/x.ttf"
    assert "testuser" not in out


def test_home_itself_redacts_to_tilde_and_leaks_nothing():
    assert diagnostics.redact_path(Path.home()) == "~"
    assert Path.home().name not in diagnostics.redact_path(Path.home())


def test_pathlike_and_str_inputs_agree():
    p = Path.home() / "fonts" / "x.ttf"
    assert diagnostics.redact_path(p) == diagnostics.redact_path(str(p))


# ───────────────────── the font actually in use ─────────────────────── #

def test_font_report_asks_about_the_face_the_overlay_would_actually_use():
    """The renderer resolves with the USER's font_family (ImageCompositor._font
    passes preferred=self.style.font_family). A diagnostic that always probes
    the default face reports a resolution the overlay never performed — the
    exact bug class this page exists to remove. So when a face is named, either
    the probe honours it or the report says it could not."""
    import image_compositor as ic

    report = diagnostics.font_report(preferred="Arial Black.ttf")
    assert report["requested"] == "Arial Black.ttf"
    if diagnostics._accepts_preferred(ic.probe_font):
        truth = ic.probe_font(24, preferred="Arial Black.ttf")
        assert report["resolved"] == (
            diagnostics.redact_path(truth["resolved"]) or "PIL.ImageFont.load_default")
    else:
        # Honest refusal, not a fabricated answer.
        assert report["resolved"] is None
        assert "preferred" in (report["error"] or "")


def test_font_report_forwards_the_preferred_face_when_the_probe_accepts_it(monkeypatch):
    """Mechanism, not intention: prove the preference reaches the resolver."""
    import image_compositor as ic
    seen = {}

    def fake_probe(size=24, *, preferred=None):
        seen["preferred"] = preferred
        return {"resolved": r"C:\Windows\Fonts\ariblk.ttf", "fallback": False,
                "kind": "truetype", "tried": [preferred], "size": size}

    monkeypatch.setattr(ic, "probe_font", fake_probe)
    report = diagnostics.font_report(preferred="Arial Black.ttf")
    assert seen["preferred"] == "Arial Black.ttf"
    assert report["requested"] == "Arial Black.ttf"
    assert report["resolved"] == "ariblk.ttf"
    assert report["fell_back"] is True     # ariblk.ttf is not what was asked for


def test_font_report_refuses_to_answer_for_a_face_the_probe_cannot_resolve(monkeypatch):
    """When probe_font takes only `size`, reporting its unqualified answer under
    the user's requested name would claim a resolution nobody ran."""
    import image_compositor as ic

    monkeypatch.setattr(ic, "probe_font", lambda size=24: {
        "resolved": "DejaVuSans.ttf", "fallback": False, "kind": "truetype",
        "tried": ["DejaVuSans.ttf"], "size": size})
    report = diagnostics.font_report(preferred="Arial Black.ttf")
    assert report["requested"] == "Arial Black.ttf"
    assert report["resolved"] is None
    assert "DejaVuSans" not in json.dumps(report)
    assert report["error"]


def test_font_report_without_a_preference_is_unchanged(monkeypatch):
    import image_compositor as ic

    report = diagnostics.font_report()
    assert report["probe_source"] == "image_compositor.probe_font"
    assert report["error"] is None


def test_collect_carries_a_backends_configured_face_into_the_font_report():
    class Backend:
        overlay_font_family = "Arial Black.ttf"

    assert diagnostics.collect(Backend())["font"]["requested"] == "Arial Black.ttf"


def test_full_report_leaks_no_secrets(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-never-appear")
    blob = json.dumps(diagnostics.collect(None))
    assert "sk-should-never-appear" not in blob
    assert str(Path.home()) not in blob


# ────────────────────────────── browser ─────────────────────────────── #

def test_browser_section_exists_before_a_browser_reports():
    """curl of /api/health must be able to tell 'no browser reported' from
    'the browser reported nothing'."""
    browser = diagnostics.browser_placeholder()
    assert browser["collected"] is False
    for key in ("secure_context", "get_user_media", "media_recorder",
                "audio_context_sample_rate", "mime_types"):
        assert key in browser


def test_browser_probe_observes_the_granted_rate_rather_than_asserting_24k():
    """§2.4: 'Safari/iOS returns the hardware rate' was INFERRED from a Chrome
    error message and never observed. The probe must read ctx.sampleRate back,
    not hardcode a claim about it."""
    js = diagnostics.BROWSER_PROBE_JS
    assert "ctx.sampleRate" in js
    assert "audio_context_rate_honoured" in js
    assert "isSecureContext" in js
    assert "getUserMedia" in js
    assert "MediaRecorder.isTypeSupported" in js


def test_format_text_reports_browser_facts_once_collected():
    report = diagnostics.collect(None)
    report["browser"] = {
        "collected": True, "secure_context": False, "get_user_media": False,
        "media_recorder": True, "audio_context_sample_rate": 48000,
        "mime_types": {"audio/webm": False, "audio/mp4": True},
    }
    text = diagnostics.format_text(report)
    assert "secure context: False" in text
    assert "48000" in text
    assert "audio/mp4=True" in text


def test_format_text_says_the_browser_half_is_missing_when_it_is():
    text = diagnostics.format_text(diagnostics.collect(None))
    assert "browser: not collected" in text
    # Absence of a capability is stated, not omitted.
    assert "ollama reachable:" in text
    assert "chosen local model:" in text
    assert "font resolved:" in text


# ─────────────────────────── the /api/health route ──────────────────── #

class _StubBackend:
    """Stands in for TranslationBackend at the seam the route actually uses."""

    def __init__(self, models, chosen, provider=object(), outcome="ok"):
        self._models, self._chosen, self.provider = models, chosen, provider
        self._outcome = outcome

    def probe_local_llm(self):
        return {
            "reachable": bool(self._models), "outcome": self._outcome,
            "models": list(self._models),
            "endpoint": "http://localhost:11434/api/tags",
            "timeout_seconds": 0.6, "elapsed_ms": 3, "detail": "",
        }

    def available_local_models(self):
        return list(self._models)

    def choose_local_model(self):
        return self._chosen


def _health(backend):
    from TranslationUI import TranslationUI

    # Real construction path — the same call start_ui() makes.
    ui_app = TranslationUI(backend=backend)
    response = asyncio.run(ui_app.api_health(None))
    return json.loads(bytes(response.body).decode("utf-8"))


def test_health_carries_the_full_diagnostic_and_keeps_its_original_keys():
    payload = _health(_StubBackend(["qwen2.5:7b"], "qwen2.5:7b"))
    # Original contract preserved: something health-checks this route.
    for key in ("status", "version", "hosted_provider", "local_models", "local_default"):
        assert key in payload
    assert payload["status"] == "ok"
    report = payload["diagnostics"]
    for section in diagnostics.SECTIONS:
        assert section in report, f"{section} missing from /api/health"
    assert report["local_llm"]["chosen_model"] == "qwen2.5:7b"
    assert report["local_llm"]["reachable"] is True


def test_health_with_no_local_model_says_so_instead_of_dropping_the_section():
    payload = _health(_StubBackend([], None, provider=None, outcome="refused"))
    report = payload["diagnostics"]
    assert "local_llm" in report
    assert report["local_llm"]["reachable"] is False
    assert report["local_llm"]["chosen_model"] is None
    assert report["local_llm"]["outcome"] == "refused"   # why, not just "no"
    assert payload["hosted_provider"] is False
    # And the browser half is still declared, not silently absent.
    assert report["browser"]["collected"] is False


def test_diagnostics_page_is_registered_on_the_app():
    """The route, not just the method: an earlier health endpoint was 'added'
    in a commit whose diff never registered it, so it 404'd and everything
    checking the deploy was checking nothing."""
    import inspect

    import TranslationUI as module

    source = inspect.getsource(module.start_ui)
    assert 'ui.page("/diagnostics")' in source
    assert hasattr(module.TranslationUI, "diagnostics_page")
