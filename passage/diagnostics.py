"""What THIS machine actually resolved — the answer to "does it work on an M1?".

PORTABILITY_PLAN.md §5 Phase B. The point is not to predict other hardware; it
is to make the system self-describing so the other machine answers for itself in
one page visit. Everything here is a runtime capability probe, never a platform
guess: §1 of the plan is explicit that the code must not ask "am I on a Mac?".

Two rules shape every field below.

1. **Absent is a value, never a missing key.** §2.2: on a machine with no Ollama
   every local path silently falls back to hosted and the app looks perfect. A
   diagnostic that simply omits ``local`` when Ollama is down is indistinguishable
   from one that never probed. So every probe reports its failure in place, with
   ``available: False`` and a reason.

2. **Fallbacks are reported as fallbacks.** The overlay font bug (see
   ``image_compositor._FONT_FALLBACKS``) was invisible precisely because falling
   back to a bitmap face is silent and still "succeeds". ``font.fell_back`` and
   ``font.bitmap_fallback`` exist so that class of bug shows up as a fact on a
   page instead of as mojibake in someone's output.

Nothing here emits API keys, tokens, or a path that would expose a user's home
directory: filesystem paths are reduced to a basename, or to a ``~``-relative
form when they live under the home directory.
"""

from __future__ import annotations

import os
import platform
import re
from inspect import signature
from pathlib import Path
from typing import Any, Callable

SCHEMA_VERSION = 1

#: The keys /diagnostics and /api/health always carry. Asserted by the tests:
#: a field that disappears when a capability is missing is its own bug class.
SECTIONS = ("platform", "local_llm", "speech", "font", "browser")


# ──────────────────────────── path hygiene ──────────────────────────── #

#: Both separators, always, on every host. ``pathlib``/``os.path`` understand
#: only the separator of the machine they run on, which made this function
#: silently host-dependent: on POSIX, ``Path(r"C:\\Windows\\Fonts\\arial.ttf").name``
#: is the ENTIRE string, so the "redacted" value leaked the whole path. The CI
#: matrix caught exactly that on its first run. The fix is not to branch on the
#: host OS — that is the assumption this phase exists to delete — it is to stop
#: asking the host and parse both styles unconditionally.
_SEPARATORS = re.compile(r"[\\/]+")


def _path_components(text: str) -> list[str]:
    """Split on ``/`` and ``\\`` alike, dropping empties and no-op ``.`` parts.

    Separator-agnostic by construction: a Windows path redacts identically on
    Linux, and a POSIX path redacts identically on Windows.
    """
    return [part for part in _SEPARATORS.split(text) if part and part != "."]


def redact_path(raw: str | os.PathLike | None) -> str | None:
    """A path safe to paste into a bug report, on any host, in any path style.

    A diagnostic is meant to be copied out of a browser and pasted somewhere
    public, so ``C:\\Users\\david.neish\\...`` must never appear in it. Paths
    under the home directory become ``~/…``; anything else is reduced to its
    final component, which is all a reader needs to tell arial.ttf from
    DejaVuSans.ttf.

    The result depends only on the input, never on ``os.sep``. Home matching is
    case-insensitive so ``C:/Users/David/x`` still collapses to ``~/x`` on
    Windows, where the same directory is spelled several ways; a false match
    can only ever redact more, never less.
    """
    if raw is None:
        return None
    text = str(raw)
    if not text:
        return None
    parts = _path_components(text)
    if not parts:
        # A bare root ("/", "\\", "C:" is not root-only) carries no name and no
        # secret; there is nothing safe left to say about it.
        return None
    try:
        home_parts = _path_components(str(Path.home()))
    except Exception:
        home_parts = []
    if home_parts:
        head = [part.casefold() for part in parts[:len(home_parts)]]
        if head == [part.casefold() for part in home_parts]:
            rest = parts[len(home_parts):]
            return "~/" + "/".join(rest) if rest else "~"
    return parts[-1]


# ────────────────────────────── platform ────────────────────────────── #

def platform_report() -> dict[str, Any]:
    """OS / arch / Python. Descriptive only — nothing branches on it.

    Recorded because it is the context a reader needs for every other section
    (an arm64 Darwin row is what makes a missing CUDA line unsurprising), not
    because the app decides anything from it.
    """
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor_arch": platform.architecture()[0],
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "in_docker": Path("/.dockerenv").exists(),
    }


# ────────────────────────────── local LLM ───────────────────────────── #

def redact_url(raw: str | None) -> str | None:
    """A URL with any embedded credentials stripped.

    OLLAMA_BASE_URL is normally http://localhost:11434, but it is an env var and
    an env var can hold ``http://user:token@host``. A diagnostic meant to be
    pasted into a bug report must not be the thing that leaks it.
    """
    if not raw:
        return raw
    if "@" in raw and "//" in raw:
        scheme, _, rest = raw.partition("//")
        return f"{scheme}//…@{rest.rpartition('@')[2]}"
    return raw


def local_llm_report(
    list_models: Callable[[], list[str]] | None = None,
    choose_model: Callable[[], str | None] | None = None,
    probe: Callable[[], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Ollama reachability, the models installed, and the one that would run.

    ``reachable`` is derived from the model listing rather than probed
    separately, because the listing is exactly what the app itself uses
    (§2.6: probe what the app uses, not a generic notion of capability).

    ``reachable`` is tri-state: True, False, or ``None`` for "not asked yet"
    (a probe still in flight, which reports ``outcome: probing``). None is
    never coerced to False — see the update block below.

    An empty list is reported as ``reachable: False`` with ``models: []`` —
    never as an absent section. The plan's §2.3 warning applies here: a 0.6 s
    probe timeout on a loaded machine yields a false negative that looks
    identical to "nothing installed", so the reader is told the probe ran and
    came back empty rather than being left to guess.
    """
    report: dict[str, Any] = {
        "reachable": False,
        "models": [],
        "model_count": 0,
        "chosen_model": None,
        # From TranslationBackend.probe_local_llm when available: a TIMEOUT and
        # a REFUSED both produce an empty model list, and §2.3 says those are
        # different bugs — a 0.6 s probe on a loaded machine is a false negative
        # that looks exactly like "nothing installed".
        "outcome": None,
        "endpoint": None,
        "timeout_seconds": None,
        "elapsed_ms": None,
        "detail": "",
        "error": None,
    }
    if probe is None and list_models is None:
        try:
            from TranslationBackend import probe_local_llm as probe  # type: ignore
        except Exception:
            probe = None
    if probe is not None:
        try:
            raw = probe() or {}
        except Exception as error:
            report["error"] = str(error)[:200]
            return report
        models = [m for m in (raw.get("models") or [])
                  if "embed" not in m]
        # "we have not asked yet" (reachable=None, outcome "probing") is NOT
        # the same fact as "we asked and nothing answered" (reachable=False),
        # and bool() used to collapse them — asserting unreachability for the
        # whole probe budget before a single packet had left the machine. A
        # probe dict with no ``reachable`` key at all is still False: that is
        # a degenerate answer, not an unasked question.
        raw_reachable = raw["reachable"] if "reachable" in raw else False
        report.update({
            "reachable": None if raw_reachable is None else bool(raw_reachable),
            "models": models,
            "model_count": len(models),
            "outcome": raw.get("outcome"),
            "endpoint": redact_url(raw.get("endpoint")),
            "timeout_seconds": raw.get("timeout_seconds"),
            "elapsed_ms": raw.get("elapsed_ms"),
            "detail": (raw.get("detail") or "")[:200],
        })
        if choose_model is not None:
            try:
                report["chosen_model"] = choose_model()
            except Exception as error:
                report["error"] = str(error)[:200]
        return report
    if list_models is None:
        report["error"] = "no probe available"
        return report
    try:
        models = list(list_models() or [])
    except Exception as error:  # a dead endpoint must not 500 the diagnostic
        report["error"] = str(error)[:200]
        return report
    report["models"] = models
    report["model_count"] = len(models)
    report["reachable"] = bool(models)
    if choose_model is not None:
        try:
            report["chosen_model"] = choose_model()
        except Exception as error:
            report["error"] = str(error)[:200]
    return report


# ─────────────────────────────── speech ─────────────────────────────── #

def speech_report(status_fn: Callable[[], dict] | None = None) -> dict[str, Any]:
    """Local speech stack: what imported, what is on disk, what would serve.

    Derived from ``passage.local_voice.status()`` so this page cannot
    contradict /engines — that module's docstring records what happened the
    last time two surfaces derived the same claim from two different probes.
    """
    report: dict[str, Any] = {
        "available": False,
        "mode": None,
        "setting": None,
        "unrecognised_setting": None,
        "stt_installed": False,
        "stt_ready": False,
        "stt_engine": "hosted",
        "tts_installed": False,
        "tts_ready": False,
        "voices": [],
        "voice_count": 0,
        "forced_but_missing": False,
        "error": None,
    }
    try:
        from passage import local_voice
    except Exception as error:
        report["error"] = f"local_voice import failed: {str(error)[:160]}"
        return report

    try:
        state = (status_fn or local_voice.status)()
    except Exception as error:
        report["error"] = str(error)[:200]
        return report

    try:
        report["stt_installed"] = bool(local_voice.whisper_installed())
        report["tts_installed"] = bool(local_voice.piper_installed())
    except Exception as error:
        report["error"] = str(error)[:200]

    report.update({
        "mode": state.get("mode"),
        "setting": state.get("setting"),
        "unrecognised_setting": state.get("unrecognised_setting"),
        "stt_ready": bool(state.get("stt_ready")),
        "stt_engine": state.get("stt_engine") or "hosted",
        "tts_ready": bool(state.get("piper_ready")),
        "voices": list(state.get("voices") or []),
        "forced_but_missing": bool(state.get("forced_but_missing")),
    })
    report["voice_count"] = len(report["voices"])
    report["available"] = report["stt_ready"] or report["tts_ready"]
    return report


# ──────────────────────────────── font ──────────────────────────────── #

def _accepts_preferred(delegate: Callable[..., Any]) -> bool:
    """Can the compositor's probe be told which face to resolve?

    Asked of the object, never assumed: this module does not own
    image_compositor.py, so the capability is discovered, and its absence is
    reported rather than papered over.
    """
    try:
        params = signature(delegate).parameters
    except (TypeError, ValueError):
        return False
    if "preferred" in params:
        return True
    return any(p.kind is p.VAR_KEYWORD for p in params.values())


def font_report(size: int = 24, preferred: str | None = None) -> dict[str, Any]:
    """The overlay font that WOULD be used, and whether that is a fallback.

    This is the section the plan singles out: "this alone would have caught the
    font bug". PIL does not search Windows' font directory for
    ``DejaVuSans.ttf``, so every overlay silently fell through to
    ``load_default()`` — a bitmap face with no Unicode coverage — and rendered
    "Padrón" as "Padr▯n" while reporting complete success.

    Font resolution is delegated to ``image_compositor``; this function never
    resolves a font a second way, because a parallel resolution order can name
    a face the overlay would never load.

    HONEST SCOPE, and it is narrower than this docstring used to claim. The
    renderer resolves with the *user's* face:
    ``resolve_font(size, preferred=self.style.font_family, candidates=_FONT_FALLBACKS)``
    (``ImageCompositor._font``), and ``font_family`` is user-editable —
    TranslationUI.py's "Font family" input feeds ``OverlayStyle(font_family=…)``
    in TranslationBackend.py. An unqualified probe therefore agreed with the
    renderer only while the configured family equalled ``FONT_CANDIDATES[0]``;
    set "Arial Black.ttf" and it described a resolution the overlay never
    performed.

    ``probe_font`` now accepts ``preferred``, so ``preferred`` here means
    "report the resolution for THIS face" and the answer is the renderer's own.
    The capability is still DISCOVERED rather than assumed, via
    ``inspect.signature``: against an older or stubbed ``image_compositor`` that
    cannot take a face, the report refuses to guess — it names the requested
    face and reports an error, rather than printing a ``resolved`` value for a
    resolution nobody ran.

    Still open, and not fixed here: TranslationUI.py / TranslationBackend.py
    build the compositor per request, so no configured family reaches
    ``collect()`` yet. ``collect()`` reads ``backend.overlay_font_family`` when
    a backend chooses to expose it.
    """
    report: dict[str, Any] = {
        "requested": None,
        "resolved": None,
        "fell_back": False,
        "bitmap_fallback": False,
        "scalable": False,
        "candidates_tried": 0,
        "probe_source": "diagnostics",
        "error": None,
    }
    try:
        import image_compositor as ic
    except Exception as error:
        report["error"] = f"image_compositor import failed: {str(error)[:160]}"
        return report

    # image_compositor.probe_font is the compositor's own resolution, added in
    # Phase A by the agent that owns that file. Preferred over anything here:
    # a diagnostic that resolves the font by a second, parallel code path can
    # report a face the renderer would never actually use.
    delegate = getattr(ic, "probe_font", None) or getattr(ic, "resolve_overlay_font", None)
    if callable(delegate):
        kwargs: dict[str, Any] = {}
        if preferred:
            if _accepts_preferred(delegate):
                kwargs["preferred"] = preferred
            else:
                # The renderer would resolve `preferred` first. This probe
                # cannot, and reporting the unqualified answer under the user's
                # requested name is the exact lie this section exists to kill.
                report["requested"] = redact_path(preferred)
                report["probe_source"] = f"image_compositor.{delegate.__name__}"
                report["error"] = (
                    f"{delegate.__name__} accepts no 'preferred' face, so the "
                    f"resolution the overlay performs for {redact_path(preferred)!r} "
                    "cannot be reported without resolving fonts a second way here"
                )
                return report
        try:
            resolved = dict(delegate(size, **kwargs))
        except Exception as error:
            report["error"] = str(error)[:200]
            return report
        requested = preferred or resolved.get("requested")
        if requested is None:
            tried = resolved.get("tried") or []
            requested = tried[0] if tried else ic.OverlayStyle().font_family
        fell_back = resolved.get(
            "fell_back",
            bool(resolved.get("fallback")) or (
                resolved.get("resolved") is not None
                and resolved.get("resolved") != requested),
        )
        report.update({
            "requested": redact_path(requested),
            "resolved": redact_path(resolved.get("resolved"))
            or "PIL.ImageFont.load_default",
            "fell_back": bool(fell_back),
            "bitmap_fallback": bool(
                resolved.get("bitmap_fallback",
                             resolved.get("kind") == "bitmap_default"
                             or resolved.get("resolved") is None)),
            "scalable": bool(resolved.get(
                "scalable", resolved.get("kind") == "truetype")),
            "candidates_tried": len(resolved.get("tried") or []),
            "probe_source": f"image_compositor.{delegate.__name__}",
        })
        return report

    report["error"] = (
        "image_compositor exposes no probe_font; refusing to resolve the font a "
        "second way here, since a parallel resolution can name a face the "
        "renderer would never load"
    )
    report["probe_source"] = "unavailable"
    return report


# ─────────────────────────────── browser ────────────────────────────── #

def browser_placeholder() -> dict[str, Any]:
    """The browser half, before a browser has filled it in.

    Present in the /api/health payload on purpose: a curl of the API must be
    able to tell "no browser has reported" apart from "the browser reported
    nothing". Only the /diagnostics page can populate this, and only from the
    live browser — §2.4 is the whole reason. The 24 kHz AudioContext note in
    the audio code says "Safari/iOS returns the hardware rate"; that was
    INFERRED from a Chrome error message and has never been observed. The page
    therefore reports the sample rate the browser actually granted, as an
    observation, and makes no claim about what any other browser would do.
    """
    return {
        "collected": False,
        "note": "browser facts are collected by the /diagnostics page, not the server",
        "secure_context": None,
        "get_user_media": None,
        "media_recorder": None,
        "audio_context_sample_rate": None,
        "mime_types": {},
    }


# ─────────────────────────────── assembly ───────────────────────────── #

def collect(backend: Any = None) -> dict[str, Any]:
    """The whole resolved picture. Never raises: a broken probe is a finding."""
    return {
        "schema_version": SCHEMA_VERSION,
        "platform": platform_report(),
        "local_llm": local_llm_report(
            None,
            getattr(backend, "choose_local_model", None),
            # A backend may carry its own probe (tests do; a future backend
            # pointed at a non-default endpoint would too). Otherwise the
            # module-level TranslationBackend.probe_local_llm is auto-wired.
            getattr(backend, "probe_local_llm", None),
        ),
        "speech": speech_report(),
        # If the caller knows which face the overlay is configured with, the
        # report describes THAT resolution; otherwise it describes the default
        # one and says so rather than mislabelling it.
        "font": font_report(preferred=getattr(backend, "overlay_font_family", None)),
        "browser": browser_placeholder(),
    }


def describe_reachable(value: Any) -> str:
    """How the pasteable block renders a tri-state reachability.

    ``None`` means the probe had not answered when the block was built. It is
    printed in words, because "False" read on a phone three time zones away is
    indistinguishable from a real negative result."""
    if value is None:
        return "unknown (not asked yet)"
    return str(bool(value))


def describe_probe_freshness(llm: dict[str, Any]) -> str:
    """The age line the COPY block carries.

    The page renders freshness in its own ui.label, but the label is not what
    Copy puts on the clipboard — and the clipboard is the artifact designed to
    leave the machine (PORTABILITY_PLAN Phase E: a human pasting this back from
    an M1 or a phone). An undated paste is a guess presented as a reading, so
    the age travels inside the text, not beside it."""
    if llm.get("pending"):
        return "probe freshness: probe in flight — no answer yet (cached: False)"
    age = llm.get("age_seconds")
    if age is None:
        return "probe freshness: taken inline for this report (cached: False)"
    ttl = llm.get("ttl_seconds")
    return (f"probe freshness: {float(age):.1f}s old"
            f" (cached: {bool(llm.get('cached'))}"
            + (f", re-probed after {ttl} s)" if ttl is not None else ")"))


def format_text(report: dict[str, Any]) -> str:
    """A copy-pasteable block. Phase E is a human on a phone pasting this back,
    so it has to survive a mobile clipboard: flat ``key: value`` lines, no
    box-drawing, no colour, no wrapping-sensitive alignment."""
    p = report.get("platform", {})
    llm = report.get("local_llm", {})
    sp = report.get("speech", {})
    ft = report.get("font", {})
    br = report.get("browser", {})
    lines = [
        f"passage diagnostics (schema {report.get('schema_version')})",
        f"os: {p.get('system')} {p.get('release')} / {p.get('machine')} / {p.get('processor_arch')}",
        f"python: {p.get('python_version')} ({p.get('python_implementation')})"
        + (" in docker" if p.get("in_docker") else ""),
        f"ollama reachable: {describe_reachable(llm.get('reachable'))}"
        f" (outcome: {llm.get('outcome') or 'n/a'},"
        f" {llm.get('elapsed_ms')} ms, limit {llm.get('timeout_seconds')} s)"
        + (f" — {llm.get('detail') or llm.get('error')}"
           if (llm.get("detail") or llm.get("error")) else ""),
        describe_probe_freshness(llm),
        f"ollama models ({llm.get('model_count')}): "
        + (", ".join(llm.get("models") or []) or "none"),
        f"chosen local model: {llm.get('chosen_model') or 'none'}",
        f"speech available: {sp.get('available')} (mode {sp.get('mode')})",
        f"stt: installed={sp.get('stt_installed')} ready={sp.get('stt_ready')}"
        f" engine={sp.get('stt_engine')}",
        f"tts: installed={sp.get('tts_installed')} ready={sp.get('tts_ready')}",
        f"voices ({sp.get('voice_count')}): "
        + (", ".join(sp.get("voices") or []) or "none"),
        f"speech forced but missing: {sp.get('forced_but_missing')}",
        f"font requested: {ft.get('requested')}",
        f"font resolved: {ft.get('resolved')} (fell back: {ft.get('fell_back')},"
        f" bitmap: {ft.get('bitmap_fallback')})",
    ]
    if br.get("collected"):
        lines += [
            f"secure context: {br.get('secure_context')}",
            f"getUserMedia: {br.get('get_user_media')}",
            f"MediaRecorder: {br.get('media_recorder')}",
            f"AudioContext sample rate granted (observed): {br.get('audio_context_sample_rate')}",
            "mime support: " + (", ".join(
                f"{k}={v}" for k, v in (br.get("mime_types") or {}).items()
            ) or "none reported"),
        ]
    else:
        lines.append("browser: not collected (open /diagnostics in the browser)")
    return "\n".join(lines)


#: Gathers the browser-side facts the server cannot know. Returned as a JS
#: expression string so the page can hand it straight to NiceGUI's
#: run_javascript. Every value is an OBSERVATION made in the visiting browser;
#: nothing here is inferred from a user-agent string.
BROWSER_PROBE_JS = """
(async () => {
  const out = {
    collected: true,
    secure_context: window.isSecureContext === true,
    origin: String(location.origin),
    user_agent: String(navigator.userAgent),
    get_user_media: !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia),
    media_recorder: typeof window.MediaRecorder !== 'undefined',
    audio_context_available: !!(window.AudioContext || window.webkitAudioContext),
    audio_context_sample_rate: null,
    audio_context_requested_rate: 24000,
    audio_context_rate_honoured: null,
    mime_types: {},
    errors: []
  };
  const AC = window.AudioContext || window.webkitAudioContext;
  if (AC) {
    // Ask for 24 kHz and record what we were GIVEN. Whether a browser honours
    // the request is the open question in PORTABILITY_PLAN.md §2.4 — never
    // assumed here, only observed on the machine that loads this page.
    try {
      const ctx = new AC({ sampleRate: 24000 });
      out.audio_context_sample_rate = ctx.sampleRate;
      out.audio_context_rate_honoured = ctx.sampleRate === 24000;
      ctx.close && ctx.close();
    } catch (e) {
      out.errors.push('AudioContext(24000): ' + e);
      try {
        const ctx = new AC();
        out.audio_context_sample_rate = ctx.sampleRate;
        out.audio_context_rate_honoured = false;
        ctx.close && ctx.close();
      } catch (e2) { out.errors.push('AudioContext(): ' + e2); }
    }
  }
  const candidates = ['audio/webm', 'audio/webm;codecs=opus', 'audio/ogg;codecs=opus',
                      'audio/mp4', 'audio/mpeg', 'audio/wav'];
  for (const t of candidates) {
    out.mime_types[t] = (typeof window.MediaRecorder !== 'undefined'
      && typeof MediaRecorder.isTypeSupported === 'function')
      ? MediaRecorder.isTypeSupported(t) : false;
  }
  return out;
})()
"""
