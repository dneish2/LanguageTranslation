import asyncio
import base64
import logging
import os
import json
import time
import secrets
import uuid
from dataclasses import asdict
from io import BytesIO
from pathlib import Path
from threading import Lock, Thread
from typing import Any, Callable
from urllib.parse import quote

from nicegui import ui, app
from fastapi import Request, UploadFile, File, Form
from starlette.responses import Response, JSONResponse, StreamingResponse

import theme
from api_security import ApiGuard, client_ip, gate_disabled, MAX_TEXT_CHARS, MAX_UPLOAD_BYTES
from TranslationBackend import (
    LIVE_LOCAL_MODEL,
    LIVE_LOCAL_PREFERENCE,
    LIVE_PROBE_TTL_SECONDS,
    OLLAMA_BASE_URL,
    TEXT_MODEL,
    VISION_MODEL,
    TranslationBackend,
    TranslationRunState,
    SUPPORTED_DOCUMENT_EXTENSIONS,
    ollama_suits_translation,
    probe_local_llm,
)
from passage.ui.common import LANGUAGES, log_event as _log_event
from passage import __version__ as passage_version
from passage import diagnostics
from passage import engine_ledger
from passage import local_voice
from passage import policy
from passage import traces
from passage import usage
from passage import provider_profiles
from passage.auth.jwt_verify import identity_from_auth_header
from passage.ui.voice_page import VoicePageMixin

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("translation.ui")

# Signs the session cookie behind app.storage.user (per-visitor data like
# Recent Threads). Fresh per process start — no env var to configure, and a
# restart resets everyone's history, an acceptable tradeoff for a
# convenience feature versus hardcoding (or persisting) a secret.
_STORAGE_SECRET = secrets.token_urlsafe(32)


# ─────────────────── ONE local probe, off the event loop ────────────────── #
#
# /api/health used to call available_local_models(), diagnostics.collect()
# (which probes again) and choose_local_model() (which probes a third and a
# fourth time), SYNCHRONOUSLY, from an `async def`. Measured against an
# endpoint that accepts the connection and then says nothing: 15.31 s wall
# clock, four outbound probes, and a 50 ms heartbeat task that recorded ZERO
# ticks for the whole 15.31 s — the loop was blocked, so a single
# unauthenticated GET stalled every other connected client. /diagnostics and
# /engines had the same shape (7-8 s renders).
#
# Two fixes, both needed:
#   1. the probe work runs in a thread (asyncio.to_thread), so the loop keeps
#      serving while it waits;
#   2. it is ONE probe per request, and everything else — the model list, the
#      chosen model, the diagnostics section — is DERIVED from that single
#      result instead of re-probing.
#
# The answer is then reused for LIVE_PROBE_TTL_SECONDS, the same 60 s window
# the keystroke path already trusts (TranslationBackend.LIVE_PROBE_TTL_SECONDS)
# rather than a second, differently-tuned cache. A cached diagnostic that does
# not say how old it is would be worse than a slow one, so every consumer is
# handed `age_seconds`/`cached` and both surfaces print it.

#: Where a backend's snapshot lives. An attribute on the backend instance, not
#: a module global keyed by id(), so the shared production backend shares one
#: probe while a test's stub backend keeps its own.
_SNAPSHOT_ATTR = "_passage_probe_snapshot"
_snapshot_lock = Lock()


def _choose_from_probe(models: list[str]) -> str | None:
    """The model that would run, decided from an ALREADY-TAKEN probe.

    Mirrors TranslationBackend.choose_local_model — explicit override wins,
    then measured preference order, then any installed model that suits
    translation — but reads the model list off the probe result instead of
    issuing another HTTP call for it.
    """
    if LIVE_LOCAL_MODEL:
        return LIVE_LOCAL_MODEL
    installed = set(models)
    for candidate in LIVE_LOCAL_PREFERENCE:
        if candidate in installed:
            return candidate
    return sorted(installed)[0] if installed else None


def _image_source_chars(result: Any) -> int:
    """Characters of source text a photo turned out to contain.

    The metered unit is source characters (passage/usage.py), and for an image
    that number does not exist until the model has read the picture — so it is
    counted from the result rather than guessed from the file size.
    """
    blocks = (result or {}).get("translated_blocks") or []
    return sum(len(b.get("source_text") or "") for b in blocks)


def _take_local_snapshot(backend: Any) -> dict[str, Any]:
    """Run exactly one probe and derive everything from it. Blocking: callers
    must run this in a thread (or accept a stale snapshot)."""
    probe = getattr(backend, "probe_local_llm", None) or probe_local_llm
    try:
        report = dict(probe() or {})
    except Exception as error:  # a dead endpoint is a finding, never a 500
        report = {"reachable": False, "outcome": "error", "models": [],
                  "endpoint": None, "timeout_seconds": None,
                  "elapsed_ms": None, "detail": str(error)[:300]}
    models = [name for name in (report.get("models") or [])
              if "embed" not in name and ollama_suits_translation(name)]
    snapshot = {
        "probe": report,
        "models": models,
        "chosen_model": _choose_from_probe(models),
        "probed_at": time.time(),
        "ttl_seconds": LIVE_PROBE_TTL_SECONDS,
    }
    # Parity with available_local_models(), which records the same thing so a
    # caller can tell "refused" from "too slow".
    try:
        backend.last_local_probe = report
    except Exception:  # pragma: no cover - read-only stub
        pass
    with _snapshot_lock:
        try:
            setattr(backend, _SNAPSHOT_ATTR, snapshot)
        except Exception:  # pragma: no cover - read-only stub
            pass
    return snapshot


def _pending_snapshot() -> dict[str, Any]:
    """What a page renders before its first probe has landed. Deliberately
    NOT reported as reachable=False: "we have not asked yet" and "we asked and
    nothing answered" are different facts, and collapsing them is the exact
    mistake the probe outcomes exist to prevent."""
    return {
        "probe": {"reachable": None, "outcome": "probing", "models": [],
                  "endpoint": None, "timeout_seconds": None,
                  "elapsed_ms": None, "detail": "probe in flight"},
        "models": [],
        "chosen_model": None,
        "probed_at": time.time(),
        "ttl_seconds": LIVE_PROBE_TTL_SECONDS,
        "pending": True,
    }


def _fresh_snapshot(backend: Any) -> dict[str, Any] | None:
    """The cached snapshot if it is still inside the TTL, else None. Never
    probes, never blocks."""
    with _snapshot_lock:
        snapshot = getattr(backend, _SNAPSHOT_ATTR, None)
    if not isinstance(snapshot, dict):
        return None
    if time.time() - snapshot["probed_at"] >= snapshot.get(
            "ttl_seconds", LIVE_PROBE_TTL_SECONDS):
        return None
    return snapshot


def _cached_snapshot(backend: Any) -> dict[str, Any] | None:
    """The last snapshot whatever its age, or None. Never probes, never
    blocks. Callers MUST report the age — see age_seconds()."""
    with _snapshot_lock:
        snapshot = getattr(backend, _SNAPSHOT_ATTR, None)
    return snapshot if isinstance(snapshot, dict) else None


def snapshot_age_seconds(snapshot: dict[str, Any]) -> float:
    return max(0.0, time.time() - snapshot["probed_at"])


def describe_snapshot_age(snapshot: dict[str, Any] | None) -> str:
    """The freshness line both surfaces print. A cache that hides its age is
    how a diagnostic starts lying."""
    if snapshot is None or snapshot.get("pending"):
        return "probing this machine now…"
    age = snapshot_age_seconds(snapshot)
    if age < 1.0:
        return "probed just now"
    return f"probed {int(age)}s ago (re-probed after {int(snapshot['ttl_seconds'])}s)"


async def local_snapshot_async(backend: Any) -> dict[str, Any]:
    """A snapshot without blocking the event loop: the cached one when fresh,
    otherwise ONE probe on a worker thread."""
    fresh = _fresh_snapshot(backend)
    if fresh is not None:
        return fresh
    return await asyncio.to_thread(_take_local_snapshot, backend)


def snapshot_payload(snapshot: dict[str, Any]) -> dict[str, Any]:
    """The freshness facts that ride along with any cached answer."""
    age = snapshot_age_seconds(snapshot)
    return {
        "age_seconds": round(age, 3),
        "cached": age >= 1.0,
        # An un-landed probe has an age, but it is the age of the WAIT, not of
        # an answer. Carried so the pasteable block can say "no answer yet"
        # instead of dating a reading that does not exist.
        "pending": bool(snapshot.get("pending")),
        "ttl_seconds": snapshot["ttl_seconds"],
        "outcome": snapshot["probe"].get("outcome"),
        "elapsed_ms": snapshot["probe"].get("elapsed_ms"),
    }


class _SnapshotBackend:
    """A read-only stand-in handed to diagnostics.collect() so it reuses the
    snapshot's single probe instead of taking its own (collect() calls
    ``probe_local_llm`` and ``choose_local_model`` off whatever it is given).

    diagnostics.py is not modified for this — it already accepts an injected
    probe, which is exactly the seam needed.
    """

    def __init__(self, snapshot: dict[str, Any]) -> None:
        self._snapshot = snapshot

    def probe_local_llm(self) -> dict[str, Any]:
        return self._snapshot["probe"]

    def choose_local_model(self) -> str | None:
        return self._snapshot["chosen_model"]


def collect_diagnostics(snapshot: dict[str, Any]) -> dict[str, Any]:
    """diagnostics.collect() driven by an already-taken probe, with the
    probe's age stamped into the local_llm section so a reader can never
    mistake a 59-second-old answer for a live one."""
    report = diagnostics.collect(_SnapshotBackend(snapshot))
    section = report.get("local_llm")
    if isinstance(section, dict):
        section.update(snapshot_payload(snapshot))
    return report


class TranslationUI(VoicePageMixin):
    def __init__(self, *, backend: TranslationBackend | None = None, api_guard: ApiGuard | None = None):
        # ── CORE BACKEND ─────────────────────────────────────────────────
        # Shared across every client in the real app (see start_ui): backend
        # holds the OpenAI/Ollama client + job store (already job-id-keyed,
        # safe to share) and the translation cache; api_guard issues/checks
        # the short-lived per-page tokens. Defaulting to a fresh instance
        # keeps `TranslationUI()` with no args working for tests.
        self.backend = backend if backend is not None else TranslationBackend()
        self.api_guard = api_guard if api_guard is not None else ApiGuard()

        # ── UI CONTAINERS ────────────────────────────────────────────────
        self.upload_container = None
        self.progress_container = None
        self.result_container = None
        self.stats_container = None

        # ── TRANSLATION STATE ───────────────────────────────────────────
        self.uploaded_file: BytesIO | None = None
        self.uploaded_file_name: str | None = None
        self.uploaded_file_extension: str | None = None
        self.current_target_language: str | None = None
        self.overlay_show_original = False
        self.overlay_font_size = 24
        self.overlay_font_family = "DejaVuSans.ttf"
        self.overlay_preview_visible = True
        self.current_correlation_id: str | None = None
        self.cancel_button = None
        self.translate_button = None
        self.active_job_id: str | None = None
        self.job_poll_timer = None

        # ── SEGMENT EDITING ─────────────────────────────────────────────
        self.original_segments_map: dict[str, str] = {}
        self.translated_segments_map: dict[str, str] = {}
        # seg_id -> the live textarea element rendered by show_result(). Save All
        # Edits reads THESE; without the registry it could only re-serialise
        # segments the per-segment Update button had already applied, i.e. it
        # silently discarded every in-page edit while reporting success.
        self.segment_editors: dict[str, Any] = {}
        # This client's own document state — never read self.backend.segment_map
        # /.output_stream/etc. (those ambient properties proxy the backend's
        # SHARED self._active_run_state pointer, reassigned by get_job_result()
        # every time ANY client's job completes). Confirmed exploitable live
        # (2026-07-06): User B uploading a new file wiped User A's still-open
        # segment editor ("Segment not found") via that shared pointer. Always
        # a private, real object (never None) so nothing falls through to the
        # ambient default.
        self.document_run_state: TranslationRunState = TranslationRunState()
        #: Trace for the document currently open, so a segment edit knows which
        #: run it is correcting. None until a document completes.
        self.document_trace_id: str | None = None

        # ── DRAWER & ADVANCED MODE ──────────────────────────────────────
        self.drawer = None
        # Segment review renders whenever a document has segments; the old
        # Default/Advanced toggle confused more than it gated.
        self.advanced_mode = True
        # Threads: chats (text translations) and documents, newest first.
        # Backed by app.storage.user (see the recent_threads property) —
        # per-visitor session cookie, not shared across users.
        self.mode_tab_row = None
        self.text_output_label = None

        # ── MOBILE FLOW ────────────────────────────────────────────────
        # Text is the first tab and the lightest way in; Document/Image are a click away.
        self.input_mode = "Text"
        # DOM id prefix for the Text-mode workspace elements; must match the
        # hardcoded `scope` in _inject_workspace_text_live_translation_js.
        self.text_status_scope = "workspace_text"
        self.mobile_input_mode = "Text"
        self.source_language_input = None
        self.target_language_input = None
        self.text_source_input = None
        self.mobile_target_input = None
        self.mobile_voice_input = None
        self.image_source_upload = None
        self.image_capture_upload = None
        self.image_upload_bytes: bytes | None = None
        self.image_upload_name: str | None = None
        self.image_translation_result: dict[str, Any] | None = None
        self.current_source_language = "English"
        self.current_target_language = "Spanish"
        # Languages whose Piper voice this page load has already asked for.
        # Per-instance, i.e. per page load, which is the right scope: the
        # download itself is process-wide and idempotent.
        self._voice_prefetch_requested: set[str] = set()

        # ── UI CONSISTENCY STANDARDS (Passage "Press" tokens, theme.py) ──
        self.button_primary_classes = theme.BTN_PRIMARY
        self.button_secondary_classes = theme.BTN_SECONDARY
        self.banner_classes = theme.BANNER

        # ── USAGE STATS ─────────────────────────────────────────────────
        self.current_count = 0
        self.current_tokens = 0

    def _inject_theme(self) -> None:
        ui.add_head_html(theme.HEAD_HTML)
        # Quasar's brand colors would otherwise leak default blue into toggles,
        # uploads, spinners, and any color=primary props.
        ui.colors(
            primary=theme.PALETTE["accent"],
            secondary=theme.PALETTE["muted"],
            accent=theme.PALETTE["accent"],
            positive=theme.PALETTE["ok"],
            negative=theme.PALETTE["err"],
            warning=theme.PALETTE["warn"],
            info=theme.PALETTE["muted"],
        )

    def _inject_workspace_text_live_translation_js(self) -> None:
        # Injected ONCE per page build (main_page). Bindings are delegated on
        # `document`, so workspace re-renders (swap ⇄, mode tabs) need no
        # re-injection — calling add_body_html from a handler whose element was
        # just cleared crashes with "parent slot deleted".
        ui.add_body_html("""
<script>
(() => {
    const scope = 'workspace_text';
    const DEBOUNCE_MS = 350;
    const stateLabels = { READY: 'Ready', TRANSLATING: 'Translating…', UPDATED: 'Updated', ERROR: 'Error' };
    let debounceTimer = null;
    let activeRequestToken = 0;
    const byId = (id) => document.getElementById(id);
    const sourceEl = () => byId(`${scope}_source`);
    const sourceLangEl = () => byId(`${scope}_source_lang`);
    const targetEl = () => byId(`${scope}_target`);
    const outputEl = () => byId(`${scope}_output`);
    const statusEl = () => byId(`${scope}_status`);
    function setStatus(text) {
        const node = statusEl();
        if (node) node.textContent = text;
    }
    async function requestTranslation() {
        const source = sourceEl();
        const target = targetEl();
        const output = outputEl();
        if (!source || !target || !output) return;
        const text = (source.value || '').trim();
        const language = (target.value || 'Spanish').trim();
        if (!text) {
            output.textContent = '';
            setStatus(stateLabels.READY);
            return;
        }
        const fromLanguage = (sourceLangEl()?.value || '').trim();
        if (fromLanguage && fromLanguage.toLowerCase() === language.toLowerCase()) {
            setStatus('From and To are the same language — swap ⇄ or pick a different target.');
            return;
        }
        const token = ++activeRequestToken;
        setStatus(stateLabels.TRANSLATING);
        try {
            const fd = new FormData();
            fd.append('text', text);
            fd.append('language', language || 'Spanish');
            const resp = await fetch('/api/text_translate', { method: 'POST', body: fd, headers: { 'X-Passage-Token': window.PASSAGE_TOKEN || '' } });
            const data = await resp.json();
            if (token !== activeRequestToken) return;
            if (!resp.ok) throw new Error(data?.error || 'Translation failed.');
            output.textContent = data.translated_text || '';
            // Name the model that answered. Local vs hosted changes both cost
            // and latency, so it is state the user should be able to see.
            const engine = data.engine ? ` · ${data.engine}` : '';
            setStatus(`${stateLabels.UPDATED}${engine}`);
        } catch (err) {
            if (token !== activeRequestToken) return;
            setStatus(`${stateLabels.ERROR}: ${err?.message || 'unknown error'}`);
        }
    }
    function scheduleDebouncedTranslation() {
        if (debounceTimer) window.clearTimeout(debounceTimer);
        debounceTimer = window.setTimeout(requestTranslation, DEBOUNCE_MS);
    }
    // Delegated bindings: the workspace DOM is torn down and rebuilt on every
    // language swap / mode change, so listeners must live on `document`.
    document.addEventListener('input', (e) => {
        if (e.target && e.target.id === `${scope}_source`) scheduleDebouncedTranslation();
    });
    document.addEventListener('click', (e) => {
        if (e.target && e.target.closest && e.target.closest(`#${scope}_manual_translate`)) requestTranslation();
    });
    window.workspaceTextLiveTranslation = { requestTranslation };
})();
</script>
        """)

    # ──────────────────────────────────── MAIN PAGE ─────────────────────────────────────────

    def request_voice_prefetch(self, language: str | None = None) -> None:
        """Fetch the Piper voice for the language the user is ACTUALLY
        targeting, in the background, as soon as that language is known — and
        again whenever it changes.

        This used to fire once from main_page() with `self.current_target_language`,
        which was always the "Spanish" hardcoded in __init__: start_ui builds a
        FRESH TranslationUI per page load, and the prefetch ran before the
        language input (and therefore its bind_value) existed. en_US/es_ES ship
        with the image, so prefetch_voice no-opped every time and the fetch path
        was unreachable for every other language — a user translating into
        German got hosted TTS forever. The language now has to be passed in by
        whoever knows it (the "To" input's change handler, the voice page), so
        no caller can accidentally read a default.

        Same shape as `backend.prewarm_live()`, for the same reason: a voice is
        ~63MB, so doing it lazily at the moment someone presses speak would put
        a minutes-long download on the request path. Here nobody is waiting.

        Deliberately non-blocking and deliberately silent. `prefetch_voice`
        returns immediately, spawns a daemon thread, no-ops when local voice is
        off / the voice is already installed / the language has no Piper voice,
        and swallows its own failures. Page render must never wait on it, and a
        failed download must never reach the UI - a translation that finds no
        voice simply uses hosted TTS and names it in meta["tts"].
        """
        language = (language if language is not None else self.current_target_language) or ""
        language = language.strip()
        if not language:
            return
        # A change handler fires on every keystroke; one request per language
        # per page is enough (prefetch_voice itself is idempotent, but there is
        # no reason to hand it "G", "Ge", "Ger"... either).
        if language.lower() in self._voice_prefetch_requested:
            return
        self._voice_prefetch_requested.add(language.lower())
        try:
            local_voice.prefetch_voice(language)
        except Exception as error:
            # Belt and braces: even the act of *starting* the prefetch must not
            # be able to take the page down.
            logging.info("[UI] voice prefetch skipped (%s)", error)

    def main_page(self, mode: str | None = None):
        # A fresh instance per page load (see start_ui) has no memory of what
        # mode was active on /voice — carry it across the navigation via
        # ?mode= instead of self, which no longer survives the page change.
        if mode in {"Text", "Document", "Image/Camera"}:
            self.input_mode = mode
            self.mobile_input_mode = mode
        self._inject_theme()
        self._inject_api_token()
        self._inject_workspace_text_live_translation_js()
        # Get the local live model into VRAM while the user is still reading
        # the page, not on their first keystroke. Fire-and-forget.
        self.backend.prewarm_live()
        # The voice prefetch deliberately does NOT happen here: at this point
        # the "To" input does not exist yet, so the only language available is
        # the __init__ default. It fires from the input's change handler (and
        # once with that input's real value) — see request_voice_prefetch.
        # Header: wordmark goes home; the mode tabs are the only navigation.
        with ui.header().classes(f"items-center {theme.HEADER} px-4 py-1"):
            with ui.row().classes("w-full items-center gap-3"):
                # The only way into Recent Threads on a phone.
                ui.button(icon="menu", on_click=lambda: self.drawer.toggle())\
                    .props("flat round dense")\
                    .classes("p-mode-tab")\
                    .tooltip("Recent threads")
                ui.html(f'<span class="{theme.WORDMARK}">Passage<b>.</b></span>')\
                    .on("click", lambda: ui.navigate.to("/"))
                ui.element("div").classes("p-header-sep")
                self.mode_tab_row = ui.row().classes("items-center gap-0")
                self._render_mode_tabs()

        # Recent Threads drawer (renders itself into self.drawer)
        # show-if-above keeps the drawer open on wide screens and closed on a
        # phone, where it is opened by the header's menu button. Without that
        # button the drawer was simply unreachable at 390px: the page's only
        # controls were the mode tabs, swap and Translate.
        self.drawer = ui.drawer(side='left').props("show-if-above").classes(theme.DRAWER)
        self.show_document_list()

        # Default workspace page
        with ui.column().classes("w-full h-full items-center justify-start p-4"):
            self.upload_container = ui.column().classes("w-full max-w-6xl")
            self.progress_container = ui.column().classes("w-full max-w-6xl")
            self.result_container = ui.column().classes("w-full max-w-6xl")
            self.stats_container = ui.column().classes("w-full max-w-6xl")
            self.refresh_upload_ui()


    def _render_mode_tabs(self) -> None:
        if self.mode_tab_row is None:
            return
        self.mode_tab_row.clear()
        with self.mode_tab_row:
            for label, mode in (("Text", "Text"), ("Document", "Document"), ("Image", "Image/Camera")):
                active = " p-mode-tab-active" if self.input_mode == mode else ""
                ui.button(label, on_click=lambda _, m=mode: self.set_workspace_mode(m))\
                    .props("flat no-caps")\
                    .classes(f"p-mode-tab{active}")
            ui.button("Voice", on_click=lambda: ui.navigate.to("/voice"))\
                .props("flat no-caps")\
                .classes("p-mode-tab")
            profile = self.active_profile
            ui.button("Engines", icon="insights",
                      on_click=lambda: ui.navigate.to("/engines"))                .props("flat no-caps dense").classes("p-mode-tab ml-auto")                .tooltip("Where your text goes")
            ui.button(profile.describe() if profile else "Engine",
                      icon="tune", on_click=self.open_engine_settings)\
                .props("flat no-caps dense")\
                .classes("p-mode-tab ml-auto")\
                .tooltip("Choose where translation runs")

    def engines_page(self):
        """Where your text has actually been going.

        Deliberately not a settings screen. It answers "what happened to my
        text this session, and what could this machine do instead", which is
        the question the engine picker creates and nothing else answers.
        """
        self._inject_theme()
        self._inject_api_token()
        with ui.header().classes(f"items-center {theme.HEADER} px-4 py-1"):
            with ui.row().classes("w-full items-center gap-3"):
                ui.html(f'<span class="{theme.WORDMARK}">Passage<b>.</b></span>')\
                    .on("click", lambda: ui.navigate.to("/"))
                ui.element("div").classes("p-header-sep")
                ui.button("Workspace", on_click=lambda: ui.navigate.to("/"))\
                    .props("flat no-caps").classes("p-mode-tab")
                ui.button("Compare", on_click=lambda: ui.navigate.to("/compare"))\
                    .props("flat no-caps").classes("p-mode-tab")
                ui.button("Engines").props("flat no-caps")\
                    .classes("p-mode-tab p-mode-tab-active")

        profile = self.active_profile
        summary = engine_ledger.summarise(self.engine_runs)
        # This render must not probe. It used to call choose_local_model() and
        # available_local_models() inline — three blocking probes on the event
        # loop, a 7.6 s render against a silent Ollama that also froze every
        # other client. Render whatever was last probed (labelled with its
        # age), and let a timer fill in a fresh probe from a worker thread.
        snapshot = _cached_snapshot(self.backend) or _pending_snapshot()
        with ui.column().classes("w-full items-center p-4"):
            with ui.column().classes("w-full max-w-5xl gap-4"):
                ui.label("Where your text goes").classes("p-display text-xl")

                with ui.column().classes(f"w-full gap-2 p-4 {theme.WELL}"):
                    ui.label("Next translation").classes(theme.DATA)
                    privacy_label = ui.label("").classes("text-base")
                    detail_label = ui.label("").classes(theme.DATA)
                    freshness_label = ui.label("").classes("text-xs p-muted-text")

                    def render_right_now(snap: dict[str, Any]) -> None:
                        local_default = snap["chosen_model"]
                        # A FORECAST from a capability probe, now labelled as
                        # one. What actually happened is the section below,
                        # read from the ledger. This block used to be phrased
                        # as a guarantee and could sit directly above a ledger
                        # that contradicted it.
                        privacy_label.set_text(policy.describe_privacy(
                            profile, local_first_model=local_default,
                            hosted_available=getattr(self.backend, "provider", None) is not None))
                        bits = [f"engine: {profile.describe() if profile else 'auto (local first)'}"]
                        bits.append("metered" if policy.is_metered(
                            profile, local_first_model=local_default) else "not metered")
                        bits.append(f"local default: {local_default or 'none installed'}")
                        detail_label.set_text(" · ".join(bits))
                        freshness_label.set_text(describe_snapshot_age(snap))

                    render_right_now(snapshot)
                    ui.button("Change engine", on_click=self.open_engine_settings)\
                        .classes(f"{theme.BTN_SECONDARY_SM} mt-1")

                with ui.column().classes(f"w-full gap-2 p-4 {theme.WELL}"):
                    ui.label("This session").classes(theme.DATA)
                    if not summary["total_runs"]:
                        ui.label("Nothing translated yet.").classes("text-sm p-muted-text")
                    else:
                        local, remote = summary["local"], summary["remote"]
                        share = int(summary["local_share_of_chars"] * 100)
                        ui.label(f"{share}% of your text stayed on this machine")\
                            .classes("text-base")
                        # A bar, because a ratio is the whole point and a
                        # number makes you do the comparison yourself.
                        with ui.element("div").classes("w-full flex h-3 rounded overflow-hidden")\
                                .style("background: var(--p-rule, #d6cbb4)"):
                            if share:
                                ui.element("div").classes("h-full")\
                                    .style(f"width:{share}%; background: var(--p-ok, #3F6B4A)")
                        ui.label(
                            f"local: {local['runs']} runs · {local['chars']} chars"
                            + (f" · {local['median_ms']} ms median" if local["median_ms"] is not None else "")
                        ).classes(theme.DATA)
                        ui.label(
                            f"sent out: {remote['runs']} runs · {remote['chars']} chars"
                            + (f" · {remote['median_ms']} ms median" if remote["median_ms"] is not None else "")
                        ).classes(theme.DATA)
                        # Never folded into either side above: an engine label
                        # nobody could classify is not evidence the text
                        # stayed here, and not evidence it left.
                        if summary["unknown"]["runs"]:
                            ui.label(
                                f"destination not known: {summary['unknown']['runs']} runs · "
                                f"{summary['unknown']['chars']} chars"
                            ).classes(theme.DATA)
                        for name, count in summary["engines"].items():
                            ui.label(f"{name} — {count} run{'s' if count != 1 else ''}")\
                                .classes(theme.DATA)

                with ui.column().classes(f"w-full gap-2 p-4 {theme.WELL}"):
                    ui.label("Voice").classes(theme.DATA)
                    # One probe feeds both lines. They used to be derived from
                    # two different places, which is how the privacy sentence
                    # came to contradict the capability line under it.
                    voice = engine_ledger.voice_state()
                    ui.label(voice["privacy"]).classes("text-base")
                    ui.label(voice["detail"]).classes(theme.DATA)

                with ui.column().classes(f"w-full gap-2 p-4 {theme.WELL}"):
                    ui.label("Metered usage").classes(theme.DATA)
                    used = usage.summary(self.usage_store)
                    # `total_runs` keeps the zero case honest: "everything ran
                    # locally or on your own key" used to be printed on a
                    # session where nothing had run at all.
                    ui.label(usage.describe(used, total_runs=summary["total_runs"]))\
                        .classes("text-base")
                    if used.metered_runs:
                        with ui.element("div").classes("w-full flex h-3 rounded overflow-hidden")\
                                .style("background: var(--p-rule, #d6cbb4)"):
                            ui.element("div").classes("h-full")\
                                .style(f"width:{int(used.share_used * 100)}%; "
                                       "background: var(--p-accent, #802F3D)")
                    ui.label(
                        "Only work Passage paid for is counted. Anything that ran on this "
                        "machine or on your own key never reaches the counter."
                    ).classes("text-xs p-muted-text")

                with ui.column().classes(f"w-full gap-2 p-4 {theme.WELL}"):
                    ui.label("Models on this machine").classes(theme.DATA)
                    bench = self._load_bench_rows()
                    models_box = ui.column().classes("w-full gap-1")

                    def render_models(snap: dict[str, Any]) -> None:
                        models_box.clear()
                        with models_box:
                            if snap.get("pending"):
                                ui.label("Checking this machine…")\
                                    .classes("text-sm p-muted-text")
                                return
                            installed = snap["models"]
                            if not installed:
                                ui.label("No local models reachable — everything runs hosted.")\
                                    .classes("text-sm p-muted-text")
                            for name in installed:
                                row = bench.get(name)
                                detail = "not benchmarked yet"
                                if row:
                                    detail = (f"{row.get('size_gb', 0):.1f} GB · {row.get('median_ms')} ms"
                                              f" · agreement {row.get('consensus')}")
                                marker = " (default)" if name == snap["chosen_model"] else ""
                                ui.label(f"{name}{marker} — {detail}").classes(theme.DATA)

                    render_models(snapshot)

                    async def refresh_local_state() -> None:
                        """One probe, on a worker thread, after the page is
                        already on screen. The render never waits for it."""
                        snap = await local_snapshot_async(self.backend)
                        render_right_now(snap)
                        render_models(snap)

                    ui.timer(0.05, refresh_local_state, once=True)
                    ui.label(
                        "Agreement is how closely a model matches the others on a fixed "
                        "suite; it rewards the mainstream reading, so a low score is a "
                        "reason to look rather than proof of error."
                    ).classes("text-xs p-muted-text")

    def _load_bench_rows(self) -> dict[str, dict]:
        """Latest benchmark row per model from data/model_bench.jsonl."""
        rows: dict[str, dict] = {}
        path = Path(__file__).resolve().parent / "data" / "model_bench.jsonl"
        try:
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    rows[row.get("model", "?")] = row      # later runs win
        except (OSError, ValueError):
            return {}
        return rows

    def compare_page(self):
        """Same sentence, every engine, side by side.

        Answers "I brought my own key — how does it compare?", which no single
        translation view can. Ranks nothing: there is no reference translation
        here, so the honest signal is how far each engine sits from the others.
        """
        self._inject_theme()
        self._inject_api_token()
        with ui.header().classes(f"items-center {theme.HEADER} px-4 py-1"):
            with ui.row().classes("w-full items-center gap-3"):
                ui.html(f'<span class="{theme.WORDMARK}">Passage<b>.</b></span>')\
                    .on("click", lambda: ui.navigate.to("/"))
                ui.element("div").classes("p-header-sep")
                ui.button("Workspace", on_click=lambda: ui.navigate.to("/"))\
                    .props("flat no-caps").classes("p-mode-tab")
                ui.button("Compare").props("flat no-caps").classes("p-mode-tab p-mode-tab-active")

        with ui.column().classes("w-full items-center p-4"):
            with ui.column().classes("w-full max-w-5xl gap-3"):
                ui.label("Compare engines").classes("p-display text-xl")
                ui.label(
                    "One sentence, every engine you can reach. Latency and tokens are "
                    "measured per run. Agreement is how closely each output matches the "
                    "others — there's no reference translation to score against, so an "
                    "outlier is a prompt to look, not a verdict."
                ).classes("text-sm p-muted-text")
                ui.label(
                    "Local engines run one at a time — they share a GPU, and running "
                    "them together measures contention rather than the model. A local "
                    "model's first run still includes loading it into memory, so read "
                    "the second run for steady-state speed."
                ).classes(f"text-xs {theme.DATA}")

                with ui.row().classes(f"w-full items-center gap-3 flex-wrap {theme.WELL} p-3"):
                    # A default with idiom, a figure and a domain term: engines
                    # agree trivially on "Where is the pharmacy?", so seeding
                    # with an easy sentence would make the feature look useless.
                    source = ui.textarea(
                        label="Text",
                        value="The board pushed back on the buyback, arguing it "
                              "would leave the balance sheet stretched heading "
                              "into a soft quarter.",
                    ).props("autogrow rows=2").classes("flex-grow")
                    target = ui.input("To", value="Spanish").classes("w-40")
                run_row = ui.row().classes("w-full items-center gap-3")
                results_box = ui.column().classes("w-full gap-2")

                def render(results) -> None:
                    results_box.clear()
                    with results_box:
                        if not results:
                            ui.label("No engines available.").classes("text-sm p-muted-text")
                            return
                        fastest = min((r.latency_ms for r in results if r.ok), default=None)
                        for r in sorted(results, key=lambda x: (not x.ok, x.latency_ms or 10**9)):
                            with ui.column().classes(f"w-full gap-1 p-3 {theme.WELL}"):
                                with ui.row().classes("w-full items-baseline justify-between gap-2"):
                                    ui.label(r.label).classes("p-display")
                                    bits = [r.engine]
                                    if r.latency_ms is not None:
                                        fast = " (fastest)" if r.latency_ms == fastest else ""
                                        bits.append(f"{r.latency_ms} ms{fast}")
                                    if r.output_tokens is not None:
                                        bits.append(f"{r.output_tokens} tok")
                                    bits.append("free" if r.is_local
                                                else (f"${r.cost_usd:.6f}" if r.cost_usd is not None
                                                      else "metered · unpriced"))
                                    if r.agreement is not None:
                                        bits.append(f"agreement {r.agreement:.2f}")
                                    ui.label(" · ".join(bits)).classes(theme.DATA)
                                if r.ok:
                                    ui.label(r.text).classes(f"w-full p-2 {theme.PANEL_TARGET}")
                                else:
                                    ui.label(f"Failed: {r.error}")\
                                        .classes(f"w-full p-2 {theme.BANNER['negative']}")

                async def run() -> None:
                    text = (source.value or "").strip()
                    if not text:
                        ui.notify("Enter some text to compare.", type="warning")
                        return
                    run_row.clear()
                    with run_row:
                        ui.spinner(size="sm")
                        ui.label("Running every engine…").classes(theme.DATA)
                    candidates = self.backend.comparison_candidates(self.active_profile)
                    results = await asyncio.to_thread(
                        self.backend.compare_translations, text, target.value or "Spanish", candidates)
                    render(results)
                    run_row.clear()
                    with run_row:
                        ui.button("Run comparison", on_click=run).classes(theme.BTN_PRIMARY)
                        ui.label(f"{sum(1 for r in results if r.ok)}/{len(results)} engines answered")\
                            .classes(theme.DATA)

                with run_row:
                    ui.button("Run comparison", on_click=run).classes(theme.BTN_PRIMARY)

    def open_engine_settings(self) -> None:
        """Choose where translation runs: Passage's key, a local model, or your
        own endpoint. Session-scoped — see the active_profile property."""
        current = self.active_profile
        with ui.dialog() as dialog, ui.card().classes(f"w-full max-w-xl {theme.WELL} p-5 gap-3"):
            ui.label("Translation engine").classes("p-display text-lg")
            ui.label(
                "Bring your own endpoint and key and Passage stops metering the "
                "work — it isn't paying for it, so it shouldn't bill for it."
            ).classes("text-sm p-muted-text")

            kind = ui.radio(
                {
                    provider_profiles.KIND_APP: "Passage hosted (metered)",
                    provider_profiles.KIND_LOCAL: "Local model on this machine",
                    provider_profiles.KIND_BYO: "My own endpoint and key",
                },
                value=current.kind if current else provider_profiles.KIND_APP,
            ).props("inline")

            base_url = ui.input(
                "Base URL", placeholder="https://api.example.com/v1",
                value=(current.base_url if current else "") or "",
            ).classes("w-full")
            api_key = ui.input(
                "API key", placeholder="sk-…",
                value="",
            ).props("type=password").classes("w-full")
            if current and current.api_key:
                ui.label(f"A key is already saved for this session ({current.redacted()['api_key_hint']}). "
                         "Leave blank to keep it.").classes("text-xs p-muted-text")
            model = ui.input(
                "Model", placeholder="qwen2.5:7b",
                value=(current.model if current else "") or "",
            ).classes("w-full")
            status = ui.label("").classes("text-sm")

            def build() -> tuple[Any, str | None]:
                chosen = kind.value
                if chosen == provider_profiles.KIND_APP:
                    return None, None
                if chosen == provider_profiles.KIND_LOCAL and not base_url.value.strip():
                    base_url.value = OLLAMA_BASE_URL
                key = api_key.value.strip() or (current.api_key if current else "")
                problem = provider_profiles.validate(base_url.value, key, model.value)
                if problem:
                    return None, problem
                return provider_profiles.ProviderProfile(
                    label=model.value.strip() or "Custom",
                    kind=chosen,
                    base_url=base_url.value.strip(),
                    api_key=key,
                    model=model.value.strip(),
                ), None

            async def test() -> None:
                profile, problem = build()
                if problem:
                    status.text = problem
                    status.classes(replace="text-sm p-banner p-banner-negative")
                    return
                if profile is None:
                    status.text = "Passage's hosted key needs no test."
                    return
                status.text = "Testing…"
                status.classes(replace="text-sm p-muted-text")
                result = await asyncio.to_thread(self.backend.test_profile, profile)
                if result.get("ok"):
                    status.text = (f"Connected — {result['model']} answered in "
                                   f"{result['latency_ms']} ms: “{result['sample']}”")
                    status.classes(replace="text-sm p-banner p-banner-positive")
                else:
                    status.text = f"Couldn't connect: {result.get('error', 'unknown error')}"
                    status.classes(replace="text-sm p-banner p-banner-negative")

            def save() -> None:
                profile, problem = build()
                if problem:
                    status.text = problem
                    status.classes(replace="text-sm p-banner p-banner-negative")
                    return
                self._set_active_profile(profile)
                ui.notify(
                    "Using Passage's hosted key." if profile is None
                    else f"Using {profile.describe()} — this session is unmetered.",
                    type="positive",
                )
                dialog.close()
                # The header chip names the active engine, so it has to
                # re-render too — refresh_upload_ui only rebuilds the workspace.
                self._render_mode_tabs()
                self.refresh_upload_ui()

            with ui.row().classes("w-full justify-end gap-2 mt-1"):
                ui.button("Test connection", on_click=test).classes(theme.BTN_SECONDARY_SM)
                ui.button("Save", on_click=save).classes(theme.BTN_PRIMARY)
        dialog.open()

    def set_workspace_mode(self, mode: str) -> None:
        self.input_mode = mode
        self.mobile_input_mode = mode
        self._render_mode_tabs()
        self.refresh_upload_ui()

    def _show_banner(self, container, message: str, kind: str = "info") -> None:
        with container:
            ui.label(message).classes(self.banner_classes.get(kind, self.banner_classes["info"]))

    def _set_translate_button_busy(self, busy: bool) -> None:
        """Disable Translate while a request is in flight — re-enabled from
        every terminal path (show_error, show_result, show_mobile_*_result)."""
        if self.translate_button is not None:
            self.translate_button.set_enabled(not busy)

    def _render_progress_ui(self, message: str, *, show_cancel: bool = False):
        """Shared loading surface for Text/Image/Document translation: clears
        progress/result/stats, renders one circular-progress + status label
        (+ optional Cancel), returns (progress_ui, label_ui)."""
        self.progress_container.clear()
        self.result_container.clear()
        self.stats_container.clear()
        with self.progress_container:
            progress_ui = ui.circular_progress(value=0, max=100, show_value=True)\
                .classes("mx-auto mt-4")
            label_ui = ui.label(message).classes("text-center mt-2")
            if show_cancel:
                self.cancel_button = ui.button("Cancel Translation", on_click=self.cancel_translation)\
                    .classes(f"{theme.BTN_DANGER} mt-2")
        return progress_ui, label_ui

    def swap_languages(self):
        source = self.source_language_input.value if self.source_language_input else self.current_source_language
        target = self.target_language_input.value if self.target_language_input else self.current_target_language
        self.current_source_language, self.current_target_language = target, source
        # The swap is a language CHANGE like any other; the new target's voice
        # should start downloading now, not when someone presses speak.
        self.request_voice_prefetch(self.current_target_language)
        self.refresh_upload_ui()

    def _describe_segment_for_editor(self, index: int, seg_info: dict) -> str:
        location = seg_info.get("location")
        if location:
            return f"{index}. {location}"

        seg_type = seg_info.get("type", "segment").replace("_", " ")
        if seg_type == "pdf block":
            page_idx = seg_info.get("page_idx")
            if page_idx is not None:
                return f"{index}. PDF page {page_idx + 1}"
        return f"{index}. {seg_type.title()}"

    def _request_confirmation(self, title, message, on_confirm, confirm_label="Confirm"):
        with ui.dialog() as dialog:
            with ui.card().classes(f"w-[420px] max-w-full {theme.WELL}"):
                ui.label(title).classes("p-display text-lg")
                ui.label(message).classes("text-sm p-muted-text")
                with ui.row().classes("justify-end space-x-2 mt-4"):
                    ui.button("Cancel", on_click=dialog.close)\
                        .classes(theme.BTN_SECONDARY_SM)

                    def confirm_and_close():
                        dialog.close()
                        on_confirm()

                    ui.button(confirm_label, on_click=confirm_and_close)\
                        .classes(theme.BTN_DANGER_SM)
        dialog.open()

    def show_document_list(self):
        # Threads = chats (text translations) + documents translated this
        # session, newest first. In-memory until per-user storage (Phase 4).
        if self.drawer is None:
            return
        self.drawer.clear()
        with self.drawer:
            with ui.row().classes("w-full items-center justify-between mb-1"):
                ui.label("Recent Threads").classes("p-display text-lg")
                ui.button(icon="refresh", on_click=self.show_document_list)\
                    .props("flat round size=sm")\
                    .classes("p-mode-tab")

            threads = sorted(self.recent_threads, key=lambda t: t.get("when", 0), reverse=True)
            if not threads:
                ui.label("No threads yet.").classes("text-sm p-muted-text")
                return
            for t in threads[:20]:
                if t["kind"] == "chat":
                    handler = lambda _, thread=t: self._open_chat_thread(thread)
                    kind_label = f"chat · {t.get('language', '')}"
                else:
                    handler = lambda _, thread=t: self._open_document_thread(thread)
                    kind_label = f"document · {t.get('language', '')}"
                with ui.row().classes("w-full items-center gap-0"):
                    with ui.button(on_click=handler).classes("p-thread-item flex-grow"):
                        with ui.column().classes("gap-0 items-start"):
                            ui.label(t["label"][:44]).classes("text-sm")
                            ui.label(kind_label).classes("p-thread-kind")
                    ui.button(icon="close", on_click=lambda _, tid=t.get("id"): self._delete_thread(tid))\
                        .props("flat round size=sm")\
                        .classes("p-mode-tab")\
                        .tooltip("Remove this thread")

    def _open_document_thread(self, thread: dict) -> None:
        if thread.get("label") == self.uploaded_file_name and self.original_segments_map:
            self.show_result()
        else:
            ui.notify(
                "Re-upload this file to open it again — saved documents arrive with accounts.",
                type="info",
            )

    def _open_chat_thread(self, thread: dict) -> None:
        """Reload a text translation into the workspace."""
        self.current_target_language = thread.get("language") or self.current_target_language
        self.input_mode = "Text"
        self.mobile_input_mode = "Text"
        self._render_mode_tabs()
        self.refresh_upload_ui()
        if self.text_source_input is not None:
            self.text_source_input.value = thread.get("original", "")
        if self.target_language_input is not None:
            self.target_language_input.value = self.current_target_language
        if self.text_output_label is not None:
            self.text_output_label.text = thread.get("translated", "")

    def refresh_upload_ui(self):
        # reset UI + clear segments
        self.upload_container.clear()
        self.progress_container.clear()
        self.result_container.clear()
        self.stats_container.clear()
        # A fresh, private TranslationRunState — NOT self.backend.segment_map
        # .clear(), which used to mutate the backend's SHARED ambient state
        # and could wipe another concurrently connected client's segments.
        self.document_run_state = TranslationRunState()

        self.render_unified_workspace()
        # Pick up threads recorded by the API layer since the last render.
        self.show_document_list()

    def render_unified_workspace(self):
        max_width = "max-w-6xl"
        with self.upload_container:
            with ui.column().classes(f"w-full {max_width} gap-3"):
                with ui.row().classes(f"w-full items-end gap-2 flex-wrap {theme.WELL} p-3"):
                    # bind_value: language edits must survive workspace
                    # re-renders (mode tabs) — an unbound input's value was
                    # silently reset to the last committed language.
                    self.source_language_input = ui.input(
                        label="From",
                        placeholder="Source language",
                        autocomplete=LANGUAGES,
                    ).bind_value(self, "current_source_language")\
                     .props(f"for={self.text_status_scope}_source_lang").classes("min-w-[120px] flex-1")
                    ui.button("⇄", on_click=self.swap_languages).classes(self.button_secondary_classes)
                    self.target_language_input = ui.input(
                        label="To",
                        placeholder="Target language",
                        autocomplete=LANGUAGES,
                    ).bind_value(self, "current_target_language").classes("min-w-[120px] flex-1")
                    # The voice for the language they are ACTUALLY targeting,
                    # fetched now and again on every change. This is the only
                    # place the real target language is first known on a page
                    # load, so it is the only place the prefetch can start.
                    self.target_language_input.on_value_change(
                        lambda event: self.request_voice_prefetch(event.value))
                    self.request_voice_prefetch(self.target_language_input.value)
                    self.translate_button = ui.button("Translate", on_click=self.start_mobile_translation).classes(self.button_primary_classes)
                    self.translate_button.props(f"id={self.text_status_scope}_manual_translate")

                # Facing pages: source sits on paper, translation on panel;
                # stacks to one column on narrow viewports.
                with ui.element("div").classes("w-full grid grid-cols-1 md:grid-cols-2 gap-3"):
                    with ui.column().classes(f"w-full p-4 gap-2 {theme.PANEL_SOURCE}"):
                        ui.label("Source").classes(theme.DATA)
                        self._render_source_input_panel()
                    with ui.column().classes(f"w-full p-4 gap-2 {theme.PANEL_TARGET}"):
                        with ui.row().classes("w-full items-baseline justify-between"):
                            ui.label("Translation").classes(theme.DATA)
                            ui.label("").bind_text_from(self.target_language_input, "value")\
                                .classes(theme.DATA)
                        if self.input_mode == "Text":
                            ui.label("Ready").classes(self.banner_classes["info"]).props(
                                f"id={self.text_status_scope}_status"
                            )
                            self.text_output_label = ui.label("").classes(
                                "w-full min-h-[140px] p-1 text-base"
                            ).props(f"id={self.text_status_scope}_output")
                        else:
                            self._show_banner(self.progress_container, "Status: Ready to translate.", "info")

    def _render_source_input_panel(self):
        if self.input_mode == "Text":
            self.text_source_input = ui.textarea(
                label="Enter text to translate",
                placeholder="Type or paste text…",
            ).props(f"autogrow rows=8 for={self.text_status_scope}_source").classes("w-full")
            # Quasar's `for` prop sets the id on the NATIVE control; a plain
            # `id` prop never reaches it in NiceGUI 3, which left the live
            # translation JS unable to find these fields.
            if self.target_language_input:
                self.target_language_input.props(f"for={self.text_status_scope}_target")
            return
        if self.input_mode == "Document":
            # auto_upload so picking a file IS the upload — without it the file
            # sits queued at 0% and Translate says "no file uploaded".
            # accept= matches the label: without it the picker offered every
            # file, and choosing e.g. a CSV export produced a green "Selected"
            # toast and "Ready to translate" before failing on Translate with
            # "Unsupported file extension: csv". Say no at selection time.
            ui.upload(
                label="Click or drop DOCX, PPTX, or PDF",
                multiple=False,
                auto_upload=True,
                on_upload=self.handle_mobile_upload,
            ).props(f"accept={','.join('.' + e for e in sorted(SUPPORTED_DOCUMENT_EXTENSIONS))}")\
             .classes("w-full")
            if self.uploaded_file_name:
                ui.label(f"Selected file: {self.uploaded_file_name}").classes("text-sm p-muted-text")
            return

        # One affordance: the file input's accept/capture props let phones
        # offer the camera directly, so no separate fallback uploader.
        ui.upload(
            label="Click or drop an image (PNG, JPG, WEBP)",
            multiple=False,
            auto_upload=True,
            on_upload=self.handle_mobile_image_upload,
        ).props('accept=".png,.jpg,.jpeg,.webp" capture=environment').classes("w-full")
        if self.image_upload_name:
            ui.label(f"Selected image: {self.image_upload_name}").classes("text-sm p-muted-text")

    def set_mobile_input_mode(self, mode):
        self.mobile_input_mode = mode
        self.input_mode = mode
        self.refresh_upload_ui()

    async def handle_mobile_upload(self, event):
        # NiceGUI 3.x: the payload lives on event.file (FileUpload) and reads async.
        name = event.file.name
        extension = name.split(".")[-1].lower() if "." in name else ""
        # accept= on the input is only a picker hint — drag-and-drop ignores it
        # entirely. Reject here too, so an unsupported file never gets a green
        # "Selected" toast and a "Ready to translate" status it can't honour.
        if extension not in SUPPORTED_DOCUMENT_EXTENSIONS:
            supported = ", ".join(sorted(e.upper() for e in SUPPORTED_DOCUMENT_EXTENSIONS))
            ui.notify(
                f"Passage can't translate .{extension or 'unknown'} files yet — "
                f"upload a {supported}.",
                type="negative",
            )
            return
        if self._reject_oversize_upload(event.file):
            return
        self.uploaded_file_name = name
        self.uploaded_file_extension = extension
        self.uploaded_file = BytesIO(await event.file.read())
        ui.notify(f"Selected '{self.uploaded_file_name}'", type="positive")
        self.refresh_upload_ui()

    def start_mobile_translation(self):
        language = self.target_language_input.value if self.target_language_input else self.current_target_language
        self.current_target_language = language
        # Last-chance backstop: even if no change handler ever fired (an API
        # caller, a restored thread), the language is unambiguous here.
        self.request_voice_prefetch(language)
        if not language:
            self.show_error("Please enter a valid target language.")
            return
        source_language = (
            self.source_language_input.value if self.source_language_input else self.current_source_language
        ) or ""
        if source_language.strip().lower() == language.strip().lower():
            self.show_error("From and To are the same language — swap ⇄ or pick a different target.")
            return

        if self.input_mode == "Document":
            if not self.uploaded_file:
                self.show_error("Please upload a file before translating.")
                return
            self._set_translate_button_busy(True)
            self.handle_translation(language)
            return

        if self.input_mode == "Image/Camera":
            if not self.image_upload_bytes or not self.image_upload_name:
                self.show_error("Please upload or capture an image before translating.")
                return
            self._set_translate_button_busy(True)
            progress_ui, label_ui = self._render_progress_ui("Reading image text...")

            # Bound on THIS thread, where session storage exists (see
            # _engine_recorder): the work below runs off the event loop.
            recorder = self._engine_recorder()
            # Same reason, second fact: the session's ENGINE choice also only
            # exists on this thread (active_profile reads app.storage.user),
            # and the ContextVar the backend consults does not cross into a
            # bare Thread. Without capturing it here the image path ran on
            # Passage's own hosted key even for a user who had pointed the app
            # at their own endpoint — their photo went to the wrong place.
            profile = self.active_profile

            def image_task():
                # Established INSIDE the worker, for the same reason
                # queue_translation_job does it: ContextVars are per-thread.
                with self.backend.using_profile(profile):
                    self._run_mobile_image_translation(language, progress_ui, label_ui,
                                                       recorder)

            Thread(target=image_task).start()
            return

        source_text = (self.text_source_input.value or "").strip() if self.text_source_input else ""
        if not source_text:
            self.show_error("Please provide source text before translating.")
            return

        self._set_translate_button_busy(True)
        progress_ui, label_ui = self._render_progress_ui("Translating text...")

        recorder = self._engine_recorder()

        def voice_task():
            self._run_mobile_voice_translation(source_text, language, progress_ui,
                                               label_ui, recorder)

        Thread(target=voice_task).start()

    def _run_mobile_voice_translation(self, voice_text, language, progress_ui, label_ui,
                                      recorder=None):
        try:
            progress_ui.set_value(40)
            label_ui.text = "Calling translation model..."
            # translate_text always runs on Passage's own hosted key, so this
            # is metered work — and it was invisible to /engines until now.
            translated = self._run_recorded(
                recorder or self._engine_recorder(),
                self.backend.translate_text, (voice_text, language),
                surface=policy.Surface.VOICE, engine=f"hosted:{TEXT_MODEL}",
                chars=len(voice_text or ""))
            progress_ui.set_value(100)
            label_ui.text = "Translation complete."
            self.current_count = 1
            self.current_tokens = 0
            self.show_mobile_voice_result(voice_text, translated, language)
        except Exception as ex:
            logging.error("[UI] Mobile voice translation error: %s", ex, exc_info=True)
            self.show_error(ex, retry=self.start_mobile_translation)

    def _run_mobile_image_translation(self, language, progress_ui, label_ui,
                                      recorder=None):
        # `recorder` is the engine-ledger binding captured on the request
        # thread, where session storage exists. The vision path runs on
        # Passage's hosted key, so this is metered work — and /engines never
        # saw it. chars come from the result: the source text is inside the
        # photo and its length isn't knowable until the model has read it.
        try:
            progress_ui.set_value(40)
            label_ui.text = "Reading and translating image text..."
            self.image_translation_result = self._run_recorded(
                recorder or self._engine_recorder(),
                self.backend.translate_image_text_blocks,
                (self.image_upload_bytes, self.image_upload_name, language),
                surface=policy.Surface.IMAGE, engine=f"hosted:{VISION_MODEL}",
                chars=None, chars_of=_image_source_chars,
            )
            progress_ui.set_value(100)
            label_ui.text = "Translation complete."
            self.show_mobile_image_result(language)
        except Exception as ex:
            logging.error("[UI] Mobile image translation error: %s", ex, exc_info=True)
            self.show_error(ex, retry=self.start_mobile_translation)

    @staticmethod
    def upload_exceeds_limit(file_upload) -> bool:
        """True when an upload is over MAX_UPLOAD_BYTES, asked BEFORE reading it.

        The limit used to be enforced at exactly one place — the image API
        route — and only after the whole body was already in memory. The two
        browser upload handlers checked the extension and then did
        ``await event.file.read()`` on anything, so a 900 MB file named .png
        was a memory event, not a rejection. NiceGUI's FileUpload knows its
        size without reading (bytes already in a buffer, or a stat on the
        spooled temp file), so the answer costs nothing.
        """
        size = getattr(file_upload, "size", None)
        if not callable(size):
            return False  # unknown size: fall through to the downstream limit
        return size() > MAX_UPLOAD_BYTES

    def _reject_oversize_upload(self, file_upload) -> bool:
        if not self.upload_exceeds_limit(file_upload):
            return False
        limit_mb = MAX_UPLOAD_BYTES / (1024 * 1024)
        ui.notify(
            f"That file is too large. The limit is {limit_mb:.0f} MB.",
            type="negative",
        )
        return True

    async def handle_mobile_image_upload(self, event):
        if self._reject_oversize_upload(event.file):
            return
        self.image_upload_name = event.file.name
        self.image_upload_bytes = await event.file.read()
        ui.notify(f"Selected image '{self.image_upload_name}'", type="positive")
        self.refresh_upload_ui()

    def show_mobile_image_result(self, language: str) -> None:
        self._set_translate_button_busy(False)
        self.progress_container.clear()
        self.result_container.clear()
        result = self.image_translation_result or {}
        blocks = result.get("translated_blocks", [])
        confidence = result.get("confidence_metadata", {})
        overlay_png = result.get("overlay_png")
        with self.result_container:
            with ui.column().classes(f"w-full max-w-3xl mx-auto gap-3 p-4 {theme.WELL}"):
                ui.label(f"Image OCR translation → {language}").classes("p-display text-lg")
                ui.label(
                    f"Confidence avg: {confidence.get('average_confidence', 0)} "
                    f"across {confidence.get('block_count', 0)} blocks"
                ).classes(theme.DATA)
                # The translated-in-place picture. This is the answer for
                # "point your phone at a menu": reading a two-column list of
                # blocks means holding the phone AND the menu and matching them
                # up by eye, which is most of the work the feature exists to do.
                if overlay_png:
                    encoded = base64.b64encode(overlay_png).decode("ascii")
                    ui.label("Translated in place").classes("p-data")
                    ui.image(f"data:image/png;base64,{encoded}")\
                        .classes("w-full rounded")\
                        .style("max-height: 70vh; object-fit: contain")
                    ui.label(
                        f"{result.get('placed_block_count', 0)} of "
                        f"{confidence.get('block_count', 0)} blocks positioned"
                    ).classes(theme.DATA)
                for idx, block in enumerate(blocks, start=1):
                    with ui.grid(columns=2).classes("w-full gap-2"):
                        with ui.column().classes("w-full gap-1"):
                            ui.label(f"Extracted #{idx}").classes("p-data")
                            ui.label(block.get("source_text", "")).classes(f"text-sm {theme.PANEL_SOURCE} p-2")
                        with ui.column().classes("w-full gap-1"):
                            ui.label(f"Translated #{idx}").classes("p-data")
                            ui.label(block.get("translated_text", "")).classes(f"text-sm {theme.PANEL_TARGET} p-2")

    def show_mobile_voice_result(self, original_text, translated_text, language):
        self._set_translate_button_busy(False)
        self.progress_container.clear()
        self.result_container.clear()
        self.stats_container.clear()
        with self.result_container:
            with ui.column().classes(f"w-full max-w-3xl mx-auto gap-3 p-4 {theme.WELL}"):
                ui.label(f"Voice translation → {language}").classes("p-display text-lg")
                ui.label("Original").classes("p-data")
                ui.label(original_text).classes(f"w-full p-3 {theme.PANEL_SOURCE} text-base")
                ui.label("Translated").classes("p-data")
                ui.label(translated_text).classes(f"w-full p-3 {theme.PANEL_TARGET} text-base")

                with ui.column().classes("w-full gap-2"):
                    ui.button(
                        "Copy Translation",
                        on_click=lambda: ui.run_javascript(
                            f"navigator.clipboard.writeText({json.dumps(translated_text)})"
                        )
                    ).classes(theme.BTN_PRIMARY_XL)
                    ui.button("Start Over", on_click=self.refresh_upload_ui).classes(theme.BTN_SECONDARY_XL)

    def handle_translation(self, target_language, font_size=None, autofit=False):
        if not target_language:
            self.show_error("Please enter a valid target language.")
            return
        self.current_target_language = target_language
        correlation_id = str(uuid.uuid4())
        self.current_correlation_id = correlation_id
        _log_event(
            "ui.translation_requested",
            correlation_id=correlation_id,
            file_name=self.uploaded_file_name,
            file_extension=self.uploaded_file_extension,
            target_language=target_language,
        )
        logging.info(f"[UI] Translating '{self.uploaded_file_name}' → {target_language}")

        progress_ui, label_ui = self._render_progress_ui("Preparing translation...", show_cancel=True)

        self._start_job_and_poll(
            progress_ui=progress_ui,
            label_ui=label_ui,
            correlation_id=correlation_id,
            processed=False,
            font_size=font_size,
            autofit=autofit,
            target_language=target_language,
            complete_event="ui.translation_complete",
            failed_event="ui.translation_failed",
            cancelled_event="ui.translation_cancelled",
        )

    def cancel_translation(self):
        # Defensive: self.cancel_button only exists while self.active_job_id
        # is set (both happen synchronously in handle_translation before any
        # click could land), so this should never be the no-job-id case. Not
        # falling back to backend.request_cancel() deliberately — that flag
        # is a single value shared across every client's backend instance,
        # found while auditing the segment-editor cross-user bug (same
        # session) — no need to introduce a fourth reachable path to it.
        if self.active_job_id:
            self.backend.cancel_job(self.active_job_id)
        self.show_error("Translation was canceled. Please upload or try again.")

    def _start_job_and_poll(
        self,
        *,
        progress_ui,
        label_ui,
        correlation_id: str,
        processed: bool,
        font_size: int | None,
        autofit: bool,
        target_language: str,
        complete_event: str,
        failed_event: str,
        cancelled_event: str,
    ) -> None:
        started = time.time()
        if self.job_poll_timer:
            self.job_poll_timer.active = False
            self.job_poll_timer = None

        # Documents run on the session's chosen endpoint too. Until now the
        # job always used the process default, so "bring your own key" quietly
        # excluded the surface that costs the most.
        profile = self.active_profile
        # Bound here, on the request context: the poll callback that books the
        # run may have no session storage of its own.
        doc_recorder = self._engine_recorder()
        self.active_job_id = self.backend.start_translation_job(
            input_stream=self.uploaded_file,
            file_extension=self.uploaded_file_extension,
            target_language=target_language,
            processed=processed,
            font_size=font_size,
            autofit=autofit,
            correlation_id=correlation_id,
            profile=profile,
            file_name=self.uploaded_file_name,
        )

        def poll_job():
            if not self.active_job_id:
                return
            job = self.backend.get_job(self.active_job_id)
            if not job:
                return
            progress_ui.set_value(job.progress)
            label_ui.text = job.status_message

            if job.state in {"queued", "running"}:
                return

            if self.job_poll_timer:
                self.job_poll_timer.active = False
                self.job_poll_timer = None

            if job.state == "canceled":
                _log_event(cancelled_event, correlation_id=correlation_id)
                return

            if job.state == "failed":
                _log_event(failed_event, correlation_id=correlation_id, error=job.error or "unknown")
                self.show_error(job.error or "Translation failed", retry=self.start_mobile_translation)
                return

            result = self.backend.get_job_result(job.result_handle) if job.result_handle else None
            if not result:
                _log_event(failed_event, correlation_id=correlation_id, error="missing_result_handle")
                self.show_error("Translation failed: missing result payload.", retry=self.start_mobile_translation)
                return
            # This client's OWN run_state, fetched by result_handle — NOT the
            # backend's shared _active_run_state (get_job_result reassigns
            # that pointer as a side effect on EVERY client's job completion,
            # which is exactly the bug this line avoids reintroducing).
            job_run_state = self.backend.get_run_state_for_result(job.result_handle)
            if job_run_state is not None:
                self.document_run_state = job_run_state

            count = result["count"]
            tokens = result["tokens"]
            seg_map = result["segment_map"]

            self.current_count = count
            self.current_tokens = 0 if processed else tokens
            # (regenerated runs keep the real target language — writing
            # "Processed" into it now leaks into the bound To input)

            self.original_segments_map.clear()
            self.translated_segments_map.clear()
            for seg_id, seg_info in seg_map.items():
                self.original_segments_map[seg_id] = seg_info["original"]
                self.translated_segments_map[seg_id] = seg_info["translated"]

            # Open a trace for this run and record what the model produced, so
            # any later human edit has something to be a correction OF. Policy
            # decides whether these are written at all (documents: yes).
            if not processed:
                profile = self.active_profile
                engine = profile.describe() if profile else f"hosted:{TEXT_MODEL}"
                self.document_trace_id = traces.record_trace(traces.Trace(
                    name=self.uploaded_file_name or "document",
                    target_language=target_language,
                    engine=engine,
                    segment_count=len(seg_map),
                    metadata={"extension": self.uploaded_file_extension},
                ))
                for seg_id, seg_info in seg_map.items():
                    traces.record_generation(
                        trace_id=self.document_trace_id, segment_id=seg_id,
                        source=seg_info.get("original", ""),
                        output=seg_info.get("translated", ""),
                        engine=engine,
                    )
                # The surface engine_ledger.summarise was written for ("one
                # document is one entry but a lot of text") and the one it
                # never saw: /engines said "Nothing translated yet" after a
                # whole document had gone to a hosted model on Passage's key.
                # `processed` runs are excluded because no model ran there —
                # that path only regenerates the output file.
                doc_recorder(
                    surface=policy.Surface.DOCUMENT, engine=engine,
                    latency_ms=int((time.time() - started) * 1000),
                    chars=sum(len(s.get("original") or "") for s in seg_map.values()))

            if not processed and self.uploaded_file_name:
                self._record_thread({
                    "kind": "document",
                    "label": self.uploaded_file_name,
                    "language": target_language,
                    "when": time.time(),
                })
                self.show_document_list()

            _log_event(
                complete_event,
                correlation_id=correlation_id,
                elapsed_seconds=round(time.time() - started, 3),
                metrics=result.get("metrics", {}),
            )
            self.show_result()
            self.active_job_id = None

        self.job_poll_timer = ui.timer(0.2, poll_job)

    def get_fresh_download_stream(self):
        # re-generate with edits and return a fresh BytesIO
        self.backend.regenerate_output_stream(run_state=self.document_run_state)
        fresh = BytesIO()
        output_stream = self.document_run_state.output_stream
        output_stream.seek(0)
        fresh.write(output_stream.read())
        fresh.seek(0)
        return fresh

    def show_result(self):
        logging.info(f"[UI] Rendering results – advanced={self.advanced_mode}, segments={len(self.original_segments_map)}")
        self._set_translate_button_busy(False)
        self.progress_container.clear()
        self.result_container.clear()
        self.stats_container.clear()
        # clearing progress_container above deleted the cancel button with it
        self.cancel_button = None

        with self.result_container:
            with ui.column().classes("max-w-3xl mx-auto w-full space-y-6 mt-4"):
                ui.label(f"{self.uploaded_file_name} → {self.current_target_language}")\
                    .classes("p-display text-2xl")

                # ── SEGMENT EDITOR ──────────────────────────────
                # Rebuilt on every render, so the registry is rebuilt too:
                # stale elements from a previous render must never be read.
                self.segment_editors.clear()
                if self.advanced_mode and self.original_segments_map:
                    ui.separator().classes("my-4")
                    ui.label("Segment review").classes("p-display text-xl mb-2")
                    ui.label(f"{len(self.original_segments_map)} segments").classes(f"{theme.DATA} mb-4")

                    # bulk actions
                    with ui.row().classes("space-x-2 mb-4"):
                        ui.button("Approve All", on_click=self.approve_all_segments)\
                          .classes(theme.BTN_OK_SM)
                        ui.button("Save All Edits", on_click=self.save_all_edits)\
                          .classes(theme.BTN_PRIMARY_SM)

                    # per-segment UI
                    for i, seg_id in enumerate(list(self.original_segments_map.keys())):
                        orig = self.original_segments_map[seg_id]
                        trans = self.translated_segments_map[seg_id]
                        seg_info = self.document_run_state.segment_map.get(seg_id, {})
                        location = self._describe_segment_for_editor(i + 1, seg_info)

                        with ui.expansion(location, value=i == 0)\
                                .classes(f"w-full mb-2 {theme.WELL}"):
                            with ui.column().classes("p-3 pb-16 md:pb-4 gap-3"):
                                with ui.row().classes("w-full justify-end items-center gap-2"):
                                    with ui.button(icon="check", on_click=lambda _, s=seg_id: self.approve_segment_callback(s))\
                                            .props("size=sm no-caps").classes(theme.BTN_OK_SM):
                                        ui.tooltip("Approve this translation")
                                    with ui.button(icon="close", on_click=lambda _, s=seg_id: self.decline_segment_callback(s))\
                                            .props("size=sm no-caps").classes(theme.BTN_SECONDARY_SM):
                                        ui.tooltip("Reject — restore the machine translation")
                                    with ui.button(icon="delete_outline", on_click=lambda _, s=seg_id: self.delete_segment_callback(s))\
                                            .props("size=sm no-caps").classes(theme.BTN_DANGER_SM):
                                        ui.tooltip("Remove this segment from the document")

                                with ui.column().classes("w-full"):
                                    ui.label("Original:")\
                                      .classes("p-data")
                                    ui.html(
                                        f'<div class="text-sm p-2 p-panel-source '
                                        f'max-h-20 overflow-y-auto">{orig[:300]}'
                                        f'{"..." if len(orig)>300 else ""}</div>'
                                    )

                                with ui.column().classes("w-full"):
                                    ui.label("Translation:")\
                                      .classes("p-data")
                                    textarea = ui.textarea(value=trans)\
                                      .props("autogrow rows=3")\
                                      .classes(f"w-full text-sm {theme.PANEL_TARGET}")
                                    textarea.segment_id = seg_id
                                    self.segment_editors[seg_id] = textarea

                                with ui.row().classes("w-full items-center gap-2"):
                                    refine = ui.input(placeholder="Refinement instructions (optional)")\
                                      .classes("flex-grow text-sm")
                                    ui.button("Update",
                                              on_click=lambda _, s=seg_id, ta=textarea, ri=refine:
                                                self.update_segment_callback(s, ta, ri)
                                             ).props("size=sm no-caps").classes(theme.BTN_PRIMARY_SM)
                                    ui.button("Re-translate",
                                              on_click=lambda _, s=seg_id, ta=textarea:
                                                self.retranslate_segment_callback(s, ta)
                                             ).props("size=sm no-caps").classes(theme.BTN_SECONDARY_SM)

                # ── FIDELITY NOTICES ────────────────────────────
                # Formatting that could not be carried across translation is
                # stated, not silently dropped.
                self._render_fidelity_notes()

                if self.uploaded_file_extension in {"png", "jpg", "jpeg", "webp"}:
                    ui.separator().classes("my-3")
                    ui.label("Image overlay controls").classes("p-display text-lg")
                    with ui.row().classes("items-center gap-2 flex-wrap"):
                        ui.number("Font size", value=self.overlay_font_size, min=8, max=64, step=1, on_change=lambda e: setattr(self, "overlay_font_size", int(e.value))).classes("w-32")
                        ui.input("Font family", value=self.overlay_font_family, on_change=lambda e: setattr(self, "overlay_font_family", e.value)).classes("w-48")
                        ui.switch("Show original overlay", value=self.overlay_show_original, on_change=lambda e: setattr(self, "overlay_show_original", bool(e.value)))
                        ui.switch("Preview visible", value=self.overlay_preview_visible, on_change=lambda e: setattr(self, "overlay_preview_visible", bool(e.value)) or self.show_result())
                        ui.button("Refresh overlay", on_click=self.refresh_image_overlay).classes(theme.BTN_SECONDARY_SM)
                    if self.overlay_preview_visible and self.document_run_state.output_stream is not None:
                        import base64
                        output_stream = self.document_run_state.output_stream
                        output_stream.seek(0)
                        encoded = base64.b64encode(output_stream.read()).decode("ascii")
                        ui.html(f'<img alt="overlay preview" style="max-width:100%;border:1px solid #ddd;border-radius:8px" src="data:image/png;base64,{encoded}"/>')

                # ── DOWNLOAD & NAV ───────────────────────────────
                ui.separator().classes("my-4")
                with ui.row().classes("justify-center space-x-4 mt-6 flex-wrap"):
                    ui.button("Download Translated File", on_click=self.download_file)\
                      .classes(theme.BTN_PRIMARY)
                    ui.button("Upload Another File", on_click=self.request_refresh_upload_ui)\
                      .classes(theme.BTN_SECONDARY)

        # stats footer
        with self.stats_container:
            with ui.row().classes("w-full max-w-3xl mx-auto gap-4 mt-1"):
                ui.label(f"{self.current_count} segments translated").classes(theme.DATA)
                if self.current_tokens > 0:
                    ui.label(f"{self.current_tokens:,} tokens").classes(theme.DATA)

    def refresh_image_overlay(self):
        try:
            self.backend.process_image(
                BytesIO(self.uploaded_file.getvalue()),
                self.current_target_language,
                show_original=self.overlay_show_original,
                font_size=self.overlay_font_size,
                font_family=self.overlay_font_family,
                run_state=self.document_run_state,
            )
            self.show_result()
        except Exception as ex:
            logging.error(f"[UI] refresh_image_overlay failed: {ex}", exc_info=True)
            ui.notify(f"Overlay refresh failed: {ex}", type="negative")

    def download_file(self):
        try:
            stream = self.get_fresh_download_stream()
            ui.download(stream.read(), f"translated_{self.uploaded_file_name}")
            ui.notify("Download started with all your edits included!", type="positive")
        except Exception as e:
            logging.error(f"[UI] Download error: {e}", exc_info=True)
            ui.notify(f"Download failed: {e}", type="negative")

    # ──────────────────────────────────── SEGMENT ACTIONS ────────────────────────────────────

    def update_segment_callback(self, seg_id, textarea, refine_input):
        try:
            instructions = refine_input.value or None
            # Captured before the call: this is the machine's output, and the
            # difference between it and what the human settles on is the only
            # ground truth this app ever gets.
            machine_output = self.translated_segments_map.get(seg_id, "")
            updated = self.backend.update_segment(
                seg_id, textarea.value, self.current_target_language, instructions,
                run_state=self.document_run_state,
            )
            textarea.value = updated
            self.translated_segments_map[seg_id] = updated
            refine_input.value = ""
            traces.record_edit(
                trace_id=self.document_trace_id, segment_id=seg_id,
                before=machine_output, after=updated,
            )
            ui.notify("Segment updated successfully!", type="positive")
        except Exception as ex:
            logging.error(f"[UI] Error updating segment {seg_id}: {ex}", exc_info=True)
            ui.notify(f"Update failed: {ex}", type="negative")

    def retranslate_segment_callback(self, seg_id, textarea):
        try:
            seg_info = self.document_run_state.segment_map.get(seg_id)
            if not seg_info:
                ui.notify("Segment not found", type="negative")
                return
            ui.notify("Re-­translating...", type="info")
            original = seg_info["original"]
            new_trans = self._run_recorded(
                self._engine_recorder(), self.backend.translate_text,
                (original, self.current_target_language),
                surface=policy.Surface.DOCUMENT, engine=f"hosted:{TEXT_MODEL}",
                chars=len(original or ""))
            self.backend.update_segment(
                seg_id, new_trans, self.current_target_language, run_state=self.document_run_state,
            )
            textarea.value = new_trans
            self.translated_segments_map[seg_id] = new_trans
            ui.notify("Re-translation complete!", type="positive")
        except Exception as ex:
            logging.error(f"[UI] Error re-translating segment {seg_id}: {ex}", exc_info=True)
            ui.notify(f"Re-translation failed: {ex}", type="negative")

    def delete_segment_callback(self, seg_id):
        self._request_confirmation(
            "Delete this segment?",
            "This removes the segment from the generated output and cannot be undone.",
            lambda: self._delete_segment(seg_id),
            confirm_label="Delete Segment"
        )

    def _delete_segment(self, seg_id):
        try:
            self.backend.delete_segment(seg_id, run_state=self.document_run_state)
            self.original_segments_map.pop(seg_id, None)
            self.translated_segments_map.pop(seg_id, None)
            self.current_count = len(self.document_run_state.segment_map)
            ui.notify("Segment deleted successfully!", type="info")
            self.show_result()
        except Exception as ex:
            logging.error(f"[UI] Error deleting segment {seg_id}: {ex}", exc_info=True)
            ui.notify(f"Delete failed: {ex}", type="negative")

    def request_refresh_upload_ui(self):
        if not self.original_segments_map and not self.translated_segments_map:
            self.refresh_upload_ui()
            return
        self._request_confirmation(
            "Start over with a new file?",
            "Current translated segments and unsaved in-page edits will be lost.",
            self.refresh_upload_ui,
            confirm_label="Start Over"
        )

    def approve_segment_callback(self, seg_id):
        try:
            orig = self.original_segments_map.get(seg_id, "")
            trans = self.translated_segments_map.get(seg_id, "")
            self.backend.record_feedback(
                approved=True,
                original=orig,
                translated=trans,
            )
            ui.notify("Segment approved ✓", type="positive")
        except Exception as ex:
            logging.error(f"[UI] Error approving segment {seg_id}: {ex}", exc_info=True)
            ui.notify(f"Approval failed: {ex}", type="negative")

    def decline_segment_callback(self, seg_id):
        try:
            orig = self.original_segments_map.get(seg_id, "")
            trans = self.translated_segments_map.get(seg_id, "")
            self.backend.record_feedback(
                approved=False,
                original=orig,
                translated=trans,
            )
            ui.notify("Segment declined ✗", type="warning")
        except Exception as ex:
            logging.error(f"[UI] Error declining segment {seg_id}: {ex}", exc_info=True)
            ui.notify(f"Decline failed: {ex}", type="negative")

    def approve_all_segments(self):
        try:
            count = 0
            for seg_id in self.original_segments_map.keys():
                orig = self.original_segments_map[seg_id]
                trans = self.translated_segments_map[seg_id]
                self.backend.record_feedback(approved=True, original=orig, translated=trans)
                count += 1
            ui.notify(f"Approved {count} segments ✓", type="positive")
        except Exception as ex:
            logging.error(f"[UI] Error bulk approving: {ex}", exc_info=True)
            ui.notify(f"Bulk approval failed: {ex}", type="negative")

    def _render_fidelity_notes(self) -> None:
        """Show what the rewrite could NOT preserve (DOCX runs / hyperlinks)."""
        notes = list(getattr(self.document_run_state, "fidelity_notes", []) or [])
        if not notes:
            return
        grouped: dict[str, list[str]] = {}
        for note in notes:
            grouped.setdefault(note["message"], []).append(note["location"])
        ui.separator().classes("my-3")
        with ui.expansion(
            f"Formatting notes ({len(notes)})", icon="report_problem"
        ).classes(f"w-full {theme.WELL}"):
            ui.label(
                "Translated wording does not line up with the original text spans, "
                "so some inline formatting could not be carried across:"
            ).classes("text-sm mb-2")
            for message, locations in grouped.items():
                shown = ", ".join(locations[:5])
                if len(locations) > 5:
                    shown += f" (+{len(locations) - 5} more)"
                ui.label(f"- {message}").classes("text-sm")
                ui.label(shown).classes(f"{theme.DATA} text-xs mb-2")

    def save_all_edits(self):
        """Apply every segment editor's CURRENT contents to the document.

        This used to call regenerate_output_stream() and nothing else, which
        re-serialised the document from segments that only the per-segment
        Update button ever wrote. Typing in a box and pressing Save All Edits
        therefore produced a green success toast, an unchanged download, and no
        trace row - the edit was lost from both the document and the dataset.
        Now the editors are the source of truth, edits are recorded as traces
        exactly like the per-segment path, and a save with nothing to save says
        so instead of claiming success.
        """
        try:
            if not self.segment_editors:
                ui.notify(
                    "No segment editors are open, so there is nothing to save. "
                    "Turn on segment review to edit translations.",
                    type="warning",
                )
                return

            changed, failed = [], []
            for seg_id, textarea in list(self.segment_editors.items()):
                if seg_id not in self.document_run_state.segment_map:
                    continue  # deleted since this render
                current = textarea.value or ""
                # The machine's output, captured before we overwrite it: the gap
                # between it and the human's text is the only ground truth here.
                machine_output = self.translated_segments_map.get(seg_id, "")
                if current.strip() == (machine_output or "").strip():
                    continue
                try:
                    updated = self.backend.update_segment(
                        seg_id,
                        current,
                        self.current_target_language,
                        regenerate=False,
                        run_state=self.document_run_state,
                    )
                except Exception as seg_ex:  # one bad segment must not eat the rest
                    logging.error(
                        f"[UI] save_all_edits failed on {seg_id}: {seg_ex}", exc_info=True
                    )
                    failed.append(seg_id)
                    continue
                textarea.value = updated
                self.translated_segments_map[seg_id] = updated
                traces.record_edit(
                    trace_id=self.document_trace_id,
                    segment_id=seg_id,
                    before=machine_output,
                    after=updated,
                )
                changed.append(seg_id)

            # Rebuild the download stream once, after the edits are in the doc.
            self.backend.regenerate_output_stream(run_state=self.document_run_state)

            if failed and not changed:
                ui.notify(
                    f"Nothing was saved: {len(failed)} segment(s) failed to apply.",
                    type="negative",
                )
            elif failed:
                ui.notify(
                    f"Saved {len(changed)} edit(s); {len(failed)} failed and were not applied.",
                    type="warning",
                )
            elif changed:
                ui.notify(f"Saved {len(changed)} edit(s) to the document.", type="positive")
            else:
                ui.notify(
                    "No changes to save - the document already matches the editors.",
                    type="info",
                )
        except Exception as ex:
            logging.error(f"[UI] Error saving edits: {ex}", exc_info=True)
            ui.notify(f"Save failed: {ex}", type="negative")

    @staticmethod
    def _is_technical_error(detail: str) -> bool:
        return len(detail) > 140 or any(
            marker in detail for marker in ("Error code", "Traceback", "Exception", "HTTP/")
        )

    def show_error(self, error, *, retry: Callable[[], None] | None = None):
        """Single error surface: short guidance in the banner; raw provider/
        stack detail tucked into an expansion instead of dumped on the page.
        Every failure path funnels here, so it's also where the in-flight
        Translate button gets re-enabled and any stuck progress spinner
        (rendered by _render_progress_ui before the failure) is cleared."""
        self._set_translate_button_busy(False)
        self.progress_container.clear()
        self.cancel_button = None
        self.result_container.clear()
        self.stats_container.clear()
        detail = str(error).strip() or "Unknown error."
        if not self._is_technical_error(detail):
            self._show_banner(self.result_container, f"Error: {detail}", "negative")
        else:
            self._show_banner(
                self.result_container,
                "The translation service hit an error and nothing was changed. Please try again.",
                "negative",
            )
            with self.result_container:
                with ui.expansion("Technical detail").classes(f"w-full {theme.WELL}"):
                    ui.label(detail).classes(f"{theme.DATA} text-xs break-all")
        if retry is not None:
            with self.result_container:
                ui.button("Try again", on_click=lambda: retry()).classes(f"{theme.BTN_SECONDARY_SM} mt-2")

    # ─────────────────────────── VOICE TRANSLATION API ────────────────────────────────────

    @property
    def recent_threads(self) -> list:
        """Per-visitor (app.storage.user, keyed by session cookie), NOT
        per-process — a shared instance-level deque here previously leaked
        every user's translated text to every other concurrently connected
        user (found in code review, 2026-07-06). Backed by NiceGUI's
        ObservableList, so in-place mutation (.insert, slicing, .clear)
        persists correctly; requires storage_secret on ui.run().

        Falls back to a throwaway, unpersisted list outside a real request
        context (unit tests calling API methods directly with a fake
        Request; any future NiceGUI edge case) — Recent Threads silently
        not recording is fine; breaking the translation response over it
        is not.
        """
        try:
            return app.storage.user.setdefault("recent_threads", [])
        except RuntimeError:
            return []

    @property
    def active_profile(self):
        """This visitor's chosen endpoint, or None for Passage's own default.

        Same session-cookie-scoped storage as recent_threads, for the same
        reason and one sharper: this may hold the user's own API key, which
        must never be shared across sessions or written anywhere a second
        visitor can read. Degrades to None outside a request context so a
        storage failure falls back to the app default instead of raising on
        the translation path.
        """
        try:
            raw = app.storage.user.get("provider_profile")
        except RuntimeError:
            return None
        if not raw:
            return None
        try:
            return provider_profiles.from_stored(raw)
        except (TypeError, ValueError):
            return None

    @property
    def engine_runs(self) -> list:
        """This session's record of which engine served what. Same
        session-cookie storage as recent_threads, and degrades to a throwaway
        list outside a request context — transparency must never be the reason
        a translation fails."""
        try:
            return app.storage.user.setdefault("engine_runs", [])
        except RuntimeError:
            return []

    @property
    def usage_store(self) -> dict:
        """Metered usage for this session (see passage/usage.py)."""
        try:
            return app.storage.user.setdefault("usage", {})
        except RuntimeError:
            return {}

    def _record_engine_run(self, *, surface, engine: str, latency_ms: int, chars: int,
                           runs_store: list | None = None,
                           usage_store: dict | None = None,
                           profile=None) -> policy.EngineRun | None:
        """Book one COMPLETED request against what actually served it.

        Everything comes from `policy.classify_run`, including whether the text
        left the machine and whether Passage pays. This used to decide for
        itself with `engine.startswith("local")`, which meant a cache hit —
        nothing ran, nothing sent — was filed as "sent out" and billed, while
        the same request's JSON body told the user it was free.

        `runs_store`/`usage_store` may be passed in by a caller that already
        has them: the mobile paths do their work on a worker thread, where
        `app.storage.user` raises and the properties degrade to throwaway
        containers — i.e. the record would be written to nothing.
        """
        try:
            run = policy.classify_run(engine, profile if profile is not None
                                      else self.active_profile)
            engine_ledger.record_run(
                self.engine_runs if runs_store is None else runs_store,
                surface=str(getattr(surface, "value", surface)), run=run,
                latency_ms=latency_ms, chars=chars, when=time.time())
            # Only work Passage paid for reaches the counter — a local, cached
            # or BYO run is never recorded, not recorded-then-zeroed.
            usage.record(self.usage_store if usage_store is None else usage_store,
                         chars=chars, metered=run.metered)
            return run
        except Exception:  # never let bookkeeping break a translation
            LOGGER.debug("engine run not recorded", exc_info=True)
            return None

    def _engine_recorder(self):
        """A recorder bound to THIS request's session storage, safe to call
        from a worker thread.

        Documents, images and voice all finish off the event loop, where
        `app.storage.user` is unavailable; capturing the containers here (on
        the request context) is what makes those surfaces visible on /engines
        at all. Until this existed, `_record_engine_run` had exactly one call
        site and the page said "Nothing translated yet" after a document had
        demonstrably been sent to a hosted model on Passage's key.
        """
        runs, used, profile = self.engine_runs, self.usage_store, self.active_profile

        def record(*, surface, engine: str, latency_ms: int, chars: int):
            return self._record_engine_run(
                surface=surface, engine=engine, latency_ms=latency_ms, chars=chars,
                runs_store=runs, usage_store=used, profile=profile)

        return record

    def _cache_hits(self) -> int | None:
        """The backend's cache-hit counter, or None if it can't be read.

        `translate_text` and the vision path don't return an engine label, so
        the only honest way to tell a cache hit from a hosted call is to watch
        this counter across the call. Without it every re-translation of the
        same sentence would be billed as a fresh hosted run.
        """
        try:
            return int(getattr(self.backend.metrics, "cache_hits"))
        except Exception:
            return None

    def _run_recorded(self, record, fn, args=(), *, surface, engine: str,
                      chars: int | None = None, chars_of=None):
        """Call `fn`, then record it under the engine that really served it —
        downgrading to "cache" when the backend's cache answered instead.

        `chars_of` covers the image case, where the source text is inside a
        photo and its length is not knowable until the model has read it.
        """
        before = self._cache_hits()
        started = time.perf_counter()
        result = fn(*args)
        after = self._cache_hits()
        if chars is None:
            try:
                chars = int(chars_of(result)) if chars_of else 0
            except Exception:
                chars = 0
        label = engine
        if not chars:
            label = "none"
        elif before is not None and after is not None and after > before:
            label = "cache"
        try:
            record(surface=surface, engine=label,
                   latency_ms=int((time.perf_counter() - started) * 1000), chars=chars)
        except Exception:
            LOGGER.debug("engine run not recorded", exc_info=True)
        return result

    def _set_active_profile(self, profile) -> None:
        try:
            if profile is None:
                app.storage.user.pop("provider_profile", None)
            else:
                app.storage.user["provider_profile"] = asdict(profile)
        except RuntimeError:
            LOGGER.info("No session storage; provider profile not persisted")

    def _record_thread(self, entry: dict) -> None:
        """Newest-first with dedupe: repeating a translation moves its thread
        to the top instead of stacking duplicates.

        Also collapses live typing into ONE thread. Text mode re-translates on
        every 350ms typing pause, so composing a single sentence used to emit
        ~8 separate threads — one per growing prefix ("The", "The quarterly",
        "The quarterly report"…) — because the dedupe key is the label and a
        growing prefix never matches itself. Recent Threads filled up with
        fragments of one sentence, which reads as history but isn't.
        `_continuation_of` treats a prefix-extension (or a backspace-shortening)
        of the newest thread as the SAME utterance and updates it in place,
        keeping its id and original timestamp.
        """
        entry.setdefault("id", str(uuid.uuid4()))
        threads = self.recent_threads

        superseded = self._continuation_of(entry, threads)
        if superseded is not None:
            # Same utterance still being composed: update in place rather than
            # stack a new row. Keep the id (so a delete button already rendered
            # against it still works) and when the utterance started, but let
            # `when` advance to this edit — it drives both the newest-first sort
            # and the continuation window, which must measure from the last
            # keystroke, not from the start (otherwise composing for longer than
            # the window silently forks a second thread mid-sentence).
            entry["id"] = superseded.get("id", entry["id"])
            entry["started"] = superseded.get("started", superseded.get("when"))
            threads[threads.index(superseded)] = entry
            return

        key = (entry.get("kind"), entry.get("label"), entry.get("language"))
        survivors = [
            t for t in threads
            if (t.get("kind"), t.get("label"), t.get("language")) != key
        ]
        threads.clear()
        threads.extend(survivors)
        threads.insert(0, entry)
        del threads[20:]

    #: How long after the last keystroke a further edit still counts as the
    #: same utterance rather than a new one.
    THREAD_CONTINUATION_WINDOW_S = 180.0

    def _continuation_of(self, entry: dict, threads: list) -> dict | None:
        """Return the existing thread `entry` is a continuation of, else None.

        Only chat threads continue — a document translation is always its own
        thread. A continuation must be the most recent thread (you can only be
        typing in one box at a time), same target language, within the window,
        and its source text must be a prefix-extension of the stored one or a
        shortening of it (backspacing mid-sentence must not spawn a new row).
        """
        if entry.get("kind") != "chat" or not threads:
            return None
        newest = max(threads, key=lambda t: t.get("when", 0))
        if newest.get("kind") != "chat":
            return None
        if newest.get("language") != entry.get("language"):
            return None
        if entry.get("when", 0) - newest.get("when", 0) > self.THREAD_CONTINUATION_WINDOW_S:
            return None
        old = (newest.get("original") or "").strip()
        new = (entry.get("original") or "").strip()
        if not old or not new:
            return None
        return newest if (new.startswith(old) or old.startswith(new)) else None

    def _delete_thread(self, thread_id: str) -> None:
        threads = self.recent_threads
        threads[:] = [t for t in threads if t.get("id") != thread_id]
        self.show_document_list()

    def _record_chat_thread(self, original: str, translated: str, language: str,
                            surface: policy.Surface = policy.Surface.TEXT) -> None:
        # Retention is decided in one place (passage/policy.py) rather than by
        # each call site, because a data-retention rule that drifts is the kind
        # of bug you hear about from someone else.
        if not policy.may_persist(surface, "session_history"):
            return
        self._record_thread({
            "kind": "chat",
            "label": original[:48],
            "original": original,
            "translated": translated,
            "language": language,
            "when": time.time(),
        })

    def _check_api_access(self, request: Request, correlation_id: str) -> JSONResponse | None:
        """Gate /api/* behind the app-issued session token plus a per-IP rate
        limit. Returns an error response to send, or None if allowed."""
        if gate_disabled():
            return None
        token = request.headers.get("x-passage-token", "")
        if not self.api_guard.validate_token(token):
            _log_event("api.rejected_no_token", correlation_id=correlation_id, ip=client_ip(request))
            return JSONResponse(
                {"error": "This API is used by the Passage app. Open the app to translate."},
                status_code=401,
                headers={"X-Correlation-Id": correlation_id},
            )
        if not self.api_guard.allow_request(client_ip(request)):
            _log_event("api.rejected_rate_limited", correlation_id=correlation_id, ip=client_ip(request))
            return JSONResponse(
                {"error": "Too many requests. Try again in a minute."},
                status_code=429,
                headers={"X-Correlation-Id": correlation_id, "Retry-After": "60"},
            )
        return None

    def _inject_api_token(self) -> None:
        """Expose a short-lived token to the page's own fetch() calls."""
        ui.add_head_html(
            f"<script>window.PASSAGE_TOKEN = {json.dumps(self.api_guard.issue_token())};</script>"
        )


    async def api_text_translate(
        self,
        request: Request,
        text: str = Form(...),
        language: str = Form(...)
    ) -> JSONResponse:
        correlation_id = str(uuid.uuid4())
        denied = self._check_api_access(request, correlation_id)
        if denied:
            return denied
        try:
            cleaned_text = (text or "").strip()
            if not cleaned_text:
                return JSONResponse(
                    {"error": "Transcript text is required."},
                    status_code=400,
                    headers={"X-Correlation-Id": correlation_id},
                )
            if len(cleaned_text) > MAX_TEXT_CHARS:
                return JSONResponse(
                    {"error": f"Text is too long ({len(cleaned_text)} characters). The limit is {MAX_TEXT_CHARS} — split it or upload it as a document."},
                    status_code=413,
                    headers={"X-Correlation-Id": correlation_id},
                )
            if not language or language.lower() in ('undefined', 'null', ''):
                language = 'es'
            _log_event(
                "ui.text_translate_requested",
                correlation_id=correlation_id,
                language=language,
                chars=len(cleaned_text),
            )
            # The keystroke path: prefers a local model when one is reachable
            # (free, and measurably faster here), hosted otherwise. The engine
            # comes back with the translation so the UI can show which model
            # answered rather than leaving the user guessing.
            started = time.perf_counter()
            translated, engine = await asyncio.to_thread(
                self.backend.translate_live,
                cleaned_text,
                language,
                self.active_profile,
            )
            run = self._record_engine_run(
                surface=policy.Surface.LIVE_TEXT, engine=engine,
                latency_ms=int((time.perf_counter() - started) * 1000),
                chars=len(cleaned_text)) or policy.classify_run(engine, self.active_profile)
            _log_event(
                "ui.text_translate_succeeded",
                correlation_id=correlation_id,
                language=language,
                engine=engine,
            )
            # LIVE_TEXT keeps a SESSION entry (thread continuation already
            # collapses one sentence into one row) but is never persisted
            # durably — see passage/policy.py for the distinction.
            self._record_chat_thread(cleaned_text, translated, language,
                                     surface=policy.Surface.LIVE_TEXT)
            return JSONResponse(
                {
                    "original_text": cleaned_text,
                    "translated_text": translated,
                    "target_language": language,
                    "engine": engine,
                    # Both derived from the engine that ACTUALLY served this
                    # request, never from a reachability snapshot. They used
                    # to come from `local_snapshot["chosen_model"]`, so one
                    # body could report engine "hosted:gpt-5.4-nano" while
                    # promising the text ran on a local model and was not
                    # metered. A claim about where data went has to be a
                    # function of what ran.
                    "privacy": run.privacy,
                    "metered": run.metered,
                    "ran_on": run.ran.value,
                    "left_machine": run.left_machine,
                },
                headers={"X-Correlation-Id": correlation_id},
            )
        except Exception as e:
            _log_event("ui.text_translate_failed", correlation_id=correlation_id, error=str(e))
            return JSONResponse(
                {"error": f"Transcript translation failed: {e}"},
                status_code=500,
                headers={"X-Correlation-Id": correlation_id},
            )

    async def api_text_translate_stream(
        self,
        request: Request,
        text: str = Form(...),
        language: str = Form(...),
    ) -> Response:
        correlation_id = str(uuid.uuid4())
        denied = self._check_api_access(request, correlation_id)
        if denied:
            return denied
        cleaned_text = (text or "").strip()
        if not cleaned_text:
            return JSONResponse(
                {"error": "Transcript text is required."},
                status_code=400,
                headers={"X-Correlation-Id": correlation_id},
            )
        if len(cleaned_text) > MAX_TEXT_CHARS:
            return JSONResponse(
                {"error": f"Text is too long ({len(cleaned_text)} characters). The limit is {MAX_TEXT_CHARS} — split it or upload it as a document."},
                status_code=413,
                headers={"X-Correlation-Id": correlation_id},
            )
        if not language or language.lower() in ('undefined', 'null', ''):
            language = 'es'

        live_streaming_enabled = os.getenv("LIVE_TEXT_STREAMING", "false").lower() in {"1", "true", "yes", "on"}
        threshold = int(os.getenv("LIVE_TEXT_STREAMING_CHAR_THRESHOLD", "250"))
        should_stream = live_streaming_enabled or len(cleaned_text) >= threshold

        # This endpoint used to pass NO profile, so a visitor who had
        # configured their own endpoint had their transcript translated on
        # PASSAGE'S key — the credential boundary the picker exists to draw,
        # crossed silently. Resolved here, on the request context (session
        # storage is unreadable from a worker thread), applied inside it.
        profile = self.active_profile
        engine = self._text_engine_label(profile)
        started = time.perf_counter()

        if not should_stream:
            translated = await asyncio.to_thread(
                self._translate_text_on_profile, cleaned_text, language, profile)
            self._record_engine_run(
                surface=policy.Surface.LIVE_TEXT, engine=engine,
                latency_ms=int((time.perf_counter() - started) * 1000),
                chars=len(cleaned_text))
            self._record_chat_thread(cleaned_text, translated, language)
            return JSONResponse(
                {
                    "fallback": True,
                    "original_text": cleaned_text,
                    "translated_text": translated,
                    "target_language": language,
                    "engine": engine,
                },
                # The /voice page has a reader for this header; until now no
                # server code produced it, so the transcript engine line was
                # permanently stuck on "the server did not name the engine".
                headers={"X-Correlation-Id": correlation_id, "X-Text-Engine": engine},
            )

        async def event_generator():
            yield f"event: start\ndata: {json.dumps({'target_language': language, 'original_text': cleaned_text, 'engine': engine})}\n\n"
            try:
                final_text, partials = await asyncio.to_thread(
                    self._stream_translate_text_on_profile,
                    cleaned_text,
                    language,
                    profile,
                )
                for partial in partials[:-1]:
                    yield f"event: partial\ndata: {json.dumps({'translated_text': partial})}\n\n"
                self._record_engine_run(
                    surface=policy.Surface.LIVE_TEXT, engine=engine,
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    chars=len(cleaned_text))
                self._record_chat_thread(cleaned_text, final_text, language)
                yield f"event: complete\ndata: {json.dumps({'translated_text': final_text, 'canonical': True, 'engine': engine})}\n\n"
            except Exception as e:
                yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Correlation-Id": correlation_id,
                     "X-Text-Engine": engine},
        )

    @staticmethod
    def _text_engine_label(profile) -> str:
        """Name the engine a text call will ACTUALLY run on.

        Derived from the profile that is about to be applied rather than
        assumed hosted, so the label, the header and the work cannot disagree.
        """
        if profile is not None and getattr(profile, "kind", None) != provider_profiles.KIND_APP:
            return profile.describe()
        return f"hosted:{TEXT_MODEL}"

    def _translate_text_on_profile(self, text: str, language: str, profile) -> str:
        """Runs in a worker thread: contextvars do not propagate into a thread,
        so the endpoint override is established HERE, not at the route."""
        with self.backend.using_profile(profile):
            return self.backend.translate_text(text, language)

    def _stream_translate_text_on_profile(self, text: str, language: str, profile):
        with self.backend.using_profile(profile):
            return self.backend.stream_translate_text(text, language)

    async def api_image_translate(
        self,
        request: Request,
        file: UploadFile = File(...),
        language: str = Form(...),
    ) -> JSONResponse:
        correlation_id = str(uuid.uuid4())
        denied = self._check_api_access(request, correlation_id)
        if denied:
            return denied
        try:
            payload = await file.read()
            if len(payload) > MAX_UPLOAD_BYTES:
                return JSONResponse(
                    {"error": "Image is too large. The limit is 8 MB."},
                    status_code=413,
                    headers={"X-Correlation-Id": correlation_id},
                )
            recorder = self._engine_recorder()
            result = await asyncio.to_thread(
                self._run_recorded, recorder,
                self.backend.translate_image_text_blocks,
                (payload, file.filename or "uploaded_image", language or "es"),
                surface=policy.Surface.IMAGE, engine=f"hosted:{VISION_MODEL}",
                chars=None, chars_of=_image_source_chars,
            )
            return JSONResponse(result, headers={"X-Correlation-Id": correlation_id})
        except ValueError as err:
            return JSONResponse(
                {"error": str(err)},
                status_code=400,
                headers={"X-Correlation-Id": correlation_id},
            )
        except Exception as err:
            return JSONResponse(
                {"error": f"Image translation failed: {err}"},
                status_code=500,
                headers={"X-Correlation-Id": correlation_id},
            )

    async def api_me(self, request: Request) -> JSONResponse:
        """Identity from the Authorization header, or anonymous. Not gated
        by the paid-API session token — this costs nothing and a client
        may need it before it has a page-issued PASSAGE_TOKEN."""
        user_id, email = identity_from_auth_header(request.headers.get("authorization"))
        if user_id is None:
            return JSONResponse({"authenticated": False, "user_id": None, "email": None})
        return JSONResponse({"authenticated": True, "user_id": user_id, "email": email})

    async def api_health(self, request: Request) -> JSONResponse:
        """Liveness plus what this instance can actually do right now.

        Commit 546a291 said it added this; its diff was two lines in
        passage/__init__.py and the route 404'd, so anything health-checking
        the deploy was checking nothing. Ungated on purpose — a health check
        that needs a page-issued token is not a health check.

        It reports capability, not just "alive": whether a hosted provider is
        configured and which local models are reachable, because "up but
        unable to translate" is the failure worth catching.
        """
        # ONE probe, on a worker thread. This route is unauthenticated by
        # design, and the previous version took four blocking probes directly
        # on the event loop: a single GET against an unreachable Ollama froze
        # the whole server for 8-15 s (measured), which is a denial of service
        # anyone could trigger. Nothing here may block the loop.
        snapshot = await local_snapshot_async(self.backend)
        report = await asyncio.to_thread(collect_diagnostics, snapshot)
        return JSONResponse({
            "status": "ok",
            "version": passage_version,
            "hosted_provider": self.backend.provider is not None,
            "local_models": list(snapshot["models"]),
            "local_default": snapshot["chosen_model"],
            # How old the local answer is. Reusing a probe is only acceptable
            # if the reader is told; an undated cache turns a diagnostic into
            # a confident guess.
            "local_probe": snapshot_payload(snapshot),
            # PORTABILITY_PLAN.md §5 Phase B. Kept alongside the original keys
            # rather than replacing them: something out there health-checks this
            # route, and a diagnostic is not worth breaking a deploy probe for.
            # Ungated like the rest of /api/health, and carrying no secret and
            # no home-directory path, so it is safe for an unauthenticated
            # caller (see passage/diagnostics.redact_path).
            "diagnostics": report,
        })

    def diagnostics_page(self) -> None:
        """One page a real device can answer "does this work here?" with.

        The plan (§4) is blunt that real iOS Safari, real Android Chrome, OS
        permission dialogs and actual mic hardware are NOT reachable from CI or
        from this development machine. For those the diagnostic IS the test, so
        this page is built to be read and copied on a phone: a plain text block,
        a copy button, no interaction required beyond loading it.

        The browser half is gathered in the browser (see
        diagnostics.BROWSER_PROBE_JS) and merged in, because the server cannot
        observe a secure context or the sample rate a browser granted.
        """
        self._inject_theme()
        # Same rule as /engines and /api/health: no probe on the render path.
        # This page took 7.8 s against a silent endpoint and blocked the loop
        # for everyone while it did. Start from the last probe (its age is
        # printed under the block) and refresh from a thread.
        snapshot = _cached_snapshot(self.backend) or _pending_snapshot()
        report = collect_diagnostics(snapshot)

        with ui.header().classes(f"items-center {theme.HEADER} px-4 py-1"):
            with ui.row().classes("w-full items-center gap-3"):
                ui.html(f'<span class="{theme.WORDMARK}">Passage<b>.</b></span>')\
                    .on("click", lambda: ui.navigate.to("/"))
                ui.element("div").classes("p-header-sep")
                ui.label("Diagnostics").classes(theme.DATA)

        with ui.column().classes("w-full items-center p-4"):
            with ui.column().classes("w-full max-w-3xl gap-4"):
                ui.label("What this machine resolved").classes("p-display text-xl")
                ui.label(
                    "Every line is a probe of this device, not a guess from its "
                    "name. Copy the block and paste it back with any report."
                ).classes("text-sm p-muted-text")

                text_area = ui.markdown(
                    f"```\n{diagnostics.format_text(report)}\n```"
                ).classes("w-full text-xs")

                state: dict[str, Any] = {"report": report}

                age_label = ui.label(describe_snapshot_age(snapshot))\
                    .classes("text-xs p-muted-text")

                async def refresh_local_llm() -> None:
                    """The local-LLM section, re-derived from ONE probe taken
                    on a worker thread. Never blocks the page or the loop."""
                    snap = await local_snapshot_async(self.backend)
                    fresh = await asyncio.to_thread(collect_diagnostics, snap)
                    fresh["browser"] = state["report"].get(
                        "browser", diagnostics.browser_placeholder())
                    state["report"] = fresh
                    age_label.set_text(describe_snapshot_age(snap))
                    text_area.set_content(
                        f"```\n{diagnostics.format_text(fresh)}\n```")

                async def gather_browser() -> None:
                    try:
                        browser = await ui.run_javascript(
                            diagnostics.BROWSER_PROBE_JS, timeout=10.0)
                    except Exception as error:  # pragma: no cover - browser only
                        browser = {"collected": False,
                                   "note": f"browser probe failed: {error}"}
                    if isinstance(browser, dict):
                        state["report"]["browser"] = browser
                        text_area.set_content(
                            f"```\n{diagnostics.format_text(state['report'])}\n```")

                async def copy_all() -> None:
                    payload = json.dumps(diagnostics.format_text(state["report"]))
                    await ui.run_javascript(
                        f"navigator.clipboard && navigator.clipboard.writeText({payload})")
                    ui.notify("Copied")

                ui.button("Copy", on_click=copy_all).classes(theme.BTN_SECONDARY_SM)
                ui.timer(0.05, refresh_local_llm, once=True)
                ui.timer(0.1, gather_browser, once=True)


def start_ui() -> None:
    """App bootstrap: one shared backend/api_guard for the whole process
    (translation cache, job store — already job-id-keyed, safe to share),
    but a FRESH TranslationUI() per page load for everything else.

    TranslationUI used to be a single instance whose main_page() reassigned
    self.upload_container/progress_container/result_container/stats_container
    (and every other piece of per-visitor state — uploaded_file, segments,
    mode, languages) on every client's page load. Two people using the app
    at once raced for those same attributes: live-verified (2026-07-06) that
    a second visitor loading "/" while a first visitor's document job was
    still polling made the first visitor's job silently never render its
    result — the completion callback wrote into whichever client's
    containers happened to be current by the time it fired, not
    necessarily its own. Constructing a new TranslationUI() per page load
    gives each client truly private state; the API-route methods never
    touch UI containers, so binding them to one throwaway instance is safe.
    """
    shared_backend = TranslationBackend()
    shared_api_guard = ApiGuard()

    def new_page_ui() -> "TranslationUI":
        return TranslationUI(backend=shared_backend, api_guard=shared_api_guard)

    api_service = new_page_ui()
    app.add_api_route("/api/voice_translate", api_service.api_voice_translate, methods=["POST"])
    app.add_api_route("/api/text_translate", api_service.api_text_translate, methods=["POST"])
    app.add_api_route(
        "/api/text_translate_stream", api_service.api_text_translate_stream, methods=["POST"],
    )
    app.add_api_route("/api/image_translate", api_service.api_image_translate, methods=["POST"])
    # Phase 4 (accounts): identity-only, not gated by the paid-API token —
    # degrades to anonymous with zero config until SUPABASE_URL is set.
    app.add_api_route("/api/me", api_service.api_me, methods=["GET"])
    app.add_api_route("/api/health", api_service.api_health, methods=["GET"])

    # `mode` is a plain FastAPI-style query param (?mode=Document) — NiceGUI
    # wires ui.page function parameters the same way. See main_page()/
    # _go_workspace() for why this replaces setting self.input_mode directly.
    def index(mode: str | None = None) -> None:
        new_page_ui().main_page(mode=mode)

    ui.page("/")(index)
    ui.page("/voice")(lambda: new_page_ui().voice_translation_page())
    ui.page("/compare")(lambda: new_page_ui().compare_page())
    ui.page("/engines")(lambda: new_page_ui().engines_page())
    ui.page("/diagnostics")(lambda: new_page_ui().diagnostics_page())
    # /mobile is retired — one responsive layout; keep old bookmarks working
    ui.page("/mobile")(lambda: ui.navigate.to("/"))
    app.add_static_files("/static", str(Path(__file__).resolve().parent / "static"))
    # run on single port; Cloud Run injects PORT
    ui.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8080")),
        title="Passage",
        favicon=str(Path(__file__).resolve().parent / "static" / "favicon.svg"),
        reload=False,
        storage_secret=_STORAGE_SECRET,
    )


if __name__ in {"__main__", "__mp_main__"}:
    start_ui()
