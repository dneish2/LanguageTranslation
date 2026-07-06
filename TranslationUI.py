import asyncio
import logging
import os
import glob
import json
import time
import uuid
from collections import deque
from io import BytesIO
from pathlib import Path
from threading import Thread
from typing import Any
from urllib.parse import quote

from nicegui import ui, app
from fastapi import Request, UploadFile, File, Form
from starlette.responses import Response, JSONResponse, StreamingResponse

import theme
from api_security import ApiGuard, client_ip, gate_disabled, MAX_TEXT_CHARS, MAX_UPLOAD_BYTES
from TranslationBackend import TranslationBackend

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("translation.ui")

# Static language list for type-ahead fields. Free text stays allowed —
# the backend takes any language name; this is autofill, not a whitelist.
LANGUAGES = [
    "English", "Spanish", "French", "German", "Italian", "Portuguese",
    "Dutch", "Russian", "Ukrainian", "Polish", "Turkish", "Arabic",
    "Hebrew", "Hindi", "Chinese", "Japanese", "Korean", "Vietnamese",
    "Thai", "Indonesian", "Swedish", "Norwegian", "Danish", "Finnish",
    "Greek", "Czech", "Romanian", "Hungarian",
]


def _log_event(event: str, correlation_id: str | None = None, **fields: Any) -> None:
    payload: dict[str, Any] = {"event": event, **fields}
    if correlation_id:
        payload["correlation_id"] = correlation_id
    LOGGER.info(json.dumps(payload, default=str))


class TranslationUI:
    def __init__(self):
        # ── CORE BACKEND ─────────────────────────────────────────────────
        self.backend = TranslationBackend()
        self.api_guard = ApiGuard()

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
        self.active_job_id: str | None = None
        self.job_poll_timer = None

        # ── SEGMENT EDITING ─────────────────────────────────────────────
        self.original_segments_map: dict[str, str] = {}
        self.translated_segments_map: dict[str, str] = {}

        # ── DRAWER & ADVANCED MODE ──────────────────────────────────────
        self.drawer = None
        # Segment review renders whenever a document has segments; the old
        # Default/Advanced toggle confused more than it gated.
        self.advanced_mode = True
        # Threads: chats (text translations) and documents, newest first.
        # In-memory until per-user workspaces land (Phase 4).
        self.recent_threads = deque(maxlen=20)
        self.mode_tab_row = None
        self.text_output_label = None
        self.document_editor_dialog = None
        self.document_editor_container = None
        self.document_editor_inputs: dict[str, Any] = {}
        self.document_editor_original_values: dict[str, str] = {}

        # ── MOBILE FLOW ────────────────────────────────────────────────
        self.mobile_mode = False
        self.input_mode = "Document"
        # DOM id prefix for the Text-mode workspace elements; must match the
        # hardcoded `scope` in _inject_workspace_text_live_translation_js.
        self.text_status_scope = "workspace_text"
        self.mobile_input_mode = "Document"
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

        # ── UI CONSISTENCY STANDARDS (Passage "Press" tokens, theme.py) ──
        self.button_primary_classes = theme.BTN_PRIMARY
        self.button_secondary_classes = theme.BTN_SECONDARY
        self.banner_classes = theme.BANNER

        # ── USAGE STATS ─────────────────────────────────────────────────
        self.current_count = 0
        self.current_tokens = 0

        # ── VOICE TRANSLATION API ──────────────────────────────────────
        # registers /api/voice_translate on the same port
        app.add_api_route(
            "/api/voice_translate",
            self.api_voice_translate,
            methods=["POST"],
        )
        app.add_api_route(
            "/api/text_translate",
            self.api_text_translate,
            methods=["POST"],
        )
        app.add_api_route(
            "/api/text_translate_stream",
            self.api_text_translate_stream,
            methods=["POST"],
        )
        app.add_api_route(
            "/api/image_translate",
            self.api_image_translate,
            methods=["POST"],
        )

    def start_ui(self):
        # register pages
        ui.page("/")(self.main_page)
        ui.page("/voice")(self.voice_translation_page)
        ui.page("/mobile")(self.mobile_page)
        app.add_static_files("/static", str(Path(__file__).resolve().parent / "static"))
        # run on single port; Cloud Run injects PORT
        ui.run(
            host="0.0.0.0",
            port=int(os.getenv("PORT", "8080")),
            title="Passage",
            favicon=str(Path(__file__).resolve().parent / "static" / "favicon.svg"),
        )

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

    def _inject_auto_device_routing(self, page: str) -> None:
        ui.add_body_html(f"""
<script>
(() => {{
    const page = {json.dumps(page)};
    const currentPath = window.location.pathname;
    const uaMobile = /Android|iPhone|iPad|iPod|Mobile|Opera Mini|IEMobile/i.test(navigator.userAgent || '');
    const viewportMobile = window.matchMedia('(max-width: 768px)').matches;
    const coarsePointer = window.matchMedia('(pointer: coarse)').matches;
    const isMobile = uaMobile || (viewportMobile && coarsePointer);
    const targetPath = isMobile ? '/mobile' : '/';
    const key = 'device_auto_redirect';

    if ((page === 'desktop' && targetPath === '/mobile' && currentPath !== '/mobile')
        || (page === 'mobile' && targetPath === '/' && currentPath !== '/')) {{
        const redirected = sessionStorage.getItem(key);
        if (!redirected) {{
            sessionStorage.setItem(key, '1');
            window.location.replace(targetPath);
            return;
        }}
    }}
    sessionStorage.removeItem(key);
}})();
</script>
        """)

    def _inject_voice_frontend_helpers(self) -> None:
        ui.add_head_html("""
<script>
window.voiceUx = window.voiceUx || (() => {
    const states = {
        READY: 'Ready: record audio or paste transcript',
        REQUESTING_AUDIO: 'Requesting microphone access…',
        RECORDING: 'Recording audio…',
        STOPPING: 'Stopping recording…',
        PROCESSING_AUDIO: 'Processing audio…',
        TRANSLATING_TEXT: 'Translating transcript…',
        COMPLETE: 'Complete: output ready',
    };

    const resolve = (scope, key) => document.getElementById(`${scope}_${key}`);

    function setStatus(scope, message) {
        const node = resolve(scope, 'status');
        if (node) node.textContent = message;
    }

    function setDebug(scope, message) {
        const node = resolve(scope, 'debug');
        if (node) node.textContent = message || '';
    }

    function setRecordingButtons(scope, recording) {
        const start = resolve(scope, 'start_recording');
        const stop = resolve(scope, 'stop_recording');
        if (start) {
            start.disabled = recording;
            start.style.opacity = recording ? '0.5' : '1';
        }
        if (stop) {
            stop.disabled = !recording;
            stop.style.opacity = recording ? '1' : '0.5';
        }
    }

    function init(scope) {
        setStatus(scope, states.READY);
        setDebug(scope, '');
        setRecordingButtons(scope, false);
    }

    return { states, init, setStatus, setDebug, setRecordingButtons };
})();
</script>
        """)

    def _render_voice_status_block(self, scope: str) -> None:
        ui.label("Status").classes("text-sm font-semibold text-gray-700 mt-3")
        ui.label("")\
            .classes("text-base text-gray-800")\
            .props(f"id={scope}_status")
        ui.label("Debug").classes("text-xs font-semibold text-gray-600 mt-1")
        ui.label("")\
            .classes("text-xs text-gray-500 min-h-[20px]")\
            .props(f"id={scope}_debug")

    def _inject_workspace_text_live_translation_js(self) -> None:
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
            setStatus(stateLabels.UPDATED);
        } catch (err) {
            if (token !== activeRequestToken) return;
            setStatus(`${stateLabels.ERROR}: ${err?.message || 'unknown error'}`);
        }
    }
    function scheduleDebouncedTranslation() {
        if (debounceTimer) window.clearTimeout(debounceTimer);
        debounceTimer = window.setTimeout(requestTranslation, DEBOUNCE_MS);
    }
    function init() {
        const source = sourceEl();
        const manualBtn = byId(`${scope}_manual_translate`);
        if (source && !source.dataset.liveTranslateBound) {
            source.dataset.liveTranslateBound = '1';
            source.addEventListener('input', scheduleDebouncedTranslation);
        }
        if (manualBtn && !manualBtn.dataset.liveTranslateBound) {
            manualBtn.dataset.liveTranslateBound = '1';
            manualBtn.addEventListener('click', requestTranslation);
        }
        setStatus(stateLabels.READY);
    }
    window.workspaceTextLiveTranslation = { init, requestTranslation };
    window.addEventListener('load', init);
    window.setTimeout(init, 0);
})();
</script>
        """)

    # ──────────────────────────────────── MAIN PAGE ─────────────────────────────────────────

    def main_page(self):
        self.mobile_mode = False
        self._inject_theme()
        self._inject_api_token()
        self._inject_auto_device_routing("desktop")
        # Header: wordmark goes home; the mode tabs are the only navigation.
        with ui.header().classes(f"items-center {theme.HEADER} px-4 py-1"):
            with ui.row().classes("w-full items-center gap-3"):
                ui.html(f'<span class="{theme.WORDMARK}">Passage<b>.</b></span>')\
                    .on("click", lambda: ui.navigate.to("/"))
                ui.element("div").classes("p-header-sep")
                self.mode_tab_row = ui.row().classes("items-center gap-0")
                self._render_mode_tabs()

        # Drawer for recent docs
        self.drawer = ui.drawer(side='left').classes(theme.DRAWER)
        with self.drawer:
            self.show_document_list()

        # Default workspace page
        with ui.column().classes("w-full h-full items-center justify-start p-4"):
            self.upload_container = ui.column().classes("w-full max-w-6xl")
            self.progress_container = ui.column().classes("w-full max-w-6xl")
            self.result_container = ui.column().classes("w-full max-w-6xl")
            self.stats_container = ui.column().classes("w-full max-w-6xl")
            self.refresh_upload_ui()

        self.build_document_editor_dialog()

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

    def set_workspace_mode(self, mode: str) -> None:
        self.input_mode = mode
        self.mobile_input_mode = mode
        self._render_mode_tabs()
        self.refresh_upload_ui()

    def _show_banner(self, container, message: str, kind: str = "info") -> None:
        with container:
            ui.label(message).classes(self.banner_classes.get(kind, self.banner_classes["info"]))

    def swap_languages(self):
        source = self.source_language_input.value if self.source_language_input else self.current_source_language
        target = self.target_language_input.value if self.target_language_input else self.current_target_language
        self.current_source_language, self.current_target_language = target, source
        self.refresh_upload_ui()

    def build_document_editor_dialog(self):
        if self.document_editor_dialog is not None:
            return

        with ui.dialog() as dialog:
            self.document_editor_dialog = dialog
            with ui.card().classes("w-[900px] max-w-full max-h-[80vh] flex flex-col"):
                ui.label("Editable Document View")\
                    .classes("text-xl font-semibold text-gray-800")
                ui.label("Review every translated segment in a continuous document layout.")\
                    .classes("text-sm text-gray-600 mb-2")

                with ui.scroll_area().classes(f"w-full flex-1 {theme.WELL} p-3 pb-28 md:pb-6"):
                    self.document_editor_container = ui.column().classes("space-y-4")

                with ui.row().classes("justify-end space-x-2 mt-4"):
                    ui.button("Cancel", on_click=self.request_close_document_editor)\
                        .classes(theme.BTN_SECONDARY_SM)
                    ui.button("Save Changes", on_click=self.save_document_editor)\
                        .classes(theme.BTN_PRIMARY_SM)

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

    def populate_document_editor(self):
        if not self.document_editor_container:
            return

        self.document_editor_container.clear()
        self.document_editor_inputs.clear()
        self.document_editor_original_values.clear()

        segments = list(self.backend.segment_map.items())
        if not segments:
            with self.document_editor_container:
                ui.label("No segments available yet. Translate a document first.")\
                    .classes("text-sm text-gray-600")
            return

        current_page = None
        with self.document_editor_container:
            segment_count = len(segments)
            for index, (seg_id, seg_info) in enumerate(segments, start=1):
                seg_type = seg_info.get("type")
                if seg_type == "pdf_block":
                    page_idx = seg_info.get("page_idx")
                    if page_idx is not None and current_page != page_idx:
                        current_page = page_idx
                        ui.label(f"Page {page_idx + 1}")\
                            .classes("text-sm font-semibold text-gray-700 mt-2")

                header = self._describe_segment_for_editor(index, seg_info)
                translated_value = self.translated_segments_map.get(
                    seg_id,
                    seg_info.get("translated", "")
                )
                self.document_editor_original_values[seg_id] = translated_value or ""

                with ui.expansion(f"Step {index}/{segment_count} — {header}", value=index == 1)\
                        .classes(f"w-full {theme.WELL}"):
                    with ui.column().classes("space-y-2 p-2 pb-16 md:pb-4"):
                        ui.label("Edit translation below (mobile-safe layout).")\
                            .classes("text-xs text-gray-500")
                        textarea = ui.textarea(value=translated_value or "")\
                            .props("autogrow rows=3")\
                            .classes(f"w-full text-base {theme.PANEL_TARGET} p-2")
                        textarea.segment_id = seg_id
                        self.document_editor_inputs[seg_id] = textarea

    def open_document_editor(self):
        if not self.document_editor_dialog:
            ui.notify("Document editor not initialised yet.", type="warning")
            return
        self.populate_document_editor()
        if not self.document_editor_inputs:
            ui.notify("No segments available to edit yet.", type="warning")
            return
        self.document_editor_dialog.open()

    def close_document_editor(self):
        if self.document_editor_dialog:
            self.document_editor_dialog.close()
        self.document_editor_inputs.clear()
        self.document_editor_original_values.clear()

    def _request_confirmation(self, title, message, on_confirm, confirm_label="Confirm"):
        with ui.dialog() as dialog:
            with ui.card().classes("w-[420px] max-w-full"):
                ui.label(title).classes("text-lg font-semibold text-gray-800")
                ui.label(message).classes("text-sm text-gray-600")
                with ui.row().classes("justify-end space-x-2 mt-4"):
                    ui.button("Cancel", on_click=dialog.close)\
                        .classes(theme.BTN_SECONDARY_SM)

                    def confirm_and_close():
                        dialog.close()
                        on_confirm()

                    ui.button(confirm_label, on_click=confirm_and_close)\
                        .classes(theme.BTN_DANGER_SM)
        dialog.open()

    def _document_editor_has_unsaved_changes(self):
        for seg_id, textarea in self.document_editor_inputs.items():
            current = textarea.value or ""
            original = self.document_editor_original_values.get(seg_id, "")
            if current != original:
                return True
        return False

    def request_close_document_editor(self):
        if not self.document_editor_inputs:
            self.close_document_editor()
            return

        if not self._document_editor_has_unsaved_changes():
            self.close_document_editor()
            return

        self._request_confirmation(
            "Discard unsaved edits?",
            "You have unsaved document editor changes. Close anyway and discard them?",
            self.close_document_editor,
            confirm_label="Discard Changes"
        )

    def save_document_editor(self):
        if not self.document_editor_inputs:
            self.close_document_editor()
            return

        try:
            changed = 0
            for seg_id, textarea in self.document_editor_inputs.items():
                new_text = textarea.value or ""
                if new_text != self.translated_segments_map.get(seg_id, ""):
                    self.backend.update_segment(
                        seg_id,
                        new_text,
                        self.current_target_language or "",
                        regenerate=False
                    )
                    self.translated_segments_map[seg_id] = new_text
                    changed += 1

            if changed:
                self.backend.regenerate_output_stream()
                ui.notify(
                    f"Saved {changed} segment{'s' if changed != 1 else ''} from the document editor.",
                    type="positive"
                )
            else:
                ui.notify("No changes detected in the document editor.", type="warning")
        except Exception as ex:
            logging.error(f"[UI] Error saving document editor changes: {ex}", exc_info=True)
            ui.notify(f"Failed to save document edits: {ex}", type="negative")
        finally:
            self.close_document_editor()
            if self.advanced_mode and self.original_segments_map and not self.mobile_mode:
                self.show_result()

    def show_document_list(self):
        # Threads = chats (text translations, in-memory) + translated documents
        # (files, until per-user storage lands in Phase 4), newest first.
        self.drawer.clear()
        with ui.row().classes("w-full items-center justify-between mb-1"):
            ui.label("Recent Threads").classes("p-display text-lg")
            ui.button(icon="refresh", on_click=self.show_document_list)\
                .props("flat round size=sm")\
                .classes("p-mode-tab")

        threads: list[dict] = []
        for chat in self.recent_threads:
            threads.append(chat)
        for fn in sorted(glob.glob("translated_*"), key=os.path.getmtime, reverse=True)[:20]:
            threads.append({
                "kind": "document",
                "label": fn,
                "when": os.path.getmtime(fn),
                "file": fn,
            })
        threads.sort(key=lambda t: t.get("when", 0), reverse=True)

        if not threads:
            ui.label("Nothing here yet — translations you run will appear as threads.")\
                .classes("text-sm text-gray-600")
            return
        for t in threads[:20]:
            if t["kind"] == "chat":
                handler = lambda _, thread=t: self._open_chat_thread(thread)
                kind_label = f"chat · {t.get('language', '')}"
            else:
                handler = lambda _, f=t["file"]: self.load_processed_document(f)
                kind_label = "document"
            with ui.button(on_click=handler).classes("p-thread-item"):
                with ui.column().classes("gap-0 items-start"):
                    ui.label(t["label"][:44]).classes("text-sm")
                    ui.label(kind_label).classes("p-thread-kind")

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

    def load_processed_document(self, filename):
        # load a file that was already translated
        for c in (self.progress_container, self.result_container, self.stats_container):
            c.clear()
        if not os.path.exists(filename):
            ui.notify(f"File {filename} not found.")
            return
        with open(filename, 'rb') as f:
            data = f.read()
        self.uploaded_file = BytesIO(data)
        self.uploaded_file_name = filename
        self.uploaded_file_extension = filename.split(".")[-1].lower()
        ui.notify(f"Loaded processed file: {filename}")
        self.handle_translation_processed()

    def refresh_upload_ui(self):
        # reset UI + clear segments
        self.upload_container.clear()
        self.progress_container.clear()
        self.result_container.clear()
        self.stats_container.clear()
        self.backend.segment_map.clear()

        self.render_unified_workspace()

    def render_unified_workspace(self):
        max_width = "max-w-md" if self.mobile_mode else "max-w-6xl"
        with self.upload_container:
            with ui.column().classes(f"w-full {max_width} gap-3"):
                with ui.row().classes(f"w-full items-end gap-2 flex-wrap {theme.WELL} p-3"):
                    self.source_language_input = ui.input(
                        label="From",
                        value=self.current_source_language or "English",
                        placeholder="Source language",
                        autocomplete=LANGUAGES,
                    ).classes("min-w-[120px] flex-1")
                    ui.button("⇄", on_click=self.swap_languages).classes(self.button_secondary_classes)
                    self.target_language_input = ui.input(
                        label="To",
                        value=self.current_target_language or "Spanish",
                        placeholder="Target language",
                        autocomplete=LANGUAGES,
                    ).classes("min-w-[120px] flex-1")
                    if self.mobile_mode:
                        mode_selector = ui.toggle(
                            {"Text": "Text", "Document": "Document", "Image/Camera": "Image/Camera"},
                            value=self.input_mode,
                        ).classes("h-10 min-h-0 px-2 rounded-md")
                        mode_selector.on_value_change(lambda e: self.set_mobile_input_mode(e.value))
                    translate_button = ui.button("Translate", on_click=self.start_mobile_translation).classes(self.button_primary_classes)
                    translate_button.props(f"id={self.text_status_scope}_manual_translate")

                # Facing pages: source sits on paper, translation on panel.
                with ui.grid(columns=1 if self.mobile_mode else 2).classes("w-full gap-3"):
                    with ui.column().classes(f"w-full p-4 gap-2 {theme.PANEL_SOURCE}"):
                        ui.label("Source").classes(theme.DATA)
                        self._render_source_input_panel()
                    with ui.column().classes(f"w-full p-4 gap-2 {theme.PANEL_TARGET}"):
                        with ui.row().classes("w-full items-baseline justify-between"):
                            ui.label("Translation").classes(theme.DATA)
                            ui.label(self.current_target_language or "").classes(theme.DATA)
                        if self.input_mode == "Text":
                            ui.label("Ready").classes(self.banner_classes["info"]).props(
                                f"id={self.text_status_scope}_status"
                            )
                            self.text_output_label = ui.label("").classes(
                                "w-full min-h-[140px] p-1 text-base"
                            ).props(f"id={self.text_status_scope}_output")
                        else:
                            self._show_banner(self.progress_container, "Status: Ready to translate.", "info")
        if self.input_mode == "Text":
            self._inject_workspace_text_live_translation_js()

    def _render_source_input_panel(self):
        if self.input_mode == "Text":
            self.text_source_input = ui.textarea(
                label="Enter text to translate",
                placeholder="Type or paste text…",
            ).props(f"autogrow rows=8 id={self.text_status_scope}_source").classes("w-full")
            if self.target_language_input:
                self.target_language_input.props(f"id={self.text_status_scope}_target")
            return
        if self.input_mode == "Document":
            ui.upload(
                label="Click or drop DOCX, PPTX, or PDF",
                multiple=False,
                on_upload=self.handle_mobile_upload,
            ).classes("w-full")
            if self.uploaded_file_name:
                ui.label(f"Selected file: {self.uploaded_file_name}").classes("text-sm text-gray-600")
            return

        # One affordance: the file input's accept/capture props let phones
        # offer the camera directly, so no separate fallback uploader.
        ui.upload(
            label="Click or drop an image (PNG, JPG, WEBP) — on a phone this opens the camera",
            multiple=False,
            auto_upload=True,
            on_upload=self.handle_mobile_image_upload,
        ).props("accept=image/* capture=environment").classes("w-full")
        if self.image_upload_name:
            ui.label(f"Selected image: {self.image_upload_name}").classes("text-sm text-gray-600")

    def mobile_page(self):
        self.mobile_mode = True
        self._inject_theme()
        self._inject_api_token()
        self._inject_auto_device_routing("mobile")
        with ui.header().classes(f"items-center {theme.HEADER} p-3"):
            with ui.row().classes("w-full items-center justify-between"):
                ui.html(f'<span class="{theme.WORDMARK}">Passage<b>.</b></span>')
                ui.button("Desktop", on_click=lambda: ui.navigate.to("/")).classes(self.button_secondary_classes)

        with ui.column().classes("w-full items-center p-3"):
            self.upload_container = ui.column().classes("w-full max-w-md space-y-3")
            self.progress_container = ui.column().classes("w-full max-w-md space-y-2")
            self.result_container = ui.column().classes("w-full max-w-md space-y-3")
            self.stats_container = ui.column().classes("w-full max-w-md space-y-1")
            self.refresh_upload_ui()

    def set_mobile_input_mode(self, mode):
        self.mobile_input_mode = mode
        self.input_mode = mode
        self.refresh_upload_ui()

    def handle_mobile_upload(self, event):
        self.uploaded_file_name = event.name
        self.uploaded_file_extension = self.uploaded_file_name.split(".")[-1].lower()
        self.uploaded_file = BytesIO(event.content.read())
        ui.notify(f"Selected '{self.uploaded_file_name}'", type="positive")
        self.refresh_upload_ui()

    def start_mobile_translation(self):
        language = self.target_language_input.value if self.target_language_input else self.current_target_language
        self.current_target_language = language
        if not language:
            self.show_error("Please enter a valid target language.")
            return

        if self.input_mode == "Document":
            if not self.uploaded_file:
                self.show_error("Please upload a file before translating.")
                return
            self.handle_translation(language)
            return

        if self.input_mode == "Image/Camera":
            if not self.image_upload_bytes or not self.image_upload_name:
                self.show_error("Please upload or capture an image before translating.")
                return
            self.progress_container.clear()
            self.result_container.clear()
            self.stats_container.clear()
            try:
                self.image_translation_result = self.backend.translate_image_text_blocks(
                    self.image_upload_bytes,
                    self.image_upload_name,
                    language,
                )
                self.show_mobile_image_result(language)
            except Exception as ex:
                self.show_error(str(ex))
            return

        source_text = (self.text_source_input.value or "").strip() if self.text_source_input else ""
        if not source_text:
            self.show_error("Please provide source text before translating.")
            return

        self.progress_container.clear()
        self.result_container.clear()
        self.stats_container.clear()

        with self.progress_container:
            progress_ui = ui.circular_progress(value=0, max=100, show_value=True).classes("mt-3")
            label_ui = ui.label("Translating text...").classes("text-sm text-gray-700")

        def voice_task():
            self._run_mobile_voice_translation(source_text, language, progress_ui, label_ui)

        Thread(target=voice_task).start()

    def _run_mobile_voice_translation(self, voice_text, language, progress_ui, label_ui):
        try:
            progress_ui.set_value(40)
            label_ui.text = "Calling translation model..."
            translated = self.backend.translate_text(voice_text, language)
            progress_ui.set_value(100)
            label_ui.text = "Translation complete."
            self.current_count = 1
            self.current_tokens = 0
            self.show_mobile_voice_result(voice_text, translated, language)
        except Exception as ex:
            logging.error("[UI] Mobile voice translation error: %s", ex, exc_info=True)
            self.show_error(ex)

    def handle_mobile_image_upload(self, event):
        self.image_upload_name = event.name
        self.image_upload_bytes = event.content.read()
        ui.notify(f"Selected image '{self.image_upload_name}'", type="positive")
        self.refresh_upload_ui()

    def show_mobile_image_result(self, language: str) -> None:
        self.result_container.clear()
        result = self.image_translation_result or {}
        blocks = result.get("translated_blocks", [])
        confidence = result.get("confidence_metadata", {})
        with self.result_container:
            with ui.card().classes("w-full p-4 space-y-3"):
                ui.label(f"Image OCR translation → {language}").classes("text-lg font-semibold")
                ui.label(
                    f"Confidence avg: {confidence.get('average_confidence', 0)} "
                    f"across {confidence.get('block_count', 0)} blocks"
                ).classes("text-xs text-gray-600")
                for idx, block in enumerate(blocks, start=1):
                    with ui.grid(columns=2).classes("w-full gap-2 border rounded p-2"):
                        with ui.column().classes("w-full"):
                            ui.label(f"Extracted #{idx}").classes("text-xs font-semibold text-gray-600")
                            ui.label(block.get("source_text", "")).classes(f"text-sm {theme.PANEL_SOURCE} p-2")
                        with ui.column().classes("w-full"):
                            ui.label(f"Translated #{idx}").classes("text-xs font-semibold text-gray-600")
                            ui.label(block.get("translated_text", "")).classes(f"text-sm {theme.PANEL_TARGET} p-2")

    def show_mobile_voice_result(self, original_text, translated_text, language):
        self.result_container.clear()
        self.stats_container.clear()
        with self.result_container:
            with ui.card().classes("w-full p-4 space-y-3"):
                ui.label(f"Voice translation → {language}").classes("text-lg font-semibold")
                ui.label("Original").classes("text-xs font-semibold text-gray-600")
                ui.label(original_text).classes(f"w-full p-3 {theme.PANEL_SOURCE} text-base")
                ui.label("Translated").classes("text-xs font-semibold text-gray-600")
                ui.label(translated_text).classes(f"w-full p-3 {theme.PANEL_TARGET} text-base")

                with ui.column().classes("w-full gap-2"):
                    ui.button(
                        "Copy Translation",
                        on_click=lambda: ui.run_javascript(
                            f"navigator.clipboard.writeText({json.dumps(translated_text)})"
                        )
                    ).classes(theme.BTN_PRIMARY_XL)
                    ui.button("Start Over", on_click=self.refresh_upload_ui).classes(theme.BTN_SECONDARY_XL)

    def handle_upload(self, event):
        # file picked → ask for language & PPTX options
        self.uploaded_file_name = event.name
        self.uploaded_file_extension = self.uploaded_file_name.split(".")[-1].lower()
        self.uploaded_file = BytesIO(event.content.read())
        logging.info(f"[UI] Uploaded '{self.uploaded_file_name}'")

        self.upload_container.clear()
        with self.upload_container:
            ui.label(f"File '{self.uploaded_file_name}' uploaded successfully!")\
                .style("font-size: 18px; color: #333; margin-bottom: 6px; text-align: center;")
            ui.label("Select a target language for translation:")\
                .style("font-size: 16px; color: #555; margin-bottom: 8px; text-align: center;")

            self.lang_input = ui.input(label="Target Language", placeholder="e.g., Spanish")
            font_size_input = None
            autofit_checkbox = None
            if self.uploaded_file_extension == 'pptx':
                font_size_input = ui.number(
                    label="Max font size (pt)",
                    value=18,
                    min=1
                ).classes("mb-2")
                autofit_checkbox = ui.checkbox("Enable auto-fit").classes("mb-4")

            ui.button(
                "Translate",
                on_click=lambda: self.handle_translation(
                    self.lang_input.value,
                    int(font_size_input.value) if font_size_input else None,
                    autofit_checkbox.value if autofit_checkbox else False
                )
            ).classes(f"{theme.BTN_PRIMARY} mt-2")

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

        # clear old UI
        self.progress_container.clear()
        self.result_container.clear()
        self.stats_container.clear()

        with self.progress_container:
            progress_ui = ui.circular_progress(value=0, max=100, show_value=True)\
                .classes("mx-auto mt-4")
            label_ui = ui.label("Preparing translation...")\
                .classes("text-center mt-2")
            self.cancel_button = ui.button("Cancel Translation", on_click=self.cancel_translation)\
                .classes(f"{theme.BTN_DANGER} mt-2")

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

    def handle_translation_processed(self):
        # display a pre-translated file without re-calling openai
        self.progress_container.clear()
        self.result_container.clear()
        self.stats_container.clear()

        correlation_id = str(uuid.uuid4())
        self.current_correlation_id = correlation_id
        _log_event(
            "ui.processed_file_open_requested",
            correlation_id=correlation_id,
            file_name=self.uploaded_file_name,
            file_extension=self.uploaded_file_extension,
        )
        with self.progress_container:
            progress_ui = ui.circular_progress(value=0, max=100, show_value=True)\
                .classes("mx-auto mt-4")
            label_ui = ui.label("Loading processed document...")\
                .classes("text-center mt-2")
            self.cancel_button = ui.button("Cancel", on_click=self.cancel_translation)\
                .classes(f"{theme.BTN_DANGER} mt-2")

        self._start_job_and_poll(
            progress_ui=progress_ui,
            label_ui=label_ui,
            correlation_id=correlation_id,
            processed=True,
            font_size=None,
            autofit=False,
            target_language="",
            complete_event="ui.processed_file_open_complete",
            failed_event="ui.processed_file_open_failed",
            cancelled_event="ui.translation_cancelled",
        )

    def cancel_translation(self):
        if self.active_job_id:
            self.backend.cancel_job(self.active_job_id)
        else:
            self.backend.request_cancel()
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

        self.active_job_id = self.backend.start_translation_job(
            input_stream=self.uploaded_file,
            file_extension=self.uploaded_file_extension,
            target_language=target_language,
            processed=processed,
            font_size=font_size,
            autofit=autofit,
            correlation_id=correlation_id,
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
                self.show_error(job.error or "Translation failed")
                return

            result = self.backend.get_job_result(job.result_handle) if job.result_handle else None
            if not result:
                _log_event(failed_event, correlation_id=correlation_id, error="missing_result_handle")
                self.show_error("Translation failed: missing result payload.")
                return

            count = result["count"]
            tokens = result["tokens"]
            seg_map = result["segment_map"]

            self.current_count = count
            self.current_tokens = 0 if processed else tokens
            if processed:
                self.current_target_language = "Processed"

            self.original_segments_map.clear()
            self.translated_segments_map.clear()
            for seg_id, seg_info in seg_map.items():
                self.original_segments_map[seg_id] = seg_info["original"]
                self.translated_segments_map[seg_id] = seg_info["translated"]

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
        self.backend.regenerate_output_stream()
        fresh = BytesIO()
        self.backend.output_stream.seek(0)
        fresh.write(self.backend.output_stream.read())
        fresh.seek(0)
        return fresh

    def show_result(self):
        logging.info(f"[UI] Rendering results – advanced={self.advanced_mode}, segments={len(self.original_segments_map)}")
        self.progress_container.clear()
        self.result_container.clear()
        self.stats_container.clear()
        if self.cancel_button:
            self.cancel_button.visible = False

        with self.result_container:
            with ui.column().classes("max-w-3xl mx-auto w-full space-y-6 mt-4"):
                ui.label(f"'{self.uploaded_file_name}' → {self.current_target_language}")\
                    .classes("text-2xl font-semibold text-gray-800")

                # ── SEGMENT EDITOR ──────────────────────────────
                if self.advanced_mode and self.original_segments_map and not self.mobile_mode:
                    ui.separator().classes("my-4")
                    ui.label("Advanced Mode: Segment Editor")\
                        .classes("text-xl font-bold mb-2")
                    ui.label(f"Review and edit {len(self.original_segments_map)} segments below:")\
                        .classes("text-sm text-gray-600 mb-4")

                    with ui.row().classes("space-x-2 mb-2"):
                        ui.button("Open Document Editor", on_click=self.open_document_editor)\
                          .classes(theme.BTN_SECONDARY_SM)
                        ui.label("Launch a single-page editor with every segment ready to edit.")\
                          .classes("text-xs text-gray-600")

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
                        seg_info = self.backend.segment_map.get(seg_id, {})
                        location = seg_info.get("location", f"segment_{i+1}")

                        with ui.expansion(f"Step {i+1}: {location}", value=i == 0)\
                                .classes(f"w-full mb-2 {theme.WELL}"):
                            with ui.column().classes("p-3 pb-16 md:pb-4 gap-3"):
                                with ui.row().classes("justify-between items-center"):
                                    ui.label(f"#{i+1}: {location}")\
                                      .classes("text-sm font-medium text-gray-700")
                                    with ui.row().classes("space-x-1"):
                                        ui.button("✓", on_click=lambda _, s=seg_id: self.approve_segment_callback(s))\
                                          .props("size=sm color=positive")
                                        ui.button("✗", on_click=lambda _, s=seg_id: self.decline_segment_callback(s))\
                                          .props("size=sm color=negative")
                                        ui.button("🗑️", on_click=lambda _, s=seg_id: self.delete_segment_callback(s))\
                                          .props("size=sm color=grey")

                                with ui.column().classes("w-full"):
                                    ui.label("Original:")\
                                      .classes("text-xs font-semibold text-gray-600")
                                    ui.html(
                                        f'<div class="text-sm p-2 p-panel-source '
                                        f'max-h-20 overflow-y-auto">{orig[:300]}'
                                        f'{"..." if len(orig)>300 else ""}</div>'
                                    )

                                with ui.column().classes("w-full"):
                                    ui.label("Translation:")\
                                      .classes("text-xs font-semibold text-gray-600")
                                    textarea = ui.textarea(value=trans)\
                                      .props("autogrow rows=3")\
                                      .classes(f"w-full text-sm {theme.PANEL_TARGET}")
                                    textarea.segment_id = seg_id

                                with ui.row().classes("w-full items-center gap-2"):
                                    refine = ui.input(placeholder="Refinement instructions (optional)")\
                                      .classes("flex-grow text-sm")
                                    ui.button("Update",
                                              on_click=lambda _, s=seg_id, ta=textarea, ri=refine:
                                                self.update_segment_callback(s, ta, ri)
                                             ).props("size=sm color=primary")
                                    ui.button("Re-translate",
                                              on_click=lambda _, s=seg_id, ta=textarea:
                                                self.retranslate_segment_callback(s, ta)
                                             ).props("size=sm color=secondary")

                if self.uploaded_file_extension in {"png", "jpg", "jpeg", "webp"}:
                    ui.separator().classes("my-3")
                    ui.label("Image overlay controls").classes("text-lg font-semibold")
                    with ui.row().classes("items-center gap-2 flex-wrap"):
                        ui.number("Font size", value=self.overlay_font_size, min=8, max=64, step=1, on_change=lambda e: setattr(self, "overlay_font_size", int(e.value))).classes("w-32")
                        ui.input("Font family", value=self.overlay_font_family, on_change=lambda e: setattr(self, "overlay_font_family", e.value)).classes("w-48")
                        ui.switch("Show original overlay", value=self.overlay_show_original, on_change=lambda e: setattr(self, "overlay_show_original", bool(e.value)))
                        ui.switch("Preview visible", value=self.overlay_preview_visible, on_change=lambda e: setattr(self, "overlay_preview_visible", bool(e.value)) or self.show_result())
                        ui.button("Refresh overlay", on_click=self.refresh_image_overlay).classes(theme.BTN_SECONDARY_SM)
                    if self.overlay_preview_visible and self.backend.output_stream is not None:
                        import base64
                        self.backend.output_stream.seek(0)
                        encoded = base64.b64encode(self.backend.output_stream.read()).decode("ascii")
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
            ui.label(f"elements translated: {self.current_count}")\
              .classes(theme.DATA)
            if self.current_tokens > 0:
                ui.label(f"tokens used: {self.current_tokens:,}")\
                  .classes(theme.DATA)

    def refresh_image_overlay(self):
        try:
            self.backend.process_image(
                BytesIO(self.uploaded_file.getvalue()),
                self.current_target_language,
                show_original=self.overlay_show_original,
                font_size=self.overlay_font_size,
                font_family=self.overlay_font_family,
                run_state=self.backend._active_run_state,
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
            updated = self.backend.update_segment(
                seg_id, textarea.value, self.current_target_language, instructions
            )
            textarea.value = updated
            self.translated_segments_map[seg_id] = updated
            refine_input.value = ""
            ui.notify("Segment updated successfully!", type="positive")
        except Exception as ex:
            logging.error(f"[UI] Error updating segment {seg_id}: {ex}", exc_info=True)
            ui.notify(f"Update failed: {ex}", type="negative")

    def retranslate_segment_callback(self, seg_id, textarea):
        try:
            seg_info = self.backend.segment_map.get(seg_id)
            if not seg_info:
                ui.notify("Segment not found", type="negative")
                return
            ui.notify("Re-­translating...", type="info")
            original = seg_info["original"]
            new_trans = self.backend.translate_text(original, self.current_target_language)
            self.backend.update_segment(seg_id, new_trans, self.current_target_language)
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
            self.backend.delete_segment(seg_id)
            self.original_segments_map.pop(seg_id, None)
            self.translated_segments_map.pop(seg_id, None)
            self.current_count = len(self.backend.segment_map)
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

    def save_all_edits(self):
        try:
            self.backend.regenerate_output_stream()
            ui.notify("All edits saved to document!", type="positive")
        except Exception as ex:
            logging.error(f"[UI] Error saving edits: {ex}", exc_info=True)
            ui.notify(f"Save failed: {ex}", type="negative")

    def show_error(self, error):
        self.result_container.clear()
        self.stats_container.clear()
        self._show_banner(self.result_container, f"Error: {error}", "negative")

    # ─────────────────────────────── VOICE TRANSLATION PAGE ─────────────────────────────────

    def voice_translation_page(self):
        self._inject_theme()
        self._inject_api_token()
        self._inject_voice_frontend_helpers()
        ui.label("Live Voice Translation").classes("text-2xl mb-4")

        with ui.row().classes("items-center space-x-2 mb-4"):
            ui.label("Target language:").classes("font-medium")
            options = "".join(f'<option value="{lang}"></option>' for lang in LANGUAGES)
            ui.html(
                f'<input id="language_select" list="passage_languages" value="Spanish" '
                f'placeholder="Type a language…" class="px-3 py-2 p-well">'
                f'<datalist id="passage_languages">{options}</datalist>'
            )

        self._render_voice_status_block("desktop_voice")

        with ui.row().classes("space-x-4 mb-4"):
            ui.html('<button id="desktop_voice_start_recording" class="p-btn p-btn-ok px-4 py-2" '
                    'onclick="startRecording()">🎤 Start Recording</button>')
            ui.html('<button id="desktop_voice_stop_recording" class="p-btn p-btn-danger px-4 py-2" '
                    'onclick="stopRecording()" disabled>⏹️ Stop Recording</button>')
            ui.button("← Back", on_click=lambda: ui.navigate.to("/"))\
              .classes(theme.BTN_SECONDARY)

        ui.label("Transcript fallback (when recording is unavailable)").classes("text-sm font-semibold text-gray-700 mt-2")
        ui.textarea(
            label="Transcript (fallback)",
            placeholder="Paste text if your browser cannot record audio.",
        ).props("id=desktop_voice_transcript autogrow").classes("w-full")
        ui.button("Translate", on_click=lambda: ui.run_javascript("translateTranscriptFallback()"))\
            .classes(f"{theme.BTN_PRIMARY} mt-2")

        ui.audio(src="data:audio/wav;base64,")\
          .props("id=out_audio controls")\
          .classes("w-full")

        ui.label("Original:").classes("font-bold mt-4")
        ui.label("").classes(f"p-2 {theme.PANEL_SOURCE} min-h-[40px]")\
          .props("id=original_text")

        ui.label("Translation:").classes("font-bold mt-2")
        ui.label("").classes(f"p-2 {theme.PANEL_TARGET} min-h-[40px]")\
          .props("id=translated_text")

        ui.add_head_html("""
<script>
    let recorder = null, stream = null, chunks = [], isRecording = false;
    let selectedMimeType = null;
    const DESKTOP_SCOPE = 'desktop_voice';

    function updateStatus(msg) {
        window.voiceUx.setStatus(DESKTOP_SCOPE, msg);
    }
    function updateDebug(msg) {
        window.voiceUx.setDebug(DESKTOP_SCOPE, msg);
    }
    function updateButtons(recording) {
        window.voiceUx.setRecordingButtons(DESKTOP_SCOPE, recording);
        isRecording = recording;
    }
    function setRecordingControlsEnabled(enabled) {
        const start = document.getElementById('desktop_voice_start_recording');
        const stop = document.getElementById('desktop_voice_stop_recording');
        if (start) {
            start.disabled = !enabled;
            start.style.opacity = enabled ? '1' : '0.5';
        }
        if (stop) {
            stop.disabled = true;
            stop.style.opacity = '0.5';
        }
        if (!enabled) isRecording = false;
    }

    function resolveRecorderMimeType() {
        if (!window.MediaRecorder || typeof MediaRecorder.isTypeSupported !== 'function') return undefined;
        const candidates = ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4'];
        for (const mime of candidates) {
            if (MediaRecorder.isTypeSupported(mime)) return mime;
        }
        if (MediaRecorder.isTypeSupported('')) return undefined;
        return null;
    }

    function mapRecordingError(err) {
        const name = err?.name || 'Error';
        if (name === 'NotAllowedError') {
            return 'Microphone permission denied. Allow mic access in browser/site settings and retry.';
        }
        if (name === 'NotFoundError') {
            return 'No microphone found. Connect/enable a mic and try again.';
        }
        if (name === 'NotSupportedError') {
            return 'Audio recording is not supported in this browser. Try a current Chrome/Edge/Safari.';
        }
        return err?.message || 'Unexpected recording error.';
    }

    async function startRecording() {
        const hasGetUserMedia = !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
        const isLocalhost = ['localhost', '127.0.0.1', '::1'].includes(window.location.hostname);
        const secureOk = window.isSecureContext || isLocalhost;
        if (!hasGetUserMedia || !secureOk) {
            setRecordingControlsEnabled(false);
            updateStatus("Recording unavailable. Use HTTPS or localhost in a supported browser.");
            updateDebug(`preflight getUserMedia=${hasGetUserMedia} secure=${secureOk}`);
            return;
        }

        selectedMimeType = resolveRecorderMimeType();
        if (selectedMimeType === null) {
            setRecordingControlsEnabled(false);
            updateStatus("No supported recording format found. Try a different browser.");
            updateDebug(`mime=none secure=${secureOk}`);
            return;
        }

        updateStatus("Requesting mic…");
        updateDebug(`mime=${selectedMimeType || 'browser-default'} secure=${secureOk} permission=requesting`);
        try {
            const constraints = {
                audio:{echoCancellation:true,noiseSuppression:true,autoGainControl:true}
            };
            stream = await navigator.mediaDevices.getUserMedia(constraints);
            updateDebug(`mime=${selectedMimeType || 'browser-default'} secure=${secureOk} permission=granted`);
            const recorderOptions = selectedMimeType ? { mimeType: selectedMimeType } : undefined;
            recorder = recorderOptions ? new MediaRecorder(stream, recorderOptions) : new MediaRecorder(stream);
            chunks = [];
            recorder.ondataavailable = e => { if(e.data.size>0) { chunks.push(e.data); updateDebug(`Chunks:${chunks.length}`); } };
            recorder.onstart = () => { updateStatus("🔴 Recording…"); updateButtons(true); };
            recorder.onerror = e => {
                updateStatus("Error: " + mapRecordingError(e.error || e));
                updateDebug(`mime=${selectedMimeType || 'browser-default'} secure=${secureOk} permission=granted`);
                updateButtons(false);
            };
            recorder.start(1000);
        } catch (err) {
            const mapped = mapRecordingError(err);
            updateStatus("Error: " + mapped);
            updateDebug(`mime=${selectedMimeType || 'browser-default'} secure=${secureOk} permission=denied (${err?.name || 'unknown'})`);
            if(stream) stream.getTracks().forEach(t=>t.stop());
        }
    }

    async function stopRecording() {
        if(!recorder || recorder.state!=='recording') {
            window.voiceUx.setDebug(DESKTOP_SCOPE, 'No active recording session.');
            return;
        }
        window.voiceUx.setStatus(DESKTOP_SCOPE, window.voiceUx.states.STOPPING);
        window.voiceUx.setRecordingButtons(DESKTOP_SCOPE, false);
        recorder.onstop = async () => {
            window.voiceUx.setStatus(DESKTOP_SCOPE, window.voiceUx.states.PROCESSING_AUDIO);
            const blob = new Blob(chunks, { type: recorder.mimeType });
            let lang = document.getElementById('language_select')?.value || 'es';
            let fd = new FormData();
            fd.append('file', blob, `rec.${blob.type.includes('webm')?'webm':'mp4'}`);
            fd.append('language', lang);
            try {
                const resp = await fetch('/api/voice_translate', { method:'POST', body:fd, headers: { 'X-Passage-Token': window.PASSAGE_TOKEN || '' } });
                if(!resp.ok) throw new Error(await resp.text());
                const audio = await resp.blob();
                const origHeader = resp.headers.get('X-Original-Text') || '';
                const transHeader = resp.headers.get('X-Translated-Text') || '';
                const decodeHeader = (value) => {
                    if (!value) return '';
                    try { return decodeURIComponent(value); } catch (_err) { return value; }
                };
                const orig = decodeHeader(origHeader);
                const trans = decodeHeader(transHeader);
                document.getElementById('original_text').textContent   = orig;
                document.getElementById('translated_text').textContent= trans;
                if(audio.size>0){
                    let url = URL.createObjectURL(audio);
                    let player = document.getElementById('out_audio');
                    player.src = url; player.play();
                    window.voiceUx.setStatus(DESKTOP_SCOPE, window.voiceUx.states.COMPLETE);
                }
            } catch(e) {
                window.voiceUx.setStatus(DESKTOP_SCOPE, "Error: " + e.message);
                window.voiceUx.setDebug(DESKTOP_SCOPE, e.message);
            } finally {
                if(stream) stream.getTracks().forEach(t=>t.stop());
                recorder = null; chunks = [];
            }
        };
        recorder.stop();
    }

    async function translateTranscriptFallback() {
        const lang = document.getElementById('language_select')?.value || 'es';
        const transcript = document.getElementById('desktop_voice_transcript')?.value || '';
        const cleaned = transcript.trim();
        if (!cleaned) {
            window.voiceUx.setStatus(DESKTOP_SCOPE, 'Please provide transcript text before translating.');
            return;
        }
        window.voiceUx.setStatus(DESKTOP_SCOPE, window.voiceUx.states.TRANSLATING_TEXT);
        window.voiceUx.setDebug(DESKTOP_SCOPE, `chars=${cleaned.length}`);
        try {
            const fd = new FormData();
            fd.append('text', cleaned);
            fd.append('language', lang);
            const resp = await fetch('/api/text_translate_stream', { method: 'POST', body: fd, headers: { 'X-Passage-Token': window.PASSAGE_TOKEN || '' } });
            const contentType = resp.headers.get('content-type') || '';
            if (contentType.includes('text/event-stream')) {
                const reader = resp.body.getReader();
                const decoder = new TextDecoder();
                let buffer = '';
                let streamed = false;
                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    buffer += decoder.decode(value, { stream: true });
                    const events = buffer.split('\n\n');
                    buffer = events.pop() || '';
                    for (const evt of events) {
                        const eventLine = evt.split('\n').find(line => line.startsWith('event: '));
                        const dataLine = evt.split('\n').find(line => line.startsWith('data: '));
                        const eventType = eventLine ? eventLine.replace('event: ', '').trim() : '';
                        const payload = dataLine ? JSON.parse(dataLine.replace('data: ', '')) : {};
                        if (eventType === 'start') {
                            document.getElementById('original_text').textContent = payload.original_text || cleaned;
                        } else if (eventType === 'partial') {
                            streamed = true;
                            document.getElementById('translated_text').textContent = payload.translated_text || '';
                        } else if (eventType === 'complete') {
                            document.getElementById('translated_text').textContent = payload.translated_text || '';
                        } else if (eventType === 'error') {
                            throw new Error(payload.error || 'Transcript streaming failed.');
                        }
                    }
                }
                if (!streamed) {
                    window.voiceUx.setDebug(DESKTOP_SCOPE, 'Streaming unavailable; translation returned without partial chunks.');
                }
            } else {
                const data = await resp.json();
                if (!resp.ok) throw new Error(data?.error || 'Transcript translation failed.');
                document.getElementById('original_text').textContent = data.original_text || cleaned;
                document.getElementById('translated_text').textContent = data.translated_text || '';
            }
            window.voiceUx.setStatus(DESKTOP_SCOPE, window.voiceUx.states.COMPLETE);
        } catch (e) {
            window.voiceUx.setStatus(DESKTOP_SCOPE, "Error: " + e.message);
            window.voiceUx.setDebug(DESKTOP_SCOPE, e.message || 'unknown error');
        }
    }

    window.translateTranscriptFallback = translateTranscriptFallback;

    window.addEventListener('load', () => {
        const hasGetUserMedia = !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
        const isLocalhost = ['localhost', '127.0.0.1', '::1'].includes(window.location.hostname);
        const secureOk = window.isSecureContext || isLocalhost;
        selectedMimeType = resolveRecorderMimeType();
        const canRecord = hasGetUserMedia && secureOk && selectedMimeType !== null;
        setRecordingControlsEnabled(canRecord);
        if (!hasGetUserMedia || !secureOk) {
            updateStatus("Recording unavailable. Use HTTPS or localhost in a supported browser.");
        } else if (selectedMimeType === null) {
            updateStatus("No supported recording format found. Try a different browser.");
        } else {
            updateStatus("Ready to record");
        }
        updateDebug(`mime=${selectedMimeType === null ? 'none' : (selectedMimeType || 'browser-default')} secure=${secureOk} getUserMedia=${hasGetUserMedia}`);
    });
</script>
        """)

    # ─────────────────────────── VOICE TRANSLATION API ────────────────────────────────────

    def _record_chat_thread(self, original: str, translated: str, language: str) -> None:
        self.recent_threads.appendleft({
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

    async def api_voice_translate(
        self,
        request: Request,
        file: UploadFile = File(...),
        language: str = Form(...)
    ) -> Response:
        correlation_id = str(uuid.uuid4())
        denied = self._check_api_access(request, correlation_id)
        if denied:
            return denied
        try:
            if not language or language.lower() in ('undefined','null',''):
                language = 'es'
                logging.warning("[API] Empty language → default to Spanish")
            _log_event("ui.voice_translate_requested", correlation_id=correlation_id, language=language)
            data = await file.read()
            if not data:
                return Response(content=b"", status_code=400, headers={"X-Error":"Empty audio data"})
            if len(data) > MAX_UPLOAD_BYTES:
                return Response(
                    content=b"Audio file is too large.",
                    status_code=413,
                    media_type="text/plain",
                    headers={"X-Correlation-Id": correlation_id},
                )
            original_text, translated_text, mp3_bytes = await asyncio.to_thread(
                self.backend.translate_audio, data, language
            )
            safe_original = (original_text or "")[:400]
            safe_translated = (translated_text or "")[:400]
            header_orig = quote(safe_original, safe="")
            header_translated = quote(safe_translated, safe="")
            return Response(
                content=mp3_bytes,
                media_type="audio/mpeg",
                headers={
                    "X-Original-Text": header_orig,
                    "X-Translated-Text": header_translated,
                    "X-Target-Language": language,
                    "X-Correlation-Id": correlation_id,
                    "Content-Length": str(len(mp3_bytes))
                }
            )
        except Exception as e:
            logging.error(f"[API] Voice error: {e}", exc_info=True)
            _log_event("ui.voice_translate_failed", correlation_id=correlation_id, error=str(e))
            msg = f"Translation error: {e}"
            return Response(content=msg.encode(), status_code=500,
                            media_type="text/plain", headers={"X-Error":msg})

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
            translated = await asyncio.to_thread(
                self.backend.translate_text,
                cleaned_text,
                language,
            )
            _log_event(
                "ui.text_translate_succeeded",
                correlation_id=correlation_id,
                language=language,
            )
            self._record_chat_thread(cleaned_text, translated, language)
            return JSONResponse(
                {
                    "original_text": cleaned_text,
                    "translated_text": translated,
                    "target_language": language,
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

        if not should_stream:
            translated = await asyncio.to_thread(self.backend.translate_text, cleaned_text, language)
            self._record_chat_thread(cleaned_text, translated, language)
            return JSONResponse(
                {
                    "fallback": True,
                    "original_text": cleaned_text,
                    "translated_text": translated,
                    "target_language": language,
                },
                headers={"X-Correlation-Id": correlation_id},
            )

        async def event_generator():
            yield f"event: start\ndata: {json.dumps({'target_language': language, 'original_text': cleaned_text})}\n\n"
            try:
                final_text, partials = await asyncio.to_thread(
                    self.backend.stream_translate_text,
                    cleaned_text,
                    language,
                )
                for partial in partials[:-1]:
                    yield f"event: partial\ndata: {json.dumps({'translated_text': partial})}\n\n"
                self._record_chat_thread(cleaned_text, final_text, language)
                yield f"event: complete\ndata: {json.dumps({'translated_text': final_text, 'canonical': True})}\n\n"
            except Exception as e:
                yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Correlation-Id": correlation_id},
        )

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
            result = await asyncio.to_thread(
                self.backend.translate_image_text_blocks,
                payload,
                file.filename or "uploaded_image",
                language or "es",
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


if __name__ in {"__main__", "__mp_main__"}:
    TranslationUI().start_ui()
