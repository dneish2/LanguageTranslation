import asyncio
import logging
import os
import glob
import json
import time
import uuid
from io import BytesIO
from threading import Thread
from typing import Any

from nicegui import ui, app
from fastapi import UploadFile, File, Form
from starlette.responses import Response

from TranslationBackend import TranslationBackend

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("translation.ui")


def _log_event(event: str, correlation_id: str | None = None, **fields: Any) -> None:
    payload: dict[str, Any] = {"event": event, **fields}
    if correlation_id:
        payload["correlation_id"] = correlation_id
    LOGGER.info(json.dumps(payload, default=str))


class TranslationUI:
    def __init__(self):
        # ── CORE BACKEND ─────────────────────────────────────────────────
        self.backend = TranslationBackend()

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
        self.current_correlation_id: str | None = None
        self.cancel_button = None
        self.active_job_id: str | None = None
        self.job_poll_timer = None

        # ── SEGMENT EDITING ─────────────────────────────────────────────
        self.original_segments_map: dict[str, str] = {}
        self.translated_segments_map: dict[str, str] = {}

        # ── DRAWER & ADVANCED MODE ──────────────────────────────────────
        self.drawer = None
        self.advanced_mode = False
        self.top_mode_control = None
        self._syncing_mode_control = False
        self.top_nav_control_classes = "h-8 min-h-0 px-2 rounded-md"
        self.document_editor_dialog = None
        self.document_editor_container = None
        self.document_editor_inputs: dict[str, Any] = {}
        self.document_editor_original_values: dict[str, str] = {}

        # ── MOBILE FLOW ────────────────────────────────────────────────
        self.mobile_mode = False
        self.mobile_input_mode = "upload"
        self.mobile_target_input = None
        self.mobile_voice_input = None

        # ── USAGE STATS ─────────────────────────────────────────────────
        self.current_count = 0
        self.current_cost = 0.0

        # ── VOICE TRANSLATION API ──────────────────────────────────────
        # registers /api/voice_translate on the same port
        app.add_api_route(
            "/api/voice_translate",
            self.api_voice_translate,
            methods=["POST"],
        )

    def start_ui(self):
        # register pages
        ui.page("/")(self.main_page)
        ui.page("/voice")(self.voice_translation_page)
        ui.page("/mobile")(self.mobile_page)
        # run on single port
        ui.run(host="0.0.0.0", port=8080)

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

    # ──────────────────────────────────── MAIN PAGE ─────────────────────────────────────────

    def main_page(self):
        self.mobile_mode = False
        self._inject_auto_device_routing("desktop")
        # Header with a compact unified mode control
        with ui.header().classes("items-center justify-between bg-gray-100 p-2"):
            with ui.row().classes("w-full flex justify-between items-center"):
                ui.label("Translation App").classes("text-lg font-bold text-black")
                self.top_mode_control = ui.toggle(
                    {"standard": "Standard", "advanced": "Advanced", "voice": "Voice"},
                    value=self._current_top_mode()
                ).classes(self.top_nav_control_classes)
                self.top_mode_control.on_value_change(self.handle_top_mode_change)

        # Drawer for recent docs
        self.drawer = ui.drawer(side='left').classes("bg-gray-50")
        with self.drawer:
            self.show_document_list()

        # Main upload / progress / results / stats containers
        with ui.column().classes("w-full h-full items-center justify-center p-4"):
            self.upload_container = ui.column().classes("w-full max-w-3xl items-center")
            self.progress_container = ui.column().classes("w-full max-w-3xl items-center")
            self.result_container = ui.column().classes("w-full max-w-3xl items-center")
            self.stats_container = ui.column().classes("w-full max-w-3xl items-center")
            self.refresh_upload_ui()

        self.build_document_editor_dialog()

    def toggle_advanced_mode(self):
        self.advanced_mode = not self.advanced_mode
        if self.advanced_mode:
            ui.notify("Advanced mode on.", type="positive")
        else:
            ui.notify("Advanced mode off.", type="info")
        self.sync_top_mode_control()
        if self.original_segments_map:
            self.show_result()

    def _current_top_mode(self) -> str:
        return "advanced" if self.advanced_mode else "standard"

    def sync_top_mode_control(self) -> None:
        if not self.top_mode_control:
            return
        self._syncing_mode_control = True
        self.top_mode_control.value = self._current_top_mode()
        self._syncing_mode_control = False

    def handle_top_mode_change(self, event) -> None:
        if self._syncing_mode_control:
            return

        selected_mode = event.value
        if selected_mode == "voice":
            self.sync_top_mode_control()
            ui.navigate.to("/voice")
            return

        should_enable_advanced = selected_mode == "advanced"
        if should_enable_advanced != self.advanced_mode:
            self.toggle_advanced_mode()

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

                with ui.scroll_area().classes("w-full flex-1 border rounded p-3 pb-28 md:pb-6 bg-white"):
                    self.document_editor_container = ui.column().classes("space-y-4")

                with ui.row().classes("justify-end space-x-2 mt-4"):
                    ui.button("Cancel", on_click=self.request_close_document_editor)\
                        .classes("bg-gray-200 text-gray-700 px-3 py-1 rounded")
                    ui.button("Save Changes", on_click=self.save_document_editor)\
                        .classes("bg-blue-600 text-white px-3 py-1 rounded")

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
                        .classes("w-full bg-gray-50 border rounded-lg"):
                    with ui.column().classes("space-y-2 p-2 pb-16 md:pb-4"):
                        ui.label("Edit translation below (mobile-safe layout).")\
                            .classes("text-xs text-gray-500")
                        textarea = ui.textarea(value=translated_value or "")\
                            .props("autogrow rows=3")\
                            .classes("w-full text-base border rounded p-2 bg-white")
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
                        .classes("bg-gray-200 text-gray-700 px-3 py-1 rounded")

                    def confirm_and_close():
                        dialog.close()
                        on_confirm()

                    ui.button(confirm_label, on_click=confirm_and_close)\
                        .classes("bg-red-500 text-white px-3 py-1 rounded")
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
        # clear and list up to 20 translated_* files
        self.drawer.clear()
        ui.label("Recent Documents").classes("font-bold text-lg mb-2")
        files = sorted(glob.glob("translated_*"), key=os.path.getmtime, reverse=True)[:20]
        if not files:
            ui.label("No recent documents.").classes("text-sm text-gray-600")
        else:
            for fn in files:
                ui.button(fn, on_click=lambda _, f=fn: self.load_processed_document(f))\
                    .classes("w-full text-left mb-1")

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

        if self.mobile_mode:
            self.render_mobile_flow()
            return

        with self.upload_container:
            ui.label("Upload a document (DOCX, PPTX, or PDF)")\
                .style("font-size: 20px; color: #333; margin-bottom: 10px; text-align: center;")
            ui.upload(
                label="Click or drop a file here",
                multiple=False,
                on_upload=self.handle_upload
            )

    def mobile_page(self):
        self.mobile_mode = True
        self._inject_auto_device_routing("mobile")
        with ui.header().classes("items-center bg-white p-3 border-b"):
            with ui.row().classes("w-full items-center justify-between"):
                ui.label("Mobile Translation Flow").classes("text-base font-semibold")
                ui.button("Desktop", on_click=lambda: ui.navigate.to("/"))                    .classes("bg-gray-200 text-gray-700 px-4 py-2 rounded")

        with ui.column().classes("w-full items-center p-3"):
            self.upload_container = ui.column().classes("w-full max-w-md space-y-3")
            self.progress_container = ui.column().classes("w-full max-w-md items-center space-y-2")
            self.result_container = ui.column().classes("w-full max-w-md space-y-3")
            self.stats_container = ui.column().classes("w-full max-w-md space-y-1")
            self.refresh_upload_ui()

    def render_mobile_flow(self):
        with self.upload_container:
            ui.label("1) Choose input").classes("text-sm font-semibold text-gray-700")
            mode = ui.toggle({"upload": "Upload", "voice": "Voice"}, value=self.mobile_input_mode)                .classes("w-full")
            mode.on("update:model-value", lambda e: self.set_mobile_input_mode(e.args))

            ui.label("2) Select language").classes("text-sm font-semibold text-gray-700 mt-2")
            self.mobile_target_input = ui.input(
                label="Target language",
                placeholder="e.g., Spanish",
                value=self.current_target_language or ""
            ).classes("w-full")

            if self.mobile_input_mode == "upload":
                ui.label("3) Upload a document").classes("text-sm font-semibold text-gray-700 mt-2")
                ui.upload(
                    label="Tap to pick DOCX, PPTX, or PDF",
                    multiple=False,
                    on_upload=self.handle_mobile_upload,
                ).classes("w-full")
                if self.uploaded_file_name:
                    ui.label(f"Selected: {self.uploaded_file_name}").classes("text-sm text-gray-600")
            else:
                ui.label("3) Record or paste speech text").classes("text-sm font-semibold text-gray-700 mt-2")
                self.mobile_voice_input = ui.textarea(
                    label="Voice transcript",
                    placeholder="Paste transcript or use dictation and edit if needed.",
                ).props("id=mobile_voice_text autogrow").classes("w-full")
                ui.html('<button type="button" class="w-full bg-emerald-600 text-white rounded-lg py-4 text-lg font-semibold" '
                        'onclick="startMobileDictation()">🎙️ Start Dictation</button>')
                ui.add_head_html("""
<script>
function startMobileDictation() {
    const input = document.getElementById('mobile_voice_text') || document.querySelector('textarea');
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) { alert('Speech recognition is not available on this browser.'); return; }
    const recognition = new SR();
    recognition.lang = 'en-US';
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;
    recognition.onresult = (event) => {
        const text = event.results?.[0]?.[0]?.transcript || '';
        if (input) {
            input.value = text;
            input.dispatchEvent(new Event('input', { bubbles: true }));
        }
    };
    recognition.start();
}
</script>
                """)

            ui.label("4) Translate").classes("text-sm font-semibold text-gray-700 mt-2")
            ui.button("Translate", on_click=self.start_mobile_translation)                .classes("w-full bg-blue-600 text-white py-4 text-lg rounded-lg shadow")

    def set_mobile_input_mode(self, mode):
        self.mobile_input_mode = mode
        self.refresh_upload_ui()

    def handle_mobile_upload(self, event):
        self.uploaded_file_name = event.name
        self.uploaded_file_extension = self.uploaded_file_name.split(".")[-1].lower()
        self.uploaded_file = BytesIO(event.content.read())
        ui.notify(f"Selected '{self.uploaded_file_name}'", type="positive")
        self.refresh_upload_ui()

    def start_mobile_translation(self):
        language = self.mobile_target_input.value if self.mobile_target_input else None
        if not language:
            self.show_error("Please enter a valid target language.")
            return

        if self.mobile_input_mode == "upload":
            if not self.uploaded_file:
                self.show_error("Please upload a file before translating.")
                return
            self.handle_translation(language)
            return

        voice_text = (self.mobile_voice_input.value or "").strip() if self.mobile_voice_input else ""
        if not voice_text:
            self.show_error("Please provide voice transcript text before translating.")
            return

        self.current_target_language = language
        self.progress_container.clear()
        self.result_container.clear()
        self.stats_container.clear()

        progress_ui = ui.circular_progress(value=0, max=100, show_value=True).classes("mt-3")
        label_ui = ui.label("Translating transcript...").classes("text-sm text-gray-700")
        with self.progress_container:
            progress_ui
            label_ui

        def voice_task():
            self._run_mobile_voice_translation(voice_text, language, progress_ui, label_ui)

        Thread(target=voice_task).start()

    def _run_mobile_voice_translation(self, voice_text, language, progress_ui, label_ui):
        try:
            progress_ui.set_value(40)
            label_ui.text = "Calling translation model..."
            translated = self.backend.translate_text(voice_text, language)
            progress_ui.set_value(100)
            label_ui.text = "Translation complete."
            self.current_count = 1
            self.current_cost = 0.0
            self.show_mobile_voice_result(voice_text, translated, language)
        except Exception as ex:
            logging.error("[UI] Mobile voice translation error: %s", ex, exc_info=True)
            self.show_error(ex)

    def show_mobile_voice_result(self, original_text, translated_text, language):
        self.result_container.clear()
        self.stats_container.clear()
        with self.result_container:
            with ui.card().classes("w-full p-4 space-y-3"):
                ui.label(f"Voice translation → {language}").classes("text-lg font-semibold")
                ui.label("Original").classes("text-xs font-semibold text-gray-600")
                ui.label(original_text).classes("w-full p-3 rounded border bg-gray-50 text-base")
                ui.label("Translated").classes("text-xs font-semibold text-gray-600")
                ui.label(translated_text).classes("w-full p-3 rounded border bg-blue-50 text-base")

                with ui.column().classes("w-full gap-2"):
                    ui.button(
                        "Copy Translation",
                        on_click=lambda: ui.run_javascript(
                            f"navigator.clipboard.writeText({json.dumps(translated_text)})"
                        )
                    ).classes("w-full bg-blue-600 text-white py-3 text-base rounded-lg")
                    ui.button("Start Over", on_click=self.refresh_upload_ui)                        .classes("w-full bg-gray-500 text-white py-3 text-base rounded-lg")

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
            ).classes("bg-blue-600 text-white px-4 py-2 rounded shadow mt-2")

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

        progress_ui = ui.circular_progress(value=0, max=100, show_value=True)\
            .classes("mx-auto mt-4").style("color: #ff9800;")
        label_ui = ui.label("Preparing translation...")\
            .classes("text-center text-gray-700 mt-2")
        self.cancel_button = ui.button("Cancel Translation", on_click=self.cancel_translation)\
            .classes("bg-red-500 text-white px-4 py-2 rounded shadow mt-2")

        with self.progress_container:
            progress_ui
            label_ui
            self.cancel_button

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
        progress_ui = ui.circular_progress(value=0, max=100, show_value=True)\
            .classes("mx-auto mt-4").style("color: #ff9800;")
        label_ui = ui.label("Loading processed document...")\
            .classes("text-center text-gray-700 mt-2")
        self.cancel_button = ui.button("Cancel", on_click=self.cancel_translation)\
            .classes("bg-red-500 text-white px-4 py-2 rounded shadow mt-2")

        with self.progress_container:
            progress_ui
            label_ui
            self.cancel_button

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
            self.current_cost = 0.0 if processed else tokens * 0.002 / 1000
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
                          .classes("bg-purple-600 text-white px-3 py-1")
                        ui.label("Launch a single-page editor with every segment ready to edit.")\
                          .classes("text-xs text-gray-600")

                    # bulk actions
                    with ui.row().classes("space-x-2 mb-4"):
                        ui.button("Approve All", on_click=self.approve_all_segments)\
                          .classes("bg-green-500 text-white px-3 py-1")
                        ui.button("Save All Edits", on_click=self.save_all_edits)\
                          .classes("bg-blue-500 text-white px-3 py-1")

                    # per-segment UI
                    for i, seg_id in enumerate(list(self.original_segments_map.keys())):
                        orig = self.original_segments_map[seg_id]
                        trans = self.translated_segments_map[seg_id]
                        seg_info = self.backend.segment_map.get(seg_id, {})
                        location = seg_info.get("location", f"segment_{i+1}")

                        with ui.expansion(f"Step {i+1}: {location}", value=i == 0)\
                                .classes("w-full mb-2 border rounded-lg bg-white"):
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
                                        f'<div class="text-sm p-2 bg-gray-50 border rounded '
                                        f'max-h-20 overflow-y-auto">{orig[:300]}'
                                        f'{"..." if len(orig)>300 else ""}</div>'
                                    )

                                with ui.column().classes("w-full"):
                                    ui.label("Translation:")\
                                      .classes("text-xs font-semibold text-gray-600")
                                    textarea = ui.textarea(value=trans)\
                                      .props("autogrow rows=3")\
                                      .classes("w-full text-sm bg-white")
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

                # ── DOWNLOAD & NAV ───────────────────────────────
                ui.separator().classes("my-4")
                with ui.row().classes("justify-center space-x-4 mt-6 flex-wrap"):
                    ui.button("Download Translated File", on_click=self.download_file)\
                      .classes("bg-blue-600 text-white px-6 py-2 rounded shadow")
                    ui.button("Upload Another File", on_click=self.request_refresh_upload_ui)\
                      .classes("bg-gray-500 text-white px-6 py-2 rounded shadow")

        # stats footer
        with self.stats_container:
            ui.label(f"Elements translated: {self.current_count}")\
              .classes("text-base text-gray-700")
            if self.current_cost > 0:
                ui.label(f"Estimated cost: ${self.current_cost:.4f}")\
                  .classes("text-sm text-gray-600")

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
        # clear UI, then show error label
        self.result_container.clear()
        self.stats_container.clear()
        with self.result_container:
            ui.label(f"An error occurred: {error}")\
              .style("font-size: 18px; color: #e53935;")

    # ─────────────────────────────── VOICE TRANSLATION PAGE ─────────────────────────────────

    def voice_translation_page(self):
        ui.label("Live Voice Translation").classes("text-2xl mb-4")

        with ui.row().classes("items-center space-x-2 mb-4"):
            ui.label("Target language:").classes("font-medium")
            ui.html('''
                <select id="language_select" class="px-3 py-2 border rounded bg-white">
                    <option value="en">English</option>
                    <option value="es" selected>Spanish</option>
                    <option value="fr">French</option>
                    <option value="de">German</option>
                    <option value="zh">Chinese</option>
                </select>
            ''')

        status_label = ui.label("Ready to record")\
                         .classes("text-lg mb-2")\
                         .props("id=status_label")

        with ui.row().classes("space-x-4 mb-4"):
            ui.html('<button id="start_btn" class="bg-green-500 text-white px-4 py-2 rounded" '
                    'onclick="startRecording()">🎤 Start Recording</button>')
            ui.html('<button id="stop_btn" class="bg-red-500 text-white px-4 py-2 rounded" '
                    'onclick="stopRecording()" disabled>⏹️ Stop & Translate</button>')
            ui.button("← Back", on_click=lambda: ui.navigate.to("/"))\
              .classes("bg-gray-500 text-white px-4 py-2 rounded")

        ui.label("Debug Info:").classes("font-bold mt-4")
        ui.label("").classes("text-sm text-gray-600").props("id=debug_info")

        ui.audio(src="data:audio/wav;base64,")\
          .props("id=out_audio controls")\
          .classes("w-full")

        ui.label("Original:").classes("font-bold mt-4")
        ui.label("").classes("p-2 border rounded bg-gray-50 min-h-[40px]")\
          .props("id=original_text")

        ui.label("Translation:").classes("font-bold mt-2")
        ui.label("").classes("p-2 border rounded bg-blue-50 min-h-[40px]")\
          .props("id=translated_text")

        ui.add_head_html("""
<script>
    /* Full recording/MediaRecorder logic from your original */
    let recorder = null, stream = null, chunks = [], isRecording = false;
    let selectedMimeType = null;

    function updateStatus(msg) {
        let e = document.getElementById('status_label');
        if (e) e.textContent = msg;
    }
    function updateDebug(msg) {
        let e = document.getElementById('debug_info');
        if (e) e.textContent = msg;
    }
    function updateButtons(recording) {
        let start = document.getElementById('start_btn'),
            stop  = document.getElementById('stop_btn');
        if (start) { start.disabled = recording; start.style.opacity = recording?'0.5':'1'; }
        if (stop)  { stop.disabled  = !recording; stop.style.opacity = recording?'1':'0.5'; }
        isRecording = recording;
    }
    function setRecordingControlsEnabled(enabled) {
        let start = document.getElementById('start_btn'),
            stop  = document.getElementById('stop_btn');
        if (start) { start.disabled = !enabled; start.style.opacity = enabled ? '1' : '0.5'; }
        if (stop)  { stop.disabled  = true; stop.style.opacity = '0.5'; }
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
            updateStatus("Not recording");
            return;
        }
        updateStatus("Stopping…");
        updateButtons(false);
        recorder.onstop = async () => {
            updateStatus("Processing audio…");
            const blob = new Blob(chunks, { type: recorder.mimeType });
            let lang = document.getElementById('language_select')?.value || 'es';
            let fd = new FormData();
            fd.append('file', blob, `rec.${blob.type.includes('webm')?'webm':'mp4'}`);
            fd.append('language', lang);
            try {
                const resp = await fetch('/api/voice_translate', { method:'POST', body:fd });
                if(!resp.ok) throw new Error(await resp.text());
                const audio = await resp.blob();
                const orig = resp.headers.get('X-Original-Text') || '';
                const trans = resp.headers.get('X-Translated-Text') || '';
                document.getElementById('original_text').textContent   = orig;
                document.getElementById('translated_text').textContent= trans;
                if(audio.size>0){
                    let url = URL.createObjectURL(audio);
                    let player = document.getElementById('out_audio');
                    player.src = url; player.play();
                    updateStatus("✅ Done");
                }
            } catch(e) {
                updateStatus("Error:"+e.message);
                updateDebug(e.message);
            } finally {
                if(stream) stream.getTracks().forEach(t=>t.stop());
                recorder = null; chunks = [];
            }
        };
        recorder.stop();
    }

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

    async def api_voice_translate(
        self,
        file: UploadFile = File(...),
        language: str = Form(...)
    ) -> Response:
        correlation_id = str(uuid.uuid4())
        try:
            if not language or language.lower() in ('undefined','null',''):
                language = 'es'
                logging.warning("[API] Empty language → default to Spanish")
            _log_event("ui.voice_translate_requested", correlation_id=correlation_id, language=language)
            data = await file.read()
            if not data:
                return Response(content=b"", status_code=400, headers={"X-Error":"Empty audio data"})
            original_text, mp3_bytes = await asyncio.to_thread(
                self.backend.translate_audio, data, language
            )
            # build headers
            translation_text = {
                'es':'Translated to Spanish',
                'fr':'Translated to French',
                'de':'Translated to German',
                'zh':'Translated to Chinese'
            }.get(language, f"Translated to {language.upper()}")
            safe = original_text[:200] if original_text else "Transcription failed"
            try:
                import urllib.parse
                if len(safe.encode('ascii','ignore')) > len(safe)*0.5:
                    header_orig = safe
                else:
                    header_orig = f"Text in {language}"
            except:
                header_orig = f"Text in {language}"
            return Response(
                content=mp3_bytes,
                media_type="audio/mpeg",
                headers={
                    "X-Original-Text": header_orig,
                    "X-Translated-Text": translation_text,
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


if __name__ in {"__main__", "__mp_main__"}:
    TranslationUI().start_ui()
