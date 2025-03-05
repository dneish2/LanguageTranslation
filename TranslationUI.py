from nicegui import ui
from io import BytesIO
from threading import Thread
from TranslationBackend import TranslationBackend
import logging
import os

# Light logging setup
logging.basicConfig(level=logging.INFO)

class TranslationUI:
    def __init__(self):
        self.backend = TranslationBackend()
        # UI containers
        self.upload_container = None
        self.result_container = None
        self.progress_container = None
        self.stats_container = None

        # Uploaded file info
        self.uploaded_file = None
        self.uploaded_file_name = None
        self.uploaded_file_extension = None
        self.current_target_language = None
        self.cancel_button = None

        # Segment data
        self.original_segments_map = {}
        self.translated_segments_map = {}

        # Drawer reference
        self.drawer = None

    def start_ui(self):
        ui.page("/")(self.main_page)
        ui.run(port=3030)

    def main_page(self):
        """
        Layout:
        1) A header with "Translation App" label and "Recent Documentss" button.
        2) A drawer at the same level as the header (NOT nested inside it).
        3) A column that centers the main content (upload, progress, results).
        """

        # Top bar
        with ui.header().classes("items-center justify-between bg-gray-100 p-2"):
            with ui.row().classes("w-full flex justify-between items-center"):
                ui.label("Translation App").classes("text-lg font-bold text-black")
                ui.button("Advanced Mode", on_click=self.activate_advanced_mode)\
                    .classes("bg-gray-200 text-gray-700 px-4 py-2 rounded shadow")
            
        # Drawer as a top-level layout element (sibling to the header)
        self.drawer = ui.drawer(side='left').classes("bg-gray-50")
        with self.drawer:
            ui.label("Recent Documents").classes("font-bold text-lg mb-2")
            self.show_document_list()

        # Main content area (centered)
        with ui.column().classes("w-full h-full items-center justify-center p-4"):
            self.upload_container = ui.column().classes("w-full max-w-3xl items-center")
            self.progress_container = ui.column().classes("w-full max-w-3xl items-center")
            self.result_container = ui.column().classes("w-full max-w-3xl items-center")
            self.stats_container = ui.column().classes("w-full max-w-3xl items-center")
            self.refresh_upload_ui()

    def activate_advanced_mode(self):
        # Placeholder method for your future advanced LLM review logic
        ui.notify("Advanced Mode activated! (No logic implemented yet.)")

    def show_document_list(self):
        """Populate the drawer with recent files that start with 'translated_'. """
        files = [f for f in os.listdir('.') if f.startswith("translated_")]
        if not files:
            ui.label("No recent documents.").classes("text-sm text-gray-600")
        else:
            files.sort(key=lambda f: os.path.getmtime(f), reverse=True)
            for filename in files:
                ui.button(filename, on_click=lambda _, fn=filename: self.load_processed_document(fn))\
                    .classes("w-full text-left mb-1")

    def load_processed_document(self, filename):
        if not os.path.exists(filename):
            ui.notify(f"File {filename} not found.")
            return
        with open(filename, 'rb') as f:
            file_data = f.read()
        self.uploaded_file = BytesIO(file_data)
        self.uploaded_file_name = filename
        self.uploaded_file_extension = filename.split(".")[-1].lower()
        ui.notify(f"Loaded processed file: {filename}")
        self.handle_translation_processed()

    def refresh_upload_ui(self):
        """Clear UI containers and prompt user to upload a new file."""
        self.upload_container.clear()
        self.progress_container.clear()
        self.result_container.clear()
        self.stats_container.clear()
        self.backend.segment_map.clear()

        with self.upload_container:
            ui.label("Upload a document (DOCX, PPTX, or PDF)")\
                .style("font-size: 20px; color: #333; margin-bottom: 10px; text-align: center;")
            ui.upload(
                label="Click or drop a file here",
                multiple=False,
                on_upload=self.handle_upload
            )

    def handle_upload(self, event):
        """Handle the uploaded file and ask for target language."""
        self.uploaded_file_name = event.name
        self.uploaded_file_extension = self.uploaded_file_name.split(".")[-1].lower()
        self.uploaded_file = BytesIO(event.content.read())
        logging.info(f"[UI] File '{self.uploaded_file_name}' uploaded. Extension={self.uploaded_file_extension}")

        self.upload_container.clear()
        with self.upload_container:
            ui.label(f"File '{self.uploaded_file_name}' uploaded successfully!")\
                .style("font-size: 18px; color: #333; margin-bottom: 6px; text-align: center;")
            ui.label("Select a target language for translation:")\
                .style("font-size: 16px; color: #555; margin-bottom: 8px; text-align: center;")
            lang_input = ui.input(label="Target Language", placeholder="e.g., Spanish")
            ui.button("Translate", on_click=lambda: self.handle_translation(lang_input.value))\
                .classes("bg-blue-600 text-white px-4 py-2 rounded shadow mt-2")

    def handle_translation(self, target_language):
        if not target_language:
            self.show_error("Please enter a valid target language.")
            return
        self.current_target_language = target_language
        logging.info(f"[UI] Starting translation for '{self.uploaded_file_name}' to '{target_language}'")

        self.progress_container.clear()
        self.result_container.clear()
        self.stats_container.clear()

        progress_ui = ui.circular_progress(value=0.0, max=100, show_value=True)\
            .classes("mx-auto mt-4")\
            .style("color: #ff9800;")
        label_ui = ui.label("Preparing translation...")\
            .classes("text-center text-gray-700 mt-2")

        self.cancel_button = ui.button("Cancel Translation", on_click=self.cancel_translation)\
            .classes("bg-red-500 text-white px-4 py-2 rounded shadow mt-2")

        with self.progress_container:
            progress_ui
            label_ui
            self.cancel_button

        def translation_task():
            try:
                (out_stream, count, tokens, translated_text, seg_map) = self.backend.translate_file(
                    self.uploaded_file,
                    self.uploaded_file_extension,
                    target_language,
                    progress_ui,
                    label_ui,
                    processed=False
                )
                if self.backend.cancel_requested:
                    logging.info("[UI] Translation canceled mid-way.")
                    return

                progress_ui.set_value(100)
                label_ui.text = "Translation complete."
                self.backend.regenerate_output_stream()

                self.original_segments_map.clear()
                self.translated_segments_map.clear()
                for seg_id, seg_info in seg_map.items():
                    self.original_segments_map[seg_id] = seg_info["original"]
                    self.translated_segments_map[seg_id] = seg_info["translated"]

                cost = tokens * 0.002 / 1000
                self.show_result(
                    self.uploaded_file_name,
                    target_language,
                    self.backend.output_stream,
                    count,
                    cost,
                    translated_text,
                    seg_map
                )
            except Exception as e:
                logging.error(f"[UI] Translation error: {e}", exc_info=True)
                self.show_error(e)

        Thread(target=translation_task).start()

    def handle_translation_processed(self):
        """Load a file in processed mode (no re-translation)."""
        self.progress_container.clear()
        self.result_container.clear()
        self.stats_container.clear()

        progress_ui = ui.circular_progress(value=0.0, max=100, show_value=True)\
            .classes("mx-auto mt-4")\
            .style("color: #ff9800;")
        label_ui = ui.label("Loading processed document...")\
            .classes("text-center text-gray-700 mt-2")

        self.cancel_button = ui.button("Cancel", on_click=self.cancel_translation)\
            .classes("bg-red-500 text-white px-4 py-2 rounded shadow mt-2")

        with self.progress_container:
            progress_ui
            label_ui
            self.cancel_button

        def processed_task():
            try:
                (out_stream, count, tokens, segmented_text, seg_map) = self.backend.translate_file(
                    self.uploaded_file,
                    self.uploaded_file_extension,
                    "",
                    progress_ui,
                    label_ui,
                    processed=True
                )
                progress_ui.set_value(100)
                label_ui.text = "File loaded."
                self.backend.regenerate_output_stream()

                self.original_segments_map.clear()
                self.translated_segments_map.clear()
                for seg_id, seg_info in seg_map.items():
                    self.original_segments_map[seg_id] = seg_info["original"]
                    self.translated_segments_map[seg_id] = seg_info["translated"]

                self.show_result(
                    self.uploaded_file_name,
                    "Processed",
                    self.backend.output_stream,
                    count,
                    0,
                    segmented_text,
                    seg_map
                )
            except Exception as e:
                logging.error(f"[UI] Error loading processed doc: {e}", exc_info=True)
                self.show_error(e)

        Thread(target=processed_task).start()

    def cancel_translation(self):
        self.backend.request_cancel()
        self.show_error("Translation was canceled. Please upload or try again.")

    def show_result(self, file_name, target_language, out_stream, count, cost, translated_text, seg_map):
        self.result_container.clear()
        self.stats_container.clear()

        if self.cancel_button:
            self.cancel_button.visible = False

        with self.result_container:
            with ui.column().classes("max-w-3xl mx-auto w-full space-y-6 relative mt-4"):
                ui.label(f"'{file_name}' translated to {target_language}.")\
                    .classes("text-2xl font-semibold text-gray-800")
                ui.label("Each segment shows the original (read-only) and the current translation (editable).")\
                    .classes("text-sm text-gray-600 mb-4")

                for seg_id in list(self.original_segments_map.keys()):
                    orig = self.original_segments_map[seg_id]
                    trans = self.translated_segments_map[seg_id]
                    with ui.card().classes("border-l-4 border-blue-500 shadow-md p-4 relative"):
                        # Delete button top-right
                        ui.button("Delete", on_click=lambda _, s=seg_id: self.delete_segment_callback(s))\
                            .props("size=small color=danger")\
                            .style("position: absolute; top: 8px; right: 8px;")

                        ui.label(f"Segment ID: {seg_id[:8]}...").classes("font-bold text-lg text-gray-700 mb-3")

                        with ui.column().classes("bg-gray-50 rounded p-3 mb-3"):
                            ui.label("Original").classes("font-semibold text-gray-700 mb-1")
                            ui.html(f"<div class='text-base text-gray-800 leading-relaxed'>{orig}</div>")

                        ui.label("Current Translation").classes("font-semibold text-gray-700 mb-1")
                        text_area = ui.textarea(value=trans)\
                            .props("autogrow")\
                            .classes("text-base text-gray-800 leading-relaxed w-full mb-3")

                        refine_input = ui.input(label="Refinement Instructions (optional)")\
                            .props("clearable")\
                            .classes("mb-3")

                        with ui.row().classes("justify-between"):
                            ui.button("Update", on_click=lambda _, s=seg_id, ta=text_area, rin=refine_input: self.update_segment_callback(s, ta, rin))\
                                .props("size=small color=primary")

                with ui.row().classes("justify-start space-x-4 mt-6"):
                    ui.button(
                        "Download Translated File",
                        on_click=lambda: ui.download(self.backend.output_stream.read(), f"translated_{file_name}")
                    ).classes("bg-blue-600 text-white px-4 py-2 rounded shadow")

                    ui.button(
                        "Upload Another File",
                        on_click=self.refresh_upload_ui
                    ).classes("bg-gray-200 text-gray-800 px-4 py-2 rounded shadow")

        with self.stats_container:
            ui.label(f"Elements translated: {count}").classes("text-base text-gray-700 mt-4")
            ui.label(f"Estimated cost: ${cost:.4f}").classes("text-base text-gray-700 mb-4")

    def update_segment_callback(self, seg_id, text_area, refine_input):
        try:
            updated = self.backend.update_segment(seg_id, text_area.value, self.current_target_language, refine_input.value)
            text_area.value = updated
            self.translated_segments_map[seg_id] = updated
            ui.notify(f"Segment {seg_id[:8]} updated.")
        except Exception as ex:
            ui.notify(f"Error updating segment {seg_id[:8]}: {ex}")

    def delete_segment_callback(self, seg_id):
        try:
            self.backend.delete_segment(seg_id)
            if seg_id in self.original_segments_map:
                del self.original_segments_map[seg_id]
            if seg_id in self.translated_segments_map:
                del self.translated_segments_map[seg_id]
            ui.notify(f"Segment {seg_id[:8]} deleted.")

            self.show_result(
                self.uploaded_file_name,
                self.current_target_language or "Processed",
                self.backend.output_stream,
                len(self.backend.segment_map),
                0,
                "",
                self.backend.segment_map
            )
        except Exception as ex:
            ui.notify(f"Error deleting segment {seg_id[:8]}: {ex}")

    def show_error(self, error):
        self.result_container.clear()
        self.stats_container.clear()
        with self.result_container:
            ui.label(f"An error occurred: {error}").style("font-size: 18px; color: #e53935;")

if __name__ in {"__main__", "__mp_main__"}:
    app = TranslationUI()
    app.start_ui()