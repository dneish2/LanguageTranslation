from nicegui import ui
from io import BytesIO
from TranslationBackend import TranslationBackend
from threading import Thread

class TranslationUI:
    def __init__(self):
        self.backend = TranslationBackend()  # Backend instance
        self.upload_container = None
        self.result_container = None
        self.progress_container = None
        self.stats_container = None
        self.uploaded_file = None
        self.uploaded_file_name = None
        self.uploaded_file_extension = None

    def start_ui(self):
        ui.page("/")(self.main_page)
        ui.run(port=3030)

    def main_page(self):
        """Define the main page layout."""
        with ui.column().classes("absolute-center w-full h-full"):
            self.upload_container = ui.column().classes("items-center w-full")
            self.result_container = ui.column().classes("items-center w-full")
            self.progress_container = ui.column().classes("items-center w-full")
            self.stats_container = ui.column().classes("items-center w-full")
        self.refresh_upload_ui()

    def refresh_upload_ui(self):
        """Refresh the upload container with the initial UI elements."""
        self.upload_container.clear()
        self.progress_container.clear()
        self.result_container.clear()
        self.stats_container.clear()

        with self.upload_container:
            ui.label("Please upload a document (DOCX, PPTX, or PDF)").style("font-size: 20px; color: #333;")
            ui.upload(on_upload=self.handle_upload, multiple=False)

    def handle_upload(self, e):
        """Handle the uploaded file and store it in memory. Prompt for language selection."""
        self.uploaded_file_name = e.name
        self.uploaded_file_extension = self.uploaded_file_name.split(".")[-1].lower()
        self.uploaded_file = BytesIO(e.content.read())

        # Clear the upload container and prompt for language selection
        self.upload_container.clear()
        with self.upload_container:
            ui.label(f"File '{self.uploaded_file_name}' uploaded successfully!").style("font-size: 18px; color: #333;")
            ui.label("Please select a target language for translation:").style("font-size: 16px; color: #555;")
            language_input = ui.input(label="Target Language", placeholder="e.g., Spanish")
            ui.button("Translate", on_click=lambda: self.handle_translation(language_input.value))

    def handle_translation(self, target_language):
        """Translate the uploaded file after language selection."""
        if not target_language:
            self.show_error("Please enter a valid target language.")
            return

        self.progress_container.clear()
        self.result_container.clear()
        self.stats_container.clear()

        progress_ui = ui.circular_progress(value=0.0, max=100, show_value=True).classes("mx-auto").style("color: #ff9800;")
        label_ui = ui.label("Preparing translation...").classes("text-center").style("color: #333;")

        with self.progress_container:
            progress_ui
            label_ui

        def translation_task():
            try:
                (
                    output_stream,  # The final translated file in memory
                    count,          # Number of items translated
                    tokens,         # Token usage
                    translated_text, 
                    original_segments, 
                    translated_segments
                ) = self.backend.translate_file(
                    self.uploaded_file, 
                    self.uploaded_file_extension, 
                    target_language, 
                    progress_ui, 
                    label_ui
                )
                cost = tokens * 0.002 / 1000
                self.show_result(
                    self.uploaded_file_name, 
                    target_language, 
                    output_stream, 
                    count, 
                    cost, 
                    translated_text, 
                    original_segments, 
                    translated_segments
                )
            except Exception as e:
                self.show_error(e)

        Thread(target=translation_task).start()

    def show_result(self, file_name, target_language, output_stream, count, cost, translated_text, original_segments, translated_segments):
        """Display the translated file result, statistics, and a split screen view of original vs translated text."""
        self.result_container.clear()
        self.stats_container.clear()

        with self.result_container:
            ui.label(f"The file '{file_name}' has been translated to {target_language}.").style("font-size: 18px; color: #333;")

            # Create a row with two columns: left for original, right for translated
            with ui.row().classes("w-full justify-around"):
                with ui.column().classes("w-1/2 p-4 border-r"):
                    ui.label("Original Text:").style("font-weight: bold; font-size:18px; color:#333;")
                    for orig_seg in original_segments:
                        ui.html(f"<div style='white-space:pre-wrap; font-size:16px; color:#333; margin-bottom:10px;'>{orig_seg}</div>")

                with ui.column().classes("w-1/2 p-4"):
                    ui.label("Translated Text:").style("font-weight: bold; font-size:18px; color:#333;")
                    for trans_seg in translated_segments:
                        ui.html(f"<div style='white-space:pre-wrap; font-size:16px; color:#333; margin-bottom:10px;'>{trans_seg}</div>")

            ui.button(
                "Download Translated File",
                on_click=lambda: ui.download(output_stream.read(), f"translated_{file_name}"),
            )
            ui.button("Upload Another File", on_click=self.refresh_upload_ui)

        with self.stats_container:
            ui.label(f"Number of elements translated: {count}").style("font-size: 16px; color: #333;")
            ui.label(f"Estimated cost: ${cost:.4f}").style("font-size: 16px; color: #333;")

    def show_error(self, error):
        """Display an error message."""
        self.result_container.clear()
        with self.result_container:
            ui.label(f"An error occurred: {error}").style("font-size: 18px; color: #e53935;")

if __name__ in {"__main__", "__mp_main__"}:
    app = TranslationUI()
    app.start_ui()