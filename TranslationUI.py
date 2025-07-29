import asyncio
import logging
import os
import glob
from io import BytesIO
from threading import Thread

from nicegui import ui, app
from fastapi import UploadFile, File, Form
from starlette.responses import Response

from TranslationBackend import TranslationBackend

logging.basicConfig(level=logging.INFO)


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
        self.cancel_button = None

        # ── SEGMENT EDITING ─────────────────────────────────────────────
        self.original_segments_map: dict[str, str] = {}
        self.translated_segments_map: dict[str, str] = {}

        # ── DRAWER & ADVANCED MODE ──────────────────────────────────────
        self.drawer = None
        self.advanced_mode = False
        self.advanced_button = None

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
        # run on single port
        ui.run(host="0.0.0.0", port=8080)

    # ──────────────────────────────────── MAIN PAGE ─────────────────────────────────────────

    def main_page(self):
        # Header with Advanced + Voice buttons
        with ui.header().classes("items-center justify-between bg-gray-100 p-2"):
            with ui.row().classes("w-full flex justify-between items-center"):
                ui.label("Translation App").classes("text-lg font-bold text-black")
                with ui.row().classes("space-x-2"):
                    self.advanced_button = ui.button(
                        "Enable Advanced Mode",
                        on_click=self.toggle_advanced_mode
                    ).classes("bg-gray-200 text-gray-700 px-4 py-2 rounded shadow")
                    ui.button(
                        "Live Voice Translation",
                        on_click=lambda: ui.navigate.to("/voice")
                    ).classes("bg-gray-200 text-gray-700 px-4 py-2 rounded shadow")

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

    def toggle_advanced_mode(self):
        self.advanced_mode = not self.advanced_mode
        if self.advanced_mode:
            self.advanced_button.text = "Advanced Mode Enabled"
            ui.notify("Advanced Mode activated! Segment editing features are now available.")
        else:
            self.advanced_button.text = "Enable Advanced Mode"
            ui.notify("Advanced Mode disabled. Segment editing hidden.")

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

        with self.upload_container:
            ui.label("Upload a document (DOCX, PPTX, or PDF)")\
                .style("font-size: 20px; color: #333; margin-bottom: 10px; text-align: center;")
            ui.upload(
                label="Click or drop a file here",
                multiple=False,
                on_upload=self.handle_upload
            )

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

        def translation_task():
            try:
                out_stream, count, tokens, _, seg_map = self.backend.translate_file(
                    self.uploaded_file,
                    self.uploaded_file_extension,
                    target_language,
                    progress_ui,
                    label_ui,
                    processed=False,
                    font_size=font_size,
                    autofit=autofit
                )
                if self.backend.cancel_requested:
                    logging.info("[UI] Translation canceled mid-way.")
                    return

                progress_ui.set_value(100)
                label_ui.text = "Translation complete."

                # save stats
                self.current_count = count
                self.current_cost = tokens * 0.002 / 1000

                # rebuild segment maps
                self.original_segments_map.clear()
                self.translated_segments_map.clear()
                for seg_id, seg_info in seg_map.items():
                    self.original_segments_map[seg_id] = seg_info["original"]
                    self.translated_segments_map[seg_id] = seg_info["translated"]

                self.show_result()
            except Exception as e:
                logging.error(f"[UI] Translation error: {e}", exc_info=True)
                self.show_error(e)

        Thread(target=translation_task).start()

    def handle_translation_processed(self):
        # display a pre-translated file without re-calling openai
        self.progress_container.clear()
        self.result_container.clear()
        self.stats_container.clear()

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

        def processed_task():
            try:
                out_stream, count, _, _, seg_map = self.backend.translate_file(
                    self.uploaded_file,
                    self.uploaded_file_extension,
                    "",
                    progress_ui,
                    label_ui,
                    processed=True
                )
                progress_ui.set_value(100)
                label_ui.text = "File loaded."

                self.current_count = count
                self.current_cost = 0.0
                self.current_target_language = "Processed"

                self.original_segments_map.clear()
                self.translated_segments_map.clear()
                for seg_id, seg_info in seg_map.items():
                    self.original_segments_map[seg_id] = seg_info["original"]
                    self.translated_segments_map[seg_id] = seg_info["translated"]

                self.show_result()
            except Exception as e:
                logging.error(f"[UI] Error loading processed doc: {e}", exc_info=True)
                self.show_error(e)

        Thread(target=processed_task).start()

    def cancel_translation(self):
        self.backend.request_cancel()
        self.show_error("Translation was canceled. Please upload or try again.")

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
                if self.advanced_mode and self.original_segments_map:
                    ui.separator().classes("my-4")
                    ui.label("Advanced Mode: Segment Editor")\
                        .classes("text-xl font-bold mb-2")
                    ui.label(f"Review and edit {len(self.original_segments_map)} segments below:")\
                        .classes("text-sm text-gray-600 mb-4")

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

                        with ui.card().classes("w-full mb-3 p-4 border"):
                            # header
                            with ui.row().classes("justify-between items-center mb-3"):
                                ui.label(f"#{i+1}: {location}")\
                                  .classes("text-sm font-medium text-gray-700")
                                with ui.row().classes("space-x-1"):
                                    ui.button("✓", on_click=lambda _, s=seg_id: self.approve_segment_callback(s))\
                                      .props("size=sm color=positive")
                                    ui.button("✗", on_click=lambda _, s=seg_id: self.decline_segment_callback(s))\
                                      .props("size=sm color=negative")
                                    ui.button("🗑️", on_click=lambda _, s=seg_id: self.delete_segment_callback(s))\
                                      .props("size=sm color=grey")

                            # original text
                            with ui.column().classes("w-full mb-2"):
                                ui.label("Original:")\
                                  .classes("text-xs font-semibold text-gray-600")
                                ui.html(
                                    f'<div class="text-sm p-2 bg-gray-50 border rounded '
                                    f'max-h-20 overflow-y-auto">{orig[:300]}'
                                    f'{"..." if len(orig)>300 else ""}</div>'
                                )

                            # translation (editable)
                            with ui.column().classes("w-full mb-2"):
                                ui.label("Translation:")\
                                  .classes("text-xs font-semibold text-gray-600")
                                textarea = ui.textarea(value=trans)\
                                  .props("autogrow rows=2")\
                                  .classes("w-full text-sm")
                                textarea.segment_id = seg_id

                            # update controls
                            with ui.row().classes("space-x-2"):
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
                with ui.row().classes("justify-center space-x-4 mt-6"):
                    ui.button("Download Translated File", on_click=self.download_file)\
                      .classes("bg-blue-600 text-white px-6 py-2 rounded shadow")
                    ui.button("Upload Another File", on_click=self.refresh_upload_ui)\
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

    def approve_segment_callback(self, seg_id):
        try:
            orig = self.original_segments_map.get(seg_id, "")
            trans = self.translated_segments_map.get(seg_id, "")
            self.backend.record_feedback(original=orig, translated=trans)
            ui.notify("Segment approved ✓", type="positive")
        except Exception as ex:
            logging.error(f"[UI] Error approving segment {seg_id}: {ex}", exc_info=True)
            ui.notify(f"Approval failed: {ex}", type="negative")

    def decline_segment_callback(self, seg_id):
        try:
            orig = self.original_segments_map.get(seg_id, "")
            trans = self.translated_segments_map.get(seg_id, "")
            self.backend.record_feedback(seg_id, approved=False, original=orig, translated=trans)
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

    async function startRecording() {
        updateStatus("Requesting mic…");
        updateDebug("Starting...");
        try {
            const constraints = {
                audio:{echoCancellation:true,noiseSuppression:true,autoGainControl:true,sampleRate:44100}
            };
            stream = await navigator.mediaDevices.getUserMedia(constraints);
            recorder = new MediaRecorder(stream, { mimeType:'audio/webm;codecs=opus' });
            chunks = [];
            recorder.ondataavailable = e => { if(e.data.size>0) { chunks.push(e.data); updateDebug(`Chunks:${chunks.length}`); } };
            recorder.onstart = () => { updateStatus("🔴 Recording…"); updateButtons(true); };
            recorder.onerror = e => { updateStatus("Error: "+e.error.message); updateButtons(false); };
            recorder.start(1000);
        } catch (err) {
            updateStatus("Error: "+err.message);
            updateDebug(err.message);
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
        updateButtons(false);
        updateStatus("Ready to record");
    });
</script>
        """)

    # ─────────────────────────── VOICE TRANSLATION API ────────────────────────────────────

    async def api_voice_translate(
        self,
        file: UploadFile = File(...),
        language: str = Form(...)
    ) -> Response:
        try:
            if not language or language.lower() in ('undefined','null',''):
                language = 'es'
                logging.warning("[API] Empty language → default to Spanish")
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
                    "Content-Length": str(len(mp3_bytes))
                }
            )
        except Exception as e:
            logging.error(f"[API] Voice error: {e}", exc_info=True)
            msg = f"Translation error: {e}"
            return Response(content=msg.encode(), status_code=500,
                            media_type="text/plain", headers={"X-Error":msg})


if __name__ in {"__main__", "__mp_main__"}:
    TranslationUI().start_ui()