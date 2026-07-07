import sys
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TranslationBackend import TranslationBackend, TranslationRunState
from TranslationUI import TranslationUI


class DummyContainer:
    def clear(self):
        return None


class DummyInput:
    def __init__(self, value=""):
        self.value = value


class DummyProgress:
    def __init__(self):
        self.values = []

    def set_value(self, value):
        self.values.append(value)


class DummyLabel:
    def __init__(self):
        self.text = ""


class DummyButton:
    def __init__(self):
        self.enabled = True

    def set_enabled(self, value):
        self.enabled = value


def _build_mobile_ui() -> TranslationUI:
    ui_app = TranslationUI()
    ui_app.mobile_mode = True
    ui_app.upload_container = DummyContainer()
    ui_app.progress_container = DummyContainer()
    ui_app.result_container = DummyContainer()
    ui_app.stats_container = DummyContainer()
    return ui_app


def test_mobile_upload_happy_path_invokes_translation():
    """Unified workflow: Document mode + language should kick off translation."""
    ui_app = _build_mobile_ui()
    ui_app.input_mode = "Document"
    ui_app.mobile_input_mode = "Document"
    ui_app.target_language_input = DummyInput("Spanish")
    ui_app.uploaded_file = BytesIO(b"doc")

    called = {}

    def fake_handle_translation(language, *_args, **_kwargs):
        called["language"] = language

    ui_app.handle_translation = fake_handle_translation
    ui_app.start_mobile_translation()

    assert called["language"] == "Spanish"


def test_mobile_upload_error_path_requires_file():
    """Unified workflow: document mode still requires an uploaded file."""
    ui_app = _build_mobile_ui()
    ui_app.input_mode = "Document"
    ui_app.mobile_input_mode = "Document"
    ui_app.target_language_input = DummyInput("French")
    ui_app.uploaded_file = None

    errors = []
    ui_app.show_error = lambda message: errors.append(str(message))

    ui_app.start_mobile_translation()

    assert errors == ["Please upload a file before translating."]


def test_text_mode_happy_path_updates_progress_and_result(monkeypatch):
    ui_app = _build_mobile_ui()
    ui_app.input_mode = "Text"
    ui_app.target_language_input = DummyInput("German")
    ui_app.text_source_input = DummyInput("hello")
    progress = DummyProgress()
    label = DummyLabel()

    monkeypatch.setattr(ui_app.backend, "translate_text", lambda text, lang: f"{lang}:{text}")

    rendered = {}
    ui_app.show_mobile_voice_result = lambda original, translated, language: rendered.update(
        original=original,
        translated=translated,
        language=language,
    )

    ui_app._run_mobile_voice_translation("hello", "German", progress, label)

    assert progress.values == [40, 100]
    assert label.text == "Translation complete."
    assert rendered == {
        "original": "hello",
        "translated": "German:hello",
        "language": "German",
    }


def test_text_mode_error_path_requires_text():
    ui_app = _build_mobile_ui()
    ui_app.input_mode = "Text"
    ui_app.target_language_input = DummyInput("German")
    ui_app.text_source_input = DummyInput("   ")

    errors = []
    ui_app.show_error = lambda message: errors.append(str(message))

    ui_app.start_mobile_translation()

    assert errors == ["Please provide source text before translating."]


def test_text_mode_rejects_identical_source_and_target_language():
    ui_app = _build_mobile_ui()
    ui_app.input_mode = "Text"
    ui_app.source_language_input = DummyInput("Spanish")
    ui_app.target_language_input = DummyInput("spanish")  # case-insensitive match
    ui_app.text_source_input = DummyInput("Hola, ¿qué tal?")

    errors = []
    ui_app.show_error = lambda message: errors.append(str(message))
    translated = []
    ui_app.backend.translate_text = lambda *a, **k: translated.append(1)

    ui_app.start_mobile_translation()

    assert errors == ["From and To are the same language — swap ⇄ or pick a different target."]
    assert not translated


def test_document_mode_rejects_identical_source_and_target_language():
    ui_app = _build_mobile_ui()
    ui_app.input_mode = "Document"
    ui_app.mobile_input_mode = "Document"
    ui_app.source_language_input = DummyInput("French")
    ui_app.target_language_input = DummyInput("French")
    ui_app.uploaded_file = BytesIO(b"doc")

    errors = []
    ui_app.show_error = lambda message: errors.append(str(message))
    ui_app.handle_translation = lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not run"))

    ui_app.start_mobile_translation()

    assert errors == ["From and To are the same language — swap ⇄ or pick a different target."]


def test_show_error_classifies_short_message_as_direct_banner():
    assert TranslationUI._is_technical_error("Please upload a file before translating.") is False


def test_show_error_classifies_provider_exception_as_technical():
    long_openai_error = (
        "Error code: 400 - {'error': {'message': 'Audio file might be corrupted', "
        "'type': 'invalid_request_error'}}"
    )
    assert TranslationUI._is_technical_error(long_openai_error) is True


def test_show_error_classifies_long_message_as_technical_even_without_markers():
    assert TranslationUI._is_technical_error("x" * 141) is True
    assert TranslationUI._is_technical_error("x" * 140) is False


def _use_fake_thread_storage(monkeypatch, ui_app):
    """recent_threads is backed by app.storage.user (a real request/session
    context) so cross-user data can't leak through a shared instance
    attribute (see PASSAGE_PLAN.md, 2026-07-06 code-review fix). Outside a
    real request the property degrades to a fresh throwaway list each call
    (fail-safe, not a test seam) — so dedupe/ordering tests substitute a
    plain dict-backed store here to exercise _record_thread's own logic."""
    backing: dict = {}
    monkeypatch.setattr(
        type(ui_app), "recent_threads", property(lambda self: backing.setdefault("threads", []))
    )


def test_record_thread_dedupes_repeated_chat_by_moving_it_to_the_top(monkeypatch):
    ui_app = _build_mobile_ui()
    _use_fake_thread_storage(monkeypatch, ui_app)
    ui_app._record_chat_thread("Hello", "Hola", "Spanish")
    ui_app._record_chat_thread("Goodbye", "Adiós", "Spanish")
    ui_app._record_chat_thread("Hello", "Hola de nuevo", "Spanish")  # repeat of the first

    labels = [t["label"] for t in ui_app.recent_threads]
    assert labels == ["Hello", "Goodbye"]
    assert ui_app.recent_threads[0]["translated"] == "Hola de nuevo"


def test_record_thread_keeps_distinct_languages_for_the_same_text_separate(monkeypatch):
    ui_app = _build_mobile_ui()
    _use_fake_thread_storage(monkeypatch, ui_app)
    ui_app._record_chat_thread("Hello", "Hola", "Spanish")
    ui_app._record_chat_thread("Hello", "Bonjour", "French")

    assert len(ui_app.recent_threads) == 2


def test_recent_threads_degrades_to_empty_outside_a_request_context():
    """No real request/session -> recent_threads must not raise; Recent
    Threads silently not recording is fine, breaking translation is not."""
    ui_app = _build_mobile_ui()

    ui_app._record_chat_thread("Hello", "Hola", "Spanish")

    assert ui_app.recent_threads == []


def test_delete_thread_removes_only_the_matching_entry(monkeypatch):
    ui_app = _build_mobile_ui()
    _use_fake_thread_storage(monkeypatch, ui_app)
    ui_app._record_chat_thread("Hello", "Hola", "Spanish")
    ui_app._record_chat_thread("Goodbye", "Adiós", "Spanish")
    keep_id = ui_app.recent_threads[0]["id"]
    delete_id = ui_app.recent_threads[1]["id"]
    ui_app.drawer = None  # show_document_list() no-ops without a real drawer

    ui_app._delete_thread(delete_id)

    remaining_ids = [t["id"] for t in ui_app.recent_threads]
    assert remaining_ids == [keep_id]


def _build_ui_with_backend(backend) -> TranslationUI:
    ui_app = TranslationUI(backend=backend)
    ui_app.upload_container = DummyContainer()
    ui_app.progress_container = DummyContainer()
    ui_app.result_container = DummyContainer()
    ui_app.stats_container = DummyContainer()
    return ui_app


def test_two_clients_sharing_one_backend_have_independent_document_run_state(monkeypatch):
    """The real production topology since start_ui()'s per-client fix: one
    shared TranslationBackend, many TranslationUI instances. Each instance's
    document_run_state must be its own object, never the backend's ambient
    ._active_run_state (which get_job_result() reassigns on every client's
    job completion)."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    shared_backend = TranslationBackend()

    ui_a = _build_ui_with_backend(shared_backend)
    ui_b = _build_ui_with_backend(shared_backend)

    assert ui_a.backend is ui_b.backend
    assert ui_a.document_run_state is not ui_b.document_run_state

    ui_a.document_run_state.segment_map["seg-a"] = {"original": "A", "translated": "A-translated"}
    ui_b.document_run_state.segment_map["seg-b"] = {"original": "B", "translated": "B-translated"}

    assert "seg-b" not in ui_a.document_run_state.segment_map
    assert "seg-a" not in ui_b.document_run_state.segment_map


def test_refresh_upload_ui_does_not_clear_another_clients_segments(monkeypatch):
    """The exact bug found live (2026-07-06): User B calling refresh_upload_ui
    (e.g. uploading a new file) used to call self.backend.segment_map.clear(),
    wiping the SHARED ambient state and breaking User A's still-open segment
    editor with 'Segment not found'."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    shared_backend = TranslationBackend()

    ui_a = _build_ui_with_backend(shared_backend)
    ui_b = _build_ui_with_backend(shared_backend)
    ui_a.document_run_state.segment_map["seg-a"] = {"original": "A", "translated": "A-translated"}
    # Isolate the assertion to refresh_upload_ui's document_run_state reset -
    # full rendering needs real NiceGUI containers, out of scope here.
    ui_b.render_unified_workspace = lambda: None
    ui_b.show_document_list = lambda: None

    ui_b.refresh_upload_ui()

    assert "seg-a" in ui_a.document_run_state.segment_map, "User B's refresh must not touch User A's segments"


def test_retranslate_segment_callback_reads_its_own_document_run_state(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    shared_backend = TranslationBackend()
    ui_a = _build_ui_with_backend(shared_backend)
    ui_b = _build_ui_with_backend(shared_backend)
    ui_a.document_run_state.segment_map["seg-a"] = {"original": "Hello", "translated": "stale"}
    ui_a.current_target_language = "Spanish"
    monkeypatch.setattr(shared_backend, "translate_text", lambda *a, **k: "Hola")
    monkeypatch.setattr(shared_backend, "update_segment", lambda *a, **k: "Hola")

    # B racing in with an unrelated fresh run_state must not affect A's lookup.
    ui_b.document_run_state = TranslationRunState()

    textarea = DummyInput()
    notified = []
    import TranslationUI as translation_ui_module
    monkeypatch.setattr(translation_ui_module.ui, "notify", lambda msg, **k: notified.append(msg))

    ui_a.retranslate_segment_callback("seg-a", textarea)

    assert textarea.value == "Hola"
    assert not any("not found" in str(m).lower() for m in notified)


def test_set_translate_button_busy_toggles_enabled_state():
    ui_app = _build_mobile_ui()
    ui_app.translate_button = DummyButton()

    ui_app._set_translate_button_busy(True)
    assert ui_app.translate_button.enabled is False

    ui_app._set_translate_button_busy(False)
    assert ui_app.translate_button.enabled is True


def test_set_translate_button_busy_is_a_noop_before_first_render():
    ui_app = _build_mobile_ui()
    assert ui_app.translate_button is None

    ui_app._set_translate_button_busy(True)  # must not raise


def test_mobile_voice_translation_failure_offers_retry_via_start_mobile_translation():
    ui_app = _build_mobile_ui()
    captured = {}
    ui_app.show_error = lambda error, retry=None: captured.update(error=error, retry=retry)
    ui_app.backend.translate_text = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))

    ui_app._run_mobile_voice_translation("hello", "German", DummyProgress(), DummyLabel())

    assert "boom" in str(captured["error"])
    assert captured["retry"] == ui_app.start_mobile_translation


def test_mobile_image_translation_failure_offers_retry_via_start_mobile_translation():
    ui_app = _build_mobile_ui()
    ui_app.image_upload_bytes = b"fake-bytes"
    ui_app.image_upload_name = "sign.png"
    captured = {}
    ui_app.show_error = lambda error, retry=None: captured.update(error=error, retry=retry)
    ui_app.backend.translate_image_text_blocks = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("ocr failed"))

    ui_app._run_mobile_image_translation("French", DummyProgress(), DummyLabel())

    assert "ocr failed" in str(captured["error"])
    assert captured["retry"] == ui_app.start_mobile_translation


def test_mobile_image_translation_success_updates_progress_and_renders_result():
    ui_app = _build_mobile_ui()
    ui_app.image_upload_bytes = b"fake-bytes"
    ui_app.image_upload_name = "sign.png"
    ui_app.backend.translate_image_text_blocks = lambda *a, **k: {"translated_blocks": [], "confidence_metadata": {}}
    rendered = []
    ui_app.show_mobile_image_result = lambda language: rendered.append(language)
    progress, label = DummyProgress(), DummyLabel()

    ui_app._run_mobile_image_translation("French", progress, label)

    assert progress.values == [40, 100]
    assert label.text == "Translation complete."
    assert rendered == ["French"]


def test_mode_switch_syncs_mobile_and_unified_mode():
    ui_app = _build_mobile_ui()
    refresh_calls = {"count": 0}
    ui_app.refresh_upload_ui = lambda: refresh_calls.__setitem__("count", refresh_calls["count"] + 1)

    ui_app.set_mobile_input_mode("Image/Camera")

    assert ui_app.mobile_input_mode == "Image/Camera"
    assert ui_app.input_mode == "Image/Camera"
    assert refresh_calls["count"] == 1


def test_desktop_voice_js_uses_desktop_voice_controls_and_voiceux_path():
    source = Path("passage/ui/voice_page.py").read_text(encoding="utf-8")

    assert "document.getElementById('status_label')" not in source
    assert "document.getElementById('debug_info')" not in source
    assert "document.getElementById('start_btn')" not in source
    assert "document.getElementById('stop_btn')" not in source
    assert "const DESKTOP_SCOPE = 'desktop_voice';" in source
    assert "window.voiceUx.setStatus(DESKTOP_SCOPE, msg);" in source
    assert "window.voiceUx.setDebug(DESKTOP_SCOPE, msg);" in source
    assert "window.voiceUx.setRecordingButtons(DESKTOP_SCOPE, recording);" in source
    assert "document.getElementById('desktop_voice_start_recording')" in source
    assert "document.getElementById('desktop_voice_stop_recording')" in source


def test_transcript_fallback_js_handler_is_defined_and_exported():
    source = Path("passage/ui/voice_page.py").read_text(encoding="utf-8")

    assert "async function translateTranscriptFallback()" in source
    assert "window.translateTranscriptFallback = translateTranscriptFallback;" in source
    assert "fetch('/api/text_translate_stream'" in source


def test_voice_ui_decodes_percent_encoded_translation_headers():
    source = Path("passage/ui/voice_page.py").read_text(encoding="utf-8")

    assert "decodeURIComponent(value)" in source
    assert "const transHeader = resp.headers.get('X-Translated-Text')" in source


def test_workspace_text_mode_js_has_debounce_and_auto_translation_trigger():
    source = Path("TranslationUI.py").read_text(encoding="utf-8")

    assert "const DEBOUNCE_MS = 350;" in source
    # Bindings are delegated on `document` so they survive workspace re-renders.
    assert "document.addEventListener('input', (e) => {" in source
    assert "scheduleDebouncedTranslation();" in source
    assert "window.setTimeout(requestTranslation, DEBOUNCE_MS);" in source
    assert "fetch('/api/text_translate'" in source


def test_workspace_text_js_is_injected_once_at_page_build_not_on_rerender():
    """Re-injecting on refresh_upload_ui crashed with 'parent slot deleted'
    when triggered from an element inside the cleared container (swap ⇄)."""
    source = Path("TranslationUI.py").read_text(encoding="utf-8")

    injection_calls = source.count("self._inject_workspace_text_live_translation_js()")
    assert injection_calls == 1
    main_page_body = source.split("def main_page(self")[1].split("def _render_mode_tabs")[0]
    assert "_inject_workspace_text_live_translation_js" in main_page_body


def _make_upload_event(name: str, data: bytes):
    class FakeFile:
        def __init__(self):
            self.name = name

        async def read(self):
            return data

    class FakeEvent:
        file = FakeFile()

    return FakeEvent()


def test_document_upload_reads_nicegui_3x_file_payload(monkeypatch):
    import asyncio

    ui_app = _build_mobile_ui()
    monkeypatch.setattr("TranslationUI.ui.notify", lambda *a, **k: None)
    ui_app.refresh_upload_ui = lambda: None

    asyncio.run(ui_app.handle_mobile_upload(_make_upload_event("report.pdf", b"%PDF-1.7")))

    assert ui_app.uploaded_file_name == "report.pdf"
    assert ui_app.uploaded_file_extension == "pdf"
    assert ui_app.uploaded_file.getvalue() == b"%PDF-1.7"


def test_image_upload_reads_nicegui_3x_file_payload(monkeypatch):
    import asyncio

    ui_app = _build_mobile_ui()
    monkeypatch.setattr("TranslationUI.ui.notify", lambda *a, **k: None)
    ui_app.refresh_upload_ui = lambda: None

    asyncio.run(ui_app.handle_mobile_image_upload(_make_upload_event("sign.png", b"\x89PNG")))

    assert ui_app.image_upload_name == "sign.png"
    assert ui_app.image_upload_bytes == b"\x89PNG"


def test_workspace_text_mode_js_discards_stale_responses():
    source = Path("TranslationUI.py").read_text(encoding="utf-8")

    assert "let activeRequestToken = 0;" in source
    assert "const token = ++activeRequestToken;" in source
    assert "if (token !== activeRequestToken) return;" in source
    assert "stateLabels = { READY: 'Ready', TRANSLATING: 'Translating…', UPDATED: 'Updated', ERROR: 'Error' }" in source
