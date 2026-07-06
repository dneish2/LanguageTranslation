import sys
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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


def test_mode_switch_syncs_mobile_and_unified_mode():
    ui_app = _build_mobile_ui()
    refresh_calls = {"count": 0}
    ui_app.refresh_upload_ui = lambda: refresh_calls.__setitem__("count", refresh_calls["count"] + 1)

    ui_app.set_mobile_input_mode("Image/Camera")

    assert ui_app.mobile_input_mode == "Image/Camera"
    assert ui_app.input_mode == "Image/Camera"
    assert refresh_calls["count"] == 1


def test_desktop_voice_js_uses_desktop_voice_controls_and_voiceux_path():
    source = Path("TranslationUI.py").read_text(encoding="utf-8")

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
    source = Path("TranslationUI.py").read_text(encoding="utf-8")

    assert "async function translateTranscriptFallback()" in source
    assert "window.translateTranscriptFallback = translateTranscriptFallback;" in source
    assert "fetch('/api/text_translate_stream'" in source


def test_voice_ui_decodes_percent_encoded_translation_headers():
    source = Path("TranslationUI.py").read_text(encoding="utf-8")

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
    main_page_body = source.split("def main_page(self):")[1].split("def _render_mode_tabs")[0]
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
