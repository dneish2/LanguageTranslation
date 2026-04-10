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
    """Primary mobile happy path: upload + language kicks off translation."""
    ui_app = _build_mobile_ui()
    ui_app.mobile_input_mode = "upload"
    ui_app.mobile_target_input = DummyInput("Spanish")
    ui_app.uploaded_file = BytesIO(b"doc")

    called = {}

    def fake_handle_translation(language, *_args, **_kwargs):
        called["language"] = language

    ui_app.handle_translation = fake_handle_translation
    ui_app.start_mobile_translation()

    assert called["language"] == "Spanish"


def test_mobile_upload_error_path_requires_file():
    """Primary mobile error path: translate with no file shows guidance."""
    ui_app = _build_mobile_ui()
    ui_app.mobile_input_mode = "upload"
    ui_app.mobile_target_input = DummyInput("French")
    ui_app.uploaded_file = None

    errors = []
    ui_app.show_error = lambda message: errors.append(str(message))

    ui_app.start_mobile_translation()

    assert errors == ["Please upload a file before translating."]


def test_mobile_voice_happy_path_updates_progress_and_result(monkeypatch):
    ui_app = _build_mobile_ui()
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


def test_mobile_voice_error_path_requires_transcript():
    ui_app = _build_mobile_ui()
    ui_app.mobile_input_mode = "voice"
    ui_app.mobile_target_input = DummyInput("German")
    ui_app.mobile_voice_input = DummyInput("   ")

    errors = []
    ui_app.show_error = lambda message: errors.append(str(message))

    ui_app.start_mobile_translation()

    assert errors == ["Please provide voice transcript text before translating."]
