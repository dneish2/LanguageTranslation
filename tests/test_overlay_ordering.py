"""Overlay draw order, pre-transmit image validation, and the upload size gate.

Every assertion here is deterministic: no model is consulted, no network is
touched, and the two model-shaped fakes exist to COUNT which endpoint was used,
which is the whole point of the credential tests.
"""
import json
import sys
import threading
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api_security import MAX_UPLOAD_BYTES  # noqa: E402
from image_compositor import ImageCompositor, OverlayStyle  # noqa: E402
from passage import provider_profiles as pp  # noqa: E402


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _blank_png(width: int = 400, height: int = 200) -> bytes:
    buf = BytesIO()
    Image.new("RGB", (width, height), (255, 255, 255)).save(buf, format="PNG")
    return buf.getvalue()


def _ink_pixels(png_bytes: bytes) -> set[tuple[int, int]]:
    """Coordinates of every pixel that is not background white.

    Both the canvas and the cover are pure white and the text is near-black, so
    "ink" is exactly "a glyph was painted here and nothing painted over it".
    """
    img = Image.open(BytesIO(png_bytes)).convert("RGB")
    pixels = img.load()
    return {
        (x, y)
        for y in range(img.height)
        for x in range(img.width)
        if sum(pixels[x, y]) < 3 * 200
    }


# Two boxes that OVERLAP vertically — region 2's cover band runs straight
# through the bottom of region 1's box. This is the shape a real menu produces
# (padded covers on closely spaced lines) and the shape the bug needs.
REGION_1 = {"bbox": (10, 40, 380, 120), "translated": "primero segundo tercero",
            "original": "one two three", "direction": "ltr"}
REGION_2 = {"bbox": (10, 100, 380, 170), "translated": "cuarto quinto sexto",
            "original": "four five six", "direction": "ltr"}


def _compose(regions, **kwargs) -> bytes:
    # The production compositor with production defaults. Constructed normally;
    # nothing is hand-assigned onto it.
    return ImageCompositor(OverlayStyle()).compose(_blank_png(), list(regions), **kwargs)


# --------------------------------------------------------------------------
# C3 — draw ordering
# --------------------------------------------------------------------------

def test_later_cover_does_not_erase_earlier_text():
    """No pixel of region 1's translation may go missing when region 2 joins.

    Under the one-pass loop the second region's white cover was painted after
    the first region's glyphs, deleting ~39% of them. Set containment is the
    exact form of that claim, so this fails the moment the order regresses.
    """
    alone = _ink_pixels(_compose([REGION_1]))
    together = _ink_pixels(_compose([REGION_1, REGION_2]))

    assert alone, "region 1 painted no text at all; the fixture is broken"
    erased = alone - together
    assert not erased, (
        f"{len(erased)} of region 1's {len(alone)} ink pixels were erased by a "
        f"later region's cover (worst y={max(y for _x, y in erased)})"
    )


def test_earlier_cover_does_not_erase_later_text():
    """Symmetric: reversing the input order must not move the damage."""
    alone = _ink_pixels(_compose([REGION_2]))
    together = _ink_pixels(_compose([REGION_1, REGION_2]))

    assert alone
    assert not (alone - together)


def test_render_is_independent_of_region_order():
    """[r1, r2] and [r2, r1] must produce identical pixels.

    With covers and text interleaved they could not: whichever region came last
    won the contested band. Order-independence is only achievable once every
    cover is down before any glyph is.
    """
    forward = _ink_pixels(_compose([REGION_1, REGION_2]))
    reverse = _ink_pixels(_compose([REGION_2, REGION_1]))
    assert forward == reverse


def test_covers_still_hide_the_original_ink():
    """Guard against 'fixing' the order by not covering at all."""
    base = Image.new("RGB", (400, 200), (255, 255, 255))
    from PIL import ImageDraw
    ImageDraw.Draw(base).rectangle((20, 45, 200, 110), fill=(0, 0, 0))
    buf = BytesIO()
    base.save(buf, format="PNG")

    out = ImageCompositor(OverlayStyle()).compose(buf.getvalue(), [REGION_1])
    covered = _ink_pixels(out)
    # The black slab sat inside region 1's bbox; after covering, the only ink
    # left in that band belongs to the overlay text, which is far sparser.
    slab = {(x, y) for y in range(45, 111) for x in range(20, 201)}
    assert len(covered & slab) < 0.5 * len(slab)


def test_show_original_mode_paints_no_cover():
    """show_original must leave the photo intact; the two-pass split keeps that."""
    base = Image.new("RGB", (400, 200), (0, 0, 0))
    buf = BytesIO()
    base.save(buf, format="PNG")
    out = _ink_pixels(
        ImageCompositor(OverlayStyle()).compose(buf.getvalue(), [REGION_1, REGION_2],
                                                show_original=True))
    # A cover would have punched white holes into a fully black canvas.
    assert len(out) > 0.9 * 400 * 200


# --------------------------------------------------------------------------
# C4 — validate before transmitting  /  C1 — which endpoint served the request
# --------------------------------------------------------------------------

class _CountingProvider:
    """Records every outbound chat call. The count IS the assertion."""

    def __init__(self, content: str, name: str):
        self.calls: list[dict] = []
        self.name = name
        provider = self

        class _Completions:
            def create(self, **kwargs):
                provider.calls.append(kwargs)
                return _Resp(content)

        class _Chat:
            completions = _Completions()

        class _Client:
            chat = _Chat()

        self.client = _Client()

    def create_chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        return _Resp("translated")


class _Msg:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Msg(content)


class _Resp:
    def __init__(self, content):
        self.choices = [_Choice(content)]


_OCR_JSON = json.dumps({"recognized_blocks": [
    {"text": "hola", "confidence": 0.95, "bbox": [0.1, 0.1, 0.9, 0.3]},
]})


def _backend(monkeypatch, content=_OCR_JSON):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    from TranslationUI import TranslationUI
    ui_app = TranslationUI()
    hosted = _CountingProvider(content, "passage-hosted")
    ui_app.backend.provider = hosted
    monkeypatch.setattr(ui_app.backend, "translate_text", lambda text, lang: f"{lang}:{text}")
    monkeypatch.setattr(ui_app.backend, "_translate_blocks_together",
                        lambda blocks, lang: [b.update({"translated_text": f"{lang}:{b['source_text']}"})
                                              for b in blocks])
    return ui_app, hosted


def test_non_image_named_png_is_rejected_with_zero_outbound_calls(monkeypatch):
    """A text file called fake.png must never leave the machine.

    The extension check alone let it through: it was base64'd, sent to the
    hosted vision model, and only failed afterwards. Asserting the error is not
    enough — the defect is the transmission, so the call count is the test.
    """
    ui_app, hosted = _backend(monkeypatch)

    with pytest.raises(ValueError) as err:
        ui_app.backend.translate_image_text_blocks(
            b"this is plainly not an image, it is a note to my accountant",
            "fake.png", "Spanish")

    assert "isn't a readable image" in str(err.value)
    assert hosted.calls == [], (
        f"{len(hosted.calls)} outbound call(s) were made with a non-image file")


def test_mislabelled_real_image_is_rejected_with_zero_outbound_calls(monkeypatch):
    """A genuine PNG named .jpg is still a lie about the bytes; catch it locally."""
    ui_app, hosted = _backend(monkeypatch)

    with pytest.raises(ValueError):
        ui_app.backend.translate_image_text_blocks(_blank_png(), "photo.jpg", "Spanish")

    assert hosted.calls == []


def test_real_png_does_reach_the_provider(monkeypatch):
    """Proves the counter above can count — otherwise those tests are vacuous."""
    ui_app, hosted = _backend(monkeypatch)

    result = ui_app.backend.translate_image_text_blocks(_blank_png(), "photo.png", "Spanish")

    assert len(hosted.calls) == 1
    assert result["translated_blocks"][0]["translated_text"] == "Spanish:hola"


def test_image_work_runs_on_the_session_profile_inside_the_worker_thread(monkeypatch):
    """C1: a BYO user's photo must go to THEIR endpoint, not Passage's key.

    ContextVars do not cross into a bare threading.Thread, so the session
    profile could never apply to image work. This runs the real backend call on
    a real second thread through the real ``using_profile`` contextmanager and
    asserts WHICH provider served it: the BYO one got the call, Passage's
    hosted provider got zero.
    """
    ui_app, hosted = _backend(monkeypatch)
    byo = _CountingProvider(_OCR_JSON, "byo")
    profile = pp.ProviderProfile(label="Mine", kind=pp.KIND_BYO,
                                 base_url="http://127.0.0.1:1/v1", api_key="k", model="m")
    # Only the provider FACTORY is stubbed, so no socket is opened. The
    # ContextVar plumbing under test is entirely real.
    monkeypatch.setattr(ui_app.backend, "provider_for_profile", lambda p: byo)

    captured = {}

    def worker():
        with ui_app.backend.using_profile(profile):
            captured["result"] = ui_app.backend.translate_image_text_blocks(
                _blank_png(), "photo.png", "Spanish")

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join(timeout=30)

    assert "result" in captured, "worker thread did not finish"
    assert len(byo.calls) == 1, "the session's own endpoint was not used"
    assert hosted.calls == [], "the photo went to Passage's hosted key instead"


def test_start_mobile_translation_captures_the_profile_before_spawning(monkeypatch):
    """The fix must read the profile on the request thread, not in the worker.

    ``active_profile`` reads ``app.storage.user``, which does not exist inside a
    bare thread — reading it there would silently yield None and fall back to
    Passage's key. So assert the captured value reaches the worker.
    """
    import TranslationUI as ui_module
    ui_app, hosted = _backend(monkeypatch)
    byo = _CountingProvider(_OCR_JSON, "byo")
    profile = pp.ProviderProfile(label="Mine", kind=pp.KIND_BYO,
                                 base_url="http://127.0.0.1:1/v1", api_key="k", model="m")
    monkeypatch.setattr(ui_app.backend, "provider_for_profile", lambda p: byo)
    monkeypatch.setattr(type(ui_app), "active_profile", property(lambda self: profile))

    ui_app.image_upload_bytes = _blank_png()
    ui_app.image_upload_name = "photo.png"
    ui_app.input_mode = "Image/Camera"
    ui_app.current_target_language = "Spanish"
    ui_app.current_source_language = "English"
    monkeypatch.setattr(ui_app, "request_voice_prefetch", lambda *a, **k: None)
    monkeypatch.setattr(ui_app, "_set_translate_button_busy", lambda busy: None)
    monkeypatch.setattr(ui_app, "_render_progress_ui",
                        lambda *a, **k: (_Progress(), _Label()))
    monkeypatch.setattr(ui_app, "show_mobile_image_result", lambda language: None)
    monkeypatch.setattr(ui_app, "show_error", lambda *a, **k: pytest.fail(f"show_error: {a}"))

    started: list[threading.Thread] = []

    class _Thread(threading.Thread):
        def start(self):
            started.append(self)
            super().start()

    monkeypatch.setattr(ui_module, "Thread", _Thread)

    ui_app.start_mobile_translation()
    assert started, "no worker thread was spawned"
    started[0].join(timeout=30)

    assert len(byo.calls) == 1, "worker did not run on the session profile"
    assert hosted.calls == [], "worker fell back to Passage's hosted key"


class _Progress:
    def set_value(self, _value):
        pass


class _Label:
    text = ""


# --------------------------------------------------------------------------
# D9a — the size limit, enforced before the read
# --------------------------------------------------------------------------

class _SizedUpload:
    """A FileUpload stand-in whose read() is a tripwire.

    NiceGUI's own FileUpload exposes size() without reading, which is exactly
    what the fix relies on; reading here means the limit was checked too late.
    """

    def __init__(self, name: str, byte_count: int):
        self.name = name
        self.content_type = "application/octet-stream"
        self._size = byte_count
        self.read_calls = 0

    def size(self) -> int:
        return self._size

    async def read(self) -> bytes:
        self.read_calls += 1
        return b"x" * self._size


class _Event:
    def __init__(self, file):
        self.file = file


def _run(coro):
    import asyncio
    return asyncio.run(coro)


def _silence_notify(monkeypatch, sink: list):
    import TranslationUI as ui_module
    monkeypatch.setattr(ui_module.ui, "notify",
                        lambda message, **kwargs: sink.append((message, kwargs)))


def test_oversize_image_upload_is_refused_before_the_file_is_read(monkeypatch):
    ui_app, _hosted = _backend(monkeypatch)
    notes: list = []
    _silence_notify(monkeypatch, notes)
    upload = _SizedUpload("huge.png", MAX_UPLOAD_BYTES + 1)

    _run(ui_app.handle_mobile_image_upload(_Event(upload)))

    assert upload.read_calls == 0, "the oversize file was read into memory anyway"
    assert ui_app.image_upload_bytes is None
    assert notes and "too large" in notes[0][0]


def test_oversize_document_upload_is_refused_before_the_file_is_read(monkeypatch):
    ui_app, _hosted = _backend(monkeypatch)
    notes: list = []
    _silence_notify(monkeypatch, notes)
    upload = _SizedUpload("huge.pdf", MAX_UPLOAD_BYTES + 1)

    _run(ui_app.handle_mobile_upload(_Event(upload)))

    assert upload.read_calls == 0
    assert ui_app.uploaded_file is None
    assert notes and "too large" in notes[0][0]


def test_upload_within_the_limit_is_still_accepted(monkeypatch):
    """Otherwise the two tests above pass by rejecting everything."""
    ui_app, _hosted = _backend(monkeypatch)
    notes: list = []
    _silence_notify(monkeypatch, notes)
    monkeypatch.setattr(ui_app, "refresh_upload_ui", lambda: None)
    upload = _SizedUpload("small.png", 1024)

    _run(ui_app.handle_mobile_image_upload(_Event(upload)))

    assert upload.read_calls == 1
    assert ui_app.image_upload_bytes == b"x" * 1024


def test_limit_boundary_is_inclusive(monkeypatch):
    ui_app, _hosted = _backend(monkeypatch)
    assert ui_app.upload_exceeds_limit(_SizedUpload("a.png", MAX_UPLOAD_BYTES)) is False
    assert ui_app.upload_exceeds_limit(_SizedUpload("a.png", MAX_UPLOAD_BYTES + 1)) is True
