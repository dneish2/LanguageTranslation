"""The two surfaces whose receipts were still wrong: recorded speech and the
document job. Plus the image API route, which returned 500 for every image it
successfully translated — and billed for it.

All four defects below were reproduced live, in real browser/HTTP sessions
against fake providers keyed by endpoint, before being written down:

  R4     Surface.VOICE had no producer left. ``policy.py`` still defined the
         row, ``/engines`` still had a column for it, and ZERO call sites
         emitted it: after a hosted translation ran on Passage's key from a
         recording, ``ledger: null`` and ``usage: null``. /engines could say
         "Nothing translated yet" over speech that had been transcribed,
         translated and billed.
  D-VOICE ``api_voice_translate`` called ``translate_audio`` with no
         ``using_profile``, and ``translate_audio`` takes no profile at all. A
         session with a BYO endpoint stored had its transcript sent to
         PASSAGE'S key — while the /voice paste-a-transcript fallback, on the
         same page, honoured the endpoint. Speaking a sentence and pasting the
         same sentence went to two different companies.
  D-DOC  The document job booked its profile-derived engine label
         unconditionally: the only ledger call site that never learned about
         cache provenance. The same file twice in one session made zero new
         outbound calls and still wrote a second "sent out, metered" row.
  D-IMG  ``/api/image_translate`` returned ``overlay_png`` as raw bytes to
         ``JSONResponse``, so every successful translation came back as a 500
         — after the vision call, with the ledger row already written and
         metered. The user paid for an error.

Rules this file holds itself to:

* ASSERT WHICH PATH RAN. Every fake endpoint counts its own calls and stamps
  its own identity into the bytes it returns, so a test fails if another
  engine served the request and fails if nothing ran at all.
* PRODUCTION CONSTRUCTION. Real ``TranslationBackend()`` / ``TranslationUI()``
  constructors, real routes, real ``_run_translation_job`` worker thread. The
  OpenAI client is faked at ``openai.OpenAI``, below everything under test.
  Nothing is built with ``__new__``; no attribute is hand-assigned onto a
  backend or a provider.

The seams, all presentation or environment, none of them the code under test:
``app.storage.user`` (NiceGUI's cookie store, which does not exist outside a
request), ``ui.timer`` (the document poll clock), the two speech models in
``passage.local_voice``, and ``show_result``.
"""

import asyncio
import json
import sys
import threading
import types
from io import BytesIO
from pathlib import Path
from urllib.parse import unquote

import pytest
from docx import Document

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import openai

import TranslationBackend as tb
import TranslationUI as ui_module
from TranslationBackend import TEXT_MODEL, VISION_MODEL, TranslationBackend
from TranslationUI import TranslationUI
from passage import local_voice, traces
from passage import provider_profiles as pp
from passage.ui import voice_page


HOSTED = None  # base_url of Passage's own hosted key
ENDPOINT_A = "http://user-a.private.example/v1"

TRANSCRIPT = "Where is the train station?"


# ─────────────────────── endpoint doubles, keyed by URL ────────────────── #


class Endpoints:
    """Fake OpenAI-compatible endpoints keyed by base_url (None = Passage's
    hosted key). Each counts its own calls and names itself in the bytes it
    returns, so "where did this text go" is observable from the response."""

    def __init__(self) -> None:
        self.names: dict[object, str] = {}
        self.calls: dict[object, int] = {}
        self._lock = threading.Lock()
        #: The OCR read succeeds but finds no text — a photograph of a blank
        #: wall. The photo still reached the model, which is the whole point of
        #: the tests that set this.
        self.ocr_finds_nothing = False
        #: Raised INSTEAD of connecting, and deliberately not counted: a
        #: refused connection means nothing landed anywhere.
        self.refuse_with: BaseException | None = None

    def register(self, base_url, name: str) -> None:
        self.names[base_url] = name
        self.calls.setdefault(base_url, 0)

    def count(self, base_url) -> int:
        return self.calls.get(base_url, 0)

    def total(self) -> int:
        return sum(self.calls.values())


class _FakeCompletions:
    def __init__(self, endpoints: Endpoints, base_url) -> None:
        self._endpoints = endpoints
        self._base_url = base_url

    def create(self, **kwargs):
        key = self._base_url
        if key not in self._endpoints.names:
            raise AssertionError(f"call to an endpoint no test registered: {key!r}")
        if self._endpoints.refuse_with is not None:
            raise self._endpoints.refuse_with
        with self._endpoints._lock:
            self._endpoints.calls[key] = self._endpoints.calls.get(key, 0) + 1
        name = self._endpoints.names[key]

        body = json.dumps(kwargs.get("messages") or [])
        if "image_url" in body:  # the OCR read of the photo
            content = json.dumps({"recognized_blocks": [] if self._endpoints.ocr_finds_nothing else [
                {"text": "ENTRANTES", "confidence": 0.95, "bbox": [0.1, 0.1, 0.9, 0.3]},
            ]})
        elif '"translations"' in body:  # the batched block translation
            content = json.dumps({"translations": [{"i": 0, "text": f"<<{name}>> STARTERS"}]})
        else:
            content = f"<<{name}>> translated"
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=content))]
        )


class _FakeOpenAI:
    endpoints: Endpoints

    def __init__(self, **kwargs):
        self.api_key = kwargs.get("api_key")
        self.base_url = kwargs.get("base_url")
        self.chat = types.SimpleNamespace(
            completions=_FakeCompletions(type(self).endpoints, self.base_url)
        )


@pytest.fixture
def endpoints(monkeypatch):
    registry = Endpoints()
    monkeypatch.setattr(_FakeOpenAI, "endpoints", registry, raising=False)
    monkeypatch.setattr(openai, "OpenAI", _FakeOpenAI)
    registry.register(HOSTED, "PASSAGE-HOSTED")
    registry.register(ENDPOINT_A, "PRIVATE-A")
    return registry


@pytest.fixture
def backend(monkeypatch, endpoints):
    monkeypatch.setenv("OPENAI_API_KEY", "passage-hosted-key")
    monkeypatch.setenv("TRANSLATION_PROVIDER", "openai")
    # No local text model may answer: these tests are about which REMOTE
    # endpoint received the words, and a machine running Ollama would change
    # the path under test.
    monkeypatch.setattr(tb, "LIVE_LOCAL_ENABLED", False)
    return TranslationBackend()


# ───────────────────────── the session substrate ──────────────────────── #


class _FakeStorage:
    """NiceGUI's ``app.storage``, keyed by session the way a signed cookie
    keys it. ``active_profile`` / ``session_cache_scope`` / ``engine_runs`` /
    ``usage_store`` run their real production code on top of it."""

    def __init__(self) -> None:
        self.current: dict | None = None

    @property
    def user(self) -> dict:
        if self.current is None:
            raise RuntimeError("no session storage outside a request")
        return self.current


@pytest.fixture
def storage(monkeypatch):
    fake_app = types.SimpleNamespace(storage=_FakeStorage())
    monkeypatch.setattr(ui_module, "app", fake_app)
    monkeypatch.setattr(voice_page, "app", fake_app, raising=False)
    return fake_app.storage


class Session:
    """One browser context: its own storage dict, its own TranslationUI."""

    def __init__(self, ui_app: TranslationUI, store: dict):
        self.ui = ui_app
        self.store = store

    @property
    def runs(self) -> list:
        return self.store.get("engine_runs", [])

    @property
    def usage(self) -> dict:
        return self.store.get("usage", {})

    def rows(self, surface: str) -> list:
        return [r for r in self.runs if r.get("surface") == surface]


def _make_session(backend, storage, *, scope: str, profile=None) -> Session:
    store: dict = {"cache_scope": scope}
    if profile is not None:
        store["provider_profile"] = pp.asdict(profile)
    storage.current = store
    return Session(TranslationUI(backend=backend), store)


def _request(ui_app: TranslationUI):
    return types.SimpleNamespace(
        headers={"x-passage-token": ui_app.api_guard.issue_token()},
        client=types.SimpleNamespace(host="127.0.0.1"),
    )


class _Upload:
    """Stands in for Starlette's UploadFile: the route only reads it."""

    def __init__(self, data: bytes, filename: str):
        self._data = data
        self.filename = filename

    async def read(self):
        return self._data


def _byo_profile() -> pp.ProviderProfile:
    return pp.ProviderProfile(
        label="Private A", kind=pp.KIND_BYO, base_url=ENDPOINT_A,
        api_key="sk-user-a-secret", model="private-a", id="probe-a")


# ───────────────────────────── voice fixtures ─────────────────────────── #


@pytest.fixture
def local_speech(monkeypatch):
    """Speech models are not the subject: the transcript's DESTINATION is.

    Both ends are pinned local so that anything hosted showing up in a voice
    test is the translation step — the step that carries the words.
    """
    monkeypatch.setattr(local_voice, "stt_available", lambda: True)
    monkeypatch.setattr(local_voice, "transcribe", lambda audio: TRANSCRIPT)
    monkeypatch.setattr(local_voice, "WHISPER_MODEL", "base")
    monkeypatch.setattr(local_voice, "tts_available", lambda language: True)
    monkeypatch.setattr(local_voice, "synthesize", lambda text, language: b"RIFFfake")


def _speak(session: Session, language: str = "Spanish"):
    return asyncio.run(session.ui.api_voice_translate(
        _request(session.ui), file=_Upload(b"fake-wav-bytes", "speech.wav"),
        language=language))


# ─────────── R4 + D-VOICE: the spoken path, booked and routed ─────────── #


def test_spoken_translation_runs_on_the_sessions_own_endpoint(
    backend, endpoints, storage, local_speech
):
    """D-VOICE. A session with a BYO endpoint stored records a sentence.

    The words must land on THEIR endpoint and Passage's key must receive
    nothing. This is not a labelling defect: it is a private transcript sent
    to a provider the user did not choose, on a key they did not pay for.
    """
    session = _make_session(backend, storage, scope="session:A", profile=_byo_profile())

    response = _speak(session)

    assert response.status_code == 200
    assert endpoints.count(ENDPOINT_A) == 1, "the user's own endpoint never received the transcript"
    assert endpoints.count(HOSTED) == 0, (
        "the spoken transcript was sent to Passage's hosted key while the session "
        "had its own endpoint configured"
    )
    translated = unquote(response.headers["X-Translated-Text"])
    assert "<<PRIVATE-A>>" in translated, f"a different engine produced these bytes: {translated!r}"
    assert "PASSAGE-HOSTED" not in translated


def test_a_recording_produces_a_voice_row_that_names_the_endpoint(
    backend, endpoints, storage, local_speech
):
    """R4 + D-VOICE, on the ledger. The row exists, is filed under VOICE, and
    is not metered because Passage's key did no work."""
    session = _make_session(backend, storage, scope="session:A", profile=_byo_profile())

    _speak(session)

    rows = session.rows("voice")
    assert len(rows) == 1, f"no Surface.VOICE row was written: {session.runs}"
    row = rows[0]
    assert row["engine"] == "byo:private-a", row
    assert row["ran"] == "byo"
    assert row["left_machine"] is True
    assert row["metered"] is False, "a BYO endpoint's own work must not be billed by Passage"
    assert row["chars"] == len(TRANSCRIPT)
    assert session.usage.get("metered_runs", 0) == 0
    assert endpoints.count(ENDPOINT_A) == 1


def test_hosted_voice_work_is_metered_like_every_other_surface(
    backend, endpoints, storage, local_speech
):
    """R4. No profile: the translation runs on Passage's key, so the voice row
    must say so and the meter must move. /engines saying "Nothing translated
    yet" here is the over-claim this test exists to prevent."""
    session = _make_session(backend, storage, scope="session:A")

    _speak(session)

    assert endpoints.count(HOSTED) == 1, "the hosted endpoint did not serve this recording"
    assert endpoints.count(ENDPOINT_A) == 0
    rows = session.rows("voice")
    assert len(rows) == 1, f"no Surface.VOICE row was written: {session.runs}"
    assert rows[0]["engine"] == f"hosted:{TEXT_MODEL}"
    assert rows[0]["metered"] is True
    assert rows[0]["left_machine"] is True
    assert session.usage["metered_runs"] == 1
    assert session.usage["metered_chars"] == len(TRANSCRIPT)


def test_a_hosted_transcription_is_never_booked_as_a_cache_hit(
    monkeypatch, backend, endpoints, storage
):
    """The over-claim that a naive cache fix would introduce here.

    Cache provenance describes the TRANSLATION step only. If the recording
    itself was uploaded to a hosted transcriber, "no model ran and nothing was
    sent anywhere" is false no matter how the words were served. Same audio
    twice: the second translation IS a cache hit, and the row must still say
    the audio left.
    """
    transcriptions: list[bytes] = []
    monkeypatch.setattr(local_voice, "stt_available", lambda: False)
    monkeypatch.setattr(local_voice, "tts_available", lambda language: True)
    monkeypatch.setattr(local_voice, "synthesize", lambda text, language: b"RIFFfake")

    def hosted_transcribe(audio_bytes):
        transcriptions.append(audio_bytes)
        return TRANSCRIPT

    monkeypatch.setattr(backend, "_transcribe_hosted", hosted_transcribe)
    session = _make_session(backend, storage, scope="session:A")

    _speak(session)
    calls_after_first = endpoints.count(HOSTED)
    _speak(session)

    assert len(transcriptions) == 2, "the audio was not uploaded both times"
    assert calls_after_first == 1
    assert endpoints.count(HOSTED) == 1, "sanity: the second translation must be a cache hit"

    rows = session.rows("voice")
    assert len(rows) == 2, session.runs
    second = rows[1]
    assert second["ran"] != "cache", (
        "the recording was uploaded to a hosted transcriber, so this run cannot "
        "claim that nothing was sent anywhere"
    )
    assert second["left_machine"] is True
    assert second["metered"] is True


def test_a_fully_local_repeat_is_booked_as_cache_and_not_billed(
    monkeypatch, backend, endpoints, storage, local_speech
):
    """The mirror image, so the fix above cannot be "always say hosted": with
    a local profile every step stays on this machine, and a repeat really did
    send nothing."""
    profile = pp.local_profile("http://127.0.0.1:11434/v1", "translategemma:4b")
    endpoints.register("http://127.0.0.1:11434/v1", "LOCAL-OLLAMA")
    session = _make_session(backend, storage, scope="session:A", profile=profile)

    _speak(session)
    _speak(session)

    assert endpoints.count("http://127.0.0.1:11434/v1") == 1, (
        "the repeat made a new outbound call; there is no cache hit to observe")
    assert endpoints.count(HOSTED) == 0
    rows = session.rows("voice")
    assert len(rows) == 2, session.runs
    assert rows[1]["ran"] == "cache"
    assert rows[1]["left_machine"] is False
    assert rows[1]["metered"] is False
    assert rows[1]["engine"] == profile.describe(), (
        "a cache row must still name who originally produced the words")


# ───────────────── D-DOC: the document job's ledger row ───────────────── #


def _docx(text: str) -> bytes:
    document = Document()
    document.add_paragraph(text)
    buffer = BytesIO()
    document.save(buffer)
    return buffer.getvalue()


class _Progress:
    def set_value(self, value):
        return None


class _Label:
    def __init__(self):
        self.text = ""


class _Timer:
    """What ``ui.timer`` hands back. The test drives the real ``poll_job``
    through it rather than waiting on NiceGUI's clock."""

    def __init__(self, callback):
        self.callback = callback
        self.active = True


def _run_document_job(monkeypatch, session: Session, payload: bytes, *, language="Spanish"):
    """Drive the production document path: real job thread, real poll_job."""
    ui_app = session.ui
    ui_app.uploaded_file = BytesIO(payload)
    ui_app.uploaded_file_name = "report.docx"
    ui_app.uploaded_file_extension = "docx"
    ui_app.current_target_language = language
    ui_app.drawer = None                      # show_document_list() no-ops
    ui_app.show_result = lambda: None          # presentation only
    ui_app._set_translate_button_busy = lambda busy: None

    timers: list[_Timer] = []
    monkeypatch.setattr(ui_module.ui, "timer",
                        lambda interval, callback, **kw: timers.append(_Timer(callback))
                        or timers[-1])

    ui_app._start_job_and_poll(
        progress_ui=_Progress(), label_ui=_Label(), correlation_id="test-doc",
        processed=False, font_size=None, autofit=False, target_language=language,
        complete_event="ui.translation_complete",
        failed_event="ui.translation_failed",
        cancelled_event="ui.translation_cancelled",
    )

    poll = timers[-1].callback
    for _ in range(600):
        poll()
        if ui_app.active_job_id is None:
            break
        threading.Event().wait(0.05)
    assert ui_app.active_job_id is None, "the document job never completed"


def test_the_same_document_twice_costs_nothing_the_second_time(
    monkeypatch, backend, endpoints, storage, tmp_path
):
    """D-DOC. Same session, same file. The second run makes no outbound call,
    so it must not be booked as sent and must not move the meter."""
    monkeypatch.setattr(traces, "TRACE_DIR", tmp_path)
    session = _make_session(backend, storage, scope="session:A")
    payload = _docx("The acquisition target is Meridian Holdings.")

    _run_document_job(monkeypatch, session, payload)
    calls_after_first = endpoints.count(HOSTED)
    usage_after_first = dict(session.usage)
    _run_document_job(monkeypatch, session, payload)

    assert calls_after_first >= 1, "the first document never reached an endpoint"
    assert endpoints.count(HOSTED) == calls_after_first, (
        "the repeat made NEW outbound calls; there is no cache hit to observe here")

    rows = session.rows("document")
    assert len(rows) == 2, session.runs
    first, second = rows
    assert first["ran"] == "hosted" and first["metered"] is True
    assert second["ran"] == "cache", (
        "the document job printed 'sent out' over text that never left the machine")
    assert second["left_machine"] is False
    assert second["metered"] is False
    assert second["engine"] == f"hosted:{TEXT_MODEL}", (
        "a cache row must still name who originally produced the words")

    assert session.usage["metered_runs"] == usage_after_first["metered_runs"] == 1
    assert session.usage["metered_chars"] == usage_after_first["metered_chars"]


def test_a_document_that_really_ran_is_still_booked_and_billed(
    monkeypatch, backend, endpoints, storage, tmp_path
):
    """The negative control for the test above: a different document in the
    same session is a real run and must be metered."""
    monkeypatch.setattr(traces, "TRACE_DIR", tmp_path)
    session = _make_session(backend, storage, scope="session:A")

    _run_document_job(monkeypatch, session, _docx("The first quarterly report."))
    _run_document_job(monkeypatch, session, _docx("An entirely different sentence."))

    rows = session.rows("document")
    assert len(rows) == 2, session.runs
    assert [row["ran"] for row in rows] == ["hosted", "hosted"]
    assert all(row["metered"] for row in rows)
    assert session.usage["metered_runs"] == 2
    assert endpoints.count(HOSTED) >= 2


def test_a_document_runs_on_the_sessions_own_endpoint_and_is_booked_there(
    monkeypatch, backend, endpoints, storage, tmp_path
):
    """The doc row must name the endpoint that served it, cache logic or not."""
    monkeypatch.setattr(traces, "TRACE_DIR", tmp_path)
    session = _make_session(backend, storage, scope="session:A", profile=_byo_profile())

    _run_document_job(monkeypatch, session, _docx("Board minutes, confidential."))

    assert endpoints.count(ENDPOINT_A) >= 1, "the document never reached the user's endpoint"
    assert endpoints.count(HOSTED) == 0, "a BYO session's document went to Passage's key"
    rows = session.rows("document")
    assert len(rows) == 1 and rows[0]["engine"] == "byo:private-a"
    assert rows[0]["metered"] is False


# ─────────────── D-IMG: the image API returns what it charged ──────────── #


def _photo() -> bytes:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (320, 160), "white")
    ImageDraw.Draw(image).text((20, 60), "ENTRANTES", fill="black")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _translate_image(session: Session, payload: bytes, filename="menu.png"):
    return asyncio.run(session.ui.api_image_translate(
        _request(session.ui), file=_Upload(payload, filename), language="Spanish"))


def test_the_image_api_returns_a_usable_payload_not_a_500(
    backend, endpoints, storage
):
    """D-IMG. ``overlay_png`` was raw PNG bytes and JSONResponse cannot
    serialise bytes, so a successful vision call reached the caller as
    ``500 {"error": "... Object of type bytes is not JSON serializable"}``
    while the ledger row had already been written and metered."""
    session = _make_session(backend, storage, scope="session:A")

    response = _translate_image(session, _photo())

    assert response.status_code == 200, response.body.decode()
    body = json.loads(response.body.decode())
    assert "error" not in body, body
    blocks = body["translated_blocks"]
    assert blocks and "<<PASSAGE-HOSTED>>" in blocks[0]["translated_text"], (
        f"a different engine produced these blocks: {blocks}")
    overlay = body["overlay_png_base64"]
    assert overlay, "the composed overlay was dropped instead of being encoded"
    import base64
    assert base64.b64decode(overlay)[:4] == b"\x89PNG", "not a decodable PNG"
    assert endpoints.count(HOSTED) >= 2, "expected at least an OCR read and a translation"

    rows = session.rows("image")
    assert len(rows) == 1 and rows[0]["engine"] == f"hosted:{VISION_MODEL}"
    assert rows[0]["metered"] is True


def test_an_image_request_that_errors_is_not_metered(
    backend, endpoints, storage
):
    """D-IMG's other half: the user must not pay for a request that comes back
    as an error. A file that is not a decodable image is rejected before
    anything leaves, and nothing is booked."""
    session = _make_session(backend, storage, scope="session:A")

    response = _translate_image(session, b"plain text pretending to be png\n" * 50)

    assert response.status_code == 400
    assert "error" in json.loads(response.body.decode())
    assert endpoints.total() == 0, "a non-image was sent to a model anyway"
    assert session.rows("image") == [], "an erroring request was written to the ledger"
    assert session.usage.get("metered_runs", 0) == 0, "the user was billed for an error"


def test_the_image_api_honours_the_sessions_own_endpoint(
    backend, endpoints, storage
):
    """The response the caller now actually receives must also be the one the
    ledger describes, on a BYO session."""
    session = _make_session(backend, storage, scope="session:A", profile=_byo_profile())

    response = _translate_image(session, _photo())

    assert response.status_code == 200, response.body.decode()
    body = json.loads(response.body.decode())
    assert "<<PRIVATE-A>>" in body["translated_blocks"][0]["translated_text"]
    assert endpoints.count(HOSTED) == 0, "a BYO session's photograph went to Passage's key"
    rows = session.rows("image")
    assert len(rows) == 1 and rows[0]["engine"] == "byo:private-a"
    assert rows[0]["metered"] is False


# ───────── a delivered call that then failed is still a disclosure ──────── #
#
# DECISIONS.md §8, and the last instance of the defect shape that took six
# review rounds to converge on: the app deriving a privacy claim from state
# read later instead of recording what happened when it happened. Every
# earlier instance was a claim computed too late. This one was a claim never
# computed at all — `_run_recorded`'s `record(...)` sits at the END of the
# function, so an exception from the wrapped call escaped past it and the run
# was never booked.
#
# The repro: photograph something with no readable text. The image is base64'd
# to the hosted vision model, the model answers "nothing here",
# `TranslationBackend` raises `ValueError("No text recognized in image.")`, and
# /engines says "Nothing translated yet." about a photograph that had
# demonstrably been sent to a hosted model on Passage's key.
#
# These run the real route against the fake endpoints above, so the outbound
# call genuinely happens and production's own `_note_engine` writes the
# provenance the fix reads. That matters: an earlier version of this fix
# passed a test that fed it the caller's engine LABEL, and shipped a false
# disclosure for a file that validation rejected before anything left.


def test_a_photo_the_model_read_but_found_no_text_in_is_still_disclosed(
    backend, endpoints, storage
):
    """The §8 repro, end to end through the route that had the bug."""
    endpoints.ocr_finds_nothing = True
    session = _make_session(backend, storage, scope="session:A")

    response = _translate_image(session, _photo())

    # The user still gets the real error, unchanged. Bookkeeping must not
    # replace, swallow or reshape the failure it is recording.
    assert response.status_code == 400
    assert "error" in json.loads(response.body.decode())
    # The photograph really was sent: this is what makes it a disclosure.
    assert endpoints.count(HOSTED) == 1

    rows = session.rows("image")
    assert rows, "a delivered-then-failed vision call left no ledger row"
    assert len(rows) == 1
    assert rows[0]["engine"] == f"hosted:{VISION_MODEL}"   # which path ran
    assert rows[0]["left_machine"] is True                 # the disclosure
    # NOT billed. `delivered=False` preserves the existing rule that an
    # erroring image request is never metered, rather than trading that rule
    # away in exchange for the disclosure.
    assert rows[0]["metered"] is False
    assert session.usage.get("metered_chars", 0) == 0
    # Unknowable, and reported as such rather than guessed: the source text was
    # inside the photo and the model never got as far as reporting it.
    assert rows[0]["chars"] == 0

    from passage import engine_ledger
    summary = engine_ledger.summarise(session.runs)
    assert summary["total_runs"] == 1, "/engines still says nothing was translated"
    assert summary["remote"]["runs"] == 1
    assert summary["metered_runs"] == 0


def test_a_byo_photo_that_fails_is_disclosed_against_the_users_own_endpoint(
    backend, endpoints, storage
):
    """The disclosure has to name the endpoint that actually received the
    photo. A BYO session's failure booked against Passage's hosted key would
    be a worse lie than not booking it at all."""
    endpoints.ocr_finds_nothing = True
    session = _make_session(backend, storage, scope="session:A", profile=_byo_profile())

    _translate_image(session, _photo())

    assert endpoints.count(HOSTED) == 0, "a BYO session's photograph went to Passage's key"
    rows = session.rows("image")
    assert len(rows) == 1 and rows[0]["engine"] == "byo:private-a"
    assert rows[0]["left_machine"] is True and rows[0]["metered"] is False


def test_a_connection_that_never_opened_writes_no_row(
    backend, endpoints, storage
):
    """The false disclosure this must not produce.

    An offline user reading "sent out: 1 run" about text that never left is a
    lie told to precisely the person who chose this app. Note that provenance
    IS written here — `_note_engine` runs before the outbound call, so its
    presence cannot be the only gate — which is exactly why
    `compare.reached_the_engine` is consulted as a second one. Same rule and
    same reason as the Compare page's unreached legs; see DECISIONS.md §9.
    """
    endpoints.refuse_with = ConnectionRefusedError("connection refused")
    session = _make_session(backend, storage, scope="session:A")

    _translate_image(session, _photo())

    assert endpoints.total() == 0, "nothing should have landed"
    assert session.rows("image") == [], "an offline session was told its photo was sent out"


def test_the_sdk_flattened_connection_error_also_writes_no_row(
    backend, endpoints, storage
):
    """The openai SDK collapses transport failures into `APIConnectionError`
    with the literal message "Connection error." — the residual DECISIONS.md §9
    accepts knowingly. It is read as never-sent, deliberately, because the
    offline false positive that choice prevents is both far more common and
    aimed at the user who cares most."""
    endpoints.refuse_with = RuntimeError("Connection error.")
    session = _make_session(backend, storage, scope="session:A")

    _translate_image(session, _photo())

    assert session.rows("image") == []
