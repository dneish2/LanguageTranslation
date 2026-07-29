"""The WORKSPACE surface — the buttons a user actually presses — must run as
the session that pressed them.

Everything here was reproduced live, in two real Chromium contexts with
distinct session cookies, against fake providers keyed by endpoint, before it
was written down. The previous round of fixes and the previous round of tests
both targeted the ``/api/*`` routes; the workspace Translate button does not go
through those routes, so the leak stayed open on the primary surface while the
suite was green. That is the reason this file exists and the reason every test
below drives ``TranslationUI.start_mobile_translation`` — the button's real
handler — rather than an HTTP route.

What was live:

  D1  ``start_mobile_translation`` handed the work to a bare ``Thread``.
      ContextVars do not cross a thread boundary, so inside the worker there
      was no provider profile (a BYO session's confidential text went to
      PASSAGE'S hosted key — a credential bypass, not a mislabel) and no cache
      scope (``current_cache_scope()`` fell through to the process-wide
      "shared" partition, and the next session to type the same sentence was
      served those exact bytes).
  D3  The image path recorded an unconditional literal
      ``engine=f"hosted:{VISION_MODEL}"``, so a BYO photo that went to the
      user's own endpoint was booked as Passage's vision model, metered, and
      described to the user as having gone somewhere it did not.
  D5  One Translate click on the Text tab produced a metered row under surface
      "voice", on top of the keystroke preview that had already translated the
      identical bytes — billed twice, filed under a microphone never opened.

Two rules this file holds itself to, because breaking either is how the last
round passed while the bug shipped:

* ASSERT WHICH PATH RAN. Every fake endpoint counts its own calls and stamps
  its own identity into the bytes it returns. A test fails if another engine
  served the request, and it fails if nothing ran at all. No test asserts only
  that a call came back.
* PRODUCTION CONSTRUCTION. ``TranslationBackend()`` and ``TranslationUI()``
  run their real constructors; the OpenAI client is faked at
  ``openai.OpenAI``, i.e. below everything under test. Nothing is built with
  ``__new__`` and no attribute is hand-assigned onto the backend.

The one seam: ``app.storage.user`` is NiceGUI's per-session-cookie store and
does not exist outside a request. It is replaced here with a plain dict per
simulated session — which is exactly what makes "two sessions" meaningful —
and ``active_profile`` / ``session_cache_scope`` / ``engine_runs`` /
``usage_store`` then run their real production code on top of it.
"""

import asyncio
import json
import sys
import threading
import types
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import openai

import TranslationBackend as tb
import TranslationUI as ui_module
from TranslationBackend import TEXT_MODEL, VISION_MODEL, TranslationBackend
from TranslationUI import TranslationUI
from passage import policy
from passage import provider_profiles as pp


# ─────────────────────── endpoint doubles, keyed by URL ────────────────── #


class Endpoints:
    """Fake OpenAI-compatible endpoints keyed by base_url (None = Passage's
    hosted key).

    "Where did this text go" is an observable fact here: each endpoint counts
    its own calls and names itself in the bytes it returns, so an assertion can
    distinguish "the hosted model answered", "the user's own endpoint
    answered" and "nothing ran".
    """

    def __init__(self) -> None:
        self.names: dict[object, str] = {}
        self.calls: dict[object, int] = {}
        self._lock = threading.Lock()

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
        with self._endpoints._lock:
            self._endpoints.calls[key] = self._endpoints.calls.get(key, 0) + 1
        name = self._endpoints.names[key]

        messages = kwargs.get("messages") or []
        body = json.dumps(messages)
        if "image_url" in body:  # the OCR read of the photo
            content = json.dumps({"recognized_blocks": [
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
    """Stands in for ``openai.OpenAI``. The provider objects above it are still
    built by production code."""

    endpoints: Endpoints

    def __init__(self, **kwargs):
        self.api_key = kwargs.get("api_key")
        self.base_url = kwargs.get("base_url")
        self.chat = types.SimpleNamespace(
            completions=_FakeCompletions(type(self).endpoints, self.base_url)
        )


HOSTED = None  # base_url of Passage's own hosted key
ENDPOINT_A = "http://user-a.private.example/v1"
ENDPOINT_B = "http://user-b.private.example/v1"


@pytest.fixture
def endpoints(monkeypatch):
    registry = Endpoints()
    monkeypatch.setattr(_FakeOpenAI, "endpoints", registry, raising=False)
    monkeypatch.setattr(openai, "OpenAI", _FakeOpenAI)
    registry.register(HOSTED, "PASSAGE-HOSTED")
    registry.register(ENDPOINT_A, "PRIVATE-A")
    registry.register(ENDPOINT_B, "PRIVATE-B")
    return registry


@pytest.fixture
def backend(monkeypatch, endpoints):
    monkeypatch.setenv("OPENAI_API_KEY", "passage-hosted-key")
    monkeypatch.setenv("TRANSLATION_PROVIDER", "openai")
    # No local model may answer: these tests are about WHICH remote endpoint
    # received the bytes, and a machine that happens to run Ollama would change
    # the path under test.
    monkeypatch.setattr(tb, "LIVE_LOCAL_ENABLED", False)
    return TranslationBackend()


# ───────────────────────── the session substrate ──────────────────────── #


class _FakeStorage:
    """NiceGUI's ``app.storage``, pointed at whichever session is 'current'.

    A real deployment keys this by signed session cookie; two browser contexts
    with different cookies get different dicts. That is the only thing being
    simulated. ``TranslationUI.active_profile``, ``session_cache_scope``,
    ``engine_runs`` and ``usage_store`` all run unmodified on top of it.
    """

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
    return fake_app.storage


class JoinableThread(threading.Thread):
    """A real thread — the context boundary is the whole point — that the test
    can wait on. Substituted for ``threading.Thread`` as imported into
    TranslationUI, so ``start_mobile_translation`` still spawns a genuine
    worker and a fix that only works because the test ran things inline would
    not pass."""

    spawned: list["JoinableThread"] = []

    def start(self):
        type(self).spawned.append(self)
        super().start()


@pytest.fixture
def threads(monkeypatch):
    JoinableThread.spawned = []
    monkeypatch.setattr(ui_module, "Thread", JoinableThread)

    def drain(timeout: float = 30.0):
        for thread in list(JoinableThread.spawned):
            thread.join(timeout)
            assert not thread.is_alive(), "workspace worker thread never finished"
        JoinableThread.spawned = []

    return drain


class _Progress:
    def set_value(self, value):
        return None


class _Label:
    def __init__(self):
        self.text = ""


class _Input:
    def __init__(self, value=""):
        self.value = value


class Session:
    """One browser context: its own storage dict, its own TranslationUI."""

    def __init__(self, ui_app: TranslationUI, store: dict, storage: _FakeStorage):
        self.ui = ui_app
        self.store = store
        self._storage = storage
        self.errors: list = []
        self.results: list = []

    def activate(self):
        self._storage.current = self.store

    @property
    def runs(self) -> list:
        return self.store.get("engine_runs", [])

    @property
    def usage(self) -> dict:
        return self.store.get("usage", {})

    def rows(self, surface: str) -> list:
        return [r for r in self.runs if r.get("surface") == surface]


def _make_session(monkeypatch, backend, storage, *, scope: str, profile=None) -> Session:
    """Build a session the production way and give it a cookie-scoped store.

    ``cache_scope`` is pre-pinned exactly as the real page request pins it (see
    ``TranslationUI.session_cache_scope``): resolved once where the browser
    cookie is readable, then read by every later handler and worker.
    """
    store: dict = {"cache_scope": scope}
    if profile is not None:
        store["provider_profile"] = pp.asdict(profile)

    storage.current = store
    ui_app = TranslationUI(backend=backend)
    session = Session(ui_app, store, storage)

    # Presentation only, and bound PER INSTANCE — patching the class would make
    # two "sessions" share one results list, which is the very confusion these
    # tests exist to detect. The recording, threading, profile and cache-scope
    # code under test is untouched.
    ui_app._render_progress_ui = lambda message, **kw: (_Progress(), _Label())
    ui_app._set_translate_button_busy = lambda busy: None
    ui_app.show_mobile_voice_result = (
        lambda original, translated, language: session.results.append(translated))
    ui_app.show_mobile_image_result = (
        lambda language: session.results.append(ui_app.image_translation_result))
    ui_app.show_error = lambda error, **kw: session.errors.append(error)
    ui_app.request_voice_prefetch = lambda language: None
    return session


def _type_text(session: Session, text: str, language: str = "Spanish") -> None:
    session.ui.mobile_mode = True
    session.ui.input_mode = "Text"
    session.ui.mobile_input_mode = "Text"
    session.ui.text_source_input = _Input(text)
    session.ui.target_language_input = _Input(language)
    session.ui.source_language_input = _Input("English")


def _photo() -> bytes:
    image = Image.new("RGB", (320, 160), "white")
    ImageDraw.Draw(image).text((20, 60), "ENTRANTES", fill="black")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


CONFIDENTIAL = "Board minutes: the acquisition target is Meridian Holdings, price 41 million."


# ───────── D1: the Translate button runs as the session that pressed it ──── #


def test_workspace_translate_honours_the_sessions_byo_endpoint(
    monkeypatch, backend, endpoints, storage, threads
):
    """The credential bypass, on the button.

    A session pointed at its own endpoint presses Translate. The bytes must
    land on THEIR endpoint. Landing on Passage's hosted key is not a labelling
    defect: it is confidential text sent to a provider the user did not choose,
    on a key they did not pay for.
    """
    profile = pp.ProviderProfile(label="Mine", kind=pp.KIND_BYO, base_url=ENDPOINT_A,
                                 api_key="a-key", model="private-a")
    session = _make_session(monkeypatch, backend, storage,
                            scope="browser:aaaa-1111", profile=profile)
    _type_text(session, CONFIDENTIAL)

    session.ui.start_mobile_translation()
    threads()

    assert not session.errors, f"the translation failed: {session.errors}"
    assert endpoints.count(ENDPOINT_A) == 1, (
        "the session's own endpoint never received the text — the worker thread "
        "ran as nobody again")
    assert endpoints.count(HOSTED) == 0, (
        "confidential text was sent to PASSAGE'S hosted key by a BYO session")
    assert session.results == ["<<PRIVATE-A>> translated"], (
        f"the bytes on screen were not produced by the chosen endpoint: {session.results}")


def test_two_workspace_sessions_never_share_a_cache_entry(
    monkeypatch, backend, endpoints, storage, threads
):
    """The leak itself: A translates, B types the identical sentence.

    B must cause a NEW outbound call to B's own destination and must not be
    handed A's bytes. Serving B from A's result is one session reading
    another's confidential text.
    """
    profile_a = pp.ProviderProfile(label="A", kind=pp.KIND_BYO, base_url=ENDPOINT_A,
                                   api_key="a-key", model="private-a")
    session_a = _make_session(monkeypatch, backend, storage,
                              scope="browser:aaaa-1111", profile=profile_a)
    session_b = _make_session(monkeypatch, backend, storage, scope="browser:bbbb-2222")

    session_a.activate()
    _type_text(session_a, CONFIDENTIAL)
    session_a.ui.start_mobile_translation()
    threads()

    calls_after_a = endpoints.total()
    assert endpoints.count(ENDPOINT_A) == 1
    assert calls_after_a == 1

    session_b.activate()
    _type_text(session_b, CONFIDENTIAL)
    session_b.ui.start_mobile_translation()
    threads()

    assert not session_b.errors, f"session B failed: {session_b.errors}"
    assert endpoints.total() == calls_after_a + 1, (
        "session B was served from cache — it read bytes written by session A")
    assert endpoints.count(HOSTED) == 1, "session B's own call did not reach its engine"
    assert session_b.results == ["<<PASSAGE-HOSTED>> translated"]
    assert session_a.results[0] not in session_b.results, (
        "session B's screen shows session A's exact bytes")


def test_workspace_translate_writes_no_shared_cache_partition(
    monkeypatch, backend, endpoints, storage, threads
):
    """The "shared" partition is the pooled bucket every visitor can read.

    The workspace button must never write into it. This asserts the mechanism
    directly, in the cache the backend actually keys, so a regression is caught
    even if the two-session observation above were ever weakened.
    """
    session = _make_session(monkeypatch, backend, storage, scope="browser:cccc-3333")
    _type_text(session, CONFIDENTIAL)

    session.ui.start_mobile_translation()
    threads()

    assert endpoints.total() == 1, "nothing ran; this test would pass vacuously"
    keys = [json.loads(k) if isinstance(k, str) else list(k)
            for k in _cache_keys(backend)]
    assert keys, "the translation was never cached; nothing to assert about"
    scopes = {key[0] for key in keys}
    assert scopes == {"browser:cccc-3333"}, (
        f"the workspace button wrote outside its session's partition: {scopes}")


def _cache_keys(backend) -> list:
    """Every key currently in the backend's translation cache, whatever the
    cache implementation stores them as."""
    return list(backend.translation_cache.keys())


def test_no_bare_thread_crosses_a_context_boundary_unscoped():
    """A source-level backstop for the whole class.

    D1 shipped twice. Every worker handoff in TranslationUI must re-establish
    the submitting session's identity inside the worker; the shape of that is
    ``with self.backend.cache_scope(...)`` / ``using_profile(...)`` near the
    spawn, or ``asyncio.to_thread`` (which copies the context). This test reads
    the file and fails on a bare ``Thread(target=...)`` whose target function
    does neither, which is precisely how the leak was reintroduced.
    """
    source = (ROOT / "TranslationUI.py").read_text(encoding="utf-8").splitlines()
    offenders = []
    for index, line in enumerate(source):
        if "Thread(target=" not in line:
            continue
        window = "\n".join(source[max(0, index - 25):index + 1])
        if "cache_scope(" not in window or "using_profile(" not in window:
            offenders.append(f"line {index + 1}: {line.strip()}")
    assert not offenders, (
        "worker spawned without re-establishing the submitting session's cache "
        "scope and provider profile:\n" + "\n".join(offenders))


# ─────────── D3: the image ledger names the engine that actually ran ────── #


def test_byo_image_run_is_booked_to_the_endpoint_that_read_the_photo(
    monkeypatch, backend, endpoints, storage, threads
):
    """A BYO photo goes to the user's own endpoint, so Passage pays nothing and
    the receipt must say so. It used to record the literal
    ``hosted:{VISION_MODEL}``: metered, "left machine", and naming a model that
    never saw the image."""
    profile = pp.ProviderProfile(label="A", kind=pp.KIND_BYO, base_url=ENDPOINT_A,
                                 api_key="a-key", model="private-a")
    session = _make_session(monkeypatch, backend, storage,
                            scope="browser:aaaa-1111", profile=profile)
    session.ui.mobile_mode = True
    session.ui.input_mode = "Image/Camera"
    session.ui.mobile_input_mode = "Image/Camera"
    session.ui.image_upload_bytes = _photo()
    session.ui.image_upload_name = "menu.png"
    session.ui.target_language_input = _Input("Spanish")
    session.ui.source_language_input = _Input("English")

    session.ui.start_mobile_translation()
    threads()

    assert not session.errors, f"the image translation failed: {session.errors}"
    assert endpoints.count(ENDPOINT_A) >= 1, "the photo never reached the user's endpoint"
    assert endpoints.count(HOSTED) == 0, "the photo was sent to Passage's vision model"

    rows = session.rows(policy.Surface.IMAGE.value)
    assert len(rows) == 1, f"expected one image row, got {rows}"
    row = rows[0]
    assert row["engine"] == profile.describe(), (
        f"the ledger names an engine that did not run: {row['engine']!r}")
    assert row["engine"] != f"hosted:{VISION_MODEL}"
    assert row["metered"] is False, "Passage billed a BYO user for inference it did not buy"
    assert row["left_machine"] is False or "hosted" not in row["engine"]
    assert session.usage.get("metered_chars", 0) == 0, (
        f"metered usage recorded for a BYO image run: {session.usage}")


def test_hosted_image_run_is_still_booked_as_hosted(
    monkeypatch, backend, endpoints, storage, threads
):
    """The negative control for the test above: with no profile the photo does
    go to Passage's vision model, and the receipt must say that too. Without
    this, "never say hosted" would pass D3 while lying in the other direction.
    """
    session = _make_session(monkeypatch, backend, storage, scope="browser:dddd-4444")
    session.ui.mobile_mode = True
    session.ui.input_mode = "Image/Camera"
    session.ui.mobile_input_mode = "Image/Camera"
    session.ui.image_upload_bytes = _photo()
    session.ui.image_upload_name = "menu.png"
    session.ui.target_language_input = _Input("Spanish")
    session.ui.source_language_input = _Input("English")

    session.ui.start_mobile_translation()
    threads()

    assert not session.errors, f"the image translation failed: {session.errors}"
    assert endpoints.count(HOSTED) >= 1
    rows = session.rows(policy.Surface.IMAGE.value)
    assert len(rows) == 1
    assert rows[0]["engine"] == f"hosted:{VISION_MODEL}"
    assert rows[0]["metered"] is True
    assert session.usage.get("metered_chars", 0) > 0


# ──────────── D5: one action, one bill, under the right surface ─────────── #


def test_one_text_click_bills_once_under_the_text_surface(
    monkeypatch, backend, endpoints, storage, threads
):
    """One click on the Text tab is one action.

    It used to be filed under ``voice`` — ``_run_mobile_voice_translation``
    hardcoded ``Surface.VOICE``, so /engines attributed Text-tab work to a
    microphone the user never opened.
    """
    session = _make_session(monkeypatch, backend, storage, scope="browser:eeee-5555")
    _type_text(session, "The quarterly report is ready for review.")

    session.ui.start_mobile_translation()
    threads()

    assert not session.errors, f"the translation failed: {session.errors}"
    assert endpoints.count(HOSTED) == 1, "the hosted model did not serve this click"
    assert len(session.runs) == 1, f"one click produced {len(session.runs)} ledger rows"
    row = session.runs[0]
    assert row["surface"] == policy.Surface.TEXT.value, (
        f"Text-tab work filed under surface {row['surface']!r}")
    assert row["metered"] is True and session.usage.get("metered_runs", 0) == 1


def test_preview_then_click_is_billed_once_not_twice(
    monkeypatch, backend, endpoints, storage, threads
):
    """Typing a sentence and then pressing Translate is ONE action.

    The keystroke preview already translated these exact bytes. The button goes
    through the same door and the same partition, so it is answered from that
    preview's cache: no second outbound call, and no second bill. The row still
    names whoever produced the bytes, because a cache hit is two facts (nothing
    ran for THIS request; someone made these words earlier) and neither may be
    dropped.
    """
    session = _make_session(monkeypatch, backend, storage, scope="browser:ffff-6666")
    text = "The quarterly report is ready for review."
    _type_text(session, text)

    request = types.SimpleNamespace(
        headers={"x-passage-token": session.ui.api_guard.issue_token()},
        client=types.SimpleNamespace(host="127.0.0.1"),
    )
    asyncio.run(session.ui.api_text_translate(request, text=text, language="Spanish"))
    assert endpoints.count(HOSTED) == 1, "the keystroke preview did not actually run"

    session.ui.start_mobile_translation()
    threads()

    assert not session.errors, f"the translation failed: {session.errors}"
    assert endpoints.count(HOSTED) == 1, (
        "the button paid to translate bytes the preview had already translated")
    assert session.usage.get("metered_runs", 0) == 1, (
        f"one user action was billed {session.usage.get('metered_runs')} times: {session.usage}")
    assert session.usage.get("metered_chars", 0) == len(text)

    text_rows_ = session.rows(policy.Surface.TEXT.value)
    assert len(text_rows_) == 1
    row = text_rows_[0]
    assert row["metered"] is False, "a cache hit was billed"
    assert row["left_machine"] is False, (
        "a cache hit — nothing ran, nothing sent — was reported as having left the machine")
    assert row["engine"] == f"hosted:{TEXT_MODEL}", (
        "the cache hit must still name the model that produced the bytes on screen")
    assert not session.rows(policy.Surface.VOICE.value), (
        "Text-tab work was filed under the voice surface")


def test_a_second_sessions_cache_hit_cannot_relabel_this_click(
    monkeypatch, backend, endpoints, storage, threads
):
    """D2 on the workspace surface.

    Cache-hit detection used to sample ``backend.metrics.cache_hits``, a
    process-wide counter every session bumps. Session B getting a keystroke hit
    during session A's in-flight call rewrote A's receipt to "no model ran and
    nothing was sent anywhere" — printed over text that had just been sent out.
    Here B's hits are made to happen around A's click; A's receipt must still
    say it ran.
    """
    session_b = _make_session(monkeypatch, backend, storage, scope="browser:bbbb-2222")
    session_b.activate()
    warm = "B is typing the same sentence over and over."
    for _ in range(2):
        _type_text(session_b, warm)
        session_b.ui.start_mobile_translation()
        threads()
    hits_before = backend.metrics.cache_hits

    session_a = _make_session(monkeypatch, backend, storage, scope="browser:aaaa-1111")
    session_a.activate()

    real_translate_live = backend.translate_live

    def translate_live_with_b_hitting_cache(*args, **kwargs):
        # Exactly what another session's keystroke preview does to the shared
        # metrics object while this call is in flight.
        backend.metrics.record_cache_hit()
        result = real_translate_live(*args, **kwargs)
        backend.metrics.record_cache_hit()
        return result

    monkeypatch.setattr(backend, "translate_live", translate_live_with_b_hitting_cache)

    calls_before = endpoints.total()
    _type_text(session_a, "A is translating something for the first time.")
    session_a.ui.start_mobile_translation()
    threads()

    assert backend.metrics.cache_hits > hits_before, "the interference never happened"
    assert endpoints.total() == calls_before + 1, "A's call did not actually go out"
    row = session_a.runs[-1]
    assert row["engine"] == f"hosted:{TEXT_MODEL}", (
        f"another session's cache hit relabelled this call: {row['engine']!r}")
    assert row["metered"] is True and row["left_machine"] is True, (
        "a call that demonstrably went to a hosted endpoint was reported as "
        "'no model ran and nothing was sent anywhere'")
