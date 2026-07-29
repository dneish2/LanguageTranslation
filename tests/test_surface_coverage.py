"""Every surface must reach the engine ledger — and Compare, the fifth one to
forget, must reach it correctly.

D1, reproduced live against endpoint-keyed fake providers: a session whose only
action was Compare sent a 56-character sentence to Passage's hosted key and
/engines still read "Nothing translated yet." Worse, translating one sentence
locally and then comparing another printed "100% of the 44 characters you
translated stayed on this machine · sent out: 0 runs" immediately after those
56 characters had left. compare_page called compare_translations and recorded
nothing; policy.Surface had no compare member at all.

That is the FIFTH time this defect has been fixed one surface at a time (Text,
Voice, Document, Image, now Compare), because ledger recording is opt-in per
call site and nothing fails when someone forgets. So the first test here is not
about Compare: it enumerates policy.Surface and fails if any member has no
recorder call site IN THE SHIPPED SOURCE. The producer set is scanned out of the
code rather than listed here, because a hand-maintained list is the same
forgettable step one layer up.

The Compare tests below drive the real page: TranslationUI().compare_page()
builds the real elements and the real "Run comparison" click handler is invoked.
Nothing is built with __new__ and no attribute is hand-assigned. The only seams
are NiceGUI's per-request cookie store (which does not exist outside a request)
and the Ollama reachability probe (a network call, faked so the local candidate
is deterministic).
"""

import asyncio
import re
import sys
import threading
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import openai
from nicegui import background_tasks, context, core, ui

import TranslationBackend as tb
import TranslationUI as ui_module
from TranslationBackend import TEXT_MODEL, TranslationBackend
from TranslationUI import TranslationUI
from passage import policy


# ───────────────── the invariant: no surface without a producer ───────────── #

#: A ledger call site looks like `<some recorder>(… surface=policy.Surface.X …)`.
#: Both halves matter: the enum alone appears in retention tables and function
#: defaults, which record nothing.
_RECORDER = re.compile(r"\b(?:recorder|record|doc_recorder|_record_engine_run|_run_recorded)\(")
_SURFACE = re.compile(r"surface=policy\.Surface\.([A-Z_]+)")


def shipped_sources() -> dict[str, str]:
    """The code that actually runs in production. Tests are excluded on
    purpose: a producer that only exists in a test is not a producer."""
    return {
        str(path.relative_to(ROOT)): path.read_text(encoding="utf-8")
        for path in list(ROOT.glob("*.py")) + list((ROOT / "passage").rglob("*.py"))
    }


def producers(sources: dict[str, str]) -> dict[str, str]:
    """Surface name → the file that records it."""
    found: dict[str, str] = {}
    for name, text in sources.items():
        for call in _RECORDER.finditer(text):
            # Bounded to the call itself; a later, unrelated call must not
            # count as this one's producer.
            match = _SURFACE.search(text, call.end(), call.end() + 400)
            if match:
                found.setdefault(match.group(1), name)
    return found


def surfaces_without_a_producer(sources: dict[str, str]) -> list[str]:
    found = producers(sources)
    return sorted(s.name for s in policy.Surface if s.name not in found)


def test_every_surface_has_a_ledger_producer_in_the_shipped_code():
    missing = surfaces_without_a_producer(shipped_sources())
    assert not missing, (
        f"These surfaces are invisible on /engines and unmetered: {', '.join(missing)}. "
        "Text sent from them does not appear in the session ledger, and the privacy "
        "percentage over-claims because their characters are never counted. Fix: at the "
        "call site that performs the work, call the recorder from self._engine_recorder() "
        "with surface=policy.Surface.<NAME> and the engine that actually served the run "
        "(see compare_page for the fan-out case). Do not delete the surface to silence "
        "this — deleting it deletes the receipt, not the request."
    )


def test_the_coverage_check_fails_when_a_producer_is_removed():
    """Mutation check, so the test above cannot pass vacuously.

    The shipped source is read, Compare's recorder call is deleted from the
    copy in memory (nothing on disk is touched), and the check must then name
    COMPARE and say what to do about it.
    """
    sources = shipped_sources()
    ui_source = sources["TranslationUI.py"]
    mutated = re.sub(
        r"\n *recorder\(surface=policy\.Surface\.COMPARE[^)]*\)", "\n            pass",
        ui_source)
    assert mutated != ui_source, "the mutation did not apply; this check proves nothing"

    sources["TranslationUI.py"] = mutated
    assert surfaces_without_a_producer(sources) == ["COMPARE"]


def test_each_surface_has_a_retention_decision():
    """The other half: a new surface must not fall out of policy either."""
    for surface in policy.Surface:
        assert policy.retention_for(surface).reason


# ─────────────────────── Compare, on the production page ──────────────────── #

SENTENCE = "Merger terms for project QZX7 are confidential and final."
HOSTED = None                                  # Passage's own hosted key
LOCAL_URL = tb.OLLAMA_BASE_URL                 # a model on this machine
LOCAL_TAG = "qwen2.5:7b"


class Endpoints:
    """Fake OpenAI-compatible endpoints keyed by base_url, each counting its
    own calls and stamping its identity into the bytes it returns — so "which
    engine served this" is observable rather than assumed."""

    def __init__(self) -> None:
        self.calls: dict[object, int] = {HOSTED: 0, LOCAL_URL: 0}
        self.texts: dict[object, list[str]] = {HOSTED: [], LOCAL_URL: []}
        self.lock = threading.Lock()

    def name(self, base_url) -> str:
        return "PASSAGE-HOSTED" if base_url is HOSTED else "LOCAL-OLLAMA"


class _FakeCompletions:
    def __init__(self, endpoints: Endpoints, base_url) -> None:
        self._endpoints = endpoints
        self._base_url = base_url

    def create(self, **kwargs):
        key = self._base_url
        if key not in self._endpoints.calls:
            raise AssertionError(f"call to an endpoint no test registered: {key!r}")
        sent = " ".join(str(m.get("content")) for m in (kwargs.get("messages") or []))
        with self._endpoints.lock:
            self._endpoints.calls[key] += 1
            self._endpoints.texts[key].append(sent)
        content = f"<<{self._endpoints.name(key)}>> traducido"
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=content))])


class _FakeOpenAI:
    endpoints: Endpoints

    def __init__(self, **kwargs):
        self.api_key = kwargs.get("api_key")
        self.base_url = kwargs.get("base_url")
        self.chat = types.SimpleNamespace(
            completions=_FakeCompletions(type(self).endpoints, self.base_url))


class _FakeStorage:
    """NiceGUI's per-session cookie store, which does not exist outside a
    request. active_profile / engine_runs / usage_store run their real
    production code on top of it."""

    def __init__(self) -> None:
        self.current: dict = {"cache_scope": "session:compare"}

    @property
    def user(self) -> dict:
        return self.current


@pytest.fixture
def endpoints(monkeypatch):
    registry = Endpoints()
    monkeypatch.setattr(_FakeOpenAI, "endpoints", registry, raising=False)
    monkeypatch.setattr(openai, "OpenAI", _FakeOpenAI)
    return registry


@pytest.fixture
def compare_ui(monkeypatch, endpoints):
    """A real TranslationUI rendering the real /compare page."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-passage-hosted-key")
    monkeypatch.setenv("TRANSLATION_PROVIDER", "openai")
    store = _FakeStorage()
    monkeypatch.setattr(ui_module, "app", types.SimpleNamespace(storage=store))
    backend = TranslationBackend()
    # The Ollama reachability probe is a network call; pin the machine's model
    # list so the local candidate is deterministic. Everything downstream —
    # profile construction, provider selection, the call itself — is real.
    monkeypatch.setattr(TranslationBackend, "available_local_models", lambda self: [LOCAL_TAG])
    app_ui = TranslationUI(backend=backend)
    app_ui.compare_page()
    return app_ui, store.current


def _run_comparison(text: str = SENTENCE) -> None:
    """Type into the page's textarea and click its Run comparison button.

    The click goes through NiceGUI's own event dispatch, which schedules the
    async handler as a background task — so this drives a real event loop and
    waits for those tasks rather than calling the closure directly.
    """
    client = context.client
    areas = [e for e in client.elements.values() if isinstance(e, ui.textarea)]
    assert areas, "the compare page has no text input"
    areas[-1].value = text
    buttons = [e for e in client.elements.values()
               if isinstance(e, ui.button) and "Run comparison" in (e.text or "")]
    assert buttons, "the compare page has no Run comparison button"
    listener = list(buttons[-1]._event_listeners.values())[-1]

    async def click() -> None:
        core.loop = asyncio.get_running_loop()
        listener.handler(types.SimpleNamespace(args=None, sender=buttons[-1]))
        while background_tasks.running_tasks:
            await asyncio.gather(*list(background_tasks.running_tasks))

    asyncio.run(click())


def _rows(store: dict) -> list[dict]:
    return [r for r in store.get("engine_runs", []) if r.get("surface") == "compare"]


def test_a_compare_run_reaches_the_ledger_naming_every_engine_that_answered(compare_ui):
    """D1. The session's only action is Compare. Both engines really ran, and
    both must appear: the page previously wrote nothing at all."""
    _, store = compare_ui
    _run_comparison()

    rows = _rows(store)
    assert {r["engine"] for r in rows} == {f"hosted:{TEXT_MODEL}", f"local:{LOCAL_TAG}"}, rows
    assert all(r["chars"] == len(SENTENCE) for r in rows), rows
    # A fan-out is not one run by one invented engine.
    assert not any(r["engine"] in ("compare", "hosted:compare") for r in rows), rows


def test_the_hosted_leg_of_a_comparison_is_metered_and_the_local_leg_is_not(
    compare_ui, endpoints
):
    """The text really is sent to Passage's key, and Passage really pays for
    that one call — while the model on this machine costs it nothing."""
    _, store = compare_ui
    _run_comparison()

    assert endpoints.calls[HOSTED] == 1, "the hosted endpoint never received the sentence"
    assert endpoints.calls[LOCAL_URL] == 1, "the local model never ran"
    assert any(SENTENCE in t for t in endpoints.texts[HOSTED]), endpoints.texts[HOSTED]

    by_engine = {r["engine"]: r for r in _rows(store)}
    hosted = by_engine[f"hosted:{TEXT_MODEL}"]
    local = by_engine[f"local:{LOCAL_TAG}"]
    assert (hosted["metered"], hosted["left_machine"], hosted["ran"]) == (True, True, "hosted")
    assert (local["metered"], local["left_machine"], local["ran"]) == (False, False, "local")
    assert store["usage"]["metered_runs"] == 1
    assert store["usage"]["metered_chars"] == len(SENTENCE)


def test_a_comparison_cannot_claim_the_whole_session_stayed_on_this_machine(compare_ui):
    """The over-claim, on the summary /engines renders.

    Compare's local leg used to be the only thing anyone could see: a local
    run followed by a Compare left the page saying 100% of the session's
    characters stayed on this machine, seconds after the sentence had been
    delivered to a hosted model on Passage's key. Half the compared characters
    left, and the summary now says so.
    """
    from passage import engine_ledger

    _, store = compare_ui
    _run_comparison()

    summary = engine_ledger.summarise(store["engine_runs"])
    assert summary["remote"]["runs"] == 1, summary
    assert summary["remote"]["chars"] == len(SENTENCE), summary
    assert summary["local_share_of_chars"] == 0.5, summary
    assert summary["metered_runs"] == 1, summary


def test_an_engine_that_failed_is_not_recorded_as_a_run(compare_ui, monkeypatch):
    """Only work that produced an answer is a run. A failed candidate sent
    text out but got nothing back; booking it as a delivered translation would
    make the ledger disagree with the page beside it."""
    _, store = compare_ui

    def only_hosted(self, profile=None):
        return [{"label": "Passage hosted", "engine": f"hosted:{TEXT_MODEL}",
                 "model": TEXT_MODEL, "is_local": False, "profile": None},
                {"label": "Broken", "engine": "local:broken", "model": "broken",
                 "is_local": True, "profile": None}]

    real_compare = TranslationBackend.compare_translations

    def failing(self, text, target_language, candidates):
        results = real_compare(self, text, target_language, candidates)
        for r in results:
            if r.engine == "local:broken":
                r.error, r.text = "connection refused", ""
        return results

    monkeypatch.setattr(TranslationBackend, "comparison_candidates", only_hosted)
    monkeypatch.setattr(TranslationBackend, "compare_translations", failing)

    _run_comparison()

    assert [r["engine"] for r in _rows(store)] == [f"hosted:{TEXT_MODEL}"], _rows(store)
