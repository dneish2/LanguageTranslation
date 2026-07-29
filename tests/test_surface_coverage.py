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
    delivered to a hosted model on Passage's key. All of the compared text
    left, and the summary now says so.

    The share is 0.0, not the 0.5 this test used to assert. The user typed ONE
    sentence and every character of it went to Passage's hosted key; that a
    local model also answered the same sentence does not unsend it, and
    dividing per-row would make the number a function of how many local models
    happen to be installed (six of them read "83% stayed on this machine").
    """
    from passage import engine_ledger

    _, store = compare_ui
    _run_comparison()

    summary = engine_ledger.summarise(store["engine_runs"])
    assert summary["remote"]["runs"] == 1, summary
    assert summary["remote"]["chars"] == len(SENTENCE), summary
    assert summary["local_share_of_chars"] == 0.0, summary
    assert summary["new_chars"] == len(SENTENCE), summary
    assert summary["metered_runs"] == 1, summary


def test_a_local_engine_that_failed_is_not_recorded_as_a_run(compare_ui, monkeypatch):
    """A LOCAL leg that failed writes no row, because nothing left the machine.

    Connection refused on localhost: the sentence never reached anybody, so
    there is no destination to disclose and no work to report. This test used
    to be called "an engine that failed" and demonstrated the rule with this
    local case while locking it in for hosted ones too — where the bytes had
    already been delivered. See the hosted case below: it is the opposite rule
    for the opposite reason, and conflating them is what reprinted "100% of
    the characters you translated stayed on this machine".
    """
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


def _hosted_fails(monkeypatch, endpoints, *, how: str):
    """Make the hosted leg fail AFTER the bytes are delivered.

    Exactly what a 429, a timeout or an empty answer looks like from here: the
    endpoint counts the call — the sentence is in `endpoints.texts[HOSTED]`,
    it is on someone else's machine — and only then does the reply come back
    unusable. `how="raise"` errors, `how="empty"` answers with nothing.
    """
    real_create = _FakeCompletions.create

    def create(self, **kwargs):
        completion = real_create(self, **kwargs)
        if self._base_url is not HOSTED:
            return completion
        if how == "raise":
            raise RuntimeError("429 rate limit")
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=""))])

    monkeypatch.setattr(_FakeCompletions, "create", create)


@pytest.mark.parametrize("how", ["raise", "empty"])
def test_a_hosted_leg_that_failed_is_still_disclosed_as_sent_out(
    compare_ui, endpoints, monkeypatch, how
):
    """F1. A failed hosted leg changes what the user GOT, not where their words
    WENT — so it is disclosed, and it is not billed.

    Booking rows only `if r.ok` printed, live, for a comparison whose hosted
    leg 429'd on a 57-character secret: "100% of the 285 characters you
    translated stayed on this machine · sent out: 0 runs · 0 chars", with the
    sentence sitting in the hosted provider's logs. The bytes are gone; the
    only thing the failure removes is the bill.
    """
    from passage import engine_ledger

    _, store = compare_ui
    _hosted_fails(monkeypatch, endpoints, how=how)

    _run_comparison()

    # WHICH PATH RAN: the hosted endpoint really received these bytes, and the
    # local model really answered. Neither is assumed.
    assert endpoints.calls[HOSTED] == 1, "the hosted endpoint never saw the sentence"
    assert endpoints.calls[LOCAL_URL] == 1, "the local model never ran"
    assert any(SENTENCE in t for t in endpoints.texts[HOSTED]), endpoints.texts[HOSTED]

    by_engine = {r["engine"]: r for r in _rows(store)}
    assert set(by_engine) == {f"hosted:{TEXT_MODEL}", f"local:{LOCAL_TAG}"}, by_engine
    hosted = by_engine[f"hosted:{TEXT_MODEL}"]
    assert hosted["left_machine"] is True, hosted
    assert engine_ledger.LedgerEntry(**hosted).destination == "sent out", hosted
    assert hosted["chars"] == len(SENTENCE), hosted
    # Told, not billed. The user gets no translation and no charge.
    assert hosted["metered"] is False, hosted
    assert store.get("usage", {}).get("metered_runs", 0) == 0, store.get("usage")

    summary = engine_ledger.summarise(store["engine_runs"])
    assert summary["remote"]["runs"] == 1, summary
    assert summary["remote"]["chars"] == len(SENTENCE), summary
    assert summary["local_share_of_chars"] == 0.0, summary   # was 1.0 — "100% stayed"
    assert summary["metered_runs"] == 0, summary


def test_the_privacy_share_counts_the_users_sentence_once_per_comparison(
    compare_ui, endpoints, monkeypatch
):
    """F2. The denominator is characters the user typed, not chars x engines.

    With six engines answering one 57-character sentence the page said "83% of
    the 342 characters you translated stayed on this machine" — a claim that
    climbs toward 100% the more local models you install, about a sentence that
    went out in full. One comparison is one piece of text.
    """
    from passage import engine_ledger

    _, store = compare_ui
    monkeypatch.setattr(TranslationBackend, "available_local_models",
                        lambda self: [LOCAL_TAG, "llama3.2:3b", "phi4:14b"])

    _run_comparison()

    rows = _rows(store)
    assert len(rows) == 4, rows                      # 1 hosted + 3 local, all answered
    assert endpoints.calls[LOCAL_URL] == 3, endpoints.calls
    assert len({r["text_id"] for r in rows}) == 1, "one comparison is one piece of text"

    summary = engine_ledger.summarise(store["engine_runs"])
    assert summary["new_chars"] == len(SENTENCE), summary     # was 4 x 57 = 228
    assert summary["local_share_of_chars"] == 0.0, summary    # was 0.75


def _hosted_is_unreachable(monkeypatch):
    """The hosted endpoint refuses the connection: the socket never opens.

    The fake therefore never counts the call and never records the bytes, so
    `endpoints.calls[HOSTED] == 0` is independent evidence that nothing left
    this machine — the assertion the disclosure has to answer to. Contrast
    `_hosted_fails`, which counts first and fails after.
    """
    real_create = _FakeCompletions.create

    def create(self, **kwargs):
        if self._base_url is HOSTED:
            # What the openai SDK raises when it cannot reach the host; its
            # APIConnectionError message is literally "Connection error."
            raise RuntimeError("Connection error: could not reach api.openai.com")
        return real_create(self, **kwargs)

    monkeypatch.setattr(_FakeCompletions, "create", create)


def _engines_page_text(app_ui) -> str:
    """Render the real /engines page and return every line it prints.

    The page is the surface the claim is made on, so it is the surface the
    claim is read back from: a summary dict that is right while the sentence
    over it is wrong is the defect this file keeps finding.
    """
    client = context.client
    before = set(client.elements)
    app_ui.engines_page()
    return "\n".join(
        element.text for eid, element in list(client.elements.items())
        if eid not in before and isinstance(element, ui.label) and element.text)


def test_a_hosted_leg_that_never_connected_is_not_disclosed_as_sent_out(
    compare_ui, endpoints, monkeypatch
):
    """D-A. Nothing left this machine, so the page must not say anything did.

    `r.ok or not r.is_local` booked a row for every remote leg, delivered or
    not. Live, with an unreachable provider, that printed "0% of the 57
    characters you translated stayed on this machine · local: 3 runs · 57
    chars · sent out: 1 runs · 57 chars" while the provider's own counter sat
    at zero — a false disclosure, and a false LOSS of the local share, shown
    to the offline user this app exists for.

    A failed local leg and an unreached remote leg are the same fact and now
    get the same treatment: no bytes, no destination, no row.
    """
    from passage import engine_ledger

    app_ui, store = compare_ui
    _hosted_is_unreachable(monkeypatch)

    _run_comparison()

    # WHICH PATH RAN, from the endpoints' own counters. Nothing reached the
    # hosted endpoint; the local model really did answer.
    assert endpoints.calls[HOSTED] == 0, "the hosted endpoint was reached after all"
    assert endpoints.texts[HOSTED] == [], endpoints.texts[HOSTED]
    assert endpoints.calls[LOCAL_URL] == 1, "the local model never ran"
    assert any(SENTENCE in t for t in endpoints.texts[LOCAL_URL]), endpoints.texts[LOCAL_URL]

    rows = _rows(store)
    assert [r["engine"] for r in rows] == [f"local:{LOCAL_TAG}"], rows
    assert not any(engine_ledger.LedgerEntry(**r).destination == "sent out" for r in rows), rows

    summary = engine_ledger.summarise(store["engine_runs"])
    assert summary["remote"] == {"runs": 0, "chars": 0, "median_ms": None}, summary
    # Not filed as unknown either: a refusal is knowably nothing-left, and an
    # unknown row in this fan-out's group would pull the share off 100%.
    assert summary["unknown"]["runs"] == 0, summary
    assert summary["local_share_of_chars"] == 1.0, summary
    assert summary["new_chars"] == len(SENTENCE), summary

    page = _engines_page_text(app_ui)
    assert f"100% of the {len(SENTENCE)} characters you translated stayed on this machine" \
        in page, page
    assert "sent out: 0 runs · 0 chars" in page, page
    assert "destination not known" not in page, page


def test_the_usage_block_does_not_contradict_a_disclosed_run(
    compare_ui, endpoints, monkeypatch
):
    """D-B. One page, two blocks, opposite claims about the same sentence.

    A delivered-then-failed hosted leg is disclosed and unbilled, so the
    metered counter stays at zero — and the zero-metered copy read "Nothing
    metered this session — everything so far ran on this machine, from cache,
    or on your own key" four blocks under "sent out: 1 runs · 57 chars". True
    about the bill, false about the destination, and it only claimed the
    second because it assumed unmetered implies local.
    """
    app_ui, store = compare_ui
    _hosted_fails(monkeypatch, endpoints, how="raise")

    _run_comparison()

    # WHICH PATH RAN: these bytes really were delivered before the failure.
    assert endpoints.calls[HOSTED] == 1, "the hosted endpoint never saw the sentence"
    assert any(SENTENCE in t for t in endpoints.texts[HOSTED]), endpoints.texts[HOSTED]
    assert store.get("usage", {}).get("metered_runs", 0) == 0, store.get("usage")

    page = _engines_page_text(app_ui)
    assert f"sent out: 1 runs · {len(SENTENCE)} chars" in page, page
    assert "Nothing metered this session." in page, page
    assert "everything so far ran on this machine" not in page, page
