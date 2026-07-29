"""What Passage SAYS about a request must follow what actually served it.

Every defect below was reproduced in a live browser session before it was
written down here. They share one cause: privacy sentences and metered flags
were derived from a capability snapshot — what was reachable when the page
rendered — rather than from the engine that answered. The worst single case
was one response body that simultaneously reported engine
"hosted:gpt-5.4-nano", promised "Runs on ghost-model:999b on this machine",
and set "metered": false.

Two rules these tests hold themselves to:

* Assert WHICH PATH RAN. Every test below either forces a specific engine
  label into the real production call, or drives the real routing and then
  asserts the resulting label — never merely that a call returned 200. Where
  a capability snapshot exists, it is deliberately loaded with a DIFFERENT
  answer than the engine that serves the request, so a regression that goes
  back to reading the snapshot fails loudly instead of coincidentally passing.
* No bypasses. Nothing here builds an object with __new__ and hand-assigned
  attributes; the routes, `_record_engine_run`, `_run_recorded` and the mobile
  handlers are the production functions.

The one seam: `engine_runs` / `usage_store` are backed by `app.storage.user`,
which does not exist outside a request and degrades to a throwaway container
(fail-safe, not a test seam). These tests substitute plain containers at the
property, exactly as tests/test_mobile_ui_flow.py does for recent_threads, so
the production bookkeeping has somewhere real to write.
"""
import asyncio
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import TranslationUI as ui_module
from TranslationUI import TranslationUI
from TranslationBackend import TEXT_MODEL, VISION_MODEL
from passage import engine_ledger, policy, usage
from passage import provider_profiles as pp


class DummyProgress:
    def set_value(self, value):
        return None


class DummyLabel:
    def __init__(self):
        self.text = ""


def _session(monkeypatch, ui_app):
    """Give this instance real session containers and hand them back."""
    runs: list = []
    used: dict = {}
    monkeypatch.setattr(type(ui_app), "engine_runs", property(lambda self: runs))
    monkeypatch.setattr(type(ui_app), "usage_store", property(lambda self: used))
    return runs, used


def _app_request(ui_app: TranslationUI):
    return types.SimpleNamespace(
        headers={"x-passage-token": ui_app.api_guard.issue_token()},
        client=types.SimpleNamespace(host="127.0.0.1"),
    )


def _snapshot_claiming(monkeypatch, model):
    """Make the capability probe report `model` as the local default.

    Loaded with a model that does NOT serve the request under test, so any
    sentence still derived from the snapshot is identifiable on sight.
    """
    snap = {
        "probe": {"reachable": bool(model), "outcome": "ok", "models": [model] if model else []},
        "models": [model] if model else [],
        "chosen_model": model,
        "probed_at": 0.0,
        "pending": False,
    }
    monkeypatch.setattr(ui_module, "_take_local_snapshot", lambda backend: dict(snap))
    return snap


def _text_translate(ui_app, text="hola", language="es"):
    resp = asyncio.run(ui_app.api_text_translate(_app_request(ui_app), text=text, language=language))
    return json.loads(resp.body.decode())


# ---------------------------------------------------------------------------
# G1 — the exact live contradiction, as a regression test.
# ---------------------------------------------------------------------------

def test_a_hosted_answer_is_never_described_as_running_on_this_machine(monkeypatch):
    """One body said engine "hosted:gpt-5.4-nano", "Runs on ghost-model:999b on
    this machine", and "metered": false. Reproduced here exactly: the probe is
    loaded with ghost-model:999b (so the old snapshot-derived code would still
    print it) while a recording fake serves the request from a hosted model."""
    ui_app = TranslationUI()
    _session(monkeypatch, ui_app)
    _snapshot_claiming(monkeypatch, "ghost-model:999b")
    monkeypatch.setattr(ui_app.backend, "translate_live",
                        lambda text, language, profile=None: ("hola", "hosted:gpt-5.4-nano"))

    payload = _text_translate(ui_app, "hello")

    # Which path ran: the hosted fake served it, and the snapshot said otherwise.
    assert payload["engine"] == "hosted:gpt-5.4-nano"
    assert asyncio.run(ui_module.local_snapshot_async(ui_app.backend))["chosen_model"] \
        == "ghost-model:999b"

    assert payload["metered"] is True
    assert payload["left_machine"] is True
    assert payload["ran_on"] == "hosted"
    assert "ghost-model:999b" not in payload["privacy"]
    assert "on this machine" not in payload["privacy"]
    assert "gpt-5.4-nano" in payload["privacy"] and "metered" in payload["privacy"]


def test_a_local_answer_is_described_as_local_even_when_the_probe_disagrees(monkeypatch):
    """The mirror image, so the fix can't be "always say hosted": the probe
    claims nothing is installed, and a local model serves the request anyway."""
    ui_app = TranslationUI()
    _session(monkeypatch, ui_app)
    _snapshot_claiming(monkeypatch, None)
    monkeypatch.setattr(ui_app.backend, "translate_live",
                        lambda text, language, profile=None: ("hola", "local:translategemma:4b"))

    payload = _text_translate(ui_app, "hello")

    assert payload["engine"] == "local:translategemma:4b"
    assert asyncio.run(ui_module.local_snapshot_async(ui_app.backend))["chosen_model"] is None
    assert payload["metered"] is False
    assert payload["left_machine"] is False
    assert "translategemma:4b" in payload["privacy"] and "this machine" in payload["privacy"]


def test_hosted_after_a_local_failure_is_metered_through_the_real_router(monkeypatch):
    """No forced label at all: the REAL translate_live routes here. A local
    provider is reachable and fails, so the hosted fallback serves the text —
    and that is metered even though the user configured nothing hosted."""
    ui_app = TranslationUI()
    _session(monkeypatch, ui_app)
    _snapshot_claiming(monkeypatch, "translategemma:4b")

    class BoomProvider:
        text_model = "translategemma:4b"

        def create_chat_completion(self, **_kw):
            raise RuntimeError("local endpoint died mid-sentence")

    monkeypatch.setattr(ui_app.backend, "_live_local_provider", lambda: BoomProvider())
    monkeypatch.setattr(ui_app.backend, "translate_text", lambda text, language: "hola")

    payload = _text_translate(ui_app, "hello there")

    # Which path ran: local was tried, raised, and hosted answered.
    assert payload["engine"] == f"hosted:{TEXT_MODEL}"
    assert payload["metered"] is True
    assert "this machine" not in payload["privacy"]


def test_a_byo_endpoint_is_sent_out_but_not_metered(monkeypatch):
    ui_app = TranslationUI()
    runs, used = _session(monkeypatch, ui_app)
    profile = pp.ProviderProfile(label="mine", kind=pp.KIND_BYO, model="gpt-4o-mini")
    monkeypatch.setattr(type(ui_app), "active_profile", property(lambda self: profile))
    monkeypatch.setattr(ui_app.backend, "translate_live",
                        lambda text, language, prof=None: ("hola", profile.describe()))

    payload = _text_translate(ui_app, "hello")

    assert payload["engine"] == "byo:gpt-4o-mini"
    assert payload["left_machine"] is True     # it did leave the machine
    assert payload["metered"] is False         # Passage did not pay for it
    assert usage.summary(used).metered_runs == 0
    assert engine_ledger.LedgerEntry(**runs[0]).destination == "sent out"


# ---------------------------------------------------------------------------
# T1 / G2 — a cache hit was booked as "sent out" AND metered.
# ---------------------------------------------------------------------------

def test_a_cache_hit_is_not_sent_out_and_is_not_metered(monkeypatch):
    ui_app = TranslationUI()
    runs, used = _session(monkeypatch, ui_app)
    _snapshot_claiming(monkeypatch, "translategemma:4b")
    monkeypatch.setattr(ui_app.backend, "translate_live",
                        lambda text, language, profile=None: ("hola", "cache"))

    payload = _text_translate(ui_app, "38 characters of text, near enough")

    assert payload["engine"] == "cache"          # which path ran
    assert payload["metered"] is False
    assert payload["left_machine"] is False
    entry = engine_ledger.LedgerEntry(**runs[0])
    assert entry.engine == "cache"
    assert entry.destination == "this machine"   # was "sent out"
    assert entry.metered is False
    assert usage.summary(used).metered_runs == 0  # was 1


def test_the_engines_page_no_longer_contradicts_itself_on_a_cache_hit(monkeypatch):
    """The live /engines contradiction, reassembled from the same three
    functions the page renders from. It read, top to bottom:

        Runs on translategemma:4b on this machine …
        64% of your text stayed on this machine  /  sent out: 1 runs
        cache - 1 run
        38 of 50,000 free characters used across 1 metered run.

    One page, one session, no misconfiguration.
    """
    ui_app = TranslationUI()
    runs, used = _session(monkeypatch, ui_app)
    _snapshot_claiming(monkeypatch, "translategemma:4b")
    monkeypatch.setattr(ui_app.backend, "translate_live",
                        lambda text, language, profile=None: ("hola", "cache"))

    _text_translate(ui_app, "x" * 38)

    summary = engine_ledger.summarise(runs)
    assert summary["total_runs"] == 1
    assert summary["engines"] == {"cache": 1}
    assert summary["remote"]["runs"] == 0          # nothing was sent out
    assert summary["local_share_of_chars"] == 1.0  # was 0.64
    assert summary["metered_runs"] == 0

    line = usage.describe(usage.summary(used), total_runs=summary["total_runs"])
    assert "metered run" not in line               # was "…across 1 metered run."


def test_empty_input_is_not_sent_out_and_is_not_metered(monkeypatch):
    """translate_live returns the "none" label for empty text — which failed
    `engine.startswith("local")` exactly as "cache" did, and was filed as sent
    out. Booked through the production recorder."""
    ui_app = TranslationUI()
    runs, used = _session(monkeypatch, ui_app)

    run = ui_app._record_engine_run(surface=policy.Surface.LIVE_TEXT, engine="none",
                                    latency_ms=1, chars=0)

    assert run.ran is policy.Ran.NOTHING
    assert engine_ledger.LedgerEntry(**runs[0]).destination == "this machine"
    assert usage.summary(used).metered_runs == 0
    assert "nothing was sent anywhere" in run.privacy.lower()


def test_an_unrecognised_engine_is_admitted_not_assumed_local(monkeypatch):
    """The optimistic-guess class of bug, in the one place left for it: an
    engine label nobody can classify must not be rendered as either promise."""
    ui_app = TranslationUI()
    runs, _used = _session(monkeypatch, ui_app)

    run = ui_app._record_engine_run(surface=policy.Surface.TEXT, engine="mystery-box",
                                    latency_ms=1, chars=10)

    assert run.left_machine is None
    assert "can't tell" in run.privacy
    entry = engine_ledger.LedgerEntry(**runs[0])
    assert entry.destination == "not known"
    summary = engine_ledger.summarise(runs)
    assert summary["local"]["runs"] == 0 and summary["remote"]["runs"] == 0
    assert summary["unknown"]["runs"] == 1


# ---------------------------------------------------------------------------
# D1 / C2 — documents, images and voice were invisible to /engines entirely.
# ---------------------------------------------------------------------------

def test_a_workspace_text_translation_is_recorded_and_metered(monkeypatch):
    ui_app = TranslationUI()
    runs, used = _session(monkeypatch, ui_app)
    monkeypatch.setattr(ui_app.backend, "translate_live",
                        lambda text, language, profile=None: ("hallo", f"hosted:{TEXT_MODEL}"))
    ui_app.show_mobile_voice_result = lambda *a, **k: None

    assert runs == []  # the old behaviour: "Nothing translated yet", forever

    ui_app._run_mobile_text_translation("hello there", "German", DummyProgress(), DummyLabel())

    entry = engine_ledger.LedgerEntry(**runs[0])
    # The Text tab, filed under the Text tab. This row used to say "voice",
    # so /engines attributed typing to a microphone that was never opened.
    assert entry.surface == "text"
    assert entry.engine == f"hosted:{TEXT_MODEL}"   # which path ran
    assert entry.destination == "sent out" and entry.metered is True
    assert usage.summary(used).metered_chars == len("hello there")


def test_an_image_translation_is_recorded_and_metered_by_its_source_characters(monkeypatch):
    ui_app = TranslationUI()
    runs, used = _session(monkeypatch, ui_app)
    ui_app.image_upload_bytes, ui_app.image_upload_name = b"fake-bytes", "menu.png"
    ui_app.show_mobile_image_result = lambda *a, **k: None
    monkeypatch.setattr(ui_app.backend, "translate_image_text_blocks",
                        lambda payload, name, language: {
                            "translated_blocks": [
                                {"source_text": "Bienvenue", "translated_text": "Welcome"},
                                {"source_text": "Plat du jour", "translated_text": "Dish of the day"},
                            ],
                        })

    ui_app._run_mobile_image_translation("English", DummyProgress(), DummyLabel())

    entry = engine_ledger.LedgerEntry(**runs[0])
    assert entry.surface == "image"
    assert entry.engine == f"hosted:{VISION_MODEL}"   # which path ran
    assert entry.metered is True
    # Counted from the text the photo turned out to contain, not the file size.
    assert entry.chars == len("Bienvenue") + len("Plat du jour")
    assert usage.summary(used).metered_chars == entry.chars


def test_the_image_api_route_records_the_run_too(monkeypatch):
    ui_app = TranslationUI()
    runs, used = _session(monkeypatch, ui_app)
    monkeypatch.setattr(ui_app.backend, "translate_image_text_blocks",
                        lambda payload, name, language: {
                            "translated_blocks": [{"source_text": "Salida", "translated_text": "Exit"}]})

    class Upload:
        filename = "sign.png"

        async def read(self):
            return b"bytes"

    resp = asyncio.run(ui_app.api_image_translate(_app_request(ui_app), file=Upload(), language="en"))

    assert resp.status_code == 200
    entry = engine_ledger.LedgerEntry(**runs[0])
    assert entry.engine == f"hosted:{VISION_MODEL}" and entry.metered is True
    assert usage.summary(used).metered_chars == len("Salida")


def test_a_document_run_is_one_ledger_entry_carrying_all_its_characters(monkeypatch):
    """engine_ledger.summarise exists because "one document is one entry but a
    lot of text" — and documents were the one surface it never saw. A document
    must outweigh the typing beside it in the local-vs-sent-out share."""
    ui_app = TranslationUI()
    runs, used = _session(monkeypatch, ui_app)
    record = ui_app._engine_recorder()

    record(surface=policy.Surface.LIVE_TEXT, engine="local:translategemma:4b",
           latency_ms=40, chars=55)
    record(surface=policy.Surface.DOCUMENT, engine=f"hosted:{TEXT_MODEL}",
           latency_ms=9000, chars=8000)

    summary = engine_ledger.summarise(runs)
    assert summary["total_runs"] == 2
    assert summary["remote"]["chars"] == 8000
    assert summary["local_share_of_chars"] < 0.01   # by count alone it would be 50%
    assert usage.summary(used).metered_chars == 8000


def test_the_recorder_survives_a_worker_thread_without_session_storage(monkeypatch):
    """Documents, images and voice all finish off the event loop, where
    `app.storage.user` raises and the properties degrade to throwaway
    containers — i.e. the record would be written to nothing. The recorder is
    bound on the request thread for exactly this reason."""
    from threading import Thread

    ui_app = TranslationUI()
    runs, _used = _session(monkeypatch, ui_app)
    record = ui_app._engine_recorder()
    # Now make the properties behave as they do off the request context.
    monkeypatch.setattr(type(ui_app), "engine_runs", property(lambda self: []))

    Thread(target=lambda: record(surface=policy.Surface.DOCUMENT,
                                 engine=f"hosted:{TEXT_MODEL}",
                                 latency_ms=10, chars=1200)).start()
    import time as _t
    for _ in range(100):
        if runs:
            break
        _t.sleep(0.01)

    assert len(runs) == 1 and runs[0]["chars"] == 1200


def test_a_backend_cache_hit_is_unmetered_but_still_names_its_producer(monkeypatch):
    """A cache hit is TWO facts. Nothing ran and nothing was sent for this
    request (so: not metered, "this machine"), and the words on screen were
    made earlier by a specific engine (so: that engine is still named).

    Driven through the REAL cache — a planted entry and the real
    `translate_live` — rather than by nudging a counter, because a counter is
    exactly what could not answer this question: it is process-wide, so one
    session's hit used to relabel another session's in-flight hosted call."""
    ui_app = TranslationUI()
    runs, used = _session(monkeypatch, ui_app)

    key = ui_app.backend._normalize_cache_key(
        "hello there", "Spanish", mode="live", profile=None)
    ui_app.backend._cache_put(key, "hola", f"hosted:{TEXT_MODEL}")
    ui_app.show_mobile_voice_result = lambda *a, **k: None

    ui_app._run_mobile_text_translation("hello there", "Spanish", DummyProgress(), DummyLabel())

    entry = engine_ledger.LedgerEntry(**runs[0])
    assert entry.engine == f"hosted:{TEXT_MODEL}"   # who produced the bytes
    assert entry.destination == "this machine" and entry.metered is False
    assert usage.summary(used).metered_runs == 0


# ---------------------------------------------------------------------------
# G3 — hosted was promised even with no provider and no local model.
# ---------------------------------------------------------------------------

def test_no_provider_and_no_local_model_does_not_promise_hosted():
    line = policy.describe_privacy(None, local_first_model=None, hosted_available=False)
    assert "hosted" not in line.lower() or "no hosted provider" in line
    assert "can't translate" in line


def test_the_engines_page_passes_what_health_already_knows(monkeypatch):
    """/api/health reports `hosted_provider` from `backend.provider`, so the
    page had the fact available and guessed anyway. Asserted against the real
    health route rather than a constant, so the two can't drift apart."""
    ui_app = TranslationUI()
    monkeypatch.setattr(ui_app.backend, "provider", None)
    # The real snapshot builder, over a probe that finds nothing — so the
    # "no local model" half of the claim is produced the way production
    # produces it rather than asserted into existence.
    monkeypatch.setattr(ui_module, "probe_local_llm",
                        lambda: {"reachable": False, "outcome": "unreachable", "models": []})

    health = json.loads(asyncio.run(ui_app.api_health(_app_request(ui_app))).body.decode())
    assert health["hosted_provider"] is False
    assert health["local_default"] is None

    line = policy.describe_privacy(None, local_first_model=health["local_default"],
                                   hosted_available=health["hosted_provider"])
    assert "can't translate" in line


def test_hosted_is_still_promised_when_a_provider_is_configured():
    """The fix must not turn into "never say hosted"."""
    line = policy.describe_privacy(None, local_first_model=None, hosted_available=True)
    assert "hosted" in line and "metered" in line


# ---------------------------------------------------------------------------
# G6 — "everything ran locally" before anything had run.
# ---------------------------------------------------------------------------

def test_zero_runs_is_not_evidence_of_privacy():
    empty = usage.summary({})
    assert usage.describe(empty, total_runs=0) == \
        "Nothing translated yet, so nothing has been metered."
    assert "ran locally" not in usage.describe(empty, total_runs=0)
    # Once something HAS run unmetered, the claim is sayable.
    assert "own key" in usage.describe(empty, total_runs=3)
    # And with no ledger in hand, no claim is made either way.
    assert "own key" not in usage.describe(empty)


# ---------------------------------------------------------------------------
# The classifier itself, over the whole vocabulary of engine labels.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("engine,ran,left,metered", [
    ("local:translategemma:4b", policy.Ran.LOCAL, False, False),
    ("cache", policy.Ran.CACHE, False, False),
    ("none", policy.Ran.NOTHING, False, False),
    ("", policy.Ran.NOTHING, False, False),
    ("hosted:gpt-5.4-nano", policy.Ran.HOSTED, True, True),
    ("app:gpt-5.4-nano", policy.Ran.HOSTED, True, True),
    ("byo:gpt-4o-mini", policy.Ran.BYO, True, False),
    ("who-knows", policy.Ran.UNKNOWN, None, False),
])
def test_every_engine_label_classifies_the_same_way_everywhere(engine, ran, left, metered):
    run = policy.classify_run(engine)
    assert (run.ran, run.left_machine, run.metered) == (ran, left, metered)
    # A run Passage did not pay for is never metered, and a metered run always
    # left the machine — the two claims can't come apart.
    assert not (run.metered and run.left_machine is not True)


def test_a_privacy_sentence_never_says_this_machine_about_a_hosted_run():
    for engine in ("hosted:gpt-5.4-nano", "app:gpt-5.4-nano", "byo:gpt-4o-mini"):
        assert "on this machine" not in policy.classify_run(engine).privacy
    for engine in ("local:x", "cache", "none"):
        assert "sent" in policy.classify_run(engine).privacy


# ---------------------------------------------------------------------------
# The /engines FORECAST must be the router's answer, not the installed list.
# Verified live before it was written down: with PASSAGE_LIVE_LOCAL=0 the page
# still printed "Your next translation will run on translategemma:4b on this
# machine" while the router was sending everything to a hosted model, because
# the snapshot chose from report["models"] (everything INSTALLED) instead of
# report["chosen_model"] / report["can_serve"] (what will actually serve).
# ---------------------------------------------------------------------------

def _probing_backend(report):
    return types.SimpleNamespace(probe_local_llm=lambda: dict(report))


def test_the_forecast_names_the_model_that_will_really_serve():
    snap = ui_module._take_local_snapshot(_probing_backend({
        "reachable": True, "outcome": "ok",
        "models": ["translategemma:4b", "qwen2.5:7b"],
        "routing_enabled": True, "chosen_model": "translategemma:4b",
        "can_serve": True,
    }))
    assert snap["chosen_model"] == "translategemma:4b"
    assert snap["can_serve"] is True


def test_no_local_forecast_when_the_router_will_not_serve_locally():
    """Reachable daemon, models installed, routing off. "The endpoint is up"
    and "local will answer your next request" are different facts."""
    snap = ui_module._take_local_snapshot(_probing_backend({
        "reachable": True, "outcome": "ok",
        "models": ["translategemma:4b", "qwen2.5:7b"],
        "routing_enabled": False, "chosen_model": None, "can_serve": False,
    }))
    assert snap["can_serve"] is False
    assert snap["chosen_model"] is None, "forecast a local run the router will not make"
    # Diagnostics still needs the installed list; only the FORECAST is gated.
    assert snap["models"] == ["translategemma:4b", "qwen2.5:7b"]
    # And the sentence the page prints follows chosen_model, so it cannot
    # promise this machine.
    assert "on this machine" not in policy.describe_privacy(
        None, local_first_model=snap["chosen_model"], hosted_available=True)
