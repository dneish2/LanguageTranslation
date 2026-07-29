"""Cache provenance is a PER-CALL fact, not an inference from shared state.

The bug these tests pin down (reproduced live in two real browser sessions):

  D4  A live-text cache hit returned the STORED engine label ("hosted:…"),
      which is correct for the LABEL, and the caller then metered it as a fresh
      hosted call. One hosted call, billed twice.
  D2  Whether a call was a cache hit was inferred by sampling
      ``backend.metrics.cache_hits`` before and after it. That counter is
      PROCESS-WIDE. While session B's photo was in flight against a hosted
      vision model, session A got a keystroke cache hit, and B's receipt became
      engine "cache" / is_local true / left_machine false — "no model ran and
      nothing was sent anywhere", printed over a photograph that had just been
      base64'd and sent out.

So the assertions here are deliberately of the form "WHICH endpoint was called,
how many times" — every fake endpoint counts its own calls and stamps its own
identity into the bytes it returns, so a test fails if another engine served
the request or if nothing ran at all. Nothing is constructed with ``__new__``
and no attribute is hand-assigned onto a backend: the real
``TranslationBackend()`` constructor runs, against fake OpenAI endpoints keyed
by base_url.

D6 is covered too: /engines forecast a local run on a server where local
routing was switched off, or where PASSAGE_LIVE_LOCAL_MODEL named a model that
was not installed.
"""

import sys
import threading
import time
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import openai

import TranslationBackend as tb
from TranslationBackend import TranslationBackend


# ─────────────────────────── endpoint doubles ─────────────────────────── #


class _Endpoints:
    """Fake OpenAI-compatible endpoints keyed by base_url (None = hosted).

    Each records its own call count and returns text naming itself, so an
    assertion proves WHICH engine ran rather than merely that a call returned.
    """

    def __init__(self) -> None:
        self.replies: dict[object, str] = {}
        self.calls: dict[object, int] = {}
        self.delays: dict[object, float] = {}
        self.entered = threading.Event()
        self._lock = threading.Lock()

    def register(self, base_url, reply: str, delay: float = 0.0) -> None:
        self.replies[base_url] = reply
        self.calls.setdefault(base_url, 0)
        self.delays[base_url] = delay

    def count(self, base_url) -> int:
        return self.calls.get(base_url, 0)


class _FakeCompletions:
    def __init__(self, endpoints: _Endpoints, base_url) -> None:
        self._endpoints = endpoints
        self._base_url = base_url

    def create(self, **_kwargs):
        key = self._base_url
        if key not in self._endpoints.replies:
            raise AssertionError(f"unexpected call to unregistered endpoint {key!r}")
        with self._endpoints._lock:
            self._endpoints.calls[key] = self._endpoints.calls.get(key, 0) + 1
        self._endpoints.entered.set()
        delay = self._endpoints.delays.get(key, 0.0)
        if delay:
            time.sleep(delay)
        content = self._endpoints.replies[key]
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=content))]
        )


class _FakeOpenAI:
    """Stands in for ``openai.OpenAI`` so the provider is built by the
    production path, not assembled by the test."""

    endpoints: _Endpoints

    def __init__(self, **kwargs):
        self.api_key = kwargs.get("api_key")
        self.base_url = kwargs.get("base_url")
        self.chat = types.SimpleNamespace(
            completions=_FakeCompletions(type(self).endpoints, self.base_url)
        )


@pytest.fixture
def endpoints(monkeypatch):
    registry = _Endpoints()
    monkeypatch.setattr(_FakeOpenAI, "endpoints", registry, raising=False)
    monkeypatch.setattr(openai, "OpenAI", _FakeOpenAI)
    return registry


@pytest.fixture
def backend(monkeypatch, endpoints):
    monkeypatch.setenv("OPENAI_API_KEY", "hosted-key")
    monkeypatch.setenv("TRANSLATION_PROVIDER", "openai")
    # No local model may answer in these tests: the point is to observe hosted
    # calls leaving, and a machine that happens to be running Ollama would
    # otherwise change which path ran.
    monkeypatch.setattr(tb, "LIVE_LOCAL_ENABLED", False)
    endpoints.register(None, "<<HOSTED>> translation")
    return TranslationBackend()


# ───────────────────── D4: a cache hit is not a run ───────────────────── #


def test_live_cache_hit_reports_original_engine_but_no_run(backend, endpoints):
    """The D4 reproduction: same session, identical text twice.

    Second time: no outbound call, same bytes, and the call itself reports
    ``from_cache`` — which is what metering and the privacy sentence must
    follow — while the engine LABEL still names the hosted model that really
    produced the bytes.
    """
    with backend.cache_scope("session-A"):
        first = backend.translate_live_detailed("the quarterly report", "Spanish")
        calls_after_first = endpoints.count(None)
        second = backend.translate_live_detailed("the quarterly report", "Spanish")

    assert calls_after_first == 1, "the first live translation did not reach the hosted endpoint"
    assert endpoints.count(None) == 1, "the second call made a NEW outbound hosted call"
    assert first.text == second.text == "<<HOSTED>> translation"

    assert first.from_cache is False
    assert first.engine.startswith("hosted:")

    assert second.from_cache is True, "a cache hit was reported as a fresh run — this is the double bill"
    assert second.engine == first.engine, "the cache hit must still name the engine that made the bytes"
    assert second.engine != "cache", "the label names the producer, not the cache"


def test_cache_hit_provenance_survives_the_policy_layer(backend, endpoints):
    """End to end through the code that actually meters: cached => not metered,
    nothing left the machine; label => still the hosted model."""
    from passage import policy

    with backend.cache_scope("session-A"):
        backend.translate_live_detailed("hello there", "Spanish")
        hit = backend.translate_live_detailed("hello there", "Spanish")

    assert endpoints.count(None) == 1
    fresh_run = policy.classify_run(hit.engine)
    assert fresh_run.metered is True, "sanity: this label alone would be metered"

    # The label is not enough; the per-call fact is. This is what the caller
    # must key metering on.
    assert hit.from_cache is True
    assert policy.classify_run("cache").metered is False
    assert policy.classify_run("cache").left_machine is False


def test_empty_text_reports_no_run_and_no_cache(backend, endpoints):
    result = backend.translate_live_detailed("   ", "Spanish")
    assert result.engine == "none"
    assert result.from_cache is False
    assert endpoints.count(None) == 0


# ──────── D2: another session's activity cannot rewrite my receipt ─────── #


def test_provenance_ignores_the_process_wide_counter(backend, endpoints):
    """The mechanism, isolated. A hosted call whose duration overlaps a bump of
    the shared cache-hit counter must still report that it ran and was sent."""
    with backend.capture_provenance() as prov:
        # Exactly what another session's keystroke cache hit does to the
        # shared metrics object while this call is in flight.
        backend.metrics.record_cache_hit()
        backend.metrics.record_cache_hit()
        with backend.cache_scope("session-B"):
            out = backend.translate_text("a photograph's caption", "Spanish")

    assert out == "<<HOSTED>> translation"
    assert endpoints.count(None) == 1, "the hosted endpoint did not actually serve this"
    assert prov.from_cache is False, (
        "another session's cache hit was allowed to relabel this call as cached — "
        "this is the false 'nothing was sent anywhere' over a sent photograph"
    )
    assert prov.engine.startswith("hosted:")


def test_two_concurrent_sessions_each_report_their_own_truth(backend, endpoints):
    """Drive both sessions at once, the way the live reproduction did.

    Session B makes a slow hosted call (stand-in for the vision call). While it
    is in flight, session A gets a live-text cache hit. Each call must report
    its own provenance.
    """
    slow_url = "https://vision.example.com/v1"
    endpoints.register(slow_url, "<<VISION>> caption", delay=0.6)

    # Prime session A's cache with one real hosted call.
    with backend.cache_scope("session-A"):
        primed = backend.translate_live_detailed("shared sentence", "Spanish")
    assert primed.from_cache is False
    assert endpoints.count(None) == 1

    results: dict[str, object] = {}
    errors: list[BaseException] = []

    def session_b() -> None:
        try:
            profile = tb.pp.ProviderProfile(
                label="BYO", kind=tb.pp.KIND_BYO, base_url=slow_url,
                api_key="secret", model="vision-model",
            )
            with backend.cache_scope("session-B"):
                results["b"] = backend.translate_live_detailed(
                    "a photograph's caption", "Spanish", profile=profile)
        except BaseException as error:  # noqa: BLE001 - reported, not swallowed
            errors.append(error)

    thread = threading.Thread(target=session_b)
    thread.start()
    assert endpoints.entered.wait(5), "session B never reached its endpoint"
    # B is now inside its slow call. A's cache hit happens right here.
    with backend.cache_scope("session-A"):
        a_hit = backend.translate_live_detailed("shared sentence", "Spanish")
    assert thread.is_alive(), "session B finished too early to overlap; test is vacuous"
    thread.join(10)
    assert not errors, errors

    b_result = results["b"]
    assert endpoints.count(slow_url) == 1, "session B's endpoint was not called"
    assert b_result.text == "<<VISION>> caption"
    assert b_result.from_cache is False, (
        "session A's cache hit corrupted session B's receipt — B's bytes WERE sent out"
    )
    assert "endpoint:" in b_result.engine or b_result.engine != "cache"

    assert endpoints.count(None) == 1, "session A's repeat made a new outbound call"
    assert a_hit.from_cache is True
    assert a_hit.text == primed.text
    assert a_hit.engine == primed.engine


def test_a_mixed_document_is_never_reported_as_cached(backend, endpoints):
    """One request, several sub-translations: "nothing was sent" is only true
    when every one of them was served from cache."""
    with backend.cache_scope("session-A"):
        backend.translate_text("first sentence", "Spanish")
        calls = endpoints.count(None)
        with backend.capture_provenance() as prov:
            backend.translate_text("first sentence", "Spanish")   # cached
            backend.translate_text("second sentence", "Spanish")  # fresh

    assert endpoints.count(None) == calls + 1, "the fresh sentence never reached the endpoint"
    assert prov.cache_hits == 1
    assert prov.engine_runs == 1
    assert prov.from_cache is False


# ───────────── D6: do not forecast a local run that cannot happen ───────── #


def _probe_against(monkeypatch, models: list[str]) -> dict:
    """Run the REAL probe against a real HTTP server that answers /api/tags."""
    import json as _json
    from http.server import BaseHTTPRequestHandler, HTTPServer

    payload = _json.dumps({"models": [{"name": n} for n in models]}).encode()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        return tb.probe_local_llm(f"http://127.0.0.1:{server.server_port}", timeout=5.0)
    finally:
        server.shutdown()
        server.server_close()


def test_forecast_is_local_when_local_can_really_serve(monkeypatch):
    monkeypatch.setattr(tb, "LIVE_LOCAL_ENABLED", True)
    monkeypatch.setattr(tb, "LIVE_LOCAL_MODEL", "")
    report = _probe_against(monkeypatch, ["translategemma:4b"])
    assert report["reachable"] is True
    assert report["can_serve"] is True
    assert report["chosen_model"] == "translategemma:4b"


def test_no_local_forecast_when_local_routing_is_disabled(monkeypatch):
    """PASSAGE_LIVE_LOCAL=0: the endpoint is up and the model is installed, but
    nothing will ask it. /engines said "your next translation will run on
    translategemma:4b on this machine" directly above "0% of your text stayed
    on this machine"."""
    monkeypatch.setattr(tb, "LIVE_LOCAL_ENABLED", False)
    monkeypatch.setattr(tb, "LIVE_LOCAL_MODEL", "")
    report = _probe_against(monkeypatch, ["translategemma:4b"])
    assert report["reachable"] is True, "the endpoint really is up; that is not the question"
    assert report["routing_enabled"] is False
    assert report["chosen_model"] is None
    assert report["can_serve"] is False


def test_no_local_forecast_when_the_named_model_is_not_installed(monkeypatch):
    monkeypatch.setattr(tb, "LIVE_LOCAL_ENABLED", True)
    monkeypatch.setattr(tb, "LIVE_LOCAL_MODEL", "gemma3:270m-not-pulled")
    report = _probe_against(monkeypatch, ["translategemma:4b"])
    assert report["reachable"] is True
    assert report["chosen_model"] is None, "forecast a model the daemon does not have"
    assert report["can_serve"] is False


def test_unreachable_endpoint_never_forecasts_local(monkeypatch):
    import socket

    monkeypatch.setattr(tb, "LIVE_LOCAL_ENABLED", True)
    monkeypatch.setattr(tb, "LIVE_LOCAL_MODEL", "")
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    report = tb.probe_local_llm(f"http://127.0.0.1:{port}", timeout=1.0)
    assert report["reachable"] is False
    assert report["chosen_model"] is None
    assert report["can_serve"] is False


def test_live_routing_and_forecast_agree_on_a_missing_named_model(monkeypatch, backend):
    """The forecast is only honest if the router makes the same decision."""
    monkeypatch.setattr(tb, "LIVE_LOCAL_ENABLED", True)
    monkeypatch.setattr(tb, "LIVE_LOCAL_MODEL", "gemma3:270m-not-pulled")
    monkeypatch.setattr(
        TranslationBackend, "available_local_models_detailed",
        lambda self: [{"name": "translategemma:4b"}],
    )
    assert backend.choose_local_model() is None
    assert tb.usable_local_model(["translategemma:4b"]) is None


# ───────────────── D7: whitespace around a hyperlink survives ──────────── #


def test_hyperlink_paragraph_keeps_the_space_before_the_link(backend, monkeypatch, tmp_path):
    """"See " + link("the documentation") rendered as "Verla documentación" in
    the DOCUMENT: the boundary space was stripped before translation and never
    written back."""
    from io import BytesIO

    from docx import Document

    from test_document_fidelity import _add_hyperlink, _docx_bytes, _output_xml

    sent: list[str] = []

    def fake_translate_text(text, target_language, correlation_id=None, file_metrics=None):
        sent.append(text)
        return {"See": "Ver", "the documentation": "la documentación",
                "for details.": "para más detalles."}.get(text, text)

    monkeypatch.setattr(backend, "translate_text", fake_translate_text)

    def build(doc):
        para = doc.add_paragraph()
        para.add_run("See ")
        _add_hyperlink(para, "https://example.com/docs", "the documentation")
        para.add_run(" for details.")

    out_stream, _count, _tokens, _text, _segs = backend.process_docx(
        BytesIO(_docx_bytes(build)), target_language="Spanish", do_translate=True
    )

    # The model is still handed trimmed text (a dangling space confuses it)...
    assert "See" in sent and "the documentation" in sent

    xml = _output_xml(out_stream)
    assert xml.count("<w:hyperlink") == 1, "the link element was destroyed"

    out_stream.seek(0)
    text = "".join(p.text for p in Document(out_stream).paragraphs)
    # ...but the DOCUMENT keeps the boundary whitespace.
    assert "Verla" not in text, f"the space before the link was lost in the document: {text!r}"
    assert "Ver la documentación" in text, text
    assert "documentación para más detalles." in text, text
