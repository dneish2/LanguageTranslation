"""R2: the model's own stream reaches the page, and a superseded request stops.

Every fake here stamps what it did (pieces fed, calls made, streams closed),
so a test fails if the stream was faked after the fact, if a cancelled call
kept generating, or if a second engine was spliced into a first one's answer.
"""
import asyncio
import json
import threading
import time
import types

import pytest

import TranslationBackend as tb
from passage import streaming
from passage.streaming import StreamCancelled, StreamSink
from TranslationUI import TranslationUI


def _request(ui_app, disconnected=False):
    async def is_disconnected():
        return disconnected

    return types.SimpleNamespace(
        headers={"x-passage-token": ui_app.api_guard.issue_token()},
        client=types.SimpleNamespace(host="127.0.0.1"),
        is_disconnected=is_disconnected,
    )


def _events(resp) -> list[tuple[str, dict]]:
    async def collect():
        parts = []
        async for part in resp.body_iterator:
            parts.append(part.decode() if isinstance(part, bytes) else part)
        return "".join(parts)

    out = []
    for block in asyncio.run(collect()).split("\n\n"):
        lines = block.split("\n")
        kind = next((l[7:] for l in lines if l.startswith("event: ")), None)
        data = next((l[6:] for l in lines if l.startswith("data: ")), None)
        if kind:
            out.append((kind, json.loads(data) if data else {}))
    return out


def _feeding(*pieces, result=None):
    """A model stand-in that writes `pieces` into the installed sink."""
    def fake(*_a, **_k):
        sink = streaming.current()
        assert sink is not None, "no stream sink installed: the endpoint is not streaming"
        for piece in pieces:
            sink.feed(piece)
            time.sleep(0.02)
        return result if result is not None else "".join(pieces)
    return fake


# ── the endpoints ────────────────────────────────────────────────────────

def test_stream_endpoint_sends_the_models_own_partials_before_complete(monkeypatch):
    """It used to wait for the whole answer, then slice it into 80-character
    'partials': nothing arrived any sooner than without streaming."""
    monkeypatch.setenv("LIVE_TEXT_STREAMING", "true")
    ui_app = TranslationUI()
    monkeypatch.setattr(ui_app.backend, "translate_text", _feeding("hola", " mundo"))

    events = _events(asyncio.run(ui_app.api_text_translate_stream(
        _request(ui_app), text="hello world", language="es")))

    kinds = [k for k, _ in events]
    assert kinds[0] == "start" and kinds[-1] == "complete"
    partials = [d["translated_text"] for k, d in events if k == "partial"]
    assert partials[0] == "hola" and partials[-1] == "hola mundo"
    assert events[-1][1]["translated_text"] == "hola mundo"


def test_stream_endpoint_reports_a_failure_as_an_error_event(monkeypatch):
    monkeypatch.setenv("LIVE_TEXT_STREAMING", "true")
    ui_app = TranslationUI()

    def boom(*_a, **_k):
        raise RuntimeError("boom")

    monkeypatch.setattr(ui_app.backend, "translate_text", boom)
    events = _events(asyncio.run(ui_app.api_text_translate_stream(
        _request(ui_app), text="hello world", language="es")))

    assert events[-1][0] == "error" and "boom" in events[-1][1]["error"]


def test_live_endpoint_streams_and_completes_with_the_same_receipt(monkeypatch):
    ui_app = TranslationUI()
    runs: list = []
    monkeypatch.setattr(type(ui_app), "engine_runs", property(lambda self: runs))
    monkeypatch.setattr(type(ui_app), "usage_store", property(lambda self: {}))
    monkeypatch.setattr(ui_app.backend, "_live_local_provider", lambda: None)
    monkeypatch.setattr(ui_app.backend, "_require_provider",
                        lambda: types.SimpleNamespace(max_input_chars=None))
    calls = []

    def chunk(text, language, *a, **k):
        calls.append(text)
        tb._note_engine(f"hosted:{tb.TEXT_MODEL}")
        return _feeding("La reunión", " se movió.")()

    monkeypatch.setattr(ui_app.backend, "_translate_chunk", chunk)

    events = _events(asyncio.run(ui_app.api_text_translate_live(
        _request(ui_app), text="The meeting moved.", language="Spanish")))

    assert [d["translated_text"] for k, d in events if k == "partial"][0] == "La reunión"
    kind, receipt = events[-1]
    assert kind == "complete" and receipt["translated_text"] == "La reunión se movió."
    assert receipt["engine"] == f"hosted:{tb.TEXT_MODEL}"
    assert receipt["metered"] is True and receipt["left_machine"] is True
    assert len(calls) == 1 and len(runs) == 1


# ── providers stream through the one egress method ───────────────────────

class _FakeStream:
    def __init__(self, pieces):
        self.pieces = pieces
        self.consumed = 0
        self.closed = False

    def __iter__(self):
        for piece in self.pieces:
            self.consumed += 1
            yield types.SimpleNamespace(choices=[types.SimpleNamespace(
                delta=types.SimpleNamespace(content=piece))])

    def close(self):
        self.closed = True


def _hosted_with(stream):
    provider = tb.ChatCompletionsProvider(api_key="k", text_model="gpt-test")
    seen = {}

    def create(**kwargs):
        seen.update(kwargs)
        return stream

    provider.client = types.SimpleNamespace(chat=types.SimpleNamespace(
        completions=types.SimpleNamespace(create=create)))
    return provider, seen


def test_hosted_provider_streams_into_the_sink_and_still_returns_the_whole_answer():
    stream = _FakeStream(["Hola", " mundo"])
    provider, seen = _hosted_with(stream)
    shown = []

    with streaming.streaming(StreamSink(shown.append)):
        completion = provider.create_chat_completion(messages=[], max_tokens=10)

    assert seen["stream"] is True
    assert shown == ["Hola", "Hola mundo"]
    assert completion.choices[0].message.content == "Hola mundo"
    assert stream.closed


def test_cancel_closes_the_hosted_stream_and_stops_reading_it():
    """Closing the response is what stops output tokens being generated and
    billed; a cancelled call that kept reading would still pay for them."""
    stream = _FakeStream(["Uno", " dos", " tres", " cuatro"])
    provider, _ = _hosted_with(stream)
    cancel = threading.Event()

    def on_text(text):
        cancel.set()  # the next keystroke arrives after the first piece

    with pytest.raises(StreamCancelled), streaming.streaming(StreamSink(on_text, cancel)):
        provider.create_chat_completion(messages=[], max_tokens=10)

    assert stream.closed and stream.consumed == 2


def test_without_a_sink_nothing_streams():
    """Documents and every other caller are untouched."""
    whole = types.SimpleNamespace(choices=[types.SimpleNamespace(
        message=types.SimpleNamespace(content="Hola"))])
    provider, seen = _hosted_with(whole)

    assert provider.create_chat_completion(messages=[], max_tokens=10) is whole
    assert "stream" not in seen


def test_native_ollama_streams_and_drops_thinking(monkeypatch):
    lines = [json.dumps(x).encode() + b"\n" for x in (
        {"message": {"thinking": "let me think"}},
        {"message": {"content": "Hola"}},
        {"message": {"content": " mundo"}},
        {"done": True, "message": {"content": ""}},
    )]

    class Response:
        def __init__(self):
            self.closed = False

        def __iter__(self):
            return iter(lines)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.closed = True

    response = Response()
    sent = {}

    def urlopen(request, timeout=None):
        sent.update(json.loads(request.data))
        return response

    monkeypatch.setattr("passage.ollama_native.urllib.request.urlopen", urlopen)
    provider = tb.NativeOllamaProvider(base_url="http://127.0.0.1:11434/v1", text_model="translategemma:4b")
    shown = []

    with streaming.streaming(StreamSink(shown.append)):
        completion = provider.create_chat_completion(messages=[{"role": "user", "content": "x"}])

    assert sent["stream"] is True
    assert shown == ["Hola", "Hola mundo"]
    assert completion.choices[0].message.content == "Hola mundo"
    assert response.closed


# ── routing while streaming ──────────────────────────────────────────────

class _Counting:
    text_model = "hosted-fake"

    def __init__(self, answer="Hola desde la nube"):
        self.calls = 0
        self.answer = answer

    def create_chat_completion(self, **_kw):
        self.calls += 1
        return types.SimpleNamespace(choices=[types.SimpleNamespace(
            message=types.SimpleNamespace(content=self.answer))])


class _DiesMidSentence:
    text_model = "translategemma:4b"
    base_url = "http://127.0.0.1:11434/v1"

    def create_chat_completion(self, **_kw):
        streaming.current().feed("Hola")
        raise ConnectionError("local model died mid-sentence")


def test_local_first_does_not_splice_hosted_onto_local_text():
    hosted = _Counting()
    router = tb.LocalFirstProvider(_DiesMidSentence(), hosted)

    with pytest.raises(ConnectionError), streaming.streaming(StreamSink(lambda t: None)):
        router.create_chat_completion(messages=[], max_tokens=10)

    assert hosted.calls == 0


def test_local_first_still_falls_back_before_any_text():
    hosted = _Counting()

    class Refused:
        text_model, base_url = "translategemma:4b", "http://127.0.0.1:11434/v1"

        def create_chat_completion(self, **_kw):
            raise ConnectionError("refused")

    shown = []
    router = tb.LocalFirstProvider(Refused(), hosted)
    with streaming.streaming(StreamSink(shown.append)):
        completion = router.create_chat_completion(messages=[], max_tokens=10)

    assert hosted.calls == 1 and completion.choices[0].message.content == "Hola desde la nube"


def test_a_cancelled_live_translation_is_not_cached(monkeypatch):
    """A superseded request's half answer must never be served later as a
    finished translation."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    backend = tb.TranslationBackend()
    calls = []

    def chunk(text, language, *a, **k):
        calls.append(text)
        sink = streaming.current()
        if sink is not None:
            sink.feed("Hola")
            sink.cancel.set()
            sink.check()
        return "Hola mundo"

    monkeypatch.setattr(backend, "_live_local_provider", lambda: None)
    monkeypatch.setattr(backend, "_require_provider", lambda: types.SimpleNamespace(max_input_chars=None))
    monkeypatch.setattr(backend, "_translate_chunk", chunk)

    with pytest.raises(StreamCancelled), streaming.streaming(StreamSink(lambda t: None)):
        backend.translate_live("Hello world", "Spanish")
    assert backend.translate_live("Hello world", "Spanish")[0] == "Hola mundo"
    assert len(calls) == 2


def test_an_abandoned_request_that_reached_a_model_is_still_booked(monkeypatch):
    """Its input was sent (and on a metered API, paid for). Booked as not
    delivered, so the visitor is not charged for an answer they never saw."""
    ui_app = TranslationUI()
    booked = []
    reached = threading.Event()

    def work():
        tb._note_engine(f"hosted:{tb.TEXT_MODEL}")
        reached.set()
        for _ in range(100):
            time.sleep(0.02)
            streaming.current().check()
        return "never"

    async def drive():
        async for _ in ui_app._run_streamed(
                _request(ui_app, disconnected=True), None, work,
                recorder=lambda **kw: booked.append(kw), surface="live_text", chars=11):
            pass
        for _ in range(100):
            if booked:
                break
            await asyncio.sleep(0.02)

    asyncio.run(drive())

    assert reached.is_set()
    assert len(booked) == 1
    assert booked[0]["engine"] == f"hosted:{tb.TEXT_MODEL}" and booked[0]["delivered"] is False


def test_partial_placeholders_never_reach_the_screen():
    shown = []
    sink = StreamSink(shown.append)
    for piece in ["See ", "[[PS", "G:0]]", " now"]:
        sink.feed(piece)

    assert shown == ["See", "See", "See", "See  now"]
    assert all("[[" not in s for s in shown)


def test_the_live_box_script_is_valid_javascript_as_served(tmp_path):
    """The script lives in a non-raw Python string. A '\n' written there is
    a real newline by the time the browser sees it, which is a syntax error:
    the live box then does nothing at all, with no Python error and no test
    failure. passage/live_bench.py caught it (zero requests per sentence)."""
    import shutil
    import subprocess
    from pathlib import Path

    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    rendered = []
    ui_app = TranslationUI()
    import TranslationUI as ui_module
    original = ui_module.ui.add_body_html
    ui_module.ui.add_body_html = rendered.append
    try:
        ui_app._inject_workspace_text_live_translation_js()
    finally:
        ui_module.ui.add_body_html = original
    script = rendered[0].split("<script>", 1)[1].rsplit("</script>", 1)[0]
    path = Path(tmp_path) / "live.js"
    path.write_text(script, encoding="utf-8")

    result = subprocess.run([node, "--check", str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
