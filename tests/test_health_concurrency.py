"""/api/health, /diagnostics and /engines must not freeze the server.

Measured defect (independent audit, this branch): a single unauthenticated
GET /api/health against an Ollama endpoint that accepts the connection and
then says nothing took 15.31 s, issued FOUR outbound probes, and a heartbeat
task ticking every 50 ms recorded ZERO ticks for the whole 15.31 s. The route
was an ``async def`` calling four sequential blocking urllib probes directly
on the event loop, so every other connected client stalled with it. Against a
refused port: 8.32 s.

The governing question here (PORTABILITY_PLAN §2.2, restated for concurrency):
*would this test fail if the fix were reverted?* A test that asserts
``/api/health`` returns 200 would not — it returned 200 before, fifteen
seconds later. So these tests assert the MECHANISM:

  1. a concurrent heartbeat task keeps ticking WHILE a health request against
     a dead endpoint is in flight (the loop is not blocked), and
  2. the request costs ONE outbound probe, not four (no regression to
     probe-per-consumer),

and they do it against a REAL socket that behaves like the failure being
defended against — a real listening-but-silent server and a real refused port
— through the real ``TranslationUI(backend=...)`` construction path and the
real ``api_health`` coroutine. No ``__new__``, no assigning the attribute
under test, no stubbing of the route itself.
"""

import asyncio
import json
import socket
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import TranslationUI as ui_module


# ───────────────────────── real, hostile endpoints ───────────────────── #

class SilentServer:
    """A socket that ACCEPTS the connection and then never answers.

    This is the worst case and the one the auditor measured: a refused port
    fails fast-ish, a silent one burns the whole probe timeout. Nothing is
    faked — urllib really connects and really waits.
    """

    def __init__(self) -> None:
        self._sock = socket.socket()
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(8)
        self.port = self._sock.getsockname()[1]
        self._held: list[socket.socket] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._accept_forever, daemon=True)
        self._thread.start()

    def _accept_forever(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except OSError:
                return
            self._held.append(conn)  # held open, deliberately unanswered

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def close(self) -> None:
        self._stop.set()
        try:
            self._sock.close()
        except OSError:
            pass
        for conn in self._held:
            try:
                conn.close()
            except OSError:
                pass


def _refused_port() -> int:
    """A port with nothing on it: bound, read back, then closed."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


class CountingBackend:
    """The production probe against a hostile endpoint, with a call counter.

    ``probe_local_llm`` here is the REAL TranslationBackend function, pointed
    at the test's socket. That matters: a fake sleep would prove nothing about
    urllib blocking the loop, which is the actual defect.
    """

    provider = None

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url
        self.probe_calls = 0
        self._lock = threading.Lock()

    def probe_local_llm(self):
        from TranslationBackend import probe_local_llm

        with self._lock:
            self.probe_calls += 1
        return probe_local_llm(base_url=self.base_url)

    # Present so nothing falls back to them silently: if the route ever calls
    # these again, the counter jumps and the probe-count test fails.
    def available_local_models(self):
        return [m for m in self.probe_local_llm()["models"] if "embed" not in m]

    def choose_local_model(self):
        return (self.probe_local_llm()["models"] or [None])[0]


@pytest.fixture()
def fast_probe(monkeypatch):
    """Keep the probe honest but the suite quick.

    The timeouts are lowered, NOT removed: a 0.4 s wait is still a real wait
    on a real socket, and a blocked event loop shows up just as clearly at
    0.4 s as at 2.5 s (the heartbeat simply misses fewer ticks). Production
    values would make this file take a minute.
    """
    import TranslationBackend as backend_module

    monkeypatch.setattr(backend_module, "LIVE_PROBE_TIMEOUT_SECONDS", 0.4)
    monkeypatch.setattr(backend_module, "PROBE_REFUSAL_CHECK_SECONDS", 0.3)
    # Never reuse a snapshot across tests: each one must pay for its own probe.
    monkeypatch.setattr(ui_module, "LIVE_PROBE_TTL_SECONDS", 60.0)


# ─────────────────────────── the mechanism ───────────────────────────── #

async def _health_with_heartbeat(backend, interval: float = 0.05):
    """Run /api/health while a heartbeat task ticks every `interval`.

    Returns (payload, ticks, elapsed_seconds). The tick count is the whole
    point: it is a direct measurement of whether the event loop was free to
    run anything else during the request.
    """
    ui_app = ui_module.TranslationUI(backend=backend)
    ticks = 0
    stop = False

    async def heartbeat():
        nonlocal ticks
        while not stop:
            await asyncio.sleep(interval)
            ticks += 1

    beat = asyncio.create_task(heartbeat())
    await asyncio.sleep(interval)  # let the heartbeat get going
    before = ticks
    started = time.perf_counter()
    response = await ui_app.api_health(None)
    elapsed = time.perf_counter() - started
    during = ticks - before
    stop = True
    beat.cancel()
    try:
        await beat
    except asyncio.CancelledError:
        pass
    return json.loads(bytes(response.body).decode("utf-8")), during, elapsed


def test_health_against_a_silent_endpoint_leaves_the_loop_responsive(fast_probe):
    """The regression itself. Before the fix this recorded ZERO ticks while
    the loop sat inside urllib; a health check that stops the world is a
    denial of service on an unauthenticated route."""
    server = SilentServer()
    try:
        backend = CountingBackend(server.base_url)
        payload, ticks, elapsed = asyncio.run(_health_with_heartbeat(backend))
    finally:
        server.close()

    assert payload["status"] == "ok"
    # The probe really did have to wait — otherwise this proves nothing.
    assert elapsed > 0.3, f"probe returned too fast to be a real wait ({elapsed:.3f}s)"
    # …and the loop kept serving through it. Expected ticks ≈ elapsed/0.05.
    assert ticks >= 3, (
        f"event loop blocked: {ticks} heartbeat ticks in {elapsed:.2f}s "
        "— blocking I/O is back on the loop")
    # And it stayed unreachable-but-honest rather than pretending.
    assert payload["local_models"] == []
    assert payload["local_probe"]["outcome"] in {"timeout", "refused", "error"}


def test_health_against_a_refused_port_leaves_the_loop_responsive(fast_probe):
    """The second endpoint the auditor measured (8.32 s, loop frozen)."""
    port = _refused_port()
    backend = CountingBackend(f"http://127.0.0.1:{port}")
    payload, ticks, elapsed = asyncio.run(_health_with_heartbeat(backend))

    assert payload["status"] == "ok"
    assert ticks >= 2, (
        f"event loop blocked: {ticks} heartbeat ticks in {elapsed:.2f}s")


def test_health_costs_exactly_one_probe(fast_probe):
    """Four consumers, one probe. The route used to probe reachability, then
    collect diagnostics (which probed again), then choose a model (twice
    more), multiplying every second of an unreachable endpoint by four."""
    port = _refused_port()
    backend = CountingBackend(f"http://127.0.0.1:{port}")
    asyncio.run(ui_module.TranslationUI(backend=backend).api_health(None))
    assert backend.probe_calls == 1, (
        f"{backend.probe_calls} probes for one health request — the "
        "single-probe snapshot regressed")


def test_a_second_health_request_reuses_the_probe_and_says_how_old_it_is(fast_probe):
    """Reuse is only acceptable if the age is reported: a cache that hides its
    age turns a diagnostic into a confident guess, which is worse than a slow
    one."""
    port = _refused_port()
    backend = CountingBackend(f"http://127.0.0.1:{port}")
    ui_app = ui_module.TranslationUI(backend=backend)

    first = json.loads(bytes(asyncio.run(ui_app.api_health(None)).body).decode())
    second = json.loads(bytes(asyncio.run(ui_app.api_health(None)).body).decode())

    assert backend.probe_calls == 1, "the cached probe was not reused"
    assert "age_seconds" in second["local_probe"]
    assert second["local_probe"]["age_seconds"] >= first["local_probe"]["age_seconds"]
    assert second["local_probe"]["ttl_seconds"] > 0
    # The diagnostics section carries the same age, so /diagnostics cannot
    # print a stale answer as a live one.
    assert "age_seconds" in second["diagnostics"]["local_llm"]


def test_an_expired_snapshot_is_re_probed_rather_than_served_forever(
        fast_probe, monkeypatch):
    """The other half of caching: a stale answer must not outlive its TTL."""
    monkeypatch.setattr(ui_module, "LIVE_PROBE_TTL_SECONDS", 0.0)
    port = _refused_port()
    backend = CountingBackend(f"http://127.0.0.1:{port}")
    ui_app = ui_module.TranslationUI(backend=backend)
    asyncio.run(ui_app.api_health(None))
    asyncio.run(ui_app.api_health(None))
    assert backend.probe_calls == 2, "an expired snapshot was served anyway"


def test_concurrent_health_requests_do_not_serialise_the_server(fast_probe):
    """Two clients hitting the unauthenticated route at once. Before the fix
    they queued behind each other on the loop; the second waited out the
    first's full probe before its own even started."""
    server = SilentServer()

    async def scenario():
        backend = CountingBackend(server.base_url)
        app_a = ui_module.TranslationUI(backend=backend)
        app_b = ui_module.TranslationUI(backend=backend)
        started = time.perf_counter()
        await asyncio.gather(app_a.api_health(None), app_b.api_health(None))
        return time.perf_counter() - started

    try:
        elapsed = asyncio.run(scenario())
    finally:
        server.close()

    # Two full serial probes would be ~2x the single-probe cost (0.4s timeout
    # + 0.3s refusal check each). Overlapped, they cost roughly one.
    assert elapsed < 1.2, (
        f"two concurrent health requests took {elapsed:.2f}s — they are "
        "still serialising on the event loop")


# ───────────────────────── the page renders ──────────────────────────── #

def test_page_render_helpers_never_probe_on_the_render_path(fast_probe):
    """/diagnostics and /engines build from the CACHED snapshot and refresh
    from a worker thread. The render path itself must issue no probe — those
    pages measured 7.81 s and 7.56 s, blocking every other client."""
    port = _refused_port()
    backend = CountingBackend(f"http://127.0.0.1:{port}")

    started = time.perf_counter()
    snapshot = ui_module._cached_snapshot(backend) or ui_module._pending_snapshot()
    report = ui_module.collect_diagnostics(snapshot)
    elapsed = time.perf_counter() - started

    assert backend.probe_calls == 0, "the render path probed"
    assert elapsed < 0.2, f"render path blocked for {elapsed:.2f}s"
    # Pending is reported as pending, not as "unreachable": "we have not asked
    # yet" and "we asked and nothing answered" are different facts.
    assert snapshot["pending"] is True
    assert report["local_llm"]["outcome"] == "probing"
    assert "probing" in ui_module.describe_snapshot_age(snapshot)


def test_a_landed_probe_is_labelled_with_its_age_not_presented_as_live(fast_probe):
    port = _refused_port()
    backend = CountingBackend(f"http://127.0.0.1:{port}")
    snapshot = ui_module._take_local_snapshot(backend)
    assert "just now" in ui_module.describe_snapshot_age(snapshot)

    snapshot["probed_at"] -= 12.0
    text = ui_module.describe_snapshot_age(snapshot)
    assert "12s ago" in text, text
