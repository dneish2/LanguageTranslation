"""Suite-wide defaults, and the isolation that makes the suite order-independent.

Tests run as a Cloud deployment unless they opt into Local. Local-first routing
probes a real Ollama on this machine, so without this default the same test
passes or fails depending on whether a developer happens to have Ollama running
(the dev box does, CI does not). A test about local routing says so with
``@pytest.mark.local_deployment``, or replaces ``_live_local_provider`` outright.

The rest of this file exists because a suite whose result depends on which other
tests ran is not evidence. Three process-global latches were deciding outcomes
by collection order; see ``_isolate_nicegui`` and ``_isolate_voice_latches``.
"""
import pytest
from nicegui import core
from nicegui.client import Client
from nicegui.slot import Slot

import TranslationBackend as tb
from passage import local_voice


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "local_deployment: run as a Local deployment (local-first routing on)")


@pytest.fixture(autouse=True)
def _deployment_mode(request, monkeypatch):
    local = request.node.get_closest_marker("local_deployment") is not None
    monkeypatch.setattr(tb, "DEPLOYMENT_MODE", "local" if local else "cloud")


# ─────────────────── nicegui: no latch survives a test ────────────────────── #

@pytest.fixture(autouse=True)
def _isolate_nicegui():
    """Give every test pristine nicegui globals.

    WHY THIS IS NOT OPTIONAL. `nicegui/context.py` creates a process-global
    pseudo-client the first time anything renders UI without an explicit
    `Client`, and both of its conditions are one-way doors:

        if not stack and not core.script_mode and not core.app.is_started:
            if not Client.instances:          # never shrinks: nothing calls
                core.script_mode = True       # delete_client in a test
                core.script_client = Client(page('/')).__enter__()

    So the FIRST client-less render in the process latches `script_mode`, leaves
    a slot on `Slot.stacks` that is never popped, and every later client-less
    render in the run silently rides on it. Run a test that builds a real
    `Client` first instead and `Client.instances` is non-empty forever, the
    pseudo-client is never created, and the client-less renders in
    `test_surface_coverage.py` fail with "The current slot cannot be determined"
    — an error that blames a background task and points nowhere near the cause.

    That was the real state of this suite: green only because
    `test_api_security.py` sorts first alphabetically and renders client-less.
    Measured before this fixture existed, same commit, same machine:

        alphabetical order   656 passed
        reverse order        647 passed, 9 errors
        3 random orders      2 of 3 failed

    Renaming files would have preserved that luck rather than removed it.

    `core.reset()` is nicegui's own helper for this ("Reset core variables.
    (Useful for testing.)"). The library also ships the much broader
    `nicegui.testing.general.nicegui_reset_globals()`, deliberately NOT used
    here, on two measurements: it costs 143s against this suite's 35s (a
    per-test `gc.collect()` plus a walk of every `ui.element` subclass), and
    alone it is insufficient anyway — it leaves `Slot.stacks` untouched, so the
    stale slot outlives the element it points at and the failure merely changes
    to "The parent element this slot belongs to has been deleted". These three
    lines are both cheaper and more correct: 656 passed in every order tried, at
    no measurable cost.

    `Slot.stacks` is a private ClassVar, so a nicegui upgrade could move it. The
    reverse-order job in CI is what catches that loudly, instead of the suite
    quietly going back to being order-dependent.
    """
    def clean():
        Slot.stacks.clear()
        Client.instances.clear()
        core.reset()

    clean()
    yield
    clean()


# ───────────── local voice: no latch, and no thread outliving its test ────── #

@pytest.fixture(autouse=True)
def _isolate_voice_latches():
    """Reset the single-flight guards in `local_voice`.

    Same bug class as `_isolate_nicegui`, one module over. `prefetch_whisper`
    sets `_whisper_prefetch_started = True` and never unsets it, so the first
    test to reach it turns the pre-fetch into a permanent no-op for every test
    that follows — and `_download_locks` accumulates a lock per voice stem, keyed
    to directories that no longer exist. Only `test_stt_readiness.py` resets
    these, for itself. Doing it here means no test inherits another test's
    single-flight decision.
    """
    def clean():
        local_voice._whisper_prefetch_started = False
        local_voice._whisper_prefetch_lock = None
        local_voice._download_locks.clear()

    clean()
    yield
    clean()


@pytest.fixture(autouse=True)
def _join_voice_prefetches(monkeypatch):
    """Let a test's background downloads finish BEFORE its patches come off.

    `prefetch_voice` is deliberately fire-and-forget, and `ensure_voice` reads
    `VOICE_DIR` when it reaches the lock — not when the thread was started. A
    thread still running after `monkeypatch` restores `VOICE_DIR` therefore
    writes its download into the REAL `models/piper/`, and since the test's
    `_fetch_url` stub writes a 12-byte placeholder, that installs a voice
    `installed_voices()` reports as present and `synthesize()` then fails on —
    exactly the silent-failure shape this module's probes exist to prevent.

    Not hypothetical. The CI run that first went red here printed
    `[LocalVoice] voice ready: es_ES-davefx-medium` from its TEARDOWN phase.

    The ordering is the entire mechanism, and it is why this fixture asks for
    `monkeypatch` without using it: a fixture is finalised BEFORE the fixtures it
    depends on, so requesting `monkeypatch` guarantees this teardown runs while
    the patches are still in place. Verified, not assumed.

    This replaces the fixed sleeps that stood in for it — `test_piper_fetch.py`
    had `time.sleep(0.1)  # give any daemon fetch thread the chance to be
    caught`, which is a guess about a race rather than an end to it.
    """
    yield
    still_running = local_voice.wait_for_prefetches(timeout=10.0)
    assert not still_running, (
        "a local-voice pre-fetch thread outlived its test and may now be writing "
        f"into the real VOICE_DIR: {still_running}"
    )


# ─────────────────────── rendering pages, the one way ─────────────────────── #

class PageBuild:
    """One page builder, rendered into its OWN nicegui client.

    The client is explicit rather than ambient on purpose. Without it nicegui
    falls back to the process-wide pseudo-client described in
    `_isolate_nicegui` — shared state, so a page test that rides on it is a page
    test whose result depends on what else ran.

    Promoted here from `test_voice_language_routing.py`, which worked this out
    for its own file first. New page tests should use this rather than
    rediscovering the fallback.
    """

    def __init__(self, func):
        self.func = func
        self.client = None

    def __call__(self):
        self.client = Client(lambda: None, request=None)
        with self.client:
            return self.func()

    def texts(self) -> list:
        """Every non-empty label text this page produced, from its own registry."""
        self()
        found = []
        for element in self.client.elements.values():
            text = getattr(element, "text", None)
            if isinstance(text, str) and text:
                found.append(text)
        return found
