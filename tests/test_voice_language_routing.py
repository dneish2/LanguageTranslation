"""The Piper pre-fetch must follow the language the USER picked.

Every assertion here is about WHICH LANGUAGE the local voice machinery was
asked for — never that a call succeeded. That distinction is the whole file.
The pre-fetch used to run once per page load with the "Spanish" hardcoded in
`TranslationUI.__init__`, before the "To" input (and its bind_value) existed;
start_ui builds a fresh TranslationUI per page load, so the bound write landed
on an instance thrown away at the next navigation. en_US and es_ES ship on
disk, so `prefetch_voice` no-opped every time and `_fetch_url` was unreachable
in production for every user — a German target got hosted TTS forever, which is
the exact failure DECISIONS.md §3 exists to prevent. Nothing about that was
observable from "the page rendered" or "prefetch_voice was called".

So the tests below drive the REAL construction path: `start_ui`'s page builder,
which calls `new_page_ui().main_page()`. No `TranslationUI.__new__`, and
nothing hand-assigns the attribute under test — that bypass is how the defect
survived a green suite.
"""
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import TranslationUI as ui_module
from TranslationUI import TranslationUI
from passage import engine_ledger, local_voice, policy


# ───────────────────────────── harness ─────────────────────────────────── #

class _PageBuild:
    """One registered page builder, rendered into its OWN nicegui client.

    The client is explicit rather than ambient on purpose. Without it nicegui
    silently falls back to a process-wide "script mode" pseudo-client, and that
    fallback only happens while `Client.instances` is empty — so these tests
    passed alone and failed as soon as any other test file had constructed a
    Client first. A page test whose result depends on which other files ran is
    not evidence about the page.
    """

    def __init__(self, func):
        self.func = func
        self.client = None

    def __call__(self):
        from nicegui.client import Client
        self.client = Client(lambda: None, request=None)
        with self.client:
            return self.func()


class _Pages:
    """`start_ui`'s registered page builders, plus every TranslationUI they built."""

    def __init__(self, routes: dict, built: list):
        self.routes, self.built = routes, built

    def __getitem__(self, path: str) -> _PageBuild:
        return _PageBuild(self.routes[path])

    @property
    def current(self) -> TranslationUI:
        """The instance the most recent page load built — NOT one the test made.

        Found this way on purpose: a test that constructs its own TranslationUI
        cannot notice that page loads discard theirs, which is exactly the
        defect. `built` is populated by subclassing the production class, so
        every line of __init__ and main_page still runs unmodified.
        """
        assert self.built, "no page has been rendered yet"
        return self.built[-1]


def _page_builders(monkeypatch) -> _Pages:
    """Run the REAL `start_ui` and hand back its registered page builders.

    Only the process-level side effects are stubbed (ui.run, route/static
    registration). Everything that matters — one shared backend, a FRESH
    TranslationUI per page load — is the production code, because the
    freshness is precisely what broke the pre-fetch.
    """
    routes: dict = {}
    built: list = []

    class _Recording(TranslationUI):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            built.append(self)

    monkeypatch.setattr(ui_module, "TranslationUI", _Recording)
    monkeypatch.setattr(ui_module.ui, "run", lambda *a, **k: None)
    monkeypatch.setattr(ui_module.ui, "page",
                        lambda path, **k: (lambda func: routes.setdefault(path, func) or func))
    monkeypatch.setattr(ui_module.app, "add_api_route", lambda *a, **k: None)
    monkeypatch.setattr(ui_module.app, "add_static_files", lambda *a, **k: None)
    ui_module.start_ui()
    return _Pages(routes, built)


def _record_prefetch(monkeypatch) -> list:
    """Capture the LANGUAGE the app asks for, at the seam the UI calls."""
    asked: list = []
    monkeypatch.setattr(ui_module.local_voice, "prefetch_voice",
                        lambda language: asked.append(language))
    return asked


def _rendered_texts(build: "_PageBuild") -> list[str]:
    """Every label text a page produced, via its own client element registry."""
    build()
    texts = []
    for element in build.client.elements.values():
        text = getattr(element, "text", None)
        if isinstance(text, str) and text:
            texts.append(text)
    return texts


def _fake_voice_stack(monkeypatch, tmp_path, *, installed=("en_US-lessac-medium",)):
    """A real VOICE_DIR on disk with exactly these voices, and piper importable.

    The production probes (`enabled`, `voice_file_for`, `tts_available`) are
    left alone — they are what is under test. Only the `piper` import and the
    network are faked.
    """
    monkeypatch.setenv("PASSAGE_LOCAL_VOICE", "1")
    monkeypatch.setitem(sys.modules, "piper", types.ModuleType("piper"))
    voice_dir = tmp_path / "piper"
    voice_dir.mkdir(parents=True, exist_ok=True)
    for stem in installed:
        (voice_dir / f"{stem}.onnx").write_bytes(b"onnx-weights")
        (voice_dir / f"{stem}.onnx.json").write_text('{"sample_rate": 22050}', encoding="utf-8")
    monkeypatch.setattr(local_voice, "VOICE_DIR", voice_dir)
    return voice_dir


# ─────────────── (a) the prefetch follows the chosen language ───────────── #

def test_page_load_prefetches_the_language_the_user_is_actually_targeting(monkeypatch):
    """Three page loads, three different targets, three different requests.

    The auditor's probe over this same path returned ['Spanish', 'Spanish',
    'Spanish']. If the prefetch ever goes back to reading the __init__ default,
    this fails on the second load.
    """
    asked = _record_prefetch(monkeypatch)
    pages = _page_builders(monkeypatch)

    pages["/"]()                  # visitor 1 leaves the default alone
    pages["/"]()                  # visitor 2: a brand new instance, per start_ui
    assert asked == ["Spanish", "Spanish"]

    # Visitor 2 now picks a language. The change has to reach the instance the
    # CURRENT page owns — the write that used to land on a discarded one.
    pages.current.target_language_input.value = "German"
    assert asked[-1] == "German", asked

    # And a third page load starts clean again rather than inheriting it.
    pages["/"]()
    pages.current.target_language_input.value = "Japanese"
    assert asked[-1] == "Japanese", asked


def test_changing_the_target_language_requests_that_voice(monkeypatch):
    """A change, not just the initial value. Someone who loads the page and
    then types "Japanese" must get the Japanese voice fetched — this is the
    only signal there is that the fetch-on-demand path is reachable at all."""
    asked = _record_prefetch(monkeypatch)
    pages = _page_builders(monkeypatch)
    pages["/"]()
    page_ui = pages.current

    page_ui.target_language_input.value = "German"
    page_ui.target_language_input.value = "Japanese"

    assert "German" in asked and "Japanese" in asked
    # And the bound attribute really did move with it, i.e. the language the
    # rest of the app will translate into is the same one we fetched for.
    assert page_ui.current_target_language == "Japanese"


def test_repeated_keystrokes_do_not_re_request_the_same_language(monkeypatch):
    asked = _record_prefetch(monkeypatch)
    pages = _page_builders(monkeypatch)
    pages["/"]()
    page_ui = pages.current

    page_ui.target_language_input.value = "German"
    page_ui.target_language_input.value = "Japanese"
    page_ui.target_language_input.value = "German"

    assert asked.count("German") == 1


def test_swapping_languages_requests_the_new_target(monkeypatch):
    """⇄ changes what the user is translating into just as much as typing."""
    asked = _record_prefetch(monkeypatch)
    pages = _page_builders(monkeypatch)
    pages["/"]()
    page_ui = pages.current
    page_ui.source_language_input.value = "French"
    page_ui.target_language_input.value = "German"
    asked.clear()

    page_ui.swap_languages()

    assert "French" in asked


# ───── (b) the fetch path is REACHABLE in production, for a real language ── #

def test_a_page_load_targeting_german_actually_reaches_the_download(monkeypatch, tmp_path):
    """End to end through the production seam, with only the socket faked.

    `prefetch_voice` is NOT stubbed here: the page load has to drive the real
    `prefetch_voice` -> `ensure_voice` -> `_fetch_url`, for de_DE, with a voice
    directory that (like the shipped image) holds only en_US. Before the fix
    `_fetch_url` was never called in production by anybody, so a test that
    stopped at "prefetch_voice was called" would have passed on the broken code.
    """
    _fake_voice_stack(monkeypatch, tmp_path, installed=("en_US-lessac-medium",))
    urls: list[str] = []

    def fake_fetch(url, destination, *, timeout):
        urls.append(url)
        Path(destination).write_bytes(b"onnx-weights")

    monkeypatch.setattr(local_voice, "_fetch_url", fake_fetch)

    pages = _page_builders(monkeypatch)
    pages["/"]()
    pages.current.target_language_input.value = "German"

    deadline = time.time() + 10
    while time.time() < deadline and not any("de_DE" in url for url in urls):
        time.sleep(0.05)

    assert any("de_DE-thorsten-medium.onnx" in url for url in urls), urls
    assert local_voice.voice_installed("German") is True
    # And the page never waited on it: the download ran on its own thread.
    assert local_voice.tts_available("German") is True


def test_english_default_page_load_does_not_download_a_bundled_voice(monkeypatch, tmp_path):
    """The other half: an installed voice must not re-fetch 63MB per page."""
    _fake_voice_stack(monkeypatch, tmp_path,
                      installed=("en_US-lessac-medium", "es_ES-davefx-medium"))
    urls: list[str] = []
    monkeypatch.setattr(local_voice, "_fetch_url",
                        lambda url, destination, *, timeout: urls.append(url))

    pages = _page_builders(monkeypatch)
    pages["/"]()
    time.sleep(0.2)

    assert urls == []


# ──────── (c) /engines must not promise local playback it can't give ─────── #

def test_engines_page_does_not_claim_local_playback_for_uninstalled_voices(
        monkeypatch, tmp_path):
    """/engines is rendered by a fresh instance that CANNOT know the visitor's
    target language, so it must describe capability, never guarantee both-local.

    It previously read the always-"Spanish" default, found the bundled es_ES
    voice, and told a visitor targeting German that "recording and playback
    both run on this machine" while their translated text went to the hosted
    voice. Same class of bug as printing "hosted — metered" above "100% stayed
    on this machine".
    """
    _fake_voice_stack(monkeypatch, tmp_path,
                      installed=("en_US-lessac-medium", "es_ES-davefx-medium"))
    monkeypatch.setattr(local_voice, "stt_available", lambda: True)
    engines = _page_builders(monkeypatch)["/engines"]

    texts = _rendered_texts(engines)
    voice_lines = [t for t in texts if "machine" in t and "ecording" in t]

    assert voice_lines, texts
    line = voice_lines[0]
    assert "playback both run on this machine" not in line
    # It has to name what IS installed — "some languages are hosted" with no
    # list is not something a user can act on.
    assert "en_US-lessac-medium" in line and "es_ES-davefx-medium" in line
    # German has no voice here; nothing on the page may suggest otherwise.
    assert local_voice.tts_available("German") is False
    assert "German" not in line


def test_engines_voice_block_is_rendered_from_voice_state(monkeypatch, tmp_path):
    """One probe feeds the page. `voice_state()` used to have no caller but the
    tests, /engines assembled its own voice block from two other sources, and
    the contradiction it was written to prevent duly reappeared there."""
    _fake_voice_stack(monkeypatch, tmp_path)
    sentinel = {
        "mode": "auto", "engine": "local:probe", "is_local": True,
        "detail": "DETAIL-FROM-VOICE-STATE", "forced_but_missing": False,
        "voices": ["xx_XX"], "privacy": "PRIVACY-FROM-VOICE-STATE",
    }
    monkeypatch.setattr(engine_ledger, "voice_state", lambda *a, **k: sentinel)
    engines = _page_builders(monkeypatch)["/engines"]

    texts = _rendered_texts(engines)

    assert "PRIVACY-FROM-VOICE-STATE" in texts
    assert "DETAIL-FROM-VOICE-STATE" in texts


def test_voice_state_privacy_agrees_with_what_tts_would_do(monkeypatch, tmp_path):
    """The helper's own contract, since the page now depends on it."""
    _fake_voice_stack(monkeypatch, tmp_path, installed=("es_ES-davefx-medium",))
    monkeypatch.setattr(local_voice, "stt_available", lambda: True)

    row = engine_ledger.voice_state()
    assert "es_ES-davefx-medium" in row["privacy"]
    assert "playback both run on this machine" not in row["privacy"]

    # With a language in hand the answer may be specific — and must match the
    # probe the synthesiser itself uses.
    assert local_voice.tts_available("Spanish") is True
    assert "both run on this machine" in policy.describe_voice_privacy("Spanish")
    assert local_voice.tts_available("German") is False
    assert "sent out to be spoken" in policy.describe_voice_privacy("German")


def test_no_local_speech_means_no_local_claim(monkeypatch, tmp_path):
    _fake_voice_stack(monkeypatch, tmp_path)
    monkeypatch.setattr(local_voice, "stt_available", lambda: False)
    assert policy.describe_voice_privacy() == \
        "Recordings are sent to Passage's hosted speech models."
    assert policy.describe_voice_privacy("Spanish") == \
        "Recordings are sent to Passage's hosted speech models."
