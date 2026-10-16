"""The browser half of the mobile work: real WebKit and Chromium at real
device sizes, asserting the properties the 2026-10-15 review measured.

OPT-IN. Skipped unless PASSAGE_MOBILE_E2E=1 and Playwright is importable with
its browsers installed:

    uv pip install playwright && python -m playwright install webkit chromium
    PASSAGE_MOBILE_E2E=1 python -m pytest tests/mobile -q

Kept out of the default suite on purpose. It starts a real server, drives two
browser engines and takes about a minute, which is three times the whole rest
of the suite; and a browser download failing must never be the reason a code
change looks broken. The ordinary suite asserts the same fixes at the source
level in tests/test_mobile_resilience.py, so deleting a fix fails there too.

WHAT THIS CANNOT DO, stated here because the repo's standing rule is that a
tier stays UNVERIFIED until a real run says otherwise: WebKit on a developer
machine is not iOS Safari, Chromium at 412px is not a Pixel, and nothing here
touches a real digitiser, a real on-screen keyboard, VoiceOver, TalkBack, a
real camera or a real cellular handoff. It catches regressions in properties
that are already known to be true. It does not certify a device.
"""
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.skipif(
    os.getenv("PASSAGE_MOBILE_E2E") != "1",
    reason="browser tier is opt-in; set PASSAGE_MOBILE_E2E=1",
)

playwright_api = pytest.importorskip(
    "playwright.sync_api", reason="playwright is not installed",
)

#: name, engine, width, height — the five the review used.
DEVICES = [
    ("iphone-se", "webkit", 375, 667),
    ("iphone-15", "webkit", 393, 852),
    ("pixel-8", "chromium", 412, 915),
    ("ipad-mini", "webkit", 744, 1133),
    ("desktop", "chromium", 1440, 900),
]
ROUTES = ["/", "/voice", "/compare", "/engines", "/diagnostics"]

#: Apple HIG minimum is 44pt; Material is 48dp. 44 is the floor asserted here.
TAP_TARGET_MIN = 44
#: iOS Safari zooms the page when a focused input is smaller than this.
IOS_ZOOM_THRESHOLD = 16


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="module")
def app_url():
    """The real app, started the way production starts it."""
    port = _free_port()
    # No key and no Ollama needed: this tier is about layout and touch, and a
    # page that renders is all it asserts. Which engine would have served a
    # translation is tested elsewhere, properly.
    env = {**os.environ, "PORT": str(port)}
    # nicegui.helpers.is_pytest() is just `'PYTEST_CURRENT_TEST' in os.environ`,
    # and a subprocess inherits it. ui.run() then takes its screen-test branch
    # and dies on a missing NICEGUI_SCREEN_TEST_PORT. This app is the subject
    # of the test, not a participant in it, so it starts as production starts.
    env.pop("PYTEST_CURRENT_TEST", None)
    process = subprocess.Popen(
        [sys.executable, "TranslationUI.py"], cwd=str(ROOT), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    url = f"http://127.0.0.1:{port}"
    deadline = time.time() + 90
    try:
        import urllib.request
        while time.time() < deadline:
            if process.poll() is not None:
                output = (process.stdout.read() or b"").decode("utf-8", "replace")
                pytest.fail(f"the app exited before serving:\n{output[-2000:]}")
            try:
                with urllib.request.urlopen(url, timeout=2):
                    break
            except Exception:
                time.sleep(0.5)
        else:
            pytest.fail(f"the app never answered on {url}")
        yield url
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()


@pytest.fixture(scope="module")
def browsers():
    with playwright_api.sync_playwright() as play:
        engines = {"webkit": play.webkit.launch(), "chromium": play.chromium.launch()}
        try:
            yield engines
        finally:
            for engine in engines.values():
                engine.close()


def _page(browsers, engine, width, height):
    context = browsers[engine].new_context(
        viewport={"width": width, "height": height}, has_touch=width < 700,
    )
    return context


MEASURE = """
() => {
  const vw = window.innerWidth;
  const de = document.documentElement;
  const small = [], tinyInputs = [];
  for (const el of document.querySelectorAll('button, .q-btn, a[href], input, textarea, select')) {
    const r = el.getBoundingClientRect();
    const cs = getComputedStyle(el);
    if (cs.display === 'none' || cs.visibility === 'hidden' || !r.width || !r.height) continue;
    const label = (el.innerText || el.getAttribute('aria-label')
                   || el.getAttribute('placeholder') || el.tagName).trim().slice(0, 30);
    if (r.width < 44 || r.height < 44) {
      small.push(`${Math.round(r.width)}x${Math.round(r.height)} ${label}`);
    }
    if (el.matches('input, textarea, select')
        && parseFloat(cs.fontSize) < 16) {
      tinyInputs.push(`${parseFloat(cs.fontSize)}px ${label}`);
    }
  }
  return {overflowPx: de.scrollWidth - vw, small, tinyInputs};
}
"""


@pytest.mark.parametrize("name,engine,width,height", DEVICES,
                         ids=[d[0] for d in DEVICES])
@pytest.mark.parametrize("route", ROUTES)
def test_no_page_scrolls_sideways(app_url, browsers, name, engine, width, height, route):
    """The layout held at every width the review tried, with one exception:
    /diagnostics printed an unwrapped 1916px code block that overflowed even a
    1440px desktop. Pass 2 wraps it; until then this records the fact."""
    context = _page(browsers, engine, width, height)
    try:
        page = context.new_page()
        page.goto(app_url + route, wait_until="networkidle", timeout=60000)
        page.wait_for_timeout(1500)
        measured = page.evaluate(MEASURE)
        if route == "/diagnostics":
            pytest.xfail("known: the diagnostics code block does not wrap (Pass 2)")
        assert measured["overflowPx"] <= 0, (
            f"{name}{route} scrolls {measured['overflowPx']}px sideways"
        )
    finally:
        context.close()


@pytest.mark.parametrize("name,engine,width,height", DEVICES,
                         ids=[d[0] for d in DEVICES])
def test_no_input_triggers_the_ios_zoom(app_url, browsers, name, engine, width, height):
    """Three inputs on / were 14px, so focusing one zoomed the whole page and
    left the user scrolled sideways inside a layout that never overflowed."""
    context = _page(browsers, engine, width, height)
    try:
        page = context.new_page()
        page.goto(app_url + "/", wait_until="networkidle", timeout=60000)
        page.wait_for_timeout(1500)
        measured = page.evaluate(MEASURE)
        assert not measured["tinyInputs"], (
            f"{name}: inputs under {IOS_ZOOM_THRESHOLD}px: {measured['tinyInputs']}"
        )
    finally:
        context.close()


def test_the_capture_card_is_the_tap_target(app_url, browsers):
    """The dashed card was not tappable at all -- elementFromPoint at its
    centre resolved to .q-uploader__list -- and the only control that opened
    the camera was 32x32, well under either platform's minimum."""
    context = _page(browsers, "webkit", 393, 852)
    try:
        page = context.new_page()
        page.goto(app_url + "/?mode=Image/Camera", wait_until="networkidle", timeout=60000)
        page.wait_for_timeout(2000)
        card = page.locator(".p-capture").first
        box = card.bounding_box()
        assert box, "no capture card rendered"

        button = page.evaluate("""() => {
            const b = document.querySelector('.p-capture .q-uploader__header .q-btn');
            if (!b) return null;
            const r = b.getBoundingClientRect();
            return {w: Math.round(r.width), h: Math.round(r.height)};
        }""")
        assert button, "the capture button is gone"
        assert button["h"] >= TAP_TARGET_MIN, f"capture button {button}"

        with page.expect_file_chooser(timeout=5000):
            page.mouse.click(box["x"] + box["width"] / 2,
                             box["y"] + box["height"] * 0.72)
    finally:
        context.close()


def test_a_dropped_connection_is_visible_before_a_tap_is_wasted(app_url, browsers):
    """nicegui's own 'Connection lost' popup is wired at about 4s and CSS-
    delayed 2000ms on top, so it lands at about 6s. Until then every tap is a
    silent no-op that reads as a missed tap. socket.io knows in about one."""
    context = _page(browsers, "chromium", 393, 852)
    try:
        page = context.new_page()
        page.goto(app_url + "/", wait_until="networkidle", timeout=60000)
        page.wait_for_timeout(2000)
        context.set_offline(True)
        page.wait_for_timeout(2500)
        assert page.evaluate("document.body.getAttribute('data-connection')") == "lost"
        pointer = page.evaluate("""() => {
            const b = document.querySelector('.p-btn-primary');
            return b ? getComputedStyle(b).pointerEvents : null;
        }""")
        assert pointer == "none", f"primary action still accepts taps: {pointer!r}"
    finally:
        context.set_offline(False)
        context.close()


def test_the_typed_draft_outlives_a_reload(app_url, browsers):
    """The measured failure: text typed, connection dropped, text gone. The
    reconnect window covers a short drop; this covers every longer one."""
    context = _page(browsers, "chromium", 393, 852)
    sentence = "Where is the train station?"
    try:
        page = context.new_page()
        page.goto(app_url + "/", wait_until="networkidle", timeout=60000)
        page.wait_for_timeout(2000)
        page.locator("textarea").first.fill(sentence)
        page.wait_for_timeout(1200)
        page.reload(wait_until="networkidle")
        page.wait_for_timeout(3000)
        assert page.evaluate("(document.querySelector('textarea')||{}).value||''") == sentence
    finally:
        context.close()
