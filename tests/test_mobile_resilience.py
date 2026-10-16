"""Pass 1 of the mobile work: the session survives a drop, and the camera is
reachable with a thumb.

Every assertion here corresponds to a measurement taken against the running app
on 2026-10-15 (the storyboard review). The browser-level proof lives in
`tests/mobile/test_mobile_viewports.py`, which needs Playwright; these are the
checks that hold without a browser, so that deleting the fix fails the ordinary
suite rather than only the optional one.
"""
import ast
import inspect
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import TranslationUI as module
from conftest import PageBuild

STYLESHEET = ROOT / "static" / "passage.css"


# ── the reconnect window ────────────────────────────────────────────────

def test_reconnect_window_is_longer_than_niceguis_desktop_default():
    """3.0 s is nicegui's default and it deleted the client -- and the typed
    source text with it -- after any drop longer than a held breath."""
    from nicegui import ui

    default = inspect.signature(ui.run).parameters["reconnect_timeout"].default
    assert default == 3.0, (
        "nicegui's default moved; re-check whether this override is still "
        f"needed and what it should be (saw {default!r})"
    )
    assert module.reconnect_timeout() >= 30.0


def test_start_ui_actually_passes_the_timeout_to_ui_run():
    """The value is useless if the call does not carry it. Asserted on the
    parsed call rather than on a substring, so a commented-out line or a
    mention in a docstring cannot satisfy it."""
    tree = ast.parse(inspect.getsource(module.start_ui))
    run_calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "run"
    ]
    assert run_calls, "start_ui no longer calls ui.run"
    keywords = {kw.arg for call in run_calls for kw in call.keywords}
    assert "reconnect_timeout" in keywords


@pytest.mark.parametrize("raw,expected", [
    ("45", 45.0),
    ("5.5", 5.5),
])
def test_the_deploy_can_retune_the_window(monkeypatch, raw, expected):
    monkeypatch.setenv("PASSAGE_RECONNECT_TIMEOUT", raw)
    assert module.reconnect_timeout() == expected


@pytest.mark.parametrize("raw", ["", "soon", "0", "-1"])
def test_a_nonsense_window_falls_back_rather_than_crashing_the_app(monkeypatch, raw):
    """A zero or negative window deletes the client on the first missed
    heartbeat, which is worse than the default this exists to replace. A
    typo in a deploy variable must not be the thing that breaks sessions."""
    monkeypatch.setenv("PASSAGE_RECONNECT_TIMEOUT", raw)
    assert module.reconnect_timeout() == module.RECONNECT_TIMEOUT_SECONDS


# ── the capture card ────────────────────────────────────────────────────

def _upload_elements(page_ui, mode):
    """Render the workspace in `mode` and return its ui.upload elements."""
    page_ui.input_mode = mode
    build = PageBuild(page_ui._render_source_input_panel)
    build()
    from nicegui import ui as nicegui_ui

    return [
        element for element in build.client.elements.values()
        if isinstance(element, nicegui_ui.upload)
    ]


@pytest.mark.parametrize("mode,marker", [
    ("Image/Camera", "p-capture-image"),
    ("Document", "p-capture-doc"),
])
def test_the_upload_card_is_marked_as_a_capture_target(mode, marker):
    """`p-capture` is what the click delegation and the 48px sizing both hang
    off. Without the class the card is back to a 32x32 button nobody can hit."""
    page_ui = module.TranslationUI()
    uploads = _upload_elements(page_ui, mode)
    assert len(uploads) == 1, f"expected one uploader in {mode}, got {len(uploads)}"
    classes = uploads[0]._classes
    assert "p-capture" in classes
    assert marker in classes


def test_the_camera_still_opens_directly_on_a_phone():
    """capture=environment was already right before this work. Restyling the
    card must not drop it, or the picker offers the photo library instead of
    the camera -- the whole point of the screen."""
    page_ui = module.TranslationUI()
    uploads = _upload_elements(page_ui, "Image/Camera")
    props = uploads[0]._props
    assert props.get("capture") == "environment"
    assert ".png" in str(props.get("accept", ""))


def test_the_capture_label_is_not_baked_into_the_server_render():
    """The copy differs by input device -- a phone has no pointer to click
    with and nothing to drop -- and the server cannot know which it is. The
    label therefore comes from CSS, so the Python label must stay empty."""
    page_ui = module.TranslationUI()
    for mode in ("Image/Camera", "Document"):
        uploads = _upload_elements(page_ui, mode)
        assert not uploads[0]._props.get("label"), (
            f"{mode}: a server-rendered label would show on both device kinds"
        )


# ── the stylesheet carries the rules the measurements closed ────────────

def _stylesheet() -> str:
    return STYLESHEET.read_text(encoding="utf-8")


def test_no_input_is_left_under_the_ios_zoom_threshold():
    """iOS Safari zooms the whole page when a focused input is under 16px and
    the page cannot undo it. All three inputs on / were 14px."""
    css = _stylesheet()
    assert "max(16px, 1rem)" in css, "the 16px floor on inputs is gone"
    assert ".q-field__native" in css


def test_the_capture_card_rules_survive():
    css = _stylesheet()
    for rule in (".p-capture", "min-height: 48px", "--p-capture-copy"):
        assert rule in css, f"{rule!r} missing: the capture card is unstyled"


def test_a_lost_connection_is_visible_on_the_primary_action():
    """nicegui's own popup lands at about 6s. socket.io knows in about 1.
    Those seconds are when a tap reads as a missed tap rather than a dead one."""
    css = _stylesheet()
    assert 'body[data-connection="lost"]' in css


def test_the_stylesheet_copy_in_the_design_system_has_not_drifted():
    """design-system/.design-sync/NOTES.md states src/styles.css is a
    byte-identical copy of the live stylesheet. A silent drift makes the
    design system describe a UI that no longer exists."""
    copy = ROOT / "design-system" / "src" / "styles.css"
    if not copy.exists():
        pytest.skip("design-system package not present")
    assert copy.read_bytes() == STYLESHEET.read_bytes(), (
        "static/passage.css and design-system/src/styles.css have diverged; "
        "copy the stylesheet across (the DS build copies it verbatim)"
    )


# ── the scripts reach every page, not just the workspace ────────────────

@pytest.mark.parametrize("script_method,needle", [
    ("_inject_mobile_resilience_js", "passage.draft.v1"),
    ("_inject_capture_target_js", "p-capture"),
])
def test_the_mobile_shell_scripts_are_injected(monkeypatch, script_method, needle):
    """A dropped connection is not a property of one route, so these ride on
    _inject_theme -- the one call every page build makes."""
    captured = []
    monkeypatch.setattr(module.ui, "add_body_html", lambda html, *a, **k: captured.append(html))
    monkeypatch.setattr(module.ui, "add_head_html", lambda *a, **k: None)
    monkeypatch.setattr(module.ui, "colors", lambda *a, **k: None)

    page_ui = module.TranslationUI()
    PageBuild(page_ui._inject_theme)()

    assert any(needle in html for html in captured), (
        f"{script_method}'s script never reached the page shell"
    )
