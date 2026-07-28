"""Capability probes: does the machine report what it actually resolved?

The trap these guard (PORTABILITY_PLAN.md §2.2): every local path in this app
falls back silently — local LLM to hosted, real font to PIL's bitmap face. On a
machine with no Ollama and no fonts the app still looks perfect, so a test that
only asserts "the call succeeded" is worthless. Every test here asserts WHICH
path ran and, when a fallback fires, that the fallback was RECORDED.
"""
import logging
import os
import socket
import subprocess
import sys
import urllib.error
from pathlib import Path

import pytest
from PIL import Image, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import TranslationBackend as tb  # noqa: E402
from image_compositor import (  # noqa: E402
    FONT_CANDIDATES,
    ImageCompositor,
    OverlayStyle,
    probe_font,
    resolve_font,
)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _closed_port() -> int:
    """A port nothing is listening on, so a connect is REFUSED."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


@pytest.fixture
def black_hole_port():
    """A port that accepts the TCP connection and then never answers.

    This is the shape of the dangerous failure: the endpoint is *present*, so
    the connection succeeds, but the reply does not arrive inside the probe
    window. It must be classified as a timeout, never as a refusal.
    """
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    # Deep backlog on purpose: connections are never accepted, and a full
    # backlog would start REFUSING them, turning this fixture into the
    # opposite of what it is for.
    server.listen(128)
    try:
        yield server.getsockname()[1]
    finally:
        server.close()


# --------------------------------------------------------------------------
# 1. timeout vs refusal must not collapse into "unavailable"
# --------------------------------------------------------------------------

def test_classify_probe_error_separates_timeout_from_refusal():
    timed_out = urllib.error.URLError(socket.timeout("timed out"))
    refused = urllib.error.URLError(ConnectionRefusedError(61, "refused"))
    assert tb.classify_probe_error(timed_out) == tb.PROBE_TIMEOUT
    assert tb.classify_probe_error(refused) == tb.PROBE_REFUSED
    assert tb.PROBE_TIMEOUT != tb.PROBE_REFUSED


def test_classify_probe_error_unwraps_nested_reasons():
    nested = urllib.error.URLError(urllib.error.URLError(TimeoutError()))
    assert tb.classify_probe_error(nested) == tb.PROBE_TIMEOUT


def test_classify_probe_error_dns_http_and_other():
    assert tb.classify_probe_error(socket.gaierror(11001, "no such host")) == tb.PROBE_DNS
    http = urllib.error.HTTPError("http://x/api/tags", 500, "boom", {}, None)
    assert tb.classify_probe_error(http) == tb.PROBE_HTTP_ERROR
    assert tb.classify_probe_error(ValueError("something else")) == tb.PROBE_ERROR


def test_probe_reports_refused_for_a_dead_port():
    report = tb.probe_local_llm(f"http://127.0.0.1:{_closed_port()}", timeout=1.0)
    assert report["reachable"] is False
    assert report["outcome"] == tb.PROBE_REFUSED, report
    assert report["models"] == []
    assert report["endpoint"].endswith("/api/tags")


def test_probe_reports_timeout_not_refusal_for_a_slow_endpoint(black_hole_port):
    """The false negative §2.3 warns about: present but slow.

    If this ever reported PROBE_REFUSED, a live-but-loaded Ollama would look
    exactly like an uninstalled one — the failure mode the whole probe exists
    to make visible.
    """
    report = tb.probe_local_llm(f"http://127.0.0.1:{black_hole_port}", timeout=0.25)
    assert report["reachable"] is False
    assert report["outcome"] == tb.PROBE_TIMEOUT, report
    assert report["outcome"] != tb.PROBE_REFUSED
    assert report["timeout_seconds"] == 0.25
    assert report["elapsed_ms"] >= 200
    assert report["detail"]


def test_probe_timeout_is_bounded_by_its_argument(black_hole_port):
    """Forgiving, but never a hang: startup must not wait forever."""
    report = tb.probe_local_llm(f"http://127.0.0.1:{black_hole_port}", timeout=0.2)
    assert report["outcome"] == tb.PROBE_TIMEOUT
    assert report["elapsed_ms"] < 5000


# --------------------------------------------------------------------------
# 2. the constant itself
# --------------------------------------------------------------------------

def test_live_probe_timeout_default_is_more_forgiving_than_the_dev_machine():
    # 0.6s was fitted to this 5090 box, where Ollama answers instantly.
    assert tb.LIVE_PROBE_TIMEOUT_SECONDS >= 2.0
    assert tb.LIVE_PROBE_TIMEOUT_SECONDS <= 10.0  # still bounded


def test_live_probe_timeout_is_env_overridable():
    """Checked in a fresh interpreter: the module constant is read at import,
    so patching it in-process would prove nothing about the env var."""
    env = dict(os.environ, PASSAGE_LIVE_PROBE_TIMEOUT="7.5")
    out = subprocess.run(
        [sys.executable, "-c",
         "import TranslationBackend as t; print(t.LIVE_PROBE_TIMEOUT_SECONDS)"],
        cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=180,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert "7.5" in out.stdout


# --------------------------------------------------------------------------
# 3. the backend records WHY local was unavailable
# --------------------------------------------------------------------------

def test_available_local_models_records_the_refusal_reason(monkeypatch):
    monkeypatch.setattr(tb, "OLLAMA_BASE_URL", f"http://127.0.0.1:{_closed_port()}")
    backend = tb.TranslationBackend()
    assert backend.available_local_models() == []
    # The empty list alone cannot distinguish "nothing installed" from
    # "nothing answered". The recorded probe can.
    assert backend.last_local_probe is not None
    assert backend.last_local_probe["outcome"] == tb.PROBE_REFUSED
    assert backend.last_local_probe["reachable"] is False


def test_available_local_models_records_a_timeout_distinctly(monkeypatch, black_hole_port):
    monkeypatch.setattr(tb, "OLLAMA_BASE_URL", f"http://127.0.0.1:{black_hole_port}")
    monkeypatch.setattr(tb, "LIVE_PROBE_TIMEOUT_SECONDS", 0.25)
    backend = tb.TranslationBackend()
    assert backend.available_local_models() == []
    assert backend.last_local_probe["outcome"] == tb.PROBE_TIMEOUT
    assert backend.last_local_probe["outcome"] != tb.PROBE_REFUSED


def test_live_local_fallback_to_hosted_is_recorded_as_refusal(monkeypatch):
    """The hosted fallback fired — and said why. Production path, no bypass."""
    monkeypatch.setattr(tb, "LIVE_LOCAL_ENABLED", True)
    monkeypatch.setattr(tb, "OLLAMA_BASE_URL", f"http://127.0.0.1:{_closed_port()}")
    backend = tb.TranslationBackend()
    assert backend._live_local_provider() is None  # fell back to hosted
    assert backend.last_live_probe is not None
    assert backend.last_live_probe["outcome"] == tb.PROBE_REFUSED
    assert backend.last_live_probe["reachable"] is False


def test_live_local_fallback_records_timeout_distinctly(monkeypatch, black_hole_port):
    monkeypatch.setattr(tb, "LIVE_LOCAL_ENABLED", True)
    monkeypatch.setattr(tb, "OLLAMA_BASE_URL", f"http://127.0.0.1:{black_hole_port}")
    monkeypatch.setattr(tb, "LIVE_PROBE_TIMEOUT_SECONDS", 0.25)
    backend = tb.TranslationBackend()
    assert backend._live_local_provider() is None
    assert backend.last_live_probe["outcome"] == tb.PROBE_TIMEOUT
    assert backend.last_live_probe["outcome"] != tb.PROBE_REFUSED


# --------------------------------------------------------------------------
# 4. font resolution reports its fallback
# --------------------------------------------------------------------------

def test_resolve_font_reports_bitmap_fallback_when_no_face_exists():
    report = resolve_font(24, preferred="", candidates=("nope-not-a-face.ttf",))
    assert report["fallback"] is True
    assert report["kind"] == "pil_default"
    assert report["resolved"] is None
    assert report["tried"] == ["nope-not-a-face.ttf"]
    # Pillow >= 10.1 bundles a scalable default, so "is it a FreeTypeFont?"
    # no longer separates fallback from success. The recorded flag does.
    assert resolve_font(24, candidates=("nope-not-a-face.ttf",))["fallback"] is True


def test_resolve_font_reports_which_face_resolved():
    report = resolve_font(20)
    if report["fallback"]:
        pytest.skip("this machine has no vector face installed; nothing to resolve")
    assert report["kind"] == "truetype"
    assert report["resolved"], "a resolved face must be named, not implied"
    assert isinstance(report["font"], ImageFont.FreeTypeFont)
    assert report["resolved"] == report["tried"][-1]


def test_preferred_face_is_tried_first():
    report = resolve_font(18, preferred="zzz-missing.ttf")
    assert report["tried"][0] == "zzz-missing.ttf"


def test_probe_font_is_json_shaped_and_carries_no_font_object():
    report = probe_font(18)
    assert "font" not in report
    assert set(report) == {"resolved", "fallback", "kind", "tried", "size"}
    assert isinstance(report["fallback"], bool)


def test_font_candidates_cover_more_than_one_host_family():
    joined = " ".join(FONT_CANDIDATES)
    assert "C:\\Windows" in joined          # Windows
    assert "/usr/share/fonts" in joined     # Linux
    assert "/System/Library/Fonts" in joined  # macOS (path presence only —
    # whether it loads there is UNVERIFIED from this Windows machine)


def test_compositor_records_the_bitmap_fallback_through_the_real_compose_path(monkeypatch, caplog):
    """No bypass: a real ImageCompositor renders a real image.

    Only the candidate list is narrowed, to force the fallback that a machine
    with no fonts would hit. The assertion is that the fallback FIRED and was
    recorded — not that compose() returned bytes, which it does either way.
    """
    monkeypatch.setattr(ImageCompositor, "_FONT_FALLBACKS", ("nope-not-a-face.ttf",))
    compositor = ImageCompositor(OverlayStyle(font_family="also-missing.ttf"))
    from io import BytesIO
    buf = BytesIO()
    Image.new("RGB", (200, 80), (255, 255, 255)).save(buf, format="PNG")

    with caplog.at_level(logging.WARNING):
        out = compositor.compose(
            buf.getvalue(),
            [{"bbox": [10, 10, 190, 50], "translated": "Padrón €8.50"}],
        )

    assert out.startswith(b"\x89PNG")
    assert compositor.font_resolution is not None
    assert compositor.font_resolution["fallback"] is True
    assert compositor.font_resolution["kind"] == "pil_default"
    assert compositor.font_report()["fallback"] is True
    assert "font" not in compositor.font_report()
    assert any("PIL's default face" in r.getMessage() for r in caplog.records), \
        "a silent font fallback is exactly the failure this phase removes"


def test_compositor_reports_a_real_face_when_one_exists():
    compositor = ImageCompositor()
    report = compositor.font_report()
    if report["fallback"]:
        pytest.skip("this machine has no vector face installed")
    assert report["kind"] == "truetype"
    assert report["resolved"]


# --------------------------------------------------------------------------
# 5. no platform sniffing anywhere in the probe surface
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["TranslationBackend.py", "image_compositor.py"])
def test_no_os_branching_in_probe_owning_modules(name):
    """§7: 'No code branches on operating system.' A probe reads a fact; a
    platform check encodes a guess."""
    source = (ROOT / name).read_text(encoding="utf-8")
    for banned in ("sys.platform", "os.name ==", "platform.system()", "platform.machine()"):
        assert banned not in source, f"{name} branches on the host via {banned}"
