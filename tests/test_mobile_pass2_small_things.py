"""Frame 07 of the mobile review: the small true things.

Source-level on purpose, like the pass 1 tests: deleting a fix fails the
ordinary suite, not only the opt-in browser tier in tests/mobile/.
"""
from pathlib import Path

import pytest

import TranslationUI
from TranslationUI import resolve_workspace_mode

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("raw, expected", [
    ("Text", "Text"), ("text", "Text"),
    ("Document", "Document"), ("doc", "Document"), (" DOCUMENT ", "Document"),
    ("Image/Camera", "Image/Camera"), ("Image", "Image/Camera"),
    ("image", "Image/Camera"), ("camera", "Image/Camera"), ("photo", "Image/Camera"),
])
def test_the_mode_a_person_would_type_is_accepted(raw, expected):
    assert resolve_workspace_mode(raw) == expected


@pytest.mark.parametrize("raw", ["", None, "voice", "Imagee", "../etc"])
def test_an_unknown_mode_is_not_quietly_mapped_to_something(raw):
    assert resolve_workspace_mode(raw) is None


def test_main_page_says_so_when_the_mode_is_unknown():
    """?mode=Image used to render Text with no word said. The page now uses
    the resolver and notifies on a miss instead of falling through."""
    src = Path(TranslationUI.__file__).read_text(encoding="utf-8")
    body = src.split("def main_page(", 1)[1].split("\n    def ", 1)[0]
    assert "resolve_workspace_mode(mode)" in body
    assert "ui.notify(" in body


def test_the_diagnostics_code_block_wraps():
    src = Path(TranslationUI.__file__).read_text(encoding="utf-8")
    body = src.split("def diagnostics_page(", 1)[1].split("\n    def ", 1)[0]
    assert "p-diagnostics" in body
    css = (ROOT / "static" / "passage.css").read_text(encoding="utf-8")
    rule = css.split(".p-diagnostics pre", 1)[1].split("}", 1)[0]
    assert "pre-wrap" in rule and "overflow-wrap: anywhere" in rule


def test_the_retired_mobile_mode_field_is_gone():
    src = Path(TranslationUI.__file__).read_text(encoding="utf-8")
    assert "mobile_input_mode" not in src
