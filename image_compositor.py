from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import logging
from collections.abc import Sequence
from typing import Any

from PIL import Image, ImageDraw, ImageFont


#: Faces worth trying, most-wanted first. This is deliberately NOT a
#: per-OS list: the code never asks "am I on Windows?", it asks "does this
#: face load here?" — a fact only the running machine can answer. Bare names
#: come first because PIL searches its own font directories for them; the
#: absolute paths are the well-known homes of the same faces on the machines
#: this app targets, tried blindly and cheaply.
#:
#: Why this matters: PIL does not search the Windows font directory for
#: "DejaVuSans.ttf", so every call used to fall through to load_default(), a
#: tiny bitmap face with no Unicode coverage. On a Spanish menu that rendered
#: "Padrón" as "Padr▯n" and "€8.50" as "▯8.50", in text far too small for its
#: box. A missing glyph in a translation overlay is not cosmetic: the overlay
#: is the entire output. So the fallback is now *reported*, not silent.
FONT_CANDIDATES: tuple[str, ...] = (
    "DejaVuSans.ttf",
    "Arial.ttf",
    "arial.ttf",
    "segoeui.ttf",
    "Helvetica.ttc",
    "NotoSans-Regular.ttf",
    "LiberationSans-Regular.ttf",
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\segoeui.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/Library/Fonts/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
)


def resolve_font(
    size: int = 24,
    *,
    preferred: str | None = None,
    candidates: Sequence[str] = FONT_CANDIDATES,
) -> dict[str, Any]:
    """Probe for a real vector face and report exactly what happened.

    Returns a dict with:
      ``font``      the loaded PIL font object
      ``resolved``  the candidate string that loaded, or None
      ``fallback``  True when no requested face loaded and PIL's own default
                    face is in use. On old Pillow that is a tiny unscalable
                    bitmap with almost no Unicode coverage; on Pillow >= 10.1
                    it is a bundled scalable face, better but still not the
                    typeface anyone asked for. Either way it is a fallback and
                    must be reported.
      ``kind``      "truetype" or "pil_default"
      ``tried``     every candidate attempted, in order
      ``size``      the requested pixel size

    Side-effect free apart from reading font files; safe to call from a
    diagnostics endpoint.
    """
    tried: list[str] = []
    for candidate in (preferred, *candidates):
        if not candidate:
            continue
        if candidate in tried:
            continue
        tried.append(candidate)
        try:
            font = ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
        return {
            "font": font, "resolved": candidate, "fallback": False,
            "kind": "truetype", "tried": tried, "size": size,
        }
    # Last resort. Newer Pillow lets the default face be scaled, which at
    # least keeps the text legible even without full Unicode.
    try:
        font = ImageFont.load_default(size=size)
    except TypeError:
        font = ImageFont.load_default()
    return {
        "font": font, "resolved": None, "fallback": True,
        "kind": "pil_default", "tried": tried, "size": size,
    }


def probe_font(size: int = 24) -> dict[str, Any]:
    """Structured font capability report, without the font object.

    Importable and side-effect free — this is the shape /diagnostics consumes.
    """
    return {k: v for k, v in resolve_font(size).items() if k != "font"}


@dataclass
class OverlayStyle:
    font_size: int = 24
    font_family: str = "DejaVuSans.ttf"
    text_color: tuple[int, int, int] = (20, 20, 20)
    cover_color: tuple[int, int, int] = (255, 255, 255)
    padding: int = 3


class ImageCompositor:
    def __init__(self, style: OverlayStyle | None = None) -> None:
        self.style = style or OverlayStyle()
        self._font_cache: dict[int, Any] = {}
        #: Filled the first time a font is resolved; see font_report().
        self.font_resolution: dict[str, Any] | None = None

    def compose(self, image_bytes: bytes, regions: list[dict[str, Any]], *, show_original: bool = False) -> bytes:
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        draw = ImageDraw.Draw(img)
        for region in regions:
            bbox = region.get("bbox")
            if not bbox:
                continue
            x0, y0, x1, y1 = map(int, bbox)
            if not show_original:
                draw.rectangle((x0, y0, x1, y1), fill=self.style.cover_color)

            text = region.get("original", "") if show_original else region.get("translated") or ""
            if not text:
                continue
            direction = region.get("direction", "ltr")
            font = self._fit_font(draw, text, x1 - x0, y1 - y0)
            wrapped = self._wrap_text(draw, text, font, max(1, x1 - x0 - 2 * self.style.padding))
            block = "\n".join(wrapped)
            anchor_x = x1 - self.style.padding if direction == "rtl" else x0 + self.style.padding
            draw.multiline_text(
                (anchor_x, y0 + self.style.padding),
                block,
                fill=self.style.text_color,
                font=font,
                align="right" if direction == "rtl" else "left",
                spacing=2,
                anchor="ra" if direction == "rtl" else "la",
            )

        out = BytesIO()
        img.save(out, format="PNG")
        return out.getvalue()

    #: See module-level FONT_CANDIDATES. Kept as a class attribute so a test or
    #: a caller can narrow the search (e.g. to force the bitmap fallback).
    _FONT_FALLBACKS = FONT_CANDIDATES

    def _font(self, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        cached = self._font_cache.get(size)
        if cached is not None:
            return cached
        resolution = resolve_font(
            size,
            preferred=self.style.font_family,
            candidates=self._FONT_FALLBACKS,
        )
        # Recorded, never silent: a bitmap fallback means missing glyphs, and
        # the overlay IS the output. /diagnostics reads this.
        self.font_resolution = resolution
        if resolution["fallback"]:
            logging.warning(
                "[Overlay] no requested font resolved (tried %d candidates); using "
                "PIL's default face - expect wrong typeface and possibly missing glyphs",
                len(resolution["tried"]),
            )
        font = resolution["font"]
        self._font_cache[size] = font
        return font

    def font_report(self) -> dict[str, Any]:
        """What this compositor actually resolved, for diagnostics.

        Probes at the style's own font size if nothing has been rendered yet,
        so the answer is a fact about this machine rather than a guess.
        """
        if self.font_resolution is None:
            self._font(self.style.font_size)
        return {k: v for k, v in (self.font_resolution or {}).items() if k != "font"}

    def _fit_font(self, draw: ImageDraw.ImageDraw, text: str, width: int, height: int):
        # Start from the box, not from a fixed 24px: an overlay replacing a
        # 74px-tall headline in 24px type reads as a caption stuck over the
        # title rather than a translation of it. Still only ever shrinks to
        # fit, so nothing overflows its box.
        start = max(self.style.font_size, min(height, 160))
        for size in range(start, 7, -1):
            font = self._font(size)
            lines = self._wrap_text(draw, text, font, max(1, width - 2 * self.style.padding))
            block = "\n".join(lines)
            box = draw.multiline_textbbox((0, 0), block, font=font, spacing=2)
            if box[2] <= width and box[3] <= height:
                return font
        return self._font(8)

    def _wrap_text(self, draw: ImageDraw.ImageDraw, text: str, font, width: int) -> list[str]:
        words = text.split()
        if not words:
            return [""]
        lines: list[str] = []
        current = words[0]
        for word in words[1:]:
            trial = f"{current} {word}"
            box = draw.textbbox((0, 0), trial, font=font)
            if box[2] <= width:
                current = trial
            else:
                lines.append(current)
                current = word
        lines.append(current)
        return lines
