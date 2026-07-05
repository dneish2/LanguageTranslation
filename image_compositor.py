from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Any

from PIL import Image, ImageDraw, ImageFont


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

    def _font(self, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        try:
            return ImageFont.truetype(self.style.font_family, size=size)
        except OSError:
            return ImageFont.load_default()

    def _fit_font(self, draw: ImageDraw.ImageDraw, text: str, width: int, height: int):
        for size in range(self.style.font_size, 7, -1):
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
