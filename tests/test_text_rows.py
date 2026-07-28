"""Pixel-measured text rows, and the overlay correction built on them."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image, ImageDraw
from passage import text_rows


def _font(size=18):
    from PIL import ImageFont
    for path in (r"C:\Windows\Fonts\arial.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "/Library/Fonts/Arial.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _page(lines=6, width=400, gap=34, top=24, texture=False):
    """Real glyphs, not solid bars. The detector estimates the page background
    with a local blur, and a bar thicker than the blur radius IS its own
    background — an unrealistic fixture that no real text produces."""
    img = Image.new("RGB", (width, top + lines * gap + 40), (250, 246, 238))
    d = ImageDraw.Draw(img)
    if texture:
        # Vertical stripes, like the paper grain on a photographed menu. This
        # is what made every detected row span the full image width.
        for x in range(0, width, 3):
            d.line((x, 0, x, img.height), fill=(238, 232, 220))
    font = _font()
    for i in range(lines):
        d.text((40, top + i * gap), f"Linea numero {i} de texto", font=font, fill=(30, 28, 26))
    return img


def test_finds_one_row_per_line():
    rows = text_rows.detect_text_rows(_page(lines=6))
    assert len(rows) == 6
    assert all(r.y0 < r.y1 for r in rows)
    assert [r.y0 for r in rows] == sorted(r.y0 for r in rows)


def test_rows_do_not_span_the_full_width_on_textured_paper():
    """Paper grain and vertical rules put ink in every column. Accepting that
    made every row full-width, which is useless for placing an overlay."""
    page = _page(lines=5, texture=True)
    rows = text_rows.detect_text_rows(page)

    assert rows, "texture defeated row detection entirely"
    for r in rows:
        assert r.x1 - r.x0 < page.width * 0.9, f"row spans {r.x0}-{r.x1} of {page.width}"


def test_blank_image_yields_no_rows():
    assert text_rows.detect_text_rows(Image.new("RGB", (200, 120), (255, 255, 255))) == []


def test_fit_recovers_a_known_offset_and_scale():
    """The vision model's boxes were systematically low AND drifted further
    out down the page — an offset alone left the bottom wrong."""
    rows = text_rows.detect_text_rows(_page(lines=8))
    assert len(rows) >= 6
    true_centres = [(r.y0 + r.y1) / 2 for r in rows]
    # A realistic distortion: shifted by well under one line's spacing, plus a
    # slight scale drift. The magnitude matters — an error approaching the line
    # gap makes "nearest row" resolve to the wrong line, which is exactly why
    # per-box snapping was abandoned in favour of fitting the whole page.
    model_boxes = [[10, int((c - 9) / 1.02) - 6, 200, int((c - 9) / 1.02) + 6]
                   for c in true_centres]

    scale, offset = text_rows.fit_vertical_correction(model_boxes, rows)
    fixed = [text_rows.apply_vertical_correction(b, scale, offset, 10_000) for b in model_boxes]

    before = max(abs(c - ((b[1] + b[3]) / 2)) for c, b in zip(true_centres, model_boxes))
    after = max(abs(c - ((b[1] + b[3]) / 2)) for c, b in zip(true_centres, fixed))
    assert after < before / 3, f"correction barely helped: {before:.1f} -> {after:.1f}"


def test_fit_is_identity_without_enough_evidence():
    """A two-parameter fit from two noisy points is worse than no correction."""
    rows = text_rows.detect_text_rows(_page(lines=6))
    assert text_rows.fit_vertical_correction([[0, 10, 10, 20]], rows) == (1.0, 0.0)
    assert text_rows.fit_vertical_correction([[0, 10, 10, 20]] * 5, []) == (1.0, 0.0)


def test_fit_refuses_an_implausible_scale():
    """A wild scale means the pairing was wrong, not the geometry — fall back
    to a pure shift rather than stretching the whole layout."""
    rows = text_rows.detect_text_rows(_page(lines=6))
    absurd = [[0, y, 10, y + 4] for y in (0, 5, 9, 14, 19, 23)]

    scale, _ = text_rows.fit_vertical_correction(absurd, rows)

    assert scale == 1.0


def test_correction_never_collapses_a_box():
    fixed = text_rows.apply_vertical_correction([0, 100, 50, 106], 0.0, 0.0, 500)
    assert fixed[3] - fixed[1] >= 4
