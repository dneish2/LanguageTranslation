"""Find the text lines in an image from its pixels.

Why this exists: the camera path needs a box per line of text so a translation
can be drawn over it. Asking a vision model for bounding boxes produced boxes
that sat about 35px BELOW the text they described — every overlay covered the
following line and left the original showing. Asking qwen3-vl for coordinates
at all returns nothing usable; it reads text beautifully and cannot report
where it saw it.

So the two halves come from different places: the model reads the text (it is
very good at that), and the geometry is measured off the pixels here, where it
is arithmetic rather than estimation. Pairing them in reading order is what
makes an accurate overlay possible without any OCR engine installed.

The method is a horizontal projection profile: a line of text is a band of
rows containing markedly more ink than the page around it. That is robust to
photographs, works on any script, and needs nothing but PIL.
"""
from __future__ import annotations

from dataclasses import dataclass

from PIL import Image, ImageFilter, ImageOps


@dataclass
class TextRow:
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    def as_bbox(self) -> list[int]:
        return [self.x0, self.y0, self.x1, self.y1]


def _ink_map(image: Image.Image) -> tuple[list[list[bool]], int, int]:
    """Per-pixel "is this ink" using a local threshold.

    A global threshold fails on two things this has to survive: photographs
    are unevenly lit, and headings are often a lighter colour than body text
    (the burgundy section titles on a menu were missed entirely by a global
    cut). Comparing each pixel against a blurred copy of the page — a cheap
    local background estimate — catches light-coloured text that a global
    threshold reads as background.
    """
    grey = ImageOps.grayscale(image)
    # Radius scales with the image so the background estimate stays "the page
    # around this text" rather than "these few characters" at any resolution.
    radius = max(4, min(grey.width, grey.height) // 40)
    background = grey.filter(ImageFilter.BoxBlur(radius))
    grey_px, bg_px = grey.load(), background.load()
    width, height = grey.size
    ink = [
        [grey_px[x, y] < bg_px[x, y] - 12 for x in range(width)]
        for y in range(height)
    ]
    return ink, width, height


def detect_text_rows(
    image: Image.Image, *, min_height: int = 6, min_gap: int = 2, pad: int = 2,
) -> list[TextRow]:
    """Bands of the image that contain a line of text, top to bottom."""
    ink, width, height = _ink_map(image)
    per_row = [sum(row) for row in ink]
    # The cut is relative to THIS page, not an absolute count. A fixed
    # threshold made the whole image one band on textured paper: the texture
    # contributes a steady amount of "ink" to every row, so what separates a
    # text line from blank paper is the excess above that baseline, not the
    # total. Take the median row as the baseline and require a text row to
    # clear it by a healthy margin of the observed range.
    ordered = sorted(per_row)
    baseline = ordered[len(ordered) // 2]
    ceiling = ordered[int(len(ordered) * 0.98)]
    if ceiling <= baseline:
        return []  # a blank or uniform image has no rows to find
    row_threshold = baseline + max(3, int((ceiling - baseline) * 0.25))
    is_text = [count > row_threshold for count in per_row]

    # Close single-row gaps so the dot of an "i" or a broken stroke doesn't
    # split one line into two rows.
    for y in range(1, height - 1):
        if not is_text[y] and is_text[y - 1] and is_text[y + 1]:
            is_text[y] = True

    # Columns inked down most of the WHOLE page are vertical rules, table
    # borders or paper texture — never a letter stroke, which is interrupted
    # by the gaps between lines. Excluding them globally is what stops every
    # row from being reported as full-width; a per-band test can't tell a
    # continuous stripe from a tall glyph, because within one band they look
    # identical.
    column_totals = [sum(ink[y][x] for y in range(height)) for x in range(width)]
    structural = [total > height * 0.6 for total in column_totals]

    bands: list[tuple[int, int]] = []
    start: int | None = None
    for y, flag in enumerate(is_text):
        if flag and start is None:
            start = y
        elif not flag and start is not None:
            if y - start >= min_height:
                bands.append((start, y))
            start = None
    if start is not None and height - start >= min_height:
        bands.append((start, height))

    # Merge bands separated by less than min_gap — ascenders and descenders
    # sometimes carve a near-empty row out of the middle of a line.
    merged: list[tuple[int, int]] = []
    for band in bands:
        if merged and band[0] - merged[-1][1] < min_gap:
            merged[-1] = (merged[-1][0], band[1])
        else:
            merged.append(band)

    rows: list[TextRow] = []
    for y0, y1 in merged:
        # Horizontal extent measured only INSIDE this band, and a column has
        # to be inked through a real fraction of the band's height to count.
        # Accepting "any inked pixel" made every row span the full width:
        # paper texture and vertical rules put a little ink in every column,
        # so the test has to distinguish a stroke from a speck.
        band_height = max(1, y1 - y0)
        needed = max(2, int(band_height * 0.25))
        columns = [
            x for x in range(width)
            if not structural[x] and sum(ink[y][x] for y in range(y0, y1)) >= needed
        ]
        if not columns:
            continue
        # A band spanning essentially the whole width is texture or a rule,
        # not a line of text — drop it rather than emit a box over blank paper.
        if columns[-1] - columns[0] > width * 0.97:
            continue
        rows.append(TextRow(
            x0=max(0, columns[0] - pad),
            y0=max(0, y0 - pad),
            x1=min(width, columns[-1] + pad + 1),
            y1=min(height, y1 + pad),
        ))

    # Lines of text on one page are of broadly similar height, so a band many
    # times the typical one isn't a line — it is a region where uneven
    # lighting (a photo's vignette) lifted blank paper over the threshold and
    # merged everything below it into a single band. Drop those rather than
    # cover a quarter of the page with one overlay.
    if len(rows) >= 3:
        heights = sorted(r.height for r in rows)
        typical = heights[len(heights) // 2]
        rows = [r for r in rows if r.height <= typical * 4]
    return rows


def fit_vertical_correction(bboxes: list[list[int]], rows: list[TextRow]
                            ) -> tuple[float, float]:
    """Fit `true_y = scale * model_y + offset` from model boxes to real ink.

    A single global shift wasn't enough. Correcting the test menu by the
    median -14px fixed the top of the page and left the bottom still wrong,
    because the model's error GROWS with distance down the page — it isn't a
    constant offset, it is a slightly wrong vertical scale. Fitting both terms
    handles offset and drift together.

    Only confident pairs feed the fit: a box whose nearest row is further away
    than the typical line spacing is probably matched to the wrong line, and
    including it would bend the fit toward that mistake. With too few
    confident pairs the fit is abandoned entirely (identity), because a
    two-parameter fit from two noisy points is worse than not correcting.
    """
    if len(rows) < 3 or len(bboxes) < 3:
        return 1.0, 0.0
    centres = sorted((r.y0 + r.y1) / 2 for r in rows)
    spacings = [b - a for a, b in zip(centres, centres[1:])]
    spacing = sorted(spacings)[len(spacings) // 2] if spacings else 0
    if spacing <= 0:
        return 1.0, 0.0

    pairs: list[tuple[float, float]] = []
    for x0, y0, x1, y1 in bboxes:
        model_y = (y0 + y1) / 2
        nearest = min(centres, key=lambda c: abs(c - model_y))
        if abs(nearest - model_y) <= spacing:
            pairs.append((model_y, nearest))
    if len(pairs) < 3:
        return 1.0, 0.0

    n = len(pairs)
    mean_x = sum(p[0] for p in pairs) / n
    mean_y = sum(p[1] for p in pairs) / n
    variance = sum((p[0] - mean_x) ** 2 for p in pairs)
    if variance <= 0:
        return 1.0, mean_y - mean_x
    covariance = sum((p[0] - mean_x) * (p[1] - mean_y) for p in pairs)
    scale = covariance / variance
    # A wildly non-unit scale means the pairing was wrong, not the geometry.
    if not 0.8 <= scale <= 1.25:
        return 1.0, mean_y - mean_x
    return scale, mean_y - scale * mean_x


def apply_vertical_correction(bbox: list[int], scale: float, offset: float,
                              height: int) -> list[int]:
    x0, y0, x1, y1 = bbox
    new_y0 = int(round(scale * y0 + offset))
    new_y1 = int(round(scale * y1 + offset))
    if new_y1 - new_y0 < 4:            # never collapse a box to nothing
        new_y1 = new_y0 + max(4, y1 - y0)
    return [x0, max(0, new_y0), x1, min(height, new_y1)]


def estimate_vertical_bias(bboxes: list[list[int]], rows: list[TextRow]) -> int:
    """How far the model's boxes sit from the ink, as one number.

    The vision model's error is SYSTEMATIC, not random: on the test menu every
    box sat about 35px below its text. That distinction decides the fix.
    Snapping each box to its nearest row individually made the overlay worse,
    because line spacing (~43px) is close to the bias (~35px) — so "nearest"
    frequently resolved to the line below, and each box was confidently moved
    somewhere wrong.

    Estimating one shift for the whole page can't do that. Every box moves
    together, so the layout's relative structure is preserved and a bad
    estimate degrades to "still slightly off" rather than "scrambled".
    The median is used rather than the mean so a few boxes with no matching
    row can't drag the estimate.
    """
    if not rows or not bboxes:
        return 0
    offsets = []
    for x0, y0, x1, y1 in bboxes:
        centre = (y0 + y1) / 2
        nearest = min(rows, key=lambda r: abs((r.y0 + r.y1) / 2 - centre))
        offsets.append((nearest.y0 + nearest.y1) / 2 - centre)
    offsets.sort()
    return int(offsets[len(offsets) // 2])


def apply_vertical_bias(bbox: list[int], bias: int, height: int) -> list[int]:
    x0, y0, x1, y1 = bbox
    return [x0, max(0, y0 + bias), x1, min(height, y1 + bias)]


def snap_bbox_to_rows(bbox: list[int], rows: list[TextRow], *,
                      max_drift_ratio: float = 1.5) -> list[int]:
    """Correct a model-reported box's vertical placement using detected rows.

    This is deliberately NOT a pairing of rows to OCR lines. Zipping the two
    sequences in order looks reasonable and is unsafe: on the test menu the
    detector finds 12 bands where the model reads 24 lines, because a dish and
    its price share one visual row. Zipping them would slide every caption one
    line further out of place the further down the page you go, and text drawn
    over the WRONG line is worse than no overlay at all.

    Snapping avoids the problem entirely. Each source does what it is good at:
    the model says what the text is and roughly where, the pixels say exactly
    where the ink sits. A box moves to the nearest detected row only when one
    is close enough to plausibly be the same line — otherwise it is left alone,
    so an undetected row degrades to the old behaviour rather than a wrong one.
    """
    if not rows:
        return bbox
    x0, y0, x1, y1 = bbox
    centre = (y0 + y1) / 2
    height = max(1, y1 - y0)
    nearest = min(rows, key=lambda r: abs((r.y0 + r.y1) / 2 - centre))
    drift = abs((nearest.y0 + nearest.y1) / 2 - centre)
    if drift > height * max_drift_ratio:
        return bbox
    # Keep the model's horizontal extent: it tracks the specific phrase, while
    # a detected row spans everything on that line (a dish AND its price).
    return [x0, nearest.y0, x1, nearest.y1]
