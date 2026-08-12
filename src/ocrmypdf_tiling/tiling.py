"""tiling core for OCRmyPDF.

This module contains *pure* geometry and stitching helpers with zero knowledge
of any OCR engine. It is shared by:

- the ``ocrmypdf_tiling`` wrapper plugin (wraps any OCR engine), and
- any other consumer that needs to slice a page image into overlapping tiles,
  remap coordinates back to the full-page coordinate system, and deduplicate
  words that appear in overlapping tile zones.

Coordinate convention: image pixels, top-left origin, Y increasing downward
(identical to hOCR and ``ocrmypdf.models.ocr_element.BoundingBox``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from PIL import Image

try:  # ocrmypdf>=17
    from ocrmypdf import OcrElement
    from ocrmypdf.models.ocr_element import BoundingBox, OcrClass
except ImportError:  # pragma: no cover - only used for type annotations at import
    OcrElement = None  # type: ignore[assignment]

# Defaults (overridable via plugin CLI options)
A4_MM: tuple[float, float] = (210.0, 297.0)
TILE_PX: tuple[int, int] = (2480, 3508)  # A4 @ 300 dpi
OVERLAP: float = 0.20
IOU_THRESHOLD: float = 0.45
_DEFAULT_DPI: tuple[float, float] = (300.0, 300.0)


@dataclass
class Word:
    """A single recognized word with an axis-aligned bounding box.

    ``bbox`` is expressed in the current working coordinate system: tile-local
    coordinates while collecting, full-page coordinates after ``remap_bbox``.
    """

    text: str
    bbox: BoundingBox
    confidence: float | None = None
    source_tile: int = -1
    tile_bbox: BoundingBox | None = field(default=None)

    def center(self) -> tuple[float, float]:
        return (
            (self.bbox.left + self.bbox.right) / 2.0,
            (self.bbox.top + self.bbox.bottom) / 2.0,
        )


def page_size_mm(image: Image.Image) -> tuple[float, float]:
    """Return the physical page size in millimeters from pixels + DPI.

    DPI is read from ``image.info['dpi']`` (PIL may return a 2-tuple or a
    scalar); falls back to 300 DPI when absent.
    """
    dpi = image.info.get('dpi', _DEFAULT_DPI)
    if isinstance(dpi, (tuple, list)):
        dpi_x, dpi_y = float(dpi[0]), float(dpi[1])
    else:
        dpi_x = dpi_y = float(dpi)
    dpi_x = dpi_x or 300.0
    dpi_y = dpi_y or 300.0
    w_mm = image.width * 25.4 / dpi_x
    h_mm = image.height * 25.4 / dpi_y
    return w_mm, h_mm


def needs_tiling(
    image: Image.Image, max_mm: tuple[float, float] = A4_MM
) -> bool:
    """Return True if the physical page exceeds ``max_mm`` in either axis."""
    w_mm, h_mm = page_size_mm(image)
    return w_mm > max_mm[0] or h_mm > max_mm[1]


def make_tiles(
    image: Image.Image,
    tile_size: tuple[int, int] = TILE_PX,
    overlap: float = OVERLAP,
) -> list[tuple[Image.Image, int, int]]:
    """Slice an image into overlapping tiles using a sliding window.

    Pure crops -- no resizing, so pixel density (and therefore DPI) is
    preserved. The last tile in each direction is clamped to the image bounds,
    so its effective overlap may exceed the nominal value.

    Args:
        image: The full-page image to tile.
        tile_size: Desired tile (width, height) in pixels.
        overlap: Fractional overlap between adjacent tiles in [0, 1).

    Returns:
        List of ``(crop, off_x, off_y)`` where ``off_x``/``off_y`` are the
        tile's top-left corner in full-page coordinates.
    """
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f'overlap must be in [0, 1), got {overlap}')
    tile_w, tile_h = tile_size
    if tile_w <= 0 or tile_h <= 0:
        raise ValueError(f'tile_size must be positive, got {tile_size}')

    page_w, page_h = image.size
    step_x = max(int(round(tile_w * (1.0 - overlap))), 1)
    step_y = max(int(round(tile_h * (1.0 - overlap))), 1)

    tiles: list[tuple[Image.Image, int, int]] = []
    off_y = 0
    while off_y < page_h:
        off_x = 0
        y1 = min(off_y + tile_h, page_h)
        while off_x < page_w:
            x1 = min(off_x + tile_w, page_w)
            crop = image.crop((off_x, off_y, x1, y1))
            tiles.append((crop, off_x, off_y))
            if x1 >= page_w:
                break
            off_x += step_x
        if y1 >= page_h:
            break
        off_y += step_y
    return tiles


def remap_bbox(
    bbox: BoundingBox,
    off_x: int,
    off_y: int,
    page_w: int,
    page_h: int,
) -> BoundingBox:
    """Shift a tile-local bbox into full-page coordinates and clip to the page."""
    left = bbox.left + off_x
    top = bbox.top + off_y
    right = bbox.right + off_x
    bottom = bbox.bottom + off_y

    left = max(left, 0.0)
    top = max(top, 0.0)
    right = min(right, float(page_w))
    bottom = min(bottom, float(page_h))

    if right <= left:
        right = left + max(1.0, bbox.width)
    if bottom <= top:
        bottom = top + max(1.0, bbox.height)

    return BoundingBox(left=left, top=top, right=right, bottom=bottom)


def iou(a: BoundingBox, b: BoundingBox) -> float:
    """Intersection-over-union of two axis-aligned boxes (0.0 for empty)."""
    x1 = max(a.left, b.left)
    y1 = max(a.top, b.top)
    x2 = min(a.right, b.right)
    y2 = min(a.bottom, b.bottom)

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h

    a_area = a.width * a.height
    b_area = b.width * b.height
    union = a_area + b_area - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def _same_text(a: str, b: str) -> bool:
    """Normalized text equality for dedup (case-insensitive, whitespace-folded)."""
    return ' '.join(a.casefold().split()) == ' '.join(b.casefold().split())


def _interior_score(word: Word) -> float:
    """Score how far a word's center is from its source tile's edges.

    Used to prefer the copy of a duplicated word that is most fully inside a
    tile (least likely to be clipped at a seam).
    """
    if word.tile_bbox is None:
        return 0.0
    cx, cy = word.center()
    dx = min(cx - word.tile_bbox.left, word.tile_bbox.right - cx)
    dy = min(cy - word.tile_bbox.top, word.tile_bbox.bottom - cy)
    return min(dx, dy)


def _merge_union(a: Word, b: Word) -> Word:
    """Merge two duplicate words by taking the union box and max confidence."""
    bbox = BoundingBox(
        left=min(a.bbox.left, b.bbox.left),
        top=min(a.bbox.top, b.bbox.top),
        right=max(a.bbox.right, b.bbox.right),
        bottom=max(a.bbox.bottom, b.bbox.bottom),
    )
    confidence = None
    confs = [c for c in (a.confidence, b.confidence) if c is not None]
    if confs:
        confidence = max(confs)
    return Word(
        text=a.text,
        bbox=bbox,
        confidence=confidence,
        source_tile=a.source_tile,
        tile_bbox=a.tile_bbox,
    )


def dedup_words(
    words: list[Word], iou_threshold: float = IOU_THRESHOLD
) -> list[Word]:
    """Remove duplicate words that appear in overlapping tile zones.

    Two words are considered duplicates when their IoU exceeds
    ``iou_threshold`` AND their normalized text matches. For duplicates we keep
    the higher-confidence copy, then the one whose center is most interior to
    its source tile, then merge by union.

    Candidates are bucketed by a coarse spatial grid so dense pages do not
    degrade to O(n^2) pairwise comparisons.

    Returns:
        A new list of words (input order preserved for the survivors, which is
        important for deterministic downstream grouping).
    """
    if not words:
        return []

    grid_size = 256
    buckets: dict[tuple[int, int], list[int]] = {}
    for idx, word in enumerate(words):
        cx, cy = word.center()
        key = (int(cx // grid_size), int(cy // grid_size))
        buckets.setdefault(key, []).append(idx)

    survivors = [True] * len(words)

    def _near(idx_a: int, idx_b: int) -> bool:
        b = words[idx_b].bbox
        if b.right < words[idx_a].bbox.left or b.left > words[idx_a].bbox.right:
            return False
        if b.bottom < words[idx_a].bbox.top or b.top > words[idx_a].bbox.bottom:
            return False
        return True

    def _pick(a: Word, b: Word) -> Word:
        ca = a.confidence if a.confidence is not None else -1.0
        cb = b.confidence if b.confidence is not None else -1.0
        if ca > cb:
            return a
        if cb > ca:
            return b
        ia, ib = _interior_score(a), _interior_score(b)
        if ia > ib:
            return a
        if ib > ia:
            return b
        return _merge_union(a, b)

    checked: set[tuple[int, int]] = set()
    for idx, word in enumerate(words):
        if not survivors[idx]:
            continue
        cx, cy = word.center()
        gx, gy = int(cx // grid_size), int(cy // grid_size)
        for gxx in (gx - 1, gx, gx + 1):
            for gyy in (gy - 1, gy, gy + 1):
                for jdx in buckets.get((gxx, gyy), ()):
                    if jdx == idx:
                        continue
                    if not survivors[jdx]:
                        continue
                    pair = (min(idx, jdx), max(idx, jdx))
                    if pair in checked:
                        continue
                    checked.add(pair)
                    if not _near(idx, jdx):
                        continue
                    other = words[jdx]
                    if iou(word.bbox, other.bbox) > iou_threshold and _same_text(
                        word.text, other.text
                    ):
                        keep = _pick(word, other)
                        if keep is word:
                            survivors[jdx] = False
                        else:
                            survivors[idx] = False
                            words[idx] = keep
                            break
        if not survivors[idx]:
            continue

    return [word for idx, word in enumerate(words) if survivors[idx]]


def group_words_into_lines(
    words: list[Word], line_overlap_factor: float = 0.6
) -> list[list[Word]]:
    """Cluster words into text lines by vertical proximity, then sort by X.

    This mirrors the reference algorithm used in ``ocrmypdf-paddleocr``:
    sort words by Y-center, cluster while the Y-center gap stays within
    ``line_overlap_factor * median_word_height``, and sort each resulting line
    left-to-right.

    Call this AFTER cross-tile dedup so lines split at tile seams are healed.
    """
    if not words:
        return []

    def y_center(w: Word) -> float:
        return (w.bbox.top + w.bbox.bottom) / 2.0

    def height(w: Word) -> float:
        return max(w.bbox.height, 1.0)

    sorted_words = sorted(words, key=y_center)
    heights = sorted(height(w) for w in sorted_words)
    median_height = heights[len(heights) // 2] if heights else 20.0
    threshold = median_height * line_overlap_factor

    lines: list[list[Word]] = []
    current_line = [sorted_words[0]]
    current_y = y_center(sorted_words[0])

    for word in sorted_words[1:]:
        yc = y_center(word)
        if abs(yc - current_y) <= threshold:
            current_line.append(word)
        else:
            lines.append(current_line)
            current_line = [word]
            current_y = yc
    lines.append(current_line)

    for i, line in enumerate(lines):
        lines[i] = sorted(line, key=lambda w: w.bbox.left)

    return lines


def words_to_tree(
    words: list[Word],
    page_w: int,
    page_h: int,
    dpi: float | None,
) -> tuple[OcrElement, str]:
    """Build a PAGE -> LINE -> WORD OcrElement tree from full-page words.

    Args:
        words: Words already remapped to full-page coordinates.
        page_w, page_h: Full-page image size in pixels.
        dpi: Page DPI to attach to the PAGE element (may be None).

    Returns:
        ``(page_element, plain_text)`` where ``plain_text`` is each line joined
        with newlines.
    """
    lines = group_words_into_lines(words)

    page = OcrElement(
        ocr_class=OcrClass.PAGE,
        bbox=BoundingBox(left=0, top=0, right=page_w, bottom=page_h),
        dpi=dpi,
    )

    text_lines: list[str] = []
    for line_words in lines:
        if not line_words:
            continue
        line_bbox = BoundingBox(
            left=min(w.bbox.left for w in line_words),
            top=min(w.bbox.top for w in line_words),
            right=max(w.bbox.right for w in line_words),
            bottom=max(w.bbox.bottom for w in line_words),
        )
        line = OcrElement(
            ocr_class=OcrClass.LINE,
            bbox=line_bbox,
        )
        for word in line_words:
            line.children.append(
                OcrElement(
                    ocr_class=OcrClass.WORD,
                    bbox=word.bbox,
                    text=word.text,
                    confidence=word.confidence,
                )
            )
        page.children.append(line)
        text_lines.append(' '.join(w.text for w in line_words))

    return page, '\n'.join(text_lines)
