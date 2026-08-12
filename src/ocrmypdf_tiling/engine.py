"""TilingOcrEngine: an OCR engine wrapper for OCRmyPDF.

The wrapper presents the unified ``generate_ocr`` API to OCRmyPDF
(``supports_generate_ocr() -> True``) regardless of what the inner engine
implements. For oversized pages it tiles the image, runs the inner engine on
each tile, remaps the tile-local word boxes to full-page coordinates,
deduplicates the overlap zones, and stitches everything into a single
``PAGE -> LINE -> WORD`` OcrElement tree.

For inner engines that only implement ``generate_hocr`` (such as OCRmyPDF's
built-in Tesseract engine), the wrapper uses OCRmyPDF's own ``HocrParser`` to
convert each tile's hOCR output into an OcrElement tree -- the exact machinery
OCRmyPDF uses internally for its hOCR pipeline.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from PIL import Image

from ocrmypdf.pluginspec import OcrEngine, OrientationConfidence

from .tiling import (
    Word,
    dedup_words,
    group_words_into_lines,
    make_tiles,
    needs_tiling,
    remap_bbox,
    words_to_tree,
)

log = logging.getLogger(__name__)


def _tiling_opt(options, field, default):
    """Read a value from the typed ``options.tiling`` namespace if available."""
    try:
        return getattr(options.tiling, field)
    except (AttributeError, ValueError):
        return default


def _as_mm_pair(value, name: str) -> tuple[float, float]:
    """Normalize a max-page value (tuple or 'WxH' string) to (float, float)."""
    if isinstance(value, str):
        try:
            w, h = value.lower().split('x', 1)
            return float(w), float(h)
        except ValueError as exc:
            raise ValueError(f'{name} must be formatted like "210x297"') from exc
    return float(value[0]), float(value[1])


def _as_px_pair(value, name: str) -> tuple[int, int]:
    """Normalize a tile-size value (tuple or 'WxH' string) to (int, int)."""
    if isinstance(value, str):
        try:
            w, h = value.lower().split('x', 1)
            return int(w), int(h)
        except ValueError as exc:
            raise ValueError(f'{name} must be formatted like "2480x3508"') from exc
    return int(value[0]), int(value[1])


class TilingOcrEngine(OcrEngine):
    """Wrap any OcrEngine and apply sliding-window tiling to oversized pages.

    Args:
        inner: The engine to run on each tile (or on the whole page when no
            tiling is needed).
        options: OCRmyPDF OcrOptions (used to read tiling configuration).
    """

    def __init__(self, inner: OcrEngine, options):
        self.inner = inner
        self.options = options
        self._config = {
            'max_mm': _as_mm_pair(
                _tiling_opt(options, 'max_page_mm', (210.0, 297.0)),
                'tiling_max_page_mm',
            ),
            'tile_size': _as_px_pair(
                _tiling_opt(options, 'tile_size', (2480, 3508)),
                'tiling_tile_size',
            ),
            'overlap': _tiling_opt(options, 'overlap', 0.20),
            'iou': _tiling_opt(options, 'iou', 0.45),
        }
        log.debug(
            'TilingOcrEngine wrapping %s: config=%s', inner, self._config
        )

    # ------------------------------------------------------------------ #
    # Metadata / orientation: delegate 1:1 to the inner engine.
    #
    # Note: the OcrEngine base class declares these as abstract methods, so
    # they MUST be overridden for this class to be instantiable. OCRmyPDF
    # always calls them on an engine *instance*, so instance methods that
    # delegate to the inner engine are correct.
    # ------------------------------------------------------------------ #

    def version(self) -> str:
        return self.inner.version()

    def __str__(self) -> str:
        return f'Tiling({self.inner})'

    def creator_tag(self, options) -> str:
        return self.inner.creator_tag(options)

    def languages(self, options) -> set[str]:
        return self.inner.languages(options)

    def get_orientation(
        self, input_file: Path, options
    ) -> OrientationConfidence:
        return self.inner.get_orientation(input_file, options)

    def get_deskew(self, input_file: Path, options) -> float:
        return self.inner.get_deskew(input_file, options)

    # generate_hocr / generate_pdf are never invoked by OCRmyPDF because
    # supports_generate_ocr() returns True; they exist only to satisfy the
    # abstract interface and delegate to the inner engine.
    def generate_hocr(
        self, input_file: Path, output_hocr: Path, output_text: Path, options
    ) -> None:
        return self.inner.generate_hocr(
            input_file, output_hocr, output_text, options
        )

    def generate_pdf(
        self, input_file: Path, output_pdf: Path, output_text: Path, options
    ) -> None:
        return self.inner.generate_pdf(
            input_file, output_pdf, output_text, options
        )

    # ------------------------------------------------------------------ #
    # generate_ocr: the unified entry point.
    # ------------------------------------------------------------------ #

    @staticmethod
    def supports_generate_ocr() -> bool:
        return True

    def generate_ocr(
        self,
        input_file: Path,
        options,
        page_number: int = 0,
    ) -> tuple[object, str]:
        """Run OCR on ``input_file``, tiling first when the page is oversized.

        Returns ``(OcrElement tree, plain text)``.
        """
        with Image.open(input_file) as image:
            image.load()
            page_w, page_h = image.size
            dpi_info = image.info.get('dpi')
            dpi = float(dpi_info[0]) if dpi_info else None

        if not needs_tiling(image, self._config['max_mm']):
            log.debug('Page %d not oversized; delegating directly', page_number)
            return self._run_untiled(input_file, options, page_number)

        log.debug(
            'Page %d oversized (%dx%d); tiling with %s overlap %s',
            page_number,
            page_w,
            page_h,
            self._config['tile_size'],
            self._config['overlap'],
        )
        tiles = make_tiles(
            image,
            tile_size=self._config['tile_size'],
            overlap=self._config['overlap'],
        )

        all_words: list[Word] = []
        with tempfile.TemporaryDirectory(prefix='ocrmypdf-tiling-') as tmpdir:
            tmp = Path(tmpdir)
            for tile_index, (crop, off_x, off_y) in enumerate(tiles):
                tile_path = tmp / f'tile_{tile_index}.png'
                self._save_tile(crop, tile_path, dpi_info)
                tile_words = self._ocr_tile(
                    tile_path,
                    options,
                    page_number,
                    tile_index,
                    off_x,
                    off_y,
                    page_w,
                    page_h,
                )
                all_words.extend(tile_words)

        words = dedup_words(all_words, iou_threshold=self._config['iou'])
        log.debug(
            'Page %d: %d raw words -> %d after dedup, %d lines',
            page_number,
            len(all_words),
            len(words),
            len(group_words_into_lines(words)),
        )

        page_tree, text = words_to_tree(words, page_w, page_h, dpi)
        return page_tree, text

    # ------------------------------------------------------------------ #
    # Internals.
    # ------------------------------------------------------------------ #

    def _run_untiled(
        self, input_file: Path, options, page_number: int
    ) -> tuple[object, str]:
        """Run the inner engine on the full page without tiling."""
        if self.inner.supports_generate_ocr():
            return self.inner.generate_ocr(input_file, options, page_number)

        # Inner engine is hOCR-only (e.g. Tesseract): round-trip through
        # OCRmyPDF's own HocrParser.
        from ocrmypdf.hocrtransform import HocrParser

        with tempfile.TemporaryDirectory(prefix='ocrmypdf-tiling-') as tmpdir:
            tmp = Path(tmpdir)
            hocr_tmp = tmp / 'page.hocr'
            text_tmp = tmp / 'page.txt'
            self.inner.generate_hocr(input_file, hocr_tmp, text_tmp, options)
            tree = HocrParser(hocr_tmp).parse()
            text = text_tmp.read_text(encoding='utf-8')
        return tree, text

    @staticmethod
    def _save_tile(
        crop: Image.Image, tile_path: Path, dpi_info
    ) -> None:
        """Save a tile, preserving the page DPI so scaling stays consistent."""
        save_kwargs = {}
        if dpi_info:
            try:
                dpi = tuple(round(float(v)) for v in dpi_info)
                save_kwargs['dpi'] = dpi
            except (TypeError, ValueError):
                pass
        crop.save(tile_path, format='PNG', **save_kwargs)

    def _ocr_tile(
        self,
        tile_path: Path,
        options,
        page_number: int,
        tile_index: int,
        off_x: int,
        off_y: int,
        page_w: int,
        page_h: int,
    ) -> list[Word]:
        """Run the inner engine on one tile and remap its words to the page.

        Words that touch an *interior* tile edge are dropped: they are
        fragments of text that the tile crop cut in half. The overlapping
        adjacent tile contains the complete word, so keeping the fragment
        would pollute the output with spurious characters. Words touching a
        page edge are kept.
        """
        tile_tree = self._run_inner_on_tile(tile_path, options, page_number)
        tile_bbox = self._tile_bbox(tile_path)
        crop_w, crop_h = self._tile_size(tile_path)

        interior_left = off_x > 0
        interior_right = off_x + crop_w < page_w
        interior_top = off_y > 0
        interior_bottom = off_y + crop_h < page_h

        words: list[Word] = []
        for word_elem in tile_tree.words:
            if not word_elem.text or not word_elem.text.strip():
                continue
            if word_elem.bbox is None:
                continue
            b = word_elem.bbox
            if interior_left and b.left <= 1.0:
                continue
            if interior_right and b.right >= crop_w - 1.0:
                continue
            if interior_top and b.top <= 1.0:
                continue
            if interior_bottom and b.bottom >= crop_h - 1.0:
                continue
            words.append(
                Word(
                    text=word_elem.text,
                    bbox=remap_bbox(b, off_x, off_y, page_w, page_h),
                    confidence=word_elem.confidence,
                    source_tile=tile_index,
                    tile_bbox=tile_bbox,
                )
            )
        return words

    def _run_inner_on_tile(
        self, tile_path: Path, options, page_number: int
    ):
        """Run the inner engine on a single tile, returning an OcrElement tree."""
        if self.inner.supports_generate_ocr():
            tree, _text = self.inner.generate_ocr(
                tile_path, options, page_number
            )
            return tree

        from ocrmypdf.hocrtransform import HocrParser

        with tempfile.TemporaryDirectory(prefix='ocrmypdf-tiling-') as tmpdir:
            tmp = Path(tmpdir)
            hocr_tmp = tmp / 'tile.hocr'
            text_tmp = tmp / 'tile.txt'
            self.inner.generate_hocr(tile_path, hocr_tmp, text_tmp, options)
            return HocrParser(hocr_tmp).parse()

    @staticmethod
    def _tile_bbox(tile_path: Path):
        from ocrmypdf.models.ocr_element import BoundingBox

        with Image.open(tile_path) as img:
            return BoundingBox(
                left=0, top=0, right=img.width, bottom=img.height
            )

    @staticmethod
    def _tile_size(tile_path: Path) -> tuple[int, int]:
        with Image.open(tile_path) as img:
            return img.width, img.height
