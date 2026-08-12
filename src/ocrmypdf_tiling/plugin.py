"""OCRmyPDF plugin entry points for the tiling wrapper.

Loading this plugin (auto-loaded via the ``ocrmypdf`` entry point, or
explicitly with ``--plugin ocrmypdf_tiling``) adds a ``Tiling`` option group.
When ``--tiling`` is given it registers a ``TilingOcrEngine`` that wraps
whatever inner OCR engine is selected (Tesseract by default, or PaddleOCR-VL
1.6 via ``--ocr-engine vl16`` / the ``ocrmypdf_vl16`` plugin). When tiling
is disabled the plugin is transparent and OCRmyPDF falls through to the
built-in engine selection.

Options are declared as a pydantic model (``TilingOptions``) registered via
the ``register_options`` hook. This is the OCRmyPDF 17.10 way: the CLI stores
the flat ``tiling_*`` values in ``OcrOptions.extra_attrs`` and ``OcrOptions``
materializes the typed ``options.tiling.*`` namespace lazily. Because the
model validates/normalizes its own fields, nothing needs to be mutated on the
(immutable) ``OcrOptions`` object.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field, field_validator

from ocrmypdf import hookimpl
from ocrmypdf.exceptions import MissingDependencyError

from .engine import TilingOcrEngine

if TYPE_CHECKING:
    from pluggy import PluginManager

log = logging.getLogger(__name__)

# Set by initialize() so the 'auto' inner-engine resolution can re-invoke the
# get_ocr_engine hook and ask OCRmyPDF which engine it would have selected.
_pm: PluginManager | None = None

# Re-entrancy guard: when we re-invoke the get_ocr_engine hook from inside our
# own hookimpl (for 'auto' inner-engine resolution), our own hookimpl is called
# again by pluggy. Returning None there lets the *other* plugins decide the
# inner engine, which is exactly "whatever OCRmyPDF would have selected".
_resolving = False

_ENGINE_DEFAULT: Literal['auto', 'tesseract', 'vl16'] = 'auto'
_ENGINE_NAMES: tuple[Literal['auto', 'tesseract', 'vl16'], ...] = (
    'auto',
    'tesseract',
    'vl16',
)


class TilingOptions(BaseModel):
    """Typed tiling options (exposed to plugins as ``options.tiling.*``)."""

    enabled: bool = False
    engine: Literal['auto', 'tesseract', 'vl16'] = _ENGINE_DEFAULT
    max_page_mm: tuple[float, float] = (210.0, 297.0)
    tile_size: tuple[int, int] = (2480, 3508)
    overlap: float = Field(default=0.20, ge=0.0, lt=1.0)
    iou: float = Field(default=0.45, ge=0.0, le=1.0)

    @field_validator('max_page_mm', mode='before')
    @classmethod
    def _parse_mm(cls, v):
        if isinstance(v, str):
            w, h = v.lower().split('x', 1)
            return float(w), float(h)
        return v

    @field_validator('tile_size', mode='before')
    @classmethod
    def _parse_px(cls, v):
        if isinstance(v, str):
            w, h = v.lower().split('x', 1)
            return int(w), int(h)
        return v


@hookimpl
def register_options() -> dict[str, type[BaseModel]]:
    """Register the tiling option model namespace."""
    return {'tiling': TilingOptions}


@hookimpl
def add_options(parser):
    """Add the Tiling option group."""
    tiling = parser.add_argument_group(
        'Tiling', 'Sliding-window tiling for oversized pages (engine-agnostic)'
    )
    tiling.add_argument(
        '--tiling',
        action='store_true',
        dest='tiling_enabled',
        help='Enable the tiling wrapper engine (wraps the selected OCR engine).',
    )
    tiling.add_argument(
        '--ocr-engine',
        choices=list(_ENGINE_NAMES),
        default=_ENGINE_DEFAULT,
        dest='tiling_engine',
        help=(
            'Inner OCR engine to wrap. "auto" uses whatever engine OCRmyPDF '
            'would otherwise select (Tesseract, or PaddleOCR-VL 1.6 if the '
            'ocrmypdf_vl16 plugin is loaded).'
        ),
    )
    tiling.add_argument(
        '--tiling-max-page-mm',
        metavar='WxH',
        default='210x297',
        dest='tiling_max_page_mm',
        help='Tile when the physical page exceeds this size (mm). Default: 210x297 (A4).',
    )
    tiling.add_argument(
        '--tiling-tile-size',
        metavar='WxH',
        default='2480x3508',
        dest='tiling_tile_size',
        help='Tile size in pixels. Default: 2480x3508 (A4 @ 300 dpi).',
    )
    tiling.add_argument(
        '--tiling-overlap',
        type=float,
        default=0.20,
        dest='tiling_overlap',
        help='Fractional overlap between adjacent tiles in [0, 1). Default: 0.20.',
    )
    tiling.add_argument(
        '--tiling-iou',
        type=float,
        default=0.45,
        dest='tiling_iou',
        help='IoU threshold for deduplicating words in overlap zones. Default: 0.45.',
    )


@hookimpl
def check_options(options):
    """Validate tiling options and eagerly check the inner engine dependency."""
    if not getattr(options, 'tiling_enabled', False):
        return

    # Accessing options.tiling triggers lazy materialization + validation of
    # TilingOptions (dimension parsing, overlap/iou ranges). Let pydantic raise.
    try:
        options.tiling  # noqa: B018
    except Exception as exc:
        raise ValueError(f'Invalid tiling options: {exc}') from exc

    engine = options.tiling.engine
    if engine == 'vl16':
        try:
            from ocrmypdf_vl16.plugin import PaddleVL16Engine  # noqa: F401
        except ImportError as exc:
            raise MissingDependencyError(
                '--ocr-engine vl16 requires the "ocrmypdf_vl16" package '
                '(install with: uv add ocrmypdf-vl16, or the [vl16] extra of '
                'ocrmypdf-tiling).'
            ) from exc


@hookimpl
def initialize(plugin_manager):
    """Stash the pluggy manager for the 'auto' inner-engine resolution.

    OCRmyPDF passes the raw ``pluggy.PluginManager`` here (see
    ``ocrmypdf/api.py``), so ``plugin_manager`` is the pluggy manager itself.
    """
    global _pm
    _pm = plugin_manager
    log.debug(
        'ocrmypdf_tiling initialized; pluggy manager available: %s',
        _pm is not None,
    )


@hookimpl
def get_ocr_engine(options):
    """Return a TilingOcrEngine when --tiling is enabled, else None."""
    global _resolving
    if options is None:
        return None
    if not getattr(options, 'tiling_enabled', False):
        return None
    if _resolving:
        # Re-entrant call during 'auto' inner-engine resolution: defer to the
        # other engine plugins so the inner engine matches what OCRmyPDF would
        # have selected.
        return None
    _resolving = True
    try:
        inner = _resolve_inner(options)
    finally:
        _resolving = False
    return TilingOcrEngine(inner, options)


def _resolve_inner(options):
    """Resolve the inner OCR engine to wrap."""
    engine = getattr(options, 'tiling_engine', _ENGINE_DEFAULT)

    if engine == 'tesseract':
        return _get_tesseract_engine()

    if engine == 'vl16':
        try:
            from ocrmypdf_vl16.plugin import PaddleVL16Engine
        except ImportError as exc:
            raise MissingDependencyError(
                '--ocr-engine vl16 requires the "ocrmypdf_vl16" package.'
            ) from exc
        return PaddleVL16Engine()

    # auto: ask OCRmyPDF which engine it would select, with this plugin
    # temporarily blocked so the firstresult hook does not recurse into us.
    return _resolve_auto(options)


def _get_tesseract_engine():
    """Return the built-in Tesseract engine."""
    try:
        from ocrmypdf.builtin_plugins.tesseract_ocr import TesseractOcrEngine
    except ImportError as exc:
        raise MissingDependencyError(
            'Tesseract OCR engine is not available in this OCRmyPDF build.'
        ) from exc
    return TesseractOcrEngine()


def _resolve_auto(options):
    """Resolve the engine OCRmyPDF would have selected, without this plugin.

    Re-invokes the firstresult ``get_ocr_engine`` hook from inside our own
    hookimpl; our re-entrant hookimpl returns None (see ``get_ocr_engine``), so
    the other plugins -- the exact set OCRmyPDF would have consulted --
    determine the result. Falls back to Tesseract if resolution is unavailable.
    """
    if _pm is None:
        log.warning('pluggy manager unavailable; falling back to Tesseract')
        return _get_tesseract_engine()

    inner = _pm.hook.get_ocr_engine(options=options)

    if inner is None:
        log.debug('auto resolution returned None; falling back to Tesseract')
        return _get_tesseract_engine()
    log.debug('auto inner engine resolution -> %s', inner)
    return inner
