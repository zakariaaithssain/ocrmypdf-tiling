"""tiling plugin for OCRmyPDF.

Exposes the hookimpl functions so the package can be loaded either as an
installed entry point or directly via ``ocrmypdf --plugin ocrmypdf_tiling``.
"""

from __future__ import annotations

from .engine import TilingOcrEngine
from .plugin import (
    add_options,
    check_options,
    get_ocr_engine,
    initialize,
)

__all__ = [
    'TilingOcrEngine',
    'add_options',
    'check_options',
    'get_ocr_engine',
    'initialize',
]
