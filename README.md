# ocrmypdf-tiling

tiling plugin for [OCRmyPDF](https://github.com/ocrmypdf/OCRmyPDF).
Tiles oversized pages so they fit inside the pixel constraints of OCR engines,
runs OCR per tile with *any* wrapped engine, then stitches the results back
into a single text layer with pixel-accurate, page-level word boxes.

## Install

```bash
uv pip install .
```

Requires OCRmyPDF 17+ (the modern `generate_ocr` plugin API).

## Usage

Once installed (or loaded via `--plugin ocrmypdf_tiling`), enable tiling:

```bash
ocrmypdf --tiling big_scan.pdf out.pdf
```

`--tiling` enables the wrapper.  
`--ocr-engine` (default `auto`) selects the OCR engine. `auto` resolves to whatever OCR engine OCRmyPDF would otherwise
use, Tesseract by default, or PaddleOCR-VL 1.6 when `ocrmypdf_vl16` is loaded:

```bash
# Tiling + PaddleOCR-VL 1.6 on an oversized page
ocrmypdf --tiling --ocr-engine vl16 --vl-device gpu big_scan.pdf out.pdf

# Tiling + Tesseract
ocrmypdf --tiling --ocr-engine tesseract big_scan.pdf out.pdf
```

## Options

| Option | Default | Meaning |
| ------ | ------- | ------- |
| `--tiling` | off | Enable the tiling wrapper engine. |
| `--ocr-engine {auto,tesseract,vl16}` | `auto` | Inner OCR engine to wrap. |
| `--tiling-max-page-mm WxH` | `210x297` | Pages larger than this (in mm) are tiled. |
| `--tiling-tile-size WxH` | `2480x3508` | Tile pixel size. |
| `--tiling-overlap FLOAT` | `0.20` | Fraction of the tile that overlaps neighbors. |
| `--tiling-iou FLOAT` | `0.45` | IoU threshold for merging duplicate words in overlaps. |

## License

MPL-2.0
