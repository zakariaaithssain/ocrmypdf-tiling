# ocrmypdf-tiling

Engine-agnostic tiling plugin for [OCRmyPDF](https://github.com/ocrmypdf/OCRmyPDF).
Tiles oversized pages so they fit inside the pixel constraints of OCR engines,
runs OCR per tile with *any* wrapped engine, then stitches the results back
into a single text layer with pixel-accurate, page-level word boxes.

## Install

```bash
pip install ocrmypdf-tiling
```

Requires OCRmyPDF 17+ (the modern `generate_ocr` plugin API).

## Usage

Once installed (or loaded via `--plugin ocrmypdf_tiling`), enable tiling:

```bash
ocrmypdf --tiling big_scan.pdf out.pdf
```

`--tiling` enables the wrapper; `--tiling-engine` (default `auto`) selects the
inner engine. `auto` resolves to whatever OCR engine OCRmyPDF would otherwise
use — Tesseract by default, or PaddleOCR-VL 1.6 when `ocrmypdf_vl16` is loaded:

```bash
# Tiling + PaddleOCR-VL 1.6 on an oversized page
ocrmypdf --tiling --tiling-engine vl16 --vl-device gpu big_scan.pdf out.pdf

# Tiling + Tesseract
ocrmypdf --tiling --tiling-engine tesseract big_scan.pdf out.pdf
```

## Options

| Option | Default | Meaning |
| ------ | ------- | ------- |
| `--tiling` | off | Enable the tiling wrapper engine. |
| `--tiling-engine {auto,tesseract,vl16}` | `auto` | Inner OCR engine to wrap. |
| `--tiling-max-page-mm WxH` | `210x297` | Pages larger than this (in mm) are tiled. |
| `--tiling-tile-size WxH` | `2480x3508` | Tile pixel size. |
| `--tiling-overlap FLOAT` | `0.20` | Fraction of the tile that overlaps neighbors. |
| `--tiling-iou FLOAT` | `0.45` | IoU threshold for merging duplicate words in overlaps. |

## Notes

- On 16 GB GPUs, the VL engine serializes concurrent page predicts and uses
  Paddle's `auto_growth` allocator so a card isn't exhausted by concurrent
  pages or a large model reservation (see the `ocrmypdf_vl16` README).
- Run `ocrmypdf --help` after installing to see these options.

## License

MPL-2.0
