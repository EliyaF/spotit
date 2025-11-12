# Spot It Card Generator

Generate a Spot It-style circular card from exactly 8 input images. The script randomizes rotation, scale, and placement, avoids overlaps, balances distribution around the card, and produces different layouts on each run (unless a seed is set).

## Requirements

- Python 3.10+
- Windows, macOS, or Linux

Install dependencies:

```bash
pip install -r requirements.txt
```

## Prepare images

Place exactly 8 images in `images/` (supported: PNG, JPG, JPEG, WEBP). Transparent PNGs work best.

## Usage

- Auto-discover 8 files in `images/`:

```bash
python generate_card.py --out output/card.png
```

- Or pass 8 explicit paths:

```bash
python generate_card.py images/a.png images/b.png images/c.png images/d.png images/e.png images/f.png images/g.png images/h.png --out output/card.png
```

### Options

- `--size`: Canvas size in px (square). Default: 1200
- `--card`: Card diameter in px. Default: 92% of canvas
- `--bg`: Background color (hex). Default: `#FFFFFF`
- `--seed`: Set to reproduce a layout. Omit for fresh randomness each run.
- `--verbose`: Print placement method and radial stats
- `--debug-overlay PATH`: Save a PNG overlay showing symbol bounds/centers

## Generate All Cards

To generate all 57 Spot It cards:

```bash
python generate_all_cards.py
```

This creates cards in the `output/` directory with filenames like `card_01.png`, `card_02.png`, etc.

## Create Printable PDF

To create a printable PDF from all generated cards:

```bash
python create_printable_pdf.py
```

This creates `spot_it_cards_printable.pdf` with each card sized at 8.25 cm in diameter, arranged for A4 printing.

### PDF Options

- `--output FILENAME`: Custom output PDF filename
- `--cards DIRECTORY`: Directory containing card images (default: `output/`)
- `--size DIAMETER`: Card diameter in cm (default: 8.25)
- `--add-guides`: Add cutting guide lines to help with card cutting

**Tip**: Use `--add-guides` to create a PDF with dashed cutting lines that make it easier to cut out each card precisely.

### Printing Instructions

1. Print the PDF on A4 paper
2. Cut along the card boundaries
3. Each card will be exactly 8.25 cm in diameter

## Notes

- If placement fails (extremely rare), try a larger `--size`, reduce `--card`, or reduce symbol size by editing `min_frac`/`max_frac` in `generate_card`.
- Output is saved as PNG to preserve transparency/quality.

