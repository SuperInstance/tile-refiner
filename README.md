# tile-refiner

Raw tiles to structured artifacts with dedup and keyword extraction.

Transforms raw agent knowledge submissions into structured, deduplicated PLATO tiles.

## What It Does

1. **Parse** — Extract question/answer pairs from raw text
2. **Deduplicate** — Hash-based duplicate detection across rooms
3. **Extract Keywords** — Identify domain-specific terminology
4. **Score Quality** — Confidence scoring based on length, specificity, and source
5. **Format** — Output in plato-tile-spec compatible format

## Installation

```bash
pip install tile-refiner
```

## Part of the Cocapn Fleet

Feeds the PLATO Room Server with high-quality knowledge tiles. Works with plato-tile-spec (format) and flywheel-engine (compounding loop).

## License

MIT
