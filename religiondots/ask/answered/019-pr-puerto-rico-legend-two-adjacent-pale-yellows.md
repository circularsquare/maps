# 019 — pr: Puerto Rico legend: two adjacent pale yellows look identical

Summary: Puerto Rico legend: Christianity unspecified (#f8dc4f) and Protestant unspecified (#f8dd81) sit adjacent and look almost identical (CIEDE2000 6.0). Tweak one colour, or leave? No checker flags it; palette is yours.

*Filed 2026-09-14 by session `f95259a4-super`. Anita's call; nothing is waiting on it.*

## What I did

Left the palette unchanged.

## What it costs to reverse

One colour edit in the palette, then a retile, about 20-25 minutes.

## Why it is yours rather than mine

The palette is hand-tuned by Anita, and colours are fixed per category across the map.

## The detail

After today's remap of Puerto Rico's "Otros" write-ins to the `christianity` root (`sources/pr.md`
§11), two adjacent legend rows are near-identical pale yellows:

| row | colour | Puerto Rico dots at 1:1,000 |
|---|---|---|
| Christianity, unspecified | #f8dc4f | 662 |
| Protestant, unspecified | #f8dd81 | 297 |

- **Distance:** CIE76 21.5, CIEDE2000 6.0. They differ mainly in lightness (L 64 against 74), and
  6.0 is the figure that matches what the eye sees.
- **Not flagged:** `check_overview.py` holds pairs in one family to dE 12, and `check_palette.py`
  measures whole families, where Puerto Rico's four are all at least 25 apart.
- **Map-wide:** the pair exists everywhere these two nodes appear together, not only in Puerto Rico.


---

## Ruled 2026-09-14 by Anita: leave the palette

*"yes this is fine. this is not specific to puerto rico. its fine tho."*

No colour changes. The two yellows stay as they are wherever the two nodes meet.
