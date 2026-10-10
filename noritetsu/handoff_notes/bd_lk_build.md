# Bangladesh (bd) and Sri Lanka (lk): first builds, 2026-10-08

Both built (model, tiles, check_model); both `.osm.pbf` extracts deleted. Not run:
`tools/build_regions.py` (managing session).

## Shared-file change to land: tools/rebuild.py REGISTER

```diff
     "pk": "pk_register:data/raw/pk",
+    # Sri Lanka and Bangladesh: hand lists through rinf.py (lk_register.py's engine, nafrica's
+    # recipe; bd's list in bd_lines.py). After an extract (with --station-areas):
+    # `lk_register.py --clip lk`; `bd_register.py --clip bd` then `bd_register.py --fill bd`.
+    "lk": "lk_register:data/raw/rinf/lk",
+    "bd": "bd_register:data/raw/rinf/bd",
 }
```

and in MINUTES: `"lk": 1, "bd": 1` (each builds in about 30 s plus tiles).

No other shared file needs a change: no rinf.py hook, no borders.EXTRA entry (no crossing is
drawn, below), nothing in build_model.py.

## Then

- `python tools/build_regions.py` (puts lk and bd on the map).
- India need not be rebuilt for bd: no crossing is drawn on either side (India's build already
  stops at Gede, Petrapole and Haldibari). Rebuild it if you want `split_at_borders` run anyway;
  nothing should move.

## Files (all the country agent's own)

- `lk_register.py`: the engine (nafrica_register's convert, with optional km-from-origin per
  point as the register's chainage, `--clip` with the neighbours' outlines shrunk ~1.1 km,
  `--fill`, `--trace`, `--stations`, logged path checks) plus Sri Lanka's line list.
- `bd_register.py` + `bd_lines.py`: Bangladesh's list through that engine.
- `rinf_countries/lk.py`, `rinf_countries/bd.py`: two-line settings shims (as za.py).
- `rules/lk.py`, `rules/bd.py`; `colours/lk.csv`, `colours/bd.csv` (our picks);
  `REGISTER["lk"]`, `REGISTER["bd"]` in check_model.py; `lk_sources.md`, `bd_sources.md`.

## Numbers

| cc | register lines | km | running | greyed | check_model |
|---|---|---|---|---|---|
| lk | 12 | 1,414 | 10 lines, 1,328 km | 2 lines, 86 km | chain median 1.006, none off 5%; worst register deviation 0.04 |
| bd | 30 | 3,115 | 26 lines, 3,078 km | 4 lines, 37 km | worst register deviation 0.02; 15 path checks 0.95-1.01 |

OSM lines: lk none (its 12 route relations are the register's lines or specials: named
trains); bd Dhaka Metro MRT Line 6 (18.9 km).
