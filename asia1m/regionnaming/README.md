# asia1m region naming

Turns the **main** layer of `asia1m.ase` into `regions_all.csv` — one row per colored
region with a centroid (lat/lon) and an admin-division description — to feed to Gemini for naming.

## Output
- **`regions_all.csv`** — columns: `name` (blank, to fill), `rid`, `color`, `area_px`, `lon`, `lat`,
  `country`, `province`, `admin_divisions`, `flags`.
- **`REPORT.md`** — method, admin sources per country, and edge cases to review.

## Regenerate (after drawing more regions)
```
python asia1m/regionnaming/run.py
```
Exports the `main` layer, extracts regions, rasterizes admin boundaries, writes the CSV + report.
Set `ASEPRITE=/path/to/Aseprite.exe` if Aseprite isn't at the default install location.
Needs the geo stack (geopandas, rasterio, scipy) — same env as `generateAsia.py`.

Intermediates (`main.png`, `*.npy`, `*_names.json`, `regions_prelim.csv`) are regenerable and gitignored.

## How it works
1. `geo.py` — pixel↔lon/lat using `generateAsia.py`'s Albers projection.
2. `extract_regions.py` — each exact color = a region; same-color blobs merged within ~300px.
   Excludes water/border/coast and the `#eead75` grid overlay.
3. `build_admin.py` / `build_townships.py` — rasterize per-country admin boundaries (+ China townships).
4. `describe_full.py` — intersect regions with admin units; positional phrases; roll up to a parent
   (prefecture/province) or `{parent} minus {child}` when covered; township detail for China city cores.
   Dominant units lead; small edge units are demoted to a trailing `(also …)`. Tunable thresholds at
   the top of the file (`FULL`, `PRIMARY`, etc.).
5. `finalize.py` — writes CSV + REPORT.
