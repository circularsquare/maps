# citypop — Stadestér evaluation (2026-07-08)

Interactive historical world-city population map. First task per research brief:
evaluate Stadestér 1.0 → build-on / augment / replace.

## Verdict: BUILD ON Stadestér.

It is ~80% of the merge/reconcile job the brief hoped for. It already delivers the
exact structure we want: per-city **annual** population series + lat/lon + provenance.

## What we actually have

Zenodo release (17180328) is **raster-centric** (population grids in PNG, 130MB+ zips) —
the city point data is buried in JSON inside those zips. The clean city-level data lives
in the **GitHub repo** instead:

- `input/uud/stadester_cities.json` (41MB, 24,219 cities) ← downloaded
- `input/uud/processed_stadester_cities.json` (27MB, 22,411 cities) ← downloaded, ~equivalent
- These are the populstat+buringh+chandler+devries merge, annually spline-interpolated.
- The full "41,214" figure adds GHSL modern cities + raster metro-adjustment (not in these files).

### Schema (per city)
```
"Name-Country": {
  name, province, other_names[], elevation,
  population: { "1950": 65000, ... },   // annual, interpolated
  coords: [lat, lon], country, key, type   // type = provenance source
}
```

### Coverage density by benchmark year (from stadester_cities.json)
| year  | #cities | #≥10k |
|------:|--------:|------:|
| -3000 |       6 |     6 |
| -1000 |      32 |    32 |
|     1 |      20 |    20 |
|   500 |     116 |   116 |
|  1000 |   1,398 |   236 |
|  1500 |   2,267 |   522 |
|  1800 |   2,730 | 1,156 |
|  1850 |   3,744 | 1,890 |
|  1900 |   9,107 | 4,511 |
|  1950 |  14,313 | 9,780 |
|  2000 |  13,540 |12,137 |
|  2020 |      31 |     3 |  ← tail-off, see issue 4

**Takeaway:** ~1000AD onward is genuinely dense (thousands of cities) — far beyond the
"~10 cities in 1850" sparse viz the brief complained about. Pre-500AD is inherently thin
(the underlying Chandler/Modelski scholarship is thin; no dataset fixes that).

### Historical fidelity spot-checks (good!)
- Baghdad: 1000→1.1M (peak Abbasid), 1500→90k (Mongol sack), 2000→3.8M. Correct shape.
- London: 1000→38k, 1800→863k, 1900→4.5M, 2000→7.1M.
- Xi'an/Chang'an: yr1→420k (Han), 2000→6.2M.
- Beijing: 1500→671k, 1800→1.14M, 2000→11.7M.

## Known issues → cleaning plan (all tractable)
1. **Agglomeration vs city-proper duplicates.** 1,116 parenthetical entries, 711 "(agglomeration)".
   Need one definition rule + dedup/merge. → propose: prefer agglomeration/metro where a variant
   exists, merge its series into the base city, else city-proper.
2. **Geocoding fallback collisions.** ~1,319 coord cells shared by >1 entry; some are legit
   agglomeration pairs, some are multiple towns dumped at one provincial centroid (bad geocode).
   Minor visually; flag worst offenders.
3. **Corrupt year keys:** exactly one (`19690310`). Clip years to [-4000, 2030].
4. **Modern tail missing.** Series mostly end ~2000–2003; interpolation stops there
   (only 31 cities at 2020). Graft 2001–2025 from a clean modern source (UN WUP 2024 /
   GHS-UCDB (already in repo, 9.2MB) / citypopulation.de).
5. Mojibake was a **console-display artifact only** — underlying JSON is clean UTF-8. Not a data issue.

## Next: pipeline
build.py: load → clip bad years → dedup agglomeration → graft modern tail → emit compact
per-year city array (name, lat, lon, pop, provenance) → self-contained index.html with a
year slider/play + sublinearly-scaled population bubbles (à la flights/index.html).
