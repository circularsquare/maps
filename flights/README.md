# flights

Interactive globe of estimated passengers per route, with an *unexpected
routes* engine (planned) for spotting diaspora corridors. AI-facing
architecture notes live in [CLAUDE.md](CLAUDE.md); this file is the runbook.

## Setup

Put `FR24_API_KEY=...` in `../secrets.env` (repo root, gitignored). No other
dependencies beyond Python stdlib + a static file server for the viewer.

## Run

1. **Pull a day** of FR24 departures — one airport at a time, resumable,
   credit-capped. Files land in `pull/<DAY>/<ICAO>.json`.
   ```
   # edit DAY (and HOUR_FROM/HOUR_TO if not pulling a whole day) at the top of fr24_pull.py
   python fr24_pull.py
   ```
   Full day ≈ ~273k credits, ~5h wall-clock. Safe to Ctrl-C and re-run — it
   skips airports already saved with `ok:true` matching the same window.

2. **Aggregate** raw pulls into `routes_fr24.json`, which the viewer reads.
   ```
   # set PULL_DIRS (list every pull/<DAY>/ folder) + WINDOW_HOURS in build_fr24.py
   python build_fr24.py
   ```
   YEAR_FACTOR auto-derives from `WINDOW_HOURS × len(PULL_DIRS)`, so adding
   more days only requires extending `PULL_DIRS`.

3. **View** — `fetch()` rejects `file://`, so serve over HTTP:
   ```
   python -m http.server 8000
   # open http://localhost:8000
   ```

## Deploy

The public copy lives at **`anita.garden/flights`**, served from the website
repo (`~/projects/website`, a Jekyll GitHub Pages site). The viewer is fully
self-contained: only `index.html` + `routes_fr24.json` ship — everything else
(MapLibre, fonts, basemap, FR24 data) is CDN. No build step here.

1. **Copy the two files** into the website's `flights/` folder:
   ```
   cp index.html routes_fr24.json ~/projects/website/flights/
   ```
   (After a fresh `build_fr24.py`, `routes_fr24.json` is the only file that
   changes; re-copy just it.)

2. **Commit + push** the website repo — GitHub Actions builds and deploys:
   ```
   cd ~/projects/website
   git add flights
   git commit -m "update flights map"
   git push
   ```
   Live ~1–2 min after the push.

The projects-page link lives in `~/projects/website/pages/projects/projects.md`
(under maps → interactive). Before sharing publicly, settle the FR24 attribution
+ license review in the TODO below.

## Known coverage gaps (2026-05-16 pull)

Some countries come back blank from the FR24 `flight-summary` API even though
they have real traffic. Checked both directions (departures pulled per airport
*and* arrivals seen landing from anywhere); a country is a "hole" only if blank
both ways. **None of this is fixable by the flight-count threshold — it's an
upstream FR24 data/coverage gap.**

- **Kuwait** — anomaly. Kuwait Intl (OKBK) returns `ok:true` / **0 flights in &
  out**, while every neighbour is fine (Dubai 360, Riyadh 344, Doha 233, Bahrain
  74 a 30-min hop away). Coverage in the Gulf clearly works, so this looks like
  an FR24 historical-data glitch for that one airport/day, *not* a coverage gap.
  Most likely date-specific — re-test on another day before backfilling.
- **Uzbekistan** (11 airports, incl. Tashkent/Samarkand/Bukhara) and
  **Kyrgyzstan** (Bishkek/Manas FRU, Osh OSS) — all blank. Systemic Central-Asia
  ADS-B/MLAT gap (no MLAT region, sparse receivers). FR24's *live* product does
  track these, so the data exists upstream of the summary API. Won't self-heal
  on another pull day — needs a schedule/topology backfill.
- **Ukraine** (13 airports, all blank) — blank for the *real* reason: airspace
  closed since the war. Correctly empty; do not backfill.
- **Thin / under-covered** (not blank, just sparse): Iran (33 of 43 airports
  empty), Mongolia (3 recs), Zambia (2 recs), Tajikistan (31 recs), Russia (35
  of 112 empty). Borderline real-low: South Sudan, Eswatini (0 dep / 3 arr),
  Nauru, Christmas Island.

## TODO

- [ ] Backfill the coverage holes above (Kuwait first — cheap to re-test;
      Uzbekistan/Kyrgyzstan need a schedule source). See methods discussion.

- [ ] Pull Sun 2026-05-17 (~273k credits — 345k remaining in budget).
- [ ] Unexpected-routes engine: GeoNames `cities1000` join → 3-bucket gravity
      baseline (<500 / 500-3000 / >3000 km) → residual list panel in the viewer.
- [ ] Recover the ~2k flights dropped to "unknown airport" — extend the lookup
      beyond OpenFlights (e.g. ourairports.com).
- [ ] FR24 attribution + license review before any public sharing (terms allow
      derived works + require attribution; raw-data redistribution prohibited).
- style pass, make coherent with other maps
