# train counts

A MapLibre map of average scheduled passenger trains per day, with line thickness representing frequency. The long-term scope is global, built country by country and system by system. **Finland is the first working pilot**, not complete world coverage.

## Run locally

From this directory:

```powershell
python -m http.server 8766 --bind 127.0.0.1 --directory dist
```

Open http://127.0.0.1:8766/. No frontend build or API key is required. MapLibre and OpenFreeMap tiles load over the internet. The map uses the same dark OpenFreeMap basemap and outlined-line direction as [japan rail flow](https://anita.garden/japanrail/).

Select full-week, weekday, weekend, or a single service day. Click a corridor for both-direction counts, directional breakdown, daily bars and operators. The thickness control adjusts the square-root width scale; the legend updates with it. Color indicates the dominant service category across the entire snapshot week.

## Rebuild the Finland snapshot

Python 3.9+; standard library only:

```powershell
python scripts/build_finland.py --download --start 2026-09-14 --days 7
python -m unittest discover -s scripts
```

Omit `--download` to reproduce from the cached `data/raw/finland.zip`. Choose another `--start YYYY-MM-DD` when refreshing; the script deliberately does not silently change the analysis window. It rejects dates outside the published feed interval, duplicate physical train numbers on a service day, empty service days and frequency-based feeds that this adapter does not support. The raw download is ignored by Git. Keep archived raw feeds externally for historical reproducibility. `metadata.json` records the input checksum, feed publication version, analysis dates and generation timestamp.

## Metric and method

1. Read the passenger feed **with pass-through points**. Do not substitute `gtfs-passenger-stops.zip`: express and stopping trains would then form different links and miss shared corridors.
2. Evaluate `calendar.txt`, then apply both additions and removals from `calendar_dates.txt` for each analysis date.
3. Collapse platforms to parent timetable points, retaining pass-through points. Sort by stop sequence and remove only consecutive duplicate points.
4. Count every active train traversal between adjacent points, retaining direction. Sum both directions for the map. Divide by all selected days, including zero-service days on an individual corridor.
5. Match the ordered points to the trip's GTFS shape. Collapse parallel tracks onto a representative corridor shape. The most frequent usable geometry is selected; endpoints use common parent coordinates to keep links connected. A missing or poorly matched shape is explicitly a dashed straight-line fallback.

The map uses `max(0.65, 0.48 × sqrt(trains/day)) × thickness` pixels. Width is intentionally square-root scaled for readability; it is not linearly proportional to trains. Zero-count corridors are hidden for the selected period.

Service dates are Europe/Helsinki departure dates, including trains with times after 24:00. This is **scheduled frequency**, not observed operations, ridership, capacity or seats. Coupled services are not reconciled beyond the source train number. No realtime cancellations are applied.

## Pilot coverage and limits

- Includes VR and HSL commuter rail in the Fintraffic passenger feed, plus any heritage trains present during the selected dates. Feed cross-border portions are retained.
- Metro, trams, freight and rail replacement buses are outside this first adapter's scope.
- Geometry is corridor-level, not a physical-track inventory. Parallel tracks are combined. Parent timetable points can be junctions or non-public locations.
- Different timetable-point sequences can leave shared corridors separate. No inferred shortest-path routing bridges those differences; network conflation needs a later audit.
- Monotone nearest-vertex matching uses a 1.5 km endpoint tolerance. Source geometry may be approximate even when accepted. The initial snapshot has 16 explicitly flagged straight-line fallbacks out of 454 corridors; geometry refinement remains work to do.
- The initial week is September 14–20, 2026, from the September 10 publication. It is a particular published schedule snapshot, not an annual average. Holiday and seasonal comparisons require more windows.

## Files

- `dist/`: authored static app and generated data; serve this folder.
- `scripts/build_finland.py`: Finland-specific GTFS adapter and reusable calendar evaluation.
- `scripts/test_counts.py`: regression fixtures for exceptions, bidirectional aggregation, platform normalization, pass-through points, and loop geometry.
- `dist/data/fi/metadata.json`: source and metric provenance.
- `dist/data/fi/segments.geojson`: per-corridor daily and directional counts, operator totals, route labels and geometry quality.
- `dist/data/fi/stations.geojson`: points with passenger pickup/drop-off in active trips.

## Expanding country by country

Finland was selected because one official open feed provides passenger schedules, railway shapes and intermediate pass-by points, without registration. The USA remains a useful next target, but start with Amtrak as an **operator pilot**, then add commuter operators and urban rail with explicit coverage labels. Do not label an Amtrak-only layer as all US passenger rail. The Amtrak ArcGIS GTFS directory inspected during setup redirected to sign-in; a reusable downloadable schedule feed and its terms still need verification.

For every new country/system, first record publisher, exact feed URL, license, timezone, feed window, rail route types, identifiers and coverage gaps. Then implement an adapter yielding the same GeoJSON + metadata contract. Calendar handling is reusable; geometry matching and physical-service deduplication are source-specific. Feeds without pass-through points require routing express trains over a railway graph before shared segment counts are trustworthy. Feeds using `frequencies.txt` need explicit frequency expansion. Overlapping operator feeds, cross-border trains, coupled services and seasonal coverage need deduplication policies before merging. Never sum separate national feed totals blindly.

Next Finland improvements: refine the 16 geometry fallbacks, audit differently segmented shared corridors, then broaden to a longer reference period. Next country should be chosen only after checking an actual downloadable feed and geometry quality.

## Sources and attribution

- [Fintraffic railway data documentation](https://www.digitraffic.fi/en/railway-traffic/)
- [Passenger GTFS including pass-by points](https://rata.digitraffic.fi/api/v1/trains/gtfs-passenger.zip) — Fintraffic / Digitraffic, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Derived aggregation and representative geometry by this project.
- [OpenFreeMap](https://openfreemap.org/) / OpenStreetMap contributors — basemap; map attribution is displayed.
- [MapLibre GL JS](https://maplibre.org/maplibre-gl-js/docs/) — renderer, pinned to 4.7.1 to match the reference map.
- [japan rail flow](https://anita.garden/japanrail/) — visual reference, not a schedule data source.

This project is local. No hosting or deployment has been configured.
