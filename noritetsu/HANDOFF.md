# Handoff

Written 2026-09-30. `spec.md` is the design and §13 is the todo; `README.md` is how to run
things. This file is for picking the project up cold, and for **two people working on it at
once without colliding**.

## Where it is

Japan is built end to end and the tracker works. 1,101 lines of which 593 are register lines
(28,156 km), 9,072 stations, a 10.5 MB tile archive, and rides recorded from a strip diagram or
from the map with percentages per line, operator and overall.

**Switzerland is built too** (2026-09-30): 779 lines of which 402 are register lines
(5,567 km), 2,816 stations, a 3.0 MB tile archive. The app loads it when the map moves
over Switzerland (`?region=ch` opens there), as it does every built country listed in
`dist/regions.json`.

The `.osm.pbf` extracts are deleted once extracted and checked; `data/proc/<region>/` holds
what was pulled out of them, which is all a rebuild short of re-extracting needs. Fetch the
extract again from Geofabrik to re-extract (the plain `-latest` URL for Switzerland was
redirect-looping on 2026-09-30; the dated `switzerland-YYMMDD.osm.pbf` worked).

```powershell
python serve.py                                                  # http://localhost:8767
python extract.py --region jp --pbf data/raw/japan-260928.osm.pbf
python build_model.py --region jp --register n02:data/raw/N02-24_GML.zip    # 90 s
python build_tiles.py --region jp        # AFTER build_model: it reads line colours from it
python check_model.py --region jp

python extract.py --region ch --pbf data/raw/switzerland-260929.osm.pbf      # 1 min
python build_model.py --region ch --register schienennetz:data/raw/schienennetz_2056_de.gdb.zip   # 50 s
python build_tiles.py --region ch                                            # 30 s
python check_model.py --region ch

python -m unittest discover -s tests
node tools/lint_map_expressions.js dist/index.html
```

The Swiss register needs two files in `data/raw/`, both open data, URLs in the docstring of
`schienennetz.py`: the BAV network (`schienennetz_2056_de.gdb.zip`, 3.4 MB) and the national
service-point list (`ch_servicepoints.csv`, 25 MB), which is what says which network nodes are
passenger stops.

## Next session: Korea (agreed 2026-09-30)

The country after Switzerland. Leads, none of them tried yet for this project:

- **OSM half**: Geofabrik `asia/south-korea-latest.osm.pbf`, a few hundred MB. Run
  `inspect_region.py` before trusting it, as §12a asks. The dated URL worked for
  Switzerland when `-latest` redirect-looped.
- **Register**: there is no open national line-geometry register we know of (the
  government portals want a Korean ID; see the maps memory on Korean open data). What IS
  open, no login: KRIC's rail portal (data.kric.go.kr) has station codes, names and line
  membership nationwide, and `riders/koreariders/` already reads Korail's 철도통계연보
  yearbook, whose per-line 영업거리 is the published length `check_model.REGISTER` needs.
  So the likely shape: lines and stations from KRIC plus the yearbook, geometry from OSM
  track (the `TrackGraph` tracing already built for straight-line gaps can lay a register
  line's sections along OSM track between its stations). That is a reader with no geometry
  of its own, which neither `n02.py` nor `schienennetz.py` is, so expect to extend the
  contract.
- `riders/koreariders/README.md` has the gotchas on these sources (station-name
  truncation in KRIC files, two stations sharing a name, edition changes in the yearbook).
  Read its "Picking this up" section first.
- Add `kr` to the app's `dist/regions.json` once built (the app track's file).

Smaller data items still open, all in spec §13: a per-line list of the 20 remaining
straight-line fallbacks for the app; Swiss sub-kilometre km-lines cluttering line lists; the
shorter-route-only rule when a km-line has two routes between stops.

## Between the two tracks

Open, for the app: `looksStraight` in `index.html` guesses a straight-line fallback from
geometry (over 2 km, within 1% of crow-flies). The build now traces all but 20 of those along
track, so the guess mostly flags real, straight track such as the Shinkansen. Say if a
per-line list of the true fallbacks would help and the build will write one.

Settled 2026-09-30, for the record:

- **The app reads**: `aliases.json`'s `lines` (rides on an OSM line dropped as its register
  twin move to the register line), any `src` but `osm` as a register, `j` stations as
  diagram rows that are not bubbles, search results or stops, and `REGIONS` in `index.html`
  for the countries (add a line there for each new one). A track click opens the register
  line when it is the only one there. Verified headless: a ride saved on the old OSM Tozai
  line id migrated with none lost, and a San'in click opens 山陰線 with スーパーおき under it.
- **Heavy rail now credits across the two sources.** `build_credits` compared raw kinds, and
  OSM train routes are `train` where register lines are `rail`, so no OSM train route had
  ever credited a register line: riding the Yamanote as operated counted nothing towards
  山手線. It now compares `kind_family`. That let every Shinkansen service credit the
  conventional line beside it, about 1,600 km of false credit over 16 services, so
  high-speed and conventional sections no longer credit each other
  (`section_highspeed`: OSM sections by the `highspeed=yes` ways they lie on, register
  sections by their line's `highspeed` flag). What is left, 108 km on the Ou Line, is the
  Tsubasa and Komachi really running on it. IC 2 Zürich to Lugano now completes the
  Gotthard and Ceneri base tunnels.
- **Tiles carry `c`**, the colour of the line on each piece of track: the register line's,
  else the commonest among OSM lines, lines before named trains. **So `build_tiles.py` now
  runs after `build_model.py`.** The unused `station` tile layer is gone.
- **Fewer duplicate lines.** A name match with the register now stands when the operators
  are spelled differently but most stations are shared (OSM's "Tokyo Metro" against the
  register's 東京地下鉄 kept the Hanzomon Line listed twice). OSM lines that duplicate each
  other merge too (`merge_osm_twins`): the same through service mapped twice, and direction
  pairs never put under a route_master. 40 in Japan, 5 in Switzerland. OSM line names lose
  their direction part, "(Shibuya -> Chuo-Rinkan)", unless that would give two different
  lines one English name, in which case the one that lost its direction shows its native
  name. Station names are compared through `station_key` (NFKC, the three small-ke
  spellings folded: 市ケ谷 = 市ヶ谷) and, for CJK names close together, one ending or
  starting with the other (新線新宿 = 新宿); without both, the Toei Shinjuku Line stayed
  listed twice. Every dropped id is in `aliases.json` `lines`, so the app's migration
  covers it.
- **Straight-line sections are traced along track** (`TrackGraph` in `build_model`): 471 in
  Japan down to 20, 14 of them named trains; 10 left in Switzerland. JR中央線快速 Tokyo to
  Shinjuku is 10.1 km of track, where it was a 6.1 km chord.

`check_model.py` is the one that matters: it compares built line lengths against published
figures. Register lines should stay near 1.00 except JR trunk lines, which run 5-20% long for
a known reason (§13). OSM-derived objects should stay within 2%.

## Two tracks, and who owns what

**Do not both edit `build_model.py`.** It is the only file both tracks have reason to touch.

| Track | Owns | Must not touch |
|---|---|---|
| **Countries and data** | `n02.py` and new per-region readers, `extract.py`, `build_tiles.py`, `check_model.py`, `inspect_region.py`, `probe_*.py`, `data/` | `dist/index.html` |
| **Styling and app** | `dist/index.html`, `tools/` | the Python build |

`build_model.py` belongs to the countries track. If the app track needs a new field in the
output, ask for it rather than adding it.

## Adding a country

The pipeline is region-agnostic; only the register reader and the check tables are not.

1. **OSM half.** Get the Geofabrik extract, then
   `python extract.py --region <cc> --pbf <file>` and
   `python inspect_region.py --region <cc>` before trusting any of it. That report is how you
   find out whether `usage` tags are kept in that country, how many route relations carry a
   colour, and how much of the network a passenger route relation actually covers. Delete the
   `.pbf` afterwards; keep one at a time.
2. **The register.** §12c lists where each country's authoritative line inventory comes from:
   FRA NARN for North America, ERA RINF for the EU, GTFS via the Mobility Database as the
   broad fallback, Wikidata as the glue. Write a module beside `n02.py` exposing:

   ```python
   def build(path, log) -> (lines, stations, geoms)
   ```

   and run it with `--register <module>:<path>`. Nothing else in the build needs editing.
3. **Check it.** Add the country's published line lengths to `REGISTER` in `check_model.py`
   and make them pass. A build with no outside number checked against it is not finished.
4. **Put it on the map.** `python tools/build_regions.py` once, after the first build. The
   app loads each country as the map moves over it, and `dist/regions.json` is how it knows
   where each one is before loading any of it; a country missing from it only loads when
   someone has rides there. The outline comes from religiondots' `country_shapes.geojson`.

### What `build()` has to return

`lines` — a list of dicts:

```python
{"id": str,            # stable across builds; hash the name and operator, do not use an index
 "src": "n02",         # any non-"osm" value marks it a register line, which is what counts
 "service": False,     # True only for a named train rather than a line
 "name": str, "name_en": str, "ref": str, "colour": str,
 "operator": str, "operator_en": str, "network": str,
 "kind": str,          # rail | subway | light_rail | tram | monorail | funicular | narrow_gauge
 "km": float,          # the sum of its sections
 "variants": int, "straight_sections": int,
 "display": [station_id, ...],                  # reading order for the strip diagram
 "sections": [[station_id, station_id, km], ...]}   # a 4th element is added by build_model
```

Optional line fields:

- `"highspeed": bool` — only if the register knows. Where given, a register line only matches
  OSM ways with the same `highspeed=yes`, which keeps the Shinkansen and the conventional line
  beside it apart. Omit it and nothing is filtered.
- `"km_official": float` and `"chain": {"a|b": km}` — the register's own chainage, per
  section. `check_model` then compares every line against it, not just the handful in
  `REGISTER`.

`stations` — a dict keyed by station id:

```python
{"id": str, "name": str, "name_en": str, "lon": float, "lat": float, "lines": set()}
```

plus `"junction": True` on a section end that is not a stop. A section with a junction at
either end is kept only if OpenStreetMap passenger routes run over at least half of it
(`build_model.drop_unridden_sections`); that is how base tunnels stay and freight curves go.
A section between two stops is never questioned.

`geoms` — `{line_id: {"<from>|<to>": [[lon, lat], ...]}}`, one entry per section, keys matching
`sections`.

`build_model.merge_sources` does the rest: it matches OSM stations onto register stations by
name and distance (then a second pass for compatible spellings — "Bellevue" onto "Zürich,
Bellevue", 本町3丁目 onto 本町三丁目), hands OSM colours and English names to register lines,
drops an OSM line that is its register line twice over, keeps the rest as operating patterns
and named trains, and then computes the corridor credits. `register_way_lines` then ties
register lines to the OSM ways on the map, and corrects a register line's kind from the
track it lies on (N02 has no subway code, and legally every Osaka Metro line is a tramway).

### What Switzerland taught about registers

`schienennetz.py` is the second reader and differs from `n02.py` in ways the next country
will probably share:

- **A register line need not start at a station.** Swiss km-lines start at junctions as often
  as not; the Gotthard base tunnel has no stop at all. Every end of a line is therefore a
  section end, flagged `junction`, and OSM decides which of those sections passengers ride.
- **Which nodes are passenger stops came from a second file**, the service-point list. A
  platform group ("Chur [Gleis 10-14]", "Visp MGB-bvz") is a node of its own and has to be
  resolved onto its station by name.
- **Names differ in spelling, not just in script.** "Zürich, Bellevue" against "Bellevue".
- **The register's own chainage can jump inside a segment**: Zug Nord to Zug spans 10.1 km of
  chainage for under a kilometre of track. The nine Swiss lines `check_model` lists as off
  their chainage are that, not the build.

### Two traps that cost a rebuild each

Both are in `n02.py`, with the numbers they produced:

- **A line's track is a graph, not a chain.** Merging double track end to end gives a run that
  goes out on one rail and back on the other, so slicing between two stations traverses both.
  The San'in Line came out at 1,330 km against a published 674.
- **Sections come from an absorbing search, not from an ordering.** Ordering stations by
  distance from a terminus cannot work on a loop, because there is no terminus. The Oedo Line
  came out at 178 km against 40.7. A Dijkstra from each station that stops the moment it
  reaches another station gives the inter-station sections directly.

## Working on the app

`dist/index.html` is one file: constants, map setup, data loading, rides, crediting, then the
panel. Things that will bite:

- **Paint expressions must stay plain literal `const`s in SCREAMING_CASE.**
  `tools/lint_map_expressions.js` evaluates each `addLayer` object with those hoisted and
  reports anything it cannot evaluate as unchecked. Run it on every edit; it catches an
  invalid expression that would otherwise silently drop a whole layer.
- `['*', ['interpolate', ['zoom'], ...], factor]` is invalid. `['zoom']` may only be the
  direct input of a top-level `interpolate` or `step`.
- Added layers anchor at the first symbol layer **after the last non-symbol layer**, not at
  the first symbol layer, which in this basemap is a water label sitting before the roads.
- The basemap draws its own railways in near-black; they are hidden on load.
- Station bubbles come from the model, never from the tiles.
- No CSS transitions anywhere: state changes are instant.
- **The native name is the real name.** Someone riding trains in a country reads the names on
  the trains, and those are not in English. `lineName()` and `stName()` show an English name
  where one exists and fall back to the native one, and that fallback is correct behaviour,
  not a gap to paper over: never substitute an id, a transliteration or a placeholder for a
  name the data has in its own script. 61% of register lines have no English name and the app
  has to read well anyway.
- To see a change: `python serve.py`, then drive a real browser —
  `node tools/screenshot.js http://localhost:8767/ out.png 15000 probe.js` with Chrome started
  headless on port 9222 with its own `--user-data-dir`. The header comment in
  `tools/screenshot.js` has the exact invocation. A probe file can both drive the map and
  assert against it, which is how every UI change here has been checked.

## Data files, and what is gitignored

`dist/data/<region>/` holds `lines.json`, `stations.json`, `credits.json`, `ways.json`,
`aliases.json` and `geom/<line>.json`; `dist/data/<region>.pmtiles` is the tile archive. All of
it is generated and excluded by the repo's `data/` rule, as are `data/raw/` and `data/proc/`.
Only source is tracked.

**`aliases.json` matters more than it looks.** Station ids move when a build merges OSM onto a
register, and a rider's saved trips name stations by id. The file ships the moves and the app
migrates saved rides on load. A change that moves ids without updating it silently voids
people's history. Its `lines` map does the same for line ids dropped as a register twin.
