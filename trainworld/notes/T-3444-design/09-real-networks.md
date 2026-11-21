# T-3444 design, part 9: real networks

Agent tdperf, 2026-11-21. Design draft; nothing run. Vocabulary as in `01-model.md`, stages as in
`10-build-plan.md`. Sources read: `docs/code-map.md` 595-631, the docstrings of `pipeline/realnet.py`,
`realauto.py`, `realcorridors.py`, `realshared.py`, `reallevels.py`, `realgraph.py`, `sim/src/bin/realsave.rs`,
`noritetsu/extract.py`, and research/realworld.md for what OSM and GTFS hold.

## How they are built today

41 shipped starts in `data/packs/real/` (`realnet.py SHIPPED`: seven made by hand, nyc lon par tyo seo tor sdj, plus
34 from `realauto.json`).

| Step | What it does |
|---|---|
| Sources | The seven: GTFS (New York, Toronto, Paris, London's National Rail), timetabled trip CSVs (Seoul, London's TfL, Tokyo), OSM route relations by Overpass (Sendai, with `stop_area` relations). The 34: noritetsu's local OSM extract cut to the city (`realauto.py extract`), route relations only |
| `realnet.py net` | Stations and lines, each station under a **key**. GTFS: the parent station, or a group table where the city has one (`load_gtfs`): New York's is the MTA **Complex ID** (`MTA_GROUP`), Toronto TTC groups, London CRS, Paris ZDC. OSM: the `stop_area`, else the stops of one name within 400 m (`load_osm`); the 34 have no stop areas, so always the name rule |
| `geom`, `route` | Each hop's shape, routed over the real-track lines graph (`realgraph.py`, `realroute.py`) |
| `realcorridors.py` | Parallel chains become corridors, cut at stations and where the lines change. Each piece gets a number of **pairs** (double tracks) from its load and OSM's parallel chain count (`tracks`, the median), the lines on each pair and a role (express, local) |
| `realshared.py` | The track model: one double edge per pair, pairs 8 m apart; a station node per pair group with stops, several side by side where trains need more platform tracks (`PLAT_LOAD` 0.8); junction nodes past the platforms; `fabric` and `fan_out` so no node has more than 8 ends (`MAX_PORTS`); a short dead end where a line turns back at no stop (T-3790) |
| `reallevels.py` | Node and edge levels from OSM `layer`, `tunnel`, `bridge` |
| `realsave --shared` | Traces each edge at 3 m, marks every edge `sep`, makes every dead-end station a terminal, settles and writes `LRT1`, so a start loads settled (T-3233) |
| `gamesave`, `ship` | The game file, copied to `data/packs/real/<id>.save` |

The real NYC test save (`T-007-real-nyc.save`, the older per-line build) has every line on its own double track and
its own station nodes: 109 lines, 1,756 station nodes, one complex per real station.

## How they build bundles and platforms

The pipeline already holds most of what the new records need; only `realshared.py` and `realsave.rs` change, and they
must (realsave builds `EdgeData` and `StationData` directly, which go at stage 3-4).

**Stage 3 (bundles and tracks).**
- Each piece becomes **one bundle of 2 x pairs tracks**, pairs side by side as today: ways A B A B by running side,
  slots at -6, -2, +2, +6 for two pairs (`PAIR_GAP` 8 m is 01's default). The pipeline knows the pairs, so there is
  no geometry guess.
- A line's path `[piece, +1/-1, pair]` becomes a span (bundle, slot): the pair and direction pick the slot.
- Nodes as today. Where a line changes pair at a node (today a junction there), the bundle's node gets a stored join
  from the old pair's track to the new one. That is exact, since the pipeline knows the move.
- Stations stay on nodes at stage 3 (part 7's station section), as the game's do.

**Stage 4 (platforms and stations).**
- **One station per corridor node with stops**, named from its key. A key is often a whole complex (New York's MTA
  Complex ID, Paris ZDC, London CRS, a 400 m name cluster), so the key alone would make Times Sq or every operator
  at Shinjuku one station, against 02. The corridor node is 02's rule already applied: the pairs whose platforms lie
  side by side at one place. So all pairs at one node (local and express) are one station with platforms on each
  served track (zero-width side platforms outside each, length from the longest train, `platform_len`), and the
  key's other nodes (IRT and BMT at Times Sq, lines on other corridors) are other stations of the same name,
  grouped by the complex rule as today. Today each pair group at a node is its own station node, named alike.
- **Station nodes go.** A station no longer splits the bundle, so pieces cut only at stations (`realcorridors.py`'s
  cuts "at stations") can be joined across the station wherever the lines and pair count match on both sides.
  Junction nodes past the platforms stay.
- **Extra platform tracks** (`PLAT_LOAD`, London Bridge, Clapham Junction): today several station nodes side by
  side between two junctions. They become a bundle of 2 x n tracks between those two junction nodes, with a
  platform per track. They are already real track, so the mapping is direct.
- **Complexes stay derived** by the game's rule (50 m, passages, buildings). 02's "a direct copy" does not hold as
  written: our keys are often complexes already, stations come from corridor nodes, and the game stores no complex.
  OSM's `stop_area_group` (7,123 worldwide) is too thin to import anyway.
- Terminals: every dead-end station keeps the terminal rule until stage 7 decides presets.

**Stage 7 and later.** Turn-back stubs could become crossovers on the bundle (stage 8). Express/local roles could set
AA/BB layouts where OSM shows the track order (stage 12, below).

**The node end cap.** `fabric` and `fan_out` exist only because a game node takes 8 edge ends. With bundles, a
4-track piece is one end, not two, so far fewer junctions need splitting. 01 states no cap for bundle ends: the lead
should say whether `MAX_PORTS` 8 counts bundle ends or track ends. Bundle ends is recommended; then most of `fabric`
can go.

## What OSM and GTFS offer

From research/realworld.md (taginfo, 2026-11-21) and what our data path keeps:

| Data | Worldwide | In our path today |
|---|---|---|
| `railway=platform` | 196,741 (about 1.4 per station or halt); `ref` on 33%, `local_ref` 6%, `level` 8%, `layer` 9% | not extracted: noritetsu's `TRACK` set leaves platforms out on purpose |
| `railway:track_ref` (track number in a station) | 455,649 ways, at most ~16% of rail ways | dropped: not in noritetsu's `WAY_TAGS` |
| `railway:preferred_direction` | 338,064 ways | dropped |
| `tracks=N` on a way | (not counted) | kept in `WAY_TAGS`; corridors count parallel chains instead |
| `service=crossover`, `siding`, `yard` | 113k, 386k, 688k | `service` kept; the lines graph leaves sidings and yards out (`realgraph.py`) |
| `public_transport=stop_area` | 586,670, all modes | the seven only, through Overpass; the 34 have none |
| `stop_area_group` | 7,123 | none |
| GTFS `location_type` 1 station holding 0 platforms, `platform_code`, `pathways.txt`, `levels.txt` | per feed, not global | the seven's GTFS feeds; parent stations not used for platforms |
| MTA stations and complexes | 496 rows, 493 station ids, 445 complexes; New York only | `load_gtfs` keys New York's stations by complex id (`MTA_GROUP`); `targets` reads them for ridership |

No GTFS or NeTEx term defines a track; the platform is the finest unit. OSM is the only global source of track order
and platform shape.

## Import now, later, or not

Per the game principles: the whole world first, global sources first, a one-city source only if very easy.

| What | When | Why |
|---|---|---|
| Bundles from pairs, joins at pair changes, one station per corridor node with stops, platforms on served tracks, widened bundles for extra platform tracks | **Now** (stages 3-4) | Needed for the format change; no new source; every city alike |
| Platform shape from OSM `railway=platform` areas: island or side, by which tracks the area lies between, decided once at import (02's "geometry decides only at placement, once") | **Later**, stage 12 | Global, but needs platforms in the extract |
| Track order and ways from `railway:track_ref` and `preferred_direction` (AA/BB against ABAB on real quad lines) | Later, stage 12 | Global but thin (16% of rail ways at most); needs the extract change |
| Stop areas for the 34 extract cities | Later, stage 12 | Better station keys than the 400 m name rule; needs the extract change |
| Track counts above what lines need (`tracks`, parallel chains) | Later | Extra tracks with no line add cost and nothing a player uses yet |
| GTFS `platform_code`, `pathways`, `levels` | Not planned | Per feed, the seven only. `platform_code` is a label, which the game does not show |
| MTA station and complex ids | As a **check**, not an import | One city. Compare the game's derived complexes on New York with the MTA's 445 |

**The extract change.** Platforms, stop areas and the two track tags come from noritetsu's extract, which belongs to
another project and is read-only from trainworld (`realgraph.py`: "never writes there"). So stage 12 either asks
noritetsu to keep them (`WAY_TAGS` plus `railway:track_ref`, `railway:preferred_direction`; platforms as a new
kind; `stop_area` relations) or runs a trainworld-owned extract from the same PBFs. The re-extract's run time is
unknown; measure one region first. That is why all of it is later.

## Rebuilding the shipped starts

The corridors and network JSONs are not in `data/work/realnet/` on this machine (`REALNET_OUT` staged them elsewhere,
or they were cleared), so a rebuild is a full pipeline run:
- The 34 extract cities: `realauto.py build --jobs 4`, about 25 min for 30 (T-3225), then `realauto.py ship`, which
  refuses a city that loses a route or station the shipped save serves. That refusal is the gate.
- The seven: `realnet.py net`, `geom`, `route`, corridors, `realsave`, `gamesave` per city. `realsave` itself takes
  30 s for London and 3 s or less for the rest (measured, T-3233). Their raw inputs look present
  (`data/raw/t509/` has gb, idfm, tor, tyo, `osm_sdj.json`; `data/raw/gtfs/`); a dry `net` per city confirms that
  before stage 3 promises a rebuild.
- If the staged JSONs turn up under `REALNET_OUT`, only `realsave`, `gamesave` and `ship` need running (minutes).
- Per stage that changes the format (3, 4, 7): 1-2 h of agent time plus the machine time above. Stage 12's real
  import (the extract, platform shapes, track order) is its own 8-14 h, after the extract is measured.
- A look per rebuilt city as T-3225's hand check (a screenshot at zoom 12, the lines list), and R7 from part 8 (London
  start 1 s or less natively).

## Her calls

- Real networks import **one station per place where tracks run side by side**: a 4-track stop's local and express
  pairs become one station with four platforms, instead of today's two stations named alike in one complex.
  Separate lines under one name (Times Sq's IRT and BMT) stay separate stations in one complex. The circle and the
  name stay one per complex. [Yes.]
