# Roads at 1 km

Roads are their own layer, from OSM, on the 1 km grid of our surface (`docs/surface.md`), so "no
roads" is a toggle and 10,000 BC has none. The road kinds are the 100 m level's (`docs/fine.md`,
"Roads: OSM"), and both levels come from one pass over the extracts. The page reads it at L1, L3
and L10 (T-028). Measurements: `notes/T-010.md` (the rule), `notes/T-053.md` (the world pass),
`notes/T-028.md` (the page).

## How a road changes a cell

MAP walks a whole 1 km pixel at a flat 5 km/h when any road crosses it, whatever the slope.
Against 100 m with OSM roads, MAP's L1 is 2-50% off in median on the test tiles and up to 1.5x
too fast. Two things add up there: a cell holds a few roads and a lot of fields, and on a
hillside the 100 m level walks its roads at 5 km/h times the terrain factor, as it walks
everything else.

The rule:

- **s** is the share of the cell's 100 x 100 m cells that a walkable way (path, local or main)
  passes through, 0 to 1. It is the 100 m level's own road cells counted per 1 km cell.
- **e = min(1, 3 s)** is the cell's effective road share. A walker at 100 m strings roads
  together and keeps to them, so a cell's roads serve about three times their own area; a cell
  a third covered by roads is walked as all road.
- The cell's crossing time blends road and off-road time by e:
  `time with roads = e x road time + (1 - e) x off-road time`, road time at 5 km/h times the
  cell's terrain factor, as at 100 m. Since off-road speed is the land cover speed times the
  same terrain factor, the ratio does not depend on terrain:
  `time with roads / time without = 1 - e + e x lc / 5`, lc the cell's land cover speed
  (`lc_speed.tif`, at most 5).
- **Bridges**: a water cell (no land speed) that a walkable bridge crosses is walked at 5 km/h
  times its terrain factor, as at 100 m. Ways that are not walkable (motorways, `foot=no`)
  change nothing; tunnels are left out.

Checked on eight test tiles (Cluj and New York, four each) the rule sits within a few percent of
100 m, against 2-50% for MAP's L1. Towns stay right: urban land is 5 km/h either way. What is
left in rural Romania is the off-road land cover mosaic, which a 100 m walker threads (grass
beside fields) and a 1 km mean cannot; it is there with no roads at either level too. Numbers by
tile: `notes/T-010.md`.

## What the layer stores

**The OSM part**, built once (`tools/roads/build.py`), in `data/roads/`:

| File | Per | Bytes | What |
|---|---|---|---|
| `bits/<tile>.bits.gz` | 100 m cell, 1 x 1 degree tile of 1200 x 1200 | 1 | road bits as the fine level's `.roads`: 1 path, 2 local, 4 main, 8 not walkable, 16 walkable bridge |
| `l1/<tile>.bin.gz` | 1 km cell, 120 x 120 | 2 | count of 100 m cells with a walkable way (0-100), then the OR of the cells' road bits |

`l1/` covers 16,183 one-degree tiles, 85 MB; `bits/` is kept too: it is where the fine level's
`.roads` tiles are cut from (T-057), so any tile can later get 100 m without fetching OSM again.
The page build reads `l1/` tiles directly, so no GeoTIFF of it is made.

**The page part** (`tools/build_tiles.py`, with the levels; docs/resolution.md): one byte per
cell at every level, beside the off-road speed, in `data/tiles/r1/`, `r3/` and
`data/grid_roads.bin.gz`.

- **On land**: the cell's time with roads over its time without, x 250 (1-249; 0 = no road or
  no change, as in towns, where land cover is 5 km/h already).
- **On water**: a walkable bridge's speed over 5 km/h, x 250 (1-250), which makes the cell land.
- **L1**: the rule above, cell by cell; a small-water cell blends with its 1 km/h, a cell LC100
  leaves as sea with its 3 km/h (docs/resolution.md, "Water at L1").
- **L3 and L10**: a coarse cell's speed is the mean time of its L1 cells, so its road byte is the
  mean time with roads over the mean time without. Bridge cells join the mean with roads, at
  their speed, and count toward the quarter-land rule: a coarse water cell they make land takes
  the water byte. The coarse map with roads is then exactly the average of the L1 one.

The worker folds the byte into the speed as it loads a level (`applyRoads`, `js/tiles.js`):
speed x 250 / byte on land, byte x 0.8 (km/h x 40) on water. Everything downstream (search,
hazard per km, canoe, paint) reads the folded speed. `cfg.roads` off, as "no roads" and 10,000 BC
will be, skips the road files: they are never fetched.

The byte depends on land cover, so it is rebuilt when the land cover table changes; it does not
depend on terrain, so T-011's 90 m slopes leave it as it is. The OSM part depends on neither.

## Fetching the world

The planet file (~90 GB) does not fit on C: (shared with other projects). The pass
(`tools/roads/world.py`, running as T-053) goes extract by extract:

1. **The list** (`tools/roads/regions.py`, into `data/roads/regions.json`): the leaves of
   Geofabrik's tree (index-v1.json, with outlines), minus groupings mostly covered by smaller
   leaves (US census regions, `us`, Alps, DACH, Britain and Ireland, Great Britain, south-east
   Asia, South Africa and Lesotho), plus `central-america`, the only extract covering the Lesser
   Antilles and Cayman. Europa Island (one tile, no roads) is left out. **513 extracts,
   84.6 GB**, the largest 1.2 GB (Quebec, Ontario, Sweden, Java). Every other land tile of our
   grid is in at least one; 29% are in two or more.
2. **Each extract**: download (`tools/roads/fetch.py`, md5 checked), `build.py extract <name>`
   (every highway way drawn at 100 m and OR-ed into the tiles it touches), then the extract is
   deleted. The next extract downloads while one is read, so at most two are on disk (2.4 GB).
3. **Borders**: an extract holds the whole of every way crossing its outline, and Geofabrik's
   outlines overlap a little, so a border road comes in both neighbours. OR-ing road bits is
   the same however often a cell is drawn, so it is never counted twice and never lost. A way
   with a node the extract lacks keeps its runs of located nodes on each side (none in the test).
   A segment over 180 degrees is not drawn.
4. **Resuming**: a finished extract goes in `data/roads/done.txt` and is skipped. A stop midway
   through an extract redoes that extract; tiles it half wrote are harmless, again because of OR.
   Each tile file is written to a temporary name and renamed.
5. **At the end**: `build.py reduce` (1 km counts and kinds per tile), then the page levels
   (`tools/build_tiles.py`).

**Time**: reading is ~5 min per GB of extract on one core (Romania 333 MB in 93 s; four US
states, 1.2 GB, in 6.4 min), so **~7-8 h** for the world on one core (`--jobs 1`); downloads
overlap it. **Disk**: at most two extracts plus `bits/` and `l1/`, ~4 GB at the peak; the pass
stops when a download would leave under 6 GB free on the drive (`--min-free`).

**The 100 m tiles** (T-057, "Where the fine level goes" in `docs/fine.md`) take their `.roads`
from `bits/` after the pass: no extra download or reading, a few minutes of writing.

## Licence and credit

OSM data is under the ODbL. The road bits, the 1 km counts and the page's road bytes are each a
database derived from OSM, so like the 100 m `.roads` tiles (`docs/fine.md`, "Publishing")
they are published under the ODbL with "© OpenStreetMap contributors"; the info panel credits
OpenStreetMap (ODbL) for roads and paths. The road byte is also derived from LC100 (CC BY 4.0,
credited beside it). The painted map and the travel times are produced works, not bound by the
ODbL. What the 1 km layer adds: a second ODbL database on R2 (the page's road bytes, 68 MB),
offered under the same terms as the 100 m tiles.

## Open

- **Roads on hillsides** are walked at the hillside's terrain factor, as at 100 m: in Cluj's
  hills that is 0.68-0.85 of 5 km/h on road cells, and it is most of why MAP's L1 is so fast
  there (MAP keeps every road pixel at 5 km/h). Roads are graded and wind, so the truth is
  between the two. Proposed: keep it, so the levels agree, and revisit with the street grade
  cap of fine stage 2 (T-039).
- **The factor 3** is measured on eight tiles in two countries (farm and hill villages in
  Transylvania, the New York region). Proposed: keep it for the world build and check it on a
  few T-057 tiles elsewhere (a desert town, a rice delta, a forest track network) when they
  exist.
