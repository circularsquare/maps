# Fine level (L0.1, 100 m)

A fourth level under L10, L3 and L1 (`docs/resolution.md`), so the map keeps sharpening past
zoom 9: streets, bridges, ponds, river banks and single hillsides. **Stage 1 is built**: land
cover, OSM roads and bridges, the 1 km terrain factor, walking starts, for New York and Cluj
(T-038) and for the towns and water country of "Where the fine level goes" worldwide (T-057;
the page reads their town blocks once the change in `notes/T-057.md` is made). Mountain tiles
wait for stage 2's elevation. Measurements: `notes/T-030.md`, `notes/T-038.md`, `notes/T-047.md`,
`notes/T-057.md`.

## Grid

- Cells of **1/1200 degree** (3 arc-seconds): 93 m north-south, 70 m east-west at New York.
  Exactly 10 x 10 cells in an L1 cell, and the same grid as GLO-90 south of 50 degrees. LC100
  (1/1008 degree) is resampled to it, nearest pixel.
- Squares, as the other levels (see "Hexes").

## What a tile holds

1 x 1 degree tiles, 1200 x 1200 cells, named by their south-west corner as GLO-90 is
(`data/fine/n40w074.land.bin.gz`), two gzipped files a tile:

| File | Part | Bytes | From |
|---|---|---|---|
| `.land` | land cover class | 1200 x 1200 | LC100 2019, class at the cell centre |
| `.land` | terrain factor x 250 (stage 1) | 120 x 120 | `data/surface/terrain_factor.tif`, the L1 cell's |
| `.land` | elevation, metres (stage 2, in place of the terrain factor) | 2 a cell, row deltas | GLO-90, bilinear north of 50 degrees |
| `.roads` | road bits: 1 path, 2 local, 4 main, 8 not walkable, 16 walkable bridge | 1200 x 1200 | OSM `highway=*` ways |

Road kinds: path = footway, path, steps, track, bridleway, pedestrian, cycleway; local =
residential, service, unclassified, living street, tertiary, road; main = secondary to trunk;
not walkable = motorways and ways tagged `foot=no` (or `access=no` without a foot tag allowing
it). Tunnels are left out, `building_passage` kept. A way marks every cell it passes through.

The worker turns these into speeds when it fetches a tile, so a toggle (roads off, an era, a
different speed table) changes no data. Roads are a file of their own so "no roads" and
10,000 BC never fetch them, and so the OSM part stays apart for its licence (see "Publishing").
**Town blocks.** A town tile holds only its selected 0.25 degree blocks (300 x 300 cells): every
byte of the other blocks is 0 in both files (class, terrain factor and road bits alike), so they
cost almost nothing gzipped and the files keep one layout. A tile's **block mask** is 16 bits, bit
`4 i + j` for block row i from the north and column j from the west (0-3); 0xFFFF is a whole
tile. Whole tiles are the test regions, water country and town tiles with all 16 blocks.

`data/fine/index.json`: `tile` 1200, `terrain` 120, `tfScale` 250, `block` 300; `tiles` the
whole tiles; `partial` {tile: mask} the town tiles with some blocks; `roads` the tiles with a
`.roads` file (a tile with no OSM way has none). A page that reads only `tiles` treats partial
tiles as missing, which keeps their zero blocks from reading as water. Whole tiles are
0.05-0.6 MB (median 0.3), a town block 7-51 KB (median 23); index.json is 114 KB.

## Speeds

Worked out per cell in the worker (`fineSpeed` in `js/tiles.js`), into the same uint8
km/h x 40 grid the search reads at every level:

- **Land cover**: the LC100 speed table of `docs/surface.md`, unchanged.
- **Roads**: a land cell with a walkable way (path, local or main) is walked at least at
  5 km/h, MAP's road speed, any kind. Not-walkable ways leave the cell at its land cover speed
  (not a wall either). Ferries wait for a ferry mode.
- **Bridges**: a cell LC100 calls water is walkable at 5 km/h when a walkable way tagged
  `bridge` crosses it; every other way counts on land cells only, so a riverside road or a pier
  never closes a channel. LC100 shows every New York bridge span as water.
- **Water**: LC100 inland water (80) and sea (200). No small-water rule at this level: ponds are
  walked round. Rivers narrower than a cell (the Somes in Cluj) are not water, as at L1.
- **Terrain**, stage 1: the speed times the terrain factor of the L1 cell (MAP's elevation and
  slope factors, `docs/surface.md`). Stage 2: altitude factor per cell, and slope per move from
  the two cells' elevations, directed Tobler normalised to the flat (research/friction.md 2),
  from a lookup table on the grade. Between two cells that both have a road or are urban, the
  grade counts at most 10%: streets are graded, and GLO-90 is a surface model whose Midtown
  towers would otherwise be a 171 m mountain.
- **10,000 BC**: no roads; land cover speed from the era's L1 layer (no finer source exists);
  terrain from GLO-90. Shelves that are sea today have no GLO-90 and stay at L1.

## Roads: OSM

OSM, from Geofabrik extracts (Anita, 2026-10-26: OSM is fine under the ODbL). GRIP4 has no
footpaths and no bridge or tunnel tags; without bridge tags a road over water cannot be told
from a tunnel or a ferry, and at 100 m the bridges are what matter most in a city: almost every
city cell already has a street, and urban land is 5 km/h anyway. Outside towns roads matter:
half of Cluj's non-urban cells have a way, and fields off the tracks are 2.5 km/h.

Roads are cells, not edges: the grid search stays as it is and the toggle is free. A road runs
fastest along the 8 grid directions; at a knight's angle the move averages in its off-road
neighbours. Cars, later, may want roads as edges (one-ways, limited access), like the canoe's
river steps.

## Water and the canoe

Stage 1 is walking only: a canoe start stops at L1. The canoe at L0.1 (stage 3, T-050) works
as at L1 (`docs/canoe.md`) with three changes:

- **kind**: water cells are LC100 water; sea or lake as their L1 cell says.
- **Bridge cells are both**: a land node for walking and a canoe node (`canoeOf`), so the canoe
  passes under every bridge.
- **River steps** from RiverATLAS drawn at 1/1200 degree. Its lines come from 15" hydrography,
  so they sit up to a few hundred metres off the real channel; times are right, the line is a
  little off.

## Making the tiles

1. `python isochrone/tools/fine/select.py` (~1 min): the rule of "Where the fine level goes"
   on the 1 km layers of `data/surface/`, into `data/fine/selection.json` (town tiles with
   their block masks, water country tiles, mountain tiles for later).
2. `python isochrone/tools/fine/build.py world --jobs 3` (~15 min): every town and water
   country tile, and the test regions whole. Per tile: the LC100 window from the local COG,
   the terrain factor from `data/surface/terrain_factor.tif` (the build stops if that file
   changes under it), road bits from `data/roads/bits/` (the T-053 world pass, `docs/roads.md`),
   absent blocks zeroed. Every file is written to a temporary name and renamed, roads before
   land, and `index.json` is rewritten last, so the page never sees half a tile or a tile
   before its files. `--resume` skips tiles whose land file exists.

`build.py tiles <name>...` makes whole tiles from `bits/`; `build.py region <name>` a test
region with its roads read from Geofabrik extracts in `data/raw/osm/` (`REGIONS`; Cluj 3.5 min,
New York 13 min, nearly all of it reading the extracts). The road bits of `bits/` match those
read from the extracts on 98.9-99.99% of the test tiles' cells (OSM edits since).

## In the page

- **When**: on a walking start, an L0.1 window is solved once the view alone fits in
  **1.5 M cells**: from about **zoom 9.4 on a 1400 x 900 screen** and **10.2 on 2560 x 1440**
  (New York's latitude), where a 100 m cell is about 1 px. The window is the view plus a quarter
  on each side, the margin shrinking first to stay under the cap, with edges on multiples of
  0.05 degree. Every tile of the window must be a whole tile of `index.json`; if one is not the
  window is cut back to the whole degrees round the view, and if a tile there is not either the
  view stays at L1. Partial town tiles are not read yet: the page change that reads them by block
  and fills absent blocks from L1 speeds is written out in `notes/T-057.md`.
- **Under it** the L1 window takes a margin of the view's full size on each side and the L3
  window twice that, so the L0.1 ring is seeded from well inside L1.
- **In `js/windows.js`**: `fineWindowFor`, then `solveWindow` as for L1, seeded on its ring from
  L1 (then L3, L10) by the same bilinear sample; reuse and cancelling as for L1. Tiles come
  through `js/tiles.js` with their own cache of decoded speed tiles (~100 MB, 1.44 MB a tile).
- **Times**: about 1.5 s at the cap, much less when zoomed in (notes/T-038.md). A bucket queue in
  place of the heap is 1.3-1.6x faster (stage 2).
- **Paint**: a fourth canvas source on top (`walk01`), holes cut in L1, L3 and L10 under it.
  Cells show from about zoom 12 (5 px; 10-13 px at 13, 20-26 at 14). The map's maximum zoom is 14.
- **Hover** and the route read L0.1 first where it is shown.

## Where the fine level goes

What changes the answer at 100 m, from 42 sample tiles (the table and method are in
`notes/T-047.md`): water that cuts land apart (deltas, creek cities, lake tundra, New York),
mountains (90 m slopes; the 1 km slope smooths hillsides and valleys away), and to a lesser degree
mixed farmland and towns with hills or rivers. One-piece cities (urban land is 5 km/h at both
levels), sand seas, ice, taiga, unbroken forest and flat open tundra change little.

Most of what stands out in farmland today is MAP's road model, not resolution: MAP walks a
whole 1 km pixel at road speed when a road crosses it, so rural times at 100 m are 18-52%
longer than the page shows now. That is fixed at 1 km for the whole world by our own surface
with roads (T-010, T-028), much cheaper than 100 m everywhere.

**The rule:**

| Tier | Test (from the 1 km layers) | Unit | Count | Size |
|---|---|---|---|---|
| towns | LC100 urban >= 2% of the block's pixels | 0.25 degree block | 21,644 blocks in 4,671 tiles (1,353 tiles of area) | ~1.4 GB |
| mountains | >= 70% of land sloping over 5 degrees (GMTED) | 1 degree tile | 822 | ~1.2 GB |
| water country with people | >= 10% of pixels part water (2-98%) and urban >= 1% of land | 1 degree tile | 158 | ~0.15 GB |
| everything else | stays at 1 km | | ~17,000 | |

About **2.8 GB** with stage 2's elevation (towns ~1 MB a tile of area: class, roads, flat
elevation; mountains ~1.5 MB: elevation is most of it). Quarter-degree blocks keep towns small: most of a 1 degree tile round a town is fields.
Built at stage 1 (T-057): **0.52 GB**, towns 0.48 GB (4,507 tiles), water country 0.04 GB
(157 tiles), test regions 3 MB; 0.36 MB a tile of area. Whole tiles round every town would
have been twice that. Bridges and straits fall in town blocks, passes in mountain tiles. Where no tile exists the
view stays at L1, as now.

Not picked, and measured as changing least: deserts without dunes, ice, taiga and rainforest,
flat farmland, open tundra. Dune seas gain 20% from GLO-90's slopes (Sahara) and stay out all
the same: nobody crosses them by choice.

**The coarse levels.** The same idea, coarse cells where land is uniform and fine ones where it
is not, is the portals plan (T-007, `docs/resolution.md`): each coarse cell keeps crossing times
between its edges worked out from the fine grid, few for a uniform cell, many for a rugged one.
The measurement above says where that pays: the uniform tiles (gain under 0.1) are where a
coarse cell already gives the fine answer.

## Hexes

The grid and the search stay square at every level. The hex is a paint look only (the default on
the page, `cfg.look`, `js/paint.js`): each cell is drawn as a hexagon on the canvas, and the pixel
to cell map is kept per grid geometry. A real hex grid would not pay:

- **Search**: the worst extra length on a straight line is 2.7% for our squares with 16 moves,
  15.5% for hexes with 6 neighbours, 3.5% with 12.
- **Cost**: every source is a latitude-longitude raster, so each layer would be resampled; hex
  levels do not nest (H3 children overhang their parent), which breaks seeding and windows.
- At 100 m a cell is 1-5 px at zoom 10-12, so the look shows only from about zoom 12.

## Publishing

R2 holds 135 MB for the page now (T-016). Stage 1's towns, water country and test regions are
0.52 GB in 4,672 tiles; the whole selection with elevation ~2.8 GB (R2's free tier is 10 GB of
storage).
- OSM's licence: the published `.roads` tiles are a database derived from OSM, so they are
  published under the ODbL with "© OpenStreetMap contributors" (the info panel credits OSM
  already, for the basemap). The other files and the painted map are not bound by it.
- LC100 and GLO-90 credits go in the info panel when the level ships (a licence check of
  GLO-90's credit line is part of T-039).

## Stages

| Stage | Task | What | Anita sees |
|---|---|---|---|
| 1 | T-038, built | Tiles for New York (40-42 N, 75-73 W) and Cluj (46-48 N, 23-25 E): LC100 classes, OSM roads and bridges, the L1 terrain factor. The L0.1 window in the worker, seeded from L1, painted, hovered. | Manhattan reached over its bridges; streets, parks, ponds and fields at 100 m |
| 2 | T-039 | GLO-90 elevation in the land tiles; directed slope per move with the street grade cap; the bucket queue for every level. | Hillsides and valleys round Cluj |
| 3 | T-050 | The canoe at L0.1: kind, bridge cells, river steps. | Paddling under bridges |
| 4 | T-057, towns and water country built | The tiles of "Where the fine level goes": towns and water country (stage 1 content, roads from T-053's world pass), mountain tiles once stage 2 gives them elevation; the page reading town blocks (`notes/T-057.md`). | 100 m round every town |

## Open

- **Zoom 8-9** stays L1 (cells 3-8 px). Smoothing band edges by colouring an interpolated time
  instead of whole cells would soften it at every level, with no new data.
