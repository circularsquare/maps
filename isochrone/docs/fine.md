# Fine level (L0.1, 100 m)

A fourth level under L10, L3 and L1 (`docs/resolution.md`), so the map keeps sharpening past
zoom 9: streets, bridges, ponds, river banks and single hillsides. **Built**: land cover, OSM
roads and bridges, walking starts, for New York and Cluj (T-038) and for the towns, water
country and mountains of "Where the fine level goes" worldwide (T-057, read by block in the page
since T-086), with GLO-90 heights and directed slope per move, the rule L1 was made by (T-039).
Measurements: `notes/T-030.md`, `notes/T-038.md`, `notes/T-047.md`, `notes/T-057.md`,
`notes/T-086.md`, `notes/T-039.md`.

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
| `.land` | height, whole metres, int16 little-endian, as plane deltas | 2 x 1200 x 1200 | GLO-90 at the cell centre (bilinear between its posts), from T-011's `data/terrain/elev/`; ocean 0 |
| `.roads` | road bits: 1 path, 2 local, 4 main, 8 not walkable, 16 walkable bridge | 1200 x 1200 | OSM `highway=*` ways |

Plane deltas: each height minus its west and north neighbours plus its north-west one
(`z - w - n + nw`); along the first row minus the west one, down the first column minus the
north one, the corner as it is. They gzip 15% smaller than deltas along rows.

Road kinds: path = footway, path, steps, track, bridleway, pedestrian, cycleway; local =
residential, service, unclassified, living street, tertiary, road; main = secondary to trunk;
not walkable = motorways and ways tagged `foot=no` (or `access=no` without a foot tag allowing
it). Tunnels are left out, `building_passage` kept. A way marks every cell it passes through.

The worker turns these into speeds when it fetches a tile, so a toggle (roads off, an era, a
different speed table) changes no data. Roads are a file of their own so "no roads" and
10,000 BC never fetch them, and so the OSM part stays apart for its licence (see "Publishing").
**Town blocks.** A town tile holds only its selected 0.25 degree blocks (300 x 300 cells): every
byte of the other blocks is 0 in both files (class, height and road bits alike), so they
cost almost nothing gzipped and the files keep one layout. A tile's **block mask** is 16 bits, bit
`4 i + j` for block row i from the north and column j from the west (0-3); 0xFFFF is a whole
tile. Whole tiles are the test regions, water country and town tiles with all 16 blocks.

`data/fine/index.json`: `tile` 1200, `block` 300, `elev` (the layout: heights as plane
deltas), `slope` (the slope rule's numbers, from `tools/terrain/walk.py` `SLOPE_RULE`, so the
page walks exactly the rule L1 was made by); `tiles` the
whole tiles; `partial` {tile: mask} the town tiles with some blocks; `roads` the tiles with a
`.roads` file (a tile with no OSM way has none). Whole and partial tiles are listed apart so a
page that reads only `tiles` treats partial ones as missing rather than reading their zero
blocks as water. Whole tiles are 0.11-1.53 MB (median 0.84), a town block 13-134 KB (median
67), heights most of it; index.json is 126 KB.

## Speeds

Worked out per cell in the worker (`fineDecode` in `js/tiles.js`), into the same uint8
km/h x 40 grid the search reads at every level, beside the cells' heights and which cells are
graded (urban, or a walkable way or bridge):

- **Land cover**: the LC100 speed table of `docs/surface.md`, unchanged.
- **Roads**: a land cell with a walkable way (path, local or main) is walked at least at
  5 km/h, MAP's road speed, any kind. Not-walkable ways leave the cell at its land cover speed
  (not a wall either). Ferries wait for a ferry mode.
- **Bridges**: a cell LC100 calls water is walkable at 5 km/h when a walkable way tagged
  `bridge` crosses it; every other way counts on land cells only, so a riverside road or a pier
  never closes a channel. LC100 shows every New York bridge span as water.
- **Water**: LC100 inland water (80) and sea (200). No small-water rule at this level: ponds are
  walked round. Rivers narrower than a cell (the Somes in Cluj) are not water, as at L1.
- **Terrain**: per move, in the search (`js/search.js`), the rule of the 100 m walker that made
  L1 (notes/T-011.md 1), its numbers from index.json. A move's cost (length x the mean hours
  per km of the cells it touches) is divided by directed Tobler normalised to the flat,
  `exp(-3.5 (|g + 0.05| - 0.05))` on its grade g from its two end cells' heights (a knight move's
  too), read from a table in steps of 0.1% grade; and by MAP's height factor
  `1.016 exp(-0.0001072 h)` at the mean height of its end cells. Steeper than 45 degrees
  (|g| > 1) is not walked. Between two graded cells the grade counts at most 10%: streets are
  graded, and GLO-90 is a surface model whose Midtown towers would otherwise be a 171 m
  mountain. A move with an end in a block without 100 m data takes no slope (that cell's speed is
  its L1 cell's, terrain in it already). Hazard per km stays the cells' own.
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
   their block masks, water country tiles, mountain tiles).
2. `python isochrone/tools/fine/build.py world --jobs 2 --out fine_new` (~1 h on 2): every
   town tile by its blocks, and water country, mountain tiles (a town tile among them whole) and
   the test regions whole, into `data/fine_new/` to check
   (the page's `?fine=new`, `checks/fine.mjs --fine=new`) before swapping it in for
   `data/fine/`. Per tile: the LC100 window from the local COG, heights from
   `data/terrain/elev/` (T-011's run keeps them for every tile of the selection; the build
   stops at a tile without one), road bits from `data/roads/bits/` (the T-053 world pass,
   `docs/roads.md`), absent blocks zeroed. Every file is written to a temporary name and
   renamed, roads before land, and `index.json` is rewritten last, so the page never sees half a
   tile or a tile before its files. `--resume` skips tiles whose land file exists with the
   blocks the folder's `index.json` gives them, so a copy of `data/fine/` takes only new or
   changed tiles. Tiles past 80 N are left out: LC100 has no class there (26 mountain tiles,
   north Greenland and Ellesmere), and the view stays at L1.

`build.py tiles <name>...` makes whole tiles from `bits/`; `build.py region <name>` a test
region with its roads read from Geofabrik extracts in `data/raw/osm/` (`REGIONS`; Cluj 3.5 min,
New York 13 min, nearly all of it reading the extracts). The road bits of `bits/` match those
read from the extracts on 98.9-99.99% of the test tiles' cells (OSM edits since).

## In the page

- **When**: on a walking start, an L0.1 window is solved once the view alone fits in
  **1.5 M cells**: from about **zoom 9.4 on a 1400 x 900 screen** and **10.2 on 2560 x 1440**
  (New York's latitude), where a 100 m cell is about 1 px. The window is the view plus a quarter
  on each side, the margin shrinking first to stay under the cap, with edges on multiples of
  0.05 degree. At least **5% of the view** must lie in fine blocks, or the view stays at L1: a
  window costs the same whatever its share (0.1 s at zoom 12, 0.5-1.3 s at the cap), and a
  one-block town centred in the view is about 11% of it at zoom 10 (notes/T-086.md).
- **Blocks without 100 m data** (absent from a partial tile, or no tile at all) take the speed of
  the L1 cell they lie in, roads folded in as at L1, so a window may hold any mix of fine and
  coarse land. The filled part is walked at L1's speeds on the 100 m grid: its times came out
  within 3% of the L1 window's on the Sao Paulo check, with no step at the seam.
- **Under it** the L1 window takes a margin of the view's full size on each side and the L3
  window twice that, so the L0.1 ring is seeded from well inside L1.
- **In `js/windows.js`**: `fineWindowFor`, then `solveWindow` as for L1, seeded on its ring from
  L1 (then L3, L10) by the same bilinear sample; reuse and cancelling as for L1. Tiles come
  through `js/tiles.js` with their own cache of decoded tiles (~150 MB; 5.8 MB a tile: speeds,
  heights, graded cells).
- **Times**: 1.1-1.7 s at the cap in headless Chrome on a loaded machine (median 1.4 s; the first
  window of a session 1.2-1.7 s), 0.15 s at zoom 11.5, ~0.05 s at zoom 12 (notes/T-093.md,
  notes/T-039.md "Times"); the slope per move costs about a tenth to a third more than stage 1's
  search.
- **Paint**: a fourth canvas source on top (`walk01`), holes cut in L1, L3 and L10 under it.
  Cells show from about zoom 12 (5 px; 10-13 px at 13, 20-26 at 14). The map's maximum zoom is 14.
- **Hover** and the route read L0.1 first where it is shown.

## Where the fine level goes

What changes the answer at 100 m, from 42 sample tiles (the table and method are in
`notes/T-047.md`): water that cuts land apart (deltas, creek cities, lake tundra, New York),
mountains (90 m slopes; the 1 km slope smooths hillsides and valleys away), and to a lesser degree
mixed farmland and towns with hills or rivers. One-piece cities (urban land is 5 km/h at both
levels), sand seas, ice, taiga, unbroken forest and flat open tundra change little.

Farmland is left at 1 km because L1 already walks it as the 100 m level does: L1's speeds are
the 100 m walker's crossings, roads included (`docs/resolution.md`, "Directions"), so a 100 m
window and the L1 round it agree to under 1% 25 km out (Cluj 1.00-1.01, `checks/fine.mjs`). The
gap measured before that (rural times at 100 m 18-52% longer, notes/T-047.md) was MAP's road
model, which walked a whole 1 km pixel at road speed when a road crossed it.

**The rule:**

| Tier | Test (from the 1 km layers) | Unit | Count | Size |
|---|---|---|---|---|
| towns | LC100 urban >= 2% of the block's pixels | 0.25 degree block | 21,644 blocks in 4,671 tiles (1,353 tiles of area) | ~1.4 GB |
| mountains | >= 70% of land sloping over 5 degrees (GMTED), south of 80 N | 1 degree tile | 796 (822 picked, 26 past LC100) | 0.71 GB |
| water country with people | >= 10% of pixels part water (2-98%) and urban >= 1% of land | 1 degree tile | 158 | ~0.15 GB |
| everything else | stays at 1 km | | ~17,000 | |

Built (T-057, heights since T-039): **2.15 GB** in 5,350 tiles (2,227 tiles of area), 1.0 MB a
tile of area, heights most of it; mountain tiles 0.71 GB (median 1.0 MB). Quarter-degree blocks
keep towns small: most of a 1 degree tile round a town is fields; whole tiles round every town
would have been twice the size. Bridges and straits fall in town blocks, passes in mountain tiles. Where no tile exists the
view stays at L1, as now.

Not picked, and measured as changing least: deserts without dunes, ice, taiga and rainforest,
flat farmland, open tundra. Dune seas gain 20% from GLO-90's slopes (Sahara) and stay out all
the same: nobody crosses them by choice.

**The coarse levels.** The same idea, coarse cells where land is uniform and finer ones where it
is not, is in T-007's design (`docs/resolution.md`, "Thin barriers"): a coarse cell that water
cuts into pieces of land becomes one node a piece. The measurement above says where finer cells
pay: the uniform tiles (gain under 0.1) are where a coarse cell already gives the fine answer.

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

The page's data goes to R2 when it is published (T-016): the 100 m selection is 2.15 GB in 5,350
tiles beside ~750 MB of walking levels (R2's free tier is 10 GB of storage).
- OSM's licence: the published `.roads` tiles are a database derived from OSM, so they are
  published under the ODbL with "© OpenStreetMap contributors" (the info panel credits OSM
  already, for the basemap). The other files and the painted map are not bound by it.
- LC100 and GLO-90 are credited in the info panel. GLO-90's licence (research/friction.md,
  "GLO-90 licence") asks for its "produced using" notice and a no-liability sentence, both in the
  panel, and binds whoever we let redistribute the heights in the `.land` tiles to the same.

## Stages

| Stage | Task | What | Anita sees |
|---|---|---|---|
| 1 | T-038, built | Tiles for New York (40-42 N, 75-73 W) and Cluj (46-48 N, 23-25 E): LC100 classes, OSM roads and bridges, the L1 terrain factor (until stage 2). The L0.1 window in the worker, seeded from L1, painted, hovered. | Manhattan reached over its bridges; streets, parks, ponds and fields at 100 m |
| 2 | T-039 and T-093, built | GLO-90 elevation in the land tiles; directed slope per move with the street grade cap; the bucket queue for every level. | Hillsides and valleys round Cluj |
| 3 | T-050 | The canoe at L0.1: kind, bridge cells, river steps. | Paddling under bridges |
| 4 | T-057 and T-086, built | The tiles of "Where the fine level goes": towns, water country and mountains (roads from T-053's world pass, heights from T-011's run) and the page reading them by block (T-086). | 100 m round every town and in the mountains |

## Open

- **Zoom 8-9** stays L1 (cells 3-8 px). Smoothing band edges by colouring an interpolated time
  instead of whole cells would soften it at every level, with no new data.
