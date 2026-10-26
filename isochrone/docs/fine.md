# Fine level (L0.1, 100 m)

A fourth level under L10, L3 and L1 (`docs/resolution.md`), so the map keeps sharpening past
zoom 10: streets, bridges, ponds, river banks and single hillsides. **Design for Anita's review
(T-030); nothing of it is built.** Measurements behind every number: `notes/T-030.md`.

## Grid

- Cells of **1/1200 degree** (3 arc-seconds): 93 m north-south, 70 m east-west at New York.
  Exactly 10 x 10 cells in an L1 cell, 300 in the 0.25 degree window alignment, and the same grid
  as GLO-90 south of 50 degrees. LC100 (1/1008 degree) is resampled to it, nearest pixel.
- Squares, as the other levels (see "Hexes").

## What a tile holds

1 x 1 degree tiles, 1200 x 1200 cells, named by their south-west corner as GLO-90 is
(`data/fine/n40w074.land.bin.gz`), two files a tile, gzip as the other tiles:

| File | Band | Bytes a cell | From |
|---|---|---|---|
| `.land` | land cover class | 1 | LC100 2019, class at the cell centre |
| `.land` | elevation, metres | 2 (row deltas) | GLO-90, bilinear north of 50 degrees |
| `.roads` | road bits: path, local, main, motorway, bridge | 1 | OSM `highway=*` ways |

The worker turns these into speeds when it cuts a window, so a toggle (roads off, an era, a
different speed table) changes no data. Roads are a file of their own so "no roads" and
10,000 BC never fetch them, and so the OSM part stays apart for its licence (see "Publishing").
`data/fine/index.json` lists the tiles that exist.

## Speeds

Worked out per cell in the worker, into the same uint8 km/h x 40 grid the search reads now:

- **Land cover**: the LC100 speed table of `docs/surface.md`, unchanged.
- **Roads**: a cell with a walkable way (path, local or main road) is walked at least at
  5 km/h, MAP's road speed, any class. Motorways are not walkable roads: the cell keeps its land
  cover speed (not a wall either). Tunnels are left out. Ferries wait for a ferry mode.
- **Bridges**: a way tagged `bridge` is drawn into its cells even where LC100 says water, and
  those cells become walkable at road speed. Every other way is drawn only into land cells, so a
  riverside road or a pier never closes a channel. This is how a bridge becomes a crossing:
  LC100 shows every New York bridge span as water.
- **Water**: LC100 inland water and sea are water. No small-water rule at this level: ponds are
  walked round. Rivers narrower than a cell (the Somes in Cluj) are not water, as at L1.
- **Terrain**: altitude factor per cell, as now. Slope per move, from the two cells' elevations:
  directed Tobler normalised to the flat (research/friction.md 2), from a lookup table on the
  grade. Between two cells that both have a road or are urban, the grade counts at most 10%:
  streets are graded, and GLO-90 is a surface model whose Midtown towers would otherwise be a
  171 m mountain (40% of its 90 m steps read steeper than 10%).
- **10,000 BC**: no roads; land cover speed from the era's L1 layer (no finer source exists);
  terrain from GLO-90. Shelves that are sea today have no GLO-90 and stay at L1.

## Roads: OSM

| | OSM (Geofabrik extracts) | GRIP4 |
|---|---|---|
| Footpaths, tracks, steps | yes (80% of New York's ways are footways) | no |
| Bridges and tunnels | tagged (15,448 bridge ways in the New York tile) | not marked |
| Age | current | ~2018, older sources |
| Licence | ODbL: credit, and the road data we publish is ODbL too | CC0 or CC BY 4.0 |
| Getting it | a pbf per country or state, read with pyosmium at ~5 MB a second | one 8.4 GB download |

**OSM.** Without bridge tags a road over water cannot be told from a tunnel or a ferry, and at
100 m the bridges are what matter most in a city: almost every city cell already has a street
(97% of land cells in Brooklyn and lower Manhattan, 91% of Cluj's urban cells), and urban land
is already 5 km/h, so streets change little there. Outside cities they do: half of Cluj's
non-urban cells have a way, and fields off the tracks are 2.5 km/h.

Roads are cells, not edges: the grid search stays as it is and the toggle is free. A road runs
fastest along the 8 grid directions; at a knight's angle the move averages in its off-road
neighbours. Cars, later, may want roads as edges (one-ways, limited access), like the canoe's
river steps.

## Water and the canoe

The canoe (`docs/canoe.md`) works here as at L1, with three changes:

- **kind**: water cells are LC100 water; sea or lake as their L1 cell says.
- **Bridge cells are both**: a land node for walking and a canoe node (`canoeOf`), so the canoe
  passes under every bridge.
- **River steps** from RiverATLAS drawn at 1/1200 degree. Its lines come from 15" hydrography,
  so they sit up to a few hundred metres off the real channel; times are right, the line is a
  little off. Joining OSM waterways to RiverATLAS reaches would fix that; not planned.

## Making the tiles

`tools/fine/build.py <region>` on this machine, region by region, a region being a list of
1 degree tiles. For each tile: read the LC100 window from the local COG, fetch the GLO-90 tile
(2-6 MB, deleted once used), rasterise the roads from the Geofabrik extracts that cover the
region (downloaded to `data/raw/osm/`, deleted when the region is checked). One core, roughly
5-10 s a tile plus ~2 minutes a state-sized extract.

**Regions first, not the world.** A tile is 0.5-1.7 MB (class 0.16-0.20, roads up to 0.4 in a
city, elevation 0.27 flat to 1.4 in the Alps). The world's ~26,000 land tiles would be
**15-30 GB**: more than C: can spare (25 GB free is the limit), and the planet pbf alone is 95 GB.
A whole-world build would need disk elsewhere or an AWS run, which `trainworld/tools/aws/` does
not yet cover (it is set up for trainworld's builds only).

Why not compute it all in the browser from the sources: GLO-90's bucket sends no CORS headers,
so the page cannot read it; OSM would have to come from the basemap's vector tiles, which carry
paths only at zoom 14, hundreds of tiles a window. Our own tiles are 10-50 times smaller.

## In the page

- **When**: an L0.1 window is solved once the view alone fits in **1.5 M cells**: from about
  **zoom 10.4 on a 1400 x 900 screen** and **zoom 11.2 on 2560 x 1440** (New York's latitude),
  and only where every tile of the window exists. The margin (a quarter, as L1) shrinks first.
  The L1 window under it takes the wider margin L3 has under L1, to seed it well.
- **In `js/windows.js`**: a third window level, `windowFor(0.1, ...)` (its `f / 120` already
  gives 1/1200), a cap and a zoom rule for it; seeded on its ring from the L1 window by the same
  bilinear sample; reuse and cancelling as for L1. Tiles come through `js/tiles.js` with their
  own cache (~100 MB, a decoded tile is 5.8 MB).
- **Times** (node, one core, Cluj, every cell reached):

| Window | Cells | Plain | With slope |
|---|---|---|---|
| 1400 x 900, z12 | 0.36 M | 0.16 s | 0.23 s |
| 2560 x 1440, z12 | 1.07 M | 0.5-0.6 s | 1.0 s |
| 1400 x 900, z11 | 1.46 M | 0.9 s | 1.1 s |
| at the 1.5 M cap | 1.5 M | ~0.9 s | ~1.1 s |

  A bucket queue in place of the heap is 1.3-1.6x faster, agrees to 0.3%, and helps every level.
- **Fetch**: a window touches 2-6 tiles, 1-8 MB the first time over an area.
- **Memory**: a 1.5 M cell window is ~30 MB in the worker (grid, elevation, roads, the search's
  arrays, the paint).
- **Paint**: a fourth canvas source on top. At zoom 11-12 a cell is 1-3 px; from zoom 13 cells
  show again as squares (5-6 px at 13, 10-13 px at 14), the floor of this level.
- **Zoom 9-10 stays L1**: there a 100 m view is 2.6-7.6 M cells (2-5 s). L1 cells there are 3-8
  px. Ways to soften that are in "Open".

## Hexes

They do not help enough to re-grid:

- **Look**: a hex looks rounder than a square only when cells are big on screen. At 100 m a cell
  is 1-3 px at zoom 11-12, so the shape does not show; where it does (L1 at zoom 9-10) the fix is
  finer cells or smoother band edges.
- **Search**: the worst extra length on a straight line is 2.7% for our squares with 16 moves,
  15.5% for hexes with 6 neighbours, 3.5% with 12. Hexes are rounder than 8-move squares; ours
  already do better.
- **Cost**: every source is a latitude-longitude raster, so each layer would be resampled; hex
  levels do not nest (H3 children overhang their parent), which breaks seeding and windows; and
  hexes cannot go in a MapLibre canvas source, so painting becomes polygons or a custom shader.

**Keep squares.**

## Publishing

R2 holds 135 MB for the page now (T-016). The test regions add 10-20 MB; a country like Romania
~50 MB; Europe ~1.2-1.8 GB; the world 15-30 GB (R2's free tier is 10 GB of storage).
- OSM's licence: the published `.roads` tiles are a database derived from OSM, so they are
  published under the ODbL with "© OpenStreetMap contributors" (the info panel credits OSM
  already, for the basemap). The other files and the painted map are not bound by it.
- GLO-90 and LC100 credits go in the info panel when the level ships (a licence check of GLO-90's
  credit line is part of T-039).

## Stages

| Stage | Task | What | Anita sees |
|---|---|---|---|
| 1 | T-038 | Tiles for two test regions, New York (40-42 N, 75-73 W) and Cluj (46-48 N, 23-25 E): LC100 classes, OSM roads and bridges, the L1 terrain factor in place of elevation. The L0.1 window in the worker, seeded from L1, painted. | Manhattan reached over its bridges; streets, parks, ponds and fields at 100 m |
| 2 | T-039 | GLO-90 elevation in the land tiles; directed slope per move with the street grade cap; the bucket queue for every level; times in a real browser. | Hillsides and valleys round Cluj |
| 3 | later | The canoe at L0.1, after T-022: kind, bridge cells, river steps. | Paddling under bridges |
| 4 | later | More regions, or the world, once Anita says where it is built and kept. | |

## Open

- **Roads from OSM**, with the road tiles published as ODbL? Recommended. GRIP4 has no bridges.
- **Which regions** after the two tests, and whether the world is wanted (15-30 GB: disk, an
  AWS run, R2 over its free tier).
- **Zoom 9-10**: leave it at L1 (crisp cells from T-029), or smooth band edges by colouring an
  interpolated time instead of whole cells (no new data; every level), or solve a 100 m patch
  round the view's centre at zoom 10.
