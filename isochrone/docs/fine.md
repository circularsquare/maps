# Fine level (L0.1, 100 m)

A fourth level under L10, L3 and L1 (`docs/resolution.md`), so the map keeps sharpening past
zoom 9: streets, bridges, ponds, river banks and single hillsides. **Stage 1 is built (T-038)**
for New York and Cluj: land cover, OSM roads and bridges, the 1 km terrain factor, walking
starts. Stages 2-4 are design. Which tiles get 100 m worldwide is **open**, for Anita's review
("Where the fine level goes", T-047). Measurements: `notes/T-030.md`, `notes/T-038.md`,
`notes/T-047.md`.

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
`data/fine/index.json` lists the tiles and which have roads. Stage 1 tiles are 0.19-0.54 MB
(land 0.07-0.24, roads 0.12-0.31).

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

`python isochrone/tools/fine/build.py <region>` on this machine; a region is a box of whole
degrees and the Geofabrik extracts covering it (`REGIONS` in the script; extracts go in
`data/raw/osm/` and are deleted once the region is checked). It reads the LC100 window from
the local COG and the terrain factor from `data/surface/`, then every highway way of each
extract with pyosmium, keeping those that touch the box. One core: Cluj (Romania, 332 MB)
3.5 min, New York (four state extracts, 1.2 GB) 13 min, nearly all of it reading the extracts.

A whole-world build would be ~95 GB of planet file and 15-30 GB of tiles; "Where the fine level
goes" keeps it to a few GB.

## In the page

- **When**: on a walking start, an L0.1 window is solved once the view alone fits in
  **1.5 M cells**: from about **zoom 9.4 on a 1400 x 900 screen** and **10.2 on 2560 x 1440**
  (New York's latitude), where a 100 m cell is about 1 px. The window is the view plus a quarter
  on each side, the margin shrinking first to stay under the cap, with edges on multiples of
  0.05 degree. Every tile of the window must exist; if one is missing the window is cut back to
  the whole degrees round the view, and if a tile there is missing too the view stays at L1.
- **Under it** the L1 window takes a margin of the view's full size on each side and the L3
  window twice that, so the L0.1 ring is seeded from well inside L1.
- **In `js/windows.js`**: `fineWindowFor`, then `solveWindow` as for L1, seeded on its ring from
  L1 (then L3, L10) by the same bilinear sample; reuse and cancelling as for L1. Tiles come
  through `js/tiles.js` with their own cache of decoded speed tiles (~100 MB, 1.44 MB a tile).
- **Times**, headless Chrome, 1400 x 900, every cell land (Cluj): 0.1-0.2 s at zoom 12, 0.4 s at
  zoom 11, 1.25-1.5 s at the cap (zoom 9.5). The first fetch of a region is 0.2-1.2 s. A bucket
  queue in place of the heap is 1.3-1.6x faster (stage 2).
- **Paint**: a fourth canvas source on top (`walk01`), holes cut in L1, L3 and L10 under it.
  Cells show as squares from about zoom 12 (5 px; 10-13 px at 13, 20-26 at 14). The map's
  maximum zoom is 14.
- **Hover** and the route read L0.1 first where it is shown.

## Where the fine level goes

**For Anita's review (T-047).** Measured on 42 sample tiles (`notes/T-047.md`): in each, the
page's own search from five starts at 1 km and at 100 m, counting the share of the tile whose
colour band changes or which only one level reaches.

**What changes the answer at 100 m:**

| Where | Share of the tile that looks different | Why |
|---|---|---|
| deltas, creek cities, lake tundra, New York | 20-95% | water that cuts land apart changes what is reached at all |
| mountains | 22-58% | 90 m slopes; the 1 km slope smooths hillsides and valleys away |
| mixed farmland, savanna, towns with hills or rivers | 10-25% | the search picks the faster patches the 1 km mean blurs |
| one-piece cities (London, Trenton, Tokyo) | 5-14% | urban land is 5 km/h at both levels |
| sand seas, ice, taiga, unbroken forest, flat open tundra | 2-10% | uniform at both levels |

Most of what stands out in farmland today is MAP's road model, not resolution: MAP walks a
whole 1 km pixel at road speed when a road crosses it, so rural times at 100 m are 18-52%
longer than the page shows now. That is fixed at 1 km for the whole world by our own surface
with roads (T-010, T-028), much cheaper than 100 m everywhere.

**Proposed rule:**

| Tier | Test (from the 1 km layers) | Unit | Count | Size |
|---|---|---|---|---|
| towns | LC100 urban >= 2% of the block | 0.25 degree block | 21,644 blocks (1,350 tiles of area) | ~1.4 GB |
| mountains | >= 70% of land sloping over 5 degrees (GMTED) | 1 degree tile | 822 | ~1.2 GB |
| water country with people | >= 10% of pixels part water and urban >= 1% | 1 degree tile | ~160 | ~0.15 GB |
| everything else | stays at 1 km | | ~17,000 | |

About **2.8 GB** with stage 2's elevation (towns ~1 MB a tile of area: class, roads, flat
elevation; mountains ~1.5 MB: elevation is most of it). Wider choices: mountains from 50%
rugged (2,022 tiles, ~3 GB more), or towns from 5% urban (9,595 blocks, ~0.6 GB in place of
1.4). Quarter-degree blocks keep towns small: most of a 1 degree tile round a town is fields.
Bridges and straits fall in town blocks, passes in mountain tiles. Where no tile exists the
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

They do not help enough to re-grid:

- **Look**: a hex looks rounder than a square only when cells are big on screen. At 100 m a cell
  is 1-5 px at zoom 10-12, so the shape does not show; where it does (L1 at zoom 8-9) the fix is
  finer cells or smoother band edges.
- **Search**: the worst extra length on a straight line is 2.7% for our squares with 16 moves,
  15.5% for hexes with 6 neighbours, 3.5% with 12.
- **Cost**: every source is a latitude-longitude raster, so each layer would be resampled; hex
  levels do not nest (H3 children overhang their parent), which breaks seeding and windows; and
  hexes cannot go in a MapLibre canvas source.

**Keep squares.**

## Publishing

R2 holds 135 MB for the page now (T-016). The two test regions are 3 MB (stage 1); the proposed
world selection ~2.8 GB (R2's free tier is 10 GB of storage).
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
| 4 | later | The tiles of "Where the fine level goes", once Anita has decided it. | |

## Open

- **Which tiles get 100 m**: the rule above (~2.8 GB), a wider one, or more test regions first.
- **Zoom 8-9** stays L1 (cells 3-8 px). Smoothing band edges by colouring an interpolated time
  instead of whole cells would soften it at every level, with no new data.
