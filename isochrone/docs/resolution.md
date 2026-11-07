# Resolution

How the map looks sharp up close without solving the whole world at full detail.

## Levels

| Level | Cell | Grid | Where solved | Data |
|---|---|---|---|---|
| L10 | ~10 km (1/12 degree) | 4320 x 1740, whole world | every click | `data/grid.bin.gz`, roads `data/grid_roads.bin.gz`, directions `grid_dir.bin.gz`, `grid_dir_roads.bin.gz` |
| L3 | ~3 km (1/40 degree) | 14400 x 5800, in tiles | a window round the view, from zoom 5 | `data/tiles/l3/`, roads `r3/`, directions `d3/`, `e3/` |
| L1 | ~1 km (1/120 degree) | 43200 x 17400, in tiles | a window round the view, from zoom 7 | `data/tiles/l1/`, roads `r1/`, directions `d1/`, `e1/` |
| L0.1 | ~100 m (1/1200 degree) | 1 degree tiles, where built | a window round the view once the view fits in 1.5 M cells (zoom ~9.4 on 1400 x 900), walking starts | `data/fine/` (`docs/fine.md`) |

L0.1's sources, tiles and when it is solved are in `docs/fine.md`; it is seeded (from L1, then
L3, L10), painted (canvas source `walk01`), holed and hovered as the windows below.

L1 is our own surface (`docs/surface.md`: LC100 land cover, GMTED2010 height and slope) under the
water rule below, its land cells' speeds replaced by their directed crossings wherever the 100 m
walker crosses them ("Directions" below). L3 and L10 are made by crossing the L1 cells (T-011,
the same section); a coarse cell is land when a quarter of its L1 cells are. Every level stores
the off-road speed (uint8 km/h x 40, 0 = water), beside it a road byte per cell (`docs/roads.md`,
"The page part"), which the worker folds into the speed as it loads a level (`applyRoads` in
`js/tiles.js`; `cfg.roads` off skips the road files), and the direction planes.

**Tiles** (`tools/build_tiles.py`, over 600-row strips): 5 degree tiles for both windowed levels, 600 x 600 cells at L1 and 200 x 200 at L3, 72 x 29
of them, named `l<level>/<tile row>_<tile col>.bin.gz` (row 0 at 85N, col 0 at 180W), road bytes
the same in `r<level>/`. A speed tile is written where it has land, a road tile where it has a
road byte; `data/tiles/index.json` lists both per level (as `row * 72 + col`, the road tiles under
`roads`, the direction tiles under `dirs` and `dirsRoads`). Gzipped, L1 is 1137 speed tiles,
151 MB, 1040 road tiles, 66 MB, and direction tiles of 314 MB (`d1`, 992) and 144 MB (`e1`, 938);
L3 is 1121 tiles, 17 MB, 1026 road tiles, 10 MB, and direction tiles of 28 and 18 MB; L10 is
1.8 MB, 1.2 MB and 2.3 + 1.9 MB: ~750 MB in all. The build takes ~1 h on one core (19 min on
three), in two passes (L1, then L3 and L10 crossing it), into `data/tiles_next/` or `--out <name>`; `build_tiles.py swap <backup>
[--from <name>]` serves it by renames and moves what was served to `data/<backup>/`.
`data/` is gitignored.

**Water at L1.** A cell is water when LC100's inland water and sea cover half of it or more (as
`walk_today.tif`). LC100 has some islands and archipelagos as sea outright (Samoa, Tahiti,
Kerguelen, South Georgia, Chongming, the Sundarbans, much of the Chilean archipelago): a cell
LC100 has as 98% sea or more where GMTED2010 is above 2 m is land, at 3 km/h (bare or sparse, a
middle guess) times its terrain factor. A water cell a walkable bridge crosses is land when roads
are on (`docs/roads.md`).

## Directions

Walking is directed (T-011; measurements: `notes/T-011.md`). Each L1 cell has four crossing
speeds, east, west, south and north, without roads and with them, from the 100 m walker on GLO-90
heights (`data/terrain/l1/`, `tools/terrain/walk.py` `tile_l1`).

- **The 100 m walker** walks the L0.1 grid by the page's 16 moves and the slope rule of
  `docs/fine.md` ("Terrain"): LC100 class speeds, with roads a walkable way at least 5 km/h and a
  walkable bridge over water at 5 km/h, each move divided by directed Tobler on its grade and by
  MAP's height factor. A cell's crossing in a direction is the least time from the 100 m line just
  outside the cell to its far edge, inside the cell plus a margin of one 100 m cell, over the
  cell's width; it is then scaled by the cell's 3-cell strip along that direction (the strip's
  crossing over the sum of its three cells' own). With roads the same is solved on the grid with
  roads, with a strip margin of three 100 m cells, on cells with a walkable way or bridge in
  their 3 x 3; elsewhere, and where it would be slower, a direction keeps its value without roads.
- **L1 speed byte**: the geometric mean of the four directions without roads. A direction the
  walker does not cross takes that mean; a cell it crosses in no direction (2.5% of L1 land: past
  80 N where LC100 has no class, small water, LC100's missing islands, cells walled by water at
  100 m) keeps the surface's speed, is the same every way, and takes the road rule of
  `docs/roads.md`.
- **L1 road byte**: time with roads over time without, x 250 (`docs/roads.md`), from the
  geometric means of the four directions with and without roads. A direction with roads is never
  slower than without.
- **Direction planes**, four int8 planes E W S N a level: `d` = 32 ln(speed that way / the speed
  byte), and `e` = 32 ln(speed that way with roads / the speed with the road byte folded in), less
  `d`. Values within ±1 (3%) are written 0. With roads on a direction's speed is the folded speed x
  exp((d + e) / 32), with roads off the speed x exp(d / 32). `d1/`, `e1/`, `d3/`, `e3/` hold the four
  planes one after another in a tile's file; L10's are `grid_dir.bin.gz` and `grid_dir_roads.bin.gz`
  (4 x 1740 x 4320). A tile with every value 0 is not written. The worker fetches `e` only with
  roads on.
- **L3 and L10** cross the written L1 the way the page walks it: for each coarse cell and
  direction, the least time from the L1 cells just outside the cell (its own rows or columns) to
  its far edge, inside the cell plus a margin of one L1 cell (T-080's best crossing), scaled by
  the strip of the cell and its two neighbours along the direction (the strip's crossing over the
  sum of the three cells' own crossings; the cell's own where a neighbour is not crossed). The
  same with roads, on cells within two coarse cells of a road byte. Then as at L1: the speed byte
  is the geometric mean, a direction not crossed takes it, and a land cell crossed in no direction
  keeps the mean rule (the mean of its L1 cells' time per metre, `coarse()`) the same every way.
- **The search** (`js/search.js`): a level's `L.sd` holds each land cell's four hours per km
  (`setDirs` in `js/levels.js`), or is null when the level has no planes, which leaves the
  symmetric search exactly as it was. Move k, from row r, uses each touched cell's
  `wx s_x + (1 - wx) s_y`, with `s_x` its east or west value by the sign of the move's column
  step, `s_y` south or north by its row step, and `wx = |dx| / (|dx| + |dy|)` the move's share of
  east-west km; the move takes its length times the mean over the cells it touches (two, or four
  for a knight move). Carrying is twice that; hazard per km and getting in or out use the speed
  byte. L0.1 has no planes: its blocks without 100 m data take the L1 speed byte.

A build goes to a folder of its own (`build_tiles.py --out <name>`, `data/<name>/`); the page reads
`data/tiles_<name>/` with `?tiles=<name>` and the served `data/` otherwise.

## The view refine

All of this runs in the worker (`worker.js`, `js/`); the page only shows what it sends.

1. A click solves L10 for the whole world and paints it. A newer click cancels an older solve.
2. The page sends the view (bounds and zoom) 200 ms after each camera move ends. Each side of
   the bounds is kept within 1.5 times what an untilted view at that zoom shows round the
   centre (`viewBox`), since on the globe a tilted view reaches the horizon and a visible pole
   makes the bounds every longitude. River tiles are picked from the same box.
3. Below zoom 5 only L10 shows. From zoom 5 the worker solves an **L3 window**: the view plus a
   quarter of its size on each side. From zoom 7 it solves an L3 window of the view plus its
   full size on each side (three times as wide and tall), then an **L1 window** of the view plus
   a quarter on each side. Under an L0.1 window the L1 margin is the view's full size and the L3
   margin twice that (`docs/fine.md`, "In the page").
4. Window edges are rounded outwards to multiples of 0.25 degree, which is a whole number of
   cells at every level. Latitude is clipped to 85N-60S. A window is capped at 3 million cells
   for L3 (1.5 million on a canoe start) and 2 million for L1 by shrinking the margin round the
   view centre.
5. **Seeding.** Every land cell on the window's outer ring starts at the time the next coarser
   solved level gives there: L3 from L10, L1 from the L3 window (L10 where the L3 window does not
   reach). The value is a bilinear blend of the four coarser cell centres round the point, using
   only the reached ones. The start (snapped to the nearest land cell within ~10 km) is seeded at
   0 when it is inside the window. The search then runs inside the window only.
6. **Reuse.** If the last window of a level (same start) already holds the view (and, for L3
   under L1, the new L1 window plus the view's quarter margin), it is kept and not solved again,
   so zooming in inside a solved area costs nothing.
7. **Cancelling.** The search yields every ~30 ms; a newer click or view stops an older window
   solve at the next yield. A camera move does not stop a click's L10 solve; the windows for the
   newest view follow it.
8. **Painting.** The worker paints each level into an RGBA buffer, laid out linear in Web Mercator
   y with one column per cell column (as the L10 paint always was), each canvas row taking the
   value of the cell row it falls in, and sends it with its corners. The
   page puts each into its own MapLibre canvas source (`walk`, `walk3`, `walk1`, finest on top,
   all below the basemap's first label layer, `raster-fade-duration: 0`, nearest resampling).
   Cells are painted as hexagons by default (`cfg.look`, T-060; `?look=crisp` square blocks,
   `?look=diag` cut corners); the look is paint only, the grid and search stay square. The
   canvas is uploaded to the GPU once per change, not every frame. MapLibre 5 draws a canvas
   source on the projection's tile mesh, linear in Mercator within the tile, so on the globe the
   same canvases bend onto the sphere unchanged.
9. **Holes.** Where a finer window is shown, the coarser canvas has that rectangle cleared, so
   coarse colours do not show through the window's water or unreached land. The hole is rounded
   inwards to whole canvas pixels.
10. **Hover** reads the finest shown level whose window holds the pointer.

Solve times by level and screen size: `notes/T-002.md`. The window caps (3 M cells at L3, 2 M at
L1) keep an L1 window to about a second on a 2560 x 1440 screen.

**What the seed gets wrong.** A path that leaves the window and comes back in is only as good as
the coarse answer at the border. The coarse answer can also be a little early or late at the
border (half a coarse cell). Both show up as small seams at the window edge, not as wrong times
inside it. In checks near Cluj and in the Alps the bands carry on across window edges with only
small steps. Along a window's north and south edges a strip up to one coarser cell tall (about 5
px at zoom 9) shows both levels at once and so reads a little brighter, because the hole is
rounded inwards to whole canvas rows; the edges sit in the margin, off screen once the view has
settled.

**Small water.** A connected patch of L1 water cells (8-connected, so a diagonal channel stays one
body) under 5 km², its area summed from cell areas at their latitude, is land at 1 km/h (MAP's
rate for wading or swimming) in every level: it counts as land cells when L1 is made and when L3
and L10 cells are averaged. Larger patches stay water. `tools/smallwater.py` finds the patches,
labelling each 600-row strip and merging labels that touch across strip edges and the 180 degree
seam, so a patch is measured whole. A sea or lake inlet that joins a bigger body is part of it and
stays water.

## Thin barriers (Anita's ridge question)

A coarse cell's speeds are its best crossings of the finer cells in each direction ("Directions"),
and an L1 cell's are the 100 m walker's on 90 m heights. So
a ridge inside a cell costs what the finer walk pays to get over or round it, a gap through it is
found (the pass is kept), and a path along a valley pays the valley's speed in that direction, not
the ridge's. Three things still go wrong:

- **One-way passes.** A cell's four values say how fast it is crossed east, west, south and north,
  not between which parts of its edges. A ridge with one gap, or a river crossed at one bridge,
  makes the cell fast that way for every path through it.
- **Water inside a coarse cell.** A coarse cell is one node, so where a fjord, a wide river, a
  strait or a lake cuts its land in two, it joins both banks: a coarse walk steps in on one and out
  on the other where L1 goes round (1.9% of L10 land cells, 0.17% at L3). Fjord and river country
  is too fast at L3 and L10 (Sogndal L1/L10 1.32, Coari 1.39).
- **The 1 km source** where the walker crosses a cell in no direction (2.5% of L1 land): L1 is
  then the GMTED2010 surface, whose 1 km slope already smooths a 100 m wide wall (research T-004).

The view refine fixes these wherever you look closely. T-007's design splits each coarse cell
into its L1 land pieces, every piece a node with its own crossings and links, as the canoe's water
units are (`docs/canoe.md` 4); stage 2, the build, waits on Anita (`notes/T-007.md`). Crossing
times between several points on each edge ("portals", as in HPA*) would also hold the one-way
passes; not planned (notes/T-080.md weighs it).
