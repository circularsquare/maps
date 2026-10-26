# Resolution

How the map looks sharp up close without solving the whole world at full detail.

## Levels

| Level | Cell | Grid | Where solved | Data |
|---|---|---|---|---|
| L10 | ~10 km (1/12 degree) | 4320 x 1740, whole world | every click | `data/grid.bin.gz` (`tools/build_grid.py`) |
| L3 | ~3 km (1/40 degree) | 14400 x 5800, in tiles | a window round the view, from zoom 5 | `data/tiles/l3/` |
| L1 | ~1 km (1/120 degree) | 43200 x 17400, in tiles | a window round the view, from zoom 7 | `data/tiles/l1/` |

L1 is the MAP surface itself. L3 and L10 are made from it the same way: a cell's speed is the mean
of its pixels' time per metre, and a cell is land when a quarter of its pixels are. All levels are
uint8 speed (km/h x 40, 0 = water).

**Tiles** (`tools/build_tiles.py`, ~3 min, reads the tif in 600-row strips): 5 degree tiles for
both levels, 600 x 600 cells at L1 and 200 x 200 at L3, 72 x 29 of them, named
`l<level>/<tile row>_<tile col>.bin.gz` (row 0 at 85N, col 0 at 180W). All-water tiles are not
written; `data/tiles/index.json` lists the ones that exist per level (as `row * 72 + col`).
L1 is 1141 tiles, 117.6 MB gzipped; L3 is 1130 tiles, 17.8 MB. `data/tiles/` is gitignored and
rebuilt from the tif.

## The view refine

All of this runs in `worker.js`; the page only shows what it sends.

1. A click solves L10 for the whole world and paints it. A newer click cancels an older solve.
2. The page sends the view (bounds and zoom) 200 ms after each camera move ends.
3. Below zoom 5 only L10 shows. From zoom 5 the worker solves an **L3 window**: the view plus a
   quarter of its size on each side. From zoom 7 it solves an L3 window of the view plus its
   full size on each side (three times as wide and tall), then an **L1 window** of the view plus
   a quarter on each side.
4. Window edges are rounded outwards to multiples of 0.25 degree, which is a whole number of
   cells at every level. Latitude is clipped to 85N-60S. A window is capped at 3 million cells
   for L3 and 2 million for L1 by shrinking the margin round the view centre.
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
8. **Painting.** The worker paints each level into an RGBA buffer, linear in Web Mercator y with
   one column per cell column (as the L10 paint always was), and sends it with its corners. The
   page puts each into its own MapLibre canvas source (`walk`, `walk3`, `walk1`, finest on top,
   all below the basemap's first label layer, `raster-fade-duration: 0`, linear resampling). The
   canvas is uploaded to the GPU once per change, not every frame.
9. **Holes.** Where a finer window is shown, the coarser canvas has that rectangle cleared, so
   coarse colours do not show through the window's water or unreached land. The hole is rounded
   inwards to whole canvas pixels.
10. **Hover** reads the finest shown level whose window holds the pointer.

### Measured solve times

Headless Chrome with software WebGL on Anita's machine (the map drawing competes with the
worker for CPU, so a normal browser should be a little faster). Solve = Dijkstra only; tiles
come from the worker's cache after the first fetch (fetch 1-90 ms).

| Case | Window | Solve | Paint |
|---|---|---|---|
| L10, whole world | 4320 x 1740 | 170-250 ms | 80-100 ms |
| L3, 1400 x 900 screen, zoom 6 | 930 x 410 | 166 ms | 16 ms |
| L1, 1400 x 900, zoom 7.5 | 990 x 480 | 239 ms | 24 ms |
| L1, 1400 x 900, zoom 9 | 390 x 180 | 43 ms | 4 ms |
| L3, 2560 x 1440 screen, zoom 5 | 2720 x 1050 (2.9M) | 700-800 ms | 50-75 ms |
| L1, 2560 x 1440, zoom 7 | 2580 x 990 (2.6M, before the 2M cap) | 1.2 s | 90-115 ms |
| L1, 2560 x 1440, zoom 7.2 | 2220 x 900 (2.0M) | 0.85-1.5 s | 70-100 ms |

The page's longest frame gap during an L1 window solve was ~20 ms.

**What the seed gets wrong.** A path that leaves the window and comes back in is only as good as
the coarse answer at the border. The coarse answer can also be a little early or late at the
border (half a coarse cell). Both show up as small seams at the window edge, not as wrong times
inside it. In checks near Cluj and in the Alps the bands carry on across window edges with only
small steps. Linear resampling also leaves a faint 1-3 px line along each window edge, where the
coarser canvas's hole edge is blended; the edges sit in the margin, off screen once the view has
settled.

**Small water.** MAP gives exactly 0.06 min/m (1 km/h) to seas and lakes alike, including ponds
of one 1 km pixel (about 2% of pixels round Cluj, from satellite land cover). A connected patch
of those pixels (8-connected, so a diagonal channel stays one body) under 5 km², its area summed
from pixel areas at their latitude, is land at that 1 km/h in every level: it counts as land
pixels when the L1 cells are made and when L3 and L10 cells are averaged. Larger patches stay
water. `tools/smallwater.py` finds the patches, labelling each 600-row strip and merging labels
that touch across strip edges and the 180 degree seam, so a patch is measured whole; both builds
use it. A sea or lake inlet that joins a bigger body is part of it and stays water.

## Thin barriers (Anita's ridge question)

Averaging time per metre over a cell keeps the time to cross the cell **straight through**: a
100 m ridge inside a 1 km cell adds the same minutes as in the fine grid. So a ridge is not made
10x easier to cross by averaging. Three things do go wrong:

- **Passes vanish.** A 1 km gap through the ridge gets averaged with the ridge itself, so the
  coarse cell is slow everywhere and the pass is lost. Coarse paths then go round, or pay the full
  ridge price.
- **Walking alongside is over-charged.** A path running parallel to the ridge, in the valley
  beside it, still pays the ridge's share of the cell.
- **The slope is already smoothed in the source.** MAP works at 1 km, so a 100 m wide, 1 km tall
  wall is already a gentler slope in its data. No averaging of ours can recover it; only a finer
  elevation model can (research T-004).

The view refine fixes the first two wherever you look closely. In the Alps from Brig, a point
15 km north (46.45N 7.9E) is 5.5 hours at L1 and 21.5 hours at L10: the coarse grid had lost the
way through. A later stage can fix this in the coarse solve too, by giving each coarse cell its
crossing times between edges ("portals") computed from the fine grid (as in HPA*). This also gives
Anita's idea of coarse cells in uniform land and fine ones in mountains: a uniform block needs few
portals, a rugged one many. Not planned until the view refine is in use and the canoe layer shows
how layers and portals fit together (T-007).
