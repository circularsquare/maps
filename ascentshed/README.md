# Ascentshed

Inverse watershed. Instead of water descending to basins, "flow" ascends to
peaks: every cell is labelled by the peak its steepest-ascent path ends at (its
prominence parent). Basins are then carved into **N territories of comparable
integrated heft** — the relief analog of an equal-population choropleth.

## Pipeline

| step | function | notes |
|------|----------|-------|
| 1. terrain | `terrain.synthetic` / `terrain.load_dem` | full topo+bathy; bathymetry **not** masked; returns cos-lat area weight |
| 2. ascent basins | `ascentshed.ascent_basins` | D8 steepest-ascent, vectorized successor + pointer-doubling roots |
| 3. divide tree | `ascentshed.build_tree` | union-find over **basin-graph** edges (RAG) in descending saddle order → key col, prominence, parent, adjacency, area-weighted stats. ~70× faster than the per-cell `merge_tree` and validated identical |
| 4. partition | `ascentshed.balanced_regions` | balanced multi-source growth on the basin adjacency graph |
| measure | `ascentshed.territory_volumes` | exact area-weighted heft per territory from the grid |

`merge_tree` (the original per-cell priority-flood) is kept only as a reference
oracle; `build_tree` is the one to use. On the full-res Alps tile (3.1 M cells)
the whole pipeline runs in ~3 s.

## Two metrics, kept distinct

- **prominence-volume** `Σ_basin (h − key_col)` — relative to each peak's own key
  col. Ranks which peaks matter (favors relief mass over thin spires / flat
  ground). Used for seed selection and pruning. Does **not** conserve or tile.
- **heft** `Σ_cells (h − base)` — relative to a common base (sea level / global
  min). Conserves and tiles, so it is equalizable. This is what the balanced
  partition targets.

## Key findings (why the obvious approaches fail)

1. **Greedy prune doesn't balance.** Absorbing the smallest peak first gives a
   near-identical map whether scored by prominence or volume, and surviving
   territories are wildly unequal (CV ≈ 1.8). It's an *ultras* map, not an
   equal-heft map.
2. **Cutting the rooted divide tree can't balance either.** The divide tree is
   star-like at big peaks — the global max can directly parent ~40% of all
   basins (357/858 in the fractal test). You cannot peel siblings off the root,
   so the root territory keeps all the base mass and dominates (CV ≈ 1.8). Edge-
   cutting a rooted tree is the wrong operation.
3. **Region-growing on the basin adjacency graph works.** Letting *sibling*
   basins merge (they're adjacent across a saddle, just not parent/child in the
   tree) and growing N seeds by always extending the lightest region yields
   equal-heft connected territories: **CV ≈ 0.13–0.3**, one territory per major
   massif. See `compare_synthetic.png`.

Seeds are chosen by prominence-weighted farthest-point sampling
(`seed_peaks`) so they sit on real peaks but spread out (pure prominence-ranked
seeds cluster and starve each other into slivers).

## Run

    python run_synthetic.py        # validation + compare_synthetic.png

Real DEM: swap `terrain.synthetic` for `terrain.load_dem(path)` (GeoTIFF,
topo+bathy). `load_dem` returns a `cos(lat)` cell-area weight for geographic
rasters — feed it into the heft as a per-cell area so balance is true area·height
mass, not pixel-count mass.

## Status / next

- [x] vectorized basins, divide tree (fast RAG-based `build_tree`), prominence + volume
- [x] balanced equal-heft partition (adjacency-graph region growing)
- [x] real DEM tile (Alps) sanity check — `run_dem.py` → `alps_regions.png`, CV ≈ 0.27–0.39
- [x] area-weight the heft (cos lat) end-to-end; bathymetry/sea base handling
- [ ] **first GEBCO run: big regional subset at full res** (showcases base-to-peak island scoring)
- [ ] optional boundary-rebalancing pass to kill the last slivers at high N
- [ ] cartographic render (coastlines, labels at seed peaks); smoother divides

## DEM data

`fetch_tiles.py <lon0 lat0 lon1 lat1 z out.tif>` grabs + merges AWS open
elevation tiles (no auth) for quick land tests. For topo+bathy use GEBCO_2024:
subset/export GeoTIFF at <https://download.gebco.net/>, drop it in
`../data/ascentshed/`, then `python run_dem.py <that.tif> 1 <N>`.
