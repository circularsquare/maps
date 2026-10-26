# Code map

Where things live. Update the line for a file you add, move or change the job of.

## Page

| File | What |
|---|---|
| `index.html` | Map, legend, info panel, the mode buttons (`#controls`, room for later toggles; `showMode` sets the wording), hover with the route line (GeoJSON source `route`, one request in flight), click. Holds `BANDS` (time bands and colours) and `quietBasemap()` (roads and labels of the OpenFreeMap style, numbers in `ROAD` and `LABEL`). One MapLibre canvas source per level (`walk`, `walk3`, `walk1`, `walk01`), holes cut where a finer window shows. |
| `worker.js` | Entry of the module worker (`new Worker('worker.js', { type: 'module' })`): routes `init`, `run`, `view`, `route` to `js/session.js`. The worker owns the grids and does all searching and painting. Kept at the root so data URLs stay relative to it. |
| `js/session.js` | State of one session (solved start and its mode, last windows, generations) and the flows: `runClick` (L10 world solve; loads the canoe world data once on the first canoe start), `refine` (L3, L1 and L0.1 view windows, reuse, cancelling), `post` (paint and send), `route` (a hover's route over the solved levels). |
| `js/route.js` | `trace`: the route to a point back along the predecessor codes, finest level first and on through coarser ones at window seeds; time split into walked, carried, paddled (docs/canoe.md 6). |
| `js/levels.js` | `makeLevel` (a block of cells with its geometry), `snap`, `sampleNode` (weight, time, hazard of one layer at a point, for window seeds). |
| `js/search.js` | `solve`: Dijkstra over land and canoe nodes (docs/canoe.md 5), weight = time + K x hazard, predecessor codes `P_*`, yields every ~30 ms; `solveWalk`, the small walking loop; `cellTimes` (least time of each cell's nodes, what is painted). |
| `js/canoe.js` | `buildCanoe`: a level's canoe layer from its data (`canoeOf`, the per-cell word, current, river edges); `shoreAt`. |
| `js/canoedata.js` | Canoe file layouts (docs/canoe.md 4): `loadWorld` (L10 water, sea grid, river steps, once), `loadWindowWater` (L3/L1 water tiles and river tiles cut to a window; missing tiles are open sea). |
| `js/windows.js` | `windowFor` (window bounds and caps, lower on canoe starts), `fineWindowFor` (the L0.1 window, or none: view over the cap, a canoe start, tiles missing), `contains`, `solveWindow` with the border-ring seeding from the coarser levels, per layer. |
| `js/tiles.js` | L3 and L1 tile fetch, the shared 250 MB cache of speed and canoe tiles (`getCached`), `loadWindow` (tiles cut into a window grid), `cutInto`, `forTiles`. L0.1: `fineIndex` (`data/fine/index.json`), `fineSpeed` (class, roads and terrain factor to speeds, docs/fine.md), its own ~100 MB cache of decoded tiles. |
| `js/paint.js` | `paint`: a solved level's cell times to RGBA in Web Mercator y. |
| `js/speeds.js` | `speeds()`: the table the page's speeds panel shows, sent with `ready`: MAP's walking table, its height and slope factors, and the canoe speeds worked out from `cfg`. |
| `js/riverlines.js` | Page side (dynamic import from `index.html`): `riverLines` adds the `rivers` line layer from the canoe river files, L10 / L3 / L1 by zoom, steps joined between confluences and smoothed; `window.riverLog` has each rebuild's time. The hill shading (`addRelief`, source `dem`) is in `index.html`. |
| `js/config.js` | What `init` sends (bands, alpha, time limit, the speed to hours table, canoe data folder) and the canoe model constants (K, paddling speed, upstream share and floor, transfer, carry factor, ice limit, `sea8`, `hazard`). |

Messages, page to worker: `init {..., canoeBase}`, `run {lng, lat, mode}` (mode `walk` or
`canoe`), `view {west, east, north, south, zoom}`, `route {lng, lat, seq}`.
Worker to page: `ready {speeds}`, `none`, `error {message}`, `coarse {mode}`, `win {level}`,
`clear {levels}`, `route {seq, mode, t, hz, survival, parts, lng, lat, kind}` (no `lng` when the
point is not reached), each with the `originId` of its click. `window.solveLog` on the page holds
every solve's timings, `window.lastRoute` the last route's figures.

## Data build

| File | What | Time |
|---|---|---|
| `tools/build_grid.py` | MAP walking tif to the L10 world grid `data/grid.bin.gz` + `data/grid.json` (uint8 km/h x 40, 0 = water) | ~1.5 min |
| `tools/build_tiles.py` | Same tif to L3 and L1 5-degree tiles in `data/tiles/` (gitignored) + `index.json` | ~3 min |
| `tools/smallwater.py` | Finds the water patches under 5 km² (8-connected, strip by strip, merged across strip edges and the 180 degree seam); both builds make them land | ~45 s of each build |
| `tools/inspect_tif.py` | Prints facts about the raw tif | |

Our own walking surface (`docs/surface.md`), not yet read by the page: layers on MAP's 1 km
grid in `data/surface/` (gitignored).

| File | What | Time |
|---|---|---|
| `tools/surface/common.py` | Grid, paths, the LC100 speed table `LC_SPEED`, MAP's elevation and slope factors, the GeoTIFF profile | |
| `tools/surface/elevation.py` | GMTED2010 30" to `elev.tif` and `slope.tif` | ~4.5 min |
| `tools/surface/landcover.py` | LC100 100 m to `lc_speed.tif`, `lc_water.tif`, `lc_class.tif` by exact area overlap; `[workers]` argument | ~25 min on 4 |
| `tools/surface/combine.py` | The layers to `terrain_factor.tif` and `walk_today.tif` (off-road km/h x 1000) | ~4 min |
| `tools/surface/compare.py` | `walk_today.tif` against MAP off-road: tables, `compare.json`, PNGs to `maps/artifacts/isochrone/t009_*.png` | ~4 min |

Fine level (`docs/fine.md`), into `data/fine/` (gitignored):

| File | What | Time |
|---|---|---|
| `tools/fine/build.py <region>` | 1 x 1 degree L0.1 tiles: LC100 class and the L1 terrain factor (`.land`), OSM road bits from the Geofabrik extracts in `data/raw/osm/` (`.roads`); `index` rewrites `index.json`. Regions in `REGIONS` | Cluj 3.5 min, New York ~12 min |

Canoe water and sea (`docs/canoe.md` 4, layouts in `notes/T-021.md`), into `data/canoe/`:

| File | What | Time |
|---|---|---|
| `tools/canoe/water.py` | The speed tiles' water, NE ocean and HydroLAKES to `kind`, `shore`, `links` at L10 (`water.bin.gz`), L3 and L1 tiles, `index.json`; `--cache DIR` keeps the L1 arrays | ~11 min, ~6 GB |
| `tools/canoe/sea_fetch.py` | `drifter`: GDP drifter climatology from ERDDAP; `era5 [threads]`: streams ARCO-ERA5 to a summer climatology `data/raw/era5_summer/clim.npz`; `era5ice`: August and February sea ice to `clim_ice.npz` (needs `numcodecs`) | 30 s; ~7 min; ~1.5 min |
| `tools/canoe/sea.py` | Those to the 0.25° local-summer grid `sea.bin.gz` (currents, waves, ice, sea temperature) | seconds |
| `tools/canoe/water_plots.py` | Strait and link checks, PNGs `maps/artifacts/isochrone/t021_*.png` | ~1 min |

Canoe rivers (`docs/canoe.md` 4, layouts in `notes/T-020.md`), into `data/canoe/rivers.bin.gz`,
`data/canoe/rivers/` and the `rivers` key of `data/canoe/index.json`; work files in
`data/raw/rivers_work/`:

| File | What | Time |
|---|---|---|
| `tools/canoe/rivers_osm.py` | OSM waterfalls, dams, weirs and rapids worldwide from QLever to `data/raw/osm_barriers/*.tsv` | ~1 min |
| `tools/canoe/rivers.py` | Stages: `reaches` (RiverATLAS gdb to Q >= 1 reaches), `lakes` (HydroLAKES to an L1 lake/reservoir raster, pour points), `snap` (barriers to reaches), `build` (river steps at L10, L3, L1; GMTED gradient north of 60N), `index` (re-adds the `rivers` key) | ~3.5, 4, 1, 4 min; build peaks ~8 GB |
| `tools/canoe/rivers_check.py` | `stats`: steps, km and classes per level and continent, read back from the files; `png`: river PNGs `maps/artifacts/isochrone/t020_*.png` | ~1 min each |

Run with plain `python` (C:\Python39). Raw downloads: `data/raw/` (gitignored); the MAP surface
is `data/raw/unzipped/2020_walking_only_friction_surface.geotiff` (726 MB), LC100 is
`data/raw/lc100/` (2.3 GB), GMTED `data/raw/gmted/mn30_grd/` (0.4 GB). RiverATLAS is not kept
(6.8 GB unpacked): `rivers.py reaches` needs it back in `data/raw/riveratlas/` (figshare file
20087321, ~13 min); its output `data/raw/rivers_work/reaches.npz` and `all_lines.npz` are kept.

## Checks

| File | What |
|---|---|
| `checks/smoke.mjs` | Headless Chrome on the served page (`node isochrone/checks/smoke.mjs [url] [--canoe=<folder>]`, ~30 s): default solve, a click, L3 and L1 windows, hover; when `data/fine/` has New York, a start in Midtown at zoom 12 with an L0.1 window, Brooklyn reached over the bridges and hover from L0.1; when canoe data is served, a canoe start from Constanta, the sea painted, a hover at sea with its route and split, an L1 window with sea, the mode buttons; no console errors. One PASS or FAIL line each; exit 1 on a fail. |
| `checks/ktrips.mjs` | The K check trips in node over the real L10 data (`trips <K...>`, ~30 s a K; `time`): each trip's time, survival, hours past 20 km and route waypoints, and canoe solve times with hazard off and on (notes/T-006.md). |
