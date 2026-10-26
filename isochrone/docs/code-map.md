# Code map

Where things live. Update the line for a file you add, move or change the job of.

## Page

| File | What |
|---|---|
| `index.html` | Map, legend, info panel, hover, click. Holds `BANDS` (time bands and colours) and `quietBasemap()` (roads and labels of the OpenFreeMap style, numbers in `ROAD` and `LABEL`). One MapLibre canvas source per level (`walk`, `walk3`, `walk1`), holes cut where a finer window shows. |
| `worker.js` | Entry of the module worker (`new Worker('worker.js', { type: 'module' })`): routes `init`, `run`, `view` to `js/session.js`. The worker owns the grids and does all searching and painting. Kept at the root so data URLs stay relative to it. |
| `js/session.js` | State of one session (solved start, last windows, generations) and the flows: `runClick` (L10 world solve), `refine` (L3 and L1 view windows, reuse, cancelling), `post` (paint and send). |
| `js/levels.js` | `makeLevel` (a block of cells with its geometry), `snap`, `sample`. |
| `js/search.js` | `solve`: Dijkstra with an indexed heap, 16 moves, yields every ~30 ms. |
| `js/windows.js` | `windowFor` (window bounds and caps), `contains`, `solveWindow` with the border-ring seeding from the coarser levels. |
| `js/tiles.js` | L3 and L1 tile fetch, the 200 MB cache, `loadWindow` (tiles cut into a window grid). |
| `js/paint.js` | `paint`: a solved level to RGBA in Web Mercator y. |
| `js/config.js` | What `init` sends (bands, alpha, time limit, the speed to hours table). |

Messages, page to worker: `init`, `run {lng, lat}`, `view {west, east, north, south, zoom}`.
Worker to page: `ready`, `none`, `coarse`, `win {level}`, `clear {levels}`, each with the
`originId` of its click. `window.solveLog` on the page holds every solve's timings.

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
| `checks/smoke.mjs` | Headless Chrome on the served page (`node isochrone/checks/smoke.mjs [url]`, ~20 s): default solve, a click, L3 and L1 windows, hover, no console errors. One PASS or FAIL line each; exit 1 on a fail. |
