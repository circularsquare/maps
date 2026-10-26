# Code map

Where things live. Update the line for a file you add, move or change the job of.

## Page

| File | What |
|---|---|
| `index.html` | Map, legend, info panel, hover, click. Holds `BANDS` (time bands and colours). One MapLibre canvas source per level (`walk`, `walk3`, `walk1`), holes cut where a finer window shows. |
| `worker.js` | Owns the grids and does all searching and painting. `solve` (Dijkstra, 16 moves, yields every ~30 ms), `runClick` (L10 world solve), `refine` / `windowFor` / `loadWindow` / `solveWindow` (L3 and L1 view windows, seeded from the coarser level), `paint` (RGBA in Web Mercator y), `snap` / `sample`. |

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

Run with plain `python` (C:\Python39). Raw downloads: `data/raw/` (gitignored); the MAP surface
is `data/raw/unzipped/2020_walking_only_friction_surface.geotiff` (726 MB).

## Checks

| File | What |
|---|---|
| `checks/smoke.mjs` | Headless Chrome on the served page (`node isochrone/checks/smoke.mjs [url]`, ~20 s): default solve, a click, L3 and L1 windows, hover, no console errors. One PASS or FAIL line each; exit 1 on a fail. |
