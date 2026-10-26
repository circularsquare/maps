# Code map

Where things live. Update the line for a file you add, move or change the job of.

## Page

| File | What |
|---|---|
| `index.html` | Map, legend, info panel, hover, click. Holds `BANDS` (time bands and colours). One MapLibre canvas source per level (`walk`, `walk3`, `walk1`), holes cut where a finer window shows. |
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

Run with plain `python` (C:\Python39). Raw downloads: `data/raw/` (gitignored); the MAP surface
is `data/raw/unzipped/2020_walking_only_friction_surface.geotiff` (726 MB).

## Checks

| File | What |
|---|---|
| `checks/smoke.mjs` | Headless Chrome on the served page (`node isochrone/checks/smoke.mjs [url]`, ~20 s): default solve, a click, L3 and L1 windows, hover, no console errors. One PASS or FAIL line each; exit 1 on a fail. |
