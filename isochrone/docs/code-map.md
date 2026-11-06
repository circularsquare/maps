# Code map

Where things live. Update the line for a file you add, move or change the job of.

## Page

| File | What |
|---|---|
| `index.html` | Map (MapLibre 5.24.0, globe projection set on `style.load`; `viewBox()` the view sent to the worker and the river lines, `onPlanet()` drops hovers and clicks off the globe), legend, info panel, the mode buttons (`#controls`, room for later toggles; `showMode` sets the wording), hover with the route line (GeoJSON source `route`, one request in flight), click. Holds `BANDS` (time bands and colours) and `quietBasemap()` (roads and labels of the OpenFreeMap style, numbers in `ROAD` and `LABEL`). One MapLibre canvas source per level (`walk`, `walk3`, `walk1`, `walk01`), holes cut where a finer window shows. Corridors: the `#corrbtn` switch, GeoJSON source and layers `corridors` and `corridors-edge` (below labels), the `CORR_*` ramp, `drawCorridors` (each level's lines simplified and smoothed by `simplifyLine` and Chaikin passes, cut by `clipOut` where the next finer level with lines shows), `lineAt` (the nearest line under the pointer, for its hover). Hover boxes `#tip` and `#stip` are placed by `placeAt`. The speeds panel, the hill shading (`addRelief`) and the `?look=` choice sent with `init` are also here; `loadGrid` reads the walking levels from `data/`, or `data/tiles_<name>/` with `?tiles=<name>`, and shows the GLO-90 credit when the build has direction planes. |
| `worker.js` | Entry of the module worker (`new Worker('worker.js', { type: 'module' })`): routes `init`, `run`, `view`, `route`, `corridors` to `js/session.js`. The worker owns the grids and does all searching and painting. Kept at the root so data URLs stay relative to it. |
| `js/session.js` | State of one session (solved start and its mode, last windows, generations) and the flows: `runClick` (L10 world solve; loads the L10 land hazard band once, started at `init`, and the canoe world data once on the first canoe start), `refine` (L3, L1 and L0.1 view windows, reuse, cancelling), `post` (paint and send), `route` (a hover's route over the solved levels), `countCorridors` (every level once the view's windows are painted, coarse first, stopped by a newer click or view; `postCorridors` counts and sends one) and `setCorridors` (the switch). |
| `js/corridors.js` | Corridors (docs/corridors.md): `count` (area-weighted subtree counts over the predecessor tree, with near-equal land parents re-chosen in `reparent` (T-056, T-077); flux from the coarser level at a window's edge, which nodes lead back to the start), `lines` (polylines of edges past the threshold, 8 strength steps, speed classes from `edgeSpeeds`, the coarser level's lines where a window's routes do not reach the start). |
| `js/route.js` | `trace`: the route to a point back along the predecessor codes, finest level first and on through coarser ones at window seeds; time split into walked, carried, paddled (docs/canoe.md 6). |
| `js/levels.js` | `makeLevel` (a block of cells with its geometry), `snap`, `sampleNode` (weight, time, hazard of one layer at a point, for window seeds), `HZ_RATE` and `setHazard` (the level's land hazard per km walked, `L.hq`, from its band; docs/hazard.md 4), `setDirs` (each land cell's four directed hours per km, `L.sd`, from its direction planes, or null; docs/resolution.md "Directions"). |
| `js/search.js` | `solve`: Dijkstra over land and canoe nodes (docs/canoe.md 5), weight = time + K x hazard (water, and land from `L.hq`), predecessor codes `P_*`, yields every ~30 ms, runs walking starts too; `solveWalk`, the small walking loop, for hazard off; both walk by `L.sd` when the level has it (`dirWeights`, a move's east-west share by row); `cellTimes` (least time of each cell's nodes, what is painted). |
| `js/canoe.js` | `buildCanoe`: a level's canoe layer from its data (`canoeOf` and the further nodes of a cell from the steps' and water units' slots, the per-cell word, current, each further node's access, river edges by node, the edges into river water off the line and the one-pixel ponds that count as it, T-036, T-075, T-082), `waterLinks` (the listed water links as edges in the same lists, T-088); `eachCanoeNode` (a cell's canoe nodes), `canoeFrom` (which of a cell's canoe nodes a predecessor code came from); `shoreAt`. |
| `js/canoedata.js` | Canoe file layouts (docs/canoe.md 4): `loadWorld` (L10 water, sea grid, river steps, water units and links, once), `loadWindowWater` (L3/L1 water tiles, river tiles and pieces tiles cut to a window; missing tiles are open sea); river steps with their slots and access where the file has them (L3, L10); `parsePieces` (T-088; no `pieces` key in the index, none). |
| `js/windows.js` | `windowFor` (window bounds and caps, lower on canoe starts), `fineWindowFor` (the L0.1 window, or none: view over the cap, a canoe start, under 5% of the view in fine blocks, `FINE_MIN_SHARE`), `contains`, `solveWindow` (speeds, direction planes and land hazard loaded together) with the border-ring seeding from the coarser levels, per layer. |
| `js/tiles.js` | L3 and L1 tile fetch from the build's folder (`setTiles`, the page's `?tiles=`), the shared 250 MB cache of speed, road, direction, canoe and hazard tiles (`getCached`), `loadWindow` (speed and road tiles cut into a window grid, and with `dirs` the direction planes, `e` only with roads on), `applyRoads` (road bytes folded into speeds, docs/roads.md; L10 too, from `session.js`), `cutInto`, `forTiles`. Land hazard (`data/hazard/`): `loadHazardWorld` (L10), `loadHazardWindow` (L3/L1 tiles, missing tiles 0; L0.1 the L1 pixel under each cell). L0.1: `fineIndex` (`data/fine/index.json`, each tile's block mask in `ix.mask`), `fineShare` (share of a box in fine blocks), `fineSpeed` (class, roads and terrain factor to speeds, docs/fine.md), its own ~100 MB cache of decoded tiles; `loadFineWindow` fills blocks without 100 m data from the L1 speeds (`loadWindow` at f 1, roads folded in). |
| `js/paint.js` | `paint`: a solved level's cell times to RGBA in Web Mercator y; water paler on canoe starts; the cell looks (T-060, `cfg.look`): `hex` the default (pixel-to-cell map kept per grid geometry, packed 32-bit colours), `?look=crisp` square cells, `?look=diag` cut corners. |
| `js/speeds.js` | `speeds()`: the table the page's speeds panel shows, sent with `ready`: our LC100 walking table (MAP's survey speeds), the height and slope factors, and the canoe speeds worked out from `cfg`. |
| `js/riverlines.js` | Page side (dynamic import from `index.html`): `riverLines` adds the `rivers` line layer from the canoe river files, L10 / L3 / L1 by zoom; picks the tiles and draws what `js/riverworker.js` (a module worker: fetch, parse, steps joined between confluences and smoothed) sends back; `window.riverLog` has each rebuild's time on the page (`ms`) and in the worker (`workerMs`). The hill shading (`addRelief`, source `dem`) is in `index.html`. |
| `js/config.js` | What `init` sends (bands, alpha, time limit, the speed to hours table, canoe and hazard data folders), `roads` (fold the road bytes in) and the canoe model constants (K, paddling speed, upstream share and floor, transfer, carry factor, ice limit, `sea8`, `hazard`). |

Messages, page to worker: `init {..., bytes, roads, dirs, dirsRoads, tilesBase, canoeBase, look, corridors}` (`bytes`, `roads`, `dirs`, `dirsRoads` the L10 speeds, road bytes and direction planes; `tilesBase` the L3/L1 tiles' folder, `data/tiles_<name>/tiles/` with the page's `?tiles=<name>`), `run {lng, lat, mode}` (mode `walk` or
`canoe`), `view {west, east, north, south, zoom}`, `route {lng, lat, seq}`, `corridors {on}`.
Worker to page: `ready {speeds}`, `none`, `error {message}`, `coarse {mode}`, `win {level}`,
`clear {levels}`, `route {seq, mode, t, hz, survival, parts, lng, lat, kind}` (no `lng` when the
point is not reached), `corridors {level, xy, off, s, v, ms, countMs}` (`v` the per-line byte: step, mode, speed class), each with the `originId` of
its click. `window.solveLog` on the page holds every solve's timings, `window.corrLog` each
level's corridor time and line count, `window.lastRoute` the last route's figures.

## Data build

| File | What | Time |
|---|---|---|
| `tools/build_tiles.py [workers] [--out NAME] [--planes DIRS] [--strips A-B]` | Our surface (`data/surface/`), the roads' 1 km counts (`data/roads/l1/`) and the directed terrain planes where they exist (`data/terrain/l1/`) to the page's levels in `data/NAME/` (default `tiles_next`): L1 and L3 5-degree tiles (`l1/`, `l3/` speeds, `r1/`, `r3/` road bytes, `d1/`, `d3/`, `e1/`, `e3/` direction planes, `index.json`), L10 `grid.bin.gz`, `grid_roads.bin.gz`, `grid_dir.bin.gz`, `grid_dir_roads.bin.gz`, `grid.json`. Pass 1 (`l1_strip`): the water rule, LC100's missing islands, bridges, the road byte (`road_byte`, roads at a 10% grade cap), the planes' speeds (`directed`, `dir_bytes`). Pass 2 (`coarse_strip`): L3 and L10 by crossing the written L1 with the page's directed move costs (`move_costs`, `crossings`, `strip_scaled`), the mean rule (`coarse`) where not crossed. `swap <backup> [--from NAME]` serves a build by renames, keeping the old in `data/<backup>/` | ~1 h on 1 |
| `tools/smallwater.py` | Finds the water patches under 5 km² (8-connected, strip by strip, merged across strip edges and the 180 degree seam), on any surface (`is_water`); the build makes them land | ~35 s of a build |
| `tools/inspect_tif.py` | Prints facts about the raw tif | |

Our own walking surface (`docs/surface.md`): layers on MAP's 1 km grid in `data/surface/`
(gitignored), read by `tools/build_tiles.py`. MAP's own surface, which the page read until T-028,
is in `data/tiles_map2020/` as the page's levels (to compare and roll back).

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
| `tools/fine/select.py` | Which tiles get L0.1 (docs/fine.md "Where the fine level goes"), from the 1 km layers: town blocks with their tile's block mask, water country, mountains, into `data/fine/selection.json` | ~1 min |
| `tools/fine/build.py world\|tiles\|region\|index` | 1 x 1 degree L0.1 tiles: LC100 class and the L1 terrain factor (`.land`), road bits (`.roads`). `world [--jobs N] [--resume]`: every tile of `selection.json` plus the test regions, absent town blocks zeroed, roads from `data/roads/bits/`; `tiles <name>...`: whole tiles from `bits/`; `region <name>`: a test region (`REGIONS`) with roads from its Geofabrik extracts in `data/raw/osm/`. Each ends by rewriting `index.json` (whole tiles, partial tiles with their masks, tiles with roads) | world ~15 min on 3; Cluj region 3.5 min, New York ~12 min |

Roads (`docs/roads.md`), into `data/roads/` (gitignored): `tools/roads/regions.py` (the world's Geofabrik extracts, `regions.json`), `fetch.py` (download with md5 check), `build.py extract|reduce` (100 m road bits per tile, OR-merged across extracts; 1 km counts and kinds), `world.py` (the world pass: download, read and delete each extract in turn, two on disk at most, done.txt resume, stops under 6 GB free). ~5 min a GB of extract on one core.

Terrain (T-011, design in `notes/T-011.md` until the rebuild lands), into `data/terrain/` (gitignored), GLO-90 streamed through `data/raw/glo90/`: `tools/terrain/glo90.py` (tile names, the bucket's tile list, `fetch`, `Stream`: parallel GETs in list order with a download-ahead limit and the disk guard, never under 22 GB free; `read_fine`, elevation at fine-cell centres), `walk.py` (the directed 100 m walker's move costs, `crossings` of L1 cells by block sweeps in numpy, `strip_ratio`, `tile_l1`: a tile's 8 directed L1 planes; `csr` for whole-grid checks), `cross.mjs` (the same costs and crossings in node, `walk.crossings_js`), `plan.py` (the world list `plan.json`: 18,984 tiles with L1 land, road cells, which keep elevation for the fine level), `world.py run|status` (the run: stream, convert, compute on a pool, `done.txt` resume, `world.log`, `world.pid`).

Canoe water and sea (`docs/canoe.md` 4, layouts in `notes/T-021.md`), into `data/canoe/`:

| File | What | Time |
|---|---|---|
| `tools/canoe/water.py` | The speed tiles' water, NE ocean and HydroLAKES to `kind`, `shore`, `links` at L10 (`water.bin.gz`), L3 and L1 tiles, `index.json`; `--cache DIR` keeps the L1 arrays; `--tiles DIR --grid FILE --out DIR` builds a test set from other speed levels (as `rivers.py build`) | ~8 min, ~6 GB |
| `tools/canoe/sea_fetch.py` | `drifter`: GDP drifter climatology from ERDDAP; `era5 [threads]`: streams ARCO-ERA5 to a summer climatology `data/raw/era5_summer/clim.npz`; `era5ice`: August and February sea ice to `clim_ice.npz` (needs `numcodecs`) | 30 s; ~7 min; ~1.5 min |
| `tools/canoe/sea.py` | Those to the 0.25° local-summer grid `sea.bin.gz` (currents, waves, ice, sea temperature) | seconds |
| `tools/canoe/water_plots.py` | Strait and link checks, PNGs `maps/artifacts/isochrone/t021_*.png` | ~1 min |

Canoe rivers (`docs/canoe.md` 4, layouts in `notes/T-020.md`), into `data/canoe/rivers.bin.gz`,
`data/canoe/rivers/` and the `rivers` key of `data/canoe/index.json`; work files in
`data/raw/rivers_work/`:

| File | What | Time |
|---|---|---|
| `tools/canoe/rivers_osm.py` | OSM waterfalls, dams, weirs and rapids worldwide from QLever to `data/raw/osm_barriers/*.tsv` | ~1 min |
| `tools/canoe/rivers.py` | Stages: `reaches` (RiverATLAS gdb to Q >= 1 reaches), `lakes` (HydroLAKES to an L1 lake/reservoir raster, pour points), `snap` (barriers to reaches), `desert` (which reaches are rivers in hot desert and the flow they carry in, `desert.npz`, via `tools/hotdesert.py`), `build` (river steps at L1 from the reaches, leaving out dry desert reaches, GMTED gradient north of 60N; L3 and L10 as L1 contracted, with slots and access, `contract_level`, T-082; the water units and links of L3 and L10, `water_units`, written to `pieces.bin.gz` and `pieces/l3/` with the `links` band of the coarse water files rewritten unit to unit, T-088; `--kind DIR` reads canoe water from another folder), `contract` (L3 and L10 again from the L1 river tiles as written, ~5 min), `index` (re-adds the `rivers` key) | ~3.5, 4, 1, 0.2, 7 min; build peaks ~8 GB |
| `tools/canoe/pieces.py` | `label_level`: the water pieces of a coarse level's split cells (8-connected L1 sea and lake inside each cell), the piece pairs that touch across cell borders, each piece's access, for `rivers.py` (T-088) | ~2 min of a build |
| `tools/hotdesert.py` | The hot desert rule shared by the canoe's rivers and the land hazard's water: the mask (`hot_g2.npy`, made by `hazard/build.py hot`), each reach's share in it, and the flow a desert river carries in from outside it | |
| `tools/canoe/rivers_check.py` | `stats`: steps, km and classes per level and continent, read back from the files; `png`: river PNGs `maps/artifacts/isochrone/t020_*.png` | ~1 min each |

Land hazard (`docs/hazard.md`), into `data/hazard/` (gitignored): `tools/hazard/fetch.py`
(CHELSA, OSM wells/springs/settlements from QLever, ice outlines, into `data/raw/`; ~6 min),
`tools/hazard/build.py ice|hot|dist|tiles|plot` (ice outlines to L1 north of 60 N ~1 min; the hot
desert mask ~1 min; distance to drinkable water on a 2 km grid ~3 min, after `rivers.py desert`; the uint8 rate band as `l10.bin.gz`, L3 and L1 tiles and
`index.json` ~5 min, 38.6 MB; the world PNG `t066_world.png`), one core, peak ~6 GB. `dist` and
`tiles` take land from the page's L1 speed tiles, so they run after `tools/build_tiles.py`.

Run with plain `python` (C:\Python39). Raw downloads: `data/raw/` (gitignored); the MAP surface
is `data/raw/unzipped/2020_walking_only_friction_surface.geotiff` (726 MB), LC100 is
`data/raw/lc100/` (2.3 GB), GMTED `data/raw/gmted/mn30_grd/` (0.4 GB). RiverATLAS is not kept
(6.8 GB unpacked): `rivers.py reaches` needs it back in `data/raw/riveratlas/` (figshare file
20087321, ~13 min); its output `data/raw/rivers_work/reaches.npz` and `all_lines.npz` are kept.

## Checks

| File | What |
|---|---|
| `checks/smoke.mjs` | Headless Chrome on the served page (`node isochrone/checks/smoke.mjs [url] [--canoe=<folder>]`, ~30 s): default solve, its corridors arriving with lines and the corridors switch, a click, L3 and L1 windows, hover; a walking start at Tamanrasset with survival in a hover to the south-west; when `data/fine/` has New York, a start in Midtown at zoom 12 with an L0.1 window, Brooklyn reached over the bridges and hover from L0.1; when it has Lagos as a partial tile, a start there at zoom 12 with an L0.1 window; when canoe data is served, a canoe start from Constanta, the sea painted, a hover at sea with its route and split, an L1 window with sea, the mode buttons, its L10 corridor lines all leading back to the start; no console errors. One PASS or FAIL line each; exit 1 on a fail. |
| `checks/corridors.mjs` | The corridor tree in node over the real L10 data (~2 min): for two canoe starts and a walking one, every parent reached within the limit at no more weight than its child, every line ending at the start or on another line, and the first start counted again after the others giving the same lines (T-090). |
| `checks/levels.mjs` | Level agreement in node over a build (`[--tiles=<name>] [--only=a,b] [--seams] [--updown]`, ~1-3 min): T-080's 25 starts, L1/L10, L1/L3, L3/L10 times at 20-300 km with each level solved alone; window seams against L10; walks up and down between a valley and a height, both ways (T-011). Numbers only, no pass or fail. |
| `checks/ktrips.mjs` | The K check trips in node over the real L10 data (`trips <K...>`, ~30 s a K; `time`): each canoe trip's time, survival, hours past 20 km and route waypoints, canoe solve times with hazard off and on (notes/T-006.md), and the walking trips with land hazard (T-077, notes/T-042.md). |
