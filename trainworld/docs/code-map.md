# Code map

Where things live and how they are built. Read when you need to find a file; the rules are in
`CLAUDE.md`. Keep a line here current when you add, move or split a module.

## sim/: the Rust crate (`trainworld-sim`)

Compiled to WebAssembly with wasm-pack 0.15. `cargo test --manifest-path sim/Cargo.toml` runs its
tests natively.

- `src/track/`: the track model (T-040): geometry, costs, crossings, world frames (`frame.rs`,
  T-125), the network and its edits with undo (`world.rs`; blueprint and construct, T-055),
  run-time profiles, capacity, the boarding account (`boarding.rs`: each train's load and each
  platform's crowd, T-164), saves, local time (`localtime.rs`, T-309), and `wasm.rs` (`TrackApi`,
  the clock worker's API, with the drawing tool's routes). In the default build.
  `cargo run --release --bin track_bench` times edits on a synthetic 2,000 km network; feature
  `track-bench` exports it to WASM. notes/T-040.md.
- `src/demand/`: the demand kernel, behind feature `demand` (on in the app's build): `api.rs`
  (`DemandApi`, the demand workers' API), `intercity.rs` (the regional market), `warm.rs` (the
  everyday warm-up demand worker 0 runs at startup, T-173), `days.rs` (the weekend's factors per
  purpose and period, T-260), `routes.rs` (routed walks round water and drives over OSM roads from
  the pack, T-275), `trip.rs` (the trip explorer's query, T-195).
  `cargo test --release --features demand nyc_day -- --ignored --nocapture` solves a day on the
  real pack natively. notes/T-026.md.

## app/: the browser game

Vite 8 + TypeScript 7 + MapLibre GL 6.13 (ESM only: `import * as maplibregl`, worker URL passed
via `setWorkerUrl`, no `map.transform`) + Preact 11 with signals (TSX, no Vite plugin).
`src/main.ts` wires everything; `src/wasm/` is generated: `track/` for the clock worker, `demand/`
for the demand workers (two modules from the one crate, notes/T-051.md).

### src/game/: what the UI reads

- `types.ts` (the save's inputs, worker results, `WorldView`, `Selection`), `world.ts` (the
  snapshot signal and the edit functions), `ui.ts` (tab, selection, display settings, theme).
- Time: `clock.ts`, `localTime.ts` (each region's local time and the local time in view, T-309),
  `periods.ts` (the five periods), `days.ts` (weekday and weekend, the shown day, T-260).
- Demand and riders: `demand.ts` (results: mode split, riders), `boarding.ts` (the boarding
  account's hour on the main thread: riders packed for the clock worker, platform crowds, T-164),
  `complexLive.ts` (a complex's platform crowds and trains now, T-246), `purposes.ts` (the trip
  purposes as shown, T-192), `trip.ts` (the trip explorer's state).
- Places and frames: `coords.ts` (world metres to lng/lat), `frame.ts` and `registry.json` (the
  world frames and the one city registry, read by `cities.ts`, T-125), `places.ts` (station and
  compass names for the inspectors), `placeList.ts` (each city's OSM place list, fetched on first
  search or station naming, T-177, T-065), `boundaries.ts` (each city's boundary file, fetched
  once, T-172), `search.ts` (the search box's matching, T-148), `walks.ts` (walking minutes to the
  closest stations on the demand model's walking barrier, T-312).
- Stations and lines: `complexes.ts` (station complexes from platform geometry and the player's
  links, T-146), `complexView.ts` (a click selecting the complex, a station opened alone, the
  zoomed-out dot groups; T-217, T-249), `links.ts` (the player's links, T-197), `lineActions.ts`
  (new and copied lines), `remove.ts` (removing things, with the question asked before removing
  anything constructed, T-096).
- Other: `money.ts` (cash and ledger from the worker, fare income from riders, T-028), `issues.ts`
  (the track model's refusals in plain words), `loading.ts` (what is loading, the startup gate and
  the clock hold, T-149, T-172), `palette.ts`, `format.ts`.
- The `WorldView` is built from the clock worker's states in `workers/clockClient.ts`.

### src/map/: the map and what is drawn on it

- `mapView.ts` (the MapLibre map), `basemap.ts` (the Positron style: faint repaint, labels in Zen
  Maru Gothic from `public/fonts/`), `overlay.ts` (the overlay canvas and frame loop,
  notes/T-015.md), `renderer.ts` (WebGL2 track, stations, trains), `network.ts` (the renderer's
  buffers), `geo.ts` (the draw anchor, T-307), `anchorVersion.ts` (its version as a signal, T-317),
  `live.ts` (the live map objects for UI code, set by main.ts), `synth.ts` (stress network).
- Stations: `stationGeometry.ts` (platform shapes along the track for drawing, picking and the
  ghost, T-189, T-254), `stationLabels.ts`, `capacityMarkers.ts`, `labelLayout.ts` (shared spacing
  for names and tags).
- Tools: `pick.ts`, `tools.ts` (drawing track, placing stations, the line tool), `edgeEdit.ts`
  (dragging blueprint track's PIs, T-064, T-092), `nodeDrag.ts` (dragging blueprint nodes, T-093),
  `blueprintBox.ts` (the Select tool's drag box, T-333; its state is `game/blueprintPick.ts`),
  `stopMarkers.ts` (the toggle markers of stop editing, T-233).
- Trains and trips: `trainFollow.ts` (a selected train followed onto its next trip, T-245),
  `tripView.ts` (the trip explorer's route, T-195; its panel is `ui/inspector/TripInspector.tsx`).
- Demand views (T-084, T-097): `demandViews.ts` (wiring, a station's catchment), `demandLayers.ts`
  (line width, station size, train fill), `commuterView.ts` and `bubbles.ts` (commuter bubbles and
  bubble selections), `intercityView.ts` (longer trips of a selection or station, one bubble per
  far zone, T-118, T-120; one per city zoomed far out, T-186), `catchUp.ts` (the bubbles catching
  up over a game day, T-156), `demandOverlay.ts` (their canvas, over the network's),
  `demandHover.ts` (the count tag by the pointer).
- Tags and outlines: `trackHover.ts` and `networkTips.ts` (the hover tags on track, trains and
  stations, T-157, T-194), `cityBoundaries.ts` and `boundaryGeometry.ts` (the metro outlines and
  outside shade, T-131, T-139), `zoneLoading.ts` (zones still loading drawn grey with a loading
  bar, T-343).

### src/ui/: Preact components

`Dock.tsx` (the dock, its icon tabs and resizing, T-252); `tabs.tsx` (Build, Lines, Stations and
Money); `SettingsTab.tsx`; `SavesPanel.tsx` (saved games, T-251); `widgets.tsx` (chips, steppers,
checkboxes and the inline SVG icons); `TopBar.tsx`; `StatusTag.tsx` (the demand status tag on the map's top row); `Toolbar.tsx` (tool hints and the confirmation
dialog); `DemandSelector.tsx` (None/Homes/Destinations and the purpose filter); `Blueprint.tsx`
(Build's quote and Construct, and the Select tool's selection bar on the map, T-353); `Debug.tsx` (the `?debug=1` readout); `inspector/` (one inspector
per selection kind: `others.tsx` holds station, train, track, junction and cell or zone;
`ComplexMap.tsx` the complex map); `LoadingScreen.tsx` (the clock hold at startup, the loading card,
the screen when demand cannot load, T-149, T-172, T-180); `Search.tsx` (T-148); `StationsFull.tsx`
(the "N stations full" button, T-231); `keys.ts` (which keys a focused control keeps, T-183);
`tip.ts` (the hover tag for every panel and control, from `data-tip`, never `title`, T-210);
`style.css` (theme tokens).

### src/workers/

`protocol.ts` (edit ops, clock messages, the network snapshot for demand), `clock.worker.ts` (owns
the network: `TrackApi`, notes/T-025.md; money and autosave too), `saveFile.ts` (the save format
and IndexedDB, T-029), `clockClient.ts`, `demand.worker.ts`, `demandClient.ts`, `demandCache.ts`
(the last final demand result kept beside the save, T-328),
`demandProtocol.ts`, `demandTest.ts`, `cityReach.ts` (which city packs a network's stations can
reach, T-149), `commuters.worker.ts` and `commutersProtocol.ts` (the commuter bubbles' sums,
T-097), `selectionMerge.ts` (a bubble selection's everyday far end added up over the workers,
T-285).

### URL switches

`?debug=1` (readout; `window.tw` probes incl. `lagStart/lagStop`, `checkPrecision`; `window.twDemand`
with demand timings and `client.qlog` per query; `window.twViews` with the demand views'),
`?paused=1`, `?new=1` (a new game instead of the autosave; it owns no slot and never writes one
until a copy is saved, T-320, so it is safe in a browser profile holding real games),
`?synth=1&lines=500&trains=10000`, `?demandTest=1` (demand on T-005's hand-made network; both also
never autosave), `?demandWorkers=N`, `?perfOff=...` / `?perfTry=...` (CPU A/B switches,
`src/perfFlags.ts`, notes/T-045.md).

### Build

`npm run build --prefix app` = wasm-pack twice (Cargo profile `wasm`), type check, `app/dist/`
(city packs copied into `dist/packs/` by `vite.config.ts`), which serve.py serves at
`localhost:8800/trainworld/` (from the snapshot worktree, see CLAUDE.md). `npm run wasm --prefix
app` rebuilds only the WASM. `npm run check --prefix app` type-checks.

## checks/: standing checks

Run from the maps root (`node trainworld/checks/<file>`).

- `smoke.mjs [url]`: the pre-commit smoke check (CLAUDE.md "Before a commit").
- `dev-nowatch.mjs <port>`: the app's Vite with watching and HMR off, for an agent's own loop.
- Node, no browser: `t133-picking-check` (picking order), `t144-check` (station geometry, line
  widths), `t145-check` (the route preview contract), `t146-check` (complexes), `t125-frame-check`
  (world frames against PROJ), `t116-save-check` (save format), `t285-merge-check` (needs
  `--experimental-strip-types`), `t066-wasm-check` (demand WASM contract), `t066-worker-check` and
  `t149-worker-check` (the real demand worker in Node, a few minutes each), `t325-bare-check` (a
  zone opened bare against the same zone solved, bubbles and bubble clicks, to the bit; `ZONE=tor`
  for another), `t174-market-check`
  (the regional market's derived block).
- Browser and timing (Playwright, `PLAYWRIGHT_MODULE`): `startup-timing.mjs` with
  `static-serve.mjs` (startup marks on a scratch production build), `t303-measure.mjs` (demand
  loading, clicks, pans), `t328-cache-check.mjs` (the demand cache on reload: timing, cached
  against fresh final to the bit, a changed build, pack or network shows none), `t348-lag.mjs`
  (main-thread long tasks and frame gaps while demand loads: startup, pans, a re-solve, a zone
  joining the solve; optional Chrome trace) with `t348-analyze.mjs` (what each long task in a
  trace spent its time on), `t161-complexes.mjs` (the real save's complexes for the native
  comparison), `t125-fingerprint.mjs` (a save's derived geometry, before and after a frame
  change), `t277-browser.mjs` (Toronto; the template for a new city's browser check, not rerun
  since T-277).

## pipeline/: city packs (Python)

Run with `..\venv\Scripts\python.exe` (the maps venv).

- `build_city.py nyc` writes `data/packs/` (downloads cached in `data/raw/`; the zone gravity is
  solved by the sim crate's `pack_gravity` bin through cargo, built in `data/work/target`).
  `read_pack.py nyc` checks a pack, `look_pack.py nyc` draws `../artifacts/trainworld/T-004.png`.
  City configs are in `build_city.py`. `build_city.py --boundary-only [ids]` rewrites only the
  US packs' `.boundary.geojson` from TIGER/Line counties with their water (`tiger_counties.py`,
  simplified topologically once for the whole US into `data/work/tiger_counties_simple.json`, T-352).
- `regions.py nyc` adds each cell's state to a finished pack and re-solves its gravity in place
  (seconds; T-162): the way to re-solve after a decay or border factor change without a rebuild.
- `enrich_all.py phl bos` adds the non-work inputs (everyday sectors, NCES schools, IPEDS
  universities, airports from FAA/BTS/OurAirports, medical and parks from OSM) to finished packs
  in place, about 0.5-2 minutes a city plus Overpass time (T-142).
- `routes.py <ids>` adds routed walking and driving to finished packs (OSM from Geofabrik, cached
  in `data/raw/osm/`; the sim's `pack_walk` bin; 1-8 minutes a pack; rerun after anything that
  changes a pack's zones, `--check` lists stale ones; T-275).
- `osm_water.py` is the water source for cities outside the US (OSM coastline sea plus Geofabrik
  inland water, T-041; `source(extracts)` for a config's `water_source`, `mask <id>` for a
  water-only pack in `data/work/osm_water/`; docs/adding-a-city.md).
- `places.py [ids]` writes each city's OSM place list, `data/packs/<id>.places.json` (towns and
  neighbourhoods inside the boundary, one Overpass request a city cached in `data/raw/`; T-177),
  beside the pack, not in it.
- `frame_fixture.py` writes PROJ's values for the world frame (`app/src/game/frame.fixture.json`),
  which `checks/t125-frame-check.mjs` and the Rust frame test compare with.
- `tier0.py`: the adapter from global sources only (T-287); `canada.py`: Canada's adapters.
- Basemap glyph tiles (`app/public/fonts/`, checked in): `glyphs.py` rebuilds them, after a
  one-off `cargo install build_pbf_glyphs --locked` (on Windows set `CFLAGS` to a libz-sys zlib
  folder first, notes/T-056.md). Only needed to change the font, weights or fallback.

### The Northeast's build order

The runtime needs `build_city.py` outputs for all registered zones: cities `nyc` (with Mercer
since T-276), `phl`, `bos`, `wdc`, `bal`, `pvd`, `nhv`, `hfd`, `spf`; non-city zones `wor`, `wnd`,
`nlc`, `lit` (docs/zones.md). Then in order: `zones.py` (registers built non-city zones),
`outside.py` (the market's outside world, Kontur), `intercity.py` (the market's sites, then the
joint commute gravity `joint_gravity`, which rewrites every zone pack's gravity factors and the
market's commute list, about 12 minutes, then the `derived` block with `market_pack`; `--derive`
does only that, seconds: rerun it after changing `FORMULA` in `intercity.rs`, T-174),
`zones.py --trips`, and last `routes.py <ids>` for any pack whose zones changed (T-275).

Vite serves and copies all registered packs. Zone packs stay separate; demand workers share a
regional pool. Registry market places keep NYC0/PHL1/Mercer2 (no sites since T-276)/BOS3/WDC4 and
append Baltimore5/Providence6/NewHaven7/Hartford8/Springfield9 and the non-city zones. The regional
builder reads this registry.

Toronto (`tor`, T-277) is registered with `market: -1` (outside the regional market) and built by
`canada.py` (`canada.py tor --finish` reruns gravity, purposes and checks from the cached cells).
Its OSM inputs come from the Geofabrik Ontario extract, not Overpass.

Adding a city: the step-by-step checklist is `docs/adding-a-city.md`.
