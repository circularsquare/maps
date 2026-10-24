# Code map

Where things live and how they are built. Read when you need to find a file; the rules are in
`CLAUDE.md`. Keep a line here current when you add, move or split a module.

## sim/: the Rust crate (`trainworld-sim`)

Compiled to WebAssembly with wasm-pack 0.15. `cargo test --manifest-path sim/Cargo.toml` runs its
tests natively.

- `src/track/`: the track model (T-040): geometry, costs, crossings, world frames (`frame.rs`,
  T-125), the network and its edits with undo (`world.rs`; blueprint and construct, T-055),
  run-time profiles, capacity, the boarding account (`boarding.rs`: each train's load and each
  platform's crowd, T-164), each stop's service span for demand (`span.rs`, T-842), saves, local time (`localtime.rs`, T-309), and `wasm.rs` (`TrackApi`,
  the clock worker's API, with the drawing tool's routes and the crossover tool's edit, T-338; its
  exports are split by area into `wasm/money.rs`, `edits.rs`, `draw.rs`, `crossings.rs`, `links.rs`,
  `lines.rs`, `snapshot.rs` and `service.rs`, the shared helpers staying in `wasm.rs`). In the default build.
  `cargo run --release --bin track_bench` times edits on a synthetic 2,000 km network.
  notes/T-040.md. Methods only tests and bins call (`set_line`, `set_edge`, `to_lonlat` and the
  like) sit in plain `impl` blocks, not in the WASM exports; the same holds for `DemandApi`,
  `EverydayApi` and `IntercityApi`. A big file's tests are in its own `<file>/tests.rs`.
- `src/track/trace.rs`: a real railway's line as corners within a tolerance (T-727c), for
  `TrackApi::trace_route` and the real-network snap; `Tracer` runs it in steps (`RouteTrace` in
  WASM, T-754); notes/T-727.md.
- `src/demand/`: the demand kernel, behind feature `demand` (on in the app's build): `api.rs`
  (`DemandApi`, the demand workers' API), `intercity.rs` (the regional market; `intercity/market_trip.rs` its trip explorer query, T-866; on a big cluster its pass searches destinations one at a time,
  `pass_streamed` with `kernel::DestSearch` and `DestAssign`, T-761), `warm.rs` (the
  everyday warm-up demand worker 0 runs at startup, T-173), `days.rs` (the weekend's factors per
  purpose and period, T-260), `routes.rs` (routed walks round water and drives over OSM roads from
  the pack, T-275), `trip.rs` (the trip explorer's query, T-195; its two cells' own trip, T-720, checked with the
  walk bands in `t720.rs`), `lanes.rs` (f32 lanes, wasm
  SIMD or SSE2, for the commute pair loop, T-036). The commute pair loop is `kernel::mode_choice`;
  its small zone pairs are dealt in chunks there (`COMMUTE_CHUNK`, T-874; `DemandApi::set_chunked`).
  Zones of two sizes (T-852: `kernel::coarsen`, `Zones::size`): every zone-pair table goes through
  `kernel::table_layout` and `Gravity::base`/`key` (`PairKey` elsewhere), near pairs through
  `Zones::near`, a square's zone through `Zones::zone_at`, centre distances through `Zones::d2_m`.
  `calib --cells --exact` dumps every figure unrounded, for before/after checks.
  `cargo test --release --features demand nyc_day -- --ignored --nocapture` solves a day on the
  real pack natively. notes/T-026.md.
- `src/bin/`: native tools behind feature `demand`, built in `data/work/target`: `calib`
  (T-509: a weekday on a city's real network with a parameter set, for `pipeline/calib.py`),
  `realsave`, `pack_gravity`, `pack_walk`, `market_pack`, `joint_gravity`, `zone_summary`.
  `demand::network::RealNet` reads a real network's JSON for `calib` and the real-network tests
  (`route_nodes` gives the fare km per route node, the rule of the app's `hopKm`);
  `City::subset` and `inner` cut the synthetic city down for tests.

## app/: the browser game

Vite 8 + TypeScript 7 + MapLibre GL 6.13 (ESM only: `import * as maplibregl`, worker URL passed
via `setWorkerUrl`, no `map.transform`) + Preact 11 with signals (TSX, no Vite plugin).
`src/main.ts` wires everything; `src/wasm/` is generated: `track/` for the clock worker, `demand/`
for the demand workers (two modules from the one crate, notes/T-051.md).

### src/game/: what the UI reads

- `types.ts` (the save's inputs, worker results, `WorldView`, `Selection`), `world.ts` (the
  snapshot signal and the edit functions), `ui.ts` (tab, selection, display settings, theme).
- Time: `clock.ts`, `localTime.ts` (each region's local time and the local time in view, T-309),
  `sun.ts` (the sun's height for the top bar's sky, T-861; the calendar is in `days.ts`),
  `holidays.ts` (public holidays by country from `holidays.json`, made by `pipeline/holidays_json.py`, T-862;
  a line's and the view's country, T-872; one entry for a holiday several countries keep, T-1098), `holidayCalendar.ts` (the plain lookups the clock worker shares:
  a holiday's name, `kindOfIn` a country's kind of each day, T-872),
  `periods.ts` (the five periods), `days.ts` (weekday and weekend, the shown day, T-260).
- Demand and riders: `demand.ts` (results: mode split, riders), `boarding.ts` (the boarding
  account's hour on the main thread: riders packed for the clock worker, platform crowds, T-164),
  `complexLive.ts` (a complex's platform crowds and trains now, T-246), `purposes.ts` (the trip
  purposes as shown, T-192), `trip.ts` (the trip explorer's state).
- Places and frames: `coords.ts` (world metres to lng/lat), `rings.ts` (ring box, point-in-ring and near-ring tests for the real-network snap and its router), `frame.ts` (the world frames, and the earth's radius and metres per degree) and `registry.json` (the
  one city registry, read by `cities.ts`, T-125; since T-776 the bundled index
  only), `market-rows.json` (each zone's world market row, fetched by `workers/demandClient.ts` beside
  the first market pack and sent with it to the solving workers, `cities.ts` `marketWorld`), `places.ts` (station and
  compass names for the inspectors), `placeList.ts` (each city's OSM place list, fetched per zone by
  search or station naming, the view's zones first, T-177, T-065, T-400), `boundaries.ts` (each city's boundary file, fetched
  once, T-172), `zoneNames.ts` (zones' shown names and search aliases from the hand-edited
  `zone-names.json` and the generated `zone-aliases.json`, `pipeline/zone_aliases.py`; T-825),
  `zone-successors.json` (each removed zone id's successor, for old saves, `pipeline/zone_successors.py`; T-1053),
  `realms.ts` and the hand-edited `realms.json` (the country-sized realms: each zone's, which are shown, each realm's circle point from the registry, and `hiddenRealmAt` for a click on its land, T-975, T-1006), `search.ts` (the search box's matching, `cityMatch` for zones by name or alias, T-148), `walks.ts` (walking minutes to the
  closest stations on the demand model's walking barrier, T-312).
- Stations and lines: `complexes.ts` (station complexes from platform geometry and the player's
  links, T-146), `complexView.ts` (a click selecting the complex, a station opened alone, the
  zoomed-out dot groups; T-217, T-249), `links.ts` (the player's links, T-197), `lineActions.ts`
  (new and copied lines, Edit stops, construct blueprints: the N, L and C keys, T-670), `remove.ts` (removing things, with the question asked before removing
  anything constructed, T-096).
- `realGraph.ts`: a region's real-network routing graph (`packs/realgraph/<cc>.rgr`, fetched once
  and decoded), for snapping track to the real network (T-727); read by `workers/realRoute.worker.ts`.
- Account (T-972, T-973): `account.ts` (the optional anita.garden account, noritetsu's, and the
  leaderboards: which server, the session, each slot's game id and its score sends, the boards;
  shonei-server's `trainworld.go`); its panels `ui/AccountPanel.tsx` and `ui/Leaderboard.tsx`.
- Other: `money.ts` (cash and ledger from the worker, fare income from riders, T-028),
  `fareCurve.ts` (the fare curve through the player's two nodes: fare at a length, old saves' base and rate as nodes, T-883, T-1112), `fareUndo.ts` (fare changes' undo and redo while they are the last edits, handing over to the sim's, T-1130),
  `moneyTiming.ts` (each game hour's fares, rides, points and running cost from the period tables, T-736),
  `regionMoney.ts` (each line's and each construction's split between regions, the region ledgers, T-531), `points.ts`
  (each line's points per hour by region, points a day, each region's past 24 hours' points, the star's mark, T-684, T-685, T-694), `issues.ts`
  (the track model's refusals in plain words), `openedZones.ts` (the zones the game has started in, reached or crossed with track, kept for the save, T-729, T-801), `trackZones.ts` (the zones built and blueprint track crosses, by coarse outline, from each network state, T-1047), `loading.ts` (what is loading, the startup gate and
  the clock hold, T-149, T-172; whether the start menu is up, T-677), `palette.ts`, `shades.ts` over `shadeTable.ts` (the light and dark themes' map
  colours: basemap, line casing, map ink, blueprint off, outside shade, thumbnails; the table has no
  page behind it so the clock worker reads it, T-039, T-543), `format.ts`.
- The `WorldView` is built from the clock worker's states in `workers/clockClient.ts`.

### src/map/: the map and what is drawn on it

- `mapView.ts` (the MapLibre map; its scroll zoom told wheel gaps by event time, T-675; the flat map or globe setting, T-033), `basemap.ts` (the Positron style: warm paper repaint, three road weights, light or dark,
  labels in Zen Maru Gothic from `public/fonts/`), `realNetwork.ts` (the Real network reference layer: noritetsu's
  PMTiles per region, read from `../noritetsu/` or R2; vite.config.ts proxies `/noritetsu` to 8800 in dev, T-682),
  `overlay.ts` (the overlay canvas and frame loop,
  notes/T-015.md), `renderer.ts` (WebGL2 track, stations, trains; its shaders and their shared constants are in `rendererShaders.ts`, the platform crowds' sums in `platformCrowds.ts`), `gl.ts` (shader compile, uniform lookup, zoom interpolation, the GPU timer, canvas sizing, and the camera: `Camera` for the flat map or the globe, `PROJECT_GLSL` and `setCamera` for the shaders, `mercToCss` and `clipToCss` to CSS px, `behindHorizon`; T-033; shared by the renderer, `demandOverlay.ts`, `gradeMarks.ts`, `transferBubbles.ts`, `shadeLayer.ts`, `overlay.ts` and the 2D overlays), `network.ts` (the renderer's
  buffers, built in the clock worker from per-edge, line and station pieces kept between edits,
  T-496), `geo.ts` (the draw anchor, T-307; `pxPerUnitAt` and `unitPerPxAt` for the zoom's scale), `anchorVersion.ts` (its version as a signal, T-317),
  `live.ts` (the live map objects for UI code, set by main.ts; `hiddenOnGlobe` for DOM marks, T-033), `synth.ts` (stress network),
  `markData.ts` and `gradeMarks.ts` (the crossing, junction, ramp and portal marks: their data, and
  their draw pass, hit test and tag; T-629, T-655), `stationLife.ts` (from zoom 14, arrivals found
  by a timer of eight samples a second, feeding `stationSound.ts`, the WebAudio chime and pops, and
  `faces.ts`, the rising faces on a 2D canvas of its own; T-686), `transferBubbles.ts` (riders changing trains walking between stations as bubbles, drawn by the renderer after the stations, T-1124).
- Stations: `stationGeometry.ts` (platform shapes along the track for drawing, picking and the
  ghost, T-189, T-254), `stationNames.ts` (a station's name suggestions: cross street,
  neighbourhood, city or town, T-534), `stationLabels.ts` (names
  keep their elements across edits unless their text changes, T-496), `capacityMarkers.ts`, `labelLayout.ts` (shared spacing
  for names and tags: `BoxGrid`, `tagLines`, `tipBeside`, `TAG_OFFSET_PX`, `screenBounds`).
- Tools: `realSnap.ts` (the track tool's snap to the real network: ring, route ghost and tag, the
  second click's route, T-727d; only where a graph covers the click, `covers`, T-913), `pick.ts`, `cellIndex.ts` and `snapIndex.ts` (grids built in idle time that narrow a pick
  and the build tools' snap to what is near the pointer, T-724), `tools.ts` (drawing track, placing stations, the line tool), `stationSnap.ts` (which overlapping track a station goes on, the end snap, T-671), `edgeEdit.ts`
  (dragging blueprint track's PIs, T-064, T-092), `nodeDrag.ts` (dragging blueprint nodes, T-093), `endMarkers.ts` (loose-end and draggable-point markers, T-416), `curveCircle.ts` (the circle of the segment being drawn, T-418),
  `blueprintBox.ts` (the Select tool's drag box, T-333, and dragging its pick, T-475; its state is `game/blueprintPick.ts`),
  `stopMarkers.ts` (the toggle markers of stop editing, T-233), `reversalMarks.ts` (the icon over
  stations where a line reverses partway, T-797).
- Trains and trips: `trainFollow.ts` (a selected train followed onto its next trip, T-245),
  `tripView.ts` (the trip explorer's route, T-195; its panel is `ui/inspector/TripInspector.tsx`; between zones `MarketTrip.tsx`, T-866; between two groups of bubbles, the Shift+drag box and its samples in `commuterView.ts` `farTrip`, the average in `game/trip.ts` `averageTrips`, T-865; place ends and far groups between zones: `game/trip.ts` `mergeMarketPeriods`, `map/intercityView.ts` `placeTrip` and `interBox`, `IntercityApi::market_trip_between`, T-885).
- Demand views (T-084, T-097): `demandViews.ts` (wiring, a station's riders for its inspector), `demandLayers.ts`
  (line width, station use circles, train fill; the figures from `game/stationUse.ts`, T-482), `commuterView.ts` and `bubbles.ts` (commuter bubbles and
  bubble selections; the far end grouped into at least 10 trips a bubble in `game/farChunks.ts`, T-909), `intercityView.ts` (longer trips of a selection or station, one bubble per
  far zone, T-118, T-120; one per city zoomed far out, T-186), `demandOverlay.ts` (their canvas, over the network's),
  `demandHover.ts` (the count tag by the pointer), `modeBar.ts` (the tag's mode bar, T-494, and `localModes`, an open zone's local legs by mode, T-576), `airportMarks.ts` (the plane badge on an
  airport and its hover, T-448, T-674; paths shared with the icons in `glyphs.ts`).
- Tags and outlines: `trackHover.ts` and `networkTips.ts` (the hover tags on track, trains and
  stations, T-157, T-194), `hoverRest.ts` (the list of what else is under a resting pointer,
  T-732), `hovered.ts` (what is lit as hovered, from the tags and the list, read by both renderers, T-730), `trackTagRows.ts` (the elevation and speed rows with icons, shared with
  the track tool's tag, T-526), `cityBoundaries.ts` and `boundaryGeometry.ts` (the metro outlines, one
  layer for the zones near the view, exact ones fetched only above `ZONE_LOAD_ZOOM`, and the outside
  shade, T-131, T-139, T-400, T-871), `coarseOutlines.ts` (reads
  `packs/outlines.json`, which `pipeline/zone_outlines.py` writes: every registered zone's coarse
  outline, read once for the outlines and the realms' shade, T-800; `inOutline` is the point-in-outline test over its bands of latitude, for realms and the track's zones), `zoneLoading.ts` (zones still loading drawn grey with a loading
  bar, T-343), `realmOverlay.ts` (the realms not shown: one world shade with the shown zones cut out
  by `boundaryGeometry.ts`'s `ShadeCut` as triangles in `workers/shade.worker.ts` (T-1026), drawn by
  the custom layer `shadeLayer.ts` (T-1025), and a button each, T-975, T-1006), `zoneFocus.ts` (after each move: the focused and viewed zone, the nearby zones, and
  `zonesLoadForView`, the zoom test of `game/zones.ts` `ZONE_LOAD_ZOOM` that decides whether zones
  load for viewing, T-276, T-318, T-871).

### src/ui/: Preact components

`Dock.tsx` (the dock, its icon tabs and resizing, T-252); `AccountPanel.tsx` (the account panel,
the person icon, and `startAccountUi`, which mounts both account panels in a root of their
own so the start menu opens them too; styles in `account.css`, T-972); `Leaderboard.tsx` (the
leaderboards and the All cities row's trophy, T-973); `HelpPanel.tsx` (the "?" button's list of controls no button shows, T-908); `BuildTab.tsx`, `LinesTab.tsx`, `StationsTab.tsx`, `MoneyTab.tsx` (the tabs; the Cities tab is `Cities.tsx` mounted in `Dock.tsx`); `listParts.tsx` (the scope toggle, the Near column and the name order the Lines and Stations tabs share); `toolKeys.ts` (the build tools, the number and letter keys and `pickTool`); `Modal.tsx` (the native modal dialog behind the saves, account and leaderboard panels and the confirmation question, with `useModalDialog`); `useClockQuery.ts` (`useClockQuery`, a panel's question to the clock worker, and `useLatestAnswer`);
`MoneyHistory.tsx` (the Money tab's history charts, T-447); `FareGraph.tsx` (the fare curve graph, linear and zoomable, its two nodes dragged both ways, an arrow for one past the plot, and a row of steppers per node, T-864, T-883, T-1112); `Cities.tsx` (the Cities list, T-550, T-749); `SettingsTab.tsx`; `SavesPanel.tsx` (saved games, T-251; `SlotThumb`, a save's picture with each panel's city named over it, T-894); `bullet.ts` (`bulletText`, `bulletNode`: a line bullet as one SVG, its circle and its symbol placed from the canvas-measured ink box with the baseline on a whole device pixel, per font, symbol, size and scale, T-705, T-745; the sizes in it match `.chip` in style.css; used by `Chip` and `lineBullet`); `widgets.tsx` (chips, steppers,
checkboxes and the inline SVG icons); `ColourPicker.tsx` (the line colour picker: preview while dragging, one commit on release, T-672); `TopBar.tsx`; `SkyBanner.tsx` (the clock's sky and demand band, a canvas, T-861); `StatusTag.tsx` (the demand status tag on the map's top row); `ToolNotes.tsx` (tool hints, the notice and the confirmation
dialog); `DemandSelector.tsx` (None/Homes/Destinations and the purpose filter); `Blueprint.tsx`
(Build's quote and Construct, and the Select tool's selection bar on the map, T-353); `Debug.tsx` (the `?debug=1` readout); `inspector/` (one inspector
per selection kind: `StationInspector.tsx`, `TrainInspector.tsx`, `TrackInspector.tsx` and `JunctionInspector.tsx`; `LineInspector.tsx`; the trip panels share `tripText.ts` and `RouteParts.tsx`;
`ComplexMap.tsx` the complex map; `TrainStops.tsx` a train's next stops and who gets off, T-868; `NameEdit.tsx` a station's or line's name edited in place, T-534); `LoadingScreen.tsx` (the clock hold at startup, the loading card,
the screen when demand cannot load, T-149, T-172, T-180); `StartMenu.tsx` (the start menu and the URL switches that skip it, T-677); `NewGame.tsx` (the new game sheet: mode, start city, current location, search, real network; `START_CITIES`, `REAL_NETWORKS`, `zoneForLocation`, T-728, T-731, T-1132); `Spinner.tsx` (the small spinner, T-647, and the bunny's train, T-677); `Search.tsx` (T-148); `DemandStopped.tsx` (the alert when demand stops after startup, T-757); `keys.ts` (which keys a focused control keeps, T-183); `locate.ts` (the camera to a station or a line: search picks, the tabs' double click, the inspectors' locate button, T-569; `viewStart`, the camera at a new game's start);
`tip.ts` (the hover tag for every panel and control, from `data-tip`, never `title`, T-210);
`blurHold.ts` (every hover kept up through a window blur until the pointer moves, T-483);
`style.css` (theme tokens).

### src/workers/

`realRoute.worker.ts` (the real-network router: graphs joined at borders, A* with the turn limit
and families, then the fit in steps on a track model of its own, dropped for a newer request, T-727d, T-754; the outlines of the regions with a graph sent to the page, T-913); `protocol.ts` (edit ops, clock messages, the network snapshot for demand), `clock.worker.ts` (owns
the network: `TrackApi`, notes/T-025.md; money and autosave too; `slotSummary`, a save's picture
with a panel per city, T-894), `saveFile.ts` (the save format
and IndexedDB, T-029; slots' backups, T-744; removed zone ids taken as their successors on load, T-1053), `clockClient.ts`, `demand.worker.ts`, `demandClient.ts`, `demandCache.ts`
(the last final demand result kept beside the save, T-328),
`demandProtocol.ts`, `cacheWrite.worker.ts` (the demand cache's IndexedDB write, off the main thread, T-789), `demandClusters.ts` (the network split into clusters that share no rider,
each a sub-network, T-326; `reachView`, a city's own network inside its cluster, T-756; `marketReaches`, which stations the market's riders use, T-396), `demandTest.ts`, `cityReach.ts` (which city packs a network's stations can
reach, T-149; the footprints fetched on demand by registry box, T-404; `distanceTo` through a grid of each outline's edges, T-889), `commuters.worker.ts` and `commutersProtocol.ts` (the commuter bubbles' sums,
T-097), `selectionMerge.ts` (a bubble selection's everyday far end added up over the workers,
T-285), `shade.worker.ts` (the realm shade's cut, T-1026).

### URL switches

`?debug=1` (readout; `window.tw` probes incl. `lagStart/lagStop`, `checkPrecision`; `window.twDemand`
with demand timings and `client.qlog` per query; `window.twViews` with the demand views'),
`?paused=1`, `?new=1` (a new game; it owns no slot and never writes one
until a copy is saved, T-320, so it is safe in a browser profile holding real games),
`?continue=1` (the game opened last; checks that reopen a stored game use it), each of these and
the two below skipping the start menu (T-677),
`?acct=local|live|off` (which account server a localhost page uses, remembered; none by default,
T-972), `?synth=1&lines=500&trains=10000`, `?demandTest=1` (demand on T-005's hand-made network; both also
never autosave), `?backupMs=N` (wall ms between a slot's backups, T-744), `?demandWorkers=N`, `?perfOff=...` / `?perfTry=...` (CPU A/B switches,
`src/perfFlags.ts`, notes/T-045.md).

### Build

`npm run build --prefix app` = wasm-pack twice (Cargo profile `wasm`, `+simd128` from
`sim/.cargo/config.toml`), type check, `app/dist/`
(city packs, real networks and real-network graphs copied into `dist/packs/` by `vite.config.ts`), which serve.py serves at
`localhost:8800/trainworld/` (from the snapshot worktree, see CLAUDE.md). `npm run wasm --prefix
app` rebuilds only the WASM. `npm run check --prefix app` type-checks.

Every fetch of a `packs/` file goes through `game/dataBase.ts` (`dataUrl`): `packs/` beside the page,
or the R2 prefix a published index.html names in its `tw-data` meta. `tools/deploy.py` publishes a
build: `packs/` to R2 `trainworld/<version>/`, the rest of dist to the website repo (T-924,
notes/T-924.md has the commands). During a batch, publish a build on the last freeze
(`pipeline/freeze.py build`, then `deploy.py --dist` that build; T-928), never one that copied
data/packs mid-batch.

## checks/: standing checks

Run from the maps root (`node trainworld/checks/<file>`).

- `smoke.mjs [url]`: the pre-commit smoke check (CLAUDE.md "Before a commit").
- `lib.mjs`: what the checks share, imported by every check that drives a browser or the worker
  in Node: `playwright()` (finds Playwright), `launchChrome` (headless Chrome on a throwaway
  profile, deleted by its `close()`), `collectErrors`, `importSave`, `REAL_SAVE`, `demandFinal` (the
  wait for a demand final), `buildLine` and `acrossLine` (a built test line), `startVite`, and for
  Node `loadApp` (an app module bundled by Vite), `bundleDemandWorker` with `startWorker` (the real
  demand worker), `workerPack`, `marketPacks` and `network` (two test lines in the New York frame).
  A new check imports from it rather than copying a block.
- `dev-nowatch.mjs <port>`: the app's Vite with watching and HMR off, for an agent's own loop.
- `agent-tree.mjs new|list|land|remove`: agent worktrees in `maps/.tw-trees/<name>` on branch
  `tw/<name>` (CLAUDE.md "Agent worktrees"); `land` squashes a tree's trainworld/ changes onto master.
- `every-pack-opens.mjs [url] [zone ...]`: every registered zone, in a fresh new game, opens for
  the view and solves a three-station line across its centre without a demand error (T-484; about
  16 s a zone, so 7 to 8 hours for all 1,700; name the zones to run).
- `t1053-load-check.mjs [port]`: on its own Vite with a scratch registry lacking wnd, nlc and lit,
  a new game's save and the real save naming them load with Hartford in their place (T-1053, about
  a minute).
- Node, no browser: `t133-picking-check` (picking order), `t671-station-snap-check` (the station tool's choice among overlapping tracks and the layout's guess; given a dev server URL it also runs `t671-ghost-hover`, the ghost's first frame against the worker on the real save in headless Chrome), `t144-check` (station geometry, line
  widths), `t145-check` (the route preview contract), `t146-check` (complexes), `t1053-zone-ids` (removed zones have successors; `zoneNow` and `currentZones`; in smoke), `t125-frame-check`
  (world frames against PROJ), `t116-save-check` (save format), `t285-merge-check` (needs
  `--experimental-strip-types`), `t066-wasm-check` (demand WASM contract), `t066-worker-check` and
  `t149-worker-check` (the real demand worker in Node over New York, Philadelphia and Boston, a few
  minutes each; every zone at once is more than a worker's WASM holds), `t325-bare-check` (a
  zone opened bare against the same zone solved, bubbles and bubble clicks, to the bit; `ZONE=tor`
  for another), `t377-world` (the real save, then lines in Toronto and London:
  startup, edits per cluster, memory, London's longer trips by place), `t409-worker-check` (two networks with no track between
  them solve as two clusters, each as it would alone, to the bit; about a minute), `t301-worker-check`
  (packs and the market closed and reopened, also mid-solve and bare: figures to the bit),
  `t761-plain-net <label> <out.json>` (a `t756-reach.mjs capture` as plain JSON arrays for the native
  `t761_market_bench`, sim/src/demand/intercity/tests.rs).
- Python: `token_usage.py <transcript.jsonl> ...` (an agent's token use from its transcript,
  CLAUDE.md "Token counts"), `world_field_check.py` (the world field's country totals and fixed
  links, T-375; from trainworld/), and in pipeline/ `test_education_sources.py` and
  `test_university_sources.py` (`python -m unittest test_university_sources test_education_sources`
  from pipeline/, no network).
- Browser and timing (Playwright, `PLAYWRIGHT_MODULE`). A `page.evaluate` that moves the map returns
  nothing (`() => void window.tw.map.jumpTo(...)`): MapLibre's `jumpTo` returns the map, and Playwright
  serialising it is 1-2 s of main thread that reads as lag (T-1052). `startup-timing.mjs` with
  `static-serve.mjs` (startup marks on a scratch production build; `MENU=1` through the start
  menu, timed from the pick), `t326-clusters.mjs` (the real save plus a Toronto line: edits in each cluster
  timed to their final, every final figure hashed against `?perfOff=clusters`), `t409-split.mjs` (a
  Boston and a Philadelphia line, in a new game or on the real save, edits timed, against
  `?perfOff=split`), `t301-pan.mjs` (a long pan, Washington to Toronto and back, bubbles on: worker
  memory, open packs, reopening times; `SAVE=1` on the real save), `t801-no-grey.mjs` (no grey
  cities: a map start in a fresh profile turning Homes on, its zoomed-out bubbles, a click selecting
  one; a new New York game's Philadelphia an ordinary bubble zoomed out, loaded by a pan with no
  market pack; screenshots `t801-*.png`), `t800-outlines.mjs` (a game started on the map, then a jump to Europe, shows
  every zone's outline at once; `NOFILE=1` blocks the coarse file, `DELAY=<ms>` slows the boundary
  files; screenshots `t800-*.png`), `t124-periods.mjs`
  (the real save: a one-level train-count edit and an every-level edit timed to their final, hashed
  against `?perfOff=periods`), `t488-money-days.mjs` (the real save's clock run fast and pulled back
  across midnight: money days and hours one each, in order, none empty; uses the app's Vite, port
  6135 by default), `t715-load-money.mjs` (the real save opened from the saves panel while a later
  copy is open, twice: cash loads as saved and moves, no ledger day after the clock; a save with
  ledger days after its clock gets their running cost back; port 6165 by default),
  `t975-realms.mjs` (first the location button refused, in Lyon and in Sydney, and a map start showing every realm until its first station; then realms in a fresh profile: no bubbles behind the start menu, a Tokyo start shows Japan only with the shade cut around its zones, a click shows France, a reload hides it again, stations in two realms show both; a click on a hidden realm shows it, a drag does not, no circles at the world view; the zoomed-out frame and a pan across France and Germany measured; port 6165 by default),
  `t1071-train-count.mjs` (the real save's busiest line, its train count raised and lowered four times with the clock running: every train of every line placed each frame from the renderer's own buffers may move no further than top speed allows and come or go only at a terminal, and where new buffers arrive the old and new draw the same trains in the same places; `SPEED` 0 to 2 or -1 paused, `OP=headway`, `KEEP_FRAMES=1`; about 2 minutes; Rust side `sim/src/track/transition_tests.rs`),
  `t1025-shade.mjs` (Node, no browser: the realm shade's triangles for each realm alone and several groups up to all; no point of a shown zone shaded, none shaded twice, open land and sea shaded once; `quick` for the groups only, about 6 s, all of it under half a minute),
  `t728-newgame.mjs` (the new game sheet in a fresh profile: the search, a sandbox in Tokyo from
  its real network, a normal London start, a map start, the real New York save turned sandbox from
  Settings; slots left keep their bytes; screenshots t728-*.png; port 6150 by default),
  `t727-snap.mjs` (the snap to the real network in a new game: Tokyo to Shin-Osaka and Penn
  Station to Emeryville by two clicks each, one undo each, Tokyo to London refused, a click off
  real track an ordinary segment; router, fit and preview times and the longest main-thread task;
  t727-snap.json, t727d-*.png; about 2 minutes, port 6170 by default),
  `t743-slots.mjs` (through the saves panel: Load a file, New game and a backup's Restore as a copy
  each open a new slot and leave every other slot's bytes as they were, the real save and
  artifacts/trainworld/recovered-day26.save both; two backups rotate at `?backupMs=4000`; a deleted
  slot's backups go; port 6150 by default), `t894-previews.mjs` (save pictures: a game in New York,
  Boston and Philadelphia and the real save, their panels and size, screenshots of the saves panel
  and start menu to artifacts/trainworld/previews/; `SHADE=dark`; port 6120 by default),
  `t883-fare-curve.mjs` (the real save: the default fare's 12,796,392 rail trips, the same line through other nodes, three nodes kept as first and last, a doubled rate losing rail trips, a line lifted at its first node when it would be below $0 at 0 km, $0 at 0 km rising adding rail trips; the Money tab's drag sending once on release, wheel zoom's off-plot arrow and its click; fare graph screenshots), `t736-money-timing.mjs` (Node: a week's hours of `moneyTiming.ts` keep the period tables' fares,
  rides and running costs, with the largest hour-to-hour change beside the flat booking's),
  `t872-holidays.mjs` (the real save over the next Thanksgiving: that day's boarding account, fares
  and ledger are Saturday's, the days around it weekdays; the day is a weekday in Canada and the UK),
  `t745-bullets.mjs` (the real save's line bullets in the Lines tab, the line inspector and the map
  tags, plus digits, two characters and kana, light and dark, 100% and 90% UI, pixel ratio 1 and 2:
  each symbol's ink centred in its drawn circle within 0.6 device px, measured on the screenshot;
  port 6160 by default, about four minutes),
  `t486-click-on-top.mjs` (a click on a blueprint point's square over a bubble
  selects the track, beside it the bubble), `t533-draw-order.mjs` (the real save: the shorter line
  on top where lines share ground, a line with no trains under; port 6130 by default),
  `t658-lanes.mjs` (the real save: the overlay at zoom 12 and 15 pixel for pixel the baseline
  build's, 8800 by default; either of double track's two tracks picks the line at 17.5; two
  trains passing each picked on its own track; draw time; screenshots at 15 to 19, light and
  dark; port 6160 by default),
  `t586-trip-agrees.mjs` (New York's real save and a Mumbai test line: a far-end bubble's hover,
  the trip explorer opened from it and the worker's far-end sums agree within 1 point under every
  purpose filter and day; port 6196 by default),
  `t674-airport-modes.mjs` (a new Tokyo game and a line to Haneda: the plane's hover follows the
  worker's airport legs after the solve, and a game day later the destination bubble beside it
  reads the same selected or not; port 6144 by default, about four minutes),
  `t680-new-line-crowd.mjs` (a new game's Mumbai test line: its platforms are empty when its
  trains start and fill no faster than riders arrive; also runs on a production build; port 6145
  by default, about two minutes), `t684-points.mjs` (a new game's two Mumbai lines with a walk
  between them at Dadar: points count each journey once and the loads' passenger-km, a whole
  hour's ledger points match the hour's table, and a city's points are its past 24 hours'; dev
  server only, port 6153 by default, about four minutes),
  `t496-edit-latency.mjs` (the real save: click to drawn
  for placing stations; `MODE=visual` hashes the renderer's buffers after a series of edits against
  an older network.ts),
  `t648-stop-edit.mjs` (the real save: a station built on a line's track, then added as a stop
  with a click on its marker; click to world and to the frame drawn with the new network, and a
  screencast of the click's frames with a flash test, pixels the same before and after that a frame
  between draws otherwise, as a mask and a frame strip),
  `t328-cache-check.mjs` (the demand cache on reload: timing, cached
  against fresh final to the bit, a changed build, pack or network shows none), `t348-lag.mjs`
  (main-thread long tasks and frame gaps while demand loads: startup, pans, a re-solve, a zone
  joining the solve; optional Chrome trace) with `t348-analyze.mjs` (what each long task in a
  trace spent its time on), `t1008-load-lag.mjs` (long tasks, frame gaps and each demand message's
  main-thread ms in the first seconds after a save loads, with the main thread's top functions,
  then the game's figures at demand's final; a save imported into a new game, or `CONTINUE=1` the
  slot reopened with its demand cache; `EDITS=1` the long tasks after a track and a station edit),
  `t1020-click-latency.mjs` (laying track with the mouse: click to drawn, ghost latency per move,
  frame gaps, long tasks and the worker's steps per click, on a new game, the real save, Japan's
  network, Anita's London-Shanghai-London repro, whose far click must be refused at once and whose
  requests after Esc must stay in one city, and T-1039's long pieces and snapped routes zoomed out
  followed by short track in Belfast, with how long the clock worker takes to answer meanwhile),
  `t714-pan.mjs` (frame time while dragging and wheel-zooming the
  real save at zoom 11-16 with the clock running: fails over 12 ms of main thread a frame or 5% of
  frames over 50 ms; options for a throttled CPU, a trace per thread, settings, a tool),
  `t871-europe-pan.mjs` (drags across Europe in a new game: fails if a zoom at or below 7.5 opens
  a pack or fetches a water mask, or zoom 9 opens none; reports long tasks; about 2 minutes),
  `t908-longpress.mjs` (real mouse events on home bubbles: a quick drag pans, a held press then a
  drag boxes bubbles with no pan, the controls panel opens and closes; about a minute),
  `t884-loadbar.mjs` (the loading card's bar through the start menu: a new game, the real save and
  the world save, each slot twice: fails over 25% of a wait above 90%, a bar going back, or a fill
  over 30 points from the share of the wait gone; about 5 minutes),
  `t724-hover.mjs` (what a pointer move costs on the real save with the select and station tools,
  timers and the station name included, then a wheel zoom's frame placers: fails over 8 ms at the
  95th percentile in any view),
  `t737-japan.mjs` (Japan's whole N02 network, 709 lines, from `../artifacts/trainworld/t737-japan.save`:
  load time, demand's first solve, then frames, long tasks and threads' busy time panning and zooming
  across Japan at 240x and 960x; fails over 14 ms of main thread a frame or 5% of frames over 50 ms;
  about 4 minutes, `QUICK=1` 1.5; notes/T-737.md),
  `t756-reach.mjs` (`capture <save> <label>` records a save's demand network and packs from the
  page; `compare <label>` solves it with the real worker in Node, every population on its whole
  cluster and on its reach, and fails when a figure moves past the tolerance; `compare <label>
  chunks` the market and the commutes with every zone pair against pairs dealt in chunks (T-780, T-874); New York about 1.5
  minutes, Japan's seven cities about 15; notes/T-756.md),
  `t675-wheel-lag.mjs` (a mouse-wheel zoom with a main-thread stall as it starts travels as far as
  one without; `RAW=1` takes the fix in map/mapView.ts off and must fail),
  `t1007-panic.mjs` (forced panics in the clock worker, dev server only: an edit that panics is undone,
  an idle or read panic keeps the state, the fourth in a minute leaves the model down; about 2
  minutes), `t161-complexes.mjs` (the real save's complexes for the native
  comparison), `city-browser.mjs <id>` (a tier 0
  city: open, boundary, the top bar's split, a water crossing refused, a test line to a final solve,
  T-364; a row from its table, or from the file in `ROW`, which `pack_gate.py` writes for any pack;
  `ROWS`, a list of rows, checks them all in one sandbox game, `ZONES_PER_PAGE` (8) to a page load,
  T-928; a zone whose wait timed out, the solve's `SOLVE_WAIT_S` 180, tried once more in a fresh page, T-965).
- `pack_gate.py <id ...> [--all] [--browser] [--since snap] [--log id=build.log]` (Python, T-538):
  the ship-or-park gate for a finished pack, pipeline stage 6 (notes/T-535.md). Every check a
  number against `THRESHOLDS` at the top of the file: `read_pack.py`, OSM water points (polylabel on
  the extract's water cache) in the mask and OSM stations (`railway=station|halt`, from
  `data/work/<id>.stations.json`, which `stations <id> --extracts` writes from the extracts'
  `.stations.fgb`; older packs' cached Overpass answers in `data/raw/stations/`; no network, T-770)
  on land, `every_pack_everyday`'s rates (`TW_PACKS`; the runner's binary in `TW_EVERYDAY_EXE`,
  built once a run, T-931, else `cargo test`),
  the browser check on a row it makes (a test line along OSM stations near the centre, a water pair
  from the mask; several packs' rows split over `BROWSER_SESSIONS` (3) browser sessions at once,
  T-928, T-965), other packs unchanged against
  a `snapshot` (each other pack hashed once a run), and the plausibility figures (people
  against GHSL and the FUA, outline, jobs, commute, schools, hospitals, roads, T-460's OSM signal),
  plus a WARN outside the fleet's range (`pack_gate_fleet.json`, from `fleet`). JSON a pack in
  `../artifacts/trainworld/pack_gate/`; one summary line a pack. Thresholds and the 67-pack result:
  notes/T-538.md.
- `t700_walk_banks.py [id ...] [--packs dir]` (Python, T-700): the walk factors split by bank; no
  part of a zone walking to itself at over x5, and Shanghai's Middle Henan Rd to the Bund straight.
- `ceiling/` (T-1056): Anita's ceilings A and B on a scratch copy of the sim crate. `t1056_ceiling.rs` is
  the test module, `run_ceil.py [cities]` runs it per pack (`T1056_WORK` names the scratch folder).

## pipeline/: city packs (Python)

Run with `..\venv\Scripts\python.exe` (the maps venv).

- `build_city.py nyc` writes `data/packs/` (downloads cached in `data/raw/`; the zone gravity is
  solved by the sim crate's `pack_gravity` bin through cargo, built in `data/work/target`).
  `read_pack.py nyc` checks a pack, `look_pack.py nyc` draws `../artifacts/trainworld/T-004.png`.
  City configs are in `build_city.py`. `build_city.py --boundary-only [ids]` rewrites only the
  US packs' `.boundary.geojson` from TIGER/Line counties with their water (`tiger_counties.py`,
  simplified topologically once for the whole US into `data/work/tiger_counties_simple.json`, T-352).
- `curves/`: the gravity-curve fitters, run by hand from the maps root: `t277_fit.py` (Toronto),
  `t289_curve.py` (France), `t290_curve.py` (Germany), `t360_curve.py` (England), `t420_curve.py <id>`
  (Japan), and `t314_loo.py` (the leave-one-out test that writes `fitted_curves.json`; its US
  setup reads LODES OD onto a pack's zones with `t162_fit.py`).
- `simbin.py` runs or builds the sim crate's binaries for every pipeline step (cargo on PATH, the
  `data/work/target` dir, `TW_SIM_DIR` and `TW_CARGO_TARGET_DIR`); `packio.py` writes packs (the
  .bin and .json through .part files, `replace_cell_arrays`).
- `realms.py [--registry F] [--out F] [--dry]` writes each registry row's `realm` and `data/packs/realms.geojson`, the land of the realms in `app/src/game/realms.json` (T-975, about 1.5 minutes; run by the join).
- `regions.py`: each cell's state, as the build writes it (T-162). To re-solve a finished pack's
  gravity after a decay or border factor change without a rebuild, use `pack_gravity --keep-zones`.
- `enrich_all.py phl bos` adds the non-work inputs (everyday sectors, NCES schools, IPEDS
  universities, airports from FAA/BTS/OurAirports, medical and parks from OSM) to finished packs
  in place, about 0.5-2 minutes a city plus Overpass time (T-142). Wikidata's airport patronage
  answers are kept in `data/work/wikidata_boardings.json` (T-1083; delete it to ask again).
- `hospital_beds.py pool` writes `pipeline/hospital_beds.json`, the pooled hospital-size numbers
  and World Bank beds per 1,000 that `enrich_medical.py` reads (T-387); `check nyc` is the
  held-out check, `survey` every pack's beds under the rule. Rerun `pool` before re-enriching
  if packs with well-tagged beds change.
- `coarse_zones.py <ids> [--under 1000]` re-solves a res-9 pack's gravity with 4 km blocks where
  every 2 km zone of a block makes under that many trip ends a weekday (T-852, `pack_gravity
  --coarse`, `kernel::coarsen`); before routes and the merge (`world_run.py --coarse`, which also
  passes `build_city.py --coarse-next`: the build's own 2 km solve stops after one pass, since only
  its zones are read before this re-solve, T-1083). `read_pack.check_gravity` reads the decay from
  a table by square offset (T-1083).
- `pack_gravity <id>.json <id>.json --keep-zones` (the sim's bin, `cargo run --release --features
  demand --bin pack_gravity --`) re-solves a finished pack's own gravity on its zones as they are
  (blocks, merged cells and routes kept) with today's rules, its own curve and border factor: for
  a pack that joins nothing when the balance's rules change (T-1091's flat near end).
- `routes.py <ids>` adds routed walking and driving to finished packs (OSM from Geofabrik, cached
  in `data/raw/osm/`; the sim's `pack_walk` bin; 1-8 minutes a pack; rerun after anything that
  changes a pack's zones, `--check` lists stale ones; T-275).
- `merge_cells.py <ids> | --all` is every pack's last step (T-044): sparse res-9 cells into their
  res-8 parent within a zone, touching sparse children of busier parents together (T-466), and
  sparse cells still alone into a touching merged cell across the parent's edge (T-510), the
  res-9 pack kept in `data/work/res9/`; `--undo` puts it back
  before any step that rewrites a pack (enrichments, routes, `intercity.py`), `--stats` only counts;
  cells still under `FLOOR` (50) trip ends join their zone part's nearest busier cell (T-850b);
  `--stats --floor N` counts another floor.
- `nocar.py <ids> | --all` (not read by demand since T-854) adds `nocar` (people in households with no car, per cell; T-511) to
  the res-9 copy and the merged pack, append-only; rerun it after anything that rebuilds a pack
  (its sources: docs/data-register.md "Car ownership"); `--plan` prints every planned country's
  share and source, writing nothing.
- `everyday_rates.py <ids> | --all` writes the `everyday` header block (the country's survey
  outing rates by purpose, T-509 stage 4; table `SURVEYS`, worldwide median otherwise; the
  distance scales are the sim's one worldwide set, `everyday::DECAY`, T-637) into the merged pack and its res-9 copy; rerun it after anything that rebuilds a pack
  header. `show` prints the table.
- `osm_water.py` is the water source for cities outside the US (OSM coastline sea plus Geofabrik
  inland water, T-041; `source(extracts)` for a config's `water_source`, `mask <id>` for a
  water-only pack in `data/work/osm_water/`; docs/adding-a-city.md). `sea` reads the coastline
  polygons from `data/raw/osmdata/water-polygons-split-4326.fgb`, which `sea_fgb` makes from the zip
  when missing or older than it (T-963).
- `osm_features.py [extracts]` reads a Geofabrik extract's point and area features once into
  `data/raw/osm/<extract>.points.fgb`, `.stations.fgb` (the gate's, T-770) and `.areas.fgb`; `read()` gives tier0.py, canada.py and
  places.py the rows a GDAL read of the .osm.pbf would (T-363).
- `places.py [ids]` writes each city's OSM place list, `data/packs/<id>.places.json` (towns and
  neighbourhoods inside the boundary; from the extract's cached features for a tier 0 city, else
  one Overpass request a city cached in `data/raw/`; T-177), beside the pack, not in it.
- `world_field_build.py` writes the world field for the long-trip market (T-362, T-375): Kontur
  at H3 r5 with country, GDP per head and landmass (joined by Wikidata fixed links) into
  `data/work/world_field.*`, a pipeline input never shipped, about 6 minutes;
  `world_field.py` reads it (`load().around(lon, lat)`: r5 within 300 km, r4 to 1,000, r3
  beyond; `locate()`); `checks/world_field_check.py` checks it (country totals, fixed links).
- `longhaul_fit.py [--write]` fits the long-trip tail and border factor on the world field (T-376,
  about 6 minutes) and writes `pipeline/longhaul.json`, the parameters stage 3 reads.
- `frame_fixture.py` writes PROJ's values for the world frame (`app/src/game/frame.fixture.json`),
  which `checks/t125-frame-check.mjs` and the Rust frame test compare with.
- `tier0.py`: the adapter from global sources only (T-287), its borrowed curve read from
  `fitted_curves.json` (written by `pipeline/curves/t314_loo.py table`, T-314); `canada.py`: Canada's adapters;
  `england.py`: England's inputs for a tier 0 city (ITL3 outline, Census 2021 district totals;
  T-288; England's census upgrades go here, T-360); `france.py`: France's (départements
  dissolved from communes, legal population by commune, INSEE jobs by commune of work; T-289);
  `japan.py`: Japan's (census meshes, N03, table 4, P29; T-420); `japan_zones.py`: Japan's 38
  non-city zones planned (`japan_zones.json`, which `build_city.py` turns into configs), fitted and
  built as a batch (T-420 stage 3); `usa.py`: the US's (and Puerto Rico's) for planned rows (TIGER
  outlines, 2020 blocks less institutions, LODES, ACS work from home, the US curve and state regions;
  per-state caches in `data/work/usa/`; T-1016).
- `register.py <ids> [--market] [--trips]`: a built batch's registry entries, boxes, market packs
  and local trips in one call (T-441).
- `world_plan.py [check] [--fresh]`: the world plan (T-536), which metros get a pack, in what
  order, with which admin units, written to `pipeline/world_plan.json` (extents in
  `data/work/worldplan/`); `check` adds the eight-pack test and the next-50 picture; `row <pack>
  <out.json>` plans one existing pack's metro as if the pack did not exist (a test build, T-537);
  `metros --countries US` plans a FUA country's metro rows alone, touching no other row (T-1015);
  `zones --countries ES[,PT]` a country's zone rows (T-723).
- `plan_config.py`: a planned row's `CITIES` config, every field from the row or a rule, then its
  country module's `plan_hooks(row)` keys (T-555; `build_city.py` takes every planned row through
  it); `osm_units.py <id>`: its outline from the
  OSM admin relations matching the row's units (T-537, docs/adding-a-city.md 3), cut along the
  existing packs whose outline box meets it (`cut_packs`; the boxes cached in
  `data/work/pack_boxes.json` by file size and time, T-963).
- `world_run.py plan|run|join|status [--rows ..|--countries ..|--limit N] [--jobs N] [--workers W] [--no-join] [--root dir]`: the
  runner (T-540): world plan rows a region at a time through download, build, gate, registry,
  joint gravity, market and clean up, each step its own `python -I` process; status and times in
  each plan row, a report a batch in `../artifacts/trainworld/world_run/` (docs/adding-a-city.md
  "Running the pipeline"). The gate's rates test is built once at a run's start (`build_everyday`,
  T-931). `--root` points every step at a scratch tree (`Redirect`). `RunFault`
  stops a run without parking (`check_sim`, a gate with no result); `quiet_watch` dumps a silent step's stack.
  Several runs at once (T-786): `--no-join` then `join`; row claims, extract, packs, registry and
  plan locks through `shared_lock.py` (re-entrant cross-process file locks, `write_atomic`), which
  register.py, market_zones.py, zones.py, zone_boxes.py and world_plan.py take too. A clean join
  ends with `freeze.py`.
- `tools/aws/` (T-1064, rules and flow in docs/aws.md): long builds on a rented spot machine in a
  friend's AWS account. `launch.py` (spot, one-time, tagged, a hard timer, presigned URLs into
  `userdata/<kind>.sh`), `bucket.py` (our bucket only), `status.py` (what of ours exists; `cost
  <run>`), `teardown.py` (terminate and sweep by our tags, then list), `guard.py` (a PreToolUse hook
  refusing raw changing `aws` calls); `awslib.py` holds the region, tags and checks. A wave (T-1065):
  `bundle.py` (the code tar), `userdata/wave.sh` (check row, go, `world_run.py run --build-only` by
  runner region) with `wavepack.py` (a region's result tar on the machine), `land.py` (results into
  data/ under the plan lock, rows "built").
- `freeze.py [--label L] | list | check [stamp] | build [stamp] [--tree] [--out]` (T-928): the data a
  production build copies, frozen in `data/frozen/<stamp>/` as hard links to data/packs plus copies
  of the four game data files, with a size and time manifest; `build` makes a publishable build of a
  tree's code on a freeze (`TW_PACKS` in vite.config.ts), for `tools/deploy.py --dist`.
- Real networks and calibration (T-007, T-509): `realnet.py net <city>` turns a city's sources
  (GTFS, OSM route relations from `realnet.py fetch-osm <city>`, or a CSV of timetabled trips;
  `CITIES` lists each city's) into a demand test network in `data/work/realnet/`; `realnet.py`
  also has New York's older targets and the save (`geom`, `gamesave`; `ship` copies the
  seven game files a sandbox starts from to `data/packs/real/<id>.save`, T-731); `realnet_n02.py` is
  its N02 adapter, all of Japan's lines (cities `jpn`, `jpn7`: T-737's stress saves). `calib.py
  targets <city>` writes the city's targets (`data/work/calib/<city>/`), `calib.py run <city>
  [--set k=v,..] [--params p.json] [--tag t]` runs the sim's `calib` bin (a weekday, commutes and
  everyday, 3 crowding rounds, about 22 s for New York) and scores it (station entries, operator
  totals, rail share by home area); `calib.py score <run.json>` rescores. notes/T-509.md.
- `realgraph.py <cc ...> [--check] [--out DIR]` (T-727b): the real network the layer draws as a routing
  graph per noritetsu region, plus short undrawn track that fills its gaps (T-816), from noritetsu's
  extract and `ways.json` (read only), into `data/packs/realgraph/<cc>.rgr` (gzip, layout in its
  docstring; 0.01-6.6 MB, 1-32 s a region) and `index.json` (the regions built, which the track
  tool snaps inside, T-913; no regions given rewrites only it). Built for all 118 noritetsu regions
  with drawn track (39.6 MB, about 5 min at 2 cores; T-727e), copied into the build. Rerun after
  noritetsu rebuilds a region.
- Basemap glyph tiles (`app/public/fonts/`, checked in): `glyphs.py` rebuilds them, after a
  one-off `cargo install build_pbf_glyphs --locked` (on Windows set `CFLAGS` to a libz-sys zlib
  folder first, notes/T-056.md). Only needed to change the font, weights or fallback.

### The Northeast's build order

The Northeast's hand-built packs are `build_city.py` configs: cities `nyc` (with Mercer since
T-276), `phl`, `bos`, `wdc`, `bal`, `pvd`, `nhv`, `spf` and the zone `wor` (docs/zones.md); Hartford
(`hfd`, with Windham, New London and Litchfield since T-1113) is a world plan row. Then in order:
`zones.py` (registers built non-city zones and writes every entry's `box` through `zone_boxes.py`,
which also runs alone, T-302), `intercity.py --group auto [ids]` (every group of packs within 50 km
of each other by `joint_gravity --packs`, T-463, T-539, the Northeast's too since T-1014, a group
spanning countries or solve regions a unit at a time with `--fixed` neighbours, T-786; `--groups`
lists them and their solve units),
`market_zones.py [ids]` (each zone's market pack against the world field, then `market_pack`'s
rows into app/src/game/market-rows.json and each zone's `country` into the registry through
`market_rows.py`, which also drops the rows of unregistered zones, T-776; seconds a zone, T-377;
only the rows of zones whose market pack changed, came or went and of places within 250 km of
them, from `data/work/market_rows.inputs.json`, `market_pack --only`; `--rows --full` for every row, T-929),
`zones.py --trips`, `routes.py <ids>` for any pack whose zones changed (T-275), and last
`merge_cells.py --all` (T-044; the chain runs on res-9 packs, so `merge_cells.py <ids> --undo`
first).

Vite serves and copies all registered packs. Zone packs stay separate; demand workers share a
regional pool. Registry market places keep NYC0/PHL1/Mercer2 (no zone since T-276)/BOS3/WDC4,
Baltimore5/Providence6/NewHaven7/Springfield9/Worcester10; 8 and 11-13 (the old Hartford, Windham,
New London, Litchfield) are unused since T-1113. Every pack's `market` is -1 since T-1014.

Toronto (`tor`, T-277) is registered with `market: -1` (outside the regional market) and built by
`canada.py` (`canada.py tor --finish` reruns gravity, purposes and checks from the cached cells).
Its OSM inputs come from the Geofabrik Ontario extract, not Overpass.

Adding a city: the step-by-step checklist is `docs/adding-a-city.md`.
