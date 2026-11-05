# Code map

Where things live and how they are built. Read when you need to find a file; the rules are in
`CLAUDE.md`. Keep a line here current when you add, move or split a module.

## sim/: the Rust crate (`trainworld-sim`)

Compiled to WebAssembly with wasm-pack 0.15. `cargo test --manifest-path sim/Cargo.toml` runs its
tests natively.

- `src/track/`: the track model (T-040): geometry, costs, crossings, world frames (`frame.rs`,
  T-125), the network and its edits with undo (`world.rs`; blueprint and construct, T-055),
  run-time profiles, capacity (passing tracks' overtakes beside the following losses: `overtake.rs`,
  T-2231), the boarding account (`boarding.rs`: each train's load and each
  platform's crowd, T-164), each stop's service span for demand (`span.rs`, T-842), the lines related to a line for the line panel's diagram (`related.rs`, T-1168; `wasm/related.rs` builds its routes), station upgrades (`upgrades.rs`, T-1281), station buildings (`building.rs`: outlines, their members and refusals, T-2217; `wasm/buildings.rs` adds, reshapes, renames and removes them), saves, local time (`localtime.rs`, T-309), and `wasm.rs` (`TrackApi`,
  the clock worker's API, with the drawing tool's routes and the crossover tool's edit, T-338; its
  exports are split by area into `wasm/money.rs`, `edits.rs`, `draw.rs` (routes beside track: `beside.rs`), `crossings.rs`, `links.rs`,
  `lines.rs` (the gap a stop cannot cross: `gap.rs`, T-1896), `snapshot.rs`, `service.rs` and `upgrades.rs`, the shared helpers staying in `wasm.rs`). In the default build.
  `cargo run --release --bin track_bench` times edits on a synthetic 2,000 km network;
  `--bin edit_bench -- <track bytes>` times headway and count edits on a real save's network (T-1283; `TW_DUMP=<file>` writes `lines_info` at load, after the edits and after a save and reload, T-1787; `TW_LINES=<file>` a table of each line's rates, round trips, delays and trains at load, to compare two builds, T-1924);
  `--bin branch_bench -- <track bytes>` times the drawing tool's previews of branches off it (and with `TW_ADD=1` placing them, T-1779).
  Station placements swept along every piece, refusals counted by reason and by the joints under the platform: the ignored tests in `wasm/station_sweep_tests.rs` (a drawn network, and a save's network bytes in `T1964_NET`; T-1964).
  notes/T-040.md. Methods only tests and bins call (`set_line`, `set_edge`, `to_lonlat` and the
  like) sit in plain `impl` blocks, not in the WASM exports; the same holds for `DemandApi`,
  `EverydayApi` and `IntercityApi`. A big file's tests are in its own `<file>/tests.rs`.
- `src/track/bus.rs`: bus lines (T-1464, T-1465): the route (`BusRoute`: points with corner and road bits; `Poly` in road metres), its stops (`plan`: stations met, ends, crossings, about one every 2 km on the 400 m street grid, each with its stretch of route), the trip at the speed of the streets it crosses (`Pace`, `profile`); `wasm/bus.rs` its exports, tests in `wasm/bus_tests.rs`; notes/T-1464.md.
- `src/track/ferry.rs`: ferry lines (T-2193): ports (stations with a berth), the route the app searched (`FerryRoute`: points with a kind and a speed), its check, the boat's speed limits and its trip with acceleration (`limits`, `profile`), its stops as a bus `Plan`; `wasm/ferry.rs` its exports, tests in `ferry/tests.rs` and `wasm/ferry_tests.rs`; Node check `checks/t2193-ferry.mjs`; docs/track-and-service.md 6.8.
- `src/track/streets.rs`: the car's street speed by place (demand's `place_min_per_km`, kept equal by a test), a pack's grid of its zones' speeds (`StreetGrid`, `Streets` in the network, installed with `TrackApi::add_pack_streets`, T-1577), for a bus's speed.
- `src/track/cost.rs`: prices (SPEC 6.4), the water mask types and the terrain classes
  (`ClassGrid`, `Terrain`, `terrain_spans`, T-1270), installed with `TrackApi::add_terrain`.
- `src/track/trace.rs`: a real railway's line as corners within a tolerance (T-727c), for
  `TrackApi::trace_route` and the real-network snap; `Tracer` runs it in steps (`RouteTrace` in
  WASM, T-754); notes/T-727.md.
- The rest of `src/track/`: `net.rs` (the network: nodes, edges, stations and lines as flat arrays by stable id, with everything derived from them), `geom.rs` (alignment geometry: points of intersection with an arc fitted at each, straights between, ramps), `cross.rs` (where track crosses track, on a 1 km grid of edges), `bridge.rs` (a ground-level segment over water becomes a bridge, T-656), `roads.rs` (the road lines ground-level track crosses, fed tile by tile, T-1929), `capacity.rs` (resources, utilisation per demand level, Kingman's delay, holds), `profile.rs` (run-time profiles along a line's path under speed limits), `spacing.rs` (the phase of each line's departures, T-1925), `save.rs` (the network part of a save), `names.rs` (tidies station and line names from a real network, T-1442), `params.rs` (every tuning constant of the track model), `bench.rs` (the synthetic network `bin/track_bench.rs` times). Test modules sit beside their file (`frame_tests.rs`, `merc_tests.rs`, `related_tests.rs`, `transition_tests.rs`, `wasm/*_tests.rs`).
- `src/demand/`: the demand kernel, behind feature `demand` (on in the app's build): `api.rs`
  (`DemandApi`, the demand workers' API), `intercity.rs` (the regional market; `intercity/market_trip.rs` its trip explorer query, T-866; `intercity/drives.rs` the routed drives between places and the ferry, T-279; its pass goes through a window of destination zones,
  `pass_window`, searching destinations as the window needs them on a big cluster, `pass_searched`
  with `kernel::DestSearch` and `DestAssign`, T-761, T-1545; a window of zones' pairs alone, `set_window`
  and `window_loads`, for an edit's local solve, T-1735), `warm.rs` (the
  everyday warm-up demand worker 0 runs at startup, T-173), `days.rs` (Saturday's, Sunday's and a superholiday's away, going-out and coming-back days' factors per
  purpose and period and the lean's sign, T-260, T-262, T-264, T-1373), `routes.rs` (routed walks over water and drives over OSM roads from
  the pack, T-275), `trip.rs` (the trip explorer's query, T-195; its two cells' own trip, T-720, checked with the
  walk bands in `t720.rs`), `lanes.rs` (f32 lanes, wasm
  SIMD or SSE2, for the commute pair loop, T-036). The commute pair loop is `kernel::mode_choice`;
  its small zone pairs are dealt in chunks there (`COMMUTE_CHUNK`, T-874; `DemandApi::set_chunked`).
  `wasm.rs` (WASM only): the memory and heap figures, the last panic, and the allocator that grows
  the memory 32 MB ahead of need (T-1205, T-1174, T-1594).
  `lod.rs`: a zone's level of detail (T-1222), the everyday sampling and commute chunk a zone opens
  at (`DemandApi::at_level`); its checks in `everyday/lod_tests.rs` (`t1222_probe` measures memory, times and bias).
  Zones of two sizes (T-852: `kernel::coarsen`, `Zones::size`): every zone-pair table goes through
  `kernel::table_layout` and `Gravity::base`/`key` (`PairKey` elsewhere), near pairs through
  `Zones::near`, a square's zone through `Zones::zone_at`, centre distances through `Zones::d2_m`.
  `calib --cells --exact` dumps every figure unrounded, for before/after checks.
  `cargo test --release --features demand nyc_day -- --ignored --nocapture` solves a day on the
  real pack natively. notes/T-026.md. `cargo test --release --features demand --lib t2214 -- --ignored --nocapture`
  checks the land parts in local demand on the real packs (`api/islands_tests.rs`, T-2214).
- The rest of `src/demand/`: `kernel.rs` (the kernel stages: the commute pair loop, coarsening, gravity), `everyday.rs` (the home-based outings layer, `EverydayApi`; `everyday/drives.rs` its car drives), `pack.rs` (the city pack format and its reader), `network.rs` (the player's network as the kernel sees it), `synth.rs` (a synthetic New York-sized city and hand-made network), `lengths.rs` (a zone's trips by straight-line length and mode, for the trip-length graph, T-2365), `bench.rs` (the kernel stage by stage with timings and heap, run by `bin/demand_bench.rs`), `alloc.rs` (the counting allocator, native only), `clock.rs` (milliseconds since a start, native or wasm), `cross.rs` (everyday outings across a zone line, offline).
- `src/bin/`: native tools behind feature `demand`, built in `data/work/target`: `calib`
  (T-509: a weekday on a city's real network with a parameter set, for `pipeline/calib.py`),
  `realsave`, `pack_gravity`, `pack_walk`, `market_pack`, `joint_gravity`, `zone_summary`, `demand_bench` (the kernel's timings, `demand/bench.rs`),
  `cross_everyday` (T-1175: a solve unit's everyday outings across zone lines, its draw in
  `demand::cross`, native only; docs/zones.md 4a).
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
  snapshot signal and the edit functions), `ui.ts` (tab, selection, display settings, theme),
  `cloneNames.ts` (a copied line's name and symbol, T-1872), `northUp.ts` (which end of a line its
  diagram puts on top, T-1894).
- Time: `clock.ts`, `lineFigures.ts` (the 4x clock cap while demand updates, T-1665, and while a new line has no riders, T-2409), `editCap.ts` (which edits cap it, T-1803; `checks/t1803-cap.mjs`), `localTime.ts` (each region's local time and the local time in view, T-309),
  `sun.ts` (the sun's height for the top bar's sky, T-861; the calendar is in `days.ts`),
  `attractions.ts` (the world's attractions from `attractions.json`, made by `pipeline/attractions_json.py`, and their visits on a kind of day, T-263),
  `holidays.ts` (public holidays by country from `holidays.json`, made by `pipeline/holidays_json.py`, T-862;
  a line's and the view's country, T-872; one entry for a holiday several countries keep, T-1098; superholidays near, the sky's rush days and the Money ticks, T-264), `holidayCalendar.ts` (the plain lookups the clock worker shares:
  a holiday's name, `kindOfIn` a country's kind of each day from its holidays and own weekend, T-872, T-262; `superRuns`, `superPart`, T-264),
  `periods.ts` (the five periods), `days.ts` (weekday, Saturday, Sunday and a superholiday's away, going-out and coming-back days, the shown day, which are made while one is near, T-260, T-262, T-264, T-1373).
- Demand and riders: `demand.ts` (results: mode split, riders), `boarding.ts` (the boarding
  account's hour on the main thread: riders packed for the clock worker, platform crowds, T-164; the
  crowded stretches the map marks, T-2236, check `checks/t2236-crowding.mjs`),
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
  links, T-146), `buildings.ts` (station buildings' members, the stations whose platform touches
  the outline, and the pairs that join them into one complex, T-2217; the tool's rectangle offered
  and snaps, T-2218, its direction T-2363, check `checks/t2363-building-offer.mjs`), `buildingView.ts` (the buildings as panels and the map read them: members by
  building, a complex's building, the one under a point, the tool's `buildingDraft`, T-2218), `complexBuilder.ts` (`ComplexBuilder`: the same complexes kept up to date from the
  stations that changed, the world's in `complexView.ts` and the clock worker's snapshot in
  `clock.worker.ts`, T-1380), `patchMode.ts` (`?patch=` on the main thread), `relatedLines.ts` (the clock query `lineRelated` decoded: the lines beside or
  crossing a line, T-1168) and `relatedLayout.ts` (their diagram's rows, columns, turns, pills and
  the pills' hovers, drawn by `ui/inspector/RelatedStops.tsx`), `complexView.ts` (a click selecting the complex, a station opened alone, the
  zoomed-out dot groups; T-217, T-249), `links.ts` (the player's links, T-197), `lineActions.ts`
  (new and copied lines, Edit stops, construct blueprints: the N, L and C keys, T-670), `colourPick.ts`
  (the automatic line colour pick as pure functions: palette pick, CIEDE2000, the lines beside a line, T-2291;
  `lineColour.ts` re-picks a new line's colour after its stop edits; `checks/t2291-line-colour.mjs` in Node and
  `t2291-line-colour-game.mjs` in the game), `scheduleCopy.ts`
  (a line's schedule settings copied and pasted, kept in localStorage, T-1293), `remove.ts` (removing things, with the question asked before removing
  anything constructed, T-096).
- `realGraph.ts`: a region's real-network routing graph (`packs/realgraph/<cc>.rgr`, fetched once
  and decoded), for snapping track to the real network (T-727); read by `workers/realRoute.worker.ts`.
- Account (T-972, T-973): `account.ts` (the optional anita.garden account, noritetsu's, and the
  leaderboards: which server, the session, each slot's game id and its score sends, the boards;
  shonei-server's `trainworld.go`); its panels `ui/AccountPanel.tsx` and `ui/Leaderboard.tsx`.
  T-988, games on the account: `accountGames.ts` (the main thread's side: the account's list and
  pictures, the start menu's settling, downloads and deletions, the sign-in question) over
  `workers/cloud.ts`.
- `quotes.ts`: the construct-all quote and a planned line's price, asked of the clock worker when a panel shows them (T-1171; the line panel's button is `ui/inspector/ConstructButton.tsx`), and the blueprint priced piece by piece for the cost breakdown (T-2269).
- `blueprintStretches.ts`: the cost breakdown's parts and stretches from those pieces, the row under the pointer (`bpHover`) and a row's off switch (T-2269).
- Other: `money.ts` (cash and ledger from the worker, fare income from riders, T-028; `baseTrips`, each line's share of the base fares, T-2104; `moneyTerms`, `moneyWorld` and `lineGeo`, what income, points and region shares read, kept while unchanged so a state redoes only what it changed, T-1209), `fareInEffect.ts` (each line priced at the fares its riders were solved with, so a fare edit pays only once demand settles, T-2266),
  `fareCurve.ts` (the fare curve through the player's three nodes, the first the base fare at 0 km: fare at a length, the first stretch's base and rate, the knots the sim takes, old saves' lines as three nodes, T-883, T-1112, T-1143), `income.ts` and `countryIncome.json` (fares by national income, T-1277: a fare dollar's perceived minutes by country, each country's fare scale by income or the player's, GDP per head at PPP from notes/T-1277/income_table.py, and at market rates (`usd`, the one the rule reads) and the car's running cost a km (`car`) from notes/T-1521/money_table.py, T-1521; T-1906: the start country's income scale (`START`), each country's default over it, and the panel's scale as a multiple of the default; T-1286: the market's fare factor per country; T-1431: the market's three income groups per country from the quintile shares `quint`, notes/T-1431/quintile_table.py), `fareEstimate.ts` (the Money tab's estimate of rail trips and fare income a day for a dragged or unsolved fare curve, from demand's fare sensitivity, T-1300), `fareUndo.ts` (fare changes' undo and redo while they are the last edits, handing over to the sim's, T-1130),
  `moneyTiming.ts` (each game hour's fares, rides, points and running cost from the period tables, T-736),
  `regionMoney.ts` (each line's and each construction's split between regions, the region ledgers, T-531), `stretchMoney.ts` (money by edge and track stretch for the Profit track display, T-2420, and by bus or ferry stretch, T-2452), `points.ts`
  (each line's points per hour by region, points a day, each region's past 24 hours' points, the star's mark, T-684, T-685, T-694), `achievements.ts` (the achievements' tiers, firsts, their figures and words, posting the steps met to the clock worker, T-2124) and `continents.ts` (each country's continent, T-2124; their panel `ui/inspector/AchievementsView.tsx`, the medal and toasts `ui/Achievements.tsx`, styles `ui/achievements.css`, check `checks/t2124-achievements.mjs`), `realShare.ts` (the real networks' train shares the city view's Within bar marks, a table from `checks/t2145-real-share.mjs`, T-2145), `issues.ts`
  (the track model's refusals in plain words), `editNote.ts` (what the last edits did to train trips and profit a day, shown under the view rows at the map's upper left once demand settles, T-2125, T-2237; drawn by ui/EditNote.tsx; `checks/old/t2125-edit-note.mjs`), `openingDay.ts` (the lines on their opening day: the Lines tab's riders count-up and a "bung", not saved, T-1434; `checks/old/t1434-opening.mjs`), `openedZones.ts` (the zones the game has started in, reached or crossed with track, kept for the save, T-729, T-801), `trackZones.ts` (the zones built and blueprint track crosses, by coarse outline, from each network state, T-1047), `loading.ts` (what is loading, the startup gate and
  the clock hold, T-149, T-172; whether the start menu is up, T-677), `loadBar.ts` (the loading bar's
  stages, their shares and texts, each slot's remembered stage times and this browser's pace, T-1195;
  the time left's count and text, T-1479),
  `memoryCheck.ts` (a save's memory estimate and the refusal when it will not fit, T-1197),
  `saveSafety.ts` (persistent storage, T-1214), `busy.ts` (the busy tag's
  text, and work held a painted frame when it should take over 150 ms here, each kind's time per
  unit learnt as it runs; the clock client's big states, T-1200), `palette.ts`, `shades.ts` over `shadeTable.ts` (the light and dark themes' map
  colours: basemap, line casing, map ink, blueprint off, outside shade, thumbnails; the table has no
  page behind it so the clock worker reads it, T-039, T-543), `format.ts`.
- Smaller modules: `capacityKinds.ts` (the words for each kind of capacity limit, for the map's dots, the Trouble list and the line panel's alert), `carryLoads.ts` (a line's riders per stretch carried onto an edited stop list, T-1419), `fareDefault.ts` (a country's default fares in round numbers, T-1662), `fonts.ts` (loads the Nunito faces; Zen Maru Gothic's slices are `ui/zenmaru.css`), `lineLetter.ts` (the chip text of a line with no symbol), `lineRunning.ts` (which lines have no trains at the level now, T-1792), `listScope.ts` (what the Lines and Stations tabs list: the viewed place or the world, and the sort by distance from the screen's centre, T-332), `playtime.ts` (the anonymous plays and playtime ping; `?track` and `?notrack`, T-1771), `splitTip.ts` (the hover of a mode split bar and the city view's split table), `today.ts` (the past 24 hours' riders, boardings and fares for the line and station panels, and the crowd grades, T-446, T-2236), `upgrades.ts` (the station upgrade bits, their texts and a passage's price, T-1281), `worldTotals.ts` (the world as one place: the cities' sums for the Cities tab's first row and the top bar when zoomed far out, T-1110).
- The `WorldView` is built from the clock worker's states in `workers/clockClient.ts`.

### src/map/: the map and what is drawn on it

- `mapView.ts` (the MapLibre map; its scroll zoom told wheel gaps by event time, T-675; the flat map or globe setting, T-033), `basemap.ts` (the Positron style: warm paper repaint, three road weights, light or dark,
  labels in Zen Maru Gothic from `public/fonts/`; the Terrain hillshade, T-1270), `demFlat.ts` (the hillshade's tiles with the relief over flat-class ground blurred away, in `workers/demFlat.worker.ts`, T-1881), `terrainTint.ts` (the terrain classes' wash while building, from `game/terrain.ts`'s grids, T-1270), `realNetwork.ts` (the Real network reference layer: noritetsu's
  PMTiles per region, read from `../noritetsu/` or R2; vite.config.ts proxies `/noritetsu` to 8800 in dev, T-682),
  `styleReady.ts` (`styleUp`: MapLibre style calls are skipped while it has no style, a lost context, and reapplied on `style.load`, T-957),
  `moving.ts` (what runs at most every 100 ms while the camera moves: `PlaceThrottle`, T-1207), `lite.ts` (the lighter map and its auto switch, T-1207),
  `overlay.ts` (the overlay canvas and frame loop,
  notes/T-015.md), `renderer.ts` (WebGL2 track, stations, trains; the strokes kept between changes in `strokeCache.ts`, T-1638, and simplified and culled in `strokeLod.ts`, T-1639, the few that show gathered into one buffer in `strokeGather.ts`, T-1644; its shaders and their shared constants are in `rendererShaders.ts`, the platform crowds' sums in `platformCrowds.ts`), `gl.ts` (shader compile, uniform lookup, zoom interpolation, the GPU timer, canvas sizing, and the camera: `Camera` for the flat map or the globe, `PROJECT_GLSL` and `setCamera` for the shaders, `mercToCss` and `clipToCss` to CSS px, `behindHorizon`; T-033; shared by the renderer, `demandOverlay.ts`, `gradeMarks.ts`, `transferBubbles.ts`, `shadeLayer.ts`, `overlay.ts` and the 2D overlays), `network.ts` (the renderer's
  buffers, built in the clock worker from per-edge, line and station pieces kept between edits,
  T-496, and posted as changes to the last, `NetPatch` and `mergeNet`, T-1029; moved to a new draw anchor by `shiftNetCache`, T-1170), `lineBands.ts` (lines sharing track side by side: each edge's order of its lines, the ribbon at each place, and each line's sideways shift along its path as knots, smoothed along the line, T-1799, T-1848, the lines of one ribbon change easing at one place, T-2140; the Traffic view's shifts by riders, `trafficBands`; after an edit only the lines whose inputs it touched are worked out again, `BandMemo`, T-2238, which `?patch=check` compares with a whole pass), `corridors.ts` (every edge with lines sampled every 25 m, each sample's edges running beside it within 20 m, mirrored, kept between builds, T-1848, an edge's samples the same object while unchanged, T-2238), `strokePost.ts` (how the track and line strokes are posted: whole with room after the last segment, unchanged, or only the new segments written over the page's arrays, and the page's merge; T-1381), `geo.ts` (the draw anchor, T-307; `pxPerUnitAt` and `unitPerPxAt` for the zoom's scale), `anchorVersion.ts` (its version as a signal, T-317),
  `live.ts` (the live map objects for UI code, set by main.ts; `hiddenOnGlobe` for DOM marks, T-033), `synth.ts` (stress network),
  `markData.ts` and `gradeMarks.ts` (the crossing, junction and ramp marks: their data, and
  their draw pass, hit test and tag; T-629, T-655, T-1871), `stationLife.ts` (from zoom 14, arrivals found
  by a timer of eight samples a second, feeding `stationSound.ts`, the WebAudio chime and pops, and
  `faces.ts`, the rising faces on a 2D canvas of its own; T-686; `faceTones.ts`, their skin tones
  and hair colours by country, T-1907), `buildSound.ts` (which click
  plays which building sound, T-1282, T-1474) with `buildVoices.ts` (the synthesised bung, T-1413, the load chime, T-1655, and the sample player),
  `samples.ts` (the recordings in `public/sounds/`, decoded when the context opens, T-1474), `audioOut.ts` (the one audio context
  all share), `transferBubbles.ts` (riders changing trains walking between stations as bubbles, drawn by the renderer after the stations, T-1124).
- Stations: `stationGeometry.ts` (platform shapes along the track for drawing, picking and the
  ghost, T-189, T-254; the waiting room annex, `stationAnnexes`, T-1416), `facilityPlace.ts`
  (where each complex's one waiting room and car park go, in the clock worker, T-1528),
  `parkingLots.ts` (the car parks drawn as lots under the network, their hit test, T-1528), `stationNames.ts` (a station's name suggestions: cross street,
  neighbourhood, city or town, T-534), `stationLabels.ts` (names
  keep their elements across edits unless their text changes, T-496; the labels and the shadow
  layer's features follow the changes `labelNames.ts` reports, T-1380), `labelNames.ts` (`NameIndex`: the names to
  draw, one per name per complex, kept in order and patched from the stations and lines that changed;
  `labelGroups` is the whole list for checks), `capacityMarkers.ts`, `labelLayout.ts` (shared spacing
  for names and tags: `BoxGrid`, `tagLines`, `tipBeside`, `TAG_OFFSET_PX`, `screenBounds`).
- Bus lines (T-1464): `busLayer.ts` (a bus line on the map: its route, stops, corridor shade on
  selection, tags and pick, drawn by the renderer under rail), `busDraw.ts` (the bus tool: clicked
  points and their handles, drag, delete and the active end (T-1514), the worker's `busQuote` preview and price tag, `busDraft`), and the roads they follow (T-1465):
  `busRoads.ts` (zoom 14 tiles fetched from the basemap's tile server, the roads and motorways a bus drives (T-2469), snapping, `near` and `leg`, a leg with no road `refused` (T-2451), a free end landing on another reachable road (T-2467)),
  `roadGraph.ts` (the road graph from those tiles and its A* search), `roadTiles.ts` (the vector tile reader).
- Ferry routes (T-2192, track-and-service.md 6.8), in `src/ferry/` beside `map/`: `waterTiles.ts` (a zoom 12 tile's water polygons and
  bridge lines), `route.ts` (the water raster with bridges filled, port snapping, the leg search, coarse
  for long legs with fine ends, the line's rounding and speeds, `runSeconds`; pure, shared with the check),
  `ferryWater.ts` (`FerryWater`, the tool's client: `port`, `probe`, `line`), run in
  `workers/ferryWater.worker.ts` (fetches and keeps the tiles, each leg's search and rasters).
- Ferry lines (T-2194): `ferryDraw.ts` (the ferry tool: ports placed or shared, the route from `FerryWater`,
  the price tag, refusals, `ferryDraft`; `setPorts`, the panel's port list edits, T-2275); a ferry line's app row is a bus row with `ferry` set
  (`ClockBus.ferry`, the clock worker's `ferryRow`), so `busLayer.ts` draws it dashed and `BusInspector.tsx`
  is its panel (`FerryPorts`, its port list); boats are `BOAT_CAR` in `network.ts` and the train shader's hull; check `checks/t2194-ferry-tool.mjs`.
  Bus and ferry lines' Traffic view widths: `demandLayers.ts` `busWidths`, drawn by `busLayer.ts` (`setTraffic`, per segment `a_w`, T-2274);
  check `checks/t2273-ferry-panel.mjs`. Their Speed and Profit colours (T-2452): `busLayer.ts` per segment `a_col`
  (`segColours`: speeds by `runSpeed` from the network's keyframes, `network.ts` `speedAt`; profit from `stretchMoney`'s
  `buses`, handed over by `setProfit` in `main.ts`), hover rows `trackHover.ts` `busRows`; check `checks/t2452-bus-views.mjs`.
- Station buildings (T-2218, ui-inspectors.md "Station buildings"): `buildingDraw.ts` (the tool: the
  outline's corners, handles, snaps and the quote's tag, an SVG over the map), `buildingNet.ts` (T-2219,
  in the clock worker with `network.ts`: each building's outline in local units, its members as drawn
  (ST_BUILDING) and its inside, kept per building), `buildingFloors.ts` (T-2219: floors, shops and rims
  in the stroke cache's layer, the stencil mark that keeps members' rims off the floor, the hall's crowd
  and a selected building's outline each frame; `floorAt` for picks and hovers); `game/buildingLayout.ts`
  (the inside: shops, cross-hall, free floor, `hallRoom`), the hall's pool in `platformCrowds.ts`, the
  name in `stationLabels.ts`; check `checks/t2218-building-tool.mjs`.
- Tools: `realSnap.ts` (the track tool's snap to the real network: ring, route ghost and tag, the
  second click's route, T-727d; only where a graph covers the click, `covers`, T-913), `pick.ts`, `cellIndex.ts` and `snapIndex.ts` (grids built in idle time that narrow a pick
  and the build tools' snap to what is near the pointer, T-724, and carried over an edit, T-1030), `tools.ts` (drawing track, placing stations, the line tool; on touch the tap, preview and confirm and the control bar's place, T-1260), `touch.ts` (whether the last press was a finger, and what the touch control bar offers), `stationSnap.ts` (which overlapping track a station goes on, the nearest spot with room for the platform, T-671, T-1870), `edgeEdit.ts`
  (dragging blueprint track's PIs, T-064, T-092), `nodeDrag.ts` (dragging blueprint nodes, T-093; a platform's end or a curve's middle, T-1869), `dragHandles.ts` (where the curve-middle and platform-end handles are, for the markers and the drag, T-1869), `endMarkers.ts` (loose-end and draggable-point markers, T-416), `screenPlace.ts` (`placeAt`: every DOM mark placed through the overlay's projection, hidden where its point has no place on the screen, T-1777), `curveCircle.ts` (the circle of the segment being drawn, T-418),
  `blueprintBox.ts` (the Select tool's drag box, T-333, and dragging its pick, T-475; its state is `game/blueprintPick.ts`),
  `stopMarkers.ts` (the toggle markers of stop editing, T-233; the gap to a station they cannot
  reach, drawn by `gapFlash.ts`, T-1896), `reversalMarks.ts` (the icon over
  stations where a line reverses partway, T-797).
- Trains and trips: `trainGuard.ts` (trains in view held behind the train ahead on their track, drawn late by a lag per trip, T-1965; zoomed in, waiting at flat crossings and crossing junction moves by their timetabled times there, T-1966, from `NetworkBuffers.conflicts` = `TrackApi.guard_conflicts`; its line paths are `NetworkBuffers.pathTable` and the like), `trainFollow.ts` (a selected train followed onto its next trip, T-245),
  `tripView.ts` (the trip explorer's route, T-195; its panel is `ui/inspector/TripInspector.tsx`; between zones `MarketTrip.tsx`, T-866; between two groups of bubbles, the Shift+drag box and its samples in `commuterView.ts` `farTrip`, the average in `game/trip.ts` `averageTrips`, T-865; place ends and far groups between zones: `game/trip.ts` `mergeMarketPeriods`, `map/intercityView.ts` `placeTrip` and `interBox`, `IntercityApi::market_trip_between`, T-885; the car's and the plane's parts and the answer between places with no station from the market rows, `game/marketRule.ts`, T-1148).
- Demand views (T-084, T-097): `demandViews.ts` (wiring, a station's riders for its inspector), `demandLayers.ts`
  (line width, train fill), `stationCircles.ts` (the station ridership circles on a 2D canvas over the bubbles' and their hover hit, T-1448; the figures from `game/stationUse.ts`, T-482), `commuterView.ts` and `bubbles.ts` (commuter bubbles and
  bubble selections; the far end grouped into at least 10 trips a bubble in `game/farChunks.ts`, T-909), `intercityView.ts` (longer trips of a selection or station, one bubble per
  far zone, T-118, T-120; one per city zoomed far out, T-186), `demandOverlay.ts` (their canvas, over the network's; a selection's turning outline while its answer loads, T-1216),
  `demandHover.ts` (the count tag by the pointer), `modeBar.ts` (the tag's mode bar, T-494, `logShare` for log-scaled size bars, T-2271, and `localModes`, an open zone's local legs by mode, T-576), `airportMarks.ts` (the plane badge on an
  airport and its hover, T-448, T-674; paths shared with the icons in `glyphs.ts`), `attractionMarks.ts` (a mark on each attraction and its hover, T-263),
- Tags and outlines: `trackHover.ts` and `networkTips.ts` (the hover tags on track, trains and
  stations, T-157, T-194), `stretchLoad.ts` (a stretch's average load in the track tag, T-2007), `hoverRest.ts` (the list of what else is under a resting pointer,
  T-732), `hovered.ts` (what is lit as hovered, from the tags and the list, read by both renderers, T-730), `trackTagRows.ts` (the elevation and speed rows with icons, shared with
  the track tool's tag, T-526; `costTint`, a cost multiplier's colour there and in the Build tab, T-1159), `cityBoundaries.ts` and `boundaryGeometry.ts` (the metro outlines, one
  layer for the zones near the view, exact ones fetched only above `ZONE_LOAD_ZOOM`, and the outside
  shade, T-131, T-139, T-400, T-871), `coarseOutlines.ts` (reads
  `packs/outlines.json`, which `pipeline/zone_outlines.py` writes: every registered zone's coarse
  outline, read once for the outlines and the realms' shade, T-800, with the sea off each coast from
  `pipeline/zone_water.py` (its windows over `--jobs` processes and cached in
  `data/work/zone_outlines_cache`, T-1977) and that sea alone (`coarseWater`) for the focused zone's shade, T-1145; `inOutline` is the point-in-outline test (a cell of the outline's grid no edge meets answers alone, `cellSays` in `workers/cityReach.ts`, T-1348; else its bands of latitude), for realms and the track's zones; it also gives `cityWithin` a place's zone by these outlines, and `readOutlinesForPlaces` is the clock worker's lean copy for that, T-1265), `zoneLoading.ts` (zones still loading drawn grey with a loading
  bar, T-343), `realmOverlay.ts` (the realms not shown: one world shade with the shown zones cut out
  by `boundaryGeometry.ts`'s `ShadeCut` as triangles in `workers/shade.worker.ts` (T-1026), drawn by
  the custom layer `shadeLayer.ts` (T-1025), and a button each, T-975, T-1006), `zoneFocus.ts` (after each move: the focused and viewed zone, the nearby zones, and
  `zonesLoadForView`, the zoom test of `game/zones.ts` `ZONE_LOAD_ZOOM` that decides whether zones
  load for viewing, T-276, T-318, T-871).

### src/diagram/: the network diagram (docs/ui-diagram.md)

`graph.ts` (complexes, each line's chain with the stations beside its track, edges with their lines and the stroke each is drawn in there, T-2405;
built in the clock worker while the diagram is shown, T-2159), `layout.ts` (A, B with side terms, `sidePairs` T-2290, segments, the snap,
routing, line order; `LAYOUT`, `LAYOUT_R2`; run in `workers/diagram.worker.ts`, T-2160), `blocks.ts` (city blocks:
found, placed between A and B, held on a rebuild, and each block's real-to-drawn fit, T-2407), `borders.ts` (the zones'
borders, decided in the drawing: a grid round the lines and pills, each cell the zone of its nearest pill, its
boundaries fitted at 0, 45 and 90 degrees, each shared border once, T-2464), `waterReal.ts` (the water on the
ground from the zones' water masks: the sea, rivers thinned to centre lines, their crossings with the hops' tracks),
`water.ts` (rivers drawn as bands routed between their crossings, never across another line, T-2169) and
`waterSea.ts` (the sea decided in the drawing by the nearest stations' look to the sea on the ground, its coast
fitted at 0, 45 and 90 degrees, T-2169), `lsqr.ts`
(sparse least squares), `fmath.ts` (atan2 and powers from + - * / and sqrt, so every engine lays a graph out alike, T-2255), `draw.ts` (ribbons, rounded bends and pills, pieces going out of a node the same way drawn as one bundle (T-2405), made in the layout worker, painted
on a 2D canvas, the focus faded and `pickAt`, T-2163), `names.ts` (station names placed per zoom level
in the layout worker, painted, and `nameAt`, T-2162), `cityNames.ts` (a name per city block far out, placed per
level beside its box, T-2438), `view.ts` (key M, the canvas, pan and zoom, clicks,
hovers and the camera, the picture, the names picked and measured, the layout worker and its progress;
`ui/DiagramButton.tsx` its button and `DiagramTopLeft`, the save button and loading bar, T-2163, T-2235;
the station tag is `complexTip` in `map/networkTips.ts`).

### src/ui/: Preact components

`Dock.tsx` (the dock, its icon tabs and resizing, T-252); `AccountPanel.tsx` (the account panel,
the person icon, and `startAccountUi`, which mounts both account panels in a root of their
own so the start menu opens them too; styles in `account.css`, T-972); `Leaderboard.tsx` (the
leaderboards and their trophy, placed beside the achievements by `Achievements.tsx`, T-973, T-2179); `HelpPanel.tsx` (the "?" button's list of controls no button shows, T-908); `BuildTab.tsx` (the tools and the blueprint); `ToolBar.tsx` (the track and station bars on the map while their tool is on, T-2282, and `MapToolBar`, which draws the click mode's aura, with `clickMode`, T-2284); `LinesTab.tsx` (with its line groups' rows and row dragging, T-1791, T-1813; the groups from `game/lineGroups.ts`, the hand order from `game/handOrder.ts`), `StationsTab.tsx`, `GroupPill.tsx` (a group's outlined bullet, T-1813), `MoneyTab.tsx` (the tabs; the Cities tab is `Cities.tsx` mounted in `Dock.tsx`); `listParts.tsx` (the scope toggle, the Near column and the name order the Lines and Stations tabs share); `toolKeys.ts` (the build tools, the number and letter keys and `pickTool`); `hideUi.ts` (F2 hides the interface for screenshots, T-1491); `Modal.tsx` (the native modal dialog behind the saves, account and leaderboard panels and the confirmation question, with `useModalDialog`); `useClockQuery.ts` (`useClockQuery`, a panel's question to the clock worker, and `useLatestAnswer`); `scroller.ts` (`scrollerOf`, the nearest scrolling ancestor);
`MoneyHistory.tsx` (the Money tab's history charts, T-447); `FareGraph.tsx` (the fare curve graph, linear and zoomable at one fixed ratio of its scales, its three nodes each dragged one way at a time with hover arrows, an arrow for one past the plot, and rows of steppers under it, the base fare then each stretch's end and rate per km, T-864, T-883, T-1112, T-1143, T-1201); `Cities.tsx` (the Cities list, T-550, T-749); `SettingsTab.tsx`; `SavesPanel.tsx` (saved games, T-251; `thumbShade.ts` recolours a picture to the current theme, T-1852; `SlotThumb`, a save's picture with each panel's city named over it, T-894); `bullet.ts` (`bulletText`, `bulletNode`: a line bullet as one SVG, its circle and its symbol placed from the canvas-measured ink box with the baseline on a whole device pixel, per font, symbol, size and scale, T-705, T-745; the sizes in it match `.chip` in style.css; used by `Chip` and `lineBullet`); `widgets.tsx` (chips, steppers,
checkboxes and the inline SVG icons); `ColourPicker.tsx` (the line colour picker: preview while dragging, one commit on release, T-672); `TopBar.tsx`; `GoalTicks.tsx` (a region's goals as ticks over its split bar, the city view's, T-1437); `SkyBanner.tsx` (the day and night picture and demand band, a canvas, T-861; `MapSky` in TopBar.tsx puts it on the map, T-1868); `StatusTag.tsx` (the demand status tag on the map's top row); `BusyTag.tsx` (the busy tag after it, T-1200); `TroubleList.tsx` (the trouble list at the map's lower right, T-1433); `Guide.tsx` (the Tutorial card, T-1822; steps, notes and tick-keeping in game/guide.ts, its styles in ui/guide.css, `guideVerbose` in game/ui.ts for the tutorial-only longer hovers, T-2263; progress in the save header's `guide` through the clock worker); `KeepDot.tsx` (the yellow "Saved only in this browser" dot right of the search glass, T-1832, T-1868; the hour's count and when it is due in game/keepNudge.ts, kept per slot in localStorage by game/keepState.ts, which clockClient's downloads also mark; check checks/t1832-keep.mjs); `ripple.ts` (the card's ripple, T-1822: one fixed canvas with its own rAF loop, pulses on `data-guide` targets by the timing rules; `startRipple()` is called from `mountGuide`); `TouchBar.tsx` (the touch control bar's buttons, T-1260); `ToolNotes.tsx` (the click mode's descriptor, T-2284, tool hints, the notice and the confirmation
dialog); `DemandSelector.tsx` (None/Homes/Destinations and the purpose filter, the track display and station ridership rows; Drives is a Settings check, T-1477); `Blueprint.tsx`
(Build's quote, its cost breakdown by part and stretch, T-2269, and Construct, and the Select tool's selection bar on the map, T-353); `trackPrice.ts` (the track tool's price at a level and the level buttons' hovers, T-2271); `Debug.tsx` (the `?debug=1` readout); `Credits.tsx` (the data sources and licence notices, closed at the foot of Settings, T-924); `FeedbackPanel.tsx` (Settings' "Submit feedback" modal, sent through game/account.ts, T-1673); `MoneyText.tsx` (a money figure in the alert red while negative, and the hover of a line's money, T-722, T-1851); `PassingNote.tsx` (a one-line note on the map for a few seconds, under the edit note, T-2166); `zenmaru.css` (Zen Maru Gothic as woff2 slices by code point, T-1825); `inspector/` (one inspector
per selection kind: `Inspector.tsx` (the switch on the selection), `CommutersInspector.tsx` (commuter bubbles picked on the map), `ClosestStations.tsx` (the 3 stations nearest on foot, shared by the station and bubble inspectors, T-312), `PurposeBreakdown.tsx` (one purpose's departures through the day, T-193), `PurposeSplit.tsx` (riders by trip purpose as a stacked bar under a line's riders and a station's boardings, T-165); `StationInspector.tsx` (its upgrade row `StationUpgrades.tsx`, T-1281), `TrainInspector.tsx`, `TrackInspector.tsx` and `JunctionInspector.tsx`; `LineInspector.tsx`, handing a bus or ferry line to `BusInspector.tsx` (T-1464), both built from `LineTitle.tsx` (the title, its name, symbol and colour editor, the group row) and `schedule.tsx` (the Schedule table and its hovers, the average speed, the past 24 hours with the Average load, and the vehicle words, `vehicleOf`, T-2373); `CityInspector.tsx` with `CityDetails.tsx`, the city view's own rows (T-1438), and `TripGraph.tsx`, its trips by length (T-2365; the bins and curves in game/tripLengths.ts, the zones' answers and the corridors' distances in game/zoneLengths.ts); `AllCitiesView.tsx`, the Cities tab's All cities row as a view, and `HomeButton.tsx`, the home icon of both (T-2395; the home a signal in game/home.ts, set through clockClient's `setHome`, kept in the save header's `home` by the clock worker and used by ui/locate.ts's open fit); the trip panels share `tripText.ts` and `RouteParts.tsx`;
`ComplexMap.tsx` the complex map; `TrainStops.tsx` a train's next stops and who gets off, T-868; `OneRider.tsx` one sampled rider of a train (`game/riderSample.ts` draws it, `riderTrip` in `map/tripView.ts` reads its minutes), T-1436; `NameEdit.tsx` a station's or line's name edited in place, T-534; `LineGroup.tsx` the line panel's group row and Leave group, T-1791; `GroupInspector.tsx` the group view and its diagram, laid out by `game/groupLayout.ts`, T-1791); `LoadingScreen.tsx` (the clock hold at startup, the loading card,
the screen when demand cannot load, T-149, T-172, T-180); `StartMenu.tsx` (the start menu and the URL switches that skip it, T-677); `StartLogo.tsx` (the menu's logo: a NetworkRenderer of its own with a flat camera, `window.twLogo` its frame counts, T-1567) with `map/logoNetwork.ts` (the logo's letters as network buffers and timetables, no sim); `NewGame.tsx` (the new game sheet: mode, start city, current location, search, real network; `START_CITIES`, `REAL_NETWORKS`, `zoneForLocation`, T-728, T-731, T-1132); `Spinner.tsx` (the small spinner, T-647, and the bunny's train, T-677); `Search.tsx` (T-148); `DemandStopped.tsx` (the alert when demand stops after startup, T-757); `keys.ts` (which keys a focused control keeps, T-183); `locate.ts` (the camera to a station, a line or track nodes (a breakdown row, T-2269): search picks, the tabs' double click, the inspectors' locate button, T-569; `viewStart`, the camera at a new game's start);
`tip.ts` (the hover tag for every panel and control, from `data-tip`, never `title`, T-210);
`blurHold.ts` (every hover kept up through a window blur until the pointer moves, T-483);
`style.css` (theme tokens).

### src/workers/

`realRoute.worker.ts` (the real-network router: graphs joined at borders, A* with the turn limit
and families, then the fit in steps on a track model of its own, dropped for a newer request, T-727d, T-754; the outlines of the regions with a graph sent to the page, T-913); `protocol.ts` (edit ops, clock messages, the network snapshot for demand), `clock.worker.ts` (owns
the network: `TrackApi`, notes/T-025.md; money and autosave too; an edit's follow-up (state, trips, boarding hour, answer) waits behind the previews asked meanwhile, `later`, T-1407; `slotSummary`, a save's picture
with a panel per city, T-894), `stateRows.ts` (`StateRows`: the state's edge, node and line rows
kept between edits and read from the track model again only where `take_changed_lines` and
`take_name_changes` say they changed; station names kept by node, phase tables per line; T-1382, T-1540), `platformLines.ts` (each
station's drawn platform line in world units, on its node row, for the complexes; T-1859), `saveFile.ts` (the save format
and IndexedDB, T-029; slots' backups, T-744; a closed connection opened again and demand caches dropped on a full disk, T-1829; removed zone ids taken as their successors on load, T-1053; each slot's tie to the account, T-988), `cloud.ts` (games on the account, T-988: uploads, claims, settling, conflicts split off as versions; used by the clock worker and the main thread), `versions.ts` (T-2037: a save's origin and latest player change, the origin rule for older saves, saves grouped into one card per game), `clockClient.ts`, `demand.worker.ts`, `demandClient.ts`, `demandCache.ts`
(the last final demand result kept beside the save, T-328),
`demandProtocol.ts`, `cacheWrite.worker.ts` (the demand cache's IndexedDB write, off the main thread, T-789), `demandLod.ts` (which zones are held fine and which coarse, T-1223; the zones built in last are spared, T-2005), `demandFocus.ts` (the focus pool: a game load's zone in view solved alone to its last round, T-2045), `deviceMemory.ts` (the device's memory the pool sizes itself by, and whether it is a phone, tablet or 4 GB device, T-2122), `demandClusters.ts` (the network split into clusters that share no rider,
each a sub-network, T-326; `reachView`, a city's own network inside its cluster, T-756; `cityReach`, the stations a city reaches; the stations the market's riders can reach are flagged on the cluster, T-396), `demandTest.ts`, `cityReach.ts` (which city packs a network's stations can
reach, T-149; the footprints fetched on demand by registry box, T-404; `distanceTo` through a grid of each outline's edges, T-889; `atZero`, whether a place is in an outline, from a grid of cells each crossed by an edge or wholly in or out, so `placeAt` and `cityWithin` find the zone holding a place before measuring any distance, T-1348), `commuters.worker.ts` and `commutersProtocol.ts` (the commuter bubbles' sums,
T-097), `selectionMerge.ts` (a bubble selection's everyday far end added up over the workers,
T-285), `demandSums.ts` and `demandSums.worker.ts` (demand's sums off the main thread: each round's
views of every kind of day, a selection's commuters and longer trips, a station's, T-1208, T-1210), `shade.worker.ts` (the realm shade's cut, T-1026), `focusShade.worker.ts` (the focused
zone's own outline joined with its sea, for the shade outside it, T-1145), `roadFeed.ts` (the clock worker's fetch of the basemap's zoom 14 road tiles under the track being drawn, put in the track model to count and price crossings, T-1929), `marketParts.ts` (the regional market's `part_summary` on another kind of day; plain arithmetic a Node check imports, T-1229).

### URL switches

`?debug=1` (readout; `window.tw` probes incl. `lagStart/lagStop`, `checkPrecision`; `window.twDemand`
with demand timings and `client.qlog` per query; `window.twViews` with the demand views'),
`?paused=1`, `?new=1` (a new game; it owns no slot and never writes one
until a copy is saved, T-320, so it is safe in a browser profile holding real games),
`?continue=1` (the game opened last; checks that reopen a stored game use it), each of these and
the two below skipping the start menu (T-677),
`?acct=local|live|off` (which account server a localhost page uses, remembered; none by default,
T-972), `?synth=1&lines=500&trains=10000`, `?demandTest=1` (demand on T-005's hand-made network; both also
never autosave), `?backupMs=N` (wall ms between a slot's backups, T-744), `?demandWorkers=N`, `?focusWorkers=N` (the focus pool's size),
`?memFree=MB` (the free memory a check pretends), `?lod=` (a fixed level of detail, `workers/demandLod.ts`), `?opened=all` (no realm
hidden; checks that visit zones in every realm), `?patch=check|shiftcheck|0` (patched against whole builds, `game/patchMode.ts`),
`?diagram=r2` (the diagram's round 2 layout options), `?rawdem=1` (the hillshade as it comes), `?logo=0` (no start menu logo),
`?quietMs=N` (how long the loading card may hear nothing before the failure screen), `?keepMs=N` (the hour after which the keep dot appears), `?track` / `?notrack` (plays and playtime
pings on or off, `game/playtime.ts`), and the start menu's `?city=<zone id>` and `?mode=sandbox` (with `?new=1`), `?slot=<key>` and
`?load=file|restore|new`; `?perfOff=...` (CPU A/B switches, `src/perfFlags.ts`, which lists them; notes/T-045.md).

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

The standing checks stay in `checks/`. One-off checks of finished tasks (screenshot makers, before and after
measurements, a feature's first test) are in `checks/old/`, written `old/<file>` below: run them as
`node trainworld/checks/old/<file>`; they still work, nobody runs them as a habit (T-1586).

- `smoke.mjs [url]`: the pre-commit smoke check (CLAUDE.md "Before a commit"). Writes the real
  save's weekday rail trips per url to `artifacts/trainworld/smoke-figures.json`, which t883 compares
  against (T-1135). It prints a WARN line when over 20 `tw-*` folders older than a day sit in the
  temp folder (profiles an old check left; T-1213).
- `doc_lint.py [--all-links] [--since DATE]`: the doc-state check (T-2366), a few seconds, no browser. One
  line per finding, `file:line: [kind] what`, exit 1 if any: a task in both TASKS.md and ARCHIVE.md, stages or
  blockers all archived, an archived id opening a line of for-anita.md, over-long TASKS lines and (from
  `--since`) ARCHIVE or DECISIONS entries, empty TASKS headings, "Update:" paragraphs, over-long doc sections,
  dead names and key letters in the docs, `notes/` links that resolve nowhere (`--all-links` adds the docs and
  code comments). It is what keeps a fact from standing in two files.
- `lib.mjs`: what the checks share, imported by every check that drives a browser or the worker
  in Node: `playwright()` (finds Playwright), `launchChrome` (headless Chrome on a throwaway
  profile from `tempProfile`, deleted by its `close()` and, if the check never gets there, at process
  exit, on an error, `process.exit` or Ctrl-C; a check never makes its own `--user-data-dir`, T-1213),
  `staleTempProfiles` (smoke's count), `collectErrors`, `importSave`, `REAL_SAVE`, `SMOKE_FIGURES`, `demandFinal` (the
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
- `t2088-ridership.mjs [url] [city ...]`: the seven real saves in `data/packs/real/` (nyc lon par tyo
  seo tor sdj) each solved to a weekday final (`opened=all`, 2 demand workers), their rail trips per
  resident and metro riders per km set beside the matching rows of `research/ridership.csv`; prints a
  table and a NOTE (never a FAIL) outside 0.5-2x; json to `artifacts/trainworld/t2088/`. About 10
  minutes; not in smoke (T-2088, notes/T-2088.md).
- `t2145-real-share.mjs [url] [id ...]`: the same seven saves solved to an average-day final on one
  worker, each city's train share of its trips within (weekday, Saturday, Sunday, average day), printed
  as the rows of `REAL_SHARE` in `app/src/game/realShare.ts`; json to `artifacts/trainworld/t2143/`.
  About 25 minutes; rerun after a demand change that moves shares (T-2145).
- `t1053-load-check.mjs [port]`: on its own Vite with a scratch registry lacking wnd, nlc and lit,
  a new game's save and the real save naming them load with Hartford in their place (T-1053, about
  a minute).
- Node, no browser: `t133-picking-check` (picking order), `t671-station-snap-check` (the station tool's choice among overlapping tracks, the layout's guess and the room snap; given a dev server URL it also runs `t671-ghost-hover`, the ghost's first frame against the worker on the real save in headless Chrome), `save_net.py <save> <out>` (a game save's network bytes
  for Rust tests on a real save, given as `T1895_NET`: T-1895's ignored `beside_sweep_real_save`
  counts drawing-beside previews that cross the track followed, by start kind), `t1870-station-follow <url>` (browser: the station ghost against the pointer along a straight, a curve, joints, a junction and a station at zooms 13, 14 and 16, a few minutes), `t1964-station-joints <url>` (browser: the station tool on every plain joint of two routes drawn as the track tool draws, about a minute), `t144-check` (station geometry, line
  widths), `t145-check` (the route preview contract), `t146-check` (complexes, station buildings joining them), `t2217-buildings` (station buildings in the track model on the real save: price, members agreeing with game/buildings.ts, the save), `t2218-building-tool` (the station building tool with the mouse on the real save: the offered rectangle, the snap, Construct, Edit outline, rename, remove and their money; screenshots; needs a Vite url), `t1419-carry` (an edited line's riders carried onto its new stops), `t1053-zone-ids` (removed zones have successors; `zoneNow` and `currentZones`; in smoke), `t125-frame-check`
  (world frames against PROJ), `t116-save-check` (save format), `t285-merge-check` (needs
  `--experimental-strip-types`), `t066-wasm-check` (demand WASM contract), `t066-worker-check` and
  `t149-worker-check` (the real demand worker in Node over New York, Philadelphia and Boston, a few
  minutes each; every zone at once is more than a worker's WASM holds), `t325-bare-check` (a
  zone opened bare against the same zone solved, bubbles and bubble clicks, to the bit; `ZONE=tor`
  for another), `t409-worker-check` (two networks with no track between
  them solve as two clusters, each as it would alone, to the bit; about a minute), `t301-worker-check`
  (packs and the market closed and reopened, also mid-solve and bare: figures to the bit),
  `t1862-noise-keep` (Boston's line moved by settle noise beside a Philadelphia edit keeps its
  figures to the bit; noise adding up past 0.2% trains or 0.5 s, +1% trains, +1 s, a span step, a car
  and a moved stop each re-solve it to a fresh worker's figures to the bit; `exactKeep`; about 30 s),
  `t761-plain-net <label> <out.json>` (a `t756-reach.mjs capture` as plain JSON arrays for the native
  `t761_market_bench`, sim/src/demand/intercity/tests.rs).
- Python: `token_usage.py <transcript.jsonl> ...` (an agent's token use from its transcript,
  CLAUDE.md "Token counts"), `world_field_check.py` (the world field's country totals and fixed
  links, T-375; from trainworld/), `runner_fixes.py` (the world runner's market file picking, cache
  clearing, market heal and busy-port refusal, T-1976, T-1986; the gate's stuck-session handling,
  network retries and the region pipeline with stand-ins, T-2040, T-1979, T-1978, T-1980; nothing live
  written, about a minute), `admin_totals_check.py` (tier 0's census totals by unit: a unit cut by
  the outline, the old-census shares, a config's join on made-up files and the Indonesian and
  Philippine joins, T-2084; about a minute),
  and in pipeline/ `test_education_sources.py` and
  `test_university_sources.py` (run with `python -m unittest` from pipeline/, no network).
### checks/ continued: browser and timing

- Browser and timing (Playwright, `PLAYWRIGHT_MODULE`). A `page.evaluate` that moves the map returns
  nothing (`() => void window.tw.map.jumpTo(...)`): MapLibre's `jumpTo` returns the map, and Playwright
  serialising it is 1-2 s of main thread that reads as lag (T-1052). `startup-timing.mjs` with
  `static-serve.mjs` (startup marks on a scratch production build; `MENU=1` through the start
  menu, timed from the pick), `t1493-bubbles-or-loading.mjs` (jumps between cities at several zooms with Destinations on, a new game and with `SAVE=1` the real save: FAIL when a zone in view shows neither bubbles nor a loading state, or its bubbles never come; about 3 minutes, 6 with the save), `t1533-view-first.mjs` (T-1533: seconds from a zone becoming the zone in view to its bubbles: the real save from the start menu with and without its cached result, the bubbles off and on, a new line on the real save and in a new game, five jumps on the world save, with `CHINA=1` a jump into Wuhan while the China save loads and then the save picked again zoomed out, its cached result ahead of the network (T-1539); against 5 s, the real save's cold load and its new line reported as known misses; `QUERY` adds URL parameters, `perfOff=estimate,cacheAhead` for before T-1538 and T-1539; about 12 minutes, 35 with China), `t1530-station-bubbles.mjs` (T-1530: seconds to the station bubbles' Total and Origins figures on the real save while it loads, turned on after its final and for a new line, and for two new lines in a new game; FAIL only on page errors or a figure never shown; `QUERY=perfOff=splitAhead` for before T-1530; about 8 minutes), `t1198-tour.mjs` (a new game
  viewing and selecting 16 unbuilt European cities, then 75 s out over the sea: Chrome's private
  memory, workers' WASM, open zones; fails if anything stays open or a solving worker passes 1.3 GB
  of WASM; about 6 minutes), `t1174-panic-recover.mjs` (the real save: a demand worker made to panic
  idle, mid-solve and in the aux worker is replaced, the rail trips a day equal the first final's and
  a station is answered; dev server only; about 4 minutes), `t1205-memory.mjs` (a save, default
  all-China: lines built in three far zones and removed; once the zones close the workers well above
  their need restart, WASM falls, figures equal; `QS=&memFree=64` for New York; about 20 minutes on
  all-China), `old/t1368-station-snapshot.mjs` (a new game: a line's stations removed one by one after
  the line, then a station built on no line; demand's network has the save's stations after each;
  about a minute), `old/t801-no-grey.mjs` (no grey
  cities: a map start in a fresh profile turning Homes on, its zoomed-out bubbles, a click selecting
  one; a new New York game's Philadelphia an ordinary bubble zoomed out, loaded by a pan with no
  market pack; screenshots `t801-*.png`), `old/t800-outlines.mjs` (a game started on the map, then a jump to Europe, shows
  every zone's outline at once; `NOFILE=1` blocks the coarse file, `DELAY=<ms>` slows the boundary
  files; screenshots `t800-*.png`), `old/t488-money-days.mjs` (the real save's clock run fast and pulled back
  across midnight: money days and hours one each, in order, none empty; uses the app's Vite, port
  6135 by default), `old/t715-load-money.mjs` (the real save opened from the saves panel while a later
  copy is open, twice: cash loads as saved and moves, no ledger day after the clock; a save with
  ledger days after its clock gets their running cost and the hours' upkeep back; port 6165 by default),
  `old/t975-realms.mjs` (first the location button refused, in Lyon and in Sydney, and a map start showing every realm until its first station; then realms in a fresh profile: no bubbles behind the start menu, a Tokyo start shows Japan only with the shade cut around its zones, a click shows France, a reload hides it again, stations in two realms show both; a click on a hidden realm shows it, a drag does not, no circles at the world view; the zoomed-out frame and a pan across France and Germany measured; port 6165 by default),
  `t1965-follow.mjs [url] [locals] [expresses] [hours]` (a new game's straight double track with a local and its express, or `SAVE=` with `CENTER` and `ZOOM`, at 960x: every frame's pairs of trains overlapping on one track by the timetable and as drawn, the GPU against the CPU, a meeting before and after to artifacts/trainworld/t1965/; about 2 minutes, T-1965), `t1965-jz.mjs [url]` (the shared New York save's Z held behind a J, pictures), `t1965-perf.mjs [url] [save] [scenes]` (the guard's ms a frame and the main thread's, guard on and off in turn), `t1966-cross.mjs [url] [save | test] [scenes]` (waits at flat crossings and junction moves at 1x, 60x and 960x: trains held, waits, let-overs, pairs drawn through each other, on and off; `test` builds `t1966-layout.mjs`'s Y and crossing; T-1966), `t1966-perf.mjs [url] [save] [scenes]` (their frame cost against `?perfOff=crossWait`), `t1966-pics.mjs [url] [save | test] [lng,lat,zoom] [kind] [name]` (a train waiting at a conflict, a frame sequence), `t2185-merge.mjs [url] [counts] [light | dark]` (a Y two lines merge at with a flat crossing on a branch, per train count: the dots and each line's waits, the screen's holds at the merge, the merge dot's hover and the trouble list, a picture each; T-2185), `t2185-real.mjs [url] [save ...]` (every merge in a save with its wait, and the merge dots), `t2232-passing-panels.mjs [url] [local headway s] [express headway s]` (T-2228's 180 km test line: the line panel's button adds passing tracks at the five middle stations in one edit at its price, the express loses "Behind slower trains" (or keeps under 30 s), the local gains "Waiting for faster trains", 8-15 min a round trip at 10/10 (T-2245); shots light and dark to artifacts/trainworld/t2232/; T-2232), `t2233-passing-map.mjs [url] [hours]` (T-2228's test line with passing tracks at B to F and a spur 280 m past E, 2 game hours at 60x with the whole line in view: the express on the running track at D, E's outer track shortened to the spur and every outer track back on the running track (T-2248), expresses drawn passing locals only where the local stands on a passing station's outer track, no two trains on a run drawn through each other, zoom 15 shots of E light and dark in artifacts/trainworld/t2233/; T-2233), `t2018-trail.mjs [url] [locals] [expresses] [hours]` (the same test track: the express's timetable has no stand between its stops, and as drawn it stands only behind a standing local; series to artifacts/trainworld/t2018/; about 2 minutes, T-2018),
  `t1071-train-count.mjs` (the real save's busiest line, its train count raised and lowered four times with the clock running: every train of every line placed each frame from the renderer's own buffers may move no further than top speed allows and come or go only at a terminal, and where new buffers arrive the old and new draw the same trains in the same places; `SPEED` 0 to 2 or -1 paused, `OP=headway`, `KEEP_FRAMES=1`; about 2 minutes; Rust side `sim/src/track/transition_tests.rs`),
  `old/t1029-edit.mjs` (a series of track, station, service and undo edits on the real save or Japan's network at 6x throttling: the worker's and main thread's ms per edit under a CPU profile; `DUMP=1` hashes of the world, the drawn network and the GPU stroke buffers to compare across code or `?patch=` settings; 4-5 minutes for Japan),
  `old/t1380-edit.mjs` (a series of station edits on Japan or the real save at 6x throttling: stations added, moved, constructed, linked, removed, a line's stops, undo and redo; the worker's and main thread's ms, `StationLabels.set` and the complexes named, page and clock worker both under a CPU profile; `TOP=1`; about 6 minutes for Japan), `t1380-complexes.mjs` and `t1380-labels.mjs` (random edits: the patched complexes and station names against a whole build, node only, seconds; in smoke), `t1799-bands.mjs` (the shared-track and real New York saves: bands crossing where two lines run on together, against the pairwise rule and from circles of partings, a click on each band of the busiest shared edges, trains on their band; `[url]`, own Vite, about 2 minutes), `t2222-band-sweep.mjs` (T-2222, T-2258: on the real save and Anita's two Tokyo saves opened as a load opens them, Line and Traffic view, the opened zoom's and each zoom 10-15's simplified strokes against the full ones at every merged vertex, band jumps between joined segments, metres past the range, no line or track segment under `MIN_SEG_M` but its edge's only one; `[url] [--line]`, own Vite), `t1381-strokes.mjs` (random edits to a stroke set's pieces: the strokes posted and merged against the pieces laid out whole, through a real transfer; node only, seconds; in smoke), `t2005-recent-builds.mjs` (the zones of the last two builds are kept from the levels of detail's memory drops and through a reload, with sessions played minute by minute against T-2005's rule; node only, seconds; in smoke), `t2159-diagram-graph.mjs` (the network diagram's graph on the real New York save, Tokyo and all of Japan: counts, the 2 and 3 on the 1's edges, New York's build under 50 ms; node only, about a minute) and `t2160-diagram-layout.mjs` (its layout on New York and Tokyo: octilinear stretches, spacing, time, no line doubling back at a station, a rebuild after one added station, near stations drawn on the wrong side of each other, `l8`: line 8 east of Fleetwood with no loop round it; `r2` for round 2's options; node only, about a minute), `t2290-line8-game.mjs` (builds Anita's line 8 game in Node, artifacts/trainworld/t2290/line8.save, laid out as `l8` by t2160 and captured by diagram-game.mjs; T-2290), `t2407-blocks.mjs` (city blocks on a coast-to-coast mock of New York, Sendai as Chicago and Toronto as Los Angeles: the line under 2 New York widths, no boxes overlapping, rebuilds and edits moving blocks little; the New York save one block; with `jpn`, Okinawa packed; `t2407-draw.py` draws its cases; node only, about a minute; T-2407), `t2289-diagram-intercity.mjs` (all Japan and the Northeast save: no single-line stretch between two zones drawn longer than the longest city block, with pill overlaps and close pairs counted; node only, about three minutes; T-2289), `t2464-borders.mjs` (the zone borders on all Japan, the Northeast, the real save and Toronto's real network: every zone with stations closed, every station drawn inside its own zone's borders and no other's, between blocks too, every run at 0, 45 or 90 degrees, pills a border touches, and the times; `T2464_CACHE=<folder>` keeps the layouts; node only, about two minutes; T-2464), `t2169-water.mjs` (the water on the real save, Toronto, Tokyo and all Japan: every hop crossing a drawn river on the ground crosses its band once, each of its lines once, no band crosses any other line, every station drawn on land, and the times; `--png` writes a picture of the ground and render files for `t2169-shots.mjs`; shares `T2464_CACHE`; node only, about three minutes with Japan; `t2169-game.mjs` takes pictures of it in the game, light or dark; T-2169), `t2438-city-names.mjs` (the city names far out on the same three: New York
named alone, no two names meeting, each beside its block, city names and station names on either side of the switch;
node only, about two minutes; T-2438), `t2161-diagram-pills.mjs` (its pills: none wider than two pitches, none of two stations overlapping, no hooks where there were; node only, about a minute), `t2405-diagram-hidden.mjs` (no drawn stretch of a line 90% or more under another line's stroke, lines sharing a stroke by rule aside, with the half covered counted; `--svg` draws round them; node only, about a minute; T-2405) and `t2405-express-game.mjs` (builds a New York game with a high-speed line and an express of one colour on tracks of their own beside locals, artifacts/trainworld/t2405/express.save, `xp` in the diagram checks; T-2405), `t2162-diagram-names.mjs` (its names: none over a line, pill or name or nearer another pill at any zoom level, interchanges round Times Sq and Shinjuku named, layout plus names under 5 and 8 s; `--svg` draws the views; node only, about a minute), all opening saves through `diagram-saves.mjs` and also laying out the game's own graphs (`nyc-game`, `tyo-game`) captured by `diagram-game.mjs` (`[url]`, own Vite, about 35 s; T-2255; run it after a layout change, then the checks compare their layout with the page's bit for bit and take its times), `t2163-diagram-clicks.mjs` (the diagram on the real save in Chrome: the first layout's loading bar, hover and click on Times Sq, M back and forth centred there, a line picked with the rest faded in both themes, key 1 back to the map, the saved picture; `[url]`, own Vite, about 2 minutes), `t2021-fare-multiple.mjs` (a fare multiplier of 1.2999 or 1.304 acts and shows as 1.30x, in a new game's multiples and an older save's absolute scale; node only, seconds; in smoke), `t1382-rows.mjs` (random edits on the real save or Japan in the track model: the state's rows read from changes against a fresh full read; `--self-test` and `--self-test-names` must detect a held-back change list; node only, about 30 s; in smoke),
  `old/t1283-edit-time.mjs` (the real save's busiest line switched to headways, stepped and set by hand: the clock worker's apply and total per edit, failing past 1 s; about a minute),
  `old/t1276-headway-steady.mjs` (the real save's busiest line on headways, its Medium headway stepped fast and slow both ways and typed with the clock running: every inspector change sampled, the schedule's three rows keep their order and never blank, and one headway shows one train count throughout, give or take the one train the landed edit may settle by; about 2 minutes),
  `old/t1286-market-scale.mjs <url> [country] [factor]` (the real save's weekday at the income's scales and with one country's scale at `factor` times its income's: each zone's local rail trips and the market's, by corridors touching that country; run on a build without the change too and compare the JSON in `artifacts/trainworld/t1286/`; about 3 minutes),
  `old/t1300-fare-estimate.mjs <url>` (the real save: from each solved fare curve, the Money tab's estimate for the next, set and solved; PASS when every step's riders and fare income land within 10% of demand's; about 10 minutes),
  `t1277-fare-scale.mjs` (Node, no browser: a rider in New York and one in Mumbai on a small fixture line, set up as the demand worker sets up each zone, pay the starting curve times their own scale, a player's scale replaces the income's, and a fare weighs about the same at the default scales; seconds; run after any change to fare weights or scales),
  `t1148-market-rule.mjs` (Node, no browser: `game/marketRule.ts`, the main thread's copy of the market's place-pair rule, against the sim's on about 2,000 pairs of places, near, far, across the Channel, to Hokkaido and across seas: distance, car minutes, reference minutes, plane and car shares, trips; and the longer trips' period shares on every kind of day; about 20 s; run after any change to the market's car, plane, crossing or trip rule on either side),
  `t1177-cross-everyday.mjs` (Node, the real demand worker: Boston and Providence with made-up everyday pairs across their zone line written into Boston's market pack in memory (`t1177-data.mjs`), and a line joining the zones; the per-purpose parts, nothing counted twice, loads under their own purposes, a selection's blocks by purpose and far zone with a Saturday's own factors, the trip explorer's legs, rail and walking, and the weekend scaling (`workers/marketParts.ts`); `IDS=gua,fos` runs the market-pack checks on registered neighbours whose packs carry the pairs; about 40 s) and `old/t1177-cross-everyday-page.mjs` (the same pairs in a new game on `BASE` through a Playwright route: the page's selection, explorer figures and panel with its walking row, the walking column, the far-end labels under each purpose filter, and the days; about 3 minutes; screenshots in `artifacts/trainworld/t1177/`),
  `t1854-valley.mjs <label> [url] [spots]` (the terrain tint over Wuppertal, Innsbruck and the centres of Tokyo, Nagoya and Shanghai, screenshots to `artifacts/trainworld/t1854/`), `t1881-shade.mjs <label> [url] [spots]` (the hillshade, the relief alone and the tint in a new game in Tokyo, Nagoya, Shanghai, Midtown, Hong Kong, San Francisco and Seoul; `?rawdem=1` for the shade as it comes; screenshots to `artifacts/trainworld/t1881/`), `t1270-terrain.mjs` (a new game: track on flat ground in Brooklyn at 45 US$M a km, on mountain ground above Innsbruck at 112.5 and a bored tunnel there at 99, spots found in the zones' terrain files; Construct charges the quote; the hillshade in both themes and the track tool's terrain chip, screenshots in `artifacts/trainworld/t1270/`; about a minute, port 6184 by default),
  `old/t1161-copy-delete.mjs` (a new game: a copied line starts with no trains, removing the piece past a station or between two keeps each station in place with track to its platform's end, and auto elevation draws on from +2 track at +2 by real clicks; about a minute, port 6125 by default),
  `t1025-shade.mjs` (Node, no browser: the realm shade's triangles for each realm alone and several groups up to all; no point of a shown zone shaded, none shaded twice, open land and sea shaded once; `quick` for the groups only, about 6 s, all of it under half a minute),
  `old/t728-newgame.mjs` (the new game sheet in a fresh profile: the search, a sandbox in Tokyo from
  its real network, a normal London start, a map start, the real New York save turned sandbox from
  Settings; slots left keep their bytes; screenshots t728-*.png; port 6150 by default),
  `old/t734-track-drag.mjs` (the real save with the track tool: a press and drag on a blueprint node
  moves it, a click still places track; port 6112 by default),
  `t1779-branch-ghost.mjs [url] [save]` (branches off a line on a real save, the shared-track New York by default: each preview's ms, how far the ghost is behind when the pointer stops, click to placed and to the next ghost, the clock worker's ms per step; about 3 minutes, port 6199 by default; T-1779, T-1780),
  `t2403-follow.mjs <url> <new|nyc|nycdense|japan>` (the track tool's ghost following a pointer driven every 16 ms that never waits for the page, a click every 1.6 s: pointer to ghost drawn by time since the click, frames, long tasks, the messages after each click, the achievements' figures before and after; `PROFILE=1`, `TRACE=1`, `STOPS=1` (the line tool's stop clicks), `EXTRA=patch=check`; about 4 minutes on New York, outputs in artifacts/trainworld/t2403/; T-2388, T-2403),
  `t1805-placement.mjs [url] [save | fresh]` (runs of three blueprint segments off constructed track mid-edge, then off that blueprint: each click to the next ghost, the clock worker's ms, whether demand was sent a network; `TIMELINE=1`, `PROFILE=1`; T-1805),
  `t1407-edit-ghost.mjs [url] [save]` (train count and headway edits on the busiest line with the ghost moved every frame: the first ghost's wait, the worst, the worst asked from 150 ms on, the boarding lines redone; `ROUNDS`, `CITY`; about a minute a round; T-1407),
  `t2227-roads-lag.mjs [url]` (road crossings on and off, `?perfOff=roads`: the ghost swept round Midtown cold and warm, each preview's ms, frame gaps and long tasks, then ground segments placed in fresh tiles, click to answer, state and redraw; `RUNS=new:on,real:off,...`, `PLACES`, `TAG`; about 1 minute a new game run, 4 a real save run; T-2227),
  `t1407-last-edit.mjs [url]` (a train count, headways, the only line removed, an undo and an early blueprint segment, each the page's last action: every answer arrives, the early one's state follows; about 2 minutes; T-1407),
  `old/t1155-branch.mjs` (track drawn from a station, through the track tool's clicks, on the real save
  and a fresh game with double, quad and high-speed track: the branch's junction 20 m past the
  platform, the station keeping two track ends, one undo; screenshots in t1155/; port 6122 by default),
  `old/t1260-touch.mjs` (building by touch on a 390 px phone page: track by tap, preview and confirm, a
  pan placing nothing, stations on track (and open ground, as it was then), a line by taps, delete's cancel, the
  Select tool's taps and held box, then two mouse clicks on a desktop page; screenshots in t1260/;
  port 6165 by default),
  `old/t727-snap.mjs` (the snap to the real network in a new game: Tokyo to Shin-Osaka and Penn
  Station to Emeryville by two clicks each, one undo each, Tokyo to London refused, a click off
  real track an ordinary segment; router, fit and preview times and the longest main-thread task;
  t727-snap.json, t727d-*.png; about 2 minutes, port 6170 by default),
  `old/t1196-switch.mjs` (two slots of the real save: Open reloads the page into the other, the card's
  three stages in order, all of China refused,
  persistent storage asked; port 6146 by default), `old/t1251-save-by-hand.mjs` (two tabs on one slot:
  Save with autosave on, Save in the tab without the lock takes it and the other stops autosaving,
  Ctrl+S takes it back, a double click or the pencil renames in place and Escape cancels; port 6161 by default),
  `t988-cloud.mjs` (games on the account against a local shonei-server it builds and runs, two
  profiles as two devices: the sign-in question, saved on A and opened on B, A stops, a save by
  hand takes it back, offline play, a fork, a delete crossing over, a game kept here only, Make a
  copy naming the copy in place, earlier versions as the account's list only;
  t988-question-*.png, t988-saves-*.png, t1395/built-*.png; about 5 minutes, port 6169 by default),
  `old/t743-slots.mjs` (through the saves panel: Open a file, New game and a backup's Open as a copy
  each open a new slot and leave every other slot's bytes as they were, the real save and
  artifacts/trainworld/recovered-day26.save both; two backups rotate at `?backupMs=4000`; a deleted
  slot's backups go; port 6150 by default), `old/t894-previews.mjs` (save pictures: a game in New York,
  Boston and Philadelphia and the real save, their panels and size, screenshots of the saves panel
  and start menu to artifacts/trainworld/previews/; `SHADE=dark`; port 6120 by default),
  `old/t1265-hk-preview.mjs` (a line in Hong Kong's northwest, nearer Shenzhen's centre, saved: the slot
  names Hong Kong only and the line keeps Hong Kong's holidays; port 6167 by default),
  `old/t1372-superholidays.mjs` (the calendar against the last commit's: only days away and Obon change kind; China's and Japan's runs; the real save's ordinary days equal 8800's to the bit, the away day a Sunday times `AWAY_RATE` by purpose, the day views timed; PORT 6172 by default, about three minutes; T-1372),
  `old/t1373-rush-days.mjs` (going-out and coming-back days: the calendar's kinds 4 and 5; on a save with the clock moved two days before its country's next run, the rush views made, every purpose but the longer trips a weekday's and a Sunday's, the longer trips twice a weekday's, lines leaning each way, rail trips a day, the Day choice and the calendar's click, screenshots; `SAVE=t1156-all-china.save COUNTRY=CN` for the China network; T-1373),
  `old/t1281-upgrades.mjs` (the real save's rail trips as saved, with a waiting room at every station and with a park and ride too, in one edit each; screenshots of the station panel's upgrade row and a terminal's sidings, `SHOTS_ONLY=1` for those alone; T-1281), `old/t1528-facilities.mjs` (Node: where a complex's car park and waiting room are placed on made-up layouts, clear of boxes, track, each other and the label; T-1528),
  `old/t883-fare-curve.mjs` (the real save, the start curve and fare scales read from the game (T-1535): the default fare's rail trips against smoke's last run on the same url, and again after the changes below, the same line through other nodes and an old two-node curve changing nothing, a doubled rate and a curve steeper past 20 km losing rail trips, the weekday's income being the base, the rate and demand's extra; the Money tab's hover arrows, one-way drags sending once on release, undo, zoom keeping the slope, the off-plot arrows and their click; fare graph screenshots in t1143/), `t1201-fare-rows.mjs` (a new game's Money tab at the dock's narrowest: the rows as rates, none wrapping, a rate's whole-cent step or typed value moving later fares together, never below 0, the base and a stretch's end keeping the rates, a drag's fare followed by the rows; screenshots in t1201/; about a minute), `old/t736-money-timing.mjs` (Node: a week's hours of `moneyTiming.ts` keep the period tables' fares,
  rides and running costs, with the largest hour-to-hour change beside the flat booking's),
  `old/t872-holidays.mjs` (the real save over the next Thanksgiving: that day's boarding account, fares
  and ledger are Sunday's, the days around it weekdays; the day is a weekday in Canada and the UK;
  T-262: six countries' own weeks, a substitute day, a worked Saturday and a US working holiday in
  the calendar, and a new game's Hong Kong and Shanghai lines over their own Saturday, Sunday and a
  substitute day),
  `t745-bullets.mjs` (the real save's line bullets in the Lines tab, the line inspector and the map
  tags, plus digits, two characters and kana, light and dark, 100% and 90% UI, pixel ratio 1 and 2:
  each symbol's ink centred in its drawn circle within 0.6 device px, measured on the screenshot;
  port 6160 by default, about four minutes),
  `t1792-idle-bullets.mjs [url]` (the real save with one line at 0 trains at the level now: its
  bullet dimmed with a "Not running" hover in the Lines tab, station panel, related lines view and
  map tags, light and dark, into artifacts/trainworld/t1792/; about a minute),
  `t1812-symbol.mjs [url]` (real clicks and typing in the line editor on the real save: a symbol then
  Done, rename then symbol, symbol then rename, a rename of a hand symbol, a blank symbol, an export
  and reload; about a minute), `t1812-shots.mjs <before|after> [url]` (the logo, a selected line and
  group, a scrolled line panel and the station panel, light and dark, into artifacts/trainworld/t1812/),
  `t1813-drag.mjs [url]` (real mouse drags on the real save: a drop on a row's middle groups, into a
  gap reorders and takes a line out of its group, inside an open group reorders it, a station row
  moved; the hand order and a group bullet set by hand kept on export and reload, a header click
  dropping the order; the pill's default text and colour; a group in the shared Tokyo start; light
  and dark into artifacts/trainworld/t1813/; about two minutes),
  `t1829-storage.mjs [url]` (a squeezed store saved by dropping demand caches, and full for good saying so then saving once there is room; site data cleared and the database deleted under a running game, saved again under its name and account tie; deleting a slot or a tombstone is for good, T-1827 leftovers swept at start; about three minutes),
  `t2037-versions.mjs [url]` (one card per game: the origin rule on Anita's Tokyo forks, the latest
  player change on top, the conflict and idle-tab rules through `settleSlot` against a fake account
  server in the page, pictures of the start menu and saves panel light and dark, collapsed and
  open, into artifacts/trainworld/t2037/; about three minutes),
  `t2123-tutorial.mjs [url]` (the tutorial in a fresh profile: New York by default, the folded view
  rows, the ripple repeating until its control is pressed and following the tabs, the close
  confirmation, card 1 in its order (split, Homes, bubble and trip, then track to stops) with its notes and
  Continue rippled, card 2 (trains, play, crowd, levels, Traffic, 2%), the tutorial-only longer crowding hover
  and road chip, a new game paused until the play step (T-2142), Start again, and a Choose on map start's zoom
  step, into artifacts/trainworld/t2123/; about four minutes),
  `t2266-fare-lag.mjs [url] [save] [out.json]` (the real save: fares to 10x then 5x while solving; income held
  at the old fares until riders solved with the new ones show, and ends at the new figure with every line priced
  at 5x; port 6172, 2 demand workers, about 5 minutes; T-2266), `t2411-profit-map.mjs [url]` (the real save's
  Profit view: each rail line's edges sum to its weekday fares and upkeep share, the stretches to the network,
  the hover's rows; pictures in artifacts/trainworld/profit/; port 6195, a few minutes; T-2420),
  `t2452-bus-views.mjs [url]` (t2195's real-ferries.save with three bus lines added: each bus and ferry line's
  Profit stretches sum to its weekday fares and upkeep share, its hover in Line, Traffic, Speed and Profit,
  pictures light and dark in artifacts/trainworld/busviews/; port 6189, 2 demand workers, about 4 minutes; T-2452),
  `t2092-first-service.mjs [url]` (a new game
  opens on Build; a new line's levels set one at a time; about two minutes),
  `t1826-saves.mjs [url]` (backup tiers simulated and live with `?backupMs=50`, a pull's backup, a
  newer save with an extra block or a higher level opened read-only and left byte for byte through
  an edit and a minute, Download and Make a copy of it, the leave-page question only with an
  unsaved change; about three minutes).

### checks/ continued: clicks, fares, saves, buses and ferries

- `t486-click-on-top.mjs` (a click on a blueprint point's square over a bubble
  selects the track, beside it the bubble), `old/t1515-click-priority.mjs` (a click on a rail line, bus
  line or station inside a bubble selects it with no bubble tag, one on a bubble clear of the
  network the bubble; port 6115 by default, about two minutes), `t1550-far-bubbles.mjs` (far out with
  the city bubbles on, a click where a station meets a bubble selects the bubble and shows its tag,
  just closer in the station; port 6170, about two minutes), `t1906-money-rules.mjs [url]` (new
  games in New York and Tokyo start at $2.50 and $1.30, every other country at its own round start
  fares, the panel at 1.00x and a set dial stored as shown; the real save keeps its fares and stored scales and reads 1.00x; the loan profit rule and
  the refused buttons' hovers; pictures in t1906/; port 6150, about a minute; replaces
  `old/t1554-fare-display.mjs`), `t2022-loan-schedule.mjs` (node with `--experimental-strip-types`,
  no browser: `game/loan.ts`'s 60 equal payments drained in uneven slices, the total, zero after the
  last day, a payment on short cash, a save round trip and a pre-T-2022 loan; a second),
  `t1626-hsr-fare.mjs [url]` (a new game:
  the High speed fare row only once a line runs high-speed trains; on that line multipliers 1.0, 1.5
  and 2.0 lose riders and raise the fare a ride; a shot to artifacts/trainworld/t1626-money.png;
  about three minutes), `old/t1516-fares-never-negative.mjs`
  (a new game and the real save with a bus feeder under a tapering fare curve: no line's fares a
  day below zero; port 6116 by default, about six minutes), `old/t533-draw-order.mjs` (the real save: the shorter line
  on top where lines share ground, a line with no trains under; port 6130 by default),
  `old/t658-lanes.mjs` (the real save: the overlay at zoom 12 and 15 pixel for pixel the baseline
  build's, 8800 by default; either of double track's two tracks picks the line at 17.5; two
  trains passing each picked on its own track; draw time; screenshots at 15 to 19, light and
  dark; port 6160 by default),
  `old/t586-trip-agrees.mjs` (New York's real save and a Mumbai test line: a far-end bubble's hover,
  the trip explorer opened from it and the worker's far-end sums agree within 1 point under every
  purpose filter and day, and a Walk row under 30 minutes' walk or with a walking share; port 6196
  by default, about eight minutes),
  `old/t674-airport-modes.mjs` (a new Tokyo game and a line to Haneda: the plane's hover follows the
  worker's airport legs after the solve, and a game day later the destination bubble beside it
  reads the same selected or not; port 6144 by default, about four minutes),
  `t1374-touristy.mjs` (Node, no browser: the market of Kansai, Hangzhou and New York's packs; longer trips arriving on a going-out day per weekday in named touristy zones (Kyoto, Nara, Universal Studios, West Lake, the Met) against ordinary zones of about their people, and a weekday the same with the attractions as without; under a minute; T-1374),
  `t263-attractions.mjs` (the real save's final with attractions, the marks and the Met's hover in Manhattan, the Attractions filter, a line at Ueno and the Tokyo National Museum's hover; screenshots to artifacts/trainworld/t263/, SHADE=dark for the dark theme; port 6123 by default, about four minutes),
  `old/t680-new-line-crowd.mjs` (a new game's Mumbai test line: its platforms are empty when its
  trains start and fill no faster than riders arrive; also runs on a production build; port 6145
  by default, about two minutes), `old/t684-points.mjs` (a new game's two Mumbai lines with a walk
  between them at Dadar: points count each journey once and the loads' passenger-km, a whole
  hour's ledger points match the hour's table shaped and lagged as fares are, and a city's points
  are its past 24 hours'; dev
  server only, port 6153 by default, about four minutes),
  `old/t648-stop-edit.mjs` (the real save: a station built on a line's track, then added as a stop
  with a click on its marker; click to world and to the frame drawn with the new network, and a
  screencast of the click's frames with a flash test, pixels the same before and after that a frame
  between draws otherwise, as a mask and a frame strip),
  `t328-cache-check.mjs` (the demand cache on reload: timing, cached
  against fresh final to the bit, a changed build, pack or network shows none), `old/t348-lag.mjs`
  (main-thread long tasks and frame gaps while demand loads: startup, pans, a re-solve, a zone
  joining the solve; optional Chrome trace) with `old/t348-analyze.mjs` (what each long task in a
  trace spent its time on), `old/t1008-load-lag.mjs` (long tasks, frame gaps and each demand message's
  main-thread ms in the first seconds after a save loads, with the main thread's top functions,
  then the game's figures at demand's final; a save imported into a new game, or `CONTINUE=1` the
  slot reopened with its demand cache; `EDITS=1` the long tasks after a track and a station edit),
  `old/t1020-click-latency.mjs` (laying track with the mouse: click to drawn, ghost latency per move,
  frame gaps, long tasks and the worker's steps per click, on a new game, the real save, Japan's
  network, Anita's London-Shanghai-London repro, whose far click must be refused at once and whose
  requests after Esc must stay in one city, and T-1039's long pieces and snapped routes zoomed out
  followed by short track in Belfast, with how long the clock worker takes to answer meanwhile),
  `t1777-stuck-marks.mjs` (screen-space marks left in place while the map moves: a blueprint piece in
  Manhattan with the select, track and station tools, a wheel zoom out, a jump to Singapore and a pan,
  on the globe and the flat map; about 1.5 minutes, T-1777),
  `t1811-track-tool.mjs [url] [fresh]` (the track tool with real mouse events by Central Park, real
  save or a fresh game: keep placing carries on over open ground and stops on joining a loose end or
  mid-piece; a 40 px press-drag moves a blueprint end and undoes, a short click branches; the squares
  show; about a minute, port 6183 by default, T-1811),
  `t1896-gap.mjs` (Node, the track model: the gap a stop cannot cross on made-up track, and every
  line's cut-off stations on the real save get theirs in time, T-1896),
  `t1899-stop-orders.mjs [clicks|build]` (Node, the track model: 120 click orders of stations along
  one way through a fresh city's track, none may backtrack; `build` writes the city's game file),
  `t1899-stop-logic.mjs [url]` (browser, that city: clicks along a way round run straight, a new line
  on high-speed track gets high-speed trains and back, an established line keeps its type, overlapping
  markers on a new line take the station no trains stop at; about a minute, default port 6120, T-1899),
  `t1869-drag.mjs [url]` (a new game by Central Park, real mouse drags: a blueprint curve's middle
  bends the curve through the pointer and undoes, a platform end moves its station, the Select tool's
  pick shows its squares moving and the speed labels; screenshots to artifacts/trainworld/t1869/;
  about a minute, port 6169 by default, T-1869),
  `t1898-select.mjs [url]` (the real save, real mouse drags: a Select tool box moves only the nodes
  inside it, the bar's High speed and Regular with their messages, and a 320 km/h curve bent and its
  joint dragged keeping its speed; screenshots to artifacts/trainworld/t1898/; about two minutes,
  port 6190 by default, T-1898),
  `t714-pan.mjs` (frame time while dragging and wheel-zooming the
  real save at zoom 11-16 with the clock running: fails over 12 ms of main thread a frame or 5% of
  frames over 50 ms; options for a throttled CPU, a trace per thread, settings, a tool),
  `t871-europe-pan.mjs` (drags across Europe in a new game: fails if a zoom at or below 7.5 opens
  a pack or fetches a water mask, or zoom 9 opens none; reports long tasks; about 2 minutes),
  `old/t908-longpress.mjs` (real mouse events on home bubbles: a quick drag pans, a held press then a
  drag boxes bubbles with no pan, the controls panel opens and closes; about a minute),
  `t884-loadbar.mjs` (the loading card's bar through the start menu: a new game, the real save and
  the world save, each slot twice: fails over 25% of a wait above 90%, a bar going back, or a fill
  over 30 points from the share of the wait gone; about 5 minutes),
  `old/t1480-antipode.mjs` (a new Wuhan game on the globe zoomed 6 to 13: fails if a far realm's circle,
  Brazil at the antipode, shows over Wuhan, or the near realms' circles go zoomed out; about a minute),
  `t1544-focus-race.mjs` (the draw anchor and the zone in view changing in one task, as a loaded
  game's first state does, with the bubbles built: fails on a page error or bubbles that do not come
  back; about a minute),
  `t724-hover.mjs` (what a pointer move costs on the real save with the select and station tools,
  timers and the station name included, then a wheel zoom's frame placers: fails over 8 ms at the
  95th percentile in any view),
  `t737-japan.mjs` (Japan's whole N02 network, from `../artifacts/trainworld/t737-japan-trains.save`:
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
  comparison), `old/t1168-related.mjs [save] [line ...]` (the related-lines query on every line of a
  save in Node, timed, with the named lines' groups listed), `old/t1168-related-page.mjs <port>` (the
  same query through the clock worker: a copied line made an express, and the real save against
  the Node figures), `t1791-layout.mjs` (the group view's layout on the real save: the 4, 5, 6 and a copy, two unrelated lines, a synthetic express and short turn), `t1168-layout.mjs` (the diagram's layout of the real save's 2, 4 and 6X against
  the mockups),
  `old/t1485-end-marks.mjs [url]` (the track tool's loose-end rings: only near the pointer, on drawn track ends, none on a switched-off blueprint or a stub under 4 px on screen (T-1534), none with the pointer off the map or on a control over it while keys pan, near the pointer after a drag pan, none once the tool changes; about a minute, port 6152 by default; T-1485, T-1522),
  `old/t1500-change-trip.mjs [url]` (Shanghai - Wuhan and Wuhan - Beijing sharing a station, metro lines at both ends: the trip explorer between Shanghai and Beijing finds the change at Wuhan in every period, new game and after a save and load; about 4 minutes; T-1500),
  `old/t1513-bus-delete.mjs [port]` (the delete tool on a planned and a constructed bus line: the tag, removing at once, the question, undo; T-1513),
  `old/t1465-bus-roads.mjs [port]` (a bus drawn with the mouse in Manhattan, San Francisco, Lisbon and Dongguan: it follows drawn roads, plausible road length, no hairpin turns, cheap pointer moves, Edit route keeps the corners, far from roads stays straight; then legs across divided roads found in Dongguan, Berlin and Shenzhen (T-1512); screenshots to artifacts/trainworld/t1465/; about 2 minutes, port 6365 by default; T-1465),
  `old/t1514-bus-edit.mjs [port]` (the bus tool with the mouse in Manhattan: the Build tab's Bus button, finishing by a click on the last point, a double click, Make line and Enter, Escape, then Edit route from the far end, dragging, deleting by right click, Save route in the panel, Escape on an edit; handle screenshots in light and dark to artifacts/trainworld/t1514/; about a minute, port 7014 by default; T-1514),
  `t2451-bus-refusals.mjs [port]` (the bus and ferry tools in New York with the mouse: a first bus point and a later one over the Hudson refused with the tag and message "Too far from a road", the leg to the pointer red and never in the line's colour, a route up the avenues makes a line on roads only, 14th Street to Hoboken refused as no road path, legs over the Verrazzano and the Brooklyn Bridge on roads; a ferry port previewed and placed on the shore from a pointer on the East River; screenshots to artifacts/trainworld/ferrybus/; about a minute, port 6187 by default; T-2451, T-2467, T-2468, T-2469),
  `t2192-ferry-routes.mjs [dev url]` (ferry routes on the nine legs of notes/T-2177.md, two long ones and two waters that do not meet: in Node, ports snap, every leg routes, its room on 2.5 m rasters of the check's own, bridges crossed, minutes by boat, a picture each in artifacts/trainworld/t2192/; with a dev URL the same legs through the worker in headless Chrome, failing past 2.5 s a leg after its tiles; about 2 minutes; `ONLY=venice,nyc`; T-2192),
  `t2194-ferry-tool.mjs [url]` (the Ferry tool with the mouse on a new game: Whitehall and St George placed, the route and the tag's price, Make line, Construct at US$10M a port and US$8M a boat, boats moving, a boat's and the line's panels, the Lines tab, the Finances tab's Ferry row and boat rows, a save exported and loaded with the line, the boat fleet and the ferry fare; two Central Park waters refused with "No water path found"; screenshots at zoom 12 and 15, light and dark, to artifacts/trainworld/t2194/; about 3 minutes, port 6199 by default; T-2194),
  `t2195-ferry-demand.mjs [url] [--baseline]` (ferries in demand on the real save: the plain save solved to the bit against a `--baseline` run on the code before a change; five NYC ferry lines added through the water router and constructed, Staten Island's weekday riders 20,000-100,000, rail trips up, the Ferry fare at 2.00x losing riders, the panel's rows; cold solves of the plain and ferry saves in turn, within 10%; `T2195_WORKERS` (2); about 10 minutes, port 6191 by default; artifacts/trainworld/t2195/; T-2195),
  `t2196-ferry-fans.mjs [url] [--baseline]` (the ferry fans on the real save, against a `--baseline` run on the code before a change: the plain save to the bit by purpose, line and day; t2195's real-ferries.save with Staten Island's riders by purpose, commutes and errands moved by crowding alone; the same save with `?perfOff=fans` in turn, to the bit as the baseline and the fans' workers' ms within 3% of it; `T2196_WORKERS` (3), `T2196_REPS` (3); about 8 minutes; artifacts/trainworld/t2196/; T-2196),
  `t2273-ferry-panel.mjs [url]` (on t2195's real-ferries.save: the East River ferry's Traffic view widths on the map and in its panel, the route's hover per stretch, a port taken out by its dot and put back by undo, a port dragged below another; the ferry tool's tag up through 121 pointer moves, pictures light and dark, a new line's headways 20, 30 and none; `T2273_WORKERS` (3); about 3 minutes; artifacts/trainworld/ferries/; T-2273, T-2274, T-2275),
  `t2215-no-route.mjs [url]` (the grey No route on Salamis and Cheung Chau, a new game in each: the zone's split adding up to 100% with a No route share, the bubbles' legs per cell adding up to its trips, a bubble's hover counting its bar's four parts, the city view's No route part, the trip explorer to the shore across with "No road between these two places"; a ferry built and the share falling; screenshots light and dark, with and without the ferry, in artifacts/trainworld/t2215/; 2 demand workers; about 4 minutes, port 6196 by default; T-2215),
  `t2365-trip-graph.mjs [url] [--shots]` (the city view's trips by length on a new game and the real save in New York: the graph's bins adding up to the zone's split and To other places, each layer's drawn area its trips, "Trips within New York", the graph's hover at 10 km, the top bar's hover with its Distance row; `--shots` to artifacts/trainworld/tripui/, light and dark; 2 demand workers; about 3 minutes, port 6189 by default; T-2365),
  `t1517-bus-touch.mjs [port]` (the same edit with a finger, CDP touch events: a short press, dragging a point, a pan from empty ground, a second finger, a point held 500 ms deleted, a press that wanders; `SHADE=dark`; about a minute, port 7017 by default; T-1517),
  `t1464-bus.mjs [url]` (a feeder bus to a rail line in a new game and two on the real save: riders, the rail line's gain, Construct's price, demand's corridor, the save's route; about 6 minutes, port 6164 by default; T-1464),
  `t1880-bus-feeders.mjs [url] [out.json] [ids, id@seconds]` (six buses on the real save, one at a time: each one's own profit a weekday against the change in the network's, the rail lines' fares and riders and rail trips; `id@seconds` runs one at another headway; about 5 minutes a bus, port 6164 by default; output in `artifacts/trainworld/t1880/`; each bus's own profit under the even base share too, `DEMAND_WORKERS=` the page's workers, 2 by default; T-1880, T-2104),
  `t2104-fare-share.mjs [url] [save] [out.json] [day]` (one solved save, the real one by default: each line's fares and profit under the base fare's share against the even share per boarding, the same riders; PASS when the network's fares agree to the cent; port 6169 by default, one demand worker; T-2104),
  `old/t1414-ui.mjs [url]` (the real save in headless Chrome: a station with a train on it picks the station, the "Station upgrades" heading, the stop list without "Add all" and the A key, the disabled blueprint total's double click; screenshots in `artifacts/trainworld/t1414/`; T-1414, T-1415, T-1417, T-1420),
  `old/t1491-hide-ui.mjs [url]` (F2 and Esc hide and bring back the interface, the map taking the room, the Controls panel's F2 row, light and dark; screenshots in `artifacts/trainworld/t1491/`; T-1491),
  `old/t1476-controls.mjs <port>` (the calendar icon's days, the map's rows and the Drives setting, the "+N daily rides" indicator on the map after a stop edit on the real save, light and dark into artifacts/trainworld/t1476/; needs the real save's demand final, about 4 minutes; T-1476, T-1477, T-1478),
  `old/t1391-related.mjs <port> [prefix] [desktop|phone]` (the related-lines view on a new game built
  for the edge cases: a line beside another ending at a shared station, a two-stop interline, a
  line going on past the other's end, a line through one station twice, a loop with a line beside
  it across its seam, an express, editing stops; screenshots into artifacts/trainworld/t1391/),
  `city-browser.mjs <id>` (a tier 0
  city: open, boundary, the top bar's split, a water crossing refused, a test line to a final solve,
  T-364; a row from its table, or from the file in `ROW`, which `pack_gate.py` writes for any pack;
  `ROWS`, a list of rows, checks them all in one sandbox game, `ZONES_PER_PAGE` (8) to a page load,
  T-928; a zone whose wait timed out, the solve's `SOLVE_WAIT_S` 180, tried once more in a fresh page, T-965).
### checks/ continued: other browser checks and Python

- More browser checks (headless Chrome on a running Vite, `[url]` or `<port>`):
  `t1413-render.mjs` (renders the synthesised building sounds of `map/buildVoices.ts` offline to WAV files with each one's peak and length),
  `t2291-load.mjs` (loads a pure app module such as `game/colourPick.ts` into Node with its types stripped; imported by the t2291 checks),
  `t2421-liberty.mjs` (the trip explorer from Liberty Island to Battery Park and Liberty State Park: a No route row, "No road between these two places"),
  `t2430-station-tag.mjs` (the station tool's pointer tag shows its trips within 1 km as the pointer reaches the track),
  `t2439-hover-perf.mjs` (the trip-length graph's hover cost per pointer move; fails above 2 ms median),
  `t2444-money-plot.mjs` (the Money tab's history plots on the real save, light and dark, with a column's hover tag),
  `t2462-bus-speed.mjs` (the Speed display's colour scale for bus and ferry routes on `real-ferries.save` with three bus lines),
  `old/t1285-world-water.mjs` (the world water mask: a Long Island to Lisbon line bridged and priced over water, short chords across bays inside and outside the zones' masks),
  `old/t1413-triggers.mjs` (each building action plays its one sound at the click, and the Sound switch silences them),
  `old/t1526-steppers.mjs` (five quick clicks on the line panel's train length, trains and headway steppers each move the value by five),
  `old/t1649-big-delete.mjs` (a delete of over 50 km of track or a tenth of the network asks first; one undo brings it back),
  `old/t1656-tip-over-dialog.mjs` (the hover tag shows above the saved games dialog),
  `old/t1657-follow.mjs` (station names and attraction marks against where they belong on each frame while panning and zooming),
  `old/t1804-bars.mjs` (a city's loading bar stays on its spot through a zoom),
  `old/t1810-transfers.mjs` (the clock worker's work after a demand solve, and how long track previews wait meanwhile),
  `old/t1819-keys.mjs` (every key's dock tab, tool and elevation, and the key hints in hovers),
  `old/t1820-stations.mjs` (the station tool places on track only and snaps from 15 px),
  `old/t1835-ui.mjs` (the group pill's centred text, Start again, the keep placing hover, hotkey hint colour, the bubble icon),
  `old/t1855-groups-kept.mjs` (line groups survive a save by hand, a reload, a drag and an export),
  `old/t1872-batch.mjs` (the delete tool's hover clears on a tool change, whole-minute headway steppers, a copied line's name and symbol, the city a game opens on),
  `old/t1881-prices.mjs` (the track tool's tag over named ghosts and real hills; prints each tag, no checks),
  `old/t1928-prices.mjs` (the level price table, the 30 km water rule and the upkeep rates by level),
  `old/t1929-roads.mjs` (road crossings priced at ground level: the tag, the quote row through an export and reload, Construct paying it),
  `old/t2220-building-change.mjs` (`baseline` and `check`: a station building's changes count 0.8 in demand, and a building-free save solves to the bit as before),
  `old/t751-expand.mjs` (a refused train length plus shakes and offers "Expand all", which lengthens platforms and trains and charges them).
- Python, offline: `zone_origins.py` (zone rows whose plan origin lies outside the unit they lead with, T-1762) and `zone_new_origins.py` (the origin world_plan's rule gives for such a zone, T-1765).
- `country_preflight.py <ISO> <dry plan> [--png out]` (Python, T-1982): a country's preflight on its
  dry plan (docs/adding-a-country.md 3): FLAG lines for scattered rows, far origins, zones under
  50,000 and big FUAs' centres split between rows; NOTE lines for built neighbours within 50 km,
  reused ids' old terrain files and empty far islets; a picture of the rows. Exit 1 on a FLAG.
- `build_inputs.py [--list]` (Python, T-1738): the files a production build copies for the registry
  (vite.config.ts `writeBundle`) checked without a build, and a routed-drive place for every zone; a FLAG
  line per gap, by country. Exit 1 on a FLAG.
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
- `old/t700_walk_banks.py [id ...] [--packs dir]` (Python, T-700): the walk factors split by bank; no
  part of a zone walking to itself at over x5, and Shanghai's Middle Henan Rd to the Bund straight.
- `ceiling/` (T-1056): Anita's ceilings A and B on a scratch copy of the sim crate. `t1056_ceiling.rs` is
  the test module (`T938_SET` sets parameters; its `drive` output is the car's trip-weighted road km
  and minutes by home density, T-1147), `run_ceil.py [cities]` runs it per pack (`T1056_WORK` names
  the scratch folder).

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
  changes a pack's zones, `--check` lists stale ones; T-275). It also makes the land parts (`part`,
  header `parts`, T-2213); `routes.py <id> --stage parts` makes only those, for a finished pack,
  merged or not (into the res-9 copy and the merged pack as `nocar.py` does, or merged again where a
  merged cell would span two parts).
- `parts_run.py [ids | --from file | --all] [--workers N] [--downloads N] [--dry] [--redo] [--backup
  folder]` (T-2213): the land parts of many packs, extract by extract: downloads ahead of the readers
  into `data/raw/osm_parts/` (deleted once read; a kept `.ways2.npz` is used only when written under
  the current `routes.WAYS_RULE`), pixels per pack and extract in `data/work/parts/px/`, then per pack
  a backup to `data/work/pre-T2213/` (or `--backup`'s), the parts, its zone file's part fields
  (`market_zones.py --parts`) and a summary `data/work/parts/<id>.json`; restartable. Every pack: 67 GB
  of extracts, under 2 hours on 4 readers. Checks: `checks/t2213_islands.py` (the islands and joins of
  notes/T-2207.md 3), `checks/t2213_interior.py` (the most populous parts no road leaves),
  `checks/t2213_packs_load.py` (every pack through the sim's `zone_summary`, against an earlier run),
  `checks/t2257_links.py` (every pack with two parts or more, its extracts read again: the packs whose
  parts the current rule changes, and each part joined to another by a bridge, tunnel or causeway in
  OSM, by car, path or rail; 230 extracts, about 1 h 40 min on 4 readers).
- `merge_cells.py <ids> | --all` is every pack's last step (T-044): sparse res-9 cells into their
  res-8 parent within a zone, touching sparse children of busier parents together (T-466), and
  sparse cells still alone into a touching merged cell across the parent's edge (T-510), the
  res-9 pack kept in `data/work/res9/` with a stamp of the merged pack; `--undo` puts it back
  before any step that rewrites a pack (enrichments, routes, `intercity.py`), refusing a merged pack
  changed since its stamp unless `--force` (T-1261); `<ids> --through "<module> {id} <args>"
  [--workers N] [--force]` reruns one step on finished packs (per id: undo, the step, merge; stops at
  the first failure); `routes.py` and `packio.replace_cell_arrays` (every enrichment) refuse a merged
  pack, and `world_run`'s joint solve stops with a run fault on a member out of step; `--stats` only counts;
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
- `terrain_fine.py` (T-1854): the fine terrain intermediate, built once on AWS (`--kind terrain`):
  per one-degree block, the mean slope of GEDTM30's 30 m pixels per 1/960 degree cell and the mean
  elevation per 1/240 cell, read by HTTP range from OpenGeoHub's 432 GB file, only the tiles under the
  zone boxes, the plan's unbuilt metros and `NEXT_BOXES` (format in its docstring), with tower fields
  cut out first (`tower_ground`, GHSL's 1 km built volume, T-1881; `run` fetches it to
  `data/raw/ghsl/`, `TW_GHSL_30SS` another copy). Kept at home as
  `data/work/terrain-fine/part*.tar` and `manifest.json` (3.1 GB, never unpacked); `blocks [lat_lon ...]`,
  `run`, `check` (a block against rasterio's read). `zone_terrain.py` reads a test run's tars before
  the intermediate's with `TW_FINE_OVER=<dir>`, and the dev server serves `.terrain` files from
  `TW_TERRAIN=<dir>` before `data/terrain` (`vite.config.ts`).
- `zone_terrain.py` (T-1270, T-1854): each registered zone's terrain classes from that intermediate
  (thresholds, filter and cap in its docstring), written gzip to `data/terrain/<id>.terrain` (never
  `data/packs`, so joins do not see them); `build [ids] [--force] [--jobs n]` skips files already made
  (about 25 minutes on one core for every zone), `mix` the class shares over the files, `probe lon
  lat km`. Its `Dem` reads GEDTM30 at 240 m (`data/raw/gedtm30/`) for `market_drives.py`.
- `world_water.py` (T-1285): the world's coarse water mask, `data/terrain/world.water` (served and
  published as `packs/world.water`), from the OSM coastline polygons and Natural Earth's 1:10m lakes
  at 1/60 degree; `build` takes about 6 minutes on one core. The sim reads it as `cost::WorldWater`
  (`TrackApi::add_world_water`), deciding where no pack raster covers a point (`MaskSet`).
- `osm_water.py` is the water source for cities outside the US (OSM coastline sea plus Geofabrik
  inland water, T-041; `source(extracts)` for a config's `water_source`, `mask <id>` for a
  water-only pack in `data/work/osm_water/`; docs/adding-a-city-by-hand.md). `sea` reads the coastline
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
  `fitted_curves.json` (written by `pipeline/curves/t314_loo.py table`, T-314; `scale_to_units` the
  census totals by unit, `ghsl_people` GHSL's people per polygon); `admin_totals.py`: the generic
  `admin_totals` reader, one CONFIGS entry per country (Indonesia, the Philippines; `check`, `ghsl`,
  `dry`; T-2084); `canada.py`: Canada's adapters;
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
- `seams.py [--dry]`: zone lines through one continuous city and their commute factors, written to
  `borders.json` `seamless` for `joint_gravity` (T-1192, docs/zones.md 4); about 3 minutes.
- `plan_config.py`: a planned row's `CITIES` config, every field from the row or a rule, then its
  country module's `plan_hooks(row)` keys (T-555; `build_city.py` takes every planned row through
  it); `osm_units.py <id>`: its outline from the
  OSM admin relations matching the row's units (T-537, docs/world-run.md "A planned city"), cut along the
  existing packs whose outline box meets it (`cut_packs`; the boxes cached in
  `data/work/pack_boxes.json` by file size and time, T-963).
- `world_run.py plan|run|join|status [--rows ..|--countries ..|--limit N] [--jobs N] [--workers W] [--no-join] [--root dir]`: the
  runner (T-540): world plan rows a region at a time through download, build, gate, registry,
  joint gravity, market and clean up, each step its own `python -I` process; status and times in
  each plan row, a report a batch in `../artifacts/trainworld/world_run/` (docs/world-run.md
  "Running the pipeline"). The gate's rates test is built once at a run's start (`build_everyday`,
  T-931). Regions overlap one deep, each in its own thread (`run_regions`, T-1980), and a region's
  built rows are gated in chunks while it builds (`gate_chunks`, T-1978), builds, reads and gate
  chunks sharing the run's `Slots`. `--root` points every step at a scratch tree (`Redirect`). `RunFault`
  stops a run without parking (`check_sim`, a gate with no result); `quiet_watch` dumps a silent step's stack.
  Several runs at once (T-786): `--no-join` then `join`; row claims, extract, packs, registry and
  plan locks through `shared_lock.py` (re-entrant cross-process file locks, `write_atomic`), which
  register.py, market_zones.py, zones.py, zone_boxes.py and world_plan.py take too. A clean join
  ends with `freeze.py`.
- `netget.py [ISO2...]`: a build's network requests retried in place (`request`, `cached_json`), and the
  national figures a run fetches once before its builds (`country_inputs`, T-1979).
- `tools/aws/` (T-1064, rules and flow in docs/aws.md): long builds on a rented spot machine in a
  friend's AWS account. `launch.py` (spot, one-time, tagged, a hard timer, presigned URLs into
  `userdata/<kind>.sh`; `--az` when the cheapest zone has no spot capacity), `bucket.py` (our bucket
  only; `get` fetches objects over 100 MB with curl from a presigned URL, resuming), `status.py` (what of ours exists; `cost
  <run>`), `teardown.py` (terminate and sweep by our tags, then list), `guard.py` (a PreToolUse hook
  refusing raw changing `aws` calls); `awslib.py` holds the region, tags and checks. A wave (T-1065):
  `bundle.py` (the code tar; `--head-sim` takes sim/ from HEAD), `seed.py` (home's world field and the box's
  sea polygons, unpacked first, T-2301), `userdata/wave.sh` (check row, go, `world_run.py run --build-only` by
  runner region) with `wavepack.py` (a region's result tar on the machine), `land.py` (results into
  data/ under the plan lock, rows "built"). The fine terrain (T-1854): `userdata/terrain.sh` runs
  `pipeline/terrain_fine.py run` over a block list and puts its part tars as they fill.
- `freeze.py [--label L] | list | check [stamp] | build [stamp] [--tree] [--out]` (T-928): the data a
  production build copies, frozen in `data/frozen/<stamp>/` as hard links to data/packs plus copies
  of the four game data files, with a size and time manifest; `build` makes a publishable build of a
  tree's code on a freeze (`TW_PACKS` in vite.config.ts), for `tools/deploy.py --dist`.
- Real networks and calibration (T-007, T-509): `realnet.py net <city>` turns a city's sources
  (GTFS, OSM route relations from `realnet.py fetch-osm <city>`, or a CSV of timetabled trips;
  `CITIES` lists each city's) into a demand test network in `data/work/realnet/`; `realnet.py`
  also has New York's older targets and the save: `geom` (GTFS shapes per hop), `route` (each hop
  over the real-network graph with `realroute.py`, the track tool's router ported, lines on the same
  track set side by side and levels chosen on that geometry, T-1303), the sim's `realsave` bin (every edge marked `sep`, grade-separated from the others whatever its level, T-1751; the
  track model's network, routed hops traced with `trace.rs` at 3 m, names tidied by `track/names.rs`:
  lines lose their feed word, stations get title case without "STATION", T-1442; the sim's `load`
  runs the same on older saves), `gamesave` (the game file
  `<city>_real.save`); `ship` copies the seven a sandbox starts from to `data/packs/real/<id>.save`
  (T-731; new files, since data/frozen/ links to the old ones). Tokyo's trips file is `calib_tyo.py trips` (peak trains an hour
  by tier and `REAL_PEAK`, timed at the count point, T-1751). `REALNET_OUT` stages geom, route and gamesave outside data/. `realcorridors.py <city>` (or
  `realnet.py corridors <city>`, T-1305) routes the hops again keeping their chains and writes
  `<city>_real_corridors.json`: lines routed again off other classes' track and off full pieces
  (T-1774), parallel chains grouped into corridors, cut at stations and where the
  lines change, as many double-track pairs a piece as its trains need and OSM's tracks allow and
  the lines on each, with a picture, and (`realshared.py`) the track model's nodes, edges, platform
  tracks and line paths from them, which `realsave --shared` builds into the save (`--slow` also
  prints where trains lose time: curves, the capacity pass's holds by node, resources over 90%, and
  writes `<city>_holds.json`). Each line has its real colour (T-1770,
  `realcolour.py`, called by `net`; `realnet.py colours <city>` adds it to an older network JSON, staged
  under `REALNET_OUT`, which `realsave` reads first): the GTFS `route_color` (New York, Toronto, Paris,
  London's National Rail), else the OSM relation's `colour` tag (Sendai), else noritetsu's line colour
  (read only: London and Seoul by line id, Tokyo and N02 by line name and operator), else `realsave`'s
  palette; a colour too dark or light for either map is moved to OKLab lightness 0.40-0.90. `realnet_n02.py` is
  its N02 adapter, all of Japan's lines (cities `jpn`, `jpn7`: T-737's stress saves): each line is
  cut where MLIT's passing count changes, each piece gets cars and whole trains from its count,
  Shinkansen run on high-speed track, and `n02_trim.json` holds the cuts and omissions game runs
  asked for (T-1625, notes/T-1625.md; `realsave` reads `trains` and `hsr` from the network JSON). `calib.py
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
- Country modules, each a tier 0 adapter through `tier0.py`'s hooks like `france.py` (docs/adding-a-city-by-hand.md):
  `china.py` (Shanghai first, T-291), `germany.py` (Berlin, T-290), `india.py` (Mumbai, T-292), `korea.py` (the Seoul Capital Area, T-457), `brazil.py` (Sao Paulo, T-458), `nigeria.py` (Lagos, T-459), `australia.py` (planned rows' country keys, T-1857). `china_holidays_pattern.py` fills `china_holidays.json` with the usual pattern for years the State Council has not published.
- `enrich_all.py`'s steps, each also run alone on a finished pack: `enrich_airports.py`, `enrich_everyday.py`, `enrich_parks.py`, `enrich_schools.py` (with `education_sources.py`), `enrich_universities.py`; they share `overpass.py` (one cached Overpass request) and `arcgis.py` (one cached, paged ArcGIS REST query).
- Calibration cities beside `calib_tyo.py`: `calib_lon.py`, `calib_par.py`, `calib_seo.py`, `calib_sdj.py` and `calib_tor.py` hold each city's trips file and targets for `realnet.py` and `calib.py`.
- US helpers: `spread.py` (each census block's people and jobs over H3 cells), `wfh.py` (work from home from the ACS), `group_quarters.py` (people in institutional group quarters), `fit_decay.py` (the gravity decay fitted against LODES OD), `lumps.py` (job cells far above their neighbourhood), `look_water.py` (a look-image of a pack's water mask).
- `tools/country/` (adding-a-country.md's scripts): `backup.py` and `restore.py` (the shared files a run or join rewrites, saved and put back), `setbuilt.py` (parked plan rows set back to built), `stale.py` (market packs naming a market place with no pack), `setup_sim.py` (a sim copy for a long run), `start_hidden.ps1` and `waitany.ps1` (a detached start and the one background wait), and the census inputs of `admin_totals.py`: `turkey_census.py` (Turkey's people by ilce), `india_census.py` (India's two files), `cambodia_census.py` (Cambodia by district).

### The Northeast's build order

The Northeast's hand-built packs are `build_city.py` configs: cities `nyc` (with Mercer since
T-276), `phl`, `bos`, `wdc`, `bal`, `pvd`, `nhv`, `spf` and the zone `wor` (docs/zones.md); Hartford
(`hfd`, with Windham, New London and Litchfield since T-1113) is a world plan row. Then in order:
`zones.py` (registers built non-city zones and writes every entry's `box` through `zone_boxes.py`,
which also runs alone, T-302), `intercity.py --group auto [ids]` (every group of packs within 50 km
of each other by `joint_gravity --packs`, T-463, T-539, the Northeast's too since T-1014, a group
spanning countries or solve regions a unit at a time with `--fixed` neighbours, T-786; `--groups`
lists them and their solve units; after each unit `cross_everyday.py` draws its everyday outings
across zone lines, T-1175: the members' kept shares into their packs and res-9 copies,
`data/work/<unit>.cross_everyday.json` for the market packs),
`market_zones.py [ids]` (each zone's market pack against the world field, then `market_pack`'s
rows into app/src/game/market-rows.json and each zone's `country` into the registry through
`market_rows.py`, which also drops the rows of unregistered zones, T-776; seconds a zone, T-377;
only the rows of zones whose market pack changed, came or went and of places within 250 km of
them, from `data/work/market_rows.inputs.json`, `market_pack --only`; `--rows --full` for every row, T-929;
the joint and everyday files with pairs from the named zones only, picked by `pair_files.py`'s index,
whose `unused --move` moves files that give no zone a pair to `data/work/retired/`, T-1976),
`market_drives.py build` for new or moved places (the routed drives between places and each zone's
own detour towards each partner, `data/packs/market-drives.bin`; each place's paths cached in
`data/work/market_drives/sources/`, so only new places and places with a new or moved partner are
routed, minutes; zone centres cached in `zones.json`, OSM's car ferries in `car_ferries.npy` from
Overpass tiles in `data/raw/market_drives/`; the first run makes the world slope grid there, about 5
minutes on 4 cores, and routes every place; T-279, T-2424),
`zones.py --trips`, `routes.py <ids>` for any pack whose zones changed (T-275), and last
`merge_cells.py --all` (T-044; the chain runs on res-9 packs, so `merge_cells.py <ids> --undo`
first, or one step through `merge_cells.py <ids> --through`).

Vite serves and copies all registered packs. Zone packs stay separate; demand workers share a
regional pool. Registry market places keep NYC0/PHL1/Mercer2 (no zone since T-276)/BOS3/WDC4,
Baltimore5/Providence6/NewHaven7/Springfield9/Worcester10; 8 and 11-13 (the old Hartford, Windham,
New London, Litchfield) are unused since T-1113. Every pack's `market` is -1 since T-1014.

Toronto (`tor`, T-277) is registered with `market: -1` (outside the regional market) and built by
`canada.py` (`canada.py tor --finish` reruns gravity, purposes and checks from the cached cells).
Its OSM inputs come from the Geofabrik Ontario extract, not Overpass.

Adding a country: the playbook is `docs/adding-a-country.md` (its scripts: `tools/country/`); the runner and the plan's rule are
`docs/world-run.md`; a hand-made city or a country module is `docs/adding-a-city-by-hand.md`.
