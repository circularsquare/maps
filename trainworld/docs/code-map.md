# Code map

Where things live and how they are built. Read when you need to find a file; the rules are in
`CLAUDE.md`. Keep a line here current when you add, move or split a module.
For the big picture first (the parts, how data moves, three everyday moments), read `docs/code-tour.md`;
whoever moves a file it names fixes it there too.

## sim/: the Rust crate (`trainworld-sim`)

Compiled to WebAssembly with wasm-pack 0.15. `cargo test --manifest-path sim/Cargo.toml` runs its
tests natively.

- `src/track/`: the track model (T-040): geometry, costs, crossings, world frames (`frame.rs`,
  T-125), the network and its edits with undo (`world.rs`; blueprint and construct, T-055),
  run-time profiles, capacity (passing tracks' overtakes beside the following losses: `overtake.rs`,
  T-2231), the boarding account (`boarding.rs`: each train's load and each
  platform's crowd, T-164), each stop's service span for demand (`span.rs`, T-842), the lines related to a line for the line panel's diagram (`related.rs`, T-1168; `wasm/related.rs` builds its routes), station upgrades (`upgrades.rs`, T-1281), station buildings (`building.rs`: outlines, their members and refusals, T-2217; `wasm/buildings.rs` adds, reshapes, renames and removes them), lines run on both pairs of quad track (`twin.rs`: the hidden twin's stations and route, `twin_plan`, and why not; kept in step by `world.rs` `twin_ops`; `wasm/twin.rs` the switch and its questions, T-3333), one-track stations (`StationData::one`, `Network::one_side`, `run_passes` in `net.rs`, `Run::pass` in `service.rs`; `wasm/onetrack.rs` the panel's switch and turn around, placing in `wasm/draw.rs` `one_at`, T-3332), saves, local time (`localtime.rs`, T-309), and `wasm.rs` (`TrackApi`,
  the clock worker's API, with the drawing tool's routes and the crossover tool's edit, T-338; its
  exports are split by area into `wasm/money.rs`, `edits.rs`, `draw.rs` (routes beside track: `beside.rs`; real flyovers at the junctions a route makes: `flyover.rs`, T-3092), `crossings.rs`, `links.rs` (`rebuild.rs`: the level, track-type and track-count planner and rebuilding built track in place, rail to tram and back, single to double and back, T-3059, T-3087, T-3088, T-3296, T-3321),
  `lines.rs` (the gap a stop cannot cross: `gap.rs`, T-1896), `snapshot.rs`, `service.rs` and `upgrades.rs`, the shared helpers staying in `wasm.rs`). In the default build.
  `cargo run --release --bin track_bench` times edits on a synthetic 2,000 km network;
  `--bin lanes_parity -- <track bytes | game | game-left | mirror-right | mirror-left> <out>` fingerprints what the lanes and move tables decide (paths, routes, badges, junction inspectors, the train guard's pairs, capacity, the lines' figures), run and compared with a baseline by `checks/t3137-lanes-parity.mjs` (T-3137); `mirror-left` is `mirror-right` (the small game without quad) laid mirrored and keeping left, compared by `checks/t3297-mirror.mjs` (T-3297);
  `--bin edit_bench -- <track bytes>` times headway and count edits on a real save's network (T-1283; `TW_DUMP=<file>` writes `lines_info` at load, after the edits and after a save and reload, T-1787; `TW_LINES=<file>` a table of each line's rates, round trips, delays and trains at load, to compare two builds, T-1924);
  `--bin branch_bench -- <track bytes>` times the drawing tool's previews of branches off it (and with `TW_ADD=1` placing them, T-1779);
  `--bin boarding_bench -- <track bytes>` times the boarding account's steps on made-up riders (T-2699; `TW_DUMP=<file>` writes every line's account readouts, to compare two builds).
  `src/heap.rs`: on wasm32, the modules' allocator: dlmalloc over a system side that grows the memory in big steps, methods never inlined (T-1594, T-2699, T-2716); the clock worker's module's global allocator.
  Station placements swept along every piece, refusals counted by reason and by the joints under the platform: the ignored tests in `wasm/station_sweep_tests.rs` (a drawn network, and a save's network bytes in `T1964_NET`; T-1964).
  notes/T-040.md. Methods only tests and bins call (`set_line`, `set_edge`, `to_lonlat` and the
  like) sit in plain `impl` blocks, not in the WASM exports; the same holds for `DemandApi`,
  `EverydayApi` and `IntercityApi`. A big file's tests are in its own `<file>/tests.rs`.
- `src/track/bus.rs`: bus lines (T-1464, T-1465): the route (`BusRoute`: points with corner and road bits; `Poly` in road metres), its stops (`plan`: stations met, ends, crossings, about one every 2 km on the 400 m street grid, each with its stretch of route; `plan_spaced` for a line with its own spacing, T-2885), the trip by the road it crosses (`Pace`: the car's model, 90% of it, 28 s a stop, T-2904, constants in `streets.rs`; `profile`, `spaced_profile`); `wasm/bus.rs` its exports, tests in `wasm/bus_tests.rs`; notes/T-1464.md.
- `src/track/ferry.rs`: ferry lines (T-2193): ports (stations with a berth), the route the app searched (`FerryRoute`: points with a kind and a speed), its check, the boat's speed limits and its trip with acceleration (`limits`, `profile`), its stops as a bus `Plan`; `wasm/ferry.rs` its exports, tests in `ferry/tests.rs` and `wasm/ferry_tests.rs`; Node check `checks/t2193-ferry.mjs`; docs/track-and-service.md 6.8.
- `src/track/streets.rs`: the car's street speed by place (demand's `place_min_per_km`, kept equal by a test), a pack's grid of its zones' speeds (`StreetGrid`, `Streets` in the network, installed with `TrackApi::add_pack_streets`, T-1577), for a bus's speed.
- `src/track/cost.rs`: prices (SPEC 6.4), the water mask types and the terrain classes
  (`ClassGrid`, `Terrain`, `terrain_spans`, T-1270, T-2901: one grid per degree block, by `block_key`; `BlockIndex`,
  `take_wanted` for what was asked for and not installed), installed with `TrackApi::add_terrain`.
- `src/track/trace.rs`: a real railway's line as corners within a tolerance (T-727c), for
  `TrackApi::trace_route` and the real-network snap; `Tracer` runs it in steps (`RouteTrace` in
  WASM, T-754); notes/T-727.md.
- The rest of `src/track/`: `net.rs` (the network: nodes, edges, stations and lines as flat arrays by stable id, with everything derived from them), `lanes.rs` (the tracks inside a piece, `Network::lanes`, and each node's move table, `Moves`, which routing, line checks, the flat-junction badge, junction capacity and the train guard read; the way single track runs, `edge_way`, from its run's joins to double track, `derive_ways`, `join_away`, T-3137; what turns around when the running side does, `flip_reversals`, T-3297, with `TrackWorld::flip_right_hand` in `world.rs` and `wasm/flip.rs`), `geom.rs` (alignment geometry: points of intersection with an arc fitted at each, straights between, ramps), `cross.rs` (where track crosses track, on a 1 km grid of edges), `bridge.rs` (a ground-level segment over water becomes a bridge, T-656), `roads.rs` (the road lines ground-level track crosses, fed tile by tile, T-1929), `capacity.rs` (resources, utilisation per demand level, Kingman's delay, holds), `profile.rs` (run-time profiles along a line's path under speed limits), `spacing.rs` (the phase of each line's departures, T-1925), `timing.rs` (each train's own timetable: a local's waits at passing stations and an express timed behind its local, worked out train by train after every settle, `retime`, T-3346 stage 2), `save.rs` (the network part of a save), `names.rs` (tidies station and line names from a real network, T-1442), `params.rs` (every tuning constant of the track model), `bench.rs` (the synthetic network `bin/track_bench.rs` times). Test modules sit beside their file (`frame_tests.rs`, `merc_tests.rs`, `related_tests.rs`, `transition_tests.rs`, `wasm/*_tests.rs`).
- `src/diagram.rs`: the network diagram's carry solve (`CarrySolver`: conjugate gradients with the black box multigrid,
  both axes as one f64x2), bit for bit `app/src/diagram/sheetLines.ts`' JavaScript one; behind feature `diagram` (on by
  default, so in the track module); the diagram worker loads it (T-2589, notes/T-2589.md).
- `src/demand/`: the demand kernel, behind feature `demand` (on in the app's build): `api.rs`
  (`DemandApi`, the demand workers' API), `intercity.rs` (the regional market: its types and pair rules, with `intercity/logsum.rs` the rail logsums and sparse tables, `costs.rs` drive, ferry and air costs, `market.rs` the market's methods, `build.rs` building a market and `MarketPacks`, `api.rs` `IntercityApi` and its access lists, `pass.rs` the pass and `pass_window`, `setup.rs` serving stations and the network, `exports.rs` the wasm calls, `selection.rs` the selection and corridor outputs; `intercity/market_trip.rs` its trip explorer query, T-866; `intercity/drives.rs` the routed drives between places and the ferry, T-279; its pass goes through a window of destination zones,
  `pass_window`, searching destinations as the window needs them on a big cluster, `pass_searched`
  with `kernel::DestSearch` and `DestAssign`, T-761, T-1545; a window of zones' pairs alone, `set_window`
  and `window_loads`, for an edit's local solve, T-1735), `warm.rs` (the
  everyday warm-up demand worker 0 runs at startup, T-173), `days.rs` (Saturday's, Sunday's and a superholiday's away, going-out and coming-back days' factors per
  purpose and period and the lean's sign, T-260, T-262, T-264, T-1373), `routes.rs` (routed walks over water and drives over OSM roads from
  the pack, T-275), `trip.rs` (the trip explorer's query, T-195; its two cells' own trip, T-720, checked with the
  walk bands in `t720.rs`), `lanes.rs` (f32 lanes, wasm
  SIMD or SSE2, for the commute pair loop, T-036). The commute pair loop is `kernel::mode_choice`;
  its small zone pairs are dealt in chunks there (`COMMUTE_CHUNK`, T-874; `DemandApi::set_chunked`).
  `wasm.rs` (WASM only): the memory and heap figures, the last panic, and the counting allocator over
  `heap::Heap`, which grows the memory 32 MB at a time (T-1205, T-1174, T-1594, T-2716).
  `lod.rs`: a zone's level of detail (T-1222), the everyday sampling and commute chunk a zone opens
  at (`DemandApi::at_level`); its checks in `everyday/lod_tests.rs` (`t1222_probe` measures memory, times and bias).
  Zones of two sizes (T-852: `kernel::coarsen`, `Zones::size`): every zone-pair table goes through
  `kernel::table_layout` and `Gravity::base`/`key` (`PairKey` elsewhere), near pairs through
  `Zones::near`, a square's zone through `Zones::zone_at`, centre distances through `Zones::d2_m`.
  `calib --cells --exact` dumps every figure unrounded, for before/after checks.
  `cargo test --release --features demand nyc_day -- --ignored --nocapture` solves a day on the
  real pack natively. notes/T-026.md. `cargo test --release --features demand --lib t2214 -- --ignored --nocapture`
  checks the land parts in local demand on the real packs (`api/islands_tests.rs`, T-2214).
- The rest of `src/demand/`: `kernel.rs` (the kernel's `Params` and parts, with `kernel/` one file per stage: `gravity.rs` zones, coarsening and gravity, `access.rs` station access per cell, `subzones.rs`, `search.rs` the network graph and station-to-station search, `mode.rs` mode choice and the commute pair loop, `assign.rs` assignment, `stalls.rs` crowding, riders left behind and the trains' room), `everyday.rs` (the home-based outings layer, `EverydayApi`; `everyday/drives.rs` its car drives), `pack.rs` (the city pack format and its reader), `network.rs` (the player's network as the kernel sees it), `synth.rs` (a synthetic New York-sized city and hand-made network), `lengths.rs` (a zone's trips by straight-line length and mode, for the trip-length graph, T-2365), `bench.rs` (the kernel stage by stage with timings and heap, run by `bin/demand_bench.rs`), `alloc.rs` (the counting allocator, native only), `clock.rs` (milliseconds since a start, native or wasm), `cross.rs` (everyday outings across a zone line, offline).
- `src/bin/`: native tools behind feature `demand`, built in `data/work/target`: `calib`
  (T-509: a weekday on a city's real network with a parameter set, for `pipeline/calib.py`),
  `realsave`, `pack_gravity`, `pack_walk`, `market_pack`, `joint_gravity`, `zone_summary`, `demand_bench` (the kernel's timings, `demand/bench.rs`),
  `start_difficulty` (T-2690: each start's trips with both ends in circles round its centre, for `pipeline/start_difficulty.py`),
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
  diagram puts on top, T-1894; `lineReversed`, which way around a left-to-right strip or chart of a line reads, west end first, T-2661).
- Time: `clock.ts`, `lineFigures.ts` (the 4x clock cap while the player's edits are not in the figures, T-1665, T-2409, the same from every view, T-2622; `checks/t2622-cap-view.mjs`), `editCap.ts` (which edits cap it, T-1803; `checks/t1803-cap.mjs`), `localTime.ts` (each region's local time and the local time in view, T-309),
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
  purposes as shown, T-192), `trip.ts` (the trip explorer's state), `transitModes.ts` (the mode split bars' transit
  part by kind of line, T-3695).
- Places and frames: `coords.ts` (world metres to lng/lat), `rings.ts` (ring box, point-in-ring and near-ring tests for the real-network snap and its router), `frame.ts` (the world frames, and the earth's radius and metres per degree) and `registry.json` (the
  one city registry, read by `cities.ts`, T-125; since T-776 the bundled index
  only), `market-rows.json` (each zone's world market row, fetched by `workers/demandClient.ts` beside
  the first market pack and sent with it to the solving workers, `cities.ts` `marketWorld`), `places.ts` (station and
  compass names for the inspectors), `placeList.ts` (each city's OSM place list, fetched per zone by
  search or station naming, the view's zones first, T-177, T-065, T-400), `boundaries.ts` (each city's boundary file, fetched
  once, T-172), `zoneNames.ts` (zones' shown names and search aliases from the hand-edited
  `zone-names.json` and the generated `zone-aliases.json`, `pipeline/zone_aliases.py`; T-825),
  `zone-successors.json` (each removed zone id's successor, for old saves, `pipeline/zone_successors.py`; T-1053),
  `startDifficulty.ts` and `start-difficulty.json` (each start's tier, easy, medium or hard, from `pipeline/start_difficulty.py`, and `lineSetUp`, whether a line runs yet; T-2690),
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
  zoomed-out dot groups; T-217, T-249), `complexLead.ts` (the member a complex goes by: a typed name, then
  most lines; the complex's name, `nameSource`, `complexTitle`; check `checks/t2683-keep-names.mjs`, T-2683), `stationBinding.ts` (the
  setting for new stations' names, T-3759), `links.ts` (the player's links, T-197), `lineActions.ts`
  (new and copied lines, Edit stops, construct blueprints: the N, L and C keys, T-670), `colourPick.ts`
  (the automatic line colour pick as pure functions: palette pick, CIEDE2000, the lines beside a line, T-2291;
  `lineColour.ts` re-picks a new line's colour after its stop edits; `checks/t2291-line-colour.mjs` in Node and
  `t2291-line-colour-game.mjs` in the game), `newHeadway.ts`
  (the headways every new line starts with, saved with the game, T-3543), `vehicle.ts` (a line's vehicle: train, tram, bus or boat, its noun and size, and a name's mode word dropped before the noun, T-3054), `lineShape.ts` (the bullet shapes and each mode's default, pure) and `lineShapeState.ts` (each line's own shape, saved with the game, and `shapeOf`, its own else its mode's from Settings; T-3319), `lineVisibility.ts` (the lines hidden on the map, saved with the game, and the levels hidden while the page is open; T-3420), `newLine.ts`
  (what a new train, bus or ferry line starts with: trains or headways, dwell, turnaround, T-2919), `remove.ts` (removing things, with the question asked before removing
  anything constructed, T-096).
- `realGraph.ts`: a region's real-network routing graph (`packs/realgraph/<cc>.rgr`, fetched once
  and decoded), for snapping track to the real network (T-727); read by `workers/realRoute.worker.ts`.
  `realFollow.ts`: the graph's kinds and what each track type follows (T-3069), for the router and the layer.
- Account (T-972, T-973): `account.ts` (the optional anita.garden account, noritetsu's, and the
  leaderboards: which server, the session, each slot's game id and its score sends, the boards;
  shonei-server's `trainworld.go`); its panels `ui/AccountPanel.tsx` and `ui/Leaderboard.tsx`. T-3269, the flag beside
  a name: `flags.ts` and its list `flags.json`, `ui/Flag.tsx` (the picture, a name with its flag, the account panel's
  flag button and picker), `public/flags/` and `tzCountry.json` (time zone to country) written by `tools/flags.mjs`;
  check `checks/t3279-flags.mjs` (T-3279).
  T-2887, Community: `community.ts` (share, lists, like, report, the `?share=` link, View), `ui/Community.tsx` (the
  modal, the Share button and dialog), `diagram/shared.ts` (a schematic in the server's drawing format and back),
  `sharedSave.ts` (T-3115: a save going up with a share, and a stranger's checked before the loader); Anita's side is
  `tools/feedback-inbox/`.
  T-988, games on the account: `accountGames.ts` (the main thread's side: the account's list and
  pictures, the start menu's settling, downloads and deletions, the sign-in question) over
  `workers/cloud.ts`.
- `quotes.ts`: the construct-all quote and a planned line's price, asked of the clock worker when a panel shows them (T-1171; the line panel's button is `ui/inspector/ConstructButton.tsx`), and the blueprint priced piece by piece for the cost breakdown (T-2269).
- `blueprintStretches.ts`: the cost breakdown's parts and stretches from those pieces, the row under the pointer (`bpHover`) and a row's off switch (T-2269).
- Other: `money.ts` (cash and ledger from the worker, fare income from riders, T-028; `baseTrips`, each line's share of the base fares, T-2104; `moneyTerms`, `moneyWorld` and `lineGeo`, what income, points and region shares read, kept while unchanged so a state redoes only what it changed, T-1209), `fareInEffect.ts` (each line priced at the fares its riders were solved with, so a fare edit pays only once demand settles, T-2266),
  `fareCurve.ts` (the fare curve through the player's three nodes, the first the base fare at 0 km: fare at a length, the first stretch's base and rate, the knots the sim takes, old saves' lines as three nodes, T-883, T-1112, T-1143), `income.ts` and `countryIncome.json` (fares by national income, T-1277: a fare dollar's perceived minutes by country, each country's fare scale by income or the player's, GDP per head at PPP from notes/T-1277/income_table.py, and at market rates (`usd`, the one the rule reads) and the car's running cost a km (`car`) from notes/T-1521/money_table.py, T-1521; T-1906: the start country's income scale (`START`), each country's default over it, and the panel's scale as a multiple of the default; T-1286: the market's fare factor per country; T-1431: the market's three income groups per country from the quintile shares `quint`, notes/T-1431/quintile_table.py), `fareEstimate.ts` (the Money tab's estimate of rail trips and fare income a day for a dragged or unsolved fare curve, from demand's fare sensitivity, T-1300), `fareUndo.ts` (fare changes' undo and redo while they are the last edits, handing over to the sim's, T-1130),
  `moneyTiming.ts` (each game hour's fares, rides, points and running cost from the period tables, T-736),
  `regionMoney.ts` (each line's and each construction's split between regions, the region ledgers, T-531), `stretchMoney.ts` (money by edge and track stretch for the Profit track display, T-2420, and by bus or ferry stretch, T-2452), `points.ts`
  (each line's points per hour by region, points a day, each region's forecast weekday points, the score, the star's mark, T-684, T-685, T-694), `achievements.ts` (the achievements' tiers, firsts, their figures and words, posting the steps met to the clock worker, T-2124) and `continents.ts` (each country's continent, T-2124; their panel `ui/inspector/AchievementsView.tsx`, the medal and toasts `ui/Achievements.tsx`, styles `ui/achievements.css`, check `checks/t2124-achievements.mjs`), `realShare.ts` (the real networks' train shares the city view's Within bar marks, a table from `checks/t2145-real-share.mjs`, T-2145), `issues.ts`
  (the track model's refusals in plain words), `editNote.ts` (what the last edits did to train trips and profit a day, shown under the view rows at the map's upper left once demand settles, T-2125, T-2237; drawn by ui/EditNote.tsx; `checks/t2125-edit-note.mjs`, `checks/t2682-editnote.mjs` for edits in a new game still loading or shown coarse), `profitEstimate.ts` (the network's profit change a day from a trains, headway or cars edit before demand solves it: exact running cost, fares by the wait term, the edit note's estimating state, T-2626; `checks/t2626-profit-estimate.mjs`), `openingDay.ts` (the lines on their opening day: the Lines tab's riders count-up and a "bung", not saved, T-1434; `checks/old/t1434-opening.mjs`), `openedZones.ts` (the zones the game has started in, reached or crossed with track, kept for the save, T-729, T-801), `trackZones.ts` (the zones built and blueprint track crosses, by coarse outline, from each network state, T-1047), `loading.ts` (what is loading, the startup gate and
  the clock hold, T-149, T-172; whether the start menu is up, T-677), `loadBar.ts` (the loading bar's
  stages, their shares and texts, each slot's remembered stage times and this browser's pace, T-1195;
  the time left's count and text, T-1479), `editLeft.ts` (an edit's solve's time left for the status tag's hover,
  from the workers' `work` and the last edit's pace, tail and gap, T-3599; `checks/t3599-time-left.mjs` against the
  real wait, `checks/t3599-replay.mjs` a recorded wait replayed with no browser),
  `memoryCheck.ts` (a save's memory estimate and the refusal when it will not fit, T-1197),
  `saveSafety.ts` (persistent storage, T-1214), `busy.ts` (the busy tag's
  text, and work held a painted frame when it should take over 150 ms here, each kind's time per
  unit learnt as it runs; the clock client's big states, T-1200), `palette.ts`, `shades.ts` over `shadeTable.ts` (the light and dark themes' map
  colours: basemap, line casing, map ink, blueprint off, outside shade, thumbnails; the table has no
  page behind it so the clock worker reads it, T-039, T-543), `format.ts`.
- Smaller modules: `capacityKinds.ts` (the words for each kind of capacity limit, for the map's dots, the Trouble list and the line panel's alert), `carryLoads.ts` (a line's riders per stretch carried onto an edited stop list, T-1419), `fareDefault.ts` (a country's default fares in round numbers, T-1662), `fonts.ts` (loads the Nunito faces; Zen Maru Gothic's slices are `ui/zenmaru.css`), `lineLetter.ts` (the chip text of a line with no symbol), `lineRunning.ts` (which lines have no trains at the level now, T-1792), `listScope.ts` (what the Lines and Stations tabs list: the viewed place or the world, and the sort by distance from the screen's centre, T-332), `playtime.ts` (the anonymous plays and playtime ping; `?track` and `?notrack`, T-1771), `splitTip.ts` (the hover of a mode split bar and the city view's split table), `today.ts` (the past 24 hours' riders, boardings and fares for the line and station panels, and the crowd grades, T-446, T-2236), `upgrades.ts` (the station upgrade bits, their texts and a passage's price, T-1281), `worldTotals.ts` (the world as one place: the cities' sums for the Cities tab's first row and the top bar when zoomed far out, T-1110).
- The `WorldView` is built from the clock worker's states in `workers/clockClient.ts`.

### src/i18n/: text by language (T-2583, docs/localization.md)

`index.ts` (`t` and `tRich` over the ICU messages, the `language` signal, `setLanguage`, the `tNumber`,
`tList`, `tDate`, `tCountry` helpers), `intl.ts` (`LANGS` and the cached `Intl` formatters by language,
worker-safe), `messages/` (`en.json` bundled, the other languages fetched), `context.json` (a note per key),
`glossary.json`, `sources/` (the English each translation was made from), `places.ts` (`tPlace`: a registry
city's name in Japanese or Chinese from `place-names.json`, written by `tools/place_names.mjs`, T-2681).
`game/remove.ts` is the worked example.

### src/map/: the map and what is drawn on it

- `mapView.ts` (the MapLibre map; its scroll zoom told wheel gaps by event time, T-675; the flat map or globe setting, T-033), `basemap.ts` (the Positron style: warm paper repaint, three road weights, light or dark, arrows on one-way streets while the bus tool is on and Settings makes buses obey them,
  labels in Zen Maru Gothic from `public/fonts/`; the Terrain hillshade, T-1270), `demFlat.ts` (the hillshade's tiles with the relief over flat-class ground blurred away, in `workers/demFlat.worker.ts`, T-1881), `terrainTint.ts` (the terrain classes' wash while building, from `game/terrain.ts`'s grids, in the hexes track is charged by, T-1270, T-2890), `canvasRead.ts` (whether canvas reads are made up, as in Firefox resisting fingerprinting, and a PNG decode and encode without a canvas for that case, T-3705), `realStations.ts` (the real stations: their rings and hover on the Real network layer, T-3570, and the name a station placed on one takes, T-3571), `realNetwork.ts` (the Real network reference layer, with the freight and abandoned archives built by `pipeline/real_extra.py`, T-3464: noritetsu's
  PMTiles per region, read from `../noritetsu/` or R2; vite.config.ts proxies `/noritetsu` to 8800 in dev, T-682),
  `styleReady.ts` (`styleUp`: MapLibre style calls are skipped while it has no style, a lost context, and reapplied on `style.load`, T-957),
  `moving.ts` (what runs at most every 100 ms while the camera moves: `PlaceThrottle`, T-1207), `lite.ts` (the lighter map and its auto switch, T-1207),
  `overlay.ts` (the overlay canvas and frame loop,
  notes/T-015.md), `renderer.ts` (WebGL2 track, stations, trains; the strokes kept between changes in `strokeCache.ts`, T-1638, and simplified and culled in `strokeLod.ts`, T-1639, the few that show gathered into one buffer in `strokeGather.ts`, T-1644; its shaders and their shared constants are in `rendererShaders.ts`, the platform crowds' sums in `platformCrowds.ts`), `gl.ts` (shader compile, uniform lookup, zoom interpolation, the GPU timer, canvas sizing, and the camera: `Camera` for the flat map or the globe, `PROJECT_GLSL` and `setCamera` for the shaders, `mercToCss` and `clipToCss` to CSS px, `behindHorizon`; T-033; shared by the renderer, `demandOverlay.ts`, `gradeMarks.ts`, `transferBubbles.ts`, `shadeLayer.ts`, `overlay.ts` and the 2D overlays), `network.ts` (the renderer's
  buffers, built in the clock worker from per-edge, line and station pieces kept between edits,
  T-496, and posted as changes to the last, `NetPatch` and `mergeNet`, T-1029; moved to a new draw anchor by `shiftNetCache`, T-1170), `lineBands.ts` (lines sharing track side by side: each edge's order of its lines, the ribbon at each place, and each line's sideways shift along its path as knots, smoothed along the line, T-1799, T-1848, the lines of one ribbon change easing at one place, T-2140; the Traffic view's shifts by riders, `trafficBands`; after an edit only the lines whose inputs it touched are worked out again (the edges they read and the network's pair rules they read, so a pair rule that changes redoes only the ribbons that used it), `BandMemo`, T-2238, which `?patch=check` compares with a whole pass), `corridors.ts` (every edge with lines sampled every 25 m, each sample's edges running beside it within 20 m, mirrored, kept between builds, T-1848, an edge's samples the same object while unchanged, T-2238), `strokePost.ts` (how the track and line strokes are posted: whole with room after the last segment, unchanged, or only the new segments written over the page's arrays, and the page's merge; T-1381), `geo.ts` (the draw anchor, T-307; `pxPerUnitAt` and `unitPerPxAt` for the zoom's scale), `anchorVersion.ts` (its version as a signal, T-317), `visibility.ts` (the lines and levels the player hid, as the renderer's shaders, the station boxes, the grade marks and the picks read them, T-3420),
  `live.ts` (the live map objects for UI code, set by main.ts; `hiddenOnGlobe` for DOM marks, T-033), `synth.ts` (stress network),
  `markData.ts` and `gradeMarks.ts` (the crossing, junction and ramp marks: their data, and
  their draw pass, hit test and tag; T-629, T-655, T-1871), `oneWayMarks.ts` (one-way loops' marks: a train
  line's chevrons, made in `networkFromState` as `NetworkBuffers.oneWay`, and their pass; the spacing a bus
  route's arrows in `busLayer.ts` share; T-2991), `stationLife.ts` (from zoom 14, arrivals found
  by a timer of eight samples a second, feeding `stationSound.ts`, the WebAudio chime and pops, and
  `faces.ts`, the rising faces on a 2D canvas of its own; T-686; `faceTones.ts`, their skin tones
  and hair colours by country, T-1907), `buildSound.ts` (which click
  plays which building sound, T-1282, T-1474) with `buildVoices.ts` (the synthesised bung, T-1413, the load chime, T-1655, and the sample player),
  `samples.ts` (the recordings in `public/sounds/`, decoded when the context opens, T-1474), `audioOut.ts` (the one audio context
  all share), `transferBubbles.ts` (riders changing trains walking between stations as bubbles, drawn by the renderer after the stations, T-1124).
- Stations: `stationGeometry.ts` (platform shapes along the track for drawing, picking and the
  ghost, T-189, T-254; the waiting room annex, `stationAnnexes`, T-1416), `stationSide.ts` (the side a station on
  single track draws its one platform on, T-3540), `facilityPlace.ts`
  (where each complex's one waiting room and car park go, in the clock worker, T-1528; kept per complex between builds, T-3803),
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
  `roadGraph.ts` (`joinRoads` putting back the junctions the tiles simplify away (T-2873) and keeping short service links (T-2882), the road graph from those tiles and its A* search, one-way streets only the way they run when `oneWay` is set, T-3480), `roadTiles.ts` (the vector tile reader, including `oneway`; `roadOneWay.test.ts` runs with `npx vite-node`).
- Ferry routes (T-2192, track-and-service.md 6.8), in `src/ferry/` beside `map/`: `waterTiles.ts` (a zoom 12 tile's water polygons and
  bridge lines), `route.ts` (the water raster with bridges filled, port snapping, the leg search, coarse
  for long legs with fine ends, the line's rounding and speeds, `runSeconds`; pure, shared with the check),
  `ferryWater.ts` (`FerryWater`, the tool's client: `port`, `probe`, `line`), run in
  `workers/ferryWater.worker.ts` (fetches and keeps the tiles, each leg's search and rasters).
- Ferry lines (T-2194): `ferryDraw.ts` (the ferry tool: ports placed or shared, waypoints on shift click, T-3809, the route from `FerryWater`,
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
  second click's route, T-727d; only where a graph covers the click, `covers`, T-913), `streetSnap.ts` (tram track's snap to streets, T-2927: `BusRoads.tramLeg` and `RoadGraph.turnPath`, then `trace_kind` in the router worker), `pick.ts`, `cellIndex.ts` and `snapIndex.ts` (grids built in idle time, at once for 20,000 items or more (T-2404), that narrow a pick
  and the build tools' snap to what is near the pointer, T-724, and carried over an edit, T-1030), `tools.ts` (drawing track, placing stations, the line tool; on touch the tap, preview and confirm and the control bar's place, T-1260), `touch.ts` (whether the last press was a finger, and what the touch control bar offers), `stationSnap.ts` (which overlapping track a station goes on, the nearest spot with room for the platform, T-671, T-1870), `edgeEdit.ts`
  (dragging blueprint track's PIs, T-064, T-092; a double click adding a point, T-2951), `nodeDrag.ts` (dragging blueprint nodes, T-093; a platform's end, T-1869; `endDrag`, a drag's cleanup), `dragHandles.ts` (where the platform-end handles are, for the markers and the drag, T-1869; `NODE_HIT_PX`, a node's reach for a press), `endMarkers.ts` (loose-end and draggable-point markers, T-416), `screenPlace.ts` (`placeAt`: every DOM mark placed through the overlay's projection, hidden where its point has no place on the screen, T-1777; `marksLayer`, a tool's layer of marks), `curveCircle.ts` (the circle of the segment being drawn, T-418),
  `blueprintBox.ts` (the Select tool's drag box, T-333, and dragging its pick, T-475; its state is `game/blueprintPick.ts`; the pieces a rebuild would refuse outlined in red, T-3059, and the track beyond the pick its stations ramp on, T-3241, from `game/rebuild.ts`; plain selecting's Shift+click on the same pick, `shiftClick`, T-3746),
  `busStops.ts` (a by-hand bus line's stop editing, T-2891; it snaps to stations and, by `game/busLinks.ts`, to other bus lines' stops within demand's merge, T-2992), `stopMarkers.ts` (the toggle markers of stop editing, T-233; the gap to a station they cannot
  reach, drawn by `gapFlash.ts`, T-1896), `reversalMarks.ts` (the icon over
  stations where a line reverses partway, T-797).
- Trains and trips: `trainGuard.ts` (trains in view held behind the train ahead on their track, drawn late by a lag per trip, T-1965; zoomed in, waiting at flat crossings and crossing junction moves by their timetabled times there, T-1966, from `NetworkBuffers.conflicts` = `TrackApi.guard_conflicts`; at every zoom, opposite directions taking two-way single track in turn, T-3202, from `NetworkBuffers.singles` = `TrackApi.guard_singles`, with the trains at a station on single track on its two drawn tracks; its line paths are `NetworkBuffers.pathTable` and the like), `terminalBays.ts` (where a train stands at its line's ends: both platforms of a double-track dead end, a terminal's two sidings, a single-track station's two tracks; the pick made in the clock worker from the timetable, drawn by the train shader and held by the guard, T-2980; and `singleSplitEnds`, where two-way single track is drawn splitting into two through a station, T-3212, which `network.ts` draws the track, lines and trains from), `trainFollow.ts` (a selected train followed onto its next trip, T-245), `cameraFollow.ts` (the camera follows the selected train, from the overlay's `afterDraw`; `checks/t3361-follow.mjs`, about 2 minutes, T-3361),
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
  the track tool's tag, T-526, which `placementTag` builds; `costTint`, a cost multiplier's colour there and in the Build tab, T-1159; the curve labels of the drawing tool and the three drags, `placeCurveLabels` and `CurveLabelPool`, T-417), `cityBoundaries.ts` and `boundaryGeometry.ts` (the metro outlines, one
  layer for the zones near the view, exact ones fetched only above `ZONE_LOAD_ZOOM`, and the outside
  shade, T-131, T-139, T-400, T-871; on desktop every zone's coarse outline is drawn from `coarseEdges.ts`, the packed
  file plus each point's partner across a shared border, worked out once by `workers/outlineTopo.worker.ts`, T-3491;
  a phone holds only the coarse zones near the view in `OutlineEdges`, T-2794; nothing is drawn behind the start
  menu), `coarseOutlines.ts` (reads
  `packs/outlines.json`, which `pipeline/zone_outlines.py` writes: every registered zone's coarse
  outline, read once for the outlines and the realms' shade, T-800, with the sea off each coast from
  `pipeline/zone_water.py` (its windows over `--jobs` processes and cached in
  `data/work/zone_outlines_cache`, T-1977) and that sea alone (`coarseWater`) for the focused zone's shade, T-1145; `inOutline` is the point-in-outline test (a cell of the outline's grid no edge meets answers alone, `cellSays` in `workers/cityReach.ts`, T-1348; else its bands of latitude), for realms and the track's zones; it also gives `cityWithin` a place's zone by these outlines, and `readOutlinesForPlaces` is the clock worker's lean copy for that, T-1265; the file is fetched and parsed by the page alone and kept only as `PackedOutlines` (typed arrays, decoded a zone at a time, T-2794, T-3491), and the clock and shade workers get a copy of it, T-2802), `zoneLoading.ts` (zones still loading drawn grey with a loading
  bar, T-343), `realmOverlay.ts` (the realms not shown: one world shade with the shown zones cut out
  by `boundaryGeometry.ts`'s `ShadeCut` as triangles in `workers/shade.worker.ts` (T-1026; a worker per cut, ended with its answer, decoding only the zones a cut names, T-2794, T-3493), drawn by
  the custom layer `shadeLayer.ts` (T-1025), and a button each, T-975, T-1006), `zoneFocus.ts` (after each move: the focused and viewed zone, the nearby zones, and
  `zonesLoadForView`, the zoom test of `game/zones.ts` `ZONE_LOAD_ZOOM` that decides whether zones
  load for viewing, T-276, T-318, T-871).

### src/diagram/: the network diagram (docs/ui-diagram.md)

`graph.ts` (complexes, each line's chain with the stations beside its track, edges with their lines and the stroke each is drawn in there, T-2405;
built in the clock worker while the diagram is shown, T-2159), `layout.ts` (A, B with side terms, `sidePairs` T-2290, hops parallel on the ground `parallelHops` T-2520, segments, the snap,
routing, line order, the water kept open `seaTerms` T-2541; `LAYOUT` from the distortion dial `DISTORT` and `dialled` T-2556, `LAYOUT_R2`; run in `workers/diagram.worker.ts`, T-2160), `blocks.ts` (city blocks:
found, placed between A and B, held on a rebuild, and each block's real-to-drawn fit, T-2407), `borders.ts` (the zones'
borders, decided in the drawing: a grid round the lines and pills, each cell the zone of its nearest pill, its
boundaries fitted at 0, 45 and 90 degrees, each shared border once, T-2464), `waterReal.ts` (the water on the
ground from the zones' water masks: the sea and land masses (`groundSea`, which the layout reads first, T-2541),
rivers thinned to centre lines, their crossings with the hops' tracks),
`water.ts` (the type of the drawn water; the old drawing of rivers and sea in the layout, `waterSea.ts` and
`borders.ts`' pill regions were deleted in T-2590), `regions.ts` (the network's regions, one sheet each, and their drawings put together and placed, T-3448; the clock worker builds a graph per region), `sheet.ts` (the rubber sheet, the diagram's default: a quadtree lattice
stretched by multigrid so the network has room, `rubberSheet` and `sheetWarp`, T-2568; the cut T-2609 and the floor's fall from the networks T-2625), `sheetLines.ts` (the layout's
octilinear stages run from the sheet, the map carried with the stations, and its sea and zone borders, `sheetLines` and
`sheetMap`, T-2567; `sheetLayout`, laid out again round stations a routed river finds on the wrong bank, T-3358; its carry's solve in WASM through `carryWasm.ts`, T-2589), `thin.ts` (buses and ferries fitted
thin into a laid-out region through its carried map, T-3360; a ferry over the drawn water through `ferryWater.ts`, T-3824), `busGraph.ts` (buses and ferries with no rail near, a graph
of their own, T-3360), `sheetThin.ts` (the sheet's map before its coasts are traced: thin water and land drawn again as
bands, nibs smoothed, T-2580), `riverRoute.ts` (a city map's narrow rivers routed after the layout, one band a
course, stations held on their banks, T-3358), `coastRound.ts` (the coast's small features taken out before its
corners are rounded, and fillets where bands meet the sea or each other, T-3833), `lsqr.ts`
(sparse least squares), `fmath.ts` (atan2 and powers from + - * / and sqrt, so every engine lays a graph out alike, T-2255), `draw.ts` (ribbons, rounded bends and pills, pieces going out of a node the same way drawn as one bundle (T-2405), made in the layout worker, painted
on a 2D canvas, the focus faded and `pickAt`, T-2163), `names.ts` (station names placed per zoom level
in the layout worker, painted, and `nameAt`, T-2162), `cityNames.ts` (a name per city block far out, centred on its own
city's stations, grey under the lines, T-2438, T-2521, T-2581), `view.ts` (key M, the canvas, pan and zoom, clicks,
hovers and the camera, the picture, the names picked and measured, the layout worker and its progress;
`ui/DiagramButton.tsx` its button and `DiagramTopLeft`, the save button and loading bar, T-2163, T-2235, the mode buttons
T-3360;
the station tag is `complexTip` in `map/networkTips.ts`).

### src/ui/: Preact components

`Dock.tsx` (the dock, its icon tabs and resizing, T-252); `AccountPanel.tsx` (the account panel,
the person icon, and `startAccountUi`, which mounts both account panels in a root of their
own so the start menu opens them too; styles in `account.css`, T-972); `Leaderboard.tsx` (the
leaderboards and their trophy, placed beside the achievements by `Achievements.tsx`, T-973, T-2179; its board picker is a city search using `findStart` from `NewGame.tsx`, T-2813); `HelpPanel.tsx` (the "?" button's list of controls no button shows, T-908); `BuildTab.tsx` (the tools and the blueprint); `ToolBar.tsx` (the track and station bars on the map while their tool is on, T-2282, and `MapToolBar`, which draws the click mode's aura, with `clickMode`, T-2284); `LinesTab.tsx` (with its line groups' rows and row dragging, T-1791, T-1813; the groups from `game/lineGroups.ts`, the hand order from `game/handOrder.ts`; its eye toggles and Show all, T-3420), `StationsTab.tsx`, `GroupPill.tsx` (a group's outlined bullet, T-1813), `MoneyTab.tsx` (the tabs; the Cities tab is `Cities.tsx` mounted in `Dock.tsx`); `listParts.tsx` (the scope toggle, the Near column and the name order the Lines and Stations tabs share); `toolKeys.ts` (the build tools, the number and letter keys and `pickTool`); `hideUi.ts` (F2 hides the interface for screenshots, T-1491); `Modal.tsx` (the native modal dialog behind the saves, account and leaderboard panels and the confirmation question, with `useModalDialog`); `useClockQuery.ts` (`useClockQuery`, a panel's question to the clock worker, and `useLatestAnswer`); `scroller.ts` (`scrollerOf`, the nearest scrolling ancestor);
`MoneyHistory.tsx` (the Money tab's history charts, T-447); `FareGraph.tsx` (the fare curve graph, linear and zoomable at one fixed ratio of its scales, its nodes (three, or 1-12 from the Advanced fares panel, T-3739) each dragged one way at a time with hover arrows, an arrow for one past the plot, and rows of steppers under it, the base fare then each stretch's end and rate per km, T-864, T-883, T-1112, T-1143, T-1201); `AdvancedFares.tsx` (the Advanced fares panel: the bigger graph with Add point, rows with remove, the kind and transfer rows, every country's multiple and base fare, T-3739); `Cities.tsx` (the Cities list, T-550, T-749); `SettingsTab.tsx` (`SettingsSearch.tsx`, the search box over it that hides the rows it does not find, T-3681); `SavesPanel.tsx` (saved games, T-251; `thumbShade.ts` recolours a picture to the current theme, T-1852; `SlotThumb`, a save's picture with each panel's city named over it, T-894); `bullet.ts` (`bulletText`, `bulletNode`: a line bullet as one SVG, its outline in the line's shape (`bulletDraw`, T-3319) and its symbol placed from the canvas-measured ink box with the baseline on a whole device pixel, per font, symbol, size and scale, T-705, T-745; the sizes in it match `.chip` in style.css; used by `Chip` and `lineBullet`); `widgets.tsx` (chips, steppers,
checkboxes and the inline SVG icons); `ColourPicker.tsx` (the line colour picker: preview while dragging, one commit on release, T-672); `TopBar.tsx`; `GoalTicks.tsx` (a region's goals as ticks over its split bar, the city view's, T-1437); `SkyBanner.tsx` (the day and night picture and demand band, a canvas, T-861; `MapSky` in TopBar.tsx puts it on the map, T-1868); `StatusTag.tsx` (the demand status tag on the map's top row); `BusyTag.tsx` (the busy tag after it, T-1200); `TroubleList.tsx` (the trouble list at the map's lower right, T-1433); `Guide.tsx` (the Tutorial card, T-1822; steps, notes and tick-keeping in game/guide.ts, its styles in ui/guide.css, `guideVerbose` in game/ui.ts for the tutorial-only longer hovers, T-2263; progress in the save header's `guide` through the clock worker); `KeepDot.tsx` (the yellow "Saved only in this browser" dot right of the search glass, T-1832, T-1868; the hour's count and when it is due in game/keepNudge.ts, kept per slot in localStorage by game/keepState.ts, which clockClient's downloads also mark; check checks/t1832-keep.mjs); `ripple.ts` (the card's ripple, T-1822: one fixed canvas with its own rAF loop, pulses on `data-guide` targets by the timing rules; `startRipple()` is called from `mountGuide`); `TouchBar.tsx` (the touch control bar's buttons, T-1260); `ToolNotes.tsx` (the click mode's descriptor, T-2284, tool hints, the notice and the confirmation
dialog); `DemandSelector.tsx` (None/Homes/Destinations and the purpose filter, the track display and station ridership rows; Drives is a Settings check, T-1477); `Blueprint.tsx`
(Build's quote, its cost breakdown by part and stretch, T-2269, and Construct, and the Select tool's selection bar on the map, T-353); `trackPrice.ts` (the track tool's price at a level and the level buttons' hovers, T-2271); `Debug.tsx` (the `?debug=1` readout); `Credits.tsx` (links to the game's about page on anita.garden and its sources section, which carries T-924's licence notices: the foot of Settings, and "About this game" on the start menu); `FeedbackPanel.tsx` (Settings' "Submit feedback" modal, sent through game/account.ts, T-1673); `MoneyText.tsx` (a money figure in the alert red while negative, and the hover of a line's money, T-722, T-1851); `PassingNote.tsx` (a one-line note on the map for a few seconds, under the edit note, T-2166); `zenmaru.css` (Zen Maru Gothic as woff2 slices by code point, T-1825); `inspector/` (one inspector
per selection kind: `Inspector.tsx` (the switch on the selection), `CommutersInspector.tsx` (commuter bubbles picked on the map), `ClosestStations.tsx` (the 3 stations nearest on foot, shared by the station and bubble inspectors, T-312), `PurposeBreakdown.tsx` (one purpose's departures through the day, T-193), `PurposeSplit.tsx` (riders by trip purpose as a stacked bar under a line's riders and a station's boardings, T-165); `StationInspector.tsx` (its upgrade row `StationUpgrades.tsx`, T-1281), `TrainInspector.tsx`, `TrackInspector.tsx` and `JunctionInspector.tsx`; `LineInspector.tsx`, handing a bus or ferry line to `BusInspector.tsx` (T-1464), both built from `LineTitle.tsx` (the title, its name, symbol and colour editor, the group row) and `schedule.tsx` (the Schedule table and its hovers, the average speed, the past 24 hours with the Average load, and the vehicle words, `vehicleOf`, T-2373); `CityInspector.tsx` with `CityDetails.tsx`, the city view's own rows (T-1438), and `TripGraph.tsx`, its trips by length (T-2365; the bins and curves in game/tripLengths.ts, the zones' answers and the corridors' distances in game/zoneLengths.ts); `AllCitiesView.tsx`, the Cities tab's All cities row as a view, and `HomeButton.tsx`, the home icon of both (T-2395; the home a signal in game/home.ts, set through clockClient's `setHome`, kept in the save header's `home` by the clock worker and used by ui/locate.ts's open fit); the trip panels share `tripText.ts` and `RouteParts.tsx`;
`ComplexMap.tsx` the complex map; `TrainStops.tsx` a train's next stops and who gets off, `MareyChart.tsx` the line's timetable chart from `TrackApi.line_trips` and the run profiles (T-2603), with `mareyShare.ts` the other lines over its track and the at-grade conflicts on its path (T-2660), T-868; `SchedulePanel.tsx` the schedule panel, its departures board from `TrackApi.station_departures` and the chart at its width, a group's chart too (T-3346); `LineProfile.tsx` the line's cross section at the foot of its view, from the clock worker's `workers/lineProfile.ts` (the line's points with their track height, stops, other lines' track crossing it) and `game/groundHeights.ts` (ground heights from the hillshade's Terrain Tiles, decoded per tile), T-2734; `OneRider.tsx` one sampled rider of a train (`game/riderSample.ts` draws it, `riderTrip` in `map/tripView.ts` reads its minutes), T-1436; `NameEdit.tsx` a station's or line's name edited in place, T-534; `LineGroup.tsx` the line panel's group row, T-1791; `GroupInspector.tsx` the group view and its diagram, laid out by `game/groupLayout.ts`, T-1791); `LoadingScreen.tsx` (the clock hold at startup, the loading card,
the screen when demand cannot load, T-149, T-172, T-180); `StartMenu.tsx` (the start menu and the URL switches that skip it, T-677); `StartLogo.tsx` (the menu's logo: a NetworkRenderer of its own with a flat camera, `window.twLogo` its frame counts, T-1567) with `map/logoNetwork.ts` (the logo's letters as network buffers and timetables, no sim); `NewGame.tsx` (the new game sheet: mode, start city, current location, search, real network, running side; `START_CITIES`, `REAL_NETWORKS`, `zoneForLocation`, T-728, T-731, T-1132, T-3297); `SideSwitch.tsx` (T-3297: the Trains keep row, Settings' switch of a game's side with its question and the list of what the switch fixed and left, `flipSites` in `game/world.ts`; the start's side from `game/runningSide.ts`, the railway side by country and the real networks that keep left); `Spinner.tsx` (the small spinner, T-647, and the bunny's train, T-677); `Search.tsx` (T-148); `DemandStopped.tsx` (the alert when demand stops after startup, T-757); `keys.ts` (which keys a focused control keeps, T-183); `locate.ts` (the camera to a station, a line or track nodes (a breakdown row, T-2269): search picks, the tabs' double click, the inspectors' locate button, T-569; `viewStart`, the camera at a new game's start);
`tip.ts` (the hover tag for every panel and control, from `data-tip`, never `title`, T-210);
`blurHold.ts` (every hover kept up through a window blur until the pointer moves, T-483);
`style.css` (theme tokens).

### src/workers/

`realRoute.worker.ts` (the real-network router: graphs joined at borders, A* with the turn limit
and families, then the fit in steps on a track model of its own, dropped for a newer request, T-727d, T-754; the outlines of the regions with a graph sent to the page, T-913); `protocol.ts` (edit ops, `RouteAsk` for a route edit and its preview, clock messages, the network snapshot for demand), `clock.worker.ts` (owns
the network: `TrackApi`, notes/T-025.md; money too; an edit's follow-up (state, trips, boarding hour, answer) waits behind the previews asked meanwhile, `later`, T-1407, an early track segment's until the pointer rests, `holdOn`, T-2404; `slotSummary`, a save's picture
with a panel per city, T-894), `slots.ts` (the save slot of the game in hand, T-3734: `writeBlock`, whether and why the
slot may not be written now; `publishSlots`, autosave, the save by hand, the saves panel's actions, the slot's lease
against other tabs, the cloud queue; its state is the one object `saves`, which the clock worker also reads and
writes, and it reaches the game through the `SlotHost` the worker binds; `letGoSlot` for a start or open that failed,
whose record is game/openFailures.ts, T-3773), `valuePatch.ts` (a value posted as its changes from the last one: the
money record, T-3803), `stateRows.ts` (`StateRows`: the state's edge, node and line rows
kept between edits and read from the track model again only where `take_changed_lines` and
`take_name_changes` say they changed; station names kept by node, phase tables per line; T-1382, T-1540), `platformLines.ts` (each
station's drawn platform line in world units, on its node row, for the complexes; T-1859), `saveFile.ts` (the save format
and IndexedDB, T-029; slots' backups, T-744; a closed connection opened again and demand caches dropped on a full disk, T-1829; removed zone ids taken as their successors on load, T-1053; each slot's tie to the account, T-988; a failed read tried again, then thrown, never taken as an empty slot, T-2703, the slot list too, T-2705, check `checks/t2703-storeread.mjs`), `cloud.ts` (games on the account, T-988: uploads, claims, settling, conflicts split off as versions; used by the clock worker and the main thread), `versions.ts` (T-2037: a save's origin and latest player change, the origin rule for older saves, saves grouped into one card per game), `clockClient.ts`, `demand.worker.ts`, `demandClient.ts`, `demandCache.ts`
(the last final demand result kept beside the save, T-328),
`demandProtocol.ts` (the demand workers' messages; `isQuery`, the view questions' kinds; `ALL_PERIODS`), `cacheWrite.worker.ts` (the demand cache's IndexedDB write, off the main thread, T-789), `demandLod.ts` (which zones are held fine and which coarse, T-1223; the zones built in last are spared, T-2005), `demandFocus.ts` (the focus pool: a game load's zone in view solved alone to its last round, T-2045), `deviceMemory.ts` (the device's memory the pool sizes itself by, and whether it is a phone, tablet or 4 GB device, T-2122), `demandClusters.ts` (the network split into clusters that share no rider,
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
never autosave), `?backupMs=N` (wall ms between a slot's backups, T-744), `?demandWorkers=N` (`auto`: the page's own pool rule, where a check's core plan would add a number), `?focusWorkers=N` (the focus pool's size),
`?memFree=MB` (the free memory a check pretends), `?lod=` (a fixed level of detail, `workers/demandLod.ts`), `?opened=all` (no realm
hidden; checks that visit zones in every realm), `?patch=check|shiftcheck|0` (patched against whole builds, `game/patchMode.ts`),
`window.twSheet` (the diagram's last rubber sheet, T-2568), `?rawdem=1` (the hillshade as it comes), `?logo=0` (no start menu logo),
`?quietMs=N` (how long the loading card may hear nothing before the failure screen), `?keepMs=N` (the hour after which the keep dot appears), `?track` / `?notrack` (plays and playtime
pings on or off, `game/playtime.ts`), and the start menu's `?city=<zone id>` and `?mode=sandbox` (with `?new=1`), `?slot=<key>` and
`?load=file|restore|new`; `?lang=<code>|pseudo` (a language, or every message in ⟦ ⟧, `i18n/index.ts`, T-2583); `?perfOff=...` (CPU A/B switches, `src/perfFlags.ts`, which lists them; notes/T-045.md).

### Build

`npm run build --prefix app` = wasm-pack twice (Cargo profile `wasm`, `+simd128` from
`sim/.cargo/config.toml`), type check, `app/dist/`
(city packs, real networks and real-network graphs copied into `dist/packs/` by `vite.config.ts`), which serve.py serves at
`localhost:8800/trainworld/` (from the snapshot worktree, see CLAUDE.md). `npm run wasm --prefix
app` rebuilds only the WASM. `npm run check --prefix app` type-checks.

Every fetch of a `packs/` file goes through `game/dataBase.ts` (`dataUrl`): `packs/` beside the page,
or the R2 prefix a published index.html names in its `tw-data` meta; the terrain blocks go through
`TERRAIN_BASE`, `packs/terrain-world/` locally or the fixed prefix in its `tw-terrain` meta (T-2901). `tools/deploy.py` publishes a
build: `packs/` (without the terrain blocks: `--terrain-world` uploads those once, from the build or a data folder with `--terrain-dir`, and a release is refused until they are complete) to R2 `trainworld/<version>/`, the rest of dist to the website repo (T-924,
notes/T-924.md has the commands). During a batch, publish a build on the last freeze
(`pipeline/freeze.py build`, then `deploy.py --dist` that build; T-928), never one that copied
data/packs mid-batch.
The page's link preview (`og:` and `twitter:` tags in `app/index.html`, absolute URLs on anita.garden/trainsworld/)
names `app/public/preview.png`, 1200x630, which the build copies beside index.html and `deploy.py` publishes with the
rest of dist (T-2543). Redo the picture when the look changes; X and Bluesky cache cards.
The game installs as a phone app (T-3815): `app/public/manifest.webmanifest` (relative `start_url` and `scope`, standalone,
the pink theme's background) and `icon-192.png`, `icon-512.png`, `icon-maskable-512.png` (the favicon glyph on that background),
linked from `app/index.html` with the iOS tags; `deploy.py` copies them with the rest of dist. There is no service worker and no
offline caching, on purpose: it can pin an old build after an update, and offline play is for the Steam release. Chrome
offers Install without one (checked with `Page.getInstallabilityErrors`, Chrome 157).

## checks/: standing checks

`checks/INDEX.md` lists every file in `checks/` on one line (kind, in smoke or not, runs, last commit, what it proves); grep it first. Regenerate it with `python trainworld/tools/checks_index.py` after adding or moving a check. Run from the maps root (`node trainworld/checks/<file>`); each file's header comment says what it needs and how long it takes.

The standing checks stay in `checks/`. One-off checks of finished tasks (screenshot makers, before and after
measurements, a feature's first test) are in `checks/old/`, written `old/<file>` below: run them as
`node trainworld/checks/old/<file>`; nobody runs them as a habit, and some use selectors the app no longer has (T-1586, T-2831).
`lint.cmd` runs the three linters (T-3443, about a minute warm, Rust on 3 cores): `tsc --noEmit` with `noUnusedLocals`, `noUnusedParameters`, `noImplicitReturns` and `noFallthroughCasesInSwitch` (app/tsconfig.json), `cargo clippy --all-targets -- -D warnings` on sim/ with and without `--features demand` (the style and complexity groups, `approx_constant` and `large_enum_variant` are allowed in sim/Cargo.toml), and ruff 0.14.5 with only F and E9 on pipeline/ and tools/ (`ruff.toml`; the maps venv has it). Run it before handing back work that touched app/, sim/ or pipeline/.
Scripts for the orchestrator's builds are here too: `snapbuild.cmd <hash>` and `snapbuild-frozen.cmd <hash> <freeze>` (rebuild 8800; the served build's assets/ and fonts/ stay beside the new build's and `tools/deploy.py` does the same on the website, both through `tools/keep_assets.py`, so a tab open across a rebuild can still load its lazy files, T-2962),
`smoke8800.cmd` (smoke 8800 light then dark, before a publish), `keep_hunks.py` (stage some hunks of a file), `mark_taking.py` (mark TASKS lines taken).

- Playwright checks: a `page.evaluate` that moves the map must return nothing (`() => void window.tw.map.jumpTo(...)`): MapLibre's `jumpTo` returns the map, and Playwright serialising it is 1-2 s of main thread that reads as lag (T-1052).
- `lib.mjs` is the shared plumbing (Playwright, headless Chrome); `lib/play.mjs` plays the game by pointer (how-to at its top) and `lib/reads.mjs` reads panels. `play.mjs` keeps one game open per Vite port for probes (`start`, `do`, `sim`, `edits`, `reload`, `stop`); `wait.mjs` waits for a line in a log.
- `t3137-lanes-parity.mjs [--write]` fingerprints what the lanes and move tables decide against a baseline (`t3137-parity-digest.json`; text in `artifacts/trainworld/t3137-parity/`); rerun it after any change to lanes, moves, routing or junction capacity, and write a new baseline when a change meant to alter them is reviewed. `t3297-mirror.mjs` does the same for a left-hand game against its mirror (about 4 minutes).

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
- `start_difficulty.py [--tiers] [--fresh]` (T-2690) writes `app/src/game/start-difficulty.json`, the
  easy and medium starts (any other is hard; zones.md "Start tiers"), read by `game/startDifficulty.ts`; the table in
  `../artifacts/trainworld/t2690/scores.json`. About 25 minutes on one core; `--tiers` re-cuts in seconds.
- `realms.py [--registry F] [--out F] [--dry]` writes each registry row's `realm` and `data/packs/realms.geojson`, the land of the realms in `app/src/game/realms.json` (T-975, about 1.5 minutes; run by the join).
- `regions.py`: each cell's state, as the build writes it (T-162). To re-solve a finished pack's
  gravity after a decay or border factor change without a rebuild, use `pack_gravity --keep-zones`.
- `enrich_all.py phl bos` adds the non-work inputs (everyday sectors, NCES schools, IPEDS
  universities, airports from FAA/BTS/OurAirports, medical and parks from OSM) to finished packs
  in place, about 0.5-2 minutes a city plus Overpass time (T-142). Wikidata's airport patronage
  answers are kept in `data/work/wikidata_boardings.v2.json` (T-1083, T-2896; delete it to ask again);
  national counts that come first are in `pipeline/airport_passengers.csv` (T-2896).
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
- `checks/split_zones.py` (T-3012; read-only, about 4 minutes on one core): every registered zone's land read from its
  `boundary.geojson`, the zones in more than one piece on land listed with the zones that touch each piece and the zones
  standing in the gap, into `artifacts/trainworld/split-zones.json`; slivers, water gaps and foreign exclaves ignored
  (the docstring has the rules); exit 1 when a piece of 20 km2 or more lies 2 km or more from the rest.
- `light_pass.py run` (T-3244, world-run.md "The light pass"): the demand steps over every shipped tier 0
  pack with no extract read or joint solve; `campus_outlines.py` (the world's universities and colleges
  from Overpass in longitude bands, split per pack, read by `tier0.osm_education` for a cache without
  outlines) and `wikidata_prefill.py` (the Wikidata caches, 200 ids a query) are two of its steps; the
  latter ends with `campus_pools.py build`, the world table of institutions' counts, campuses and labels
  that `tier0.university_sizes` shares each count over (T-3273).
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
  the res-9 copy and the merged pack, append-only; only `world_run.py` still runs it, and nothing
  needs it rerun by hand (T-926 drops the step and the layer; its sources: docs/data-register.md
  "Car ownership"); `--plan` prints every planned country's
  share and source, writing nothing.
- `everyday_rates.py <ids> | --all` writes the `everyday` header block (the country's survey
  outing rates by purpose, T-509 stage 4; table `SURVEYS`, worldwide median otherwise; the
  distance scales are the sim's one worldwide set, `everyday::DECAY`, T-637) into the merged pack and its res-9 copy; rerun it after anything that rebuilds a pack
  header. `show` prints the table.
- `terrain_fine.py` (T-1854): the fine terrain intermediate, built once on AWS (`--kind terrain`):
  per one-degree block, the mean slope of GEDTM30's 30 m pixels per 1/960 degree cell and the mean
  elevation per 1/240 cell, read by HTTP range from OpenGeoHub's 432 GB file (format in its docstring), with
  tower fields cut out first (`tower_ground`, GHSL's 1 km built volume, T-1881; `run` fetches it to
  `data/raw/ghsl/`, `TW_GHSL_30SS` another copy). Since T-2901 it holds every block with a non-empty source
  tile, so a new zone needs no run: kept at home as `data/work/terrain-fine-world/part*.tar` and `manifest.json`
  (9.35 GB, never unpacked); `blocks [lat_lon ...]`, `run`, `check` (a block against rasterio's read).
  `zone_terrain.Fine` reads a test run's tars before the intermediate's with `TW_FINE_OVER=<dir>`.
- `world_terrain.py` (T-2901): the world's terrain classes, one gzip file per degree block with data
  (`build <fine> <out> [--jobs n]`, into `data/terrain-world-v2/` since T-3372; the old `data/terrain-world/` is v1, still on R2 for releases made before it) and `index.bin` (which blocks have a file), read by
  the game (`game/terrainBlocks.ts`, the terrain section of `workers/clock.worker.ts`) and published by `tools/deploy.py --terrain-world`.
- `zone_terrain.py` (T-1270, T-1854): the terrain classes from that intermediate (thresholds, filter and
  cap in its docstring) and the `.terrain` encoding, which `world_terrain.py` uses; there are no per-zone
  files since T-2901. `mix [dir]` the class shares over the blocks, `probe lon lat km`. Its `Dem` reads GEDTM30 at 240 m (`data/raw/gedtm30/`) for `market_drives.py slope`, whose
  output `data/work/market_drives/slope60.npy` covers the world; the 10 GB raster is not kept on disk
  (T-2554), so download it from `zone_terrain.SRC_URL` only to rerun `slope`.
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
  neighbourhoods inside the boundary, or a node in a gap of it under about 1 km across that no other
  pack's boundary holds, such as a Nile island at Aswan; an area holding a node of its own name is that
  node's place, dropped where another pack holds the node, T-3623; from the extract's cached features for a tier 0 city, else
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
  track set side by side and levels chosen on that geometry, T-1303), the sim's `realsave` bin (every edge marked `sep`, grade-separated from the others whatever its level, T-1751; every
  dead-end station a terminal, and the network converted to the game's Mercator frame before it is settled and
  saved, so the game loads it settled, T-3233; its `_real_track.json` keeps the pack's local frame; the
  track model's network, routed hops traced with `trace.rs` at 3 m, names tidied by `track/names.rs`:
  lines lose their feed word, stations get title case without "STATION", T-1442; the sim's `load`
  runs the same on older saves), `gamesave` (the game file
  `<city>_real.save`); `ship [city ...]` copies the seven a sandbox starts from (or those named) to `data/packs/real/<id>.save`
  (T-731; new files, since data/frozen/ links to the old ones). Tokyo's trips file is `calib_tyo.py trips` (peak trains an hour
  by tier and `REAL_PEAK`, timed at the count point, T-1751). `REALNET_OUT` stages geom, route and gamesave outside data/. `realcorridors.py <city>` (or
  `realnet.py corridors <city>`, T-1305) routes the hops again keeping their chains and writes
  `<city>_real_corridors.json`: lines routed again off other classes' track and off full pieces
  (T-1774), parallel chains grouped into corridors, cut at stations and where the
  lines change, as many double-track pairs a piece as its trains need and OSM's tracks allow and
  the lines on each, with a picture, (`realshared.py`) the track model's nodes, edges, platform
  tracks and line paths from them (T-3790: no node over a game node's 8 ends, a bigger junction split into
  several joined by links, a station side over 8 through fans, and a short dead end where a line turns back at
  no stop; rules in its docstring), and (`reallevels.py`, T-1773) each node's level and each edge's
  stretches of level from OSM's layer, tunnel and bridge tags (rules in its docstring, notes/T-1773.md),
  which `realsave --shared` builds into the save, a ramp per change placed clear of platforms
  (routed on realgraph.py's lines graphs in `data/work/realgraph-lines/`, T-3790; `REALGRAPH_DIR` reads a staged
  set) (`--slow` also
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
  asked for (T-1625, notes/T-1625.md; `realsave` reads `trains` and `hsr` from the network JSON, in both its builds, and gives high-speed trains to every line with over half its length on high-speed track, the game's own rule for a new line, T-3272). `calib.py
  targets <city>` writes the city's targets (`data/work/calib/<city>/`), `calib.py run <city>
  [--set k=v,..] [--params p.json] [--tag t]` runs the sim's `calib` bin (a weekday, commutes and
  everyday, 3 crowding rounds, about 22 s for New York) and scores it (station entries, operator
  totals, rail share by home area); `calib.py score <run.json>` rescores. notes/T-509.md.
- `realside.py <in.save> <out.save> left|right` (T-3297): a real network's game file set to a running side, the
  file `realsave --left` gives for a network whose junctions all fly (realsave's).
- `realauto.py` (T-3225): real networks with no hand work for the cities in `realauto.json` (`left`: built keeping left, T-3297); `extract`
  cuts noritetsu's extract (read only) to each pack as the `osm` adapter's input, with each relation's
  ways chained as its shape (`realnet.py geom` lays off-graph hops on them), the routes it dropped by
  reason (`dropped_routes`) and its own `lines` overrides, which the `osm` adapter reads after the
  source's (S-Bahn mapped as light rail, T-3265); `build [ids] [--jobs N]`
  runs net, geom, corridors, `realsave --shared --slow` and gamesave per city, builds a city again
  without the lines that did not run (`auto_<id>_drop.json`) until every line runs (`BUILD_PASSES`), builds a city
  with a keep list (`realauto_keep/<id>.json`, written by `keep <id> <save>`; Moscow's is v0.4.2's lines, T-3790)
  from the lines it picks only, and writes a report per city
  (`auto_<id>.json`: stations, lines running, names in Latin letters, km off the graph, holds);
  `ship` ships the listed cities that passed and writes `app/src/game/realNetworks.json`. `realnet.py`
  makes each listed city a `CITIES` entry (`auto_city`). Rules in its docstring; notes/T-3225.md.
  Browser check: `checks/t3225-real-load.mjs` (every city from the new game sheet to playable).
- `realgraph.py <cc ...> [--check] [--out DIR] [--nori DIR] [--lines DIR] [--lines-only]` (T-727b; freight from the extract's ways and abandoned rail from the region's `_extra.pmtiles`, kinds 8 and 9, T-3495; run with the system Python, which has pmtiles and mapbox_vector_tile): the real network the layer draws as a routing
  graph per noritetsu region, plus short undrawn track that fills its gaps (T-816), from noritetsu's
  extract, `ways.json` and `track_extra.json` (the ways tiles flag `x`, T-3043; read only), into `data/packs/realgraph/<cc>.rgr` (gzip, layout in its
  docstring, which has each chain's OSM `layer` in its flags' top bits for the real networks' levels,
  T-1773; 0.01-7.5 MB, 1-32 s a region) and `index.json` (the regions built, which the track
  tool snaps inside, T-913; no regions given rewrites only it). Built for all 118 noritetsu regions
  with drawn track (44.2 MB, about 5 min on 1 core; T-727e, T-3043), copied into the build. Each region also
  gets its lines graph, `data/work/realgraph-lines/<cc>.rgr` (only track a line runs over plus gap fillers, cut only
  where those meet; the real networks are routed on it, never deployed, T-3790; `--lines-only` writes just these,
  in the venv). Rerun after noritetsu rebuilds a region.
- Basemap glyph tiles (`app/public/fonts/`, checked in): `glyphs.py` rebuilds them, after a
  one-off `cargo install build_pbf_glyphs --locked` (on Windows set `CFLAGS` to a libz-sys zlib
  folder first, notes/T-056.md). Only needed to change the font, weights or fallback.
- Country modules, each a tier 0 adapter through `tier0.py`'s hooks like `france.py` (docs/adding-a-city-by-hand.md):
  `china.py` (Shanghai first, T-291), `germany.py` (Berlin, T-290), `india.py` (Mumbai, T-292), `korea.py` (the Seoul Capital Area, T-457), `brazil.py` (Sao Paulo, T-458), `nigeria.py` (Lagos, T-459), `australia.py` (planned rows' country keys, T-1857), `mexico.py` (planned rows' census people and DENUE jobs, T-2080; `mexico.py check` tries the hooks on stand-in outlines). `china_holidays_pattern.py` fills `china_holidays.json` with the usual pattern for years the State Council has not published.
- `enrich_all.py`'s steps, each also run alone on a finished pack: `enrich_airports.py`, `enrich_everyday.py`, `enrich_parks.py`, `enrich_schools.py` (with `education_sources.py`), `enrich_universities.py`; they share `overpass.py` (one cached Overpass request, kept as `.json.gz`) and `arcgis.py` (one cached, paged ArcGIS REST query).
- Calibration cities beside `calib_tyo.py`: `calib_lon.py`, `calib_par.py`, `calib_seo.py`, `calib_sdj.py` and `calib_tor.py` hold each city's trips file and targets for `realnet.py` and `calib.py`.
- US helpers: `spread.py` (each census block's people and jobs over H3 cells), `wfh.py` (work from home from the ACS), `group_quarters.py` (people in institutional group quarters), `fit_decay.py` (the gravity decay fitted against LODES OD), `lumps.py` (job cells far above their neighbourhood), `look_water.py` (a look-image of a pack's water mask).
- `tools/country/` (adding-a-country.md's scripts): `backup.py` and `restore.py` (the shared files a run or join rewrites, saved and put back), `setbuilt.py` (parked plan rows set back to built), `stale.py` (market packs naming a market place with no pack), `setup_sim.py` (a sim copy for a long run), `start_hidden.ps1` and `waitany.ps1` (a detached start and the one background wait), and the census inputs of `admin_totals.py`: `turkey_census.py` (Turkey's people by ilce), `india_census.py` (India's two files), `cambodia_census.py` (Cambodia by district), `colombia_census.py` (Colombia by municipio, T-2712), `egypt_census.py` (Egypt by governorate, T-2956).

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
`market_drives.py build` for new or moved places (run by the join after its market step, T-2054; the routed drives between places and each zone's
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
