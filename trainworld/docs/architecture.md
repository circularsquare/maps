[Back to the spec](../SPEC.md)

# Architecture: performance ideas, technology and saves

Sections 2, 7 and 11 of the spec in detail. Where files live: `docs/code-map.md`.

## 2. The core performance idea

**Trains are cheap; passengers are expensive.** Passenger work is what has to stay cheap and
never come in bursts (the evidence from Subway Builder: notes/architecture-history.md).

- **Trains are a function of time, not things that get stepped.** A line has a route, a run-time
  profile (accelerate, cruise at the curve-limited speed, brake, dwell, plus junction delays,
  [track-and-service.md](track-and-service.md) 6.2) and trains per period. A train's position is
  `profile(t - departure)`; trains outside the view cost nothing.
- **Passengers are flows, not agents.** Demand is trips per period between places. Route and mode
  choice run on that in workers when the network changes, producing loads per line segment per
  period. The clock worker turns those into the boarding account [T-164,
  [track-and-service.md](track-and-service.md) 6.6]: each train's load and each platform's crowd,
  worked out a day at a time when service or demand changes, never stepped per frame. Demand's
  crowding rounds run the same rule per period, so a line that leaves riders behind loses some
  [T-188].
- **Recompute only what changed**: demand is solved only when service changes.
- **Optional flavour agents**: if the screen wants little people, sample a few from the flows for
  display; they never drive the simulation.

Borrowed from Factorio and NIMBY Rails (`research/performance-factorio-nimby.md`):

- Sleep what is idle; wake on timed events, never poll (FFF-148, FFF-421).
- Cosmetics are a pure function of the clock, evaluated only when drawn (FFF-204, FFF-421).
- Merge many things into one solved unit: riders are counts per bucket (FFF-176, FFF-416).
- Flat structure-of-arrays memory (FFF-204, FFF-209).
- Thread only independent, mostly-read work, each worker owning its memory (FFF-215, FFF-421).
- Deterministic simulation: save = inputs, replays and bug repro for free (FFF-76).
- Snapshot-then-compress saving off the main thread (FFF-201).

How NIMBY Rails does it, for comparison: notes/architecture-history.md.

### 2.1 No signals, with the door left open

Trains do not block each other: two lines on one track pass through each other. Tangles are
discouraged by capacity and junction delays computed from scheduled frequencies
([track-and-service.md](track-and-service.md) 6.2), and by the price of junctions and crossings.
If blocking trains are wanted later:

- The track is already a graph of directed segments between nodes, the shape a block system needs.
- The renderer and the passenger model never assume `position = f(t)`. They consume **keyframes**:
  constant-acceleration phases (time, offset, speed, acceleration) along a path of `(edge,
  direction)` segments. Today every trip of one line, direction and demand level shares one phase
  table and carries only its departure time (format in notes/T-040.md); an event-driven block
  simulator could give each train its own phases without touching renderer or demand code.
- The simulation is deterministic and event-driven, so "wait for block" is a new event type.

An idea for later: signals simulated by clumps of shared track with a stored repeating pattern
(T-265, notes/T-170.md).

## 7. Technology

Web first, so it can be shared as a link.

- **App and UI**: TypeScript, Vite, Preact with signals. No game engine: the map is the game. The
  main thread holds UI state and read-only snapshots from the workers (the `WorldView`: the save's
  inputs plus worker results) and changes the game only by sending edit operations to the clock
  worker ([track-and-service.md](track-and-service.md) 6.5).
- **Map**: MapLibre GL 6 for the basemap, flat or as its globe (a Settings choice, T-033). Track, stations and trains are drawn by our own WebGL2 renderer on a **separate canvas
  over the map**, neither GeoJSON sources nor a drawing custom layer (why:
  notes/architecture-history.md). Track buffers are built in the clock
  worker and replaced on each edit, made from per-edge, per-line and per-station pieces the worker
  keeps and rebuilds only when their inputs change (`NetCache` in map/network.ts, T-496); trains are instanced and positioned on the GPU from keyframes
  and the clock. World coordinates and the moving draw anchor: [world.md](world.md). A lost WebGL
  context (phones drop them) is taken on both our canvases; on restore the renderer and the demand
  canvas compile again and upload what they last drew, and the camera layer is added back to
  MapLibre's restored style (T-953). Shaders are written to WebGL2's guaranteed limits and highp
  floats in every fragment shader that measures px (notes/T-953.md).
- **A state and its trips land together** [T-648]: the clock worker posts each state, then its
  trips, before anything else. The client holds the state until those trips (at most 250 ms), so
  the world, the network and the trips are taken in one task and no frame draws old trips on new run
  profiles; a state built at another draw anchor is taken at once. A state leaves out its edges and
  nodes when they are as in the last one posted (`sameEdges`, `sameNodes`); the client puts back its
  last ones.
- **Staying in step with the basemap** [T-075; notes/T-015.md, notes/T-075.md]: an empty MapLibre
  custom layer hands over the camera matrix on every frame MapLibre draws, and the overlay is drawn
  right there with the same matrix; our own frame loop drives the overlay only for the clock and
  skips duplicate draws when MapLibre has a frame queued.
- **Flat map or globe** [T-033; [ui-map.md](ui-map.md) "Globe"]: the camera handed over is
  `map/gl.ts` `Camera`, from the custom layer's `defaultProjectionData`: the flat map's matrix
  (Mercator 0..1 to clip space) and, while MapLibre draws its globe (out to zoom 12), the globe's
  matrix (unit sphere to clip space), the horizon's plane and how far the view has blended to flat.
  Every vertex shader of ours (the renderer's, the crossing marks', the transfer bubbles', the demand
  canvas's and the realm shade's) places points with `project()` (`PROJECT_GLSL`): on the flat map
  exactly `u_matrix * p`, on the globe MapLibre's own sphere, horizon depth and blend (its
  `_projection_globe.vertex.glsl`), computed from the anchor's Mercator position plus the local
  offset in float32, which holds to a few metres where the globe is drawn (zoom 12 and out; closer in
  it is the flat map with its centimetre precision). The CPU twin is `mercToCss`, behind
  `overlay.toScreen`, the one call from local units to screen px; a point behind the horizon comes out
  NaN. Screen-space sizes and directions come from projecting two points, so they hold on the
  sphere; sizes in px per metre (`pxPerUnitAt`, a train's length) keep the flat map's scale.
- **Time on the GPU** [notes/T-018.md, notes/T-040.md]: keyframe times are relative to a render
  epoch at the start of the current game hour, so the float32 time stays under 3600 s and positions
  within about 2 cm. When the hour turns the clock worker lists the next hour's trips relative to
  the new epoch (computed in float64), and only that buffer is re-uploaded.
- **Frame loop** [T-073, T-084, T-097]: frames only while the clock runs; 10 fps while the window is
  visible but unfocused; nothing while hidden or paused; none while no trips run
  this hour (the clock wakes once a game minute instead). Otherwise the overlay draws on MapLibre's
  frames and once after a change to what is drawn. The demand views have a canvas of their own over
  it, drawn only on MapLibre's frames and after their data changes, never by the clock; bubbles and
  a selection's far end are summed in their own worker (`workers/commuters.worker.ts`).
- **While the camera moves** [T-1207; `map/moving.ts`]: work done for each MapLibre frame that need
  not be runs at most every 100 ms (`MOVING_PLACE_MS`) and in full on the frame after the move ends:
  the DOM placers (station names, capacity tags, attraction marks, zone loading bars), whose marks
  move as one with a flat pan in between; the platform crowds; the station sounds and faces sample
  half as often; the Lines tab's live counts wait. The lighter map (`map/lite.ts`): the basemap
  without buildings, the residential tint, minor roads, airfields, small rivers and minor names, no
  hillshade, and a pixel ratio of at most 1; Settings has it Auto (the default), On or Off. Auto turns
  it on for the rest of the page load once the median gap between frames drawn while the camera moves,
  over the last 40 (gaps over 250 ms and frames while a game loads left out), is over 40 ms.
- **A hidden page** [T-826; notes/T-826.md]: the browser stops animation frames and slows the main
  thread's timers to once a second (once a minute after five hidden minutes); worker timers run on
  at nearly full rate. With "Run in background" off (the default) the clock freezes while hidden, so
  the game pauses. With it on the clock runs on: it is anchored to wall time, the clock worker
  settles money and autosaves on its own timers from the last clock message, and what the main
  thread moves with frames (the shown minute and day, the render epoch with its trips and boarding
  hour, and the demand level effects that follow the minute) catches up at the first frame back.
  Hours that pass hidden pay their fares evenly within the hour (no boarding hour was worked out for
  them), with the same whole-hour totals. Station faces and sounds stop while hidden either way.
- **Simulation and demand**: Rust compiled to WebAssembly in Web Workers: one clock worker (track
  model, lines, money, the boarding account, autosave) and a pool of demand workers. The crate
  builds two WASM modules, the clock worker's and the demand workers', so each downloads only its
  own; the main thread loads neither. Both at opt-level 3 (the pair loop is the recompute;
  notes/T-051.md) and with 128-bit WASM SIMD (`+simd128`, set for the wasm target in
  `sim/.cargo/config.toml`, so both wasm-pack builds take it; Chrome 91, Firefox 89, Safari 16.4
  and later; T-036). Flat typed arrays, transferable between threads without copying.
- **Previews** [T-1039, `ClockClient.preview`]: the clock worker works out one preview (a ghost) at
  a time; a newer request replaces one waiting on the main thread and marks the one in flight
  unwanted (`previewStale`, outside the worker's queue), which is answered at once if it has not
  started or is waiting for water masks; the end of a run (Esc, another tool) and of a drag drop both.
  The router worker's route search and fit stop at their next look (every 20,000 states, every 25 ms)
  when a newer route is asked or the pointer leaves real track.
- **The demand pool** [T-149, T-173, `workers/demandClient.ts`][T-381]: 3 solving workers on 12 or more
  logical threads, 2 on 6 to 11, 1 on fewer; then `navigator.deviceMemory` 4 GB or less gives 1, and
  unreported memory at most 2; a big game may then drop some by memory [T-1205, below]. The Settings "Use more cores" restores half the threads less one, 1 to 5
  (one per period), at the next load; `?demandWorkers=N` overrides both. Plus one auxiliary worker with no periods
  that answers at once what needs no period's result (a city's cells, each bubble's commuters by
  mode). Worker i owns periods i, i + n, .... Every worker opens only the packs the game needs, each
  its own copy: the viewed zone first (only above zoom 7.5, docs/zones.md, T-871), and every zone with a built station inside its footprint or
  within 4.5 km of it, since only those can put a rider on a line (`workers/cityReach.ts`); a zone
  viewed or on screen, or crossed by built track [T-1047], that no station reaches opens bare in the
  auxiliary worker alone [T-325, [demand-runtime.md](demand-runtime.md) "Bare zones"], the viewed zone of a new game from the start [T-351]. Solving
  workers also open the regional market once a market zone is viewed or reached or a station lies
  within its reach [T-396]. A solve waits until every pack it needs is open in
  every worker, so a published result never leaves out a constructed city. Each worker solves the
  network's clusters (populations and the lines they reach, sharing no rider) apart, and after an
  edit only those it touched [T-326, [demand-runtime.md](demand-runtime.md) "Clusters"]; within a
  cluster each city's layers solve on the stations within its reach, the market on the whole
  cluster [T-756, "Each population on its reach"]. Worker 0 first opens a
  small synthetic everyday layer (`warm.rs`) and the others get packs only once it is ready, so
  every worker opens on optimised code [T-173]. The clock worker's network snapshot after each
  service change goes to every worker; a worker takes a newer one between populations. Messages:
  `workers/demandProtocol.ts`; results land in `game/demand.ts`. A sums worker with no WASM
  (`workers/demandSums.worker.ts`) adds up what the main thread used to [T-1208, T-1210]: each
  published round's views of the four kinds of day, applied on the main thread in the order published,
  and a selection's commuters (dealt for the display, `game/farChunks.ts`) and longer trips and a
  station's; the main thread does the same sums itself only if that worker fails. Timings:
  notes/demand-measurements.md, notes/demand-kernel-history.md.
- **Only final results replace final results** [T-232, T-324]: after an edit every figure keeps the
  last finished forecast until the new solve's last round lands; earlier rounds are shown only where
  nothing final is (startup, a whole-game load, a city just opened), and for lines new or changed by
  the edit, which take the free-flow estimate's riders as soon as it lands. A cluster solved ahead
  of the rest of its solve replaces its own figures with its final as soon as it is done [T-392].
  What the player sees:
  [ui-panels.md](ui-panels.md) "Loading".
- **Loading and readiness** [T-149, T-172, T-180, T-243, T-347, T-351]: input is never blocked. A
  loaded game holds only the clock until every needed city is open and the fresh solve's final
  round has landed (`HOLD_ROUND`, `holdDone`; Anita, T-347). A viewed zone open bare is final from
  its opening: a new game (no station) opens its viewed zone bare, its solve covers the market
  alone, and the card goes when the bare figures arrive (T-351). The cached result
  (section 11) is drawn behind the loading card without releasing it. A city opened later (a pan,
  construction) is a load without a hold and counts as done at its first estimate (`READY_ROUND`).
  **Startup order** (main.ts) [T-677]: the page sets up the map, overlay and UI, then waits on the
  start menu ([ui-panels.md](ui-panels.md) "Start menu"); the pick starts the demand pool, then the
  clock worker (`ClockClient.start`, which queues anything posted before it), so nothing but the
  map loads under the menu and a load from the pick runs on a map already drawn. A URL switch that
  chooses the game (`?new=1`, `?synth=1`, `?demandTest=1`, `?paused=1`, `?continue=1`;
  `choiceFromUrl` in ui/StartMenu.tsx) skips the menu, and the pool then starts before the map is
  set up. A new game, from the menu or `?new=1`, opens its viewed zone
  bare (`startDemand`'s `newGame`). **A save's packs open while its network is worked out**
  [T-889]: the clock worker reads the save's built stations (`TrackApi.save_stations`) before it
  derives the track and sends them (`stationsAhead`); the pool
  downloads and opens the zones they reach and those zones' markets at once, as prefetched zones, so
  no solve covers one the network does not reach, and a market the network does not want closes. **The registry is split** [T-776]: `game/registry.json`, bundled,
  is the index of what is needed before a zone opens (each zone's id, name, kind, country, origin,
  centre, zoom, box, place, UTC offset and `localTrips`; the market's frame, rule and place names);
  the world market's rows, the only per-zone detail, are `game/market-rows.json` (a hashed file in
  a build), downloaded once by the page beside the first market pack and sent with it to the
  solving workers, which cannot open a market pack before them. The regional market goes to the workers as bytes and is decoded there; each zone's
  boundary file is fetched once, shared by the reach test, the focus and the outlines
  (`game/boundaries.ts`). The clock worker is ready without any water mask: masks, outlines, place
  lists and boundaries are fetched per zone as the view, track or a search needs them (T-302,
  T-400, T-404; docs/zones.md 6).
  A pack download that fails is tried again after 0.5, 1.5 and 4 s, and a city pack's header is
  read again after its .bin and both fetched again if it changed (a pack rewritten mid-load). A
  zone whose pack still cannot be downloaded is left out (not needed, waited for or held for),
  its failure shows on the status tag while it is in view, and it is tried again after 10, 30 and
  then every 60 s; demand everywhere else carries on (T-484). A pack that downloads whole and
  does not open in a worker (a parse or WASM error, or a market pack whose zone count disagrees
  with its market row) is treated the same way: the worker answers its `open` naming the pack
  in `rejected`, every worker drops it, the zone is left out and flagged, and it is tried again
  once after 10 s; a second failure is final until the page is reloaded (T-492). Only an error outside opening a pack, other than a panic (below), stops the pool ("Demand could not load" at startup, the
  "Demand stopped" alert later, the figures shown kept [T-757]). T-950: a clock worker that fails
  before its first game (its init throws and posts `failed`, or its script errors) stops the pool
  too, as does an uncaught error in main.ts's setup before `setupDone()`; the loading gate's
  watchdog (ui/LoadingScreen.tsx, `markAlive` in game/loading.ts) stops a card quiet for 3 minutes,
  10 with a bar.
  `checks/every-pack-opens.mjs --broken` serves a broken city pack and market pack.
  `game/loading.ts`; timings in notes/T-172.md, frame times in notes/T-180.md.
- **A Rust panic in the track model** [T-1007] leaves its wasm-bindgen borrow held, so the clock
  worker makes a new `TrackApi` and opens the network of the last load or edit in it as a load
  does (an edit's panic always goes back to before the edit; any other uses the live network when it
  can still be read), then the main thread logs the panic and says so in the notice area. Undo
  history and the draft in hand are gone, as after a load. A fourth panic in a minute, or a ninth in all, leaves it down
  with a notice to reload (the autosave holds the game). `?debug=1` on the dev server has
  `tw.client.debugPanic("now" | "read" | "afterEdit")`.
- **A Rust panic in a demand worker** [T-1174] (out of memory aborts the same way) leaves the
  objects it was in borrowed, so the worker posts `dead` with `last_panic()`'s text (the panic
  message, or the allocation that failed) and does nothing more; a pack that fails to open with a
  panic is never taken for a broken pack. The page terminates it, starts a new worker in its place,
  downloads again and reopens everything it held, then sends the newest network again under a new
  version (`freshParts`: every worker composes its market afresh from the packs open, as the
  replacement did, so all hold the same corridor table, T-1354); the shown figures stay until that final. A second panic before the replacement has seen
  a solve through, or a fifth replacement in the page, stops the pool as before. `?debug=1` on the
  dev server has `twDemand.client.debugPanic(worker, "now" | "solve")`; check `t1174-panic-recover.mjs`.
- **Demand workers' memory** [T-1205] (`demandClient.ts` "T-1205: memory"): WASM memory never
  shrinks, so each worker reports its memory with its heap in use and its last solve's heap peak
  (`wasm_heap_bytes`, `wasm_heap_peak`; the peak is reset as each solve starts and forgotten when
  anything closes). A worker's need, what a worker started afresh reaches, is the larger of its heap
  at rest times 1.38 and that peak times 1.07, plus 64 MB. Once nothing is in flight (no solve, load,
  open, close or question, no replacement) and the network has been still for 20 s, every worker
  whose memory is at least 384 MB and 15% above its need is restarted through the panic path above,
  together and without counting as a panic; the network is solved once more, questions wait for that
  final, and a worker is not restarted again until its memory or need has moved by 384 MB. Before
  each solve the pool drops solving workers while 0.6 GB for the page, each worker's need plus 0.7 GB
  for its JS heap (or, on a cross-origin isolated page, `measureUserAgentSpecificMemory`'s measure of
  everything outside the WASM) would pass `memBudget`: 8 GB (Anita's target, T-1199), `deviceMemory`
  x 0.75 under 8, 6 GB where it is not reported. The periods are dealt again (`periods`, taken up
  between solves) and every worker composes afresh; no figure changes. It never grows back in a page
  (T-1366). `?perfOff=restart,cap` turns either off; check `t1205-memory.mjs`.
- **Threads talk by snapshots**: a worker transfers a finished buffer with a version number; the
  main thread swaps it in. Works on any host. SharedArrayBuffer is a later option; it needs
  cross-origin isolation headers, which GitHub Pages (Anita's website) cannot send and Cloudflare
  Pages can.
- **Data pipeline**: Python in the maps venv, offline, calling the sim crate natively where the game
  needs the same computation (the zone gravity, the regional market's all-pairs figures). One **pack**
  per zone (boundary, cells with population, jobs and attractor weights, the solved gravity, road
  speeds per period, fitted constants, sources; format in notes/T-004.md) and the regional market
  pack (sites, commute list, trip parameters, the `derived` block). Binary arrays with a small JSON
  header. Locally they sit beside the page (`packs/<id>.json`, `.bin`, `packs/intercity.json`);
  published, they are on R2 under a versioned prefix that `tools/deploy.py` writes into the page as
  a `tw-data` meta, never chosen by hostname (`game/dataBase.ts`, T-924; T-060, notes/T-026.md).
  The main thread fetches each once and hands each worker a copy.
- **Determinism**: the simulation is a pure function of (save, pack versions, game time). For the
  hour or two after an edit to a line's service the trains drawn also depend on the game time the
  edit took effect; the save carries it (the service's eras, T-206, T-1071).

## 11. Saves

A save is the player's inputs only: track, stations, lines, schedules, fares, each line's station
note [T-706], money (cash, cars owned of each kind, the ledgers' last 30 days and 48 hours, for the network and for each region, which are the
Money tab's history charts: fares, costs, net and train rides; T-447, T-531, T-565, T-567), clock,
empty line drafts, custom badge symbols, the line-name counter, and the player's station links. Population, jobs and travel come from versioned packs;
all results are recomputed. Sizes: notes/T-040.md, notes/T-567.md.

- **Format** [T-029, `workers/saveFile.ts`]: `TWG1`, a JSON header and the track model's network
  bytes, gzipped. The header keeps `city: nyc` and adds optional region and pack metadata. The
  network bytes are TWT6 (the Mercator world frame, T-306), which is TWT5 byte for byte but for the
  magic; its origin fields keep the game's home origin, unused by the frame, and its radii are ground
  metres. After the lines come optional trailing blocks (`sim/src/track/save.rs`): `LUI1` (custom
  badge symbols and the line-name counter, T-110/T-111), then, each only when there are any, `LNK1`
  links [T-197], `LAU1` lines kept by headway [T-625] and `LHS1` lines running high-speed trains
  [T-817]; a save without one loads with none. While a line's trains change over after a service
  edit, the network bytes end with the service's eras and shifts, their length and the tag `ERA1`
  (`Service::encode_eras`, T-1071), so the trains load where they were; the network decoder stops
  before them. Formats older than TWT4 (TWT2, TWT3) no longer open: the load says the save is from
  an older version (T-1146). A line saved with no train allocation (a frequency in trains per hour,
  as every line was before T-104) is given the whole trains that rate needs over its run time when
  the save loads (`TrackWorld::allocate_unallocated`), and its trains per hour follow from them
  like any other line's. Every zone's track, service and money is in the one
  network [T-116]. The header's `ledger`, `hours` and `regions` (per zone id, "" for outside every
  zone) hold the history charts' data, loaded back as they were, so a reload keeps the charts; money
  is saved to the dollar and rides to a tenth. Ledger entries' `points` are whole points; the
  header's `stars` keeps each region's star, the game second it was earned. A field a save lacks
  reads as empty: no `rides` or `points` as 0, no `hours`, `regions` or `stars` as none, no
  `fleetFast` as no high-speed cars. A header `points` block (per region totals, written by builds
  between T-684 and T-694) is ignored. Ledger entries after the header's clock are dropped on load
  and their net returned to the cash ([track-and-service.md](track-and-service.md) 6.4, T-715).
  `mode` (`normal` or `sandbox`), `start` (the start zone's id, absent for a start on the world map)
  and `opened` (the zones opened in this game, in order: the start zone at first, then each zone
  reached by a station or crossed by built track; T-729, T-1047, read only for the zoomed-out dots,
  T-976, docs/zones.md 6) come from T-728; a
  save without them reads as normal with no start and takes the zone in view and the zones reached
  as its opened list. The slot's summary has `mode` too, for the saves cards. The clock worker sends
  the list (`opened`, before a game's snapshot) and the main thread sends it back whole when it
  grows.
- **Removed zones** [T-1053]: a save opens if its header names the registry's world (or none, as
  legacy New York files do); the zones it names need not all exist. The header's `cities` lists
  every zone registered when it was written, so a replan that merges or re-splits zones would
  otherwise shut out every save. As it loads (`currentZones` in `workers/saveFile.ts`), each zone id
  in `cities`, `start`, `opened`, `regions`, `stars` and the line notes' cities that the registry
  lacks becomes its successor from `app/src/game/zone-successors.json` (`zoneNow`, followed through
  later removals); a removed zone's ledgers add into its successor's, its star keeps the earlier
  time, and an id with no successor is dropped (its ledgers go to "", outside every zone; a line
  note keeps the old id, which only counts that zone's notes). Track, stations and lines are
  positions and need nothing. The table is written by `pipeline/zone_successors.py` whenever zone
  ids leave the registry: the successor is the zone whose outline holds the removed zone's old
  centre, else the zone with an outline point nearest it within 30 km, else none; `--set id=succ`
  is a hand choice. Smoke's Node check `t1053-zone-ids.mjs` fails while an id in HEAD's registry is
  in neither the working registry nor the table; `checks/t1053-load-check.mjs` loads saves naming
  three removed zones through a scratch registry.
- **Sandbox** [T-728]: the track model's `set_unlimited` makes every money check pass (the edit
  still pays, so the cash can go below zero); the clock worker sets it from the header's mode and on
  the one-way `sandbox` message from Settings.
- **Older frames** [T-308; [world.md](world.md)]: a file or slot in TWT5 (the Northeast conic) or
  TWT4 (equirectangular) is converted to Mercator as it loads (`convert_save_frame` in the
  clock worker) and the slot written again at once in TWT6; the old bytes stay under a hidden key
  `old-frame-<slot>`. Cash, fleet, fares and ledger are copied as they are; track lengths follow the
  true ground.
- **When it saves**: the clock worker builds, compresses and writes to IndexedDB itself: at once
  after anything paid for, else once a wall minute if something changed, and when the page is
  hidden; a game running in the background keeps saving once a wall minute [T-826]. On start the player picks a slot or New game on the start menu [T-677]; `?continue=1`
  opens the slot opened last. `?new=1` starts a game that reads no slot and owns none, and
  does not autosave until "Save a copy" gives it one [T-320]; `?synth=1` and `?demandTest=1`
  load the last game but never autosave it.
- **No slot is written over with another game** [T-743]: New game (the start menu's beside saved
  games, and the saves panel's), Load a file and a backup's Restore as a copy each put the game in a
  new slot, after saving the game being played, which keeps its slot. A new game's slot is named
  after its start (`newGameName`: "Tokyo", "Tokyo sandbox", "World" for a map start, then " 2" and
  on if taken; T-728) and written once something changes, or at once when it starts from a real
  network; a loaded file's is
  named after the file without `.save` and written at once; a restored backup's is the slot's name
  and the backup's time, "<name> at 3:28 AM", any such time already on the name dropped first
  (`forkName` in workers/cloud.ts, also for account restores and forks; T-1363). A tab without its slot's lock must save a copy before leaving a changed
  game. On a test page the game is replaced in the page and owns no slot. A slot that cannot be
  read on start stays as it is (its bytes also under `unreadable-<time>`) and the new game gets a
  slot of its own ("Game N"). The start menu's New game always takes a new slot, even with nothing
  saved (T-728); the original slot `autosave` is only ever an old one.
- **Real networks** [T-731]: `packs/real/<id>.save` for New York, London, Paris, Tokyo, Seoul,
  Toronto and Sendai (`pipeline/realnet.py ship` copies them from realnet's game
  files into `data/packs/real/`, and the build copies that folder). Each line has its own double
  track at its own level. Between stations it follows the real track [T-1303]: each hop routed over
  the real-network graph (the track tool's router and fit, run when the saves are built, never in
  the game), lines over the same track set 6 m apart side by side; a hop with no real track near
  (trams, monorails, people movers: about a sixth in Paris and Tokyo, most of Toronto's streetcars)
  keeps its GTFS shape or a straight line. A sandbox started from one loads
  it as a new game: its network, fleet and fares, with the clock, cash, ledgers and stars of a new
  game. They are in older frames and converted as they load (above).
- **Backups** [T-744]: two per slot in the same store, `backup:<slot>:0` and `:1`, the slot's record
  without its picture. An autosave at least 5 wall minutes after the slot's newest backup (or with
  none) is also written over the older backup, so backups only follow changes. The saves panel lists
  the playing slot's under its card, each with Restore as a copy. Deleting a slot deletes them.
  `?backupMs=` shortens the interval for checks.
- **Outside the browser's storage** [T-1214]: a full disk let Chrome empty the site's IndexedDB
  (2026-10-24). Once a slot is written the page asks `navigator.storage.persist()`; a folder the player
  picks (File System Access; its handle kept as `copy-folder` in the same store) gets every slot from
  the clock worker when picked and each slot at every autosave, `<name> (<4 letters of its key>).save`
  written over each time.
- **Switching games reloads the page** [T-1196]: Open, Load a file, Restore as a copy and New game
  post `leave` (the game in hand saved, its lock let go; a file, backup or new game's spec kept as
  `pending-load` in the same store), then reload with `?slot=<key>` or `?load=<kind>`; the new page's
  clock worker takes `pending-load` as it starts. Loads posted straight to the worker still replace
  the game in place.
- **Named browser slots** [T-068, T-241]: save the current game as a new copy, open another, rename,
  delete an inactive slot after confirmation. The selected slot is remembered; the original autosave
  stays at its key ("Original game"). Each slot write stores a `SlotSummary` beside the bytes: a 384
  x 256 PNG of the track and line routes drawn in the worker on an OffscreenCanvas, the stations'
  cities, line and station counts, cash. Switching waits for the current game's save transaction to
  commit. The panel: [ui-panels.md](ui-panels.md) "Saved games".
- **One writer per slot**: a worker-held browser lock lets one tab autosave each slot. Another tab
  can play, read and export that game but shows a persistent Autosave off notice; different slots
  save concurrently. **Saving by hand** [T-1251, `saveByHand`]: the saves panel's Save or Ctrl+S
  writes the slot at once (an empty new game too). In a tab without the lock it takes it
  (`navigator.locks` `steal`): that tab then autosaves, and the tab it was taken from sees its lock's
  request reject (`SlotLease.lost`), stops autosaving and says "Autosave is off here because this
  game was saved in another tab."; a save by hand there takes it back. Unsaved changes in a blocked tab must go to a new slot before switching (or be
  exported). Rename and delete refuse slots owned by another tab. Storage failures keep the dirty
  state and a persistent notice, with file export still available; browsers without locks disable
  browser writes. notes/T-029.md, notes/T-068.md.
- **Demand cache** [T-328, `workers/demandCache.ts`]: each final demand result is kept beside its
  slot in the same IndexedDB store (`demand:<slot>:<network hash>`, one per slot; a deleted slot
  takes its cache along; sizes and timings in notes/T-328.md). It holds what
  the page shows from a final solve (every kind of day's lines, stations, purposes, each solved
  zone's split, the market's trips, and the viewed zone's weekday bubbles of every purpose when they
  were on show), not worker state; lines and stations by their place in the network, since a load
  renumbers ids. Its key is a SHA-256 of the network demand reads, of every pack the solve covered
  and of the market, the app build, the worker count and the perfOff flags; the zones must be the
  ones the new solve covers. When a whole game loads and the key matches, the result shows at once
  (the bubbles once the zone's cells are open), the loading card stays up and the clock held until
  the fresh final (T-347), the cached figures drawn behind the card with the status "Updating
  demand"; the workers solve as on any load, their estimates not shown, other purposes' or days'
  bubbles wait for them, a click is
  answered from the newest round the workers reach ([demand-runtime.md](demand-runtime.md) "The pool", T-327),
  and the fresh final replaces the cache in one step (it is bit for bit the same). A test page, `?new=1` and a tab without the slot never
  write it. The write runs in a worker of its own (`workers/cacheWrite.worker.ts`, T-789), the
  market's corridor table (shared by the days) copied once and transferred to it.
- **Old saves are a nicety** (Anita, 2026-10-12): only Anita has saves, so no real effort goes into
  migrations.
- **Games on the account** [T-988; `workers/cloud.ts`, `game/accountGames.ts`; the server's routes
  in shonei-server's `docs/trainworld-api.md`; notes/T-988.md]: while signed in (and only where the
  server's `/noritetsu/config` says `trainworldGames`), each slot is also kept on the account, the
  browser staying the working copy. A slot's tie is its record `sync:<slot>` in the same store
  (deleted with the slot): the game id (the leaderboards' id too, folded in from
  `trainworld.score-games.v1`), the account, the rev it last uploaded or downloaded and the slot's
  `at` then (the slot has changed here when its `at` differs), the last upload's time, and the
  given-way and kept-here-only marks. A slot first written while signed in is tied to that
  account; one made signed out is tied to none until the sign-in question (ui-panels.md "Account
  and leaderboards"). The device is one browser profile, a random id in localStorage
  (`trainworld.device.v1`). The main thread posts `{kind: "account", ctx}` to the clock worker on
  every session change; a 401 there posts `accountGone` back and the page signs out.
  - **Uploads** (the clock worker, after `writeSlot`, beside T-1214's copy): the save bytes as Save
    to a file writes them, the name, the card's facts and picture. A save by hand goes at once
    with `force`; an autosave at most every 2 wall minutes (`GAP_MS`), a timer sending what the
    gap held back; the page hiding sends at once (best effort); a rename at once. A save over
    4 MB is never sent (the server closes the connection past 4.6 MB).
  - **On load**: the start menu settles every slot it can lock before the worker starts
    (`settleOnStart`), and the worker again on start and on `online`: a slot changed here goes
    up; an unchanged one takes a newer account copy (slot record from the server's facts and
    picture, nothing decoded); one the account deleted (a tombstone) is deleted if unchanged, else
    kept as a new account game; deletions made here while offline are sent. The game opened is
    claimed (`claim`); an unchanged slot behind the account is replaced in place (`loadGame`
    through the game's message queue).
  - **Two devices** work as two tabs do: the last device to open a game is its writer; another
    device's upload hears 409 "elsewhere", stops autosaving the game, locally too, and says
    "Autosave is off here because this game was opened on another device."; its save by hand
    claims it back with `force`. Its slot is marked given way, so its next load takes the account
    copy without a fork. **A fork** (changed here, and the account moved on from the same rev, say
    a game opened offline): this browser's version goes into a new slot "Tokyo at 9:41 PM", a new
    account game, and the old slot takes the account's copy; a game being played carries on in the
    fork's slot.
  - **Deleting** a tied slot deletes its account game (queued as `cloud-deletes` while offline);
    a game the account deleted, under a tombstone, never comes back under its id (410): a kept
    copy gets a new id. A game opened unchanged after another device deleted it is kept in this
    browser only.
  - **The account's backups** (two a game, made by the server) are listed under the playing game
    and restored as copies through Load a file's path (`restoreAccountBackup`); the route names a
    backup by place, so the page checks its `X-Rev`.
  - **Signing out with Remove** marks the account in sessionStorage (`trainworld.signout-remove`),
    goes to the start menu if a game is open, and there settles, deletes every slot of the account
    not changed since its last sync (never from the account; its score entry is forgotten here so
    its figures stay), then signs out (`removeOnStart`).
  - A page whose `/noritetsu/config` did not answer asks again on `online` and every 2 minutes.
  - `?new=1`, `?synth=1`, `?demandTest=1` sync nothing; localhost syncs only with `?acct=local` or
    `?acct=live` given once. `checks/t988-cloud.mjs` runs it against a local server.
- **Later**: sharing a game by link (T-989).
