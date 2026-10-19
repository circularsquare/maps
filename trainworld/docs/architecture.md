[Back to the spec](../SPEC.md)

# Architecture: performance ideas, technology and saves

Sections 2, 7 and 11 of the spec in detail. Where files live: `docs/code-map.md`.

## 2. The core performance idea

**Trains are cheap; passengers are expensive.** Subway Builder's changelog points at passenger
work: a timetable search per 200-person group, "15 minute lag spikes" from every group re-deciding
at once, freezes when drawing track on big networks. It is Electron + React + MapLibre, so the map
library is not the problem; bursty per-group passenger work is.

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
- NIMBY Rails hands the renderer a small motion snapshot each frame and routes passengers over a
  graph of transfer stations only; it simulates individual riders, and overfull queues slow its
  whole sim. The flow model is meant to go far past that.

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
- **Map**: MapLibre GL 6 for the basemap (it has a globe, which our layer does not draw on yet,
  T-033). Track, stations and trains are drawn by our own WebGL2 renderer on a **separate canvas
  over the map**: not GeoJSON sources (per-frame `setData` pins a core) and not a MapLibre custom
  layer (that repaints the whole basemap on every train frame). Track buffers are built in the clock
  worker and replaced on each edit, made from per-edge, per-line and per-station pieces the worker
  keeps and rebuilds only when their inputs change (`NetCache` in map/network.ts, T-496); trains are instanced and positioned on the GPU from keyframes
  and the clock. World coordinates and the moving draw anchor: [world.md](world.md).
- **Staying in step with the basemap** [T-075; notes/T-015.md, notes/T-075.md]: an empty MapLibre
  custom layer hands over the camera matrix on every frame MapLibre draws, and the overlay is drawn
  right there with the same matrix; our own frame loop drives the overlay only for the clock and
  skips duplicate draws when MapLibre has a frame queued.
- **Time on the GPU** [notes/T-018.md, notes/T-040.md]: keyframe times are relative to a render
  epoch at the start of the current game hour, so the float32 time stays under 3600 s and positions
  within about 2 cm. When the hour turns the clock worker lists the next hour's trips relative to
  the new epoch (computed in float64), and only that buffer is re-uploaded.
- **Frame loop** [T-073, T-084, T-097]: frames only while the clock runs; 10 fps while the window is
  visible but unfocused; nothing while hidden (the clock freezes) or paused; none while no trips run
  this hour (the clock wakes once a game minute instead). Otherwise the overlay draws on MapLibre's
  frames and once after a change to what is drawn. The demand views have a canvas of their own over
  it, drawn only on MapLibre's frames and after their data changes, never by the clock; bubbles and
  a selection's far end are summed in their own worker (`workers/commuters.worker.ts`).
- **Simulation and demand**: Rust compiled to WebAssembly in Web Workers: one clock worker (track
  model, lines, money, the boarding account, autosave) and a pool of demand workers. The crate
  builds two WASM modules, the clock worker's and the demand workers', so each downloads only its
  own; the main thread loads neither. Both at opt-level 3 (the pair loop is the recompute;
  notes/T-051.md) and with 128-bit WASM SIMD (`+simd128`, set for the wasm target in
  `sim/.cargo/config.toml`, so both wasm-pack builds take it; Chrome 91, Firefox 89, Safari 16.4
  and later; T-036). Flat typed arrays, transferable between threads without copying.
- **The demand pool** [T-149, T-173, `workers/demandClient.ts`][T-381]: 3 solving workers on 12 or more
  logical threads, 2 on 6 to 11, 1 on fewer; then `navigator.deviceMemory` 4 GB or less gives 1, and
  unreported memory at most 2. The Settings "Use more cores" restores half the threads less one, 1 to 5
  (one per period), at the next load; `?demandWorkers=N` overrides both. Plus one auxiliary worker with no periods
  that answers at once what needs no period's result (a city's cells, each bubble's commuters by
  mode). Worker i owns periods i, i + n, .... Every worker opens only the packs the game needs, each
  its own copy: the viewed zone first, and every zone with a built station inside its footprint or
  within 4.5 km of it, since only those can put a rider on a line (`workers/cityReach.ts`); a zone
  viewed or on screen that no station reaches opens bare in the auxiliary worker alone [T-325,
  [demand-runtime.md](demand-runtime.md) "Bare zones"], the viewed zone of a new game from the start [T-351]. Solving
  workers also open the regional market once a market zone is viewed or reached or a station lies
  within its reach [T-396]. A solve waits until every pack it needs is open in
  every worker, so a published result never leaves out a constructed city. Each worker solves the
  network's clusters (populations and the lines they reach, sharing no rider) apart, and after an
  edit only those it touched [T-326, [demand-runtime.md](demand-runtime.md) "Clusters"]. Worker 0 first opens a
  small synthetic everyday layer (`warm.rs`) and the others get packs only once it is ready, so
  every worker opens on optimised code [T-173]. The clock worker's network snapshot after each
  service change goes to every worker; a worker takes a newer one between populations. Messages:
  `workers/demandProtocol.ts`; results land in `game/demand.ts`. Timings: [demand-kernel.md](demand-kernel.md)
  4.4, notes/demand-measurements.md.
- **Only final results replace final results** [T-232, T-324]: after an edit every figure keeps the
  last finished forecast until the new solve's last round lands; earlier rounds are shown only where
  nothing final is (startup, a whole-game load, a city just opened), and for lines new or changed by
  the edit, which take the free-flow estimate's riders as soon as it lands. A cluster solved ahead
  of the rest of its solve replaces its own figures with its final as soon as it is done [T-392].
  What the player sees:
  [ui-panels.md](ui-panels.md) "Loading".
- **Loading and readiness** [T-149, T-172, T-180, T-243, T-347, T-351]: input is never blocked. A
  loaded game holds only the clock until every needed city is open and the fresh solve's final
  round has landed (`HOLD_ROUND`, `holdDone`); T-172 released it at the first estimate and T-328 at
  the cached result, but the game then lagged through the crowding rounds, so Anita had it wait
  (T-347). A viewed zone open bare is final from its opening: a new game (no station) opens its
  viewed zone bare, its solve covers the market alone, and the card goes when the bare figures
  arrive, about 2.7 s after navigation instead of about 7 s (T-351). The cached result
  (section 11) is drawn behind the loading card without releasing it. A city opened later (a pan,
  construction) is a load without a hold and counts as done at its first estimate (`READY_ROUND`).
  **Startup order** (main.ts) [T-677]: the page sets up the map, overlay and UI, then waits on the
  start menu ([ui-panels.md](ui-panels.md) "Start menu"); the pick starts the demand pool, then the
  clock worker (`ClockClient.start`, which queues anything posted before it), so nothing but the
  map loads under the menu and a load from the pick runs on a map already drawn. A URL switch that
  chooses the game (`?new=1`, `?synth=1`, `?demandTest=1`, `?paused=1`, `?continue=1`;
  `choiceFromUrl` in ui/StartMenu.tsx) skips the menu, and the pool then starts before the map is
  set up, as it did before the menu. A new game, from the menu or `?new=1`, opens its viewed zone
  bare (`startDemand`'s `newGame`). The regional market goes to the workers as bytes and is decoded there; each zone's
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
  with the registry's row) is treated the same way: the worker answers its `open` naming the pack
  in `rejected`, every worker drops it, the zone is left out and flagged, and it is tried again
  once after 10 s; a second failure is final until the page is reloaded (T-492). A market pack
  mismatched with its row, as in a snapshot built mid pack rebuild, is isolated per zone like any
  other. Only an error outside opening a pack stops the pool ("Demand could not load").
  `checks/every-pack-opens.mjs --broken` serves a broken city pack and market pack.
  `game/loading.ts`; timings in notes/T-172.md, frame times in notes/T-180.md.
- **Threads talk by snapshots**: a worker transfers a finished buffer with a version number; the
  main thread swaps it in. Works on any host. SharedArrayBuffer is a later option; it needs
  cross-origin isolation headers, which GitHub Pages (Anita's website) cannot send and Cloudflare
  Pages can.
- **Data pipeline**: Python in the maps venv, offline, calling the sim crate natively where the game
  needs the same computation (the zone gravity, the regional market's all-pairs figures). One **pack**
  per zone (boundary, cells with population, jobs and attractor weights, the solved gravity, road
  speeds per period, fitted constants, sources; format in notes/T-004.md) and the regional market
  pack (sites, commute list, trip parameters, the `derived` block). Binary arrays with a small JSON
  header. Today they sit beside the page (`packs/<id>.json`, `.bin`, `packs/intercity.json`); the
  main thread fetches each once and hands each worker a copy. Published, they move to R2 under a
  versioned path, compressed and cached for good (T-060, notes/T-026.md).
- **Determinism**: the simulation is a pure function of (save, pack versions, game time), except
  for the hour or two after an edit to a line's service, when the trains drawn also depend on the
  game time the edit was made [T-206; not saved, so a reload re-spaces them once].

## 11. Saves

A save is the player's inputs only: track, stations, lines, schedules, fares, money (cash, cars
owned, the ledgers' last 30 days and 48 hours, for the network and for each region, which are the
Money tab's history charts: fares, costs, net and train rides; T-447, T-531, T-565, T-567), clock,
empty line drafts, custom badge symbols, the line-name counter, and the player's station links. Population, jobs and travel come from versioned packs;
all results are recomputed. About 100 KB compressed per 10,000 route-km (T-040).

- **Format** [T-029, `workers/saveFile.ts`]: `TWG1`, a JSON header and the track model's network
  bytes, gzipped. The header keeps `city: nyc` and adds optional region and pack metadata. The
  network bytes are TWT6 (the Mercator world frame, T-306), which is TWT5 byte for byte but for the
  magic; its origin fields keep the game's home origin, unused by the frame, and its radii are ground
  metres. Optional trailing blocks: TWT4 metadata (drafts, symbols, the counter, T-110/T-111) and
  `LNK1` for links, written only when there are any [T-197]. Every zone's track, service and money
  is in the one network [T-116]. The header's `ledger`, `hours` and `regions` (per zone id, "" for
  outside every zone) hold the history charts' data, loaded back as they were, so a reload keeps the
  charts; money is saved to the dollar and rides to a tenth, which gzips to about 60% of full
  precision. An entry without `rides` (a save from before T-565) reads as 0 rides, and a save with
  no `hours` or `regions` starts those empty. Sizes: notes/T-567.md. Each ledger entry's
  `points` (whole points) and `stars` (each region's star, the game second it was earned) are
  absent in saves before T-684 and T-685, which start with none. A header `points` block (per region
  totals, written by builds between T-684 and T-694) is ignored. Ledger entries after the header's
  clock are dropped on load and their net returned to the cash ([track-and-service.md](track-and-service.md)
  6.4, T-715). `mode` (`normal` or `sandbox`), `start` (the start zone's id, absent for a start on
  the world map) and `opened` (the zones opened in this game, in order: the start zone at first, then
  each zone clicked open or reached by a station; T-729, docs/zones.md 6) are absent in saves before
  T-728, which read as normal with no start and take the zone in view and the zones reached as their
  opened list; the slot's summary has `mode` too, for the saves cards. The clock worker sends the
  list (`opened`, before a game's snapshot) and the main thread sends it back whole when it grows.
- **Sandbox** [T-728]: the track model's `set_unlimited` makes every money check pass (the edit
  still pays, so the cash can go below zero); the clock worker sets it from the header's mode and on
  the one-way `sandbox` message from Settings.
- **Older frames** [T-308; [world.md](world.md)]: a file or slot in TWT5 (the Northeast conic) or
  TWT4 and older (equirectangular) is converted to Mercator as it loads (`convert_save_frame` in the
  clock worker) and the slot written again at once in TWT6; the old bytes stay under a hidden key
  `old-frame-<slot>`. Cash, fleet, fares and ledger are copied as they are; track lengths follow the
  true ground.
- **When it saves**: the clock worker builds, compresses and writes to IndexedDB itself: at once
  after anything paid for, else once a wall minute if something changed, and when the page is
  hidden. On start the player picks a slot or New game on the start menu [T-677]; `?continue=1`
  opens the slot opened last. `?new=1` starts a game that reads no slot and owns none, and
  does not autosave until "Save as a copy" gives it one [T-320]; `?synth=1` and `?demandTest=1`
  load the last game but never autosave it.
- **No slot is written over with another game** [T-743]: New game (the start menu's beside saved
  games, and the saves panel's), Load a file and a backup's Restore as a copy each put the game in a
  new slot, after saving the game being played, which keeps its slot. A new game's slot is named
  after its start (`newGameName`: "Tokyo", "Tokyo sandbox", "World" for a map start, then " 2" and
  on if taken; T-728) and written once something changes, or at once when it starts from a real
  network; a loaded file's is
  named after the file without `.save` and written at once; a restored backup's is the slot's name
  and the backup's time. A tab without its slot's lock must save a copy before leaving a changed
  game. On a test page the game is replaced in the page and owns no slot. A slot that cannot be
  read on start stays as it is (its bytes also under `unreadable-<time>`) and the new game gets a
  slot of its own ("Game N"). The start menu's New game always takes a new slot, even with nothing
  saved (T-728); the original slot `autosave` is only ever an old one.
- **Real networks** [T-731]: `packs/real/<id>.save` for New York, London, Paris, Tokyo, Seoul,
  Toronto and Sendai (7-80 KB each, `pipeline/realnet.py ship` copies them from realnet's game
  files into `data/packs/real/`, and the build copies that folder). A sandbox started from one loads
  it as a new game: its network, fleet and fares, with the clock, cash, ledgers and stars of a new
  game. They are in older frames and converted as they load (below).
- **Backups** [T-744]: two per slot in the same store, `backup:<slot>:0` and `:1`, the slot's record
  without its picture. An autosave at least 5 wall minutes after the slot's newest backup (or with
  none) is also written over the older backup, so backups only follow changes. The saves panel lists
  the playing slot's under its card, each with Restore as a copy. Deleting a slot deletes them.
  `?backupMs=` shortens the interval for checks.
- **Named browser slots** [T-068, T-241]: save the current game as a new copy, open another, rename,
  delete an inactive slot after confirmation. The selected slot is remembered; the original autosave
  stays at its key ("Original game"). Each slot write stores a `SlotSummary` beside the bytes: a 384
  x 256 PNG of the track and line routes drawn in the worker on an OffscreenCanvas, the stations'
  cities, line and station counts, cash. Switching waits for the current game's save transaction to
  commit. The panel: [ui-panels.md](ui-panels.md) "Saved games".
- **One writer per slot**: a worker-held browser lock lets one tab autosave each slot. Another tab
  can play, read and export that game but shows a persistent Autosave off notice; different slots
  save concurrently. Unsaved changes in a blocked tab must go to a new slot before switching (or be
  exported). Rename and delete refuse slots owned by another tab. Storage failures keep the dirty
  state and a persistent notice, with file export still available; browsers without locks disable
  browser writes. notes/T-029.md, notes/T-068.md.
- **Demand cache** [T-328, `workers/demandCache.ts`]: each final demand result is kept beside its
  slot in the same IndexedDB store (`demand:<slot>:<network hash>`, one per slot, 7.3 MB for the
  real New York save, 4.9 MB of it the bubbles; a deleted slot takes its cache along). It holds what
  the page shows from a final solve (every kind of day's lines, stations, purposes, each solved
  zone's split, the market's trips, and the viewed zone's weekday bubbles of every purpose when they
  were on show), not worker state; lines and stations by their place in the network, since a load
  renumbers ids. Its key is a SHA-256 of the network demand reads, of every pack the solve covered
  and of the market, the app build, the worker count and the perfOff flags; the zones must be the
  ones the new solve covers. When a whole game loads and the key matches, the result shows at once
  (about 0.1 s after the network is drawn; the bubbles once the zone's cells are open, about 3.7 s),
  the loading card stays up and the clock held until the fresh final (T-347), the cached figures
  drawn behind the card with the status "Updating demand" (hover text as before); the workers solve
  as before, their estimates not shown, other purposes' or days' bubbles wait for them, a click is
  answered from the newest round the workers reach ([demand-runtime.md](demand-runtime.md) "The pool", T-327),
  and the fresh final replaces the cache in one step (it is bit for bit the same). A test page, `?new=1` and a tab without the slot never
  write it.
- **Old saves are a nicety** (Anita, 2026-10-12): only Anita has saves, so no real effort goes into
  migrations.
- **Later**: sharing by link (upload a save, get a link, M4); accounts only for sync and sharing,
  on `shonei-server` (`~/projects/shonei-server`).
