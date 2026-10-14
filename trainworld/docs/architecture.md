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
  worker and replaced on each edit; trains are instanced and positioned on the GPU from keyframes
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
  notes/T-051.md). Flat typed arrays, transferable between threads without copying.
- **The demand pool** [T-149, T-173, `workers/demandClient.ts`]: half the logical cores less one,
  at most 5 (one per period), at least 1 solving workers, plus one auxiliary worker with no periods
  that answers at once what needs no period's result (a city's cells, each bubble's commuters by
  mode). Worker i owns periods i, i + n, .... Every worker opens only the packs the game needs, each
  its own copy: the viewed zone first, and every zone with a built station inside its footprint or
  within 4.5 km of it, since only those can put a rider on a line (`workers/cityReach.ts`); a zone
  viewed or on screen that no station reaches opens bare in the auxiliary worker alone [T-325,
  [demand.md](demand.md) "Bare zones"]. Solving
  workers also open the regional market, always. A solve waits until every pack it needs is open in
  every worker, so a published result never leaves out a constructed city. Worker 0 first opens a
  small synthetic everyday layer (`warm.rs`) and the others get packs only once it is ready, so
  every worker opens on optimised code [T-173]. The clock worker's network snapshot after each
  service change goes to every worker; a worker takes a newer one between populations. Messages:
  `workers/demandProtocol.ts`; results land in `game/demand.ts`. Timings: [demand.md](demand.md)
  4.4.
- **Only final results replace final results** [T-232, T-324]: after an edit every figure keeps the
  last finished forecast until the new solve's last round lands; earlier rounds are shown only where
  nothing final is (startup, a whole-game load, a city just opened), and for lines new or changed by
  the edit, which take the free-flow estimate's riders as soon as it lands. What the player sees:
  [ui-panels.md](ui-panels.md) "Loading".
- **Loading and readiness** [T-149, T-172, T-180, T-243]: input is never blocked. A loaded game
  holds only the clock until every needed city has its first estimate (every period's free-flow
  result); the crowding rounds refine it in the background. The demand pool starts before the map is
  set up; the regional market goes to the workers as bytes and is decoded there; each zone's
  boundary file is fetched once for the reach test and the outlines (`game/boundaries.ts`).
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
owned, the last 30 days' ledger), clock, empty line drafts, custom badge symbols, the line-name
counter, and the player's station links. Population, jobs and travel come from versioned packs;
all results are recomputed. About 100 KB compressed per 10,000 route-km (T-040).

- **Format** [T-029, `workers/saveFile.ts`]: `TWG1`, a JSON header and the track model's network
  bytes, gzipped. The header keeps `city: nyc` and adds optional region and pack metadata. The
  network bytes are TWT6 (the Mercator world frame, T-306), which is TWT5 byte for byte but for the
  magic; its origin fields keep the game's home origin, unused by the frame, and its radii are ground
  metres. Optional trailing blocks: TWT4 metadata (drafts, symbols, the counter, T-110/T-111) and
  `LNK1` for links, written only when there are any [T-197]. Every zone's track, service and money
  is in the one network [T-116].
- **Older frames** [T-308; [world.md](world.md)]: a file or slot in TWT5 (the Northeast conic) or
  TWT4 and older (equirectangular) is converted to Mercator as it loads (`convert_save_frame` in the
  clock worker) and the slot written again at once in TWT6; the old bytes stay under a hidden key
  `old-frame-<slot>`. Cash, fleet, fares and ledger are copied as they are; track lengths follow the
  true ground.
- **When it saves**: the clock worker builds, compresses and writes to IndexedDB itself: at once
  after anything paid for, else once a wall minute if something changed, and when the page is
  hidden. On start the last game loads. `?new=1` starts a game that reads no slot and owns none, and
  does not autosave until "Save as a copy" gives it one [T-320]; `?synth=1` and `?demandTest=1`
  load the last game but never autosave it.
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
- **Old saves are a nicety** (Anita, 2026-10-12): only Anita has saves, so no real effort goes into
  migrations.
- **Later**: sharing by link (upload a save, get a link, M4); accounts only for sync and sharing,
  on `shonei-server` (`~/projects/shonei-server`).
