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

### App, map and basemap

- **App and UI**: TypeScript, Vite, Preact with signals. No game engine: the map is the game. The
  main thread holds UI state and read-only snapshots from the workers (the `WorldView`: the save's
  inputs plus worker results) and changes the game only by sending edit operations to the clock
  worker ([track-and-service.md](track-and-service.md) 6.5).
- **Map**: MapLibre GL 6 for the basemap, flat or as its globe (a Settings choice, T-033). Track, stations and trains are drawn by our own WebGL2 renderer on a **separate canvas
  over the map**, neither GeoJSON sources nor a drawing custom layer (why:
  notes/architecture-history.md). Track buffers are built in the clock
  worker on each edit, made from per-edge, per-line and per-station pieces (with each line's train
  speeds and samples) the worker keeps and rebuilds only when their inputs change (`NetCache` in
  map/network.ts, T-496, T-1029). An edit's network is posted as changes to the last (`NetPatch`:
  stroke runs found by identity of the reused pieces, arrays as last posted left out; each stroke set
  is posted whole, as the last network's with no arrays, or as only its new segments, which the page
  writes over the ones they replace in the arrays it has (a segment that moves, because one was
  inserted or removed before it, makes the set whole again; `map/strokePost.ts`, T-1381); the renderer
  copies the unchanged runs on the GPU and keeps textures whose data is the same (T-1028), and the
  pick and snap grids are carried over through the same runs instead of built again (T-1030).
  Anything that cannot be taken as changes (a load, a new draw anchor, a mismatch) goes whole;
  `?patch=0` posts everything whole, `?patch=check` compares the changes with whole builds
  (notes/T-1029.md). Trains are instanced and positioned on the GPU from keyframes
  and the clock. World coordinates and the moving draw anchor: [world.md](world.md). A lost WebGL
  context (phones drop them) is taken on both our canvases; on restore the renderer and the demand
  canvas compile again and upload what they last drew, and the camera layer is added back to
  MapLibre's restored style (T-953). Shaders are written to WebGL2's guaranteed limits and highp
  floats in every fragment shader that measures px (notes/T-953.md), and each program's varyings to
  8 vectors, what Firefox with fingerprinting resistance allows (`checks/t3192-varyings.mjs`). A
  decorative layer's program the browser refuses is logged and the layer left out (`compileOptional`
  in map/gl.ts); a view wider than the context's largest renderbuffer draws its strokes every frame
  without the stroke cache (T-3192).
- **A state and its trips land together** [T-648]: the clock worker posts each state, then its
  trips, before anything else. The client holds the state until those trips (at most 250 ms), so
  the world, the network and the trips are taken in one task and no frame draws old trips on new run
  profiles; a state built at another draw anchor, or one with no trips to wait for (`noTrips`), is
  taken at once. **Previews go before an edit's follow-up** [T-1407]: after applying an edit the
  worker answers the track previews and station checks asked meanwhile, then builds and posts its
  state, trips and boarding hour and answers it (`later` in the clock worker; any other message, or
  trips sent from a timer, runs the follow-up first, so nothing else sees the model ahead of its
  state). The track tool's blueprint segment that changes no line for demand asks `early`: it is
  answered before its state, the tool asks the next ghost from the answer's node, and the ghost
  just placed is drawn as the stand-in for the new blueprint until the next network is set
  (`Renderer.keepPreviewAsPlaced`). **Its state waits for the pointer to rest** [T-2404]: the
  worker holds the follow-up while ghosts keep being asked, until none has been asked for 220 ms,
  5 s after the first segment held, or a message other than a preview, a line query, demand's
  results, the clock or the next early segment comes, which joins it (one state for both); the
  hourly trips go without it, since a blueprint changes no line's run. A segment that paid (a
  junction off built track) is not held. Each held segment is drawn as a stand-in: its ghost, or
  the edges the early answer carries (`held`) when no ghost of it was answered
  (`Renderer.addPlaced`). The tool asks for the state at once (`release`) when the run ends and
  when the pointer comes within 24 px of a held segment away from the start, since the snap sees a
  segment only once its state lands; a click there waits for that state (at most 3 s) before it
  snaps. The client takes the state after the next paint (`ghostFirst`). **A stop click's rows go
  first** [T-2766]: the line tool's stop clicks ask `rows`; the worker applies the edit, posts a state
  of the rows alone (`rows`: world and money, no network, no trips; the renderer keeps its network)
  and then answers, so the next click reads the stops this one left and a refusal for money comes at
  its click. The network, trips, boarding hour and demand's snapshot follow once no stop click has
  come for 500 ms (5 s at most); a stop click before then joins them (one full state for the run),
  and the line tool's reads of the lines (`stopTiers` and the like), demand's money results and the
  clock pass it, anything else runs it first. Meanwhile the client holds every other read until no
  stop click has been answered for 500 ms (the panels ask a dozen after each world). The clicked marker is drawn
  in its new tier at the click (`StopMarkers.guessClick`, settled when the edit is answered) and a rows world
  rebuilds no demand layers (`worldIsRows`) [T-3159]. A state carries only the edges,
  nodes and lines that differ from the last one posted, with the ids in order when they changed
  (`delta`, T-1029); the client rebuilds the full lists from its last ones before anything reads it.
  The worker itself reads from the track model only the rows an edit changed: it keeps the last
  rows, the track model lists the lines whose service or net was touched and the stations whose name
  was set since the last read, and a station's name is kept by node (`stateRows.ts`, T-1382). The
  run profiles' phase tables are kept per line the same way and laid end to end only when a line's
  differ; what the track model prices (the cost sums, the construct-all quote) asks the water mask
  once for a node while the node and the mask stay as they are (T-1540). The construct-all quote and
  a planned line's price each run a trial construction, so they are no part of a state: the Build
  tab and the line panel ask for them (T-1171).
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

### Frame loop and the draw caches

- **Frame loop** [T-073, T-084, T-097]: frames only while the clock runs; 10 fps while the window is
  visible but unfocused; nothing while hidden or paused; none while no trips run
  this hour (the clock wakes once a game minute instead). Otherwise the overlay draws on MapLibre's
  frames and once after a change to what is drawn. [T-1640] While the camera is still, the clock's
  frames come only as often as something visible moves half a pixel (`renderer.stillWaitMs`): trains
  from TRAINS_ZOOM, at 100 m/s of game time at the view's latitude (full rate at every speed from
  zoom 10); platform crowds from STATIONS_ZOOM, in steps of 5 game seconds; otherwise one draw a
  second, which keeps the hour's trips and staged states current. The game minute turns on its
  own timer meanwhile, and a MapLibre frame, `invalidate` or a display change plans the next frame
  again (`Overlay.replan`), so zooming in to the trains, selection, hover and edits draw at once.
  The camera moving, the complex map's snapshot and the 10 fps unfocused rate are unchanged.
  `?perfOff=stillThrottle` (or `overlay.throttle = false`) draws every frame. The demand views have a canvas of their own over
  it, drawn only on MapLibre's frames and after their data changes, never by the clock; bubbles and
  a selection's far end are summed in their own worker (`workers/commuters.worker.ts`).
- **The stroke cache** [T-1638; `map/strokeCache.ts`, notes/T-1618.md]: everything the renderer
  draws before the stations (car parks, bus lines, the track and line strokes, highlight, ghost and
  crossing marks) is drawn into an offscreen target with the canvas's own MSAA and stencil, resolved
  into a texture, and copied onto the canvas each frame with the strokes' premultiplied blend; the
  stations, crowds, bubbles and trains draw over it every frame as before. The track through the
  station boxes is a second texture, copied through the stencil where it was drawn. Both are drawn
  again only when the key changes: the camera (matrices, zoom, pixel ratio, canvas size), the
  selected line and edge, the hovered line and edge, the colour view (height, speed), line width,
  dimming, crossing marks, the shade, the bus lines, and any GL upload or deletion made outside the
  per-frame passes (counted on the context: a network, preview, highlight, demand view or traffic
  view change). The complex map's snapshot has a cache of its own. `?perfOff=strokeCache` (or
  `renderer.strokeCache = false`) draws the strokes every frame.
- **Fewer stroke instances** [T-1639; `map/strokeLod.ts`, notes/T-1618.md]: below zoom 15 (plus log2
  of the pixel ratio) the network's track and line strokes are drawn from a simplified copy per
  whole zoom band: each run of joined segments of one line or track, edge, level, flags, stretch
  and speed (within 10 km/h) is cut to the Douglas-Peucker points at 0.25 device px of the band's
  top zoom, each merged segment carrying its first segment's values (the demand widths and traffic
  marks gathered by that index). A copy is made after the first draw that needs it, in 8 ms slices
  between frames once the runs' significance is worked out (likewise in slices), one band at a time;
  the full set is drawn until then. Copies are dropped with the network (T-2403). From zoom 7 only the chunks whose box may show (all four corners not off one side
  of the view, 48 CSS px out) are drawn, in order, so which line is drawn over which is as before.
  Chunks are 16 instances in boxes of 8 boxes, level on level; a box wholly in the view is drawn
  without testing what is in it, and below zoom 12 the testing stops at boxes of 128 instances
  (zoom 10-12) or 1024 (below 10), since each test costs more in the globe's projection.
  [T-1644] When at most 40,000 instances show, they are gathered into one small buffer set
  (`map/strokeGather.ts`, from the arrays the strokes were made from) and each pass is one draw;
  drawing them range by range rebinds ten attribute buffers per range per pass, hundreds of ranges
  at zoom 15, which cost more than the strokes. More are drawn by ranges, gaps under 512 instances
  drawn through. `?perfOff=strokeLod`, `strokeCull` and `strokeGather` (or `renderer.strokeLod`,
  `strokeCull` and `strokeGather`) switch them off.
- **While the camera moves** [T-1207; `map/moving.ts`]: work done for each MapLibre frame that need
  not be runs at most every 100 ms (`MOVING_PLACE_MS`) and in full on the frame after the move ends:
  the DOM placers (station names, capacity tags, attraction marks), whose marks
  move as one with a flat pan in between (capacity tags and attraction marks instead follow their
  resources one by one every frame, station names too for any camera change but a flat pan,
  T-1614, T-1657, T-1621; zone loading bars are placed on every move, so they hold their spot through a zoom, T-1804); the platform crowds; the station sounds and faces sample
  half as often; the Lines tab's live counts wait. The lighter map (`map/lite.ts`): the basemap
  without buildings, the residential tint, minor roads, airfields, small rivers and minor names, no
  hillshade, and a pixel ratio of at most 1; Settings has it as Map detail, Low (on) or High (off, the default). It had an
  Auto, removed in T-2840: it went Low for the page on one slow zoom, and Low saves no resident memory; a stored "auto" reads as High.
- **A hidden page** [T-826; notes/T-826.md]: the browser stops animation frames and slows the main
  thread's timers to once a second (once a minute after five hidden minutes); worker timers run on
  at nearly full rate. With "Run in background" off (the default) the clock freezes while hidden, so
  the game pauses. With it on the clock runs on: it is anchored to wall time, the clock worker
  settles money and autosaves on its own timers from the last clock message, and what the main
  thread moves with frames (the shown minute and day, the render epoch with its trips and boarding
  hour, and the demand level effects that follow the minute) catches up at the first frame back.
  Hours that pass hidden pay their fares evenly within the hour (no boarding hour was worked out for
  them), with the same whole-hour totals. Station faces and sounds stop while hidden either way.

### Workers

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
  logical threads, 2 on 6 to 11, 1 on fewer; then a phone or tablet (by its user agent), or a device
  of 4 GB or less, gets 1 (T-2794 adds the same check's lean ways: pixel ratio at most 2, compact outlines; notes/T-2794.md; T-2802: the WASM memories grow 8 MB at a time, not 32, notes/T-2802.md; a phone or tablet by user agent, not a 4 GB desktop, also has no demand cache and no market packs, so no trips between zones, unless Settings "Calculate intercity trips" is on, T-2833, T-2838; a device reporting 4 GB or less gets a note at the bottom of the map, T-2839, `ui/DeviceNote.tsx`) (`workers/deviceMemory.ts` `smallDevice`, T-2122: a phone's tab holds far
  less than the phone, an iPhone's killed near 2 GB), and unreported memory (desktop Firefox, Safari)
  at most 2; a big game may then drop some by memory [T-1205, below]. Such a small device runs no
  focus pool ([demand-runtime.md](demand-runtime.md) "The focus pool"). The Settings "Use more cores" restores half the threads less one, 1 to 5
  (one per period), at the next load; `?demandWorkers=N` overrides both. A 32-bit browser gets 1 whatever the setting, and no focus pool (`addressLimited`, T-2867: a resizable ArrayBuffer's 4 GB maximum refused; its user agent cannot tell). A worker that cannot get memory halves the solving workers, `memoryShort`, and the rest solve again; with one left it is replaced as after a panic, and the failure offers "Play without intercity trips" (`dropMarket`, this tab, game/ui.ts; notes/T-2867.md). Plus one auxiliary worker with no periods
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
  service change or change to the built stations (one on no line included) [T-1368] reaches the
  page; stations on no line only reach zones, outside the network solved [T-2410], and a changed
  network goes to every worker, which takes a newer one between populations. Messages:
  `workers/demandProtocol.ts`; results land in `game/demand.ts`. A sums worker with no WASM
  (`workers/demandSums.worker.ts`) adds up what the main thread used to [T-1208, T-1210]: each
  published round's views of the four kinds of day, applied on the main thread in the order published,
  and a selection's commuters (dealt for the display, `game/farChunks.ts`) and longer trips and a
  station's; the main thread does the same sums itself only if that worker fails. Timings:
  notes/demand-measurements.md, notes/demand-kernel-history.md.
- **Only final results replace final results** [T-232, T-324]: after an edit every figure keeps the
  last finished forecast until the new solve's last round lands; earlier rounds are shown only where
  nothing final is (startup, a whole-game load), for a zone just opened (its own view alone, never a
  line's figures: `fillNewZones`, T-2793), and for lines new or changed by the edit, which take the
  free-flow estimate's riders as soon as it lands (from a rough solve, which is one zone's riders, only
  lines with no figures at all). A cluster solved ahead
  of the rest of its solve replaces its own figures with its final as soon as it is done [T-392].
  What the player sees:
  [ui-panels.md](ui-panels.md) "Loading".

### Loading, panics and memory

- **Loading and readiness** [T-149, T-172, T-180, T-243, T-347, T-351, T-2560]: input is never blocked. A
  loaded game holds only the clock until every needed city is open and the fresh solve's final
  round has landed (`HOLD_ROUND`, `holdDone`; Anita, T-347). A viewed zone open bare is final from
  its opening: a new game (no station) opens its viewed zone bare, its solve covers the market
  alone, and the card goes when the bare figures arrive (T-351). The cached result
  (section 11) releases it as soon as the game's network is in and the zone in view has figures
  (`cacheDone`, T-2560): it is the final of that very network. The zone in view's first figures
  from a rough solve or an estimate are drawn behind the card without releasing it (T-1533,
  [demand-runtime.md](demand-runtime.md) "The zone in view first"). A city opened later (a pan,
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
  A download the game waits on gives up after 45 s with no headers or no byte of its body (the
  account's calls 120 s, an upload counting toward the wait; `workers/patientFetch.ts`, T-3313), so a
  connection that died without an error fails like any other instead of holding every solve or sync
  behind it. A pack download that fails is tried again after 0.5, 1.5 and 4 s, and a city pack's header is
  read again after its .bin and both fetched again if it changed (a pack rewritten mid-load). A
  zone whose pack still cannot be downloaded is left out (not needed, waited for or held for),
  its failure shows on the status tag while it is in view, and it is tried again after 10, 30 and
  then every 60 s; demand everywhere else carries on (T-484). A zone boundary that cannot be
  fetched counts as failed (a station near the zone counts as reaching it; the view is placed by
  the zone centres) and is fetched again after 10, 30 and then every 60 s, and the coarse outlines
  after 1, 4, 10, 30 and 60 s, so a server answering 404 for a while (a rebuild) never leaves the
  zone in view unloaded for the page's life (T-2478). A pack that downloads whole and
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
  `tw.client.debugPanic("now" | "read" | "afterEdit")`. The first panic's own message comes from
  `last_panic()` (T-2820), since "recursive use of an object" is only the next call on the stopped model. A panic while
  a game opens replaces the model before anything else runs: the player gets `clockWorker.savePanic` with
  that message, and the game they had, or a new one, goes on.
- **A Rust panic in a demand worker** [T-1174] (out of memory aborts the same way) leaves the
  objects it was in borrowed, so the worker posts `dead` with `last_panic()`'s text (the panic
  message, or the allocation that failed) and does nothing more; a pack that fails to open with a
  panic is never taken for a broken pack. The page terminates it, starts a new worker in its place,
  downloads again and reopens everything it held, then sends the newest network again under a new
  version (`freshParts`: every worker composes its market afresh from the packs open, as the
  replacement did, so all hold the same corridor table, T-1354); the shown figures stay until that final. A second panic before the replacement has seen
  a solve through, or a fifth replacement in the page, stops the pool as before. `?debug=1` on the
  dev server has `twDemand.client.debugPanic(worker, "now" | "solve")`; check `t1174-panic-recover.mjs`.
  A demand worker whose code did not download (an error with no message before the worker said
  anything) is replaced the same way after 1 s, then 4 s, and counts as no panic; a third failure
  for that worker stops the pool [T-2704; `scriptFailed`, check `t2704-worker-retry.mjs`]. The clock
  worker's first message is `booted` (its script and imports have loaded); until then `ClockClient`
  holds every message, init first, with its transfers unsent, so a script that did not download is
  started again with all of them after 1 s, then 4 s; a third failure is the failure screen's "The
  train simulation could not start" [T-2710; `spawnWorker`]. The focus pool's workers are not
  retried (a failed focus pool falls back to the main pool).
- **Growing a WASM module's memory** [T-1594, T-2699, T-2716] (`sim/src/heap.rs`): a WASM memory
  grow costs about as much as the memory already held, not the bytes added, and dlmalloc on its own
  grows by what a request lacks (64 KB for a small one). So the modules run dlmalloc over a system
  side of their own that grows by at least 32 MB (the clock worker's module: or an eighth of the
  memory once that is more), and the small allocations after it are carved from that. A worker's
  memory ends at most one step above what it needed; the heap figures below do not count it. The
  allocator's methods are never inlined: LLVM takes `__rust_alloc_zeroed`'s result for a fresh
  object and folded dlmalloc's inlined read of a block's header to 0 (T-2716).
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
  everything outside the WASM) would pass `memBudget`: 8 GB (Anita's target, T-1199), the device's
  memory (`deviceGB`) x 0.75 under 8, 6 GB where it is not reported. The periods are dealt again (`periods`, taken up
  between solves) and every worker composes afresh; no figure changes. It never grows back in a page
  (T-1366). `?perfOff=cap` turns the shrinking off; check `t1205-memory.mjs`.

### Threads, pipeline and determinism

- **Threads talk by snapshots**: a worker transfers a finished buffer with a version number; the
  main thread swaps it in. Works on any host. SharedArrayBuffer is a later option; it needs
  cross-origin isolation headers, which GitHub Pages (Anita's website) cannot send and Cloudflare
  Pages can.
- **Data pipeline**: Python in the maps venv, offline, calling the sim crate natively where the game
  needs the same computation (the zone gravity, the regional market's all-pairs figures). One **pack**
  per zone (boundary, cells with population, jobs and attractor weights, the solved gravity, road
  speeds per period, fitted constants, sources; format in notes/T-004.md) and the regional market
  pack (sites, commute list, trip parameters, the `derived` block). Binary arrays with a small JSON
  header. Locally they sit beside the page (`packs/<id>.json`, `.bin`);
  published, they are on R2 under a versioned prefix that `tools/deploy.py` writes into the page as
  a `tw-data` meta, never chosen by hostname (`game/dataBase.ts`, T-924; T-060, notes/T-026.md).
  The main thread fetches each once and hands each worker a copy. The real-rail layer's files are
  noritetsu's: beside the page (`../noritetsu/`) first, then R2 (`data.anita.garden/noritetsu/`); a page
  with the meta asks R2 first, since nothing is beside the published page (`game/rings.ts`, T-941).
  Two glyph ranges are left out on purpose (Canadian syllabics, `pipeline/glyphs.py`): their 404s make
  MapLibre draw those letters itself, and smoke ignores them on any host.
- **Determinism**: the simulation is a pure function of (save, pack versions, game time). For the
  hour or two after an edit to a line's service the trains drawn also depend on the game time the
  edit took effect; the save carries it (the service's eras, T-206, T-1071).

## 11. Saves

A save is the player's inputs only: track, stations, lines, schedules, fares, each line's station
note [T-706], money (cash, cars owned of each kind, the ledgers' every day [T-2143] and last 48 hours, for the network and for each region, which are the
Money tab's history charts: fares, costs, net and train rides; T-447, T-531, T-565, T-567), clock,
empty line drafts, custom badge symbols, the line-name counter, and the player's station links. Population, jobs and travel come from versioned packs;
all results are recomputed. Sizes: notes/T-040.md, notes/T-567.md.

### Format and compatibility

- **Format** [T-029, `workers/saveFile.ts`]: `TWG1`, a JSON header and the track model's network
  bytes, gzipped. The header keeps `city: nyc` and adds optional region and pack metadata. The
  network bytes are TWT6 (the Mercator world frame, T-306), which is TWT5 byte for byte but for the
  magic; its origin fields keep the game's home origin, unused by the frame, and its radii are ground
  metres. After the lines come optional trailing blocks (`sim/src/track/save.rs`): `LUI1` (custom
  badge symbols and the line-name counter, T-110/T-111), then, each only when there are any, `LNK1`
  links [T-197], `LAU1` lines kept by headway [T-625] and `LHS1` lines running high-speed trains
  [T-817], then `LRT1`, the settled rates of the lines with trains to the bit and the lines the
  settle left circling [T-1787], without which a load settles in full, and `LGR1` line groups
  (each grouped line's index and group number, then each named group's number and name) [T-1791],
  `LGL1` group bullets set by hand (group number, text, colour + 1 or 0) and `LOR1` the Lines and
  Stations tabs' hand order (line indices, then node indices, first first) [T-1813], `LPH1` the
  lines' departure phases [T-1925], `LSP1` the share of a headway line's spare time stood at its last
  stop where not half (count, then per line its index and three f32 bits) [T-2972], `BLD1` the station buildings (count, then per building
  its name, corner count and corners as mm deltas from the previous) [T-2217] and last `LBD1` the
  build day of each constructed station, edge and line in dense order, as deltas (0 is never
  written: 1 is before recording began, otherwise the game day + 2), written only when some day is
  known [T-2871], then `LBP1` the bus lines' own stop spacings [T-2885] and `LHO1` the lines whose
  stop order the player set by hand (count, then line indices) [T-2895] and `LBH1` the stops placed by hand on bus lines (count, then per line its index, stop count and stops) [T-2891], and with `LRT1` when any line is paced, `LPC1` the count lines' pacing at merges (count, then per line its index and three f32 bits), which a load keeps instead of pacing again [T-3233], and `LTW1` the lines run on both pairs of quad track and their hidden twins (count, then per line its index and kind, a twin with its line's index) [T-3333]; a save without one loads with none. While a line's trains change over after a service
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
  header's `stars` keeps each region's star, the game second it was earned; `achievements` each
  achievement step earned, by key, with its game second and what earned it (T-2124,
  [ui-panels.md](ui-panels.md) "Achievements"); `home` where the game opens, a zone id or `world` for all
  cities (T-2395, absent for none, which opens as before; a removed zone's home follows its successor). A field a save lacks
  reads as empty: no `rides` or `points` as 0, no `hours`, `regions` or `stars` as none, no
  `fleetFast` as no high-speed cars; no `achievements` as not yet worked out, which the page does
  quietly once the loaded game's first final result is in. A header `points` block (per region totals, written by builds
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
- **Saves from a newer build** [T-1826]: the header's `level` is the writing build's save level
  (`LEVEL` in `sim/src/track/save.rs`, `TrackApi.save_level`; absent before T-1826, read as 0).
  It goes up by one with every new trailing block (listed in `save.rs` `BLOCKS`; a Rust test pins
  the level and the block count, so a block added without raising it fails), every new field inside
  a header field (a fares setting, a ledger figure) and every new meaning of existing bytes. A new
  top-level header field needs no raise: `headerUnread` (workers/saveFile.ts) treats a field missing
  from `HEADER_FIELDS` as unread in any save with a level. A load counts a game as unread in part
  when its level is above this build's, its header has an unknown field, or `TrackApi.save_unread`
  finds network bytes left after the last block it knows (an unknown block stops the decoder, so
  known blocks after it are unread too) or node or edge flag bits it does not know. Such a game
  plays, but nothing writes its slot (autosave, Save, the frame conversion) or reaches the account
  (no claim, no upload), so the device playing it in the newer build keeps it; the top bar and the
  saves panel say "This game was saved by a newer version. Reload the page to update." A newer file
  opened with Open a file (or a backup restored) is written to its new slot as it was loaded.
  Download gives the file as loaded, with everything; Make a copy gives a slot of this build's
  with the game as played here, without the parts it could not read, and autosaves from then on.
  Saves without a level are judged by their network bytes alone (they can carry retired header
  fields such as `points`), so every older save opens as before.
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

### When it saves, backups and versions

- **When it saves**: the clock worker builds, compresses and writes to IndexedDB itself: at once
  after anything paid for, else once a wall minute if something changed, and when the page is
  hidden; a game running in the background keeps saving once a wall minute [T-826]. On start the player picks a slot or New game on the start menu [T-677]; `?continue=1`
  opens the slot opened last. `?new=1` starts a game that reads no slot and owns none, and
  does not autosave until Make a copy gives it one [T-320]; `?synth=1` and `?demandTest=1`
  load the last game but never autosave it. A network viewed from Community (T-3257) comes as a
  `pending-load` import marked `view`: the worker loads it into no slot (`intoView`, `viewing` its
  name), nothing autosaves it, and Save (`saveView`, the copy path) gives it a slot under that name.
  The page asks before anything replaces it (`leaveViewed` in `switchGame` and `leaveToMenu`).
- **No slot is written over with another game** [T-743]: New game (the start menu's beside saved
  games, and the saves panel's), Open a file and a backup's Open as a copy each put the game in a
  new slot, after saving the game being played, which keeps its slot. A new game's slot is named
  after its start (`newGameName`: "Tokyo", "Tokyo sandbox", "World" for a map start, then " 2" and
  on if taken; T-728) and written once something changes, or at once when it starts from a real
  network; a loaded file's is
  named after the file without `.save` and written at once; a restored backup's (this browser's or
  the account's) takes the name of the game it backs up and is a version of that game (T-2037,
  "Versions of one game" below). A tab without its slot's lock must make a copy before leaving a changed
  game. On a test page the game is replaced in the page and owns no slot. A slot that cannot be
  read on start stays as it is (its bytes also under `unreadable-<time>`) and the new game gets a
  slot of its own ("Game N"). The start menu's New game always takes a new slot, even with nothing
  saved (T-728); the original slot `autosave` is only ever an old one.
- **Real networks** [T-731]: `packs/real/<id>.save` for New York, London, Paris, Tokyo, Seoul,
  Toronto and Sendai, made by hand from timetables, and [T-3225] the cities in `pipeline/realauto.json`
  (the 30 biggest metros by annual riders that have a pack), made with no hand work from noritetsu's OSM
  extract: route relations cut to the pack, intercity, high-speed and heritage trains left out, stations
  by name within 400 m, trains an hour from OSM's `interval` else a default by kind of line (a suburban
  train with no line of its own every half hour; a light-rail route on main-line track with 5 or more
  stops at least 1 km apart, as OSM maps Berlin's S-Bahn, runs commuter trains at commuter speed on its
  own track [T-3265]), names in Latin letters where OSM has them (Cyrillic
  transliterated, Chinese in pinyin), and a route whose line the track model cannot run (a terminal
  with more tracks than a node takes) left out (`pipeline/realnet.py ship` copies them from realnet's game
  files into `data/packs/real/`, `realauto.py ship` writes the app's list, and the build copies that folder). Between stations it follows the
  real track [T-1303]: each hop routed over the real-network graph (the track tool's router and fit,
  run when the saves are built, never in the game); lines on the same real track share double-track
  pairs, as many as their trains need [T-1305]; a hop with no real track near (trams, monorails, people
  movers, most of Toronto's streetcars) keeps its GTFS shape or a straight line. Track lies on its
  real level [T-1773, `pipeline/reallevels.py`]: OSM's `layer` order, else -1 in a tunnel and +1 on a
  bridge, clamped to -3..3, with bridges and cuts under 400 m dropped where no other line crosses, and
  ramps of 200 m a level placed clear of platforms; where a real line climbs faster than that, a
  station moves to a neighbour's level (Brooklyn's F and G stand at +2 at 7 Av and 15 St, really
  underground). Imported track is grade-separated from other imported track whatever the levels
  (`EdgeData.sep`, bit 7 of the edge flags in the save; T-1751): the crossings the levels miss make no
  flat crossing, hold or price, while track the player draws crosses it by the usual rule. Every
  dead-end station has a terminal, as real termini turn trains on all their platforms [T-3233]. A
  sandbox started from one loads it as a new game: its network, fleet and fares, with the clock,
  cash, ledgers and stars of a new game. They are built and settled in the game's Mercator frame
  and saved with their settled rates and pacing, so a start loads them without settling again
  (a converted save loses its rates and settles in full, 40 s and more for London) [T-3233].
- **Backups** [T-744, T-1826]: in the same store, the slot's record without its picture, in four
  tiers `backup:<slot>:t0` to `:t3` (`writeBackup`, `planBackups` in workers/saveFile.ts). An
  autosave at least 10 wall minutes after tier 0's backup becomes tier 0; the backup it replaces
  moves to tier 1 only if it is at least an hour newer than tier 1's (else it goes), and so on with
  a day for tier 2 and a week for tier 3. So the tiers reach back about 10 minutes, an hour (up to
  about 1 h 10 min), a day and a week of wall time, and backups only follow changes. Past 8 MB in
  all for a slot's tiers the oldest are dropped (tier 0 always stays; a save is usually about
  100 KB). Before the account's copy replaces a slot (`writeFetched`: a newer copy taken in, after
  giving way, or the slot's side of a fork) the slot as it was goes to `backup:<slot>:pull`.
  Backups from before T-1826 (`:0`, `:1`) seed the tiers at the next backup write. Each slot's
  backups, and the account's of each of its game's account games, are listed among the card's other
  versions (T-2037; ui-panels.md "Saved games"); opening one makes a new slot, a version of the same
  game. A backup keeps the slot's `origin` and `edit`. Deleting a slot deletes them. `?backupMs=`
  sets the 10-minute step for checks, the others scaled with it.
- **Versions of one game** [T-2037, T-2030; `workers/versions.ts`]: every slot record carries
  `origin`, the id of the game it descends from, and `edit`, the wall time of the latest player
  change in its bytes (an edit, a list order, sandbox: wherever the clock worker marks `edited`; the
  clock's own saves and a save by hand leave it). A new game, Open a file and a file loaded from the
  start menu start an origin of their own (a new game id); Make a copy, a backup or account backup
  opened, a conflict's split and a game downloaded from the account keep their parent's. The origin
  is slot metadata and account facts only, never in the save file, so a downloaded file opened again
  is a game of its own. All saves of one origin, this browser's slots, the account's games not in
  this browser (their facts' `origin`) and the backups of each, are versions of one game shown as one
  card; the card's game is the version with the latest `edit` (a save without one counts its `at`),
  and in the saves panel the slot being played. **Saves from before T-2037** get an origin at start
  (`stampAll`: the start menu's settle with the account's list, the clock worker's start without):
  a name with no ending is a game of its own (its account id, else its slot key); a name ending in a
  fork's or restore's time (" at 6:17 PM") or " copy" takes the origin of the save named as the name
  without its endings, the one with an origin already or else the lowest id, and with none of that
  name the saves sharing the base name go together under the lowest id among them
  (`assignOrigins`); a fork's or restore's time comes off the name as the origin is stamped (forks
  and restores now keep the game's name), " copy" stays. Account games listed but not in this browser get the same rule at listing, so
  every device groups them alike; a game downloaded from the account is stamped with the list in
  hand. A rename leaves the origin.
- **Saving by hand over another tab** [T-2037]: a tab that takes the slot's lock with Save first
  looks at the slot: if another tab wrote a player change there that this tab never read (`edit`
  above the one it last read or wrote), that version moves to a new slot, the same game, still tied
  to the account game it went up as, and this slot is tied to a new account game (`keepOtherTabs`).

### Storage, slots and writers

- **Deleting is for good** [T-1827 removed]: deleting a slot (saves panel, start menu, or a
  deletion from the account taking off an unchanged copy, `settleSlot`) deletes it, its backups,
  sync record and demand cache. A card's delete deletes every slot and account game of its origin
  (T-2037); one version is deleted from the card's list of other versions. T-1827 had kept deleted slots 30 days under `deleted:<ms>-<id>`;
  Anita removed it the same day, and `sweepDeleted` clears any such keys at every start.
- **Storage full or cleared while open** [T-1829]: a connection the browser closes (site data
  cleared, the database deleted from another page: `close` and `versionchange`) is forgotten, and
  every call that fails to open a transaction drops the connection and tries once more, so the
  database is opened again, made afresh if it went. A failed autosave write calls `makeRoom`
  (drops the connection; on `QuotaExceededError` or `UnknownError` also every `demand:` and
  `ahead:` key) and tries once more. A slot this page had written found gone at the next write is
  written whole under the name last listed (`writeSlot`'s `lastName`) and its sync record put back
  from the last one seen (`lastSync`), so it stays the same account game; backups and other slots
  in a cleared database are gone.
- **Leave page?** [T-1826]: while the player has changed the game (an edit, a list order, sandbox;
  not the clock's own money) and nothing can write it (autosave off for any reason above, or a
  failed write), the clock worker tells the page (`unsaved`) and the page holds a `beforeunload`
  handler, so closing or reloading it asks first (game/saveSafety.ts). A tab whose autosave is on
  never asks: the hidden-page save covers it.
- **Outside the browser's storage** [T-1214]: a full disk let Chrome empty the site's IndexedDB
  (2026-10-24). Once a slot is written the page asks `navigator.storage.persist()`. A `copy-folder`
  handle left in the store by T-1214's folder copies (dropped in T-1395) is deleted at start.
- **Switching games reloads the page** [T-1196]: Open, Open a file, Open as a copy and New game
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

### Demand cache, old saves and the account

- **Demand cache** [T-328, `workers/demandCache.ts`]: each final demand result is kept beside its
  slot in the same IndexedDB store (`demand:<slot>:<network hash>`, one per slot; a deleted slot
  takes its cache along; sizes and timings in notes/T-328.md). It holds what
  the page shows from a final solve (every kind of day's lines, stations, purposes, each solved
  zone's split, the market's trips, and the viewed zone's bubbles of every purpose for the day shown
  when they were on show, T-1904), not worker state; lines and stations by their place in the network, since a load
  renumbers ids. Its key is a SHA-256 of the network demand reads, of every pack the solve covered
  (the packs open as it was sent, one closed while it ran included, T-2579) and of the market, the app build, the worker count and the perfOff flags; the zones must be the
  ones the new solve covers. A load's network is worked out from the save alone (it waits for the
  bus routes' street speeds, [track-and-service.md](track-and-service.md) 6.7, T-2499), so a reload
  of an unchanged game finds the result its last final wrote. When a whole game loads and the key matches, the result shows at once
  (the bubbles once the zone's cells are open), and once the network is in the loading card goes
  and the clock runs (T-2560). The workers solve the
  same network as on any load, under the load's barrier (`cacheSolve`), their estimates not shown;
  until that final the levels of detail stay as they are (as under a load's card, T-1223), and an
  edit meanwhile is an edit like any other, its solve replacing that one. The status tag says
  "Trips in 81 cities being recomputed…" until the zone in view's cluster has its final (ui-panels.md
  "Status and busy tags"); other purposes' or days' bubbles wait for the workers. A question about the zone in
  view (a station's riders, a bubble's far end) goes to the focus pool, which the cache does not end (T-2533): it
  solves that zone alone to its round 0, shown nowhere, and answers from it (`focusAsked`) until the workers' solve
  of the network is sent and what was asked of it is answered (`retireFocus`, [demand-runtime.md](demand-runtime.md)
  "The zone in view first"); from then a click is answered from the newest round the workers reach ("The pool", T-327),
  and the fresh final replaces the cache in one step (it is bit for bit the same). A test page, `?new=1` and a tab without the slot never
  write it. The write runs in a worker of its own (`workers/cacheWrite.worker.ts`, T-789), the
  market's corridor table (shared by the days) copied once and transferred to it.
  **Ahead of the network** (T-1539): a result whose network is the one a load worked out from its
  save also holds that network snapshot and the key of the save's network inputs (a SHA-256 of the
  save's track bytes, which a save numbers densely, without their eras tail (the trains changing over
  after an edit, drawn only and pruned as the clock runs, T-2588), and the header fields the snapshot
  reads: `saveNetKey`), with a small pointer `ahead:<slot>` to it. The clock worker sends that key as a
  load starts (`netAhead`), before it works the network out (17-42 s on the China save), and the
  page shows the result through the same key checks, on the stored network; the network that
  arrives is compared with it: the same one takes its place with no second look, a different one
  takes the result down and is looked up as on any load (`?perfOff=cacheAhead` waits for the
  network as before).
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
  - **Uploads** (the clock worker, after `writeSlot`): the save bytes as Download
    as a file writes them, the name, the card's facts (with `origin` and `edit`, T-2037) and
    picture. A save by hand goes at once with `force`; a rename at once; an autosave only when it
    holds a player change the account lacks (`editedHere`: the slot's `edit` above the record's,
    T-2030), at most every 2 wall minutes (`GAP_MS`), a timer sending what the gap held back; the
    page hiding sends at once (best effort) on the same terms. The clock's own saves go up with the
    next player change. A save over 4 MB is never sent (the server closes the connection past 4.6 MB).
  - **On load**: the start menu settles every slot it can lock before the worker starts
    (`settleOnStart`), and the worker again on start and on `online` (`settleSlot`): a slot with a
    player change here goes up; one without (the clock's saves at most) takes a newer account copy
    (slot record from the server's facts and picture, nothing decoded; what it held goes to its
    `pull` backup); one the account deleted (a tombstone) is deleted if it has no player change
    here, else kept as a new account game; deletions made here while offline are sent. The game
    opened is claimed (`claim`); a slot with no player change here behind the account is replaced
    in place (`loadGame` through the game's message queue).
  - **Conflicts** [T-2037, T-2030]: a slot with a player change here whose account game moved on, or
    which another device opened from the same rev, keeps its version. If the account's copy has no
    player change this slot's last sync lacked (its facts' `edit`; `editedThere`), the clock alone
    moved it on and this slot is saved over it with `force` (the server keeps the copy written over
    as a backup). Otherwise this slot is split off (`splitOff`): tied to a new account game of the
    same origin and name and uploaded, while the account's copy stays under the old id; both are
    versions of the one card, the later player change on top. A game being played carries on in
    its slot, nothing swapped.
  - **Two devices** work as two tabs do: the last device to open a game is its writer; another
    device's upload hears 409 "elsewhere". If the writer's copy has no player change this device
    lacks (an idle tab that opened the game and only ran its clock, T-2030), this device saves over
    it with `force` and carries on; otherwise it stops autosaving the game, locally too, and says
    "Autosave is off here because this game was opened on another device."; its save by hand then
    splits its version off as above rather than write over the other device's change. Its slot is
    marked given way.
  - **Deleting** a tied slot deletes its account game (queued as `cloud-deletes` while offline);
    a game the account deleted, under a tombstone, never comes back under its id (410): a kept
    copy gets a new id. A game opened unchanged after another device deleted it is kept in this
    browser only.
  - **The account's backups** (up to three a game, made by the server) are listed among the card's
    other versions and opened as a new version of the game through Open a file's path, the origin
    passed with it (`restoreAccountBackup`, the start menu's `fileChoice`); the route names a
    backup by place, so the page checks its `X-Rev`.
  - **Signing out with Remove** marks the account in sessionStorage (`trainworld.signout-remove`),
    goes to the start menu if a game is open, and there settles, deletes every slot of the account
    not changed since its last sync (never from the account; its score entry is forgotten here so
    its figures stay), then signs out (`removeOnStart`).
  - A page whose `/noritetsu/config` did not answer asks again on `online` and every 2 minutes.
  - `?new=1`, `?synth=1`, `?demandTest=1` sync nothing; localhost syncs only with `?acct=local` or
    `?acct=live` given once. `checks/t988-cloud.mjs` runs it against a local server.
- **Plays and playtime** [T-1771; `game/playtime.ts`; the server's `docs/trainworld-api.md` "Plays and
  playtime"]: once `/noritetsu/config` says `trainworldPing`, a page with a game open (not the start menu)
  and its tab visible POSTs `trainworld/ping` to the account server at once and then every 60 s: a random
  id made per page load and held in memory only, a second random id kept in localStorage
  (`trainworld.browser`, never tied to the account) so the server counts distinct browsers and each one's
  days and minutes (Anita, 2026-11-06; forgotten after 13 months unseen), the zone the game began in
  (`gameStartZone`), sandbox or not, the build; signed in, the session rides along like every call, so the server counts minutes per
  account. The server keeps per-UTC-day aggregates, never the IP; `GET /trainworld/stats` needs the admin key.
  Nothing is sent with `notrack` in localStorage (the website's owner opt-out; `?notrack` sets it,
  `?track` clears it), on localhost or from an automated browser (smoke, screenshots) unless the address
  has `?track`, and a localhost page also needs `?acct=local` to have a server. Failures are silent.
- **Later**: sharing a game by link (T-989).
