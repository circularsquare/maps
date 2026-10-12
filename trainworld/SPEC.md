# anitabuilder (trainworld) — spec

**anitabuilder** is the game's name wherever a player sees it (page title, UI, file names a
player saves), lowercase. **trainworld** stays the internal name: this folder, the crate, packages,
the serve.py registration and code. A browser game for building rail on
a real world map, in the line of Subway Builder and NIMBY Rails: Subway Builder's data-driven
demand, NIMBY Rails' whole world, and neither one's performance problems.

This file says what is true **now**. It is rewritten in place when a decision changes; the
history and the reasons live in `DECISIONS.md`. How the docs fit together is in `CLAUDE.md`.
Where this spec says **open**, ask Anita; everything else is decided unless she changes it.

---

## 1. Priorities, in order

When two goals conflict, the higher one wins.

**Current development focus (Anita, 2026-10-10): content and gameplay.** Performance is
adequate. The responsiveness and CPU principles below remain architectural constraints;
proactive optimization is parked until Anita reports a concrete problem.

1. **The game never freezes.** The main thread draws and handles input, nothing else. Any
   computation that can take more than a few milliseconds runs in a worker, and the UI keeps
   showing the last finished result until the new one lands. No edit, load, save or demand
   recompute may block input. Anita's main complaint about Subway Builder is that things brick
   the game.
2. **Low CPU.** An idle game at normal speed uses little CPU. Work happens when something
   changes (an edit, a new time period), not every frame. The ~10k-train goal is about the
   **simulation**: passengers riding ~10k trains across the world. Drawing is not the limit (T-002
   drew 3M at 60 fps) and only trains near the view are drawn anyway. Passenger cost scales with
   each city's network and demand, not with the number of trains (T-005), so the budget to watch
   is demand recompute time per city and the number of cities recomputing at once.
3. **Demand that feels earned.** Ridership follows from where people live and work and how good
   the trip is, and the player can see why. **Realism is for rough calibration, not a goal in
   itself** (Anita, 2026-10-09): the game may be easier than real life, and the many reasons
   transit use differs between countries are not ours to capture. A real network should give
   ridership of the right order (section 9), not an exact match. Prefer simple, interpretable
   rules over faithful ones.
4. **Many cities, one world.** Several full cities, intercity links between them and to smaller
   towns, in one save.
5. **Cute, clear look.** Section 8.

## 2. The core performance idea

**Trains are cheap; passengers are expensive.** Subway Builder's changelog points at passenger
work: a timetable search per 200-person group, "15 minute lag spikes" from every group
re-deciding at once, freezes when drawing track on big networks. It is Electron + React +
MapLibre, so the map library is not the problem; bursty per-group passenger work is.

- **Trains are a function of time, not things that get stepped.** A line has a route, a run-time
  profile (accelerate, cruise at the curve-limited speed, brake, dwell, plus any junction delay
  from section 6.2) and trains per period. A train's position is `profile(t - departure)`.
  Nothing ticks per train; trains outside the view cost nothing.
- **Passengers are flows, not agents.** Demand is trips per time period between places. Route
  and mode choice run on that in a worker when the network changes, producing loads per line
  segment per period. A train's load is its segment's flow divided by the trains in the period.
  Platform crowds come from the same numbers.
- **Recompute only what changed.** Each city's demand is solved separately and only when its
  network changes (a dirty flag per city, like Factorio putting idle entities to sleep).
- **Optional flavour agents.** If the screen wants little people, sample a few from the flows for
  display. They never drive the simulation.

Borrowed from Factorio and NIMBY Rails (sources in `research/performance-factorio-nimby.md`):

- Sleep what is idle; wake on timed events, never poll (FFF-148, FFF-421).
- Cosmetics are a pure function of the clock, evaluated only when drawn (FFF-204, FFF-421).
- Merge many things into one solved unit (belts as gap-lists, fluid segments): riders are counts
  per bucket, each city network is one independent solve (FFF-176, FFF-416).
- Flat structure-of-arrays memory from day one; Factorio's biggest wins were layout (FFF-204,
  FFF-209).
- Thread only independent, mostly-read work, each worker owning its memory. Factorio's attempts
  to thread interacting systems were slower or rolled back (FFF-215, FFF-421).
- Deterministic fixed-tick simulation: save = inputs, replays and bug repro for free (FFF-76).
- Snapshot-then-compress saving off the main thread (FFF-201).
- NIMBY Rails hands the renderer a small motion snapshot each frame and routes passengers over a
  graph of transfer stations only. It simulates individual riders (one 2023 benchmark: 600 trains,
  70k riders) and overfull queues slow its whole sim; the flow model is meant to go far past that.

### 2.1 No signals, for now, with the door left open

Trains do not block each other: two lines on one track pass through each other. That keeps the
train side analytic. Two requirements come with it.

**Discourage tangles another way.** Track capacity and junction conflicts are computed from the
scheduled frequencies in the clock worker, before demand runs, not simulated train by train
(section 6.2). Overloaded junctions and
segments add delay to every line through them, so a tangle costs run time, frequency, fleet and
riders, visibly. Flat crossings and junctions also cost money to build; grade separation (a
different level) avoids the conflict.

**Do not rule signals out.** If blocking trains are wanted later:

- The track is stored as a graph of segments between nodes with direction, which is already the
  shape a block system needs.
- The renderer and the passenger model never assume `position = f(t)`. They consume
  **keyframes**: constant-acceleration phases (time, offset, speed, acceleration) along a path of
  `(edge, direction)` segments. Today every trip of one line, direction and demand level runs the
  same analytic profile, so those trips share one phase table and each trip carries only its
  departure time (format in notes/T-040.md); a later event-driven block simulator can give each
  train its own phases, for one city or everywhere, without touching the renderer or demand code.
- The simulation is already deterministic and event-driven (arrivals and departures are events),
  so "wait for block" is a new event type, not a rewrite.

## 3. World, cities and towns

New York, Philadelphia-Camden-Wilmington, Boston, Washington, DC, Baltimore, Providence,
New Haven, Hartford and Springfield are playable in one Northeast save (T-116, T-134, T-137). Boston includes its southern New Hampshire suburbs; Washington
includes surrounding Maryland, Virginia and Jefferson County, West Virginia. New city additions
reuse the established sources and broad metro extents, with basic pack/loading checks rather
than a new calibration exercise (Anita).
Each city keeps its own local commute pack and gravity; shared lines retain their whole routes.
Mercer County/Trenton supplies town endpoints to the separate regional trip market (T-117).
City navigation and the top bar follow the focused city. Exact pack boundaries show at regional
zooms. Town upgrades and additional metros remain the M2/M3 design.

**Coordinates (T-125).** The track model is planar metres in the game's frame. New games use a
Lambert conformal conic tangent at New York (-73.985, 40.758): conformal, so arc radii and curve
speeds are right in every direction, with a scale error of `dlat^2/2` that depends on latitude
alone (0.06% at 220 km north or south of New York, 0.24% at 450 km, under 0.4% from 35.8 to
45.8 N, any longitude). A line between cities has its true length: New York to Philadelphia,
Boston, Washington and Pittsburgh measure within 0.02% of the great circle. Saves made before
this load in the original equirectangular frame (a fixed `cos(lat)` east-west scale, 1.2% short
at Philadelphia and worse further out) and stay in it; the save's network bytes name the frame.
Every demand pack keeps its own local origin and frame; all conversions go through longitude and
latitude. `app/src/game/registry.json` is the one registry of the world, its frames, its regional
market pack and its cities; adding a region is an entry there plus its pack. notes/T-125.md.

- **World**: one real-world map. Basemap from vector tiles (Protomaps/OpenFreeMap), repainted
  faint so the player's network reads first.
- **City**: a metro area with a boundary drawn case by case (roughly Subway Builder's extent).
  Full local demand model. A subtle exterior shade and thin solid boundary identify the focused
  metropolitan pack; the normal view bounds cull the one continuous network.
- **Town**: a place too small to model as a city, or not added yet. Long-distance demand only,
  spread inside it by population and jobs. A town can become a full city mid-game once its city
  pack exists (mainly for development: add a city to a running save).
- **Cells**: the unit of local demand. H3 resolution 9 (~0.1 km², ~170 m edge), only cells with
  people or jobs. Resolution 8 (~0.74 km², Kontur's) is too coarse for walk catchment: half a
  cell is ~4 minutes of walking. Each census block is spread over the cells its polygon covers,
  by land area (T-013). New York (25 counties, 21.7M people) is 202,337 cells and an 11.7 MB pack,
  5.9 MB gzipped, including commute, customer-sector and school weights and a 1.1 MB water mask.
  **Grain matters most in city centres** (Anita): keep res 9 or finer where it is dense; the
  suburbs may be coarser if performance ever needs it (T-044). Not needed yet.

**The network is one network.** There is no urban/intercity split in the track, trains or lines.
Any line can stop anywhere, and a big city will usually have several stations that long-distance
lines call at (Tokyo, Kansai). Only the demand is split into layers (sections 4 and 5).

## 4. Demand: local trips

Local demand models commute trips on all nine registered cities' tier-1 packs, plus shopping/
errands and social/leisure outings in NYC (T-066), and attendance at matched public schools across the NYC metro (T-140/T-150). Customer travel starts from all residents,
uses retail/service and food/recreation employment plus population as destination attractors,
and has shorter distance decay and separate outward/return schedules. Non-work outings under
150 m between represented cell centres are excluded from every mode and total. Rates are
provisional, with no commute-share target. School destinations use real building enrollment;
estimated child populations supply homes, with a provisional 90% typical-school-day attendance
factor, morning arrivals and mostly midday returns. Private schools, universities,
airport passengers and patients remain future layers.
Gravity runs once between 2 km zones independently for each city;
station access, destination-specific common-line strategies, fare-sensitive mode choice and
three crowding rounds recompute after service or fare changes. Passengers are flows, with per-
line loads and on-demand place queries. Each crowding round uses the combined loads of all
cities, NYC everyday travel and the regional market. City shares and local bubbles include the
focused city's modelled local trips. Bubbles count trip legs, outward and return together;
selection shows purpose counts/shares and five departure periods separately for each direction,
with all-mode and train-only scope. Station catchment queries still explicitly count commuters.
For local trips, the distance-based walking share is reserved before rail and driving compete
for the remainder; useful rail cannot remove the baseline walk share of a short trip. Commutes
use 2 km gravity-zone distance while everyday purposes use sampled cell distance (T-153).

Read the full numbered section in [demand.md](docs/demand.md).

## 5. Demand: long-distance trips

The first Northeast regional market is implemented (T-117), separate from local commutes.
It represents personal and business trips within and between New York, Philadelphia,
Mercer County/Trenton and every registered playable city. Its constants are judgement values,
not fitted corridor OD. Market IDs are stable; new cities append after existing places;
flow layouts and corridor summaries use the pack's market count (T-134).

### 5.1 The total travel market, then rail's share

Generate **total** all-mode trips, then let the player's network earn rail's share.
The pack has 229,885 population/job-weighted endpoint sites at 600 m grain, grouped into
2,913 zones of 6 km. Each resident generates 0.015 departures/day, distributed by destination
attraction `population + 2 * jobs` and `exp(-distance/180 km) / max(distance,40 km)^1.2`.
A smooth 40–80 km onset applies within metros too; city boundaries do not change this curve.
Opposite zone-pair totals are averaged; tiny pairs are pruned. The represented all-mode market
is about 696k daily departures, including longer trips within the metros.

Rail competes with driving through a logit at 0.025 per perceived minute. Driving takes
15 minutes plus 1.25 times straight-line distance at 85 km/h. Fares, service, transfers and
shared crowding affect rail. Period shares are 20/35/25/15/5 percent, with a stronger midday
role than commutes. Measured NHTS OD calibration remains T-011; air, coach, induced travel and
international borders remain future work. Local gravity does not yet generate cross-pack
commutes, and this non-commute market does not substitute for them.

### 5.2 Where a long-distance trip starts and ends

Origin access is population-weighted; destination access uses population and jobs. Each site
offers nearby stations through the same soft fast-walk catchment as local demand. Up to eight
weighted station options per zone end feed the common strategy graph. Local feeder lines,
transfers and the intercity service receive actual loads and boardings. No express-only route
or station-city affiliation bypasses the player's local network.

**Seeing longer trips (T-118).** The commuter views cannot show them: a Philadelphia bubble's
longer trips end in New York, off the screen. So a selected bubble set and a selected station
each list them by all represented places and
draw them by far zone up close, aggregating to labeled other-city totals at regional zooms
(T-135, T-136). Same-city longer-trip bubbles are omitted at every zoom (T-139); their counts
remain in the inspector. Bubble area represents total travellers within the longer-trip layer;
colour represents the train share, red at zero and blue at 100%.
Longer trips use their own visible size scale, blue for train and red for everything else;
section 8 has the display. A selection of local cells is mapped
to the market through its 600 m squares: the pipeline built the sites from the same cells, so each
selected cell falls in one site's square, a square counts the selected share of its commuters
(homes or jobs, whichever end was selected), and a site's share of its zone's departures is its
population over the zone's (of arrivals, its attraction over the zone's). Numbers are the
market's own, summed from its zone pairs; nothing is estimated again. A station's riders come
from the same assignment that loads the lines, so they add up to its entries and exits.

### 5.3 Towns and park-and-ride

Mercer County uses actual Census population and LODES job locations, without a fabricated
local commute pack. Access uses the same fast walk; any player-built feeder line participates
in regional routing. Park-and-ride, more towns and town-to-city upgrades remain planned.

### 5.4 Size

The regional pack retains about 1.40 million directed zone pairs. It runs in the same bounded worker
pool as the local city packs, with periods divided among workers. Each period advances all populations
from one frozen aggregate flow vector per crowding round. Details, assumptions and checks:
[T-117](notes/T-117.md), [Boston/DC expansion](notes/T-134.md), [five-city batch](notes/T-137.md).

## 6. Building and running

The player draws and edits curved track, stations and lines as blueprints, then pays to
construct them. Whole-line train allocations determine congestion-adjusted headways; legacy
frequency schedules still load. The clock worker owns edits, capacity, fleet, costs and money,
with undo for reversible changes.

Read the full numbered section in [track-and-service.md](docs/track-and-service.md).

## 7. Technology

Web first, so it can be shared as a link.

- **App and UI**: TypeScript, Vite, Preact with signals (about 10 kB gzipped). No game engine: the
  map is the game. The main thread holds UI state and read-only snapshots from the workers (the
  world view: the save's inputs plus worker results) and changes the game only by sending edit
  operations to the clock worker (6.5). Layout in `CLAUDE.md`.
- **Map**: MapLibre GL 6 for the basemap (Anita's maps all use it; it has a globe for the world
  view, which our layer does not draw on yet: T-033). Track, stations and trains are drawn by our
  own WebGL2 renderer on a **separate canvas over the map**, not GeoJSON sources (per-frame
  `setData` pins a core, memory `reference_maplibre_animation_loop_cost.md`) and not a MapLibre
  custom layer (that repaints the whole basemap on every train frame: 35% of a core playing
  against 17-21%). The network draws above the basemap's place names. Track buffers are built in
  the clock worker and replaced on each edit (6.5); trains are instanced and positioned on the GPU
  from keyframes and the clock.
- **Staying in step with the basemap**: an empty MapLibre custom layer hands over the camera
  matrix on every frame MapLibre draws, and the overlay is drawn right there, in the same frame
  with the same matrix; our own frame loop drives the overlay only for the clock and skips
  duplicate draws when MapLibre has a frame queued (T-075). This keeps the network on the current
  camera during panning without drawing it twice. notes/T-015.md, notes/T-075.md.
  Station names are DOM labels above the overlay in the UI font, moved on MapLibre's frames.
  A flat pan moves their shared compositor layer; placement and the greedy overlap pass
  (transfers first, then lines) run on other camera changes, edits, font loading and showing
  labels again, and after movement ends. A 256 px margin prepares names entering the screen;
  a pan re-places them after 128 px of travel (T-074, notes/T-074.md).
  Names and their invisible collision layer are hidden below zoom 12 (T-101). Above that,
  padded collision rectangles keep names apart: 48 px horizontally and 28 px vertically until
  zoom 14, then 20 px and 12 px. Transfer and constructed-station priorities still apply.
  A MapLibre symbol layer would sit under
  the overlay with track drawn through it. They win over basemap place names through an invisible
  symbol layer of the same names above the basemap, which MapLibre's own collision then honours
  (T-032).
- **Time on the GPU**: keyframe times are relative to a render epoch at the start of the current
  game hour, so the float32 time uniform stays under 3600 s and positions stay within ~2 cm
  however long the game runs. When the hour turns, the clock worker lists the trips running in
  the next hour with their departures relative to the new epoch (computed in float64), and only
  that buffer is re-uploaded; the phase tables change only on edits. notes/T-018.md,
  notes/T-040.md.
- **Frame loop**: frames only while the clock runs; 10 fps while the window is visible but
  unfocused (the clock keeps running); nothing while hidden (the clock freezes) or paused; and
  none while no trips run this hour, when the clock wakes once a game minute instead (T-073). Apart
  from that, the overlay draws on MapLibre's own frames (camera moves, tile loads) and once after
  a change to what is drawn (selection, settings, an edit). The commuter bubbles and a station's
  catchment are on a canvas of their own over it, drawn only on MapLibre's frames and after their
  data changes, never by the clock, so they cost nothing while the game plays with the camera
  still (T-084, T-097). The city's bubbles and a selection's far end are summed in a worker of
  their own (`workers/commuters.worker.ts`).
- **Simulation and demand**: Rust compiled to WebAssembly, in Web Workers (one for the clock and
  lines, a small shared pool for local and regional demand). The crate builds two WASM
  modules: the clock worker's holds the track model (about 430 kB, 165 kB gzipped) and the
  demand workers' the demand kernel, so each worker downloads and
  compiles only its own and nothing is fetched twice; the main thread loads neither. Both are
  built at opt-level 3 (the pair loop is the recompute; smaller settings made edits 1.2-1.8x
  slower), notes/T-051.md. Flat typed arrays (structure of arrays), transferable
  between threads without copying.
- **The demand pool**: half the logical cores less one, at most 5 (one per period), at least 1
  workers that solve, and one more with no periods. Each solving worker opens all registered city packs and
  the regional market (its own copies). The regional header parses into compact numeric site
  rows; immutable trip tables release spare allocation capacity (T-123). Workers run local warm-up solves so V8 has optimised the
  kernel before the first real one, and owns a fixed set of the five periods (worker i: periods
  i, i + n, ...). The extra worker never solves, so it answers at once what needs no period's
  result: the city's cells and each bubble's commuters by mode. The clock worker's network
  snapshot after each service change goes to every worker; between populations a worker takes a
  newer one if it has arrived. A worker sends its sums for the bubbles with each round, unasked.
  Messages: `app/src/workers/demandProtocol.ts`; results land in `app/src/game/demand.ts`. The
  top bar says which step of the solve is running (the free-flow estimate, then each crowding
  round). Timings on the real network: docs/demand.md 4.4.
- **Threads talk by snapshots**: a worker transfers a finished buffer with a version number; the
  main thread swaps it in. Works on any host. SharedArrayBuffer is a later option; it needs
  cross-origin isolation headers, which GitHub Pages (Anita's website) cannot send and Cloudflare
  Pages can.
- **Data pipeline**: Python in the maps venv, offline, calling the sim crate natively where the
  game needs the same computation (the zone gravity). One **city pack** per city (boundary, cells
  with population, jobs and attractor weights, the solved zone gravity, road speeds per period,
  fitted constants, sources; format in notes/T-004.md). The current regional market contains
  endpoint sites and tunable trip parameters; a broader world pack with air routes is planned. Binary arrays
  with a small JSON header, compressed; under ~10 MB for the biggest metro. Today the packs sit
  beside the page (`packs/<city>.json`, `.bin`, and `packs/intercity.json`); the main thread fetches each once and hands
  each demand worker a copy. Published, they move to R2 under a versioned path, compressed and
  cached for good (notes/T-026.md).
- **Determinism**: the simulation is a pure function of (save, pack versions, game time).

## 8. Look and UI

The interface uses Zen Maru Gothic, a resizable dock and a responsive map top bar. Build holds
all construction controls and the cost breakdown. Network riders, commuter bubbles and train
crowding are selectable views; station names and delay tags hide below zoom 12 and leave space
around each label. None/Homes/Destinations stays at the map's upper left, preserving its selection while demand loads.
The City panel follows the viewed city with its top five external connections and train/other bars;
purpose inspection includes a colour-matched share pie and departure charts (T-143). Exact metropolitan
pack boundaries have thin solid outlines at regional zooms. The focused metropolitan area's
boundary remains visible up close, with a darker exterior edge and lighter interior edge;
its outside basemap is subtly shaded (T-139). Picking follows visible drawing order, with demand
bubbles above the network. Trains show separate cars from zoom 15.

Read the full numbered section in [interface.md](docs/interface.md).

## 9. Checking the demand model

- **Real-network test** (T-007, notes/T-007.md): New York's real rail network (subway, SIR, PATH,
  LIRR, Metro-North, NJ Transit rail and light rail; 867 stations, 109 lines) built from the
  operators' GTFS with real run times and trains per hour, solved natively through the app's
  `DemandApi` (`pipeline/realnet.py`, `sim/src/bin/realnet.rs`, 10 s a day), and compared with:
  rail commute share by county of residence (ACS 2019-2023 B08301), subway entries per complex by
  period (MTA OD estimate, October 2024 Wednesdays), riders per subway link at the busiest hour
  (nycriders' routing of that OD) and operator weekday totals. The model is commutes only, so the
  morning peak and the county shares are the fair checks. The same network is also a track-model
  save (`T-007-real-nyc.save`), one track per line; the track model runs it 35% faster than the
  real timetables. The comparison log and what is off are in notes/T-007.md. **Historical T-090
  calibration checkpoint** (2026-10-09, before fare-sensitive demand and common-line strategies;
  soft reach, driving by density and length, three crowding rounds): rail 20.5% of
  commutes against ACS's 20.3%, county error 3.5 points (weighted mean); the four big boroughs
  within 8 points (Brooklyn 41% against 49%, Manhattan 59% against 51%); Nassau, Suffolk,
  Westchester, Fairfield and the outer counties taken together within 1.5 points of ACS; inner New Jersey still 1.7x (Hudson,
  Essex, Bergen, Union, Passaic: the crow-fly gravity sending them to Manhattan, T-072); the
  busiest subway links at 76% of their measured busiest hour (commutes only); commuter rail
  0.8-1.1x its operators' all-trip totals. A lone 12 km line under Manhattan (7 stations, 14 trains
  an hour) carries 477k riders a day, its fullest train at 93% of crush. Rough on purpose: realism
  is calibration, not the goal (section 1).
- **Tier-2 check** (T-006, T-054, New York, against LODES OD 2023): the gravity's commute lengths
  match LODES to 3.2% of trips misplaced by 2 km band (mean 18.7 km against 19.6), county-pair
  flows log correlation 0.89; own-county commuting 39% against 43%, which the home weights did not
  explain (notes/T-006.md, notes/T-054.md, `../artifacts/trainworld/T-006.png`).
- Mode share per city against published figures.
- Long-distance: rail share on corridors with known rail and air numbers. Northeast Corridor first
  (Amtrak NEC ridership, NHTS OD), then Tokyo-Osaka, Paris-Lyon, Madrid-Barcelona.

## 10. Milestones

Region: start in the **Northeast US**, New York first.

- **M0, spikes** (throwaway code, numbers are the output): 10k trains in a MapLibre custom layer
  (frame time, CPU); New York city pack + the demand kernel in a Rust/WASM worker on a hand-made
  network (seconds per recompute, memory); toolchain skeleton served through `serve.py`.
- **M1, New York sandbox**: track with curve speeds, levels and junction capacity; stations, lines,
  per-period frequencies; tier-1 demand; ridership, crowding and the mode-split bar; money; local
  saves. Real-network test passes roughly.
- **M2, Northeast**: Philadelphia and Boston as full cities, towns between them (Trenton, New
  Haven, Providence and so on), the long-distance layer, park-and-ride access, town-to-city
  upgrade.
- **M3, world**: packs generated for many metros at tiers 0 and 1, world view, per-city dirty
  flags under load.
- **M4**: POIs, the look pass, sharing saves by link, accounts.

## 11. Saves and accounts

A save is the player's inputs only: track, stations, lines, schedules, fares, money (cash, cars
owned, the last 30 days' ledger), clock. Population, jobs and travel data come from versioned
packs; all results are recomputed. Even a huge network should be a few MB compressed (T-040: about
100 KB per 10,000 route-km).
Empty line drafts, custom badge symbols and the monotonic default line-name counter are saved
too. Optional TWT4 trailing metadata preserves compatibility with earlier saves (T-110/T-111).
New headers retain `city: nyc` and add optional region/city-pack metadata (the registry's world
and cities). The frame is not in the header: the network bytes say it (TWT5 is the world frame;
TWT4 and older, the original equirectangular one, T-125). Existing NYC and Northeast files and
browser slots load unchanged, in their own frame and for good; new games and their saves are in the
world frame (section 3). Philadelphia track, service and finances save in the same network (T-116).

- Local first (T-029): the clock worker builds and compresses the save (format `TWG1`: a JSON
  header and the track model's network bytes, gzip, `app/src/workers/saveFile.ts`) and writes it to
  IndexedDB itself: at once after anything paid for, else once a wall minute if something changed,
  and when the page is hidden. On start the last game loads (`?new=1` skips it). The settings pane
  saves the game to a file (`anitabuilder-nyc-day-12.save`), loads one, and starts a new game
  after a confirmation that replaces only the active slot. **Named browser slots** (T-068) let
  the player save the current game as a new copy, open another game, rename a slot, or delete
  an inactive slot after confirmation. The selected slot is remembered, and the original
  autosave remains at its existing key. Switching waits for the current writable game's save
  transaction to commit; compression and byte handling stay in the worker.
  A worker-held browser lock permits only one tab to autosave each slot. Another tab can play,
  read and export that game but shows a persistent Autosave off notice; different slots can
  save concurrently. Unsaved changes in a blocked tab must be saved to a new slot before
  switching (or exported before reloading). Closing the owner and reloading reads the latest
  saved game and reacquires the lock. Rename/delete refuse slots owned by another tab.
  Storage failures retain dirty state and a persistent notice; file export remains available.
  Browsers without locks disable browser writes and keep file export usable. notes/T-029.md,
  notes/T-068.md.
- Sharing: upload a save, get a link. Later (M4).
- Accounts only for sync and sharing, not needed to play. They run on `shonei-server`
  (`~/projects/shonei-server`, Anita's existing server), as the current plan.

## 12. Open questions

None at the moment.
