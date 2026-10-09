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

- **World**: one real-world map. Basemap from vector tiles (Protomaps/OpenFreeMap), repainted
  faint so the player's network reads first.
- **City**: a metro area with a boundary drawn case by case (roughly Subway Builder's extent).
  Full local demand model. Zoomed into a city, everything outside its boundary is greyed and only
  trains entering it are drawn.
- **Town**: a place too small to model as a city, or not added yet. Long-distance demand only,
  spread inside it by population and jobs. A town can become a full city mid-game once its city
  pack exists (mainly for development: add a city to a running save).
- **Cells**: the unit of local demand. H3 resolution 9 (~0.1 km², ~170 m edge), only cells with
  people or jobs. Resolution 8 (~0.74 km², Kontur's) is too coarse for walk catchment: half a
  cell is ~4 minutes of walking. Each census block is spread over the cells its polygon covers,
  by land area (T-013). New York (25 counties, 21.7M people) is 202,337 cells and a 6.9 MB pack,
  3.8 MB gzipped, 1.1 MB of it the water mask (notes/T-004.md, T-013.md, T-030.md).
  **Grain matters most in city centres** (Anita): keep res 9 or finer where it is dense; the
  suburbs may be coarser if performance ever needs it (T-044). Not needed yet.

**The network is one network.** There is no urban/intercity split in the track, trains or lines.
Any line can stop anywhere, and a big city will usually have several stations that long-distance
lines call at (Tokyo, Kansai). Only the demand is split into layers (sections 4 and 5).

## 4. Demand: local trips

### 4.1 Where trips come from

A **doubly constrained gravity model** from homes to jobs with a distance-decay curve. Tiers say
how good the inputs are; **only tiers 0 and 1 are built for now**:

| Tier | Inputs | Status |
|---|---|---|
| 0 | population grid + a jobs proxy (GHSL non-residential built-up, 100 m, global, free) | build |
| 1 | population + real jobs counts by small area | build |
| 2 | a real home-to-work OD matrix | not an input; used as a **check** on the tier-1 fit where it exists (LODES OD for the US) |

Subway Builder also uses gravity: LODES gives it homes and jobs, then a distance-based gravity
model picks each commuter's workplace (`research/competitors.md`). Tier 1 done well is at its level.

**Jobs data is real job counts at the place of work**, not a proxy, where it exists:

- US: LODES WAC (Workplace Area Characteristics), jobs per census block by sector and earnings.
  Covers wage and salary jobs under unemployment insurance plus federal civilian; leaves out the
  self-employed, military and informal work. Block figures carry deliberate noise, and some
  employers' jobs are placed by a model among their sites, so aggregate a little before trusting
  one block.
- Japan: Economic Census 2021 mesh, workers per 500 m square.
- England and Wales: Census 2021 workplace population by output area. **Caution: the census was in
  March 2021 lockdown**; many people working from home count at home. Usable for shape, biased
  against office districts.
- Australia: destination zones; Italy: 2011 census sections; EU: JRC ENACT 1 km (2011).
- Everyone else, China included: tier 0.

Full list with grain and access in `research/demand-data.md`. Population for the US: 2020
decennial census by block. **The home end of commutes is where job holders live**, not everyone:
in the US, LODES RAC (jobs by home block, the same jobs WAC counts at work), so a neighbourhood
of students, retirees or children sends fewer commuters than its population (T-054: county
totals now within 10% of ACS commuters, the Bronx was 23% over). Elsewhere, population until a
source by residence is found.

**People who work at their own home make no commute and are not counted** (Anita). LODES places
a remote worker at the employer's office, so commute trips are scaled down by the share working
from home (US: ACS "worked from home", by tract at the home end, by industry at the work end). Jobs whose work is at other people's homes
(home care aides) are real commutes and are spread over where people live (T-014).

**Other trips.** Commutes are only part of travel. A second layer covers shopping, school,
errands and leisure: population to attractors (jobs as the stand-in at first, POIs later). Its
size relative to commutes is a per-city constant, from travel surveys where they exist. **POIs**
(airports, stadiums, universities, tourist sites) are a later layer with their own sources.

### 4.2 Time of day

Five periods: morning peak 6-10, midday 10-16, evening peak 16-20, evening 20-24, night 0-6. They
are the player's scheduling periods too (section 6.3: high demand in the peaks, medium midday and
evening, low at night), defined once in `sim/src/track/params.rs` (`PERIOD_HOURS`,
`PERIOD_LEVEL`) for schedules and demand alike.

Each commuter makes a trip to work and a trip home a day. Of the day's trips to work, 70 / 12 / 6 /
4 / 8% start in the five periods; of the trips home, 2 / 18 / 58 / 16 / 6% (judgement for now). A
trip home goes back by the station pair the morning's trip used, so a period's rail trips are its
share of trips to work on the morning's station pairs plus its share of trips home on the same
pairs reversed. Crowding is judged at each period's busiest hour: 1.4 times the period's mean hour
in the peaks, 1.15 midday, 1.3 in the evening, 1.5 at night. The other layer (4.1, not built yet)
will lean midday and evening.

### 4.3 Mode choice and catchment

Per OD pair and period, compare rail with driving (walking for short trips) by **generalised
cost** in minutes. Buses are out of scope (Anita, 2026-10-09).

- rail = walk to a station + wait (half the headway, capped) + ride + transfers x penalty + walk
  from station + fare / value of time
- driving = door-to-door road time (below; a few constants, real road travel times are not a
  priority) + its cost

Rail share = logistic in the difference (0.05 per perceived minute), plus a **per-city rail
propensity** constant instead of income: it absorbs car ownership, parking, habit and income in one
number, set so a realistic network gives roughly realistic mode share. New York's is 0 (T-078):
rail and driving are judged on perceived time and cost alone. **Catchment is soft** (T-090): there
is no circle the player sees. A farther station makes the walk, and so the whole trip, worse, and
the station loses riders where that trip gets worse than driving. Value of time is per country so
fares compare across countries.

**Walking.** Of the trips that do not take rail, a share walks by distance:
`1 / (1 + exp((d - 1.7 km) / 0.4 km))` on zone-centre distance, so 84% of trips within a 2 km
zone, 32% to the next zone, 6% two zones away; the rest drive (car, bus or taxi). With no rail
5.9% of New York's commutes walk (ACS: about 6%). Judgement until fitted.

Weights (perceived minutes per real minute), from Subway Builder's published constants where they
exist: riding 1.0, the fast walk to and from stations 2.0 for its first 10 minutes and 8.0 after
(below), walking between platforms on a transfer 1.39, platform wait 1.37, waiting at home before a
known departure 0.4, congested driving 1.33, plus parking cost and ~3 min to park.

**Driving** (T-090; `place_min_per_km`, `drive_min` in `sim/src/demand/kernel.rs`). The road is 1.3x
the straight line. Its first and last 2 km (at most half the road each) are local streets, the rest
main roads. Each place has a speed for both, by its density (commuters living and working per km²
of land in its 2 km zone): local streets 40 km/h and main roads 80 km/h in open country, both
falling towards 15 km/h where it is dense, half way at 5,000 per km². The middle of a trip runs at
the mean of its two ends' main-road minutes a km. So a 25 km trip between suburbs takes about 27
minutes (it took 70 at the old flat 28 km/h), Brooklyn to Midtown (10 km) about 30, and 3 km inside
Manhattan about 13. On the gravity's commutes this gives 26-37 minutes in the four big boroughs
(ACS car commutes 34-38) and 36-47 in the suburbs (ACS 27-33; the gravity's suburban commutes are
too long, T-072). Parking is charged at the destination by its job density, as before.

Route choice is not all-or-nothing shortest path. NIMBY Rails' best-known complaint is riders
waiting 50 minutes for a fast train that arrives one minute earlier. Wait time counts at its
weight and the crowding loop (4.4) spreads riders.

**Access and egress.** Riders spread over the stations they could use by a logit on total cost
(scale 0.2 per perceived minute: a station 5 minutes worse gets 37% of the better one's weight),
and rail as a whole is priced at the logsum, so a second usable station helps a little and a
slightly better one does not take everyone.

**One way to reach a station: a fast walk** (Anita, 2026-10-09, as Subway Builder does). Getting
to and from stations is a single access mode at both ends, a walk at roughly bicycle speed and
willingness: **15 km/h along a path 1.3x the straight line (5.2 minutes a straight km); the first
10 minutes (1.9 km) weigh 2.0 each, every minute after that 8.0** (T-090: a 10-minute ride to the
station is taken readily, a 20-minute one rarely). So a station 1 km away costs 10 perceived
minutes, 2 km 23, 2.5 km 44, 3 km 65, 3.5 km 86. Against driving, a home 1 km out has 59% of the
odds of one beside the station, 2 km out 31%, 2.5 km 11%, 3 km 4%: a station's draw is strong for
a couple of km and fades out by about 3-3.5 km, sooner where driving is easy and later where it
is hard (into Manhattan). Between stations the logit above applies (a station 1 km further than
another within the first 1.9 km gets 12% of its riders, beyond that under 1%).

Two **computational bounds** keep the work down and are set where little rail is left: stations
are looked for within 4 km straight line (`walk_cutoff_m`), and a pair of places is offered rail
only if its walks at the two ends together cost at most 80 perceived minutes (`pair_walk_max`).
On the real New York network 0.1% of rail trips start 3.5-4 km from their station and 0.3% have
walks of 70-80 together; for a lone line in Manhattan, the worst case, 3% start 3-3.5 km out and
the bounds cost it about 4% of its riders (notes/T-090.md).

There is no drive, park-and-ride or feeder-bus leg to a station: it was most of a lone Manhattan
line's riders and doubled suburban rail on the real network (notes/T-034.md), and one mode is
easier to read. Driving is only the alternative for the whole trip (Anita, 2026-10-09). Roughly
calibrated on the real New York network (section 9, notes/T-078.md, T-090.md).

### 4.4 Cost of computing it

Measured in `sim/src/demand/` (method and all numbers in notes/T-005.md, T-019.md and T-020.md)
on the real New York pack (21.7M people, 10.2M jobs; 74,920 cells, and 202,337 since blocks are
spread over their area, T-013) and a synthetic city of 153k cells, with a hand-made 434-station,
25-line network.

**Cells are never paired with cells.** 75k-150k cells make 5.6-23 billion pairs. Truncating by
distance does not save it: commutes average 15-20 km, and a cutoff that keeps them keeps billions
of pairs. Even pairing only cells within walk of a station is 90-180M pairs on this network
(1.3-2.6 s native per period) and grows with the square of the area the network covers. So demand
works on two levels:

1. **Gravity between 2 km square zones, once per city.** Doubly constrained, commuting residents to
   commuting jobs (4.1), decay by distance only: **f(d) = d^-1 x exp(-d / 29 km)**, fitted to
   LODES OD for New York (T-006; 4,700-6,200 zones). It does not depend on the network, so edits
   never rerun it. The pipeline solves it (Furness, the sim crate's own code run natively, ~3 s)
   and ships a zone id per cell and two factors per zone in the pack (notes/T-004.md); opening a
   city rebuilds zone trips in ~2 ms in WASM instead of solving them for 8 s. Zone trips spread to
   cells by commuters at each end.
2. **Access per cell**: the 6 nearest stations within the walk's 4 km bound (4.3), found by
   scattering each station over the cells near it. 7-13 ms native.
3. **Access subzones**: the cells of one zone that share their nearest station. Each carries its
   perceived walk to each of its stations, averaged over its cells as the mode choice weighs them
   (a logsum at its 0.05 scale: the walk's cost grows faster than its length, and the near cells
   carry most of the rail), used at both ends of a trip. Cells with no station within 4 km are in
   no subzone: their commuters drive or walk. 2,750 subzones on the hand-made network, 5,032 on New
   York's real one (2,056 and 3,702 with the 2.5 km edge before T-090); access and subzones
   together are rebuilt on every edit in 16-27 ms on the hand-made network and ~57 ms on the real
   one in WASM.
4. **Station-to-station times**: one Dijkstra per station over station nodes plus one node per
   stop per direction per line. Boarding costs the weighted wait (half the headway: the first 5
   minutes at 1.37, beyond that at 0.4) plus a transfer penalty, taken off once. A rider leaving
   a train may walk to another station within 300 m straight line and board there (walk at its
   weight; walk edges start only after a ride, so walking between stations is never a rail trip).
   Snapshot stations within 100 m of each other are one demand station (side-by-side platforms of
   two lines; never two neighbours on one line). 20-40 ms for 434 stations; the trees are kept
   for loading.
5. **Mode choice per subzone pair.** The access logit factorises: each origin subzone gets one row
   over all stations, the logsum of its access options followed by every station-to-station cost
   (at most 6 options x stations multiply-adds); each pair then multiplies that row into its
   destination's egress weights, up to 6 multiply-adds, one ln and one logit per pair. Pairs whose
   two walks together pass `pair_walk_max` (80 perceived minutes, 4.3) are skipped without being
   visited: each zone's destinations are kept cheapest walk first, and the destination zones by
   their cheapest walk, so an origin stops at the first that is too far. That bound is what holds
   the pairs where the 2.5 km edge had them: hand-made network 4.2M pairs (4.1M before T-090;
   with the 4 km reach and no pair bound 9.3M), real network 13.8M (13.6M; 31.8M). What a pair
   reads about its destination is packed in visiting order. Rail trips go back onto access
   stations once per origin subzone and are kept by station pair, and by subzone and access
   station at each end for the demand views (T-084, notes/T-078.md). The views ask for them on
   demand, never pushed: commuters by mode per cell (`cellModes`), a station's catchment and far
   end (`stationRiders`), and where the commuters living or working in any set of cells go, every
   mode (`flows`, T-098 for the bubbles: the gravity's zone trips from the selection, rail along
   the kept station pairs, walk and drive the rest; totals exact and equal to `cellModes`, far
   rail by place blurred like `stationRiders`; 10-20 ms in WASM for one cell or thousands, the
   far cells holding the last 1% left out as a cut total; notes/T-098.md).
6. **Loading**: each station's shortest-path tree is walked once in reverse: under 5 ms.
7. **Crowding**: a penalty per segment from the loads, repeat 4-6, average the loads (method of
   successive averages). Each round costs as much as the first pass. The riders projects worked
   this out; read memory `reference_transit_crowding_assignment.md` (in
   `~/.claude/projects/c--Users-anita-projects-maps/memory/`) first: the penalty starts where seats
   run out, never saturates, and averaging is what stops it ringing. **Three rounds** (T-022): the
   first round leaves the segment that takes the overflow at its worst (963% of crush on the real
   New York network against 310% before it), and a player-sized network moves under 1% by the
   third. No crush cap: an overloaded line is the player's signal (notes/T-022.md).

**Cost.** One period on the 202,337-cell New York pack and the hand-made network, one stage (the
free-flow estimate or one crowding round), one thread: 0.25-0.3 s native, 0.27 s in WASM (0.23-0.24
before T-090's wider reach: the extra origin subzones each build their station row and visit the
destination zones, and the drive costs a little more per pair; a day is 15% longer in WASM). An
edit reruns everything except gravity. Periods are independent, so the game runs them in a pool
of workers (section 7): with three workers in Chrome the mode split updated 0.49-0.55 s after a
new network reached them and the third crowding round landed by 1.9-2.0 s at T-078; T-090 adds
about 15% (not measured in Chrome). Each worker holds its own copy of the city: 18 MB of WASM
memory open, about 23 MB after solving (notes/T-026.md). New York's real 867-station network is
13.8M pairs, 1.0-1.7 s a stage in WASM, 24-25 s for a whole day on one worker (20 s before T-090).

**Accuracy of subzones** against one subzone per cell with the same model (T-090, hand-made
network, free flow): rail total within 0.2%, station entries and exits within 5.0% and 6.3% (sum
of absolute differences), segment loads within 3.0%. The logit roughly halved the station error
of all-or-nothing choice at any grouping, which is why the 5-minute walk bands are gone; what is
left is the grouping (a subzone's walk is one figure for all its cells). A plain mean of the
cells' perceived walks put rail 3.3% under, since the walk's cost is convex.

**If a bigger network makes it slow** (pairs grow with the square of the subzone count): add
workers first. Skipping zone pairs under 0.2-0.5 trips cuts pairs 1.1-4x but drops long trips
(segment loads 3.5-5.4% off, T-020). **If there is time to spare**,
10-minute walk bands are the accuracy lever: 2.4x the pairs, every error about a fifth smaller.

Only cities whose network changed are recomputed, and only for service changes (6.5). The free-flow
first estimate of all five periods lands first and replaces the shown result whole; three crowding
rounds per period refine it in the background, each replacing it again. A newer network supersedes a
running solve at the next period boundary.

## 5. Demand: long-distance trips

### 5.1 The total travel market, then rail's share

Never fit to rail ridership alone (real rail ridership is high partly because the line exists).
Model **total** trips by all modes, then let mode choice give rail its share on the player's network.

- **Market by zone pair, not city centroid.** Long-distance demand runs between coarse zones
  (about municipality size; H3 res 6, ~36 km², to start) with distance measured zone to zone. So
  two cities that touch exchange far more trips across their shared edge than between their far
  ends, as Anita wants. `T_ab = k * P_a^α * A_b^β * f(distance) * border(a, b)`, with `A` an
  attraction (population and jobs) and a per-country trip rate. Inside a zone, trip ends spread to
  cells as in 5.2.
- **One distance curve across the boundary.** The local layer (section 4) has a steep decay and
  stops at roughly 90 minutes; the long-distance layer takes over above that with a long tail, and
  the two overlap smoothly so a 30 km trip is not treated differently because it crosses a city
  line. Where two full cities touch, the local layer runs across both (their cells solved as one
  region), so commuting between neighbours exists. Exact hand-over is settled in M2.
- **Fitted against**: Japan's inter-regional passenger flow survey (all modes, 207 zones); the US
  NextGen NHTS passenger OD (583 zones, 2020-2022), which covers the Northeast; Germany's
  Kreis-level forecast matrix (base 2010); air passengers by route (the `flights` project); and
  rail ridership converted to a total by dividing by the share rail would get on the real
  network's times.
- **Mode split**: logit between the player's rail, air (real routes, ~1.5-2 h airport overhead),
  car (road distance and speed), and coach where it matters.
- **Induced travel**: a small elasticity so a much better link grows the market a bit (the
  logsum feeds back into `T_ab`). Small, so it cannot run away.
- **Border factor**: domestic vs international, softer where crossing is easy (Schengen). One
  factor per class of border until data says otherwise.

### 5.2 Where a long-distance trip starts and ends

From the same cells as local demand: one end weighted by population, the other by jobs and
attractors. A station downtown, near the jobs, wins more riders than one at the edge with no
special rule. Each trip picks its best station pair on the one network, so a city with several
long-distance stations splits its riders between them by where they actually are. In a full
city the access and egress legs ride the player's local lines and load them.

### 5.3 Towns and park-and-ride

A town has no local lines, so access is the same fast walk as in cities (4.3); whether
long-distance trips need a longer reach or a drive leg back (Anita's park-and-ride idea) is decided
when M2 gets there. A town station draws some riders from
across the town and many more if placed where people and jobs are. Anita's Shizuoka case, no
extra machinery.

### 5.4 Size

Zones in full cities and towns above a population floor (`worldcities` / GHSL urban centres);
pairs below a trip floor dropped. Solved in its own worker, on change.

## 6. Building and running

Full design, numbers and sources: `notes/T-008.md`.

### 6.1 Track and stations

- **One track type**, anywhere: over roads at normal cost, over water dearer (6.4). No buildings to
  demolish, no yards, no depots. Ground is flat (no terrain in M1).
- **Geometry: straights and circular arcs**, tangent-continuous, no transition curves or splines
  (exact lengths, one speed per curve, parallel tracks stay arcs). **The points the player clicks
  are points of intersection (PIs)**: the track runs straight between them and cuts the corner at
  each with a circular curve (Anita, 2026-10-09; T-092). A PI's radius is auto unless set: the largest that fits half of each adjacent leg (the
  whole leg next to an end), capped at what the top speed needs (1,796 m). A curve under the
  minimum radius is refused ("The curve is too tight"). A route starting or ending on existing
  track gets a lead-in PI on that track's heading, so it leaves like a turnout and its first curve
  starts at the node (it never runs on top of the main line). Track drawn while the track ran
  through the clicks (T-079, saves from that day) keeps its shape; the points it was drawn through
  are kept with it, and its first reshape makes them its PIs (auto radius) like on any other track.
- **Drawing** (T-023, T-092): click where the track should turn, at the level selected; the first
  click may start on a track end, a node or track (a new junction, built flat), and a click on a
  node or track finishes the route there; a click on the last point, a double click or Enter
  finishes at that point; Backspace takes a point back, Escape drops the route. **A click leaves
  the track drawn before it alone**: everything up to where the curve at the previous click starts
  stays exactly as it was, that curve keeps its corner and can only get tighter (when the leg to
  the cursor was what limited it, since that leg is now shared with the new click's curve), and
  the new click becomes a curve as the cursor moves on (tested, notes/T-092.md). While drawing, the
  clicked points show as squares, each curve its radius and speed limit, and the cursor the
  stretch's length, average cost multiplier (level and water over its length) and price ("1.5 km,
  cost 1.33x, $25.5M", T-085), or why it cannot be built.
- **Reshaping blueprint track** (T-064, T-092): a selected stretch of blueprint track shows its PIs
  as squares on the map; dragging one previews the new shape with each curve's radius and speed
  (or why it cannot be built), letting go applies it as one undoable edit, Escape puts it back. The
  track inspector lists the corners with each curve's speed and a radius stepper (100 to 1,500 m,
  then auto) and puts the whole stretch on one level; its ends keep theirs, with ramps. A set
  radius that a drag leaves no room for goes back to auto (splitting track for a station or a
  branch sets the radii next to the split so its shape does not change). Constructed track cannot
  be reshaped (remove it and draw it again). notes/T-064.md.
- **Moving blueprint nodes** (T-093): with the select tool, a track end, junction or station can be
  dragged while every edge meeting there is blueprint and its station is not constructed; the
  blueprint track there follows, previewed like a reshape (each curve's radius and speed, or why
  not), one undoable edit when let go, Escape cancels. The edges' PIs stay where they are, except
  that a node where two or more edge ends meet keeps its heading (6.2): each edge's first PI, which
  holds that heading, moves with the node, and an edge with no PI of its own to move gets a
  lead-in PI as in drawing. A track end's heading follows its edge. Lines over it keep their stops
  and paths; their times change. A constructed node, or one where constructed track meets, stays
  put (dragging it when it is selected says so; otherwise the map pans as usual). notes/T-093.md.
- **Placing stations**: click on track (it is split there) or on a node. A new station is named
  after the nearest street on the basemap within 400 m ("W 42nd St"), skipping names in use, else
  a neighbourhood, else "Station N"; the player can rename it.
- **Curve speed** `v = sqrt(a_lat * R)`, **a_lat = 1.1 m/s²** (LGV design: 7,000 m for 320 km/h).
  100 m: 38 km/h, 300 m: 65, 1,000 m: 119, 1,800 m: 160. **Minimum radius 100 m.** The track has no
  speed cap of its own; the M1 train's top speed is **160 km/h**. Grades do not change speed.
- **Levels** -3 to +3, **8 m apart** (+1 viaduct, -1 cut and cover, -2 bored, -3 deep). **Max
  grade 4%**, so a ramp is **200 m per level**. Level is set per PI, at the level selected when it
  was clicked; a ramp is centred between vertices of different levels. Nodes and platforms sit at
  whole levels. Tracks cross freely
  where their heights differ by a level or more; at the same level they make a flat crossing; in
  between is invalid.
- **What the player builds**: a **route** drawn as one stroke, **double track by default** (4 m
  apart, running side per country from the city pack), with **single track** as a setting: cheaper,
  both directions sharing it, stations as passing points. (Anita, 2026-10-08; not NIMBY-style
  individual tracks.)
- **Junctions in plain words** (T-080): a new junction is built flat (the branch crosses the other
  direction's track at the same level: cheaper, trains may wait for each other). The junction
  inspector says so in plain words and offers "Build a flyover" with its price (paid at once on a
  constructed junction, else added to the blueprint) and the waiting it saves per train at the
  demand level in force; a junction's capacity marker opens that inspector. A blueprint flyover can
  be removed again. No setting while drawing.
- **No track at ground level over water** (level 0 is invalid where the water mask says water);
  bridges and tunnels over or under water cost 2x (6.4). The mask is 25 m pixels in the city
  pack; water narrower than ~50 m (creeks, ditches, small ponds) is not in it, since a railway
  crosses those at grade (notes/T-030.md). Ground level means within half a level (4 m) of the
  ground, so the low end of a ramp counts too: bridge and tunnel approaches start on land.
- **Stations** are nodes on a route, platform length 60-400 m in 20 m steps, level and free of
  other nodes along it; curved platforms are fine. Platform length caps train length (20 m cars).
  Platform tracks = the route's tracks. Riders change between stations up to 300 m apart on foot, and platforms within 100 m of each other count as one station (4.4).

### 6.2 Capacity and junctions

**The graph.** Nodes (end, junction, station; the kind follows from how many edge ends meet
there) and edges (one alignment between two nodes: its PIs with radius and level, and its track
count). **Flat crossings are not nodes**: where two edges cross at the same level, the crossing is
found from their geometry, costs money and loads capacity, but splits nothing, so dragging a PI
across track never rewrites the graph or line paths. Each edge gives one directed track per
direction (shared on single track): `(edge, direction)` is the segment in SPEC 2.1's keyframes.
At a node, every edge end leaves along the node's heading or its reverse; a move is any pair of
ends on opposite sides. In a **junction**, two moves conflict when their paths cross at the same
level (track positions in opposite order on the two sides); merges and diverges do not, as track
capacity counts them. A plain double-track Y has one crossing pair. A **flyover** (a flying
junction) removes it, for its price (6.1, 6.4). A **flat crossing** makes every move on one
route conflict with every move on the other.

**Resources and utilisation.** Everything shared has an occupancy `s` per train, and per period
`rho = sum(trains per hour x s) / 3600` from the scheduled frequencies:

- track section (a run with the same set of lines), per direction: `s = 90 s + v / 0.6 m/s²`, so
  33 trains/h at 40 km/h, 28 at 80, 22 at 160, 15 at 320
- single-track section between passing points: run time + 60 s, counting both directions
- platform track: dwell + 60 s per stopping train, 40 s per passing one
- terminus: (layover + 60 s) / platform tracks (a two-track terminus with 3 min turnaround: 30/h)
- junction move, with the moves it crosses; flat crossing, all moves: 90 s

**Delay per train** (seconds), Kingman's queueing formula with a small variability term because
trains are timetabled:

    mix = 1 - sum over (line, direction) streams of (share of trains)²
    W   = 0.2 x mix x s x rho / (1 - rho)        rho <= 0.9; straight-line continuation above
        + 1800 x (1 - 1/rho)                     when rho > 1 (the queue builds for an hour)

One line alone has no queueing delay (its trains are evenly spaced) until it is over capacity. A
90 s junction shared by two equal lines: 27 s at 75%, 81 s at 90%, ~3 min at 100%, ~7 min at 110%.

**One pass, in the clock worker.** Frequencies are the player's input, so delay lengthens round
trips and raises the trains needed but never changes a load: no iteration, recomputed on every
change to lines, schedules, crossings or junctions before demand runs. It is incremental: an edit
rebuilds only the resources around the edges, nodes and lines it touched and re-solves those and
the ones a rescheduled line uses, with the same result as rebuilding everything (T-050). In WASM
that is 0.02-0.2 ms an edit at 2,000 route-km with 60 lines and 1,250 resources, and 0.2-1.2 ms
at 10,000 km with 400 lines and 11,000 resources, where the whole pass takes 13 ms
(notes/T-050.md). Each delay is baked into the line's run-time profile per demand level as a hold
at the resource's entry: a slower approach when the delay is shorter than what braking to a stop
and starting again there would cost, otherwise a stop in front of it. So trains visibly crawl or
queue at a tangle.

**Shown** on the map as markers on resources at 75-90% (busy), 90-100% (near full) and over 100%
(overloaded), with the delay per train, for the demand level in force; nothing shows below 75%.
A marker reads "+1:03" (each train's wait there) or, with no wait yet, "81%"; hovering it shows
a tooltip in plain words ("Junction at 104% of capacity", "Each train waits 1m03s here at high
demand", and for a junction "Click to see the junction and its flyover"). Markers sit over the
network and under the demand views' bubbles and catchment (T-099).
The junction inspector (T-031) draws the junction as a small diagram and lists its moves (from
one station or direction to another), with the lines on each, trains an hour at the level in
force, the wait per train, which moves each crosses, and the busiest crossing's utilisation. The
line panel splits the round trip into running, dwell at stops, turning at the ends and waiting at
busy track, and names the places it waits most (notes/T-031.md).

### 6.3 Trains, lines, schedules

- **One train kind.** Later kinds, if any, differ by speed profile and cost (HSR, light rail, as in
  Subway Builder), never by urban vs intercity. The M1 train accelerates at 1.0 m/s² to 60 km/h
  and 0.5 m/s² above (a power-limited EMU, kept as two constant rates so motion stays exact) and
  brakes at 0.9 m/s²: a 1 km stop-to-stop run takes 67.5 s. The whole train must be under a curve
  limit before it may speed up again.
- **Lines**: ordered stops, plus the exact path between them stored as edges with a direction, so
  new track never reroutes a line by itself. Splitting an edge rewrites the paths through it;
  deleting one reroutes by the fastest path or marks the line broken. A line may pass a station
  without stopping.
- **Scheduling**: every line sets trains per hour for **high, medium and low demand** (the
  column header is "Demand"), always all three, no "same all day" shortcut. High is the two peaks,
  medium midday and evening, low night. The line panel shows for each the headway and the trains
  it needs.
- **Trip times on the stops list**: each stop shows the time from the start of the line, next to
  the station (e.g. Grand Central 1m23s, Long Island City 2m59s), and the panel shows the whole
  round trip.

### 6.4 Costs and money

Money is present, not tight, as in NIMBY Rails: construction, train purchase, operating cost per
car-km, a player-set fare curve by distance. One price list worldwide. Track costs a **base of
US$90M per route-km of double track** (single track 0.6x; ramps at the dearer level) times a
multiplier per level (Anita, 2026-10-08):

| Level | +3 | +2 | +1 | 0 | -1 | -2 | -3 |
|---|---|---|---|---|---|---|---|
| Multiplier | 1.8 | 1.3 | 0.8 | 0.3 | 1.0 | 1.1 | 1.2 |
| US$M per km | 162 | 117 | 72 | 27 | 90 | 99 | 108 |

- **Over or under water: 2x** the level's price. **Level 0 cannot be built over water.** Water
  comes from a mask in the pack (T-030).
- The Build tab shows only the multiplier and price of the level currently selected, not the
  whole table.
- Stations (200 m, two platform tracks): 10 at grade, 30/40/50 at +1/+2/+3, 60/100/140 at
  -1/-2/-3; scaled by `0.4 + 0.6 x length / 200`.
- Junction: 0.25 km of track at its level; a flyover adds 0.6 km one level away; flat
  crossing 0.1 km (cheap to build, dear in delay).
- **Blueprint, then construct** (Anita, 2026-10-09). New track, stations and junctions are drawn
  as a **blueprint**: free, freely edited and undone, drawn faded, shown with its total cost. The
  player pays when they click **Construct**, which builds the blueprint at once if the money is
  there: all of it, what one line still lacks (the line panel shows its price), one stretch of
  track, or one station. Only constructed track carries trains. Lines can be planned over
  blueprint track: they show their stop times, and run (load capacity, have trains, count for
  demand) once every edge and stop they use is constructed. A junction is paid when its third
  edge is constructed, a flat crossing when its second is.
- **Removing constructed track refunds nothing**, and the game asks before doing it (T-096). Undo applies to blueprint edits, lines and
  schedules. Constructing, removing anything constructed, making a constructed junction flying and
  lengthening a constructed platform are final (the last two are paid at once): they clear the
  undo history, so undo never builds or unbuilds anything for free.
- **Trains** (T-028): a 20 m car costs US$2.5M. The network owns cars; the running lines need
  their busiest schedule's trains times their cars. Any edit that makes them need more than are
  owned (a construct that starts a line, more trains an hour, a longer round trip, an undo) buys
  the difference with it, and is refused with nothing changed if construction plus trains is
  more than the cash. Cars are never sold: spares serve the next line that needs them.
- **Running cost** US$1.50 per car-km (trains an hour each way x line length x cars; set for play,
  a quarter of a real figure, T-091), and **fares**,
  are settled each game hour by the clock worker into the cash and a per-day ledger. Fare per ride
  = a base plus a rate per km of the ride, one curve for the network, set in the Money tab ($1.50
  + $0.10 a km to start); a ride that changes lines pays the base once. Fare income comes from
  demand's riders and passenger-km per period (`app/src/game/money.ts`). Demand does not respond
  to the fare yet (its rail cost has a fixed fare), so a higher fare only earns more (follow-up
  on the demand side). Running costs can take the cash below zero; edits that cost nothing still
  work then. notes/T-028.md.
- **Starting money $6B**, sized in T-008 for one real first line and rechecked in T-028: a 20 km
  line at -2 with 18 stations is US$3.78B plus about US$0.43B of trains, leaving room for a short
  second line.
- **Economy at 100x** (Anita, 2026-10-09, as Subway Builder does; T-081): fares and running costs
  per game day count 100 times the real figures (`ECONOMY` in `sim/src/track/params.rs`, the one
  constant; running costs scale in the track model, fares in `app/src/game/money.ts`), so a game
  day earns and spends what a hundred real days would. The fare the player sets and the US$1.50 a
  car-km shown are the figures before that scaling; construction and train prices stay real. The
  Money tab says so. At these prices the real New York network (T-007) earns about $1.71B a game
  day in fares against about $0.59B of running costs.

### 6.5 Edits

The clock worker owns the network. The main thread sends edit operations (each with an inverse,
for undo); the worker answers with the new state, then the edit's answer (or the reasons it was
refused). The alignment being drawn is previewed by the worker as a trial edit, applied, measured
and rolled back, so the preview knows everything an edit would refuse (0.4 ms a round trip). The
worker sends the whole network state after an edit, with the renderer's buffers already built
(it redraws only the edges and lines the edit dirtied, keeping the rest between edits); the main
thread uploads them. At 2,000 route-km an edit's round trip is 26-36 ms with 3-6 ms of it on the
main thread; at 10,000 km 130-214 ms and 23-36 ms, where sending only what changed is T-070
(notes/T-063.md). An edit dirties only what it touched:
render tiles (zoom-12 grid) of the old and new extent, crossings of the touched edges, the
resources on them, and the lines whose path uses them, then the other lines sharing those
resources (one step; loads never depend on delay). A city's demand is dirtied only by service
changes: a line's stops or frequency, or a station-to-station time moving 5 s or more. Drawing
track no line uses never wakes demand. Demand starts after edits pause (~0.5 s), and a newer edit
cancels a running solve. An edit is refused only for problems it adds (radius, ramps, crossings,
platforms, ground over water); problems already there, say from a newer water mask, do not block
edits nearby. Every edit returns its inverse, which is undo (for blueprint edits; 6.4). Blueprint
track wakes neither profiles nor demand until it is constructed, except a line planned over it,
which may show a preview. An edit and the profiles it changes
take 0.4-2.6 ms in WASM on a 2,000 km, 60-line network, and 1.5-2.5 ms at 10,000 km with 400
lines except on track many lines share (about 20 ms for an edge of 23 lines, nearly all of it
their profiles); track no line uses, under 0.1 ms (notes/T-050.md). A save holds
PIs (format TWT3, which also keeps the points a T-079 edge was drawn through; TWT2 saves load),
nodes, stations and lines: ~100 KB compressed for 10,000 route-km on a synthetic network
(notes/T-040.md; T-008 estimated 170 KB with denser drawing).

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
  with the same matrix; our own frame loop drives the overlay only for the clock. Drawn from our
  own loop alone, it trailed the basemap by one frame on every panning frame (notes/T-015.md).
  Station names are DOM labels above the overlay in the UI font, moved on MapLibre's frames, with
  a greedy overlap pass (transfers first, then lines); a MapLibre symbol layer would sit under
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
  lines, a small pool for city demand, one for long-distance later). The crate builds two WASM
  modules: the clock worker's holds the track model (about 430 kB, 165 kB gzipped) and the
  demand workers' the demand kernel (205 kB, 83 kB gzipped), so each worker downloads and
  compiles only its own and nothing is fetched twice; the main thread loads neither. Both are
  built at opt-level 3 (the pair loop is the recompute; smaller settings made edits 1.2-1.8x
  slower), notes/T-051.md. Flat typed arrays (structure of arrays), transferable
  between threads without copying.
- **The demand pool**: half the logical cores less one, at most 3, at least 1 workers. Each opens
  the city pack itself (its own copy, ~20 ms), runs a small warm-up solve so V8 has optimised the
  kernel before the first real one, and owns a fixed set of the five periods (worker i: periods
  i, i + n, ...). The clock worker's network snapshot after each service change goes to every
  worker; between periods a worker takes a newer one if it has arrived. Messages:
  `app/src/workers/demandProtocol.ts`; results land in `app/src/game/demand.ts`.
- **Threads talk by snapshots**: a worker transfers a finished buffer with a version number; the
  main thread swaps it in. Works on any host. SharedArrayBuffer is a later option; it needs
  cross-origin isolation headers, which GitHub Pages (Anita's website) cannot send and Cloudflare
  Pages can.
- **Data pipeline**: Python in the maps venv, offline, calling the sim crate natively where the
  game needs the same computation (the zone gravity). One **city pack** per city (boundary, cells
  with population, jobs and attractor weights, the solved zone gravity, road speeds per period,
  fitted constants, sources; format in notes/T-004.md) and one **world pack** (towns, long-distance zones and parameters, air routes). Binary arrays
  with a small JSON header, compressed; under ~10 MB for the biggest metro. Today the packs sit
  beside the page (`packs/<city>.json`, `.bin`); the main thread fetches a city's once and hands
  each demand worker a copy. Published, they move to R2 under a versioned path, compressed and
  cached for good (notes/T-026.md).
- **Determinism**: the simulation is a pure function of (save, pack versions, game time).

## 8. Look and UI

Cuter than Subway Builder and NIMBY Rails: a little pastel, a little Japanese, a little 2010 web,
still minimal with the map first. **The style applies to the UI only** (dock, panels, buttons,
bars). The network is not styled by us: lines, trains and stations take the line colours.

- **Fixed layout, not floating windows.** One dock on the left with tabs (build, lines, stations,
  money, city) and an inspector area under the tab content. A top bar for the mode split, money,
  day and time, the demand level ("High demand"), pause and three speeds. **The build tools
  (select, draw track, place station, level, single track) live in the Build tab of the dock**,
  not a toolbar along the map's bottom (Anita, 2026-10-09; T-082); the map keeps only the blueprint
  total, undo/redo and Construct where they are easy to reach. No floating windows; the player may
  drag the dock a little wider or narrower and move the divider between the tab area and the
  inspector (Anita, 2026-10-09), both remembered per browser; switching tabs still never moves the
  inspector. "Construct blueprints" builds every blueprint. While drawing, the tooltip gives the
  stretch's length, average cost multiplier and price ("1.5 km, cost 1.33x, $25.5M") instead of
  explanatory notes in the panel. A click where a train is drawn over a station selects the train.
  (The T-009 mock,
  `mock/`, is the approved starting point; `T-027.png` is the shell as built.)
- **Trains show who is on board** (T-083): the train inspector shows riders aboard now and its load
  against seats and crush (a bar with a tick at the seats; "Seats free", "Standing", "Over
  full"), from demand's riders on the segment it is running, divided among the trains the line
  runs in that period (an average train of the period, not its busiest hour).
- **The inspector shows whatever is selected**: a line, a station, a stretch of track, a junction,
  a train, commuter bubbles, a cell or zone. Nothing selected means it is blank (no text at all), not a default line. **Its top edge
  never moves when the tab changes**: the tab content area has a fixed height and scrolls.
- **Display settings**: the commuter bubbles (off, homes, jobs), what the track colour means (line colour, or height by level), and show or
  hide trains, station labels, capacity markers, basemap labels, the riders drawn on the network
  (line width, station size, how full trains are) and other layers as they appear.
  They live with the theme switch in a settings pane behind a gear button at the end of the tab
  row, shown in the tab area like a tab, so the inspector stays put. Remembered per browser. The
  same pane has the game's saves (section 11).
- **Height colours** (track coloured by level): warm above ground, grey at ground, cool below:
  +3 #c2410c, +2 #ea7a1a, +1 #f2b134, 0 #8c8c8c, -1 #5aa3d8, -2 #2f6fbf, -3 #23408e. Trains
  keep their line colour.
- **Picking on the map**: a click selects the nearest train (it is drawn over the station it
  stands at), else station, else (while commuter bubbles show) a bubble, else line, else junction,
  else track, within 7 px; clicking the line already selected selects the track under
  it. An empty click or Escape clears the selection.
- **Tools** in the Build tab: select, draw track (key 1), station (2), delete (3), the level with
  its price, double or single track, platform length for new stations. The number keys work from
  any tab and open the Build tab (the level and platform settings are there); the key of the tool
  already on does nothing. Each button shows its key. **Delete** (T-096): hovering shows what a
  click removes (a stretch of track drawn red, a station or a flyover ringed, its name by the
  cursor); blueprint is removed at once (undoable), and anything constructed is removed only after
  a yes in the hint area above the bottom bar ("Remove 1.2 km of constructed track? Nothing is
  refunded." with Remove and Keep; Esc is Keep). The inspectors' Remove buttons ask the same way.
  Lines are removed from the line panel. **Esc** undoes one thing at a time: a question waiting is
  answered Keep, a drag goes back, a half-drawn route is dropped (the tool stays on), then the tool
  is left for Select (drawing, stations, delete, making a line), then the selection clears. With
  select, the squares of a selected blueprint
  stretch (its corners) and any blueprint track end, junction or station can be dragged (6.1);
  the cursor turns to a move cursor over a node that can be; a line is made from the Lines tab ("New line",
  then click stations in order, Enter to end) and extended from the line panel. Ctrl+Z undoes,
  Ctrl+Y or Ctrl+Shift+Z redoes. Blueprint is drawn dashed and faded; the bar on the map's bottom
  left has undo, redo, the blueprint's cost and "Construct blueprints" (there because they are
  needed from every tab while working on the map). A hint for the tool and a refused edit's reason
  show above that bar.
- **A new game** starts on day 1 at 07:00 in an empty New York with $6B. Opening the page
  continues the last game (section 11).
- **Money tab**: cash; yesterday and today (fares, running trains, new trains, construction,
  net); the fare curve (each ride, each km); a day at the current schedule; cars owned and needed.
  The line inspector shows a running line's fares and running cost a day.
- **The goal number is always visible.** In a city: its mode split as one bar (e.g. driving 80%,
  train 16%, walking 4%); the player's job is to make the train share grow. Subway Builder does
  this well. It is the split of the day's commutes (other trips come with the second layer, 4.1),
  shown at 60% opacity while a newer network is being solved. Its three colours are the mode
  colours below, whatever the theme. Zoomed out to the world: one bubble per city or town for
  long-distance travel, sized by trips, showing rail's share.
- **Demand views** (Anita's pick, 2026-10-09; T-084, notes/T-084.md). Three layers drawn on the
  network, each switched off in the settings pane under "Riders on the network":
  - **Line width by riders**: each stretch between two stops is as wide as its riders an hour in
    the period in force on the clock, both directions together: 0.45 of the normal width when
    empty, growing with the square root to 2.2 times at 30,000 an hour. Lines sharing track stay
    side by side at their own widths, trains ride their own stroke, and a station's dot spans the
    widened bundle. Not drawn while the track is coloured by height.
  - **Station size by riders**: a station's circle grows with the square root of its boardings
    plus alightings a day, to 2.6 times its radius at 150,000.
  - **How full trains are**: a train's core fills from its back like a gauge, in the dark shade
    of its line colour over a pale tint of it: half full means every seat taken, full means packed
    to crush, and an over-full train is full with a red rim. The numbers are the train inspector's
    (an average train of the period on the stretch it is running).

  **Commuter bubbles**, the main demand view (T-097, Anita's Subway Builder style view, notes/
  T-097.md): "Commuters on the map" in the settings pane, Off, Homes or Jobs. Every commuter at
  that end of the trip is in a bubble (`cellModes`, 4.4): the city's res-9 cells summed into their
  H3 parent by zoom (resolution 9 from zoom 13.6 in, 8 from 11.8, 7 from 10, 6 further out, about
  1.4 zoom levels a step, so bubbles keep about the same spacing on screen), each bubble at its
  commuters' weighted centre. **Area is commuters** at a fixed density, 25,000 a km² for homes and
  80,000 for jobs (job centres are that much denser), the same at every level, times the "Bubble
  size" slider under the switch (0.5 to 2 times the width, 1 by default, remembered per browser,
  T-099); no minimum size,
  and nobody means no bubble. **Colour mixes the three mode colours by share** (RGB: all car red,
  all train blue, half and half magenta), at 80% opacity, biggest bubbles under smaller ones.
  The bubbles are drawn over the network and its capacity markers, under station names. A legend in the map's top left
  corner while they are on: the three colours with commuters and shares, and the mixes from all
  driving to all train.

  Hovering a bubble shows a tag by the pointer: commuters there and their split ("47k commuters
  live here", "Train 49%, walking 3%, driving 47%"). **A click on a bubble selects it**; a click
  with Shift adds it; **holding the mouse still for 0.4 s and dragging** draws a box and selects
  every bubble whose centre is in it. Stations and trains under the pointer are still picked
  before bubbles. A selection hides the other bubbles and shows the selected ones outlined and
  where their people work (or live) as bubbles sized among themselves (the biggest about its
  hexagon's area) and coloured by their own mix (`flows`, T-098). The inspector shows the
  selection's commuters and split, and the top eight far-end places, each named by the station
  nearest it within 3 km, else by distance and direction. Escape or an empty click clears it.

  **A selected station's commuters**: the station inspector lists commuters a day who live within
  its reach and ride from it ("Live nearby") and who work within its reach ("Work nearby"); the
  larger is picked first, a click picks the other. On the map, over the network, the chosen end's
  cells are opaque discs on top, the busiest a cell's area, and the places at the far end (up to
  20 gravity zones) filled discs at 55% opacity below them, the busiest half a zone wide; no
  minimum size; hovering either shows its riders. The inspector lists the top far-end places. A
  selected station hides the commuter bubbles.

  Data is asked for, not pushed (4.4): each view asks again when the demand result changes and
  keeps showing its last answer meanwhile. Bubbles are rebuilt for a solve's first estimate and
  again when its crowding rounds are done.
- **Mode colours**, Subway Builder's (Anita, T-097), fixed and hand-tunable in `game/palette.ts`:
  driving red #e61e1e, train blue #1e3ce6, walking green #1eaa3c, near-pure so that a mix reads as
  RGB (Anita kept the mix, T-099). A selected
  station's own cells #c2185b, its far end #3a3f8f.
- **Square buttons**, 1 px borders, small line icons as inline SVG.
- **Few colour tokens**: background, text, one accent, as CSS variables so other themes are a
  token swap. Light and soft, no paper texture or "ink" statement look (Anita: too statementy).
  **Three themes, all kept**: pink (cream background, sakura accent), blue (cornflower), green
  (matcha); a dark theme later.
- No decoration that does nothing: section headings are plain text (the mock's small coloured
  squares before headings are dropped).
- **Line colours are a normal transit palette, fairly saturated** (strong reds, blues, greens,
  oranges, purples, as on real metro maps), not pastel. New lines take the next default; the player
  can set any line's colour (later in M1 or after).
- **Zen Maru Gothic** (Anita's pick over M PLUS Rounded 1c) everywhere, including station names
  and the basemap's labels: Regular only, faint grey, from glyph tiles we generate and host beside
  the app (`app/public/fonts/`, `pipeline/glyphs.py`), with Noto Sans merged in for scripts Zen
  Maru lacks. Kana and kanji on the basemap are drawn by the browser in the system font for now
  (T-057). notes/T-056.md. Few font sizes and weights (the mock's 3 sizes and 2 weights is the
  ceiling).
- Instant UI state changes, no CSS transitions. Trains moving is the only motion.
- **Space bar** pauses, and resumes at the speed that was running before the pause; it, the
  number keys and Esc are ignored while typing in a field.
- **Trains** are capsules in a darker shade of the line colour with a white rim blended into the
  core over a pixel, every edge antialiased in the shader, and turned along the chord between
  their two ends rather than the track sample under their centre (notes/T-047.md). With riders
  known the core is a gauge of how full the train is (demand views, above).
- **Lines sharing track are drawn side by side**, touching, each at its own width, like a metro map: each
  keeps its place along a shared stretch, a branch line sits on the side it leaves to, trains
  ride their own line's stroke, and a station's dot spans the lines through it (notes/T-062.md).
- Player-facing text follows the copy rules in Anita's global CLAUDE.md (no dash-joined phrases,
  sentence case, no softeners, no hype).

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
  real timetables. The comparison log and what is off are in notes/T-007.md. **Where it stands**
  (T-090, the soft reach and the drive by density and length, three crowding rounds): rail 20.5% of
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
  explain (notes/T-006.md, notes/T-054.md, `T-006.png`).
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

- Local first (T-029): the clock worker builds and compresses the save (format `TWG1`: a JSON
  header and the track model's network bytes, gzip, `app/src/workers/saveFile.ts`) and writes it to
  IndexedDB itself: at once after anything paid for, else once a wall minute if something changed,
  and when the page is hidden. On start the last game loads (`?new=1` skips it). The settings pane
  saves the game to a file (`anitabuilder-nyc-day-12.save`), loads one, and starts a new game
  after a confirmation. One autosave slot. notes/T-029.md.
- Sharing: upload a save, get a link. Later (M4).
- Accounts only for sync and sharing, not needed to play. They run on `shonei-server`
  (`~/projects/shonei-server`, Anita's existing server), as the current plan.

## 12. Open questions

None at the moment.
