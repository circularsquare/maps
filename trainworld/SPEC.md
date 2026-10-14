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
  segment per period. The clock worker turns those per-period counts into a boarding account
  per line (T-164, docs/track-and-service.md 6.6): riders arrive at each platform at demand's
  rate, each train takes whoever fits up to its crush load, and the rest wait for the next. So
  each train has its own load and each platform its own crowd, worked out a day at a time when
  service or demand changes, never stepped per frame. Demand's crowding rounds run the same rule
  per period, so a line that leaves riders behind loses some of them (T-188).
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
includes surrounding Maryland, Virginia and Jefferson County, West Virginia. **Toronto** (the
Greater Toronto and Hamilton Area, 7.3M people) is the first city outside the US (T-277): the
same pack and game, built from Statistics Canada's census by Canadian adapters
(docs/adding-a-city.md "Outside the US"), with its own fitted commute curve, and not yet in the
regional market. City additions are
paused for now (Anita, 2026-10-12); when they resume, new cities reuse the established sources and broad metro extents, with basic pack/loading checks rather
than a new calibration exercise (Anita).
**Zones** (T-276, [docs/zones.md](docs/zones.md)): the world is divided into city zones (the city
packs), non-city zones (county-sized units between city zones, built as packs by the same
pipeline: Worcester, Windham, New London and Litchfield Counties) and dead space, where no trip
starts or ends. **Open** (Anita): whether modelled space should also reach a ring of counties
around each city. Mercer County (Trenton) is part of New York's pack since T-276 (New York CSA,
twice the commuting to New York's counties as to Philadelphia's). Each zone keeps its own local
pack; commutes are one gravity over all zones (section 4); shared lines retain their whole routes.
The top bar follows the focused zone, the one under the view or within 40 km of it; over dead
space it has no name and nothing loads (T-318). Exact pack boundaries show at regional zooms.

**Coordinates (T-125, T-306; [docs/world.md](docs/world.md)).** One save holds the whole Earth.
The track model is planar, in one frame for new games: spherical Mercator in metres
(`x = R lon`, `y = R atanh(sin lat)`), conformal, so curves are circles and angles at junctions
are true, with scale `k = 1/cos(lat)` (1.32 at New York, 1.61 at London). Lengths are measured on
the ground, not in the plane: every piece of track carries its ground length (a straight is a
rhumb line, exact in closed form; an arc by Simpson's rule on `1/k`), chainage everywhere is
ground metres, and metre figures entering the plane (a set radius, the radius cap, the minimum
radius, tolerances, quad-track spacing) are multiplied by `k` there. So lengths, curve speeds,
costs and run times are right anywhere: a 3 km line of one ground shape measures and runs the
same in New York and London (within 0.0003%), a London curve set to 1,500 m is 1,500 m on the
ground (within 0.03%, the rhumb line's own curvature), and track near New York matches the conic
of T-125 within 0.01%. Track is refused within 5 degrees of the poles and across the 180th
meridian. Saves are written in Mercator (TWT6). A save in an older frame, TWT5 the Northeast conic
of T-125 (a Lambert conformal conic tangent at New York, 0.4% scale error within 5 degrees of
latitude) or TWT4 and older the original equirectangular frame (1.2% short at Philadelphia), is
converted to Mercator once when it loads, through longitude and latitude (T-308, section 11).
Every demand pack keeps its own local origin
and frame; all conversions go through longitude and latitude. App code that compares world metres
with a metre figure goes through `groundScale` or `groundDist` (`game/coords.ts`).
`app/src/game/registry.json` is the one registry of the world, its frames, its regional market
pack and its cities; adding a region is an entry there plus its pack. notes/T-125.md.

- **World**: one real-world map. Basemap from vector tiles (Protomaps/OpenFreeMap), repainted
  faint so the player's network reads first.
- **City**: a metro area with a boundary drawn case by case (roughly Subway Builder's extent).
  Full local demand model. A subtle exterior shade and thin solid boundary identify the focused
  metropolitan pack; the normal view bounds cull the one continuous network.
- **Non-city zone** (T-276): a county-sized unit touching two city zones, a pack like a city's
  (same cells, purposes and views) with `kind: "zone"` in the registry. It replaces the old towns.
- **Cells**: the unit of local demand. H3 resolution 9 (~0.1 km², ~170 m edge), only cells with
  people or jobs. Resolution 8 (~0.74 km², Kontur's) is too coarse for walk catchment: half a
  cell is ~4 minutes of walking. Each census block is spread over the cells its polygon covers,
  by land area (T-013). New York (25 counties, 21.7M people, 21.6M outside institutions) is 202,330 cells and an 11.7 MB pack,
  5.9 MB gzipped, including commute, customer-sector and school weights and a 1.1 MB water mask.
  **Grain matters most in city centres** (Anita): keep res 9 or finer where it is dense; the
  suburbs may be coarser if performance ever needs it (T-044). Not needed yet.

**The network is one network.** There is no urban/intercity split in the track, trains or lines.
Any line can stop anywhere, and a big city will usually have several stations that long-distance
lines call at (Tokyo, Kansai). Only the demand is split into layers (sections 4 and 5).

## 4. Demand: local trips

Local demand models seven purposes in all nine registered cities' tier-1 packs: commutes,
shopping/errands and social/leisure outings (T-066), attendance at matched public schools
(T-140/T-150), university attendance (T-152), airport ground trips and medical visits (T-142),
every city from the same national and global sources (LODES sectors, NCES public schools, IPEDS,
FAA boardings with BTS DB1B connecting shares at OurAirports airports and OSM terminals, OSM
hospitals, clinics, doctors and dentists; T-142; New York's schools too since T-181).
**Homes leave out institutions** (T-234): people the census counts in prisons and jails, juvenile
facilities, nursing homes and other institutions (2020 PL 94-171 P5, per block) are taken off
every block's population before anything is built on it, and LODES job holders on those blocks
are capped at the people left; college dorms and military quarters stay (docs/demand.md 4.1).
Customer travel starts from all residents,
uses retail/service and food/recreation employment plus population as destination attractors,
with OSM parks taking 8% of the social/leisure draw (T-196: by area with diminishing returns and
residents nearby; ATUS 2023 puts 7.6% of leisure outings outdoors; parks move outings, they add
none), and has shorter distance decay and separate outward/return schedules. Non-work outings under
150 m between represented cell centres are excluded from every mode and total. Rates are
provisional, with no commute-share target. School destinations use each school's reported enrollment;
estimated child populations supply homes, with a provisional 90% typical-school-day attendance
factor, morning arrivals and mostly midday returns. University attendance is a separate
adult-origin purpose using IPEDS degree-seeking undergraduate and graduate enrollment less
exclusively online students, with a provisional 60% daily attendance factor and separate timing.
Each airport's daily ground trips are its boardings less connecting passengers, times the share
of its catchment inside the pack, at its terminals; the other end is residents and visitors
(jobs, hotels) by a 20 km decay, normalised so each airport keeps its total. Medical visits are
0.04 outings per resident a day (NHTS 2022: 0.091 medical legs per person) to OSM facilities
sized by beds or building floor area. Staff of both are already commuters. Private schools
remain a future layer (sources: notes/T-142.md).
**Commutes are one gravity over every zone** (T-276, docs/zones.md 4): solved offline over all
packs' 2 km zones together (`joint_gravity`), so a commute crossing a pack line exists like any
other; its within-pack part is each pack's local commutes, the cross-pack part (1.51M commuters in
the Northeast's 13 zones, 7.8%) travels in the regional layer (section 5). Inside a pack the gravity is the
pack's own, with the distance curve
d^-0.8 exp(-d / 30 km) for the US packs (T-198: flatter near home than T-162's d^-1 exp(-d / 40 km),
so residents of job-rich places commute out to suburban jobs about as often as LODES says; the
curve is carried in each pack and the game uses each pack's own: Toronto's, fitted on Canada's
census flows between municipalities, is d^-1.2 exp(-d / 20 km), T-277). Commutes between zones in
different first-level regions (US states) have their distance decay multiplied by the pack's
border factor, 0.23 for the US packs (fitted on New York and Philadelphia LODES) and 0.45 for
Washington (its own fit: DC is a city counted as a state, T-181), and a square a state line cuts
is one zone per state (T-162). **Borders outside the US** (Anita, T-181; no such pack yet):
national borders get a factor, weaker inside the Schengen area than across other national
borders; first-level regions (states, provinces, prefectures) get none, factor 1, by default.
The values are fitted when the first such pack is built. The US states are the exception because
LODES shows they measurably cut commuting (T-159, T-162); elsewhere an admin-1 line is not
assumed to (Tokyo's prefectures exchange millions of commuters). A pack without a factor has 1.
Then station access, destination-specific common-line strategies, fare-sensitive mode choice and
three crowding rounds recompute after service or fare changes. Passengers are flows, with per-
line loads and on-demand place queries. Each crowding round uses the combined loads of all
cities, their everyday travel and the regional market, and prices both crowded trains and riders
left behind by full ones (the boarding account's rule run on each period's trains, T-188). City shares and local bubbles include the
focused city's modelled local trips. Bubbles count trip legs, outward and return together;
selection shows purpose counts/shares and five departure periods separately for each direction,
with all-mode and train-only scope. Station catchment queries still explicitly count commuters.
For local trips, the distance-based walking share is reserved before rail and driving compete
for the remainder; useful rail cannot remove the baseline walk share of a short trip. Commutes
use 2 km gravity-zone distance while everyday purposes use sampled cell distance (T-153).
**Walks and drives follow real paths** (T-275): walks to stations and the walk share go round
water and over bridges (a 50 m barrier raster from the pack's water mask with OSM's walkable
bridges, searched from each station in the game), and a drive's road is the straight line times
a routed factor between the two zones (OSM roads, per pack, scaled to New York's fitted level),
at the same density speeds; `pipeline/routes.py` adds both to a pack (docs/demand.md 4.3).
Riders change trains at a station, walk after a ride to any station within 300 m, and walk
between any two stations of one station complex (section 8, the player's links included) however
far apart, each walk costed by its length (T-161, docs/demand.md 4.4).
One trip between two points can be asked of the solved model as a commute: its station pairs,
routes through the strategies with their shares, the drive and each mode's chance, the same
numbers mode choice used (T-195, docs/demand.md 4.4).
**Weekdays and weekends** (T-260): game day 1 is a Monday and days 6 and 7 of each week are the
weekend. The solve is a weekday's; a weekend is worked out from it per purpose and period, not
solved again (NHTS 2017 weekend rates: work 0.30 of a weekday, shopping 1.22, social 1.60, school
0.08, university 0.22, medical 0.24, inter-region trips 1.58; airports 1.01 from TSA; weekend timing
shifted by NHTS's period ratios), keeping the weekday's route and mode choices. The same trains run
every day. **Open** (T-261): whether weekends get their own trains per hour, which would need a
weekend solve. docs/demand.md 4.2.
**Local time** (T-309, docs/world.md "Local time"): one game clock keeps world time in UTC-5; each
region's periods, weekends, boarding hours and fares run on its own local time (a registry
`utcOffset`, a whole hour of standard time, no daylight saving; a place in no zone takes
`round(lon / 15)`), a line on its first stop's region's, and the top bar shows the local time and
day where the player is looking.

Read the full numbered section in [demand.md](docs/demand.md).

## 5. Demand: the regional layer

The regional layer carries what crosses zones or goes far (T-117, T-276; design and reasons in
[docs/zones.md](docs/zones.md)): **inter-region trips** (visits, business, leisure, 25 km and more) within
and between every zone, and **commutes between zones**. Market ids are stable and new zones
append; Trenton's id (2) stays with no sites since Mercer joined New York (T-276).

### 5.1 The total travel market, then rail's share

Generate **total** all-mode trips, then let the player's network earn rail's share.
The pack has 258,013 population/job-weighted endpoint sites at 600 m grain from every zone pack,
grouped into 3,260 zones of 6 km. **Inter-region trips** (T-276): each resident makes 0.107 departures a
weekday, shared by attraction `population + 2 * jobs` x an onset from 0 at 25 km to 1 at 50 km x
`max(distance, 25 km)^-2.37` among every zone **and an outside world** (Kontur population living in
no zone, at the zones' attraction per person), so 0.76M of the 5.13M departures leave the map
and adding a zone later moves only the trips that went where it now is. Rate and curve are fitted to
NHTS 2017 Northeast weekday trips by ground of 50-1,000 km (research/intercity/market_fit.py);
distances are great circles between zone centres (T-128). Opposite zone-pair totals are averaged;
tiny pairs are pruned: 4.37M trips a day between represented zones, 4.09M directed pairs. Local
outings still taper over 40-80 km (their decays leave almost nothing past 25 km).
**Commutes between zones** (T-276): the cross-pack part of the joint commute gravity (section 4),
1.50M commuters a day spread from 2 km gravity zones to the market's zones and kept where at least
one a day travels between two 6 km zones (172k pairs, each home zone's dropped commuters given to
its kept pairs). Each commuter travels at the local commute's timing (70/12/6/4/8% to work, the
return 2/18/58/16/6% on the same station pair reversed) and 0.30 of a weekday's at the weekend.

The figures that need every zone pair (each zone's normaliser with the outside world, the day's
trips, each corridor's, the departures leaving) are computed offline by the game's own code into
the pack's `derived` block; the game builds zone pairs and commute pairs only between zones a
station serves, the only ones that can carry a rail trip, and a selection's or far zone's all-mode
trips row by row on demand (T-174). Opening the market costs reading its sites and commute list,
not its pairs. The block names its formula and is ignored if stale (derived at opening instead,
same result): `pipeline/intercity.py --derive` refreshes it in seconds. docs/demand.md 4.4.

**Inter-region trips** compete with driving through a logit at 0.025 per perceived minute. The car takes
15 minutes plus 1.25 times the great-circle distance at 85 km/h, plus (T-276) 15 cents a road km
and the local layer's parking at the destination, both split over two occupants, at 3 perceived
minutes a dollar; no tolls (T-270). **Commutes between zones** choose as local commutes do: 0.05
per perceived minute, the drive weighted 1.33 plus the work end's parking. Fares, service,
transfers and shared crowding affect rail; the commutes' loads are the work purpose's (T-158).
Inter-region trips' period shares are 20/35/25/15/5 percent. Air, coach, induced travel, park-and-ride
and routed drives between zones (T-275 routes drives inside packs) remain future work.

### 5.2 Where a long-distance trip starts and ends

Origin access is population-weighted; destination access uses population and jobs. Each site
offers nearby stations through the same soft fast-walk catchment as local demand. Up to eight
weighted station options per zone end feed the common strategy graph. Local feeder lines,
transfers and the intercity service receive actual loads and boardings. No express-only route
or station-city affiliation bypasses the player's local network.

**Seeing inter-region trips (T-118).** The commuter views cannot show them: a Philadelphia bubble's
inter-region trips end in New York, off the screen. So a selected bubble set and a selected station
each list them by all represented places and
draw them by far zone up close, aggregating to labeled other-city totals at regional zooms
(T-135, T-136). Same-city inter-region trip bubbles are omitted at every zoom (T-139); their counts
remain in the inspector. Bubble area represents total travellers within the inter-region trip layer;
colour represents the train share, red at zero and blue at 100%.
Inter-region trips use their own visible size scale, blue for train and red for everything else;
section 8 has the display. A selection of local cells is mapped
to the market through its 600 m squares: the pipeline built the sites from the same cells, so each
selected cell falls in one site's square, a square counts the selected share of its commuters
(homes or jobs, whichever end was selected), and a site's share of its zone's departures is its
population over the zone's (of arrivals, its attraction over the zone's). Numbers are the
market's own, summed from its zone pairs; nothing is estimated again. A station's riders come
from the same assignment that loads the lines, so they add up to its entries and exits.

### 5.3 Park-and-ride

Access uses the same fast walk at both ends; any player-built feeder line participates in
regional routing. Park-and-ride is planned: without it a station's commuters between zones come
only from within walking reach, so commuter rail between zones carries a third to a fifth of
real figures on a test network of five real corridors (notes/T-276.md).

### 5.4 Size

The regional pack is 13.9 MB (8.2 MB before T-276): its sites, the outside world's 5,991 points
and the commute list. It runs in the same bounded worker pool as the local city packs, with
periods divided among workers. Each period advances all populations from one frozen aggregate
flow vector per crowding round. Details, assumptions and checks: [T-117](notes/T-117.md),
[T-276](notes/T-276.md).

## 6. Building and running

The player draws and edits curved track, stations and lines as blueprints, then pays to
construct them. Whole-line train allocations determine congestion-adjusted headways; legacy
frequency schedules still load. The clock worker owns edits, capacity, fleet, costs and money,
with undo for reversible changes, and the boarding account: each train's load and each
platform's crowd from demand's riders, with full trains leaving riders for the next (T-164).
There is one account per kind of day, and each game hour's trains and platforms, and its fares,
follow the clock's own day (T-260). Fares are paid into the cash as each hour's trains pick up
riders, and running costs drain evenly, not once an hour (T-284).
Changing a line's trains moves no train: running trains carry on, added ones enter at a terminal,
removed ones leave at one, and the same hand-over happens at each change of period (T-206).
A drawn route is one piece of track per corner clicked (T-244), as single, double or quad track
(T-255: a double track with an express pair beside it).

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
  A station complex's stations that share a name have one label at their mean position (T-169).
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
  modules: the clock worker's holds the track model (about 650 kB, 240 kB gzipped) and the
  demand workers' the demand kernel (about 610 kB, 235 kB gzipped), so each worker downloads and
  compiles only its own and nothing is fetched twice; the main thread loads neither. Both are
  built at opt-level 3 (the pair loop is the recompute; smaller settings made edits 1.2-1.8x
  slower), notes/T-051.md. Flat typed arrays (structure of arrays), transferable
  between threads without copying.
- **The demand pool**: half the logical cores less one, at most 5 (one per period), at least 1
  workers that solve, and one more with no periods. Every worker opens only the city packs the
  game needs, each in its own copy (T-149): the viewed city's first, and every city with a built
  station inside its footprint or within 4.5 km of it, since only those can put a rider on a
  line. A city without construction changes no number until it is viewed, and then loads
  without blocking the game. Solving workers also open the regional market, always. A solve
  waits until every city it needs is open in every worker, so a published result never leaves
  out a constructed city. The regional header parses into compact numeric site
  rows; immutable trip tables release spare allocation capacity (T-123); opening the market
  visits no zone pairs (T-174, 5.1). Workers run local warm-up solves so V8 has optimised the
  kernel before the first real one. Worker 0 first opens a small synthetic everyday layer, and
  the other solving workers get city packs only once it is ready, so every worker opens a city
  on optimised code (T-173: five workers opening New York at once from baseline code took 7-8 s
  each in most page loads, against 1.3-1.8 s). Each worker owns a fixed set of the five periods (worker i: periods
  i, i + n, ...). The extra worker never solves, so it answers at once what needs no period's
  result: the city's cells and each bubble's commuters by mode. The clock worker's network
  snapshot after each service change goes to every worker; between populations a worker takes a
  newer one if it has arrived. A worker sends its sums for the bubbles with each round, unasked.
  Messages: `app/src/workers/demandProtocol.ts`; results land in `app/src/game/demand.ts`. The
  top bar says which step of the solve is running (the free-flow estimate, then each crowding
  round). **Only final results replace final results** (T-232): after an edit every figure (the
  top bar's split, line and station riders, bubbles, the boarding account) keeps the last
  finished forecast, faded with "Updating demand 2/4…", until the new solve's last round lands,
  so figures change once per edit instead of stepping through the rounds. The earlier rounds are
  shown only where nothing final is: at startup, after a whole-game load, or for a city just
  opened; the top bar then says "Estimate, refining 2/4…". **A line new or changed by the edit**
  (its stops differ from the shown result's) has no figures to keep, so it takes the free-flow
  estimate's riders as soon as that lands, as do stations only such lines serve; its final round
  replaces them like everything else (T-324: a new Toronto line showed riders 5 s after it was
  built instead of 16-17 s, final figures unchanged). Until a running line has any figures its
  trains are drawn striped in the line colour and white, never solid like a full train, and the
  train panel says "Calculating". Timings on the real network: docs/demand.md 4.4.
- **Loading and readiness** (T-149, T-172): input is never blocked; the map, inspectors and
  building work as soon as the network is drawn. A loaded game (the page opening, a file, a
  browser slot, a new game) holds only the game clock, until its network has its first estimate
  (every period's free-flow result) for every city it needs; the crowding rounds then refine it
  in the background as after any edit. Meanwhile a card in the middle of the map, which takes no
  input, says "Loading data" over a filling bar and the current step (T-180, T-212: no city
  names; T-243: a bar, the percentage only on hover); a city loading later says it in the top bar
  with a thin bar, the cities named only in its hover. The bar moves in proportion to time
  (T-243): downloading, opening the packs in every worker and the first estimate each weigh as
  long as they are expected to take (a download by its own rate, opening by pack size, the
  estimate by the network's line stops scaled by this browser's last load), and inside a step it
  moves with time against that expectation until the step's real end lands (notes/T-243.md).
  If demand cannot load a screen says so and offers to reload or play
  without demand. The demand pool starts before the map is set up, the regional market goes to the
  workers as bytes and is decoded there (T-180), and each city's boundary file is fetched once for
  the reach test and the map's outlines (`game/boundaries.ts`). `app/src/game/loading.ts`,
  `app/src/workers/cityReach.ts`, notes/T-149.md, notes/T-172.md (timings), notes/T-180.md (frame
  times while loading).
- **Threads talk by snapshots**: a worker transfers a finished buffer with a version number; the
  main thread swaps it in. Works on any host. SharedArrayBuffer is a later option; it needs
  cross-origin isolation headers, which GitHub Pages (Anita's website) cannot send and Cloudflare
  Pages can.
- **Data pipeline**: Python in the maps venv, offline, calling the sim crate natively where the
  game needs the same computation (the zone gravity, the regional market's all-pairs figures). One **city pack** per city (boundary, cells
  with population, jobs and attractor weights, the solved zone gravity, road speeds per period,
  fitted constants, sources; format in notes/T-004.md). The current regional market contains
  endpoint sites, tunable trip parameters and its derived block (T-174); a broader world pack with air routes is planned. Binary arrays
  with a small JSON header, compressed; under ~10 MB for the biggest metro. Today the packs sit
  beside the page (`packs/<city>.json`, `.bin`, and `packs/intercity.json`); the main thread fetches each it needs
  once (T-149) and hands each demand worker a copy. Published, they move to R2 under a versioned path, compressed and
  cached for good (notes/T-026.md).
- **Determinism**: the simulation is a pure function of (save, pack versions, game time), except
  for the hour or two after an edit to a line's service, when the trains drawn also depend on the
  game time the edit was made (T-206; not saved, so a reload re-spaces them once).

## 8. Look and UI

The interface uses Zen Maru Gothic in a light pastel style with three colour themes, for the UI
only: the network takes the line colours. One dock on the left holds the tabs, icons with their
names on hover (Build, Lines, Stations, Money; T-252), the saved games button and the Settings
gear at the row's right end, and under them the inspector, whose top edge never moves when the
tab changes. A responsive top bar holds the mode split (the goal number), the cash (to the dollar, moving
as fares are paid and running costs drain, T-284), the day of the
week, day and time, the demand level, pause and three speeds. Every demand view (split, line
widths, station circles, bubbles, inspectors, fares a day) shows the clock's own day by default,
switching as the clock enters and leaves a weekend; Settings' "Demand shown for" can hold it on
weekdays or weekends instead, remembered per browser. The trains and platforms always follow the
clock. Line widths are scaled to the busiest stretch-day of either day, so a weekend's lines are
thinner (T-260, T-266, T-274). The saved games open in a centred panel (T-251,
section 11). Panels show labels, numbers and states, not explanatory notes (T-253). Everything
drawn that encodes a number has a hover stating it, and every hover is the game's own tag, never
a browser tooltip (T-194, T-210). A Panel and text size setting (80-120%, T-247) scales the panels
and the map's text tags, never the map.

**The network on the map.** Lines are thin (0.85-6.6 px from zoom 9 to 15, T-191, T-205), and
lines sharing track are drawn over each other, not side by side (T-190). Four track displays:
Line, Line with traffic thickness (the default: width linear in a stretch's riders over the whole
day shown, from half a line to 6.6 lines for the network's busiest stretch-day, T-227, T-274; it
does not change with the hour), Height and Max speed. Below zoom 11 only
the lines are drawn (T-205). From 11 to 15 every station is its own dot, a complex's dots
merged under one outline and joined by connectors of the same white and rim where they do not
touch, the player's links too (T-313); from 15
a station is its platform, a band of fixed ground width that follows the track's own centre line
(T-204, T-254). Platforms drawn over each other are one white shape with one dark outline, the
track running through it and each platform's long edges as thin blue lines (T-236). Trains are
pills that bend with the track, separate cars from zoom 15, filled like a gauge by their riders
with a red edge when packed full (T-207, T-218); a selected train has a halo and the selection
follows it onto its next trip (T-242, T-245). Platforms fill with grey as riders wait, red when a
full train left some behind, and further out a dot is tinted by its fullest platform (T-219,
T-228). Station names and capacity tags hide below zoom 12, and a tag never covers a name (T-184). Each city's
pack boundary is outlined at regional zooms, and the focused city's outside is shaded (T-139).

**Demand on the map.** The None / Homes / Destinations selector at the upper left draws the local
trip bubbles, coloured by mode share (T-097, T-132); a purpose filter at its end draws one trip
purpose (T-158; school and university are shown as one, T-192). The focused zone is the one
whose boundary holds the view's centre (the nearest over dead space), and up to three more zones
on screen show their bubbles beside its own; a click on one focuses that zone and selects the
bubble (T-276, [zones.md](docs/zones.md) 6). Riders, loads and fares follow a
new result at once, while the bubbles move to it over a game day (T-156). After an edit every
figure keeps the last finished forecast until the new solve's final round lands (T-232). Zoomed
out to 8.5 and below each city is one inter-region trips bubble (T-186); inter-region trip bubbles are always
drawn, except while the map shows one local purpose (T-222). Station demand circles are an option
in the Stations tab (T-213). A "5 stations full" button under the selector counts the complexes
with full platforms or riders left behind by the last train (T-231, T-284).

**Today, not a forecast** (T-293, Anita): the line and station panels and the Lines and Stations
tabs count the day so far from midnight (riders, boardings, fares, running cost, trains that left
full, riders left behind), from the boarding account's departures up to now, moving as trains
run; they show no whole-day forecast. The map's demand views stay demand's whole-day picture.

**Selecting and inspecting.** The inspector shows whatever is selected: a line, a station
complex, track, a junction, a train, bubbles, a city or a trip. Stations whose platforms come
within 50 m form a station complex, and the player can link any two stations up to 1 km apart
into one (T-146, T-197); riders change between a complex's stations on foot (T-161). A click on a
station selects its whole complex (T-249): its lines, one row each with riders waiting now and
boardings today (T-250), then a small map of its platforms with their crowds, the trains standing
or pulling in and the busiest changes between lines (T-147, T-176, T-246). Search (a magnifying
glass, `/`, Ctrl+K or Ctrl+F) finds stations, lines, cities and the places on each city's OSM
place list, and moves the camera there at once (T-148, T-177, T-211); no addresses. The trip
explorer (key 4, or Explore a trip in Settings) shows how the demand model's people would make
one trip between two places or stations (T-195, T-208, T-221). The build tools, their keys and the
cost breakdown are in the Build tab (T-100).

**Loading.** Nothing is blocked while demand loads. A loaded game's clock waits for the first
estimate behind a "Loading data" card with a filling bar; a city loading later shows a thin bar
in the top bar (T-149, T-172, T-243; section 7).

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
  real timetables. The comparison log and what is off are in notes/T-007.md. **Latest results**
  (after T-198's flatter curve, notes/T-198.md): rail 20.6% of commutes against ACS's 20.3%,
  county error 2.1 points (weighted mean); subway 2.74M riders a day, LIRR 314k, Metro-North 206k,
  NJ Transit 240k (about 250k real), PATH 49k (212k real, T-171); walk 4.9% of commutes against
  ACS's 6.2%; fullest morning segment LIRR Babylon Branch at Massapequa Park, 108% of crush. The
  busiest subway links were last compared at T-090, at 76% of their measured busiest hour (T-216
  rechecks them). Rough on purpose: realism is calibration, not the goal (section 1). Earlier
  checkpoints: notes/T-090.md, notes/T-162.md. With routed walks and drives (T-275, the same
  day with the everyday layer and three shared rounds, notes/T-275.md): commute rail 19.7% to
  19.2%, county error 2.41 to 2.49 points, North Hudson 13.2% to 9.3% (ACS 4.7%), the Hudson
  waterfront 26.9% to 28.1% (ACS 40.9%), PATH commute trips 124.5k to 128.7k.
- **Tier-2 check** (T-006, T-054, T-162, T-198, against LODES OD 2023): with d^-0.8 exp(-d/30 km)
  and the 0.23 border factor, New York's commute lengths match LODES to 2.7% of trips misplaced by
  2 km band (mean 18.8 km against 19.6), 8.7% of commutes misplaced between county pairs, own
  county 40% against 43%, between states 8.5% against 8.5%; Philadelphia 4.4% and 7.1% misplaced,
  own county 52% against 55%, between states 10.8% against 10.1% (notes/T-198.md).
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
  saves. Real-network test passes roughly. Done.
- **M2, Northeast** (under way): nine full cities and the regional long-distance market are in
  (section 3, T-116, T-117, T-134, T-137), with zones and commutes between them (T-276);
  park-and-ride access and how far modelled space reaches remain.
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
Stations the player linked into one complex are another optional trailing block (`LNK1`, T-197),
written only when there are links, so a save without them is byte for byte as before.
New headers retain `city: nyc` and add optional region/city-pack metadata (the registry's world
and cities). The frame is not in the header: the network bytes say it (TWT6 is the Mercator world
frame, T-306; TWT5 the Northeast conic, T-125; TWT4 and older the original equirectangular one).
TWT6 is TWT5 byte for byte but for the magic; its origin fields keep the game's home origin, which
the frame does not use, and its radii are ground metres. A file or browser slot in an older frame
is converted to Mercator when it loads (T-308): the clock worker runs `convert_save_frame` on the
network bytes, loads the result, and writes the slot again at once in TWT6 (the player sees
nothing; the old bytes stay under a hidden key `old-frame-<slot>` in the same database). Cash,
fleet, fares and the ledger are copied as they are; track lengths follow the true ground (the
real NYC save: -0.04% in all, no line's run time over 0.3%), so quotes and run times move by that
much and money already spent does not. Every city's track, service and finances save in the same
network (T-116).

- Local first (T-029): the clock worker builds and compresses the save (format `TWG1`: a JSON
  header and the track model's network bytes, gzip, `app/src/workers/saveFile.ts`) and writes it to
  IndexedDB itself: at once after anything paid for, else once a wall minute if something changed,
  and when the page is hidden. On start the last game loads. `?new=1` starts a new game
  that reads no slot and owns none: it does not autosave until "Save as a copy" gives it a slot of
  its own (T-320); `?synth=1` and `?demandTest=1` load the last game but do not autosave it either,
  and the top bar says so. The saved games
  panel (T-251: a centred panel opened by the floppy disk button beside the Settings gear, or by
  the top bar's Autosave off notice) saves the game to a file (`anitabuilder-nyc-day-12.save`),
  loads one, and starts a new game after a confirmation that replaces only the active slot. **Named browser slots** (T-068) let
  the player save the current game as a new copy, open another game, rename a slot, or delete
  an inactive slot after confirmation. The selected slot is remembered, and the original
  autosave remains at its existing key ("Original game"). Each slot is shown as a card (T-241,
  docs/interface.md 8): a picture of its network, its name, when it was saved, its day, cities,
  lines, stations and cash, with an Open button on every game but the one being played. Each slot write
  stores that summary beside the save bytes (`SlotSummary`: a 384 x 256 PNG, shown at half size, of the track and
  line routes drawn in the worker on an OffscreenCanvas, the stations' cities, line and station
  counts, cash); the save file itself is unchanged. Switching waits for the current writable game's save
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

- **Weekend service** (T-261, section 4): the same trains run every day now. Should weekends get
  their own trains per hour (or run a lower level), at the cost of a weekend solve?
