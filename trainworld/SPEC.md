# anitabuilder (trainworld) — spec

**anitabuilder** is the game's name wherever a player sees it (page title, UI, file names a player
saves), lowercase. **trainworld** stays the internal name: this folder, the crate, packages, the
serve.py registration and code. A browser game for building rail on a real world map, in the line
of Subway Builder and NIMBY Rails: Subway Builder's data-driven demand, NIMBY Rails' whole world,
and neither one's performance problems.

This file is a summary of what is true **now**, one short section per area with a link to its
detailed doc. It is rewritten in place when a decision changes; the history and the reasons live
in `DECISIONS.md`, finished work in `ARCHIVE.md`, measurements in `notes/`. Where it says
**open**, ask Anita; everything else is decided unless she changes it.

| Area | Detailed doc |
|---|---|
| Performance ideas, technology, saves (2, 7, 11) | [docs/architecture.md](docs/architecture.md) |
| One world: frames, coordinates, local time (3) | [docs/world.md](docs/world.md) |
| Zones and how demand is split between them (3, 5) | [docs/zones.md](docs/zones.md) |
| Local demand (4) | [docs/demand.md](docs/demand.md); abroad: [docs/demand-worldwide.md](docs/demand-worldwide.md) |
| The regional layer (5) | [docs/regional.md](docs/regional.md) |
| Track, trains, capacity, money, boarding (6) | [docs/track-and-service.md](docs/track-and-service.md) |
| Interface (8) | [panels](docs/ui-panels.md), [inspectors](docs/ui-inspectors.md), [network on the map](docs/ui-map.md), [demand views](docs/ui-demand-views.md), [building](docs/ui-building.md) |
| Adding a city | [docs/adding-a-city.md](docs/adding-a-city.md) |

---

## 1. Priorities, in order

When two goals conflict, the higher one wins. **Current focus (Anita): content and gameplay.**
Performance is adequate; the principles below stay architectural constraints, and optimisation is
parked until Anita reports a concrete problem.

1. **The game never freezes.** The main thread draws and handles input, nothing else. Anything
   that can take more than a few milliseconds runs in a worker, and the UI keeps showing the last
   finished result until the new one lands. No edit, load, save or recompute may block input.
2. **Low CPU.** Work happens when something changes, not every frame. The ~10k-train goal is about
   passengers, whose cost scales with network and demand, not train count.
3. **Demand that feels earned.** Ridership follows from where people live and work and how good the
   trip is, and the player can see why. Realism is for rough calibration, not a goal in itself: a
   real network should give ridership of the right order (section 9). Prefer simple, interpretable
   rules over faithful ones.
4. **Many cities, one world.** Several full cities, links between them and to smaller places, in
   one save.
5. **Cute, clear look** (section 8).

## 2. The core performance idea

**Trains are cheap; passengers are expensive.** A train's position is `profile(t - departure)`,
so nothing ticks per train. Passengers are flows, not agents: trips per period between places,
route and mode choice solved in workers when service changes. The clock worker turns the loads into
a **boarding account** per line [T-164]: each train's load and each platform's crowd, worked out a
day at a time. [docs/architecture.md](docs/architecture.md) 2.

**2.1 No signals, for now.** Trains do not block each other; tangles cost delay through capacity
computed from schedules (section 6). Trains are drawn from keyframes on a directed track graph, so
a block simulator could come later without touching renderer or demand.

## 3. World, zones and cities

**One network, one world** [T-125, T-306; [docs/world.md](docs/world.md)]. A save holds the whole
Earth: the track model is planar in spherical Mercator with every length measured on the ground,
so lengths, speeds, costs and run times are right anywhere (no track near the poles or across the
180th meridian). Demand packs keep their own frames; `app/src/game/registry.json` is the one
registry of frames, packs and the market. Track, trains and lines have no urban/intercity split;
only demand is split into layers. **Local time** [T-309]: the clock keeps UTC-5; each region's
periods, weekends and fares run on its own `utcOffset`, a line on its first stop's (rest: T-321).

**Zones** [T-276, [docs/zones.md](docs/zones.md)]: the world is divided into **city zones** (the
city packs), **non-city zones** (county-sized units touching two city zones, built as packs by the
same pipeline, `kind: "zone"`) and **dead space**, where no trip starts or ends.

- **City zones**: New York (with Mercer County since T-276), Philadelphia-Camden-Wilmington,
  Boston (with southern New Hampshire), Washington (with surrounding Maryland, Virginia and
  Jefferson County WV), Baltimore, Providence, New Haven, Hartford and Springfield, in one regional
  market [T-116, T-134, T-137]; **Toronto** (the Greater Toronto and Hamilton Area), the first
  outside the US, built from Statistics Canada by Canadian adapters with its own fitted commute
  curve, outside the regional market (`market: -1`) [T-277].
- **Non-city zones**: Worcester, Windham, New London and Litchfield Counties. Modelled space stays
  between city zones, with no ring round each city, for now (Anita, T-300).
- **City additions** in the US are paused; a **sampler of other countries' cities** comes next
  (London, Paris, Berlin, Shanghai, Mumbai), each a low-fidelity first pass from general sources
  (`research/world-sampler.md`). **Cells** are H3 resolution 9 (about 0.1 km²).

## 4. Demand: local trips

Each zone's local layer models seven purposes from the same national and global sources [T-066,
T-140, T-142, T-150, T-152, T-181]: **commutes** (a doubly constrained gravity from LODES homes to
jobs), **shopping and errands** and **social and leisure** outings (retail, food and recreation
jobs, population, and OSM parks for 8% of leisure, T-196), **school** (NCES public schools),
**university** (IPEDS), **airport** ground trips (FAA boardings less connecting passengers) and
**medical** visits (OSM facilities). Private schools are not modelled yet. **Homes leave out
institutions** (prisons, nursing homes and the like, T-234).

- **Commutes are one gravity over every zone** [T-276], solved offline (`joint_gravity`); the
  cross-pack part travels in the regional layer. Each pack carries its own distance curve (US
  d^-0.8 exp(-d / 30 km), T-198) and border factor (US states 0.23, Washington 0.45; abroad,
  national borders only, T-181).
- **Mode choice** recomputes after service or fare changes: station access, destination-specific
  common-line strategies, fare-sensitive choice between rail and driving, and three crowding
  rounds that price crowded trains and riders left behind by full ones [T-188]. Each round uses
  the combined loads of every zone, the everyday layer and the regional market. A distance-based
  walking share is reserved first, so useful rail never removes a short trip's walk.
- **Walks and drives follow real paths** [T-275] round water and over OSM roads. Riders walk
  between stations within 300 m after a ride, and between any two stations of a complex [T-161].
- **Weekdays and weekends** [T-260]: days 6 and 7 of each week are the weekend, worked out from the
  weekday's solve per purpose and period (NHTS 2017 rates). The same trains run every day;
  **open** (T-261): whether weekends get their own.
- **Demand rules must hold up worldwide**: every demand rule has an entry in
  [docs/demand-worldwide.md](docs/demand-worldwide.md).

Detail: [docs/demand.md](docs/demand.md).

## 5. Demand: the regional layer

The regional layer carries what crosses zones or goes far [T-117, T-276]. Detail:
[docs/regional.md](docs/regional.md).

- **5.1 The market**: total all-mode trips first, then the network earns rail's share.
  **Inter-region trips** (visits, business and leisure of 25 km and more): 0.107 departures per
  resident a weekday among every zone and an outside world (Kontur population in no zone), fitted
  to NHTS 2017, against a car that pays for time, running cost and parking. **Commutes between
  zones**: the joint gravity's cross-pack part, at commute timing and mode choice. Figures needing
  every zone pair are computed offline into the pack's `derived` block; the game builds pairs only
  between zones a station serves [T-174].
- **5.2 Access**: the same fast-walk station access as local demand, so feeder lines and transfers
  carry real loads; nothing bypasses the player's local network.
- **5.3 Park-and-ride**: planned (T-295).
- **5.4 Size**: the market pack runs in the same worker pool as the zone packs.

## 6. Building and running

The player draws curved track (straights and circular arcs through the corners clicked, single,
double or quad, at levels -3 to +3), places stations, and makes lines, all as **blueprints** that
are free to edit and undo, then pays to **construct** them. A drawn route is one piece of track per
corner [T-244]. Lines set a whole number of trains for high, medium and low demand; capacity and
junction conflicts add delay, computed from the schedules in the clock worker before demand runs.
The clock worker owns edits (with undo for reversible changes), capacity, fleet, costs, money and
the boarding account. Fares are paid into the cash as trains pick up riders and running costs
drain evenly [T-284]; the economy runs at 100x real days [T-081]. Changing a line's trains moves no
running train: added ones enter at a terminal, removed ones leave at one [T-206]. A new game starts
with $6B in an empty New York.

Detail: [docs/track-and-service.md](docs/track-and-service.md) (6.1 track and stations, 6.2
capacity, 6.3 trains and schedules, 6.4 costs and money, 6.5 edits, 6.6 the boarding account).

## 7. Technology

Web first, so it can be shared as a link. TypeScript, Vite and Preact with signals for the app;
MapLibre GL 6 for the basemap, with our own WebGL2 renderer on a canvas over it for track,
stations and trains, drawn in the same frame as MapLibre's. Rust compiled to WebAssembly in Web
Workers: one clock worker (track model, lines, money, boarding account, autosave) and a pool of
demand workers that each own a fixed set of the five periods. Threads talk by transferring
versioned snapshots. A loaded game holds only its clock until the first demand estimate lands;
input is never blocked. City packs come from an offline Python pipeline. The simulation is a pure
function of (save, pack versions, game time), bar the hour or two after a service edit [T-206].
Detail: [docs/architecture.md](docs/architecture.md) 7; file layout in `docs/code-map.md`.

## 8. Look and UI

A light pastel style in Zen Maru Gothic with three colour themes, for the UI only; the network takes
the line colours. One dock on the left holds the icon tabs (Build, Lines, Stations, Money), the
saved games button and the Settings gear, with the inspector under them, whose top edge never
moves. The top bar holds the mode split (the goal number), the cash, the local day and time, the
demand level, pause and three speeds. **Panels show what can change**: labels, numbers and states,
no explanatory notes [T-253]. **Everything drawn that encodes a number has a hover**, always the
game's own tag, never a browser tooltip [T-194, T-210]. No CSS transitions; trains are the only
motion. **Panels count today, not a forecast** [T-293]: the line and station panels count the day
so far from the boarding account, while the map's demand views (Homes and Destinations bubbles
coloured by mode share, line widths by riders, station circles, inter-region trips) show demand's
whole day, the clock's day by default. On the map, lines sharing track are drawn over each other,
stations are dots from zoom 11 and their true platforms from 15, and trains fill like a gauge with
their riders.

## 9. Checking the demand model

Rough on purpose: the bar is the right order (section 1).

- **Real-network test** [T-007, notes/T-007.md]: New York's real rail network (867 stations, 109
  lines, from the operators' GTFS) solved natively through the app's `DemandApi`
  (`pipeline/realnet.py`), against ACS rail commute share by county, MTA subway entries by period,
  the busiest subway links and operator totals. Latest (with routed walks and drives,
  notes/T-275.md): rail 19.2% of commutes against ACS's 20.3%, county error 2.5 points. Known gaps
  are open tasks (PATH low, T-270; commuter rail high, T-271, T-272; subway peak loads, T-216).
- **Tier-2 check** against LODES OD: commute lengths and county pairs (notes/T-198.md).
- To do: mode share per city against published figures; long-distance rail share on corridors
  with known rail and air numbers (Northeast Corridor first).

## 10. Milestones

- **M0, spikes** and **M1, New York sandbox**: done.
- **M2, Northeast** (under way): nine cities, the regional market, zones and commutes between them
  are in; park-and-ride remains.
- **M3, world**: packs for many metros at tiers 0 and 1, world view, per-city dirty flags under
  load. The sampler of other countries' cities starts it.
- **M4**: POIs, the look pass, sharing saves by link, accounts.

## 11. Saves and accounts

A save is the player's inputs only; everything else is recomputed from versioned packs. The clock
worker writes it to IndexedDB in named browser slots (one writing tab per slot) and to files.
Keeping old saves loading is a nicety: only Anita has saves. Sharing by link and accounts (on
`shonei-server`) come later. [docs/architecture.md](docs/architecture.md) 11.

## 12. Open questions

- **Weekend service** (T-261, section 4): should weekends get their own trains per hour (or run a
  lower level), at the cost of a weekend solve?
- Open items in the detailed docs: border factor fade (T-273), how commutes between zones are
  labelled (T-298), express and quad track design (T-257, on hold).
