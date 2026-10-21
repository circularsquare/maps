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

When two goals conflict, the higher one wins. **Current focus (Anita): the whole world, with demand
realism at the core.** Performance is open work (2026-10-15): each item's impact is measured, and
those with high impact or that let the game scale to hundreds of cities come first.

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
same pipeline, `kind: "zone"`) and **dead space**, where no trip starts or ends. Only zones a
built station reaches join the demand solves; a zone viewed or beside the view that no station
reaches opens **bare** in one auxiliary worker, its figures, bubbles and clicks from a no-rail pass
that equals a solve's to the bit, so panning never re-solves [T-325]. Every zone loads for the view
on a pan [T-801]. Zones that share no rider
(Toronto and the Northeast, London, Paris) fall into separate **clusters**, each solved on its own
lines; an edit re-solves only the clusters it touched, the others keep their figures [T-326,
[docs/demand-runtime.md](docs/demand-runtime.md) "Clusters"].

- **City zones**: New York (with Mercer County since T-276), Philadelphia-Camden-Wilmington,
  Boston (with southern New Hampshire), Washington (with surrounding Maryland, Virginia and
  Jefferson County WV), Baltimore, Providence, New Haven, Hartford and Springfield, in one regional
  market [T-116, T-134, T-137]; **Toronto** (the Greater Toronto and Hamilton Area), the first
  outside the US, built from Statistics Canada by Canadian adapters with its own fitted commute
  curve [T-277]; **London** (Greater London and
  every ITL3 unit touching it, with Gatwick's and Medway's: 14.5M people), the first sampler city,
  built by tier 0 with England's own inputs in place of its proxies (BRES jobs, Census 2021
  population by LSOA, GIAS schools, a curve fitted on the 2011 census flows), on UTC+0 [T-288,
  T-360]; **Paris** (Ile-de-France, its 8 départements: 12.5M people), tier 0
  with INSEE's jobs by commune of work and a curve fitted on its 2022 commune flows, population
  scaled to each commune's legal population, on UTC+1 [T-289]; **Shanghai**
  (the municipality's 16 districts, Chongming included: 24.9M people), tier 0 with population
  and jobs scaled to each district's census totals (2020 population census, 2023 economic census)
  and the curve's length fitted to the published mean commute, place names in Latin script where
  OSM has them, on UTC+8 [T-291]; **Mumbai** (the metropolitan region as
  Mumbai's two districts and 15 whole talukas: 28.3M people), tier 0 with population scaled to
  2011 census sub-district totals grown to 2020 by district, half the jobs proxy on population
  (informal work), the census's no-travel workers at home, and both curve parameters fitted to
  the 2011 census commute distance bands, on UTC+5 [T-292]; **Lagos** (Lagos State's 20 LGAs and
  Ado-Odo/Ota, Ifo and Obafemi Owode in Ogun: 13.8M people), tier 0 from global sources alone
  (GHSL unscaled, half the jobs proxy on population, OSM land use capped at 1 km² an area and
  spread over it, Mumbai's curve), a data-poor test, on UTC+1 [T-459]; **São Paulo** (the 39
  municipalities of the metropolitan region: 20.7M people), tier 0 with the 2022 census by tract,
  jobs, work from home and both curve parameters from the Metro's 2017 origin-destination survey
  (workers by zone of work, home-to-job distances), Portuguese place names, on UTC-3 [T-458];
  **Berlin** (Berlin,
  Potsdam, the eight Landkreise touching Berlin, Brandenburg an der Havel and Frankfurt (Oder):
  5.5M people), tier 0 with the Zensus 2022 100 m grid scaled to each Gemeinde's population, the
  Länder's Pendlerrechnung for jobs by Gemeinde of work and a curve fitted on its flows, on UTC+1
  [T-290]; **Seoul** (the Seoul Capital Area: Seoul, Incheon less Ongjin's islands and Gyeonggi
  whole, 26.0M people), tier 0 with population scaled to each si-gun-gu's resident registration
  count (November 2020), the jobs proxy, the curve's length fitted to Statistics Korea's mean
  commute, on UTC+9 [T-457]; **Tokyo** (Greater Tokyo: Tokyo, Kanagawa, Saitama, Chiba and
  Ibaraki prefectures whole, Tokyo's Izu and Ogasawara islands left out: 39.8M people), the first stage of
  covering all of Japan, tier 0 with the 2020 census 500 m mesh population, the 2021 economic
  census 500 m mesh jobs scaled to the census's workers by municipality of work, a curve fitted on
  the census's municipality flows, the census's at-home workers plus a telework share, KSJ school
  points, and place names in romaji (OSM's, else made from the kana reading or a kanji reading
  dictionary), on UTC+9 [T-420, T-426, research/japan.md]; **Osaka** (Keihanshin: Osaka, Kyoto,
  Hyogo and Nara prefectures whole, 18.2M people) and **Nagoya** (Chukyo: Aichi, Gifu and Mie
  whole, 11.3M), with the same inputs and curve [T-431]; **Fukuoka** (the prefecture, 5.1M),
  **Sapporo** (Ishikari subprefecture and Otaru, 2.5M), **Sendai** (Miyagi, 2.3M) and
  **Hiroshima** (the prefecture, 2.8M), with the same inputs, each its own fitted curve
  (Sapporo Tokyo's) and the telework share of Japan outside the three big metros [T-432].
- **Non-city zones**: Worcester, Windham, New London and Litchfield Counties. Modelled space stays
  between city zones, with no ring round each city, for now (Anita, T-300), except in Japan, which
  is covered whole: the 31 prefectures outside the metro packs are a zone each (Okinawa included)
  and Hokkaido's other subprefectures 8 zones after the under-100k rule, 38 zones and 46.6M
  people with the metros' inputs and each zone's own fitted curve [T-420 stage 3,
  `pipeline/japan_zones.py`].
- **City additions** in the US are paused; a **sampler of other countries' cities** comes next
  (London, Paris, Shanghai, Mumbai, Lagos, São Paulo, Berlin, Seoul and all of Japan built: 67
  zones in all), each a low-fidelity first pass from general sources (`research/world-sampler.md`).
  **Cells** are H3 resolution 9 (about 0.1 km²); sparse ones merge [T-044, T-466, T-510]: a res-8
  parent under 2,500 trip ends a weekday becomes one cell, in busier parents touching children under
  1,250 each merge, and a cell under 1,250 left alone joins a touching merged cell of its zone
  across its parent's edge ([docs/demand-sources.md](docs/demand-sources.md) 4.1).

## 4. Demand: local trips

Each zone's local layer models seven purposes from the same national and global sources [T-066,
T-140, T-142, T-150, T-152, T-181] (the US sources are named here; elsewhere tier 0's equivalents,
[docs/demand-worldwide.md](docs/demand-worldwide.md)): **commutes** (a doubly constrained gravity
from LODES homes to jobs), **shopping and errands** and **social and leisure** outings (retail,
food and recreation jobs, population, and OSM parks for 8% of leisure, T-196; outside the US the
same sectors' jobs by small area where published, else each 2 km zone's jobs blended with OSM's
shops and leisure places, T-453; how many from each country's national travel survey, a worldwide
median where it has none, T-509; how far by one worldwide set of distance scales, errands 1.3 km,
social and school 1.4, medical 2.0, university 3.5, T-637), **school** (NCES public schools),
**university** (IPEDS), **airport** ground trips (FAA boardings less connecting passengers) and
**medical** visits (OSM facilities). Private schools are not modelled yet. **Homes leave out
institutions** (prisons, nursing homes and the like, T-234).

- **Commutes are one gravity over each group of zones that commute to each other** [T-276,
  T-463] (the Northeast; Japan's 45 packs), solved offline (`joint_gravity`); the cross-pack part
  travels in the regional layer. Each pack carries its own distance curve (US
  d^-0.8 exp(-d / 30 km), T-198) and border factor (US states 0.23, Washington 0.45; Japan's packs
  0.4; abroad, national borders only, T-181).
- **Mode choice** recomputes after service or fare changes: station access, destination-specific
  common-line strategies (a wait is half the combined headway, in plain minutes [T-585]; a change
  of trains costs a fixed 2 perceived minutes, its wait and the walk between the two platforms'
  centres [T-717]), fare-sensitive choice between rail and driving, and up to three crowding
  rounds that price crowded trains and riders left behind by full ones [T-188]; a period stops
  once a round moves its loads under 1% [T-434]. Riders choose among station pairs by how much
  slower an option is relative to the trip's length: the scale is 20 over the trip's rough length
  in perceived minutes (25 + 3 a straight km), in four distance bands, the regional layer's trips
  too [T-718]; past 60 minutes it softens further, times sqrt(60 / T) (T the car's perceived
  minutes between cities, the band's length in a city), so a long trip minds a minute less; a
  train counts as crowded only from 70% of its crush load [T-710]. Each round uses
  the combined loads of every zone, the everyday layer and the regional market. Walking the whole
  way is a third mode in the same logit [T-558]: 1.5 perceived minutes a minute at every length,
  a bonus of 10 minutes against the car, commutes walking
  between sampled cells rather than zone centres [T-587], so a good station
  near a short trip takes some walkers and a long walk loses to the train. The choice is
  0.10 a perceived minute with no rail constant (time and money alone), car ownership $1.50 a trip
  plus $0.10 a km and 7 minutes to park [T-636, T-637]. Past 60 perceived car-minutes a minute weighs less
  (0.10 x sqrt(60 / T)), in every layer [T-495, T-502].
- **Homes without a car** [T-511, T-513]: each cell carries the people in households with no car
  (census by tract, LSOA or prefecture, else the national rate); they choose between rail, walking
  and "other" (bus, taxi and bike as one mode, on the road at 12 km/h plus a wait), shown as
  driving.
- **Walks and drives follow real paths** [T-275] round water and over OSM roads. Riders walk
  between stations within 300 m after a ride, and between any two stations of a complex [T-161].
- **Homes and places are grouped for the choice**: the cells of a 2 km zone that share their
  nearest station and lie in the same 15-minute band of walking to it share one walk per station
  [T-090, T-720]. The trip explorer answers for its two cells' own walks, and its hovers give the
  grouped figure the map's bubbles count where it differs [T-720].
- **Weekdays and weekends** [T-260]: days 6 and 7 of each week are the weekend, worked out from the
  weekday's solve per purpose and period (NHTS 2017 rates). The demand views show the average day
  (5 weekdays and 2 weekend days over 7) unless a weekday or a weekend day is chosen [T-464]. The same trains run every day
  (Anita, T-261: for now; a Saturday of its own is T-262).
- **Demand rules must hold up worldwide**: every demand rule has an entry in
  [docs/demand-worldwide.md](docs/demand-worldwide.md).

Detail: [docs/demand.md](docs/demand.md).

## 5. Demand: the regional layer

The regional layer carries what crosses zones or goes far [T-117, T-276]. Detail:
[docs/regional.md](docs/regional.md).

- **5.1 The market**: total all-mode trips first, then the network earns rail's share. One
  **world market** with every zone in it [T-362, T-377]. **Inter-region trips** (visits, business
  and leisure of 25 km and more): a ground curve fitted to NHTS 2017 and a long-haul tail (T-376),
  both scaled by GDP per head and shared over the world field (Kontur people with country,
  income and landmass), so trips to places with no zone leave the map; a border factor between
  countries; zones under 250 km apart pair 6 km zone to 6 km zone, farther ones place to place.
  Rail competes with a car that pays for time, running cost and parking (only within a landmass)
  and, between places 250 km or more apart that both have airports, a plane (T-378); other pairs
  stay fixed. **Commutes between zones**: the joint gravity's cross-pack
  part, at commute timing and mode choice. Each zone's own pack holds its part; a registry row per
  zone gives every pair of zones its trips with nothing opened; the game builds pairs only between
  zones a station serves on one track network [T-174, T-409].
- **5.2 Access**: the same walks and rides to stations as local demand, so feeder lines and
  transfers carry real loads; nothing bypasses the player's local network.
- **5.3 Park-and-ride** [T-295, T-371, T-389, T-373, T-372]: a commute or everyday trip of 5 km or more
  may ride (car, bus, bike or a lift, one mode) from home to the cheapest station of each nearby
  line within 15 km, at a fixed cost plus the drive plus parking or a drop-off at the station; the
  walk is a true 6 km/h walk. Fitted on New York (T-372): 53-66% of commuter rail riders ride to
  their station; a lone station in a small city keeps nearly all its corridor riders without
  feeder lines, and New York's commuter rail without the subway keeps 29% of the city's rail
  commutes. The car pays parking by the destination's job density with a free-parking share, and
  congestion charges in Manhattan and central London (T-380, docs/demand-mode.md 4.3). The regional layer's commutes between zones ride at the home end and
  its inter-region trips at both ends, the far end a drop-off only, through the same
  `access_options`, so a lone station in a new city draws riders from beyond walking reach. A
  station's inspector splits the commuters who live nearby into those who walked and those who
  drove, and the trip explorer shows a ride access as a drive [T-374]; the split covers
  commutes only.
- **5.4 Size**: a zone's market pack opens and closes with the zone, in the same worker pool as
  the zone packs.

## 6. Building and running

The player draws curved track (straights and circular arcs, double or quad, at elevations
-3 to +3, "levels" in the code) two clicks per segment, each segment one curve through the click leaving along the track, up
to 350 degrees round [T-418, T-422, T-442], places stations, and makes lines, all as
**blueprints** that are free to edit and undo, then pays to **construct** them. A segment is one
piece of track [T-244]. Each station on quad track gets a crossover just past it, placed by the
game in the same edit where it fits: an X between the two pairs, 120 m long, 50 km/h, priced as two
junctions, so lines can change pairs between neighbouring quad stations [T-338, T-733]. Single
track is not offered (saves and imports keep theirs, which run as before) [T-733]. Each line has
a train length in cars, by default the most its shortest platform fits (20 m cars) and never more
[T-733]. Lines set a whole number of trains for high, medium and low demand, or a headway for each, run with the fewest trains that keep it, or at the track's limit the most that fit [T-625, T-679]; capacity and
junction conflicts add delay, computed from the schedules in the clock worker before demand runs,
and no track, platform or junction carries more trains than its capacity: a line's fleet past that
waits at its ends [T-517].
The clock worker owns edits (with undo for reversible changes), capacity, fleet, costs, money and
the boarding account. Fares are paid into the cash as trains pick up riders and running costs
drain evenly [T-284], each hour's share following when riders travel and trains run, fares
landing as rides end [T-736]; the economy runs at 100x real days [T-081]. Changing a line's trains moves no
running train: added ones enter at a terminal, removed ones leave at one [T-206]. A new game starts
with $6B in an empty New York [T-805]; double track costs a base of US$90M a km times a
multiplier per elevation (US$27M at grade), priced against cheap-country metros such as Korea's
[T-805], a 20 m car costs US$2.5M, bought as a line's fleet grows and kept in the depot for any line
when it shrinks [T-028, T-631], and running costs are US$3 a car-km. The fare is typed or stepped in the Money tab [T-626].
**Points** measure each city's network: what it earned in the past 24 game hours, recomputed
once a game hour, at 20 a rail trip (a rider's whole journey, changes included, counted in the
city it starts in) and 1 a passenger-km; not a running total, and they unlock nothing yet [T-684,
T-694]. A city whose train share reaches 10% earns a star in the Cities list,
kept once earned [T-685].

Detail: [docs/track-and-service.md](docs/track-and-service.md) (6.1 track and stations, 6.2
capacity, 6.3 trains and schedules, 6.4 costs and money, 6.5 edits, 6.6 the boarding account).

## 7. Technology

Web first, so it can be shared as a link. TypeScript, Vite and Preact with signals for the app;
MapLibre GL 6 for the basemap, with our own WebGL2 renderer on a canvas over it for track,
stations and trains, drawn in the same frame as MapLibre's. Rust compiled to WebAssembly in Web
Workers: one clock worker (track model, lines, money, boarding account, autosave) and a pool of
demand workers that each own a fixed set of the five periods. Threads talk by transferring
versioned snapshots. The page opens on a start menu (New game, then the saved games); the workers
start with the pick (T-677). A loaded game holds only its clock, behind a loading card, until its
fresh demand solve's final result lands (T-347; the last final result kept beside the save, T-328,
is drawn behind the card meanwhile); input is never blocked. City packs come from an offline Python pipeline. The simulation is a pure
function of (save, pack versions, game time), bar the hour or two after a service edit [T-206].
Detail: [docs/architecture.md](docs/architecture.md) 7; file layout in `docs/code-map.md`.

## 8. Look and UI

A pastel style in Zen Maru Gothic, light or dark (Settings, Theme) with three colours each [T-039].
Panels take CSS tokens (`ui/style.css`); the map's light and dark colours (basemap, line casing, map
ink) are one table, `game/shades.ts`, whose basemap palette is warm paper land, soft teal water and sage parks [T-676]. The network takes the line colours, the same in both themes. One dock on the left holds the icon tabs (Build, Lines, Stations, Money, Cities), the
saved games button and the Settings gear, with the inspector under them, whose top edge never
moves. The top bar holds the mode split (the goal number), the cash, the local day and time, the
demand level, pause and three speeds. **Panels show what can change**: labels, numbers and states,
no explanatory notes [T-253]. **Everything drawn that encodes a number has a hover**, always the
game's own tag, never a browser tooltip [T-194, T-210]. No CSS transitions; trains are the only
motion. **Panels count the past 24 hours, not a forecast** [T-293, T-446]: the line and station panels
count the past 24 hours from the boarding account, while the map's demand views (Homes and Destinations bubbles
coloured by mode share, line widths by riders, station use circles, inter-region trips) show demand's
whole day: the average day by default, or a weekday or a weekend day [T-464]. On the map, lines sharing track are drawn over each other,
stations are dots from zoom 9 and their true platforms from 14, and trains fill like a gauge with
their riders.

## 9. Checking the demand model

Rough on purpose: the bar is the right order (section 1).

- **Calibration on real networks** [T-007, T-509, notes/T-509.md]: a city's real rail network (the
  operators' GTFS, else OSM routes; `realnet.py net <city>`) is solved natively through the sim's
  demand code (the `calib` bin) and scored by `calib.py` against census rail shares by area, station
  entries and line counts where published, and NHTS's New York everyday rail shares. New York (at
  half weight), London, Paris, Seoul, Tokyo and Sendai are fitted jointly and Toronto is held out;
  the fit set the rail constants and the everyday walking of then, which T-558's walking constants
  hold until the deep calibration fits them (section 4). The bar is
  plausible: a moderate overshoot where real riders shun transit (New York) is accepted. Known gaps
  are open tasks (Tokyo's commute rail share, T-529; PATH low, T-270; commuter rail high, T-271;
  subway peak loads, T-216).
- **Tier-2 check** against LODES OD: commute lengths and county pairs (notes/T-198.md).
- To do: long-distance rail share on corridors with known rail and air numbers (Northeast
  Corridor first).

## 10. Milestones

- **M0, spikes** and **M1, New York sandbox**: done.
- **M2, Northeast**: nine cities, the regional market, zones, commutes between them and
  park-and-ride are in; water between packs for intercity track remains (T-358).
- **M3, world** (under way): packs for many metros at tiers 0 and 1, world view, per-city dirty
  flags under load. In so far: the sampler of other countries' cities, all of Japan, the world
  market with air, and clusters that re-solve only what an edit touched.
- **M4**: POIs, the look pass, sharing saves by link, accounts.

## 11. Saves and accounts

A save is the player's inputs only; everything else is recomputed from versioned packs. The clock
worker writes it to IndexedDB in named browser slots (one writing tab per slot) and to files. New
game, Load a file and restoring a backup each start a new slot; no slot is ever written over with
another game [T-743]. A game is normal or a sandbox (nothing refused for money, the cash hidden;
a normal game can become one once, never back), starts in a city or on the world map, and a sandbox
in one of seven cities can start from its real network [T-728, T-731; docs/ui-panels.md "Saved
games"]. Each slot keeps two rotating backups, about 5 wall minutes apart, restorable
as a new slot [T-744]. Beside each slot the page keeps its last final demand result, shown at once
on reload when the network, packs and build match [T-328].
Keeping old saves loading is a nicety: only Anita has saves. Sharing by link and accounts (on
`shonei-server`) come later. [docs/architecture.md](docs/architecture.md) 11.

## 12. Open questions

- Open items in the detailed docs: border factor fade (T-273), how commutes between zones are
  labelled (T-298).
