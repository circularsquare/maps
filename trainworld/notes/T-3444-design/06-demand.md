# T-3444 design, part 6: demand and station reach

Agent tdsim, 2026-11-21. Design for the rework; nothing is built. Vocabulary as in `01-model.md`; stations as in
`02-station.md`; the sim side in `05-sim.md`. Stage numbers are `10-build-plan.md`'s. Demand rules:
docs/demand-kernel.md (the 4.4 kernel, "A change of trains", "Walks between platforms", complexes), demand-mode.md 4.3.
Her answers: `answers-2026-11-21.md`.

## The short version

Demand never sees track, and it will not start to. It sees a list of points riders walk to, lines as stops in order
with hop minutes and trains an hour, and a few per-stop and per-station extras. The rework changes how the sim works
those arrays out, not what they hold. On any save that uses nothing new, every array the workers get is the same byte
for byte, so no figure moves. The one new demand factor is the island change (section 5, stage 9c), which she
approved as a later flavour.

## 1. What demand sees today

From the snapshot (`serviceFromSnapshot`, `packService` in `workers/demandClient.ts`) into `demand::Network`
(`sim/src/demand/network.rs`):

| Array | Holds | Made from today |
|---|---|---|
| `stXY` | one point per snapshot station | each constructed station node a line uses, at the node |
| `complex` -> `st_group` | complex number per station | `game/complexes.ts` (centres 100 m, platforms 50 m, links, buildings) |
| `stops`, `lineN` | each line's stops in order | `line_run_stops` |
| `back` -> `back_stops` | run 1's station per stop, where it differs | `line_back_stops` (T-3348, per-direction stops) |
| `stop_x/y` | each stop's platform: its snapshot station's point | the stop's station node |
| `times`, `hopKm`, `span`, `tph`, `cars` | hop minutes both ways, fare km, reach of service, rates, size | the runs |
| `ups`, `passages`, `gradeStops`, `fz` | upgrades, passages, stairs by level, fare zones | the station node |

Then the kernel merges snapshot stations within 100 m (`merge_stations`, single linkage, never two neighbours on one
line, the merged point the mean in index order) into demand stations, and `st_map` carries every per-station array
over. A change of trains costs 2 fixed perceived minutes, the wait, and the walk from the platform arrived at to the
platform of the train taken (`stop_x/y`; 1 fixed minute in a station building). A rider may walk 300 m, or any
distance within a complex, to another demand station.

## 2. The snapshot's unit: one row per stop node

What one snapshot station is on the new model decides parity.

| Unit | Rows on an old save | Parity |
|---|---|---|
| A station (02's placed together, overlapping, or joined) | fewer: old twins on quad become one station, and on NYC, imported lines 6 m apart may be bundled | breaks: fewer points, new means, merged catchments |
| A stop point (one per track) | more: two per double station | breaks: points move ±2 m, platform walks appear |
| **A stop node** (01's derived cross-section: the tracks of one bundle served by platforms with the same middle) | one per old station node | holds |

**One snapshot row per stop node of a constructed station that a line uses.** On every old save each old station node
becomes exactly one stop node (a double's station: one node over both tracks; a one-track station: one node, one
served track; a quad pair's station: one node over that pair's tracks; split legs: one node per leg). The station and
the complex reach demand only through `complex` (section 4).

**Its point** is the alignment point at the stop chainage, offset sideways to the centre of the tracks the stop node
serves **together with their partners** (05-sim section 6). An old quad pair's station node lay on that pair's centre
line, 4 m off the bundle's (3 m on NYC's 6 m spacing), and this rule gives the old node exactly in every old case:

| Case | Served tracks and partners | Centre | Old node |
|---|---|---|---|
| Station on double | -2, +2 | 0 | 0 |
| One-track station on double | +2 and its partner -2 | 0 | 0 (on the double's centre line) |
| A pair of a bundled 4 tracks, or of two NYC lines bundled | -6, -2 (or -5, -1) | -4 (or -3) | the pair's centre line |
| Two-way single track | 0, its own partner | 0 | 0 |
| A one-way leg (own bundle) | 0 | 0 | 0 |

New layouts take the same rule: a by-direction station of four platforms is one row at the centre; an island station
with its spread is one row at the centre of its two tracks.

**Its id** is authored, since stop nodes are derived and demand's local solves, clusters and per-station figures key
on the snapshot station's id: the lowest platform id among the stop node's platforms. It changes only when that
platform goes, which is a real change to the station.

**Order.** `merge_stations` takes means in index order and numbers demand stations by their first member, so rows go
in platform id order, and the loader gives platform ids in old station node order. A float mean summed in another
order differs in its last bit, which is enough to move a figure.

**What the loader must do** for "one stop node per old node": keep each old double's tracks at their old offsets (NYC's
6 m spacing stays 6 m); keep each old station's platform middle where the old node was, never snapping two old twin
stations to one middle (01 gives "same middle" no tolerance, so two old nodes a few metres apart stay two stop nodes);
keep partners inside a bundle, so an old one-way single (its own bundle) has none and its point is its own track.

**Level, upgrades, fare zone**: the station's, at the stop node's bundle level (stairs as today, `gradeStops`).

## 3. Lines: stops, back stops and patterns

- `stops`: the snapshot rows of run 0's stored stop points by stop index (05-sim section 8). On an old save these are
  the old stop nodes, so the list is today's.
- `back` / `back_stops`: run 1's row per stop where its stored stop point lies on another stop node. On a double or a
  one-bundle 4-track station both runs stop at one stop node, so `back` is empty, as today. Smart allocation's pick for
  a way the clicked station doesn't serve (the nearest station within 300 m its run passes, complex first) is exactly
  what `back_stops` already carries: split legs 150 m apart, one click, one row per leg. The old-save loader stores
  each stop's two points as today's per-direction rule derives them (stage 4b), so `back` is today's. `back_stops`
  keeps its meaning and its arrays.
- **Patterns** (a line on several tracks, 05-sim section 7): each pattern is packed as its own line with its share of
  the trains, as the hidden twin is today. Riders see the combined frequency through the stations' merge, as now, so
  a two-pairs save gives today's figures and demand needs no idea of "one line, several stop sets".
- **Exact stops** (stages 11a, 11b): a line whose two runs visit the same stations in reverse order packs as today, a
  stop one run skips carrying `span` -1 for that run (the pass marker T-3332 and T-3565 use). A line whose two runs
  share no common order (out by one route, back by another) packs as two one-way lines (run 1 at no trains), since
  demand's line is one stop list both ways. The cost: the rule "never change onto the line you left towards where you
  came from" does not see the two as one line. Nothing moves for lines without exact stops.

## 4. Merging, stations and complexes

- **The 100 m merge stays geometric** and stays the only merge. Station membership merges nothing: stations the
  player joins (joining is only for naming) stay separate demand stations where their platforms are more than 100 m
  apart, each with its own catchment and the walk between them, as today. Merging by station would move those figures
  and give a long joined station one catchment point in its middle.
- **Every stop node of a station shares its complex**, so a station's platforms are always a change on foot apart.
  Joining two stations within 300 m changes nothing riders pay (a walk within 300 m is open to them anyway); farther
  apart, it opens the walk as a passage or building does today.
- **The complex rule is written once, in the sim** (stage 4a). Demand gets `complex` from the sim instead of
  `game/complexes.ts`. The rule reads platform records the way `complexes.ts` reads drawn platform lines today
  (T-1859: 50 m between platforms, 100 m between centres, links, buildings). The partition must come out the same on
  old saves; the numbering need not, since `demand_groups` renumbers by first demand station.

## 5. Changes within a station

Every change inside a demand station pays 2 fixed minutes plus the wait plus the walk between the two stops' points.
Two lines at one stop node walk nothing. Because `stop_x/y` is the stop node's point, not the track's or the
platform's, the new layouts keep that:
- A by-direction station with islands: local to express across the island walks nothing; so does changing to the
  other direction across the tracks, which riders rarely want (the "line you left" rules).
- Offset platforms in one station, stacked AA above BB: two stop nodes, so the walk between their points, as two
  station nodes walk today. Level changes inside a station cost nothing extra, as today.
- Side platforms 8 m apart, or an island 6 m wide, cost no walk: taking `stop_x/y` from the platform's own point would
  bring a cross-platform factor in by accident.

**The island change** (stage 9c, decision 23): a change between two stop points served by the same platform record,
one on each of its edges (an island, of any width, the stage 4 zero-width Island template included), pays 1 fixed
minute instead of 2, as a station building's changes do (`BUILDING_CHANGE`, 0.5). Side platforms are two records, so a
change between them stays 2 minutes. Inside a station building an island change pays the building's 1 minute (the
two do not stack).
- **Snapshot**: one platform record id per route node (2 per stop, laid out as `span` is: run 0's then run 1's),
  0 where the platform has tracks on one edge only. Sent only where some line stops at an island, so every old save
  packs as today.
- **Kernel**: the fixed change minutes at a boarding are halved when the arrival's and the departure's route nodes
  carry the same non-zero id. The search pools departures without regard to the platform arrived at (`platform_walk`),
  so the saving is added the way the platform walk is: the expected saving over the pooled departures, by frequency.
  The trip explorer shows each route's own.
- **Checks**: a kernel test (two lines across one island against the same two on side platforms: 1 minute apart per
  change); her ceiling tests unchanged (no island in them); the reference saves' arrays equal (no islands).
- About 2-3 hours, demand side.

## 6. Walking to stations: stop points against station points

Access (homes and jobs to the network) walks to the **demand station's point**, the merged mean, as today; the walk
grid never sees platforms. Changes walk between **stop node points** (`stop_x/y`). Nothing walks to a stop point (a
track): at the walk grid's scale a track 2 m off its stop node is the same place. A tapering platform (stage 10) or a
drawn one is still one stop node per bundle and middle, its point by the same rule.

## 7. Stages, checks and hours (demand side)

| Stage | Demand side | Replaces | Check | Hours |
|---|---|---|---|---|
| 3a-3g | none: stations are still nodes; the snapshot reads them as today | | the packed arrays equal on gate A | 0-1 |
| 4c | snapshot rows from stop nodes, the point rule, ids and order; `complex` from the sim's one rule; `line_back_stops` from the runs' stored stop points | `complexes.ts` for demand, the way-station source of `backStops` | the packed `DemandNetwork` byte for byte on gate A and parity B, then rail trips a day | 4-6 |
| 6c | patterns packed as lines | the twin rows | a two-pairs game's arrays equal | 1 |
| 9c | the island change at 1 fixed minute | | kernel test; ceiling tests; reference saves equal | 2-3 |
| 11a, 11b | exact stops packed (passes, or two one-way lines) | | a line listed exactly as its simple stops gives the same arrays | 2-3 |

`packService` keeps its shape: `stXY`, `complex`, `stops`, `back`, `times`, `gradeStops` and the rest are filled from
new sources with the same meanings, plus the island ids from 9c. The native tools (`bin/realsave.rs`, the calibration
tools) pack a network on their own path and carry no back stops (T-3348's note); they take the same row rule in
stage 4c or they drift from the game's figures.

**The parity check.** From stage 4c on: build the packed arrays from the real NYC save (gate A) on master and on the
branch, compare them byte for byte (station order included), and only if they match compare rail trips a day
(12,581,681 on 2026-11-16, T-3348) and the per-station boardings. Then the same on the reference saves of parity B in
order (her Amsterdam "test" save first, then Vienna, then North America day 383), where demand must also match except
where 05-sim's stage 7 rows change hop times and rates (demand follows the sim's figures; nothing of its own moves).

**No figure moves on a save that uses nothing new.** Where something could slip, and what holds it:

| Risk | Held by |
|---|---|
| The loader bundles NYC's imported lines (6 m apart, inside `twin_at`'s 4-15 m) and makes one station of them | rows per stop node, points from partners, old offsets kept |
| Means summed in another order | rows by platform id, given by the loader in old node order |
| The sim's complex rule differs from `complexes.ts` | partition compared on gate A in the stage 4c check |
| Old per-direction stops picked differently | the loader stores today's derived picks (stage 4b) |
| `stop_x/y` taken from tracks or platforms | the point rule; a test pins a double station's two stops to one point |
| The island minute reaching old saves | old saves convert to side platforms only; no island ids are sent |
| Stage 7's real loops and passing tracks change hop times | listed in 05-sim; demand follows the sim |

## Nothing open for her
