# T-3444 design, part 6: demand and station reach

Agent tdsim, 2026-11-21. Design draft for discussion with Anita; nothing is built. Vocabulary as in `01-model.md`;
stations as in `02-station.md`; the sim side in `05-sim.md`. Demand rules: docs/demand-kernel.md (the 4.4 kernel,
"A change of trains", "Walks between platforms", complexes), demand-mode.md 4.3.

## The short version

Demand never sees track, and it will not start to. It sees a list of points riders walk to, lines as stops in order
with hop minutes and trains an hour, and a few per-stop and per-station extras. The rework changes how the sim works
those arrays out, not what they hold. On any save that uses nothing new, every array the workers get must be the same
byte for byte, so no figure moves. Nothing in this part adds a demand factor; the one candidate (a cross-platform
change benefit) is a question for her at the end.

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

The choice that decides parity is what one snapshot station is on the new model.

| Unit | Rows on an old save | Parity |
|---|---|---|
| A station (02's B plus C plus join) | fewer: the loader makes one station of old twins, of old per-direction pairs, and, on NYC, maybe of imported lines 6 m apart | breaks: fewer points, new means, merged catchments |
| A stop point (one per track) | more: two per double station | breaks: points move ±2 m, platform walks appear |
| **A stop node** (01's derived cross-section: the tracks of one bundle served by platforms with the same middle) | one per old station node | holds |

**Proposed: one snapshot row per stop node of a constructed station that a line uses.** On every old save each old
station node becomes exactly one stop node (a double's station: one node over both tracks; a one-track station: one
node, one served track; a quad pair's station: one node over that pair's tracks; her split legs: one node per leg).
The station and the complex reach demand only through `complex` (section 4).

**Its point.** The stop node lies on the bundle's alignment, but an old quad pair's station node lay on that pair's
centre line, 4 m off the bundle's (3 m on NYC's 6 m spacing). So the row's point is the alignment point at the stop
chainage, offset sideways to the centre of the tracks it serves **together with their partners** (05-sim section 6).
That gives the old node exactly in every old case:

| Case | Served tracks and partners | Centre | Old node |
|---|---|---|---|
| Station on double | -2, +2 | 0 | 0 |
| One-track station on double | +2 and its partner -2 | 0 | 0 (on the double's centre line) |
| A pair of a bundled quad or of two NYC lines bundled | -6, -2 (or -5, -1) | -4 (or -3) | the pair's centre line |
| Two-way single track | 0, its own partner | 0 | 0 |
| A one-way leg (own bundle) | 0 | 0 | 0 |

New layouts take the same rule: an AA/BB station of four platforms is one row at the centre; an island station with
its spread is one row at the centre of its two tracks. The loader must keep each old double's tracks at their old
offsets (not redraw NYC's 6 m spacing as quad's default 8 m), or the points move.

**Its id.** Stop nodes are derived and their ids may change on an edit inside the bundle, but demand's local solves,
clusters and per-station figures key on the snapshot station's id. So a row's id is authored: the lowest platform id
among the stop node's platforms. It changes only when that platform goes, which is a real change to the station.

**Order.** `merge_stations` takes means in index order and numbers demand stations by their first member, so rows go
in platform id order, and the loader gives platform ids in old station node order. A float mean summed in another
order differs in its last bit, which is enough to move a figure.

**Two conditions on the loader** for "one stop node per old node": it must keep each old station's platform middle
where the old node was, never snapping two old twin stations to one middle (01 gives "same middle" no tolerance, so
two old nodes a few metres apart along a bundle stay two stop nodes); and partners stay inside a bundle, so an old
one-way single (its own bundle of one) has none and its point is its own track.

**Level, upgrades, fare zone**: the station's, at the stop node's bundle level (stairs as today, `gradeStops`).

## 3. Lines: stops, back stops and patterns

- `stops`: the snapshot rows of run 0's stop points by stop index (05-sim section 8). On an old save these are the
  old stop nodes, so the list is today's.
- `back` / `back_stops`: run 1's row per stop where it differs from run 0's. On a double or a one-bundle quad both
  runs stop at one stop node, so `back` is empty, as today. Her split legs stop at one stop node per leg, so `back`
  carries the north leg exactly as T-3348's `line_back_stops` does now. `back_stops` keeps its meaning and its arrays.
- **Patterns** (a line on several tracks, 05-sim section 7): each pattern is packed as its own line with its share of
  the trains, as the hidden twin is today. Riders see the combined frequency through the stations' merge, as now, so
  a two-pairs save gives today's figures and demand needs no new idea of "one line, several stop sets".
- **Exact stops** (stage 11): a line whose two runs visit the same stations in reverse order packs as today, a stop
  one run skips carrying `span` -1 for that run (the pass marker T-3332 and T-3565 use). A line whose two runs share no
  common order (out by one route, back by another) packs as two one-way lines (run 1 at no trains), since demand's
  line is one stop list both ways. The cost: the rule "never change onto the line you left towards where you came
  from" does not see the two as one line. Nothing moves for lines without exact stops.

## 4. Merging, stations and complexes

- **The 100 m merge stays geometric** and stays the only merge. A station's stop nodes are not merged because they
  belong to one station: her split legs 150 m apart stay two demand stations (with the walk between them as today),
  even after she joins them into one station. Merging by station would move those figures, and give a long
  joined station one catchment point in the middle.
- **Every stop node of a station shares its complex**, so a station's platforms are always a change on foot apart.
  This is the one thing station membership adds, and it adds nothing on old saves (an old pair the loader joins was
  in one complex already, or within T-3348's 300 m change walk).
- **The complex rule is written once, in the sim** (stage 4, 01's test 2 row). Demand then gets `complex` from the sim
  instead of `game/complexes.ts`. The rule reads platform records the way `complexes.ts` reads drawn platform lines
  today (T-1859: 50 m between platforms, 100 m between centres, links, buildings). The partition must come out the
  same on old saves; the numbering need not, since `demand_groups` renumbers by first demand station.

## 5. Changes within a station

Today every change inside a demand station pays 2 fixed minutes plus the wait plus the walk between the two stops'
points. Two lines at one stop node walk nothing: a cross-platform change and a change across the tracks between side
platforms cost the same (T-3348 noted this: "cross-platform changes are short already, but no shorter than any other
change within a station").

Because `stop_x/y` is the stop node's point (section 2), not the track or the platform, the new layouts keep that:
- AA/BB with islands: local to express across the island walks nothing. So does changing to the other direction
  across the tracks, which riders rarely want (the "line you left" rules).
- Offset platforms joined into one station, stacked AA above BB: two stop nodes, so the walk between their points,
  as two separate station nodes walk today. Level changes inside a station cost nothing extra, as today.
- An island 6 m wide or side platforms 8 m apart would differ by a few seconds of walk if `stop_x/y` were the
  platform's own point. That would be a cross-platform factor brought in by accident; it stays out unless she asks.

**A cross-platform benefit**, if she wants one (her notes: "perhaps this can give a slight transfer benefit for
cross platform transfers but yea thats mostly flavor"). The smallest form: a change between two stop points on the
same platform record (the two edges of one island) pays 1 fixed minute instead of 2, as a station building's changes
do (`BUILDING_CHANGE`). It needs one number per stop (its platform id) in the snapshot and one test in the kernel's
change cost; about 2-3 hours. It is a new demand factor, so it waits for her word. I would leave it out: the change
is already cheap, players see no number for it, and it would make the island a better station than side platforms
for reasons riders can't see on the map.

## 6. Walking to stations: stop points against station points

Access (homes and jobs to the network) walks to the **demand station's point**, the merged mean, as today; the walk
grid never sees platforms. Changes walk between **stop node points** (`stop_x/y`). Nothing walks to a stop point (a
track): at the scale of the walk grid a track 2 m off the stop node is the same place. A tapering platform (stage 10)
or a drawn one is still one stop node per bundle and middle, its point by the same rule.

## 7. What changes in the snapshot, stage by stage

| Stage | Demand side | Replaces | Check | Hours |
|---|---|---|---|---|
| 3 | none: stations are still nodes; the snapshot reads them as today | | the packed arrays equal on the NYC save | 0-1 |
| 4 | snapshot rows from stop nodes, the point rule, the order; `complex` from the sim's one rule; `line_back_stops` from runs' stop points | `complexes.ts` for demand, the way-station source of `backStops` | the packed `DemandNetwork` byte for byte on NYC and the small saves, then rail trips a day | 4-6 |
| 6 | patterns packed as lines | the twin rows | a two-pairs save's arrays equal | 1 |
| 11 | exact stops packed (passes, or two one-way lines) | | a line listed exactly as its simple stops gives the same arrays | 2-3 |
| (her call) | cross-platform change at 1 fixed minute | | kernel test; ceiling tests | 2-3 |

`packService` itself keeps its shape: `stXY`, `complex`, `stops`, `back`, `times`, `gradeStops` and the rest are
filled from new sources with the same meanings. The native tools (`bin/realsave.rs`, the calibration tools) pack a
network on their own path and carry no back stops (T-3348's note); they need the same row rule in stage 4 or they
drift from the game's figures.

**The parity check.** Stage 4 and every later stage: build the packed arrays from the real NYC save on master and on
the tree and compare them byte for byte (station order included). Only if they match, compare rail trips a day
(12,581,681 on 2026-11-16, T-3348) and the per-station boardings. Then the same on the small saves of 05-sim's parity
B (terminal, passing tracks, single-track stations, two pairs, split legs, one-way street pair, one-track station),
where demand must also match except for the stage 7 capacity rows 05-sim lists, which reach demand only through hop
times and rates.

**Should any figure move on a save that uses nothing new?** No. Every row of this part is built so that it doesn't:
one row per old station node at its old point in its old order, back stops as today, twins as lines, the complex
partition unchanged. The places where something could slip, and what holds each:

| Risk | Held by |
|---|---|
| The loader bundles NYC's imported lines (6 m apart, inside `twin_at`'s 4-15 m) into quads and joins their stations | rows per stop node, points from partners, old offsets kept |
| Means summed in another order | rows by platform id, given by the loader in old node order |
| The sim's complex rule differs from `complexes.ts` | partition compared on NYC in the stage 4 check |
| `stop_x/y` taken from tracks or platforms | the point rule; a test pins a double station's two stops to one point |
| Stage 7's real loops and passing tracks change hop times | listed in 05-sim; demand follows the sim's figures, nothing of its own |

## Her calls

1. Cross-platform changes stay an ordinary change (2 fixed minutes, the wait, no walk), with no new factor? (yes; if
   she wants the flavour, 1 fixed minute between the two edges of one island, about 2-3 hours)
2. Stations joined by the player stay separate points for riders where their platforms are over 100 m apart (her
   split legs 150 m apart keep two catchments and the walk between them), joining changing only the name and the
   complex? (yes)
