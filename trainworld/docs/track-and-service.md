[Back to the spec](../SPEC.md)

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
- **Reshaping blueprint track** (T-064, T-092, T-071): a selected stretch of blueprint track shows its PIs
  as squares on the map; dragging one previews the new shape with each curve's radius and speed
  (or why it cannot be built), letting go applies it as one undoable edit, Escape puts it back. The
  track inspector lists the corners with each curve's speed and a radius stepper (100 to 1,500 m,
  then auto), a level stepper and Remove for each corner. Add corner inserts an auto-radius
  point midway along the selected interval between an end and a corner, or two corners; the
  new square can then be dragged. Removing the last corner restores a straight stretch.
  Insertion, removal and per-corner levels each use the existing checked, undoable shape edit.
  The inspector can also put all corners on one level; its ends keep theirs, with ramps. A set
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
  both directions sharing it, with sections split at stations. A single-track station has one
  shared platform track; it does not automatically add a passing loop. (Anita, 2026-10-08; not NIMBY-style
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
- platform track: dwell + 60 s per stopping train, 40 s per passing one, turnaround + 60 s
  for a short-turn ending at a through station. A single-track platform shares both directions.
- dead-end terminus: (layover + 60 s) / constructed platform tracks (a two-track terminus
  with 3 min turnaround: 30/h). Unused blueprint extensions add no active platform capacity.
- double-track station reversal: 90 s per crossing move. Reversals share their arrival
  platform with same-direction traffic and cross opposite-direction through trains or reversals.
  Each move receives only the extra queue from crossing traffic, subtracting its own-move
  baseline; independent opposing through trains do not compete. No crossover conflict applies
  in a demand level with no reversing service.
- junction move, with the moves it crosses; flat crossing, all moves: 90 s

**Delay per train** (seconds), Kingman's queueing formula with a small variability term because
trains are timetabled:

    mix = 1 - sum over (line, direction) streams of (share of trains)²
    W   = 0.2 x mix x s x rho / (1 - rho)        rho <= 0.9; straight-line continuation above
        + 1800 x (1 - 1/rho)                     when rho > 1 (the queue builds for an hour)

One line alone has no queueing delay (its trains are evenly spaced) until it is over capacity. A
90 s junction shared by two equal lines: 27 s at 75%, 81 s at 90%, ~3 min at 100%, ~7 min at 110%.

**Settled in the clock worker before demand runs.** Whole-line train allocations derive their
frequencies from capacity-adjusted round trips with a bounded, damped solve on edits (T-104).
Delays and derived frequencies feed back until settled, with at most 80 iterations and continuous
delays on resources used by count schedules. Legacy fixed-frequency schedules retain one
capacity pass: delay lengthens their round trips and raises fleet needs without changing rate.
The underlying capacity pass is incremental: an edit
rebuilds only the resources around the edges, nodes and lines it touched and re-solves those and
the ones a rescheduled line uses, with the same result as rebuilding everything (T-050). In WASM
the historical fixed-frequency pass took 0.02-0.2 ms an edit at 2,000 route-km with 60 lines and 1,250 resources, and 0.2-1.2 ms
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
network and under the demand views' bubbles and catchment (T-099). Like station names, they hide
below zoom 12 and reserve space around each tag (48 px horizontally / 28 px vertically before
zoom 14, then 20 / 12 px). Larger delays win collisions, then higher utilisation (T-102).
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
  Empty and one-stop lines are valid inactive drafts with an empty path. Stop edits use the same
  turn-valid route search for every insertion position, choosing minimum total path length
  (append wins ties). Copying preserves stops, exact path, schedule, dwell, turnaround, cars,
  colour and custom badge as independent inputs; edits and removal are undoable.
- **Scheduling** (T-104, Anita): every line sets a whole number of trains on the entire line,
  both directions together, for **high, medium and low demand** (the column header is "Demand").
  High is the two peaks, medium midday and evening, low night. New lines start at 0 / 0 / 0;
  zero means no new departures. The panel shows the integer allocation and resulting headway.
  Frequency is derived from the capacity-adjusted round trip, with a bounded solve on edits;
  extending track or changing congestion adjusts the headway while preserving the allocation.
  Resources used by count schedules keep continuous delays so rounding cannot make the solve
  oscillate. Legacy saves preserve their original frequencies until a schedule is edited.
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
  An unused blueprint extension leaves a constructed line's endpoint stopping position, run
  time and capacity unchanged. Planned lines preview the full planned station geometry (T-108).
- **Removing constructed track refunds nothing**, and the game asks before doing it (T-096). Undo applies to blueprint edits, lines and
  schedules. Constructing, removing anything constructed, making a constructed junction flying and
  lengthening a constructed platform are final (the last two are paid at once): they clear the
  undo history, so undo never builds or unbuilds anything for free.
- **Trains** (T-028): a 20 m car costs US$2.5M. The network owns cars; the running lines need
  their busiest schedule's trains times their cars. Any edit that makes them need more than are
  owned (a construct that starts a line, a larger allocation, longer trains, or an undo) buys
  the difference with it, and is refused with nothing changed if construction plus trains is
  more than the cash. Cars are never sold: spares serve the next line that needs them.
- **Running cost** US$1.50 per car-km (trains an hour each way x line length x cars; set for play,
  a quarter of a real figure, T-091), and **fares**,
  are settled each game hour by the clock worker into the cash and a per-day ledger. Fare per ride
  = a base plus a rate per km of the ride, one curve for the network, set in the Money tab ($1.50
  + $0.10 a km to start); a ride that changes lines pays the base once. Fare income comes from
  demand's riders and passenger-km per period (`app/src/game/money.ts`). Changing the fare
  recalculates demand: higher prices can lose riders (T-067). Running costs can take the cash below zero; edits that cost nothing still
  work then. notes/T-028.md.
- **Starting money $6B**, sized in T-008 for one real first line and rechecked in T-028: a 20 km
  line at -2 with 18 stations is US$3.78B plus about US$0.43B of trains, leaving room for a short
  second line.
- **Economy at 100x** (Anita, 2026-10-09, as Subway Builder does; T-081): fares and running costs
  per game day count 100 times the real figures (`ECONOMY` in `sim/src/track/params.rs`, the one
  constant; running costs scale in the track model, fares in `app/src/game/money.ts`), so a game
  day earns and spends what a hundred real days would. The fare the player sets and the US$1.50 a
  car-km shown are the figures before that scaling; construction and train prices stay real. The
  Money tab says so. The historical T-028 priced example for the real New York network (T-007)
  earned about $1.71B a game
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
resources. Count schedules then settle their derived rates against changed round trips before
the worker sends service to demand (T-104). A city's demand is dirtied only by service
changes: a line's stops or frequency, a station-to-station time moving 5 s or more, or the fare
curve changing. Fare edits use the same debounce and supersede older solves. Drawing
track no line uses never wakes demand. Demand starts after edits pause (~0.5 s), and a newer edit
cancels a running solve. An edit is refused only for problems it adds (radius, ramps, crossings,
platforms, ground over water); problems already there, say from a newer water mask, do not block
edits nearby. Every edit returns its inverse, which is undo (for blueprint edits; 6.4). Blueprint
track wakes neither profiles nor demand until it is constructed, except a line planned over it,
which may show a preview. An edit and the profiles it changes
take 0.4-2.6 ms in WASM on a 2,000 km, 60-line network, and 1.5-2.5 ms at 10,000 km with 400
lines except on track many lines share (about 20 ms for an edge of 23 lines, nearly all of it
their profiles); track no line uses, under 0.1 ms (notes/T-050.md). A save holds
PIs (format TWT4 also keeps the T-079 original drawn points and optional whole-line allocations; TWT5
(T-125) is TWT4 in the Lambert conformal world frame, and the only difference is the magic;
TWT2 and TWT3 saves load),
nodes, stations and lines: ~100 KB compressed for 10,000 route-km on a synthetic network
(notes/T-040.md; T-008 estimated 170 KB with denser drawing).
