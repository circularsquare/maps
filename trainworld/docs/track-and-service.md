[Back to the spec](../SPEC.md)

## 6. Building and running

Full design, numbers and sources: `notes/T-008.md`.

### 6.1 Track and stations

- **One track type**, anywhere: over roads at normal cost, over water dearer (6.4). No buildings to
  demolish, no yards, no depots. Ground is flat (no terrain in M1).
- **Geometry: straights and circular arcs**, tangent-continuous, no transition curves or splines
  (exact lengths, one speed per curve, parallel tracks stay arcs). **Track is stored as points of
  intersection (PIs)**: it runs straight between them and cuts the corner at each with a circular
  curve (Anita, 2026-10-09; T-092). A PI's radius is auto unless set: the largest that fits half of
  each adjacent leg (the whole leg next to an end), capped at what the top speed needs (1,796 m). A
  curve under the minimum radius is refused ("The curve is too tight"). A route starting or ending
  on existing track gets a lead-in PI on that track's heading, so it leaves like a turnout and its
  first curve starts at the node (it never runs on top of the main line). Track drawn while the
  track ran through the clicks (T-079, saves from that day) keeps its shape; the points it was drawn
  through are kept with it, and its first reshape makes them its PIs (auto radius) like on any other
  track.
- **Drawing through points** (T-418; the only way to draw since T-422, Anita 2026-10-15): click
  where the track starts (open ground, a track end, a node, or track: a new junction, built flat),
  then where it goes, at the level selected. The second click places the segment at once and later
  clicks never change it; Shift held chains the next segment from its end, Backspace takes the last
  segment of a chain back (one undo step each), Escape or Enter stops. While drawing, the ghost
  shows the segment with its curve's radius and speed limit, the curve's whole circle faintly, and
  the cursor the segment's length, average cost multiplier (level and water over its length),
  price and tightest curve, the level placed and the lowest speed limit (T-085, T-411), or why it
  cannot be built; a click on a refused segment does nothing.
- **The segment's shape**: from a track end, a node or track to open ground it is **one circular
  arc through the click, leaving along the track's heading** (a tangent arc; from a node or track,
  in the direction nearer the click). For a click d away at an angle a off the heading the radius
  is d / (2 sin a), the largest any single curve through the click can have, and the track arrives
  on heading 2a. Up to a quarter turn the edge has one auto-radius PI on the heading d / (2 cos a)
  from the start, so both legs are equal end legs. **A wider turn, up to 350 degrees** (a loop or
  hairpin past a half turn, T-442), is split into equal corners of at most 90 degrees, each PI
  r tan(turn / 2n) from the arc's tangent points, so every leg between two corners is shared
  exactly and the auto radius is the arc's at each; a level change ramps on the middle leg. **Where
  that arc would be wider than the 1,796 m cap** (T-471), the segment is the capped curve from the
  start, turning until it faces the click, then a straight to the click: one round curve of any
  turn, still through the click but arriving on its own heading rather than 2a. At r = 1,796 m the
  straight is nil, so the shape grows out of the arc without a jump. A segment from a free point is straight. A segment ending on track or a node joins
  it like a turnout, so it has the lead-in shape (a lead-in PI at each end). Refused: a curve under
  100 m (the ghost and circle red), a click within 5 degrees of straight behind the start ("turn
  back on itself": the circle grows without bound), and anything the edit refuses (water, crossing
  other track at its level, a ramp, 200 m per level, that does not fit). The preview gives the arc's
  circle (centre and radius, `TrackApi::route_arc`) and an edit the segment's end node (`route_end`)
  to chain from. Segment ends are plain joints. **Quad drawn through points** (T-423): a segment
  ending on open ground leaves each pair its own loose end, the express pair's 8 m to its side; a
  segment from a loose quad end (the other pair's end within 4-12 m beside it, level with it and
  facing the same way) carries both pairs on with no swing-out or junction: the drawn pair from the
  end clicked is the tangent arc, the other pair from the end beside it, on whichever side that
  is, concentric (so a curve on the inside needs 108 m on the drawn pair). A first segment from a
  free point is two straights side by side. From other track or a node the express pair swings
  out as below, which needs a straight, so that segment keeps the lead-in shape (curve then
  straight); one ending on track or a node has both pairs meet there.
- **One piece per segment** (T-244, Anita: deleting one stretch should not delete the whole
  path): each segment drawn is one edge (each pair's, on quad track), joined to the next at a
  plain node (two edge ends, no station). The route edit still accepts clicked corners (`pis`) and
  then makes one piece per corner, cut in the middle of the leg between two corners, the shape
  unchanged; the drawing tool no longer sends any. The delete tool, Construct in the track
  inspector and the track inspector itself act on one piece, and removing a piece leaves its
  neighbours with track ends where it was. The joints are not drawn and cost nothing; trains pass
  them with no speed limit or delay (a track section runs on through a plain node, 6.2). A station placed
  within half a platform of joints first joins the pieces there into one edge, joint after joint
  until the platform has room (a long platform over several short segments; T-418), the shape
  unchanged.
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
  branch sets the radii next to the split so its shape does not change; a split inside a curve
  gives the two new PIs the radius that ends each half at the node, the curve's own to under 1%,
  or goes to the curve's end when it is within a few metres of it, T-258). A joint between two
  pieces of a route (T-244: a plain node on the straight between the corners either side of it)
  moves with a corner dragged next to it, to the same fraction of the way between the corners, and
  the preview shows the neighbouring piece too, so a corner drags as it did when the route was one
  edge. Constructed track cannot be reshaped (remove it and draw it again). notes/T-064.md.
- **Moving blueprint nodes** (T-093): with the select tool, a track end, junction or station can be
  dragged while every edge meeting there is blueprint and its station is not constructed; the
  blueprint track there follows, previewed like a reshape (each curve's radius and speed, or why
  not), one undoable edit when let go, Escape cancels. The edges' PIs stay where they are, except
  that a node where two or more edge ends meet keeps its heading (6.2): each edge's first PI, which
  holds that heading, moves with the node, and an edge with no PI of its own to move gets a
  lead-in PI as in drawing. A track end's heading follows its edge. Lines over it keep their stops
  and paths; their times change. A constructed node, or one where constructed track meets, stays
  put (dragging it when it is selected says so; otherwise the map pans as usual). notes/T-093.md.
  **Several nodes together** (T-475, `move_nodes`, the Select tool's pick): each moves by the same
  offset; an edge between two of them shifts whole, PIs included; an edge to a node that stays
  follows the one-node rule above. All must be movable, or nothing moves.
- **Placing stations**: click on track (it is split there) or on a node. A station's name
  suggestions (T-534, `map/stationNames.ts`, worked out when asked and never stored) are:
  - **the cross street**: the nearest named street on the basemap within 400 m ("W 42nd St"), when
    the map is at zoom 13 or closer (further out the tiles hold only the bigger roads), skipping
    any street whose direction where it comes nearest is within 30° of the track's there (the
    street the line runs along, or one nearly parallel) and names in use;
  - **the neighbourhood**: the nearest OSM neighbourhood, quarter or suburb on the city's place
    list (T-065, `game/placeList.ts`) that is small: another place of its kind lies within 3.5 km
    of its label, and its OSM population, if given, is at most 200,000 (so never "Manhattan" or
    "Brooklyn"). It reaches 1.5 times as far from its label as that nearest fellow, at least 800 m;
  - **the city or town**: the OSM city, town, village or hamlet whose label is nearest relative to
    its reach (hamlet 3 km, village 4, town 5, city 6 km or 4.5 km times the square root of its
    population in millions, at most 20; New York's is 13 km).

  No street, neighbourhood or tile label ending in "station" ("Station", "Sta.", "Stn", any case:
  a stop's own label in OSM) is suggested; the next candidate takes its place. A city, town,
  village or hamlet keeps its name even then (Huntington Station is a place) [T-568].

  A new station takes the cross street, else the neighbourhood, else the nearest place of any kind
  on the place list (neighbourhoods within 2.5 km of their label point, towns 5 km), else a place
  on the loaded tiles, else "Station N". A place name in use gives way to another place at most
  800 m further, else the way the station lies from it ("North Williamsburg"), else a number
  ("Williamsburg 2"). The player can rename it or cycle through the suggestions
  ([ui-inspectors.md](ui-inspectors.md) "Renaming"). Before the click, the placement ghost
  ([ui-building.md](ui-building.md)) shows whether it would work: the clock worker tries the same `addStation`
  and rolls it back (`TrackApi.preview_station`, T-311), so the ghost and the click cannot
  disagree, cash included.
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
- **Quad track** (T-255): the third setting. The stroke draws the double track as usual and an
  express double track beside it, 8 m to the left of the drawn direction (centre to centre, so
  4 m between every pair of neighbouring tracks), from the same start node to the same end node:
  it leaves each end along the node's heading like a turnout and swings out over 150 m (less on a
  short leg), and between them runs parallel, each corner concentric with the drawn one (radius
  8 m smaller on the inside of the turn, 8 m larger outside; a drawn curve under 108 m is too
  tight for quad). Lines change pairs at the ends of the quad stretch, where the two pairs meet as
  a junction (a fork with nothing behind it if the end is new track), and at crossovers (below). Both pairs are
  ordinary double-track edges, cut into pieces per corner (T-244), priced, deleted and drawn like
  any track: quad costs two double tracks, plus junctions at ends on existing track. Drawn through
  points, the pairs end apart instead (above). A station
  placed on either pair (on track or on a node) gets a twin of the same name level with it on the
  other pair, in the same edit (on the other pair's loose end or joint when one is level with it,
  T-423): the local and express platforms, one complex and one label. If
  the twin cannot be built there, the station goes in alone. **Express copy** in the line
  inspector copies a line with every stop that has a twin (a station within 12 m) moved to it and
  the path found again, so the copy runs on the express pair wherever the local one runs on quad
  track, and opens stop editing on it; the player clicks off the stops it skips.
- **Crossovers** (T-338, design notes/T-334.md): the crossover tool joins a track to the track
  running beside it, with a click. It finds the nearest other track whose centre line is 6-15 m
  from the click and parallel there (quad's other pair, or a separate route drawn alongside), splits
  both 60 m either side of the click (or uses a plain joint within 8 m of that point), and joins
  the four new nodes with two diagonals in an X, one each way, so a train can change track in
  either direction. Each diagonal is an edge flagged as a crossover (saved as edge flag 32; a save
  without it has none): an S of two reverse curves leaving each end along the track like a turnout
  (PIs a quarter of the way in from each end), the track count of the narrower of the two, a
  50 km/h limit whatever its curves allow, and the price of one junction (0.25 km of track at its
  level, 2x over water). The four end nodes are not priced as junctions unless three other tracks
  meet there, and the flat crossing between the two diagonals is part of the price, so a crossover
  costs two junctions. Routing and capacity need nothing new: each end is a junction node whose
  moves the crossing rule (6.2) prices, the diagonals' own crossing is a flat crossing, and a line
  takes a crossover when it is the fastest way to its next stop. Refused (in the tool's tag and on
  the click) where nothing runs beside within 15 m, or where 60 m either way reaches a track end, a
  junction or a station. A crossover keeps its shape: no corners to drag, no level change, its end
  nodes cannot be moved or made a flyover, no station or junction on a diagonal. Deleting either
  diagonal deletes both (the four nodes stay as plain joints); constructing either builds both.
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
  other nodes along it; curved platforms are fine. Through a station the platform is centred on
  the node; at a dead end it runs from the node along the track and trains stop at its middle
  (`service.rs`). The map draws it there, at its true length and bent as the track is, from
  zoom 14, as a box whose width is fixed on the ground too (16 m, a quarter narrower than a plain line and its rim, the same for every station and mode; never under 7.5 px on screen), so its shape is the same at every zoom from 15 in
  ([ui-map.md](ui-map.md) "Stations"; T-189, T-204, T-239). Platform length caps train length (20 m cars).
  Platform tracks = the route's tracks. Riders change between stations up to 300 m apart on foot, and platforms within 100 m of each other count as one station (4.4). Stations whose platforms come within 50 m form a station complex, which a click on a station selects as one (T-249) and the demand circles can show as one (8, T-146); the player can link any two stations up to 1 km apart into one complex and unlink them (T-197, saved with the network as `Network::links`, not in the undo history). Riders change between any two stations of a complex on foot, however far apart (4.4, T-161).

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

- track section (a run with the same set of lines), per direction: `s = 70 s + v / 0.6 m/s²`
  (v the top speed reached on it), so at most 41 trains/h at 40 km/h, 34 at 80, 29 at 120, 25 at
  160, 16 at 320: the published maxima of the Victoria line, Chuo Rapid, Thameslink and the
  Tokaido Shinkansen (T-517)
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

**No resource carries more trains than its capacity** (T-517). Whole-line fleets are cut to fit:
on every resource but a station crossover (whose load counts through trains that never meet),
`rho` stops at 100%. Where the lines a resource carries want more, each line's rate is scaled down
in proportion to the rate its fleet would give, until the first resource it uses is full
(progressive filling, `Capacity::fill`). The trains that do not fit wait at the ends of the line
(the hand-over below sends them out of service), so a fleet past capacity buys no shorter headway
and no rider waits longer than at 100%. So the shortest headway a line can have is its fullest
resource's `s`, 1m47s on track at 80 km/h. A count level held this way counts as settled, so
an edit elsewhere leaves it to the bit. Legacy fixed-frequency schedules are not cut: past 100% they
keep the delay and fluid queue above.

**Settled in the clock worker before demand runs.** Whole-line train allocations derive their
frequencies from capacity-adjusted round trips with a bounded, damped solve on edits (T-104).
Delays and derived frequencies feed back until settled, with at most 80 iterations and continuous
delays on resources used by count schedules. Each level settles on its own (levels share no
capacity): a level whose fleets already match their counts within 0.0001 trains is not stepped,
so its rates and run times stay the same to the bit (T-415). Legacy fixed-frequency schedules retain one
capacity pass: delay lengthens their round trips and raises fleet needs without changing rate.
The underlying capacity pass is incremental: an edit
rebuilds only the resources around the edges, nodes and lines it touched and re-solves those and
the ones a rescheduled line uses, with the same result as rebuilding everything (T-050; timings
in notes/T-050.md). Each delay is baked into the line's run-time profile per demand level as a hold
at the resource's entry: a slower approach when the delay is shorter than what braking to a stop
and starting again there would cost, otherwise a stop in front of it. So trains visibly crawl or
queue at a tangle.

**Shown** on the map as markers on resources at 75-90% (busy), 90-100% (near full) and over 100%
or full with some line's trains waiting for room (overloaded, red), with the delay per train, for
the demand level in force; nothing shows below 75%.
A marker reads "+1:03" (each train's wait there) or, with no wait yet, "81%"; hovering it shows
a tooltip in plain words ("Junction at 104% of capacity", "31 trains an hour each way against a
limit of 34", for a full one "Trains beyond the limit wait at the ends of their lines", "Each train
waits 1m03s here at high demand", and for a junction "Click to see the junction and its flyover").
A line whose fleet is cut shows a callout under its schedule naming the full resource ("Track full
near Grand Central") and, per demand level, how many of its trains wait at the ends, with the
resource's limit in trains an hour on hover (`TrackApi.line_held`). Markers sit over the
network and under the demand views' bubbles (T-099). Like station names, they hide
below zoom 12 and reserve space around each tag (48 px horizontally / 28 px vertically before
zoom 14, then 20 / 12 px). Larger delays win collisions, then higher utilisation (T-102).
Markers never cover a station name: each takes the first spot around its resource that is clear
of names and earlier markers (above right, above left, below right, below left, right, left), and
hides only when none is (T-184, [ui-map.md](ui-map.md)).
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
  It never exceeds what the line's track can carry (6.2): trains beyond that wait at the ends.
  Resources used by count schedules keep continuous delays so rounding cannot make the solve
  oscillate. Legacy saves preserve their original frequencies until a schedule is edited.
- **Departures** (`Service::run_deps`): per period of the line's local time (T-309: its first
  stop's region's, `Lattice::shift`; a New York to Toronto line is on one offset, a London to
  Paris line would run on London's), the first departure at the period's start and
  then one a headway; a whole-line fleet's reverse departures are offset by the forward trip so
  each train turns straight back. When the pattern changes (a new period, or an edit) a new
  departure closer than half a headway after the old pattern's last one is left out, so trains
  never leave a terminal in pairs.
- **Trains join and leave at the ends** (T-206, notes/T-206.md). An edit that changes a running
  line's departures (its train count, a whole-line fleet's headway after track or congestion
  changes, its turnaround) takes effect at the game time it is made: departures before it keep the
  old pattern (an *era*, `Service::eras`), so every train already running carries on where it was;
  departures after it follow the new one. A train running when its profile changed is drawn from
  where it was (`Service::shifts`: at the same place, or at the same stop with the dwell it had
  left). At each terminal trains are handed over (`Service::hand_over`): a train that has finished
  its run and turnaround takes the next departure from there if it comes within that departure's
  headway (an hour at most), standing at the platform until then; a departure no train waits for
  is a train entering service there; a train that finds another already waiting, or no departure
  soon enough, leaves service at the end of its turnaround. So an added train enters at a
  terminal, a removed one finishes its trip and leaves at a terminal, and headways even out as
  trains turn; the same happens at every change of period. A train that had left service before
  a change never comes back. In steady service with a whole-line fleet every train is ready at
  exactly its next departure and nothing stands longer. Each drawn trip names the departure its
  train takes next (`Trip::next`, the fourth number per trip of `TrackApi.trips`, NaN when it
  leaves service), so a selected train is followed onto its next trip (T-245). Eras are not saved: they are dropped once
  no train of theirs can be drawn (an hour plus the longest trip and two headways after the
  change), and a loaded game starts without them. Deterministic: the drawn trips are a function
  of the service, its eras and the hour asked for, and every window agrees.
- **Changing a train count is quick** (T-229, notes/T-229.md). The stepper shows the new count at
  once (`game/world.ts` `pendingTrains`), and clicks made while an edit is in flight are folded
  into one edit sent when it lands. The worker sends a service-only state for it: no geometry
  moved, so it skips rebuilding the renderer's buffers and the main thread swaps in only the new
  run profiles (`renderer.setService`). On the real New York save an edit takes 36-48 ms in the
  worker and 6-10 ms on the main thread.
- **Trip times on the stops list**: each stop shows the time from the start of the line, next to
  the station (e.g. Grand Central 1m23s, Long Island City 2m59s), and the panel shows the whole
  round trip.

### 6.4 Costs and money

Money is present, not tight, as in NIMBY Rails: construction, train purchase, operating cost per
car-km, a player-set fare curve by distance. One price list worldwide. Track costs a **base of
US$45M per route-km of double track** (single track 0.6x; ramps at the dearer level) times a
multiplier per level (Anita, 2026-10-08; all construction prices halved 2026-10-16, T-467):

| Level | +3 | +2 | +1 | 0 | -1 | -2 | -3 |
|---|---|---|---|---|---|---|---|
| Multiplier | 1.8 | 1.3 | 0.8 | 0.3 | 1.0 | 1.1 | 1.2 |
| US$M per km | 81 | 58.5 | 36 | 13.5 | 45 | 49.5 | 54 |

- **Over or under water: 2x** the level's price. **Level 0 cannot be built over water.** Water
  comes from a mask in the pack (T-030).
- The Build tab shows only the multiplier and price of the level currently selected, not the
  whole table.
- Stations (200 m, two platform tracks): 5 at grade, 15/20/25 at +1/+2/+3, 30/50/70 at
  -1/-2/-3; scaled by `0.4 + 0.6 x length / 200`.
- Junction: 0.25 km of track at its level; a flyover adds 0.6 km one level away; flat
  crossing 0.1 km (cheap to build, dear in delay). A crossover: two junctions, all told (6.1,
  T-338); the blueprint breakdown counts it as two junction switches.
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
  **Disabled blueprint** (T-333): a blueprint edge or station can be switched off (`off` on
  `EdgeData` and `StationData`, an undoable edit, saved as spare flag bits so an old reader
  ignores it). Construct all and the blueprint quote skip it, and a junction or crossing is priced
  only when the pieces that are not disabled complete it; everything else treats it as ordinary
  blueprint (lines plan over it, it can be reshaped). Constructing a piece by hand, or the
  track a line needs, builds it and clears the state; splitting keeps it on both halves and
  joining only merges pieces in the same state.
- **Removing constructed track refunds nothing**, and the game asks before doing it (T-096). Undo applies to blueprint edits, lines and
  schedules. Constructing, removing anything constructed, making a constructed junction flying and
  lengthening a constructed platform are final (the last two are paid at once): they clear the
  undo history, so undo never builds or unbuilds anything for free.
- **Trains** (T-028): a 20 m car costs US$2.5M. The network owns cars; the running lines need
  their busiest schedule's trains times their cars. Any edit that makes them need more than are
  owned (a construct that starts a line, a larger allocation, longer trains, or an undo) buys
  the difference with it, and is refused with nothing changed if construction plus trains is
  more than the cash. Cars are never sold: spares serve the next line that needs them.
- **Running cost** US$2.25 per car-km (trains an hour each way x line length x cars; set for play,
  well under a real figure, T-091, T-485), and **fares**,
  are settled by the clock worker into the cash and a ledger per day (30 kept) and per hour (48
  kept, T-447) as the game runs, not once an hour (T-284). The ledgers hold one entry per day or
  hour, in order; a load merges any day or hour an older save lists twice. Money time never goes
  back: a clock set behind the time already settled adds nothing until it passes it, and a charge
  meanwhile lands on the latest day. Both threads time the clock on `performance.timeOrigin +
  performance.now()`, never `Date.now()`, which drifts from it (T-488). Each game hour's fares are paid as that hour's trains pick up riders (the boarding
  account's trains leaving platforms, 6.6, weighted by the riders each takes; evenly in an hour
  without one), its running cost drains evenly, and both are paid in full by the hour's end, so a
  day's totals are those of whole-hour settling. The worker settles and sends the cash and today's
  and this hour's ledger four times a wall second while the clock runs; the top bar shows the cash to the dollar,
  and only it and the Money tab's cash, Today column and History re-render. Fare per ride
  = a base plus a rate per km of the ride, one curve for the network, set in the Money tab ($1.50
  + $0.10 a km to start); a ride that changes lines pays the base once. Fare income comes from
  demand's riders and passenger-km per period (`app/src/game/money.ts`). Changing the fare
  recalculates demand: higher prices can lose riders (T-067). Running costs can take the cash below zero; edits that cost nothing still
  work then. notes/T-028.md.
- **Money by region** (T-531, `app/src/game/regionMoney.ts`): beside the network's, the worker
  keeps both ledgers per region, a region being a zone as `stationZone` places it (the Lines and
  Stations tabs' place), and "" for anything outside every zone, so the regions add up to the
  network. Settling divides each running line's hour: fares by the line's day of riders, the base
  part by the region of each boarding stop and the per-km part by passenger-km, each stretch
  between two stops half to each stop's region; running cost by route km, split the same way.
  A paid edit's construction goes to regions by the cost of the track it newly built in each
  (track midpoint's region; with no track, its new stations' regions, evenly); its new trains by the
  route km of the line the edit names, else as its construction. The main thread works out these
  shares (it has the zone boundaries) and sends them; a charge waits under "" until its shares
  arrive. Cash, the fleet and the fare curve stay network-wide. Saved in the header (`regions`);
  older saves start empty.
- **Train rides** (T-565): each day and hour entry of every ledger also counts the train rides
  taken, the same rides the fares are charged on (a line's riders a day less rides that change
  lines, as `money.ts` has them, not scaled by the economy). A period's rides are the line's by that
  period's share of passenger-km, settled with the fares (an hour's rides count as its fares are
  paid) and divided between regions by the region of each ride's boarding stop.
- **Starting money $3B** (T-485): the build price of one real first line, about a 20 km line at -2
  with 18 stations plus its trains, leaves room for a few more lines.
- **Economy at 100x** (Anita, 2026-10-09, as Subway Builder does; T-081): fares and running costs
  per game day count 100 times the real figures (`ECONOMY` in `sim/src/track/params.rs`, the one
  constant; running costs scale in the track model, fares in `app/src/game/money.ts`), so a game
  day earns and spends what a hundred real days would. The fare the player sets and the US$2.25 a
  car-km shown are the figures before that scaling; construction and train prices are not scaled. The
  Money tab says so in the hover of "A day at this schedule".

### 6.5 Edits

The clock worker owns the network. The main thread sends edit operations (each with an inverse,
for undo); the worker answers with the new state, then the edit's answer (or the reasons it was
refused). The alignment being drawn is previewed by the worker as a trial edit, applied, measured
and rolled back, so the preview knows everything an edit would refuse (0.4 ms a round trip). The
worker sends the whole network state after an edit, with the renderer's buffers already built;
the main thread uploads them. Both sides redo only what the edit changed (T-496): the worker keeps
each edge's, line's and station's piece of the buffers from the last build and reuses any whose
inputs are bit for bit the same, then lays the pieces end to end; on the main thread station names
keep their elements and measured sizes unless their text or look changed, capacity tags are rebuilt
only when one changed, complexes are built once for every user, and DOM markers read the overlay's
kept size instead of forcing a layout (docs/architecture.md). An edit dirties only what it touched:
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
(T-125) is TWT4 in the Northeast conic and TWT6 (T-306) TWT4 in the Mercator world frame, the
only difference being the magic; TWT2 and TWT3 saves load),
nodes, stations and lines: ~100 KB compressed for 10,000 route-km on a synthetic network
(notes/T-040.md).

### 6.6 Boarding: each train's load, each platform's crowd

The boarding account (T-164, `sim/src/track/boarding.rs`, `app/src/game/boarding.ts`, notes/T-164.md)
turns demand's riders per period into a load per train and a crowd per platform. It adds nothing
per person and does not change who travels: demand still decides that (SPEC 4).

- **Input**: per line, direction, stop and period, demand's boardings and alightings (transfers
  included), sent by the main thread with each demand result, and each period's busiest-hour
  factor (docs/demand-timing.md). One set per kind of day (T-260): a weekday's and a weekend's riders
  make two accounts of the same trains, and each game hour's trains and platforms read their own
  day's (`day_kind_at`; a kind with no riders yet reads the weekday's). A line's panel shows the
  day chosen in the top bar.
- **Arrivals**: a stop's boardings in a period arrive spread over it by a fixed shape, scaled so the
  busiest clock hour is the busiest-hour factor times the mean hour: a tent (rising to the middle,
  falling away) for the two peaks and midday, falling from its start for the evening, highest at
  both ends for the night. The period's total is kept exactly.
- **Each train, in the order it leaves each stop**: the line's share alighting there gets off
  (demand's alightings over the riders it has arriving, per period; everyone at the last stop), then
  it takes whoever is waiting up to its crush load (160 a car). The rest wait for the next train of
  the same line and direction. Lines sharing stops keep separate queues: demand has already split
  the riders between them by frequency (4.4's common lines), and a rider left behind does not
  switch to another line.
- **Local time** (T-309): a line's account runs on its own local time (`LineAccount::shift`): its
  hours, busiest hours, kind of day and "today" from its local midnight. The money settles each
  local time's lines by their own period and day (`localGroups` in the clock worker).
- **A day at a time**: worked out for a day of departures from 03:00, from empty platforms; a
  direction where anyone is still waiting when that day ends is run again from a day earlier, so
  the queue carries round. Departures are the drawn trips to the bit (`Service::run_deps`). While
  a line has eras (T-206) its day starts six hours before its latest change instead, so the
  account holds the departures actually drawn around the change: a train that has not entered
  service carries nobody, and trains that left before the change keep their loads. Nothing
  is saved: it follows from the service and the riders. Each line's inputs are hashed, so an edit
  works out again only the lines it changed; new riders redo every line, a few lines at a time
  between the clock worker's other messages (3 a step).
- **Each game hour** the worker sends, with the hour's trips, each trip's riders per stop-to-stop
  stretch and each platform's departures with the riders each left behind, its steady arrival
  rate, its pulses of riders changing lines (below), and the hour's riders left behind and longest
  wait. The renderer reads the train's load for the stretch it is drawn on (empty once at its last
  stop; packed to crush, drawn full with a red edge, T-207, T-218), and a platform's crowd at any
  moment is what the last train left plus who has arrived since, drawn as the platform filling up
  wherever platforms are drawn (zoom 14 and in, T-189, T-204, T-219, T-553; [ui-map.md](ui-map.md)).
- **Riders changing lines arrive with their train** (T-554, `Boarding::pulses`, notes/T-554.md):
  in the drawn crowd, not the account. Demand's transfers a day between platforms (the solve
  shown, every population; `platformTransfers`) give each platform the share of its boardings
  that change there from each other platform. Each clock hour, that share of the hour's arrivals
  is split over the feeding platform's trains leaving in the hour by the riders each has aboard
  arriving, and reaches the platform a transfer walk after the train leaves (demand's: the
  straight line between the two stations x 1.3 at 1.2 m/s; nothing at the same station). The
  rest of the hour's arrivals, and the share of a feeding platform whose trains bring none or
  under one rider each that hour, arrive steadily. Each platform's arrivals in the hour are the account's; the
  account's loads, departures and riders left behind do not read the pulses, so they and
  demand's crowding feedback stay one rule. The fares' timing within the hour (T-284) counts
  the pulses with the steady arrivals.
- **Room on a platform** (T-219, `platformRoom` in `game/boarding.ts`): what counts as a full
  platform for drawing. Each direction's riders wait on one side, 2.5 m of usable width (half a
  6 m island platform less the edge strip kept clear) along the platform's length, at 0.3 m² a
  rider: the roomy end of Fruin's queuing level of service E (0.2-0.3 m² a person, standing in
  contact with others; the TCQSM's platform scale), about a train's standing density at crush. A
  150 m platform holds 1,250 a side. Only the drawing uses it: the account still lets a crowd
  grow past it (nobody is turned away from a platform).
- **Cost** on the real New York save (109 lines, 3,294 platforms): new riders 40-60 ms of the clock
  worker in steps of at most about 5 ms; an hour's slice about 11 ms (3-4 of them
  without the pulses) and about 870 kB with its 48,000 pulses, transferred; drawing the crowds 0.2-0.5 ms each game second on the main thread;
  the transfer pairs 12,000, asked of the demand workers once per result shown, about 20 ms.

- **How full trains get** (T-207, notes/T-207.md). On the real New York save, of the trains leaving
  between 08:00 and 09:00 the median is at 48% of crush at its fullest stretch, the 90th
  percentile at 78%, and 23 of 878 reach crush; between 17:00 and 18:00, 70%, 98% and 71 of 836.
  A player's Manhattan trunk (18 stops Harlem to Fulton St, 14 ten-car trains at the peaks) is
  packed full on every peak train. Trains reach crush mostly in the busiest half hour: arrivals
  peak at 1.8 times the period's mean at its middle, against demand's 1.4 for the busiest hour.
  Demand's crowding penalty (docs/demand-kernel.md step 7) starts once seats run out, so its equilibrium
  usually sits between seated and crush loads.
- **What the panels show** (T-230): every train-fullness figure is the account's: the gauge on
  trains, the train tag and inspector, the line inspector's "Trains today" (`Boarding::line_summary`:
  fullest train, trains leaving full and when, riders left behind a day, longest wait), the Lines
  tab's train count and the schedule's crowding dots (riders' extra wait after full trains
  against their ride, `Boarding::crowding`, T-491) and the platforms. Demand's busiest-hour average train is not shown: on the real New York save 28 of the
  93 lines it put under 100% leave riders behind in the account (the L: 87%, yet 13 trains leave
  full 17:51-18:34 and 4,300 are left behind), because riders arrive fastest mid-peak.

Riders left behind feed back into demand (T-188): each crowding round runs this rule on the
period's own trains and adds the average extra wait at each line, direction and stop to that
boarding's cost (docs/demand-kernel.md step 7), so a line that leaves people behind loses riders to
other routes or the car. The account itself is not read back: the clock worker's day and demand's
periods give the same rule, but demand's is worked out inside the solve, so it settles with the
crowding rounds. There are no signals (SPEC 2.1).
