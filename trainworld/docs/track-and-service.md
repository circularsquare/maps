[Back to the spec](../SPEC.md)

## 6. Building and running

Full design, numbers and sources: `notes/T-008.md`.

### 6.1 Track and stations

- **One track type**, anywhere: over roads at normal cost, over water and rough ground dearer
  (6.4). No buildings to demolish, no yards, no depots. Ground is flat in shape: levels are heights
  above or below the local ground, and the terrain only prices track (6.4, T-1270).
- **Geometry: straights and circular arcs**, tangent-continuous, no transition curves or splines
  (exact lengths, one speed per curve, parallel tracks stay arcs). **Track is stored as points of
  intersection (PIs)**: it runs straight between them and cuts the corner at each with a circular
  curve (Anita, 2026-10-09; T-092). A PI's radius is auto unless set: the largest that fits half of
  each adjacent leg (the whole leg next to an end), capped at what the top speed needs (1,796 m). A
  curve under the minimum radius is refused ("The curve is too tight"). A route starting or ending
  on existing track gets a lead-in PI on that track's heading, so it leaves like a turnout and its
  first curve starts at the node (it never runs on top of the main line). Track in a save that
  keeps the points it was drawn through (T-079) keeps its shape; its first reshape makes those
  points its PIs (auto radius).
- **Drawing through points** (T-418, T-422; the only way to draw, Anita 2026-10-15): click
  where the track starts (open ground, a track end, a node, or track: a new junction, built flat),
  then where it goes, at the level selected. The second click places the segment at once and later
  clicks never change it; Shift held chains the next segment from its end, Backspace takes the last
  segment of a chain back (one undo step each), Escape or Enter stops. While drawing, the ghost
  shows the segment with its curve's radius and speed limit, the curve's whole circle faintly, and
  the cursor the segment's length, average cost multiplier (level, water and terrain over its length),
  price and tightest curve, the level placed and the lowest speed limit (T-085, T-411), the
  terrain classes it crosses where they are charged (T-1270), or why it
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
  that arc would be wider than the design radius** (T-471), the segment is the capped curve from the
  start, turning until it faces the click, then a straight to the click: one round curve of any
  turn, still through the click but arriving on its own heading rather than 2a. At r = the design
  radius the straight is nil, so the shape grows out of the arc without a jump. **The design
  radius** (T-817) follows the Build tab's target max speed: the smallest whole-metre radius for it at
  a_lat (below), 1,796 m at 160 km/h (the default and the auto cap), 2,806 at 200, 4,385 at 250,
  7,183 at 320 (`design_radius`, `TrackApi::set_design_speed`). It also sizes the lead-in curves of
  a drawn segment (`lead_in_capped`). Above 160, a corner the tool makes whose curve comes out
  wider than 1,796 m gets that radius set, since an auto radius is always capped at 1,796 m; so
  drawn track keeps its shape and existing track is untouched. A real line's snapped route is
  fitted with its auto radii capped at the design radius too, and its corners wider than 1,796 m
  get theirs set the same way (T-1290). Trains still run at most their top speed. A segment from a free point is straight. A segment ending on track or a node joins
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
  plain node (two edge ends, no station). A route edit given clicked corners (`pis`, as the
  real-network snap sends them, T-727; the drawing tool sends none) makes one piece per corner, cut
  in the middle of the leg between two corners, the shape unchanged. The delete tool, Construct in the track
  inspector and the track inspector itself act on one piece, and removing a piece leaves its
  neighbours with track ends where it was. The joints are not drawn and cost nothing; trains pass
  them with no speed limit or delay (a track section runs on through a plain node, 6.2). A station placed
  within half a platform of joints first joins the pieces there into one edge, joint after joint
  until the platform has room (a long platform over several short segments; T-418), the shape
  unchanged. A joint between two different levels stays as a corner on the joined edge, since a
  ramp is centred between corners; a cut through a leg whose ramp lies on a straight puts a corner
  at each end of the ramp on its side (T-622), so a station there does not move the ramp. Two
  constructed pieces join as two blueprint ones do (T-624): the joined edge stays constructed, with
  no charge and no refund, and lines, trains, levels and curve radii carry over unchanged. Joints that cannot be joined
  (a different track count, constructed beside blueprint, blueprint switched on beside off) stay, and a platform that
  reaches one is refused as "the end of a piece of track". A platform may not reach a crossing
  with track on the same level (a flat crossing, 6.2), within 1 m of its ends.
- **Reshaping blueprint track** (T-064, T-092, T-071): a selected stretch of blueprint track shows its PIs
  as squares on the map; dragging one previews the new shape with each curve's radius and speed
  (or why it cannot be built), letting go applies it as one undoable edit, Escape puts it back. The
  track inspector lists the corners with each curve's speed and a radius stepper (100 to 1,796 m,
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
  the preview shows the neighbouring piece too, so a corner drags as on a route of one edge.
  Constructed track cannot be reshaped (remove it and draw it again). notes/T-064.md.
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
- **Placing stations**: click on track (it is split there) or on a node. On open ground the station
  brings its own track [T-794, `TrackApi::add_free_station`]: a straight blueprint piece at the
  selected level, double or quad as the Build tab is set, the platform plus 10 m past each end
  long, with the station at its middle (on quad its twin on the express pair), one undoable edit
  priced as the track and station together. Its ends are ordinary loose ends, so later track joins
  them by the drawing rules. The edit's own checks refuse it as any track or station (water, a
  crossing between levels, track on top of track), and so does any flat crossing on the piece
  (`PlatformCrossing`): a crossing that close would sit at the platform's end. A station's name
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
  100 m: 38 km/h, 300 m: 65, 1,000 m: 119, 1,800 m: 160, 7,183 m: 320. **Minimum radius 100 m.**
  Each piece of track runs at the **lowest of its curve speed, its track's cap and the line's
  trains' top speed** (T-817): track caps at **160 km/h**, high-speed track at **320**, a crossover
  at 50; ordinary trains top out at 160, high-speed trains at 320 (6.3). Grades do not change speed.
- **Levels** -3 to +3, **8 m apart** (+1 viaduct, -1 cut and cover, -2 bored, -3 deep). **Max
  grade 4%**, so a ramp is **200 m per level**. Level is set per PI, at the level selected when it
  was clicked; a ramp is centred between vertices of different levels. Nodes and platforms sit at
  whole levels. Tracks cross freely
  where their heights differ by a level or more; at the same level they make a flat crossing; in
  between is invalid.
- **What the player builds**: a **route** drawn as one stroke, **double track by default** (4 m
  apart, trains keeping right everywhere, T-319). (Anita, 2026-10-08; not NIMBY-style individual
  tracks.) **Single track** stays in the track model but the Build tab does not offer it (T-733;
  it may come back for rural or regional lines): saves and imports that have it load and run.
  It is cheaper, both directions share it, with sections split at stations; a single-track
  station has one shared platform track and gets no passing loop.
- **Quad track** (T-255): the other setting. The stroke draws the double track as usual and an
  express double track beside it, 8 m to the left of the drawn direction (centre to centre, so
  4 m between every pair of neighbouring tracks), from the same start node to the same end node:
  it leaves each end along the node's heading like a turnout and swings out over 150 m (less on a
  short leg), and between them runs parallel, each corner concentric with the drawn one (radius
  8 m smaller on the inside of the turn, 8 m larger outside; a drawn curve under 108 m is too
  tight for quad; a corner whose concentric radius would overrun its legs, as the capped curve of a
  nearly straight segment can once the millimetre grid tilts its metre-long legs, takes the auto radius
  instead, within millimetres of concentric, T-708). Lines change pairs at the ends of the quad stretch, where the two pairs meet as
  a junction (a fork with nothing behind it if the end is new track), and at crossovers (below). Both pairs are
  ordinary double-track edges, cut into pieces per corner (T-244), priced, deleted and drawn like
  any track: quad costs two double tracks, plus junctions at ends on existing track. Drawn through
  points, the pairs end apart instead (above). A station
  placed on either pair (on track or on a node) gets a twin of the same name level with it on the
  other pair, in the same edit (on the other pair's loose end or joint when one is level with it,
  T-423): the local and express platforms, one complex and one label. If
  the twin cannot be built there, the station goes in alone.
- **Crossovers past quad stations** (T-733, `TrackApi::with_crossovers`): a station and its twin
  get one crossover (below) in the same edit and undo step, centred half the platform plus
  `QUAD_CROSSOVER_CLEAR` (50 m) plus 60 m from the station, so its nearest end is 50 m clear of
  the platform end (the platform can grow 100 m before meeting it). It goes off the station's first
  track end, else its second; where neither has room (the crossover's rules below, or a check the
  edit fails) the stations go in without one, and so they do when the cash covers them but not it.
  Full price, two junctions. So between any two neighbouring quad stations a line can change pairs
  once, either way. Removing the station leaves its crossover as track.
- **High-speed track** (T-817, design notes/T-817.md): the third setting, always double track
  (4 m apart; there is no high-speed quad, so four tracks are HSR and double drawn side by side).
  An edge flag (`EdgeData::hsr`, `TrackApi::set_route_hsr`), so a station placed on open ground
  with it set lays high-speed track too. It costs `HSR_COST` (1.5x) double track's price at the
  same level (6.4); stations, junctions and flyovers on it cost as on double track. It meets
  ordinary track at plain nodes like any two pieces; a split keeps the flag on both halves, and
  two pieces join under a platform only when both are high speed or both are not (`unjoin_near`).
  It cannot be doubled up (`HighSpeed`). Its cap is 320 km/h, which only a line's high-speed
  trains use; ordinary trains run on it at 160. The track inspector shows the limit.
- **Doubling up** (T-708, `TrackApi::double_up`): double track already placed, blueprint or
  built, becomes quad with one undoable edit that adds the express pair as blueprint (built track
  too: Construct pays for the pair as for any blueprint). It covers the run the selected piece is
  on: through plain joints and stations into more double track, stopping each way at a junction,
  a track end, single track, a crossover or track that has a pair beside it already. The pair runs
  8 m to the left of the selected piece's direction, or to the right where the left is refused
  (crossing other track, water), every corner concentric, one piece beside each piece with a joint
  beside each joint. Each station on the run gets its twin (same name and platform) on the joint
  beside it, and a crossover past it as a quad station does (above), in the same edit; the
  preview's price includes them. At a junction end the pair swings in to meet the junction (the T-255 taper), so lines
  can change pairs there; at any other end it has its own loose end beside the drawn pair's, or
  carries on from a loose end of the pair beside that is there already (quad drawn earlier). Refused
  on single track (`Tracks`), high-speed track (`HighSpeed`), a crossover, and track with a pair
  beside it (`Alongside`); a run stops at high-speed track as at single track.
- **Crossovers** (T-338, design notes/T-334.md): a crossover joins a track to the track running
  beside it. The game places them past quad stations (above); the player has no tool for them
  (T-733), and saves keep theirs. `TrackApi::add_crossover` finds the nearest other track whose centre line is 6-15 m
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
  takes a crossover when it is the fastest way to its next stop. Refused where nothing runs beside within 15 m, or where 60 m either way reaches a track end, a
  junction or a station. A crossover keeps its shape: no corners to drag, no level change, its end
  nodes cannot be moved or made a flyover, no station or junction on a diagonal. Deleting either
  diagonal deletes both (the four nodes stay as plain joints); constructing either builds both.
- **Junctions** (T-080): a new junction is built flat (the branch crosses the other direction's
  track at the same level: cheaper, trains may wait for each other); there is no setting while
  drawing. A flyover is bought from the junction inspector ([ui-building.md](ui-building.md)), which
  gives its price and the waiting it saves per train at the demand level in force: paid at once on
  a constructed junction, else added to the blueprint, where it can be removed again.
- **Branching from a station** (T-1155, `branch_end` in `wasm/draw.rs`): a station's node takes
  no junction (`StationPorts`) and no junction may sit in a platform, so a route starting or
  ending on a through station's node, or on track within any station's platform, leaves the track
  20 m (`BRANCH_CLEAR`) past the platform's end instead: a new flat junction there, on the side the
  route's next point lies (for its end, the side its previous point lies), the branch leaving it
  away from the station. A plain joint within 8 m of that point takes the junction; a track end
  before it (a station's own piece of track ends 10 m past the platform) is carried on from. A
  station at the end of its track is extended from its node, and from its platform when the route
  heads out past that end; drawn back the other way it branches 20 m past the platform's far end.
  Refused where a junction, another station's platform or a crossover is in the way
  (`BranchNoRoom`: "Track cannot branch off just past this platform, because a junction or another
  station is too close."), and on a ramp as any split is. The junction takes the track's level
  there and the branch ramps to the level selected. On quad track only the pair clicked branches,
  inside its crossover 50 m out; high-speed track stays high speed on both halves. It is part of
  the route edit, so the ghost, the click, the real-network snap and the edit API agree, in one
  undo step.
- **No track at ground level over water** (level 0 is invalid where the water mask says water);
  bridges and tunnels over or under water cost 2x (6.4). The mask is 25 m pixels in the city
  pack; water narrower than ~50 m (creeks, ditches, small ponds) is not in it, since a railway
  crosses those at grade (notes/T-030.md). Outside every pack's mask the world's coarse mask
  decides (T-1285, zones.md 6): the sea and large lakes in cells of 1/60 degree, so open sea between
  zones is water like water inside them. Ground level means within half a level (4 m) of the
  ground, so the low end of a ramp counts too: bridge and tunnel approaches start on land.
- **A bridge by itself** (T-656, `track/bridge.rs`): a segment drawn through points at level 0
  (every vertex level 0, no clicked corners) that crosses water is raised to level 1 over each
  crossing. The ramp up (200 m) ends 14.5 m before the water and the ramp down starts 14.5 m after
  it, so both lie on land; everything else stays level 0. Crossings less than 429 m apart share
  one bridge, since the ramp down and the ramp up would not fit between them. The shape is
  unchanged: a ramp is centred between two vertices of different levels, so each ramp gets a
  vertex either side of it (at least 100 m from its centre, nearer an existing vertex where that
  is one); on a straight that is a PI with no turn, on an arc the arc is cut into arcs of its own
  radius, one PI each, and every PI of the segment gets its radius set. A ramp that a curve or a
  vertex keeps off its place moves back along the segment in steps of 25 m, up to 400 m. Quad
  track bridges each pair on its own. One edit and one undo step; the preview (`preview_route`)
  and `add_route` share the code, so the ghost shows the bridge, its price and the elevation
  reached (`0 +1` in the tag). Another level is left as drawn. A route through clicked corners (the
  edit API's `pis`, as the real-network snap sends them, T-727) is bridged the same way when every
  corner is at ground level, the bridge's vertices cut into pieces with the corners [T-667].
  Through more than 16 vertices (a snapped real route) each stretch of water is bridged on its
  own, between the corners 4 km of chainage either side, those corners keeping the radii the
  whole route gives them (water read every 3.1 m, a quarter of the usual step, since each piece
  is checked with its own samples); where that finds no room the stretch is raised whole,
  ramping on the last leg before the water and the first after it that are long enough for a
  ramp, and with no such leg (water just past an end) the route is refused as below
  [T-727d, T-1187; `bridge::bridge_long`].
  A new end over water (a free click, not track already there) stays up at level 1, with no ramp
  on its side, so a segment may end or start mid-river with one ramp, on the land side; quad
  track's express pair, sharing that end, must stay up there too. A segment drawn at ground level
  from or to an end already at level 1 is bridged the same way: it stays up to the water, or comes
  down at once where there is room (400 m) to go up again before it [T-1269].
  With no room on land otherwise (water within 200 m of the segment's start or end, so of a
  station or junction it leaves, or one a curve leaves no vertex room for) the edit is refused as
  `BridgeNoRoom`: "There is not enough land for the bridge ramps, which are 200 m long."
- **Stations** are nodes on a route, platform length 60-400 m in 20 m steps, level and free of
  other nodes along it; curved platforms are fine. Through a station the platform is centred on
  the node; at a dead end it runs from the node along the track and trains stop at its middle
  (`service.rs`). Removing the piece on one side of a through station keeps the track to its
  platform's end (half the platform and 4 m past the node; both stations' when the piece runs
  between two), so the station stays where it was rather than turning into a dead end half a
  platform back (T-1161, `station_trim_ops`; where that split is refused, on a ramp say, the piece
  goes whole). The map draws it there, at its true length and bent as the track is, from
  zoom 14 ([ui-map.md](ui-map.md) "Stations"). Platform length caps train length (20 m cars).
  Platform tracks = the route's tracks. Riders change between stations up to 300 m apart on foot,
  and platforms within 100 m of each other count as one station (4.4). Stations whose platforms
  come within 50 m form a station complex, which a click on a station selects as one (T-249) and
  the demand circles can show as one (8, T-146); the player can join any two stations up to 1 km
  apart into one complex by a **passage** and remove it (T-197, T-1281; saved with the network as
  `Network::links`, not in the undo history; links from older saves are passages, never charged).
  Riders change between any two stations of a complex on foot, however far apart (4.4, T-161),
  through a passage along the straight line at 2 m/s.
- **Station upgrades** (T-1281, `track/upgrades.rs`, notes/T-1281.md), bought from the station
  panel's row of icons like a flyover: into the blueprint with a blueprint station (built with
  it, in the quote as "Station upgrades"), paid at once on a constructed one; removing one is
  final and refunds nothing. Saved as bits 4 to 6 of the station's node flags (old readers ignore
  them), on `StationData::upgrades`, so undo, renames and platform changes keep them. No upkeep.
  - **Terminal** (turnback sidings): a line turning there holds its platforms only for the dwell
    and turns in two sidings past the platform (6.2): 30 an hour at the default turnaround. The
    map draws the two sidings between the running tracks past each platform end with track
    beyond, from where the tracks part (zoom 14).
  - **Park and ride**: a ride to or from the station costs 10 perceived minutes less ([demand-mode.md](demand-mode.md),
    `kernel::ride_fixed`); a P after the station's name.
  - **Waiting room**: a wait for a train there counts at 0.8 of its minutes in route and mode
    choice (`kernel::wait_factor`, [demand-mode.md](demand-mode.md)).
  - **Passage**: the link above, priced US$10M plus US$90M a km of the straight line between the
    two stations (`upgrades::passage_price`); paid at once between constructed stations, else
    built and paid with the second of them (`Network::passages`, quote row "Passages").
- **Train length** (T-733, `TrackApi::set_line_cars`, `build_line` in `service.rs`): each line
  stores its cars, 0 for the most its shortest platform fits; a new line starts at the length last
  set on any line in the game (0 until one is set; the save keeps it, `newCars` in its header) [T-1094]; a set length
  runs as set, at most that cap. A platform later shortened below it caps the trains while the
  setting stays, so lengthening the platform brings the set length back. Riders a train holds
  (160 a car at crush), the cars the fleet needs and the running cost (US$4 a train-km plus US$2 a car-km) follow the cars
  run. One undoable edit, buying any cars it needs as other fleet growth does (6.4).

### 6.2 Capacity and junctions

**The graph.** Nodes (end, junction, station; the kind follows from how many edge ends meet
there) and edges (one alignment between two nodes: its PIs with radius and level, and its track
count). **Flat crossings are not nodes**: where two edges cross at the same level, the crossing is
found from their geometry, costs money and loads capacity, but splits nothing, so dragging a PI
across track never rewrites the graph or line paths. Each edge gives one directed track per
direction (shared on single track): `(edge, direction)` is the segment in SPEC 2.1's keyframes.
At a node, every edge end leaves along the node's heading or its reverse; a move is any pair of
ends on opposite sides. A line's path turns back (arrives and leaves on the same side) only at a
station it stops at or at a bare track end, one with no station (T-797, 6.3). In a **junction**, two moves conflict when their paths cross at the same
level (track positions in opposite order on the two sides); merges and diverges do not, as track
capacity counts them. A plain double-track Y has one crossing pair. A **flyover** (a flying
junction) removes it, for its price (6.1, 6.4). A **flat crossing** makes every move on one
route conflict with every move on the other.

**Resources and utilisation.** Everything shared has an occupancy `s` per train, and per period
`rho = sum(trains per hour x s) / 3600` from the scheduled frequencies:

- track section (a run with the same set of lines), per direction: `s = 70 s + v / 0.6 m/s²`
  (v the top speed reached on it), so at most 41 trains/h at 40 km/h, 34 at 80, 29 at 120, 25 at
  160, 16 at 320 (T-517)
- single-track section between passing points: run time + 60 s, counting both directions
- platform track: dwell + 60 s per stopping train, 40 s per passing one, turnaround + 60 s
  for a short-turn ending at a through station (15 an hour at the default 3 minutes). A
  single-track platform shares both directions.
- dead-end terminus: max(layover + 60 s, 200 s) / constructed platform tracks (a two-track
  terminus with 3 min turnaround: 30/h; never more than 18 an hour on one track or 36 on two,
  the Victoria line's best, however short the turnaround; `upgrades::pooled_turn_s`). Unused
  blueprint extensions add no active platform capacity.
- a station with a **terminal** (T-1281): a line turning there (ending, or reversing partway)
  holds its arrival and departure platforms for dwell + 60 s each, and the terminal's sidings
  for max(turnaround + 60 s, 200 s) / 2, pooled: 30 an hour at 3 minutes, 36 at 2. It still
  crosses the other track (the 90 s move below). A dead end keeps its pooled rule.
- double-track station reversal: 90 s per crossing move. Reversals share their arrival
  platform with same-direction traffic and cross opposite-direction through trains or reversals.
  Each move receives only the extra queue from crossing traffic, subtracting its own-move
  baseline; independent opposing through trains do not compete. No crossover conflict applies
  in a demand level with no reversing service.
- a line reversing partway (T-797) uses what a short turn at that station would: at a through
  station its arrival platform for turnaround + 60 s and the crossover above, at a dead end the
  pooled terminus, and at a bare track end the same pooled terminus over the stub's tracks, with one
  side for the limiter below. It is no junction move and no through train there, and no track section joins
  across it. On top, a stopgap limiter per station and arrival side (`REVERSAL_LIMIT_S`): 300 s per
  reversing train, so at most 12 reversals an hour a side; 120 s (30 a side) at a station with a
  terminal (`TERMINAL_REVERSAL_S`). Both directions of a line reversing at a
  station usually arrive on the same side, so one such line runs every 10 minutes each way at best;
  in its queueing delay a line's two directions count as one stream (one timetable), so a line
  alone there waits only once it is over the limit.
- a loop line (T-798): a circle's first stop is a stop on the way round (dwell + 60 s on its arrival
  platform, the move from its last path entry into its first a junction move, sections joined across
  it); the hold there to the headway is not counted. A balloon's first stop is a terminus as any
  line's end (short turn or pooled dead end), with no reversing limiter. The loop itself is plain
  track, so a balloon has no turnback limit (about 34 an hour at 80 km/h).
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
an edit elsewhere leaves it to the bit.

**Settled in the clock worker before demand runs.** Whole-line train allocations derive their
frequencies from capacity-adjusted round trips with a bounded, damped solve on edits (T-104).
Delays and derived frequencies feed back until settled, with at most 80 iterations and continuous
delays on resources used by count schedules. Each level settles on its own (levels share no
capacity): a level whose fleets already match their counts within 0.0001 trains is not stepped,
so its rates and run times stay the same to the bit (T-415). A line a save loads at a frequency
rather than trains takes the whole trains that frequency needs first (architecture.md 11).
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
limit of 34", what limits it and what helps, for a full one "The line's extra trains are held out
of service at its ends", "Each train waits 1m03s here at high demand", and for a junction "Click to
see the junction and its flyover"). Each resource has a kind the app words by (`Capacity::wire_kind`,
`game/capacityKinds.ts`): track, single track, platform, terminus, junction, flat crossing, a line
turning on a through station's platform (the line ends there and holds the platform for its
turnaround plus 60 s; helped by building a terminal there), track several lines run
over, a turn across the other track at a through station, the limiter on reversing partway
("Reversing at Nerul Village is full"), and a terminal's sidings (T-1281). A line whose fleet is cut shows an
alert under its schedule naming the full resource ("Turning on the platform at Nerul Village is
full") and, per demand level, how many of its trains are held out of service (they wait at the
ends of the line instead of running), with the resource's limit in trains an hour and what helps on
hover (`TrackApi.line_held`). Markers sit over the
network and under the demand views' bubbles (T-099). Like station names, they hide below zoom 12
and keep the names' padding; larger delays win collisions, then higher utilisation (T-102); they
never cover a station name ([ui-map.md](ui-map.md) "Station names and capacity tags", T-184).
The junction inspector (T-031, [ui-building.md](ui-building.md)) lists the junction's moves (from
one station or direction to another), with the lines on each, trains an hour at the level in
force, the wait per train, which moves each crosses, and the busiest crossing's utilisation. The
line panel's round-trip split, waiting at busy track among it: [ui-inspectors.md](ui-inspectors.md).

### 6.3 Trains, lines, schedules

- **Two train kinds, chosen per line** (T-817; Anita 2026-10-21): **ordinary** (top speed 160
  km/h) and **high speed** (320, on high-speed track only; 160 elsewhere). They differ by top speed
  and cost (6.4), never by urban vs intercity; a line's choice is saved with it (`LineData::fast`,
  old saves ordinary). Both accelerate at 1.0 m/s² to 60 km/h, 0.5 m/s² to 200 and 0.3 m/s² above
  (a power-limited EMU, kept as constant rates so motion stays exact; only high-speed trains reach
  the third band: 0 to 320 in 206 s and 11 km) and brake at 0.9 m/s²: a 1 km stop-to-stop run
  takes 67.5 s. The whole train must be under a limit before it may speed up again, so where
  high-speed track meets ordinary track the train brakes to 160 before the joint as for a curve.
  The line panel's stop times carry a hover with the top speed reached from the stop before and
  what limits it (the trains, ordinary track, curves or a crossover; `TrackApi.line_stretch_speeds`).
- **Lines**: ordered stops, plus the exact path between them stored as edges with a direction, so
  new track never reroutes a line by itself. Splitting an edge rewrites the paths through it;
  deleting one reroutes by the fastest path, unless that path reverses where the old one did not,
  or marks the line broken. Removing a station a line stops at takes it out of the line's stops
  and routes the line through the rest the same way, else leaves it broken (T-859); one undo
  restores both. The two pieces left either side of a removed station become one where they
  would join under a platform (on one straight, alike in track count, type and built state;
  `unjoin_near`), in the same edit (T-1299). A line may pass a station without stopping, and may stop at a station
  twice, never twice in a row: stops that would repeat in a row count once (`distinct_stops` in
  `sim/src/track/wasm/lines.rs`), so taking out a loop's only middle stop leaves a one-stop line.
  A line with one stop or none is inactive: no route, no trains, nothing refused (T-1242).
  **Reversing partway** (T-797, notes/T-797.md): a line's path may turn back at a stop or at a
  bare track end (a node with one edge end and no station, such as a headshunt), nowhere else.
  Nothing stores it: a reversal is two path entries arriving at and leaving a node on the same
  side, and stops are matched to the path so every reversal not at a bare end falls on one. The
  route search may reverse at a hop's first stop, or at a bare end on the way, for 600 s of search
  time each (`REVERSAL_SEARCH_S`), and the stop insertion counts each reversal as 5 km of path
  (`REVERSAL_INSERT_M`), so a route without one wins unless it is much longer. A reversing train
  stands the line's turnaround there, not its dwell; its middle stands d short of the node on
  arrival and leaves from d past it in the run's unfolded offsets (`Run::jump`, the same place on
  the ground; d is half the train at a through station, half the platform at a dead end, half the
  train plus 20 m at a bare end, each at most half the way to the next place a train stands), so
  it is drawn jumping once as it stops. A bare end is no stop: the profile stands the turnaround
  there (`Run::ends`) and leaves it out of the stop times, so riders through it pay the stand in
  their ride. The stand is inside the run's duration, so the round trip, fleet
  and demand need nothing new; trains still enter and leave only at the two ends.
  **Loops** (T-798, notes/T-797.md): a line whose last stop is its first (three stops or more) is a
  loop; nothing else stores it. A **circle** carries straight on through its first stop A (its last
  path entry and first are a through move); a **balloon** turns back there (e.g. out along a stem,
  round a loop, back down the stem). The route search's closing hop asks to arrive on the side that
  carries on round, and only without such a path arrives on the other (a balloon); a reroute after
  a deletion never turns a circle into a balloon (the closing turn counts as a reversal). Run 0 goes
  round one way and run 1 the other, each from A to A with its own trains; a circle stands the dwell
  at A between laps, a balloon the turnaround. Every loop runs both ways (one-way loops are T-798's
  stage L3). Stops closed into a loop whose path only runs out and back the same way (a
  lollipop's loop stops taken out, leaving A, M, A) make no loop: a stop edit keeps the stops up
  to where the path turns, which becomes the line's end, a terminus rather than a reversal
  partway (T-1273, `edited_stops` in `sim/src/track/wasm/lines.rs`).
  Empty and one-stop lines are valid inactive drafts with an empty path. Stop edits use the same
  turn-valid route search for every insertion position, choosing minimum total path length plus
  the reversal penalty (append wins ties). Copying preserves stops, exact path, dwell, turnaround, cars,
  colour and custom badge as independent inputs, and starts with no trains (counts at 0, or a
  headway line's headways at none) until the player sets them, so a copy never doubles a
  service (T-1161); edits and removal are undoable.
- **Scheduling** (T-104, Anita): every line sets a whole number of trains on the entire line,
  both directions together, for **high, medium and low demand** (the column header is "Demand").
  High is the two peaks, medium midday and evening, low night. New lines start at 0 / 0 / 0;
  zero means no new departures. The panel shows the integer allocation and resulting headway.
  Frequency is derived from the capacity-adjusted round trip, with a bounded solve on edits (at
  most 80 rounds, ending at the first round that changes no rate or round trip, T-747);
  extending track or changing congestion adjusts the headway while preserving the allocation.
  It never exceeds what the line's track can carry (6.2): trains beyond that wait at the ends.
  A loop's lap is a run and its closing stand; its round trip is twice the longer lap, and its count
  is even, half each way (the stepper moves by 2, headway mode doubles each way's count, an odd
  count from a save or an edit rounds up), so each way's trains keep the headway over their lap.
  Resources used by count schedules keep continuous delays so rounding cannot make the solve
  oscillate. Legacy saves preserve their original frequencies until a schedule is edited.
- **Headway mode** (T-625, `LineData::auto`, `Op::AutoTrains`): instead of a count per level a
  line can keep a headway per level, in seconds (0 is no service; the panel offers 1 minute to 2
  hours). Its counts are then derived: the fewest trains with round trip over headway, the round
  trip being the capacity-adjusted one with running, dwell, turnaround and waiting (`auto_trains`;
  the real headway may run 0.5% over the chosen one, so a few seconds of drift in the round trip
  buy no train). At the track's limit the count is clamped (T-679, `clamp_auto`): when the headway
  is no shorter than the line's shortest headway (6.2, with the other lines' rates as they stand,
  `Capacity::line_limits`) or within the 0.5% of it, the line takes no more trains than fit, at least
  one, so it runs just over the headway set (4:07 for 4:00, say) and no train is held out. A headway
  well under the limit is not clamped: the trains past it wait at the ends and the panel warns.
  Every edit that changes the line, its track or shared track settles the counts
  again (`TrackWorld::settle_train_counts`: count from the round trip, settle the rates, repeat up
  to 5 rounds), and whole trains that full track would keep waiting at the ends (6.2) are dropped,
  so the fleet is what runs. On over-full track (other lines' fixed rates already past its
  capacity) they are kept, held out of service, and a fleet short of its rate there counts as
  settled (T-752). A count that grows buys cars as any fleet growth does (6.4), and the
  edit that grows it is refused with nothing changed if the cash is short. A hand-set count ends
  the headways; switching to headways starts from the headway the current trains give. Switching
  either way buys and sells no train: that edit keeps the line's counts as they stand (T-752),
  except at a level with no trains, which takes its headway's count (T-849: a new line set
  straight to headways runs). Saved in a
  trailing `LAU1` block of the track save (line index and three headways, only when a line has
  them); a save without it loads every line by hand.
- **Departures** (`Service::run_deps`): per period of the line's local time (T-309: its first
  stop's region's, `Lattice::shift`; a New York to Toronto line is on one offset, a London to
  Paris line would run on London's), the first departure half a one-way run before the period's
  start and then one a headway, until half a run before its end (T-835, `Lattice::lead`: the mean
  of the two runs at medium, at most 2 hours). So a train belongs to the period in force when it
  reaches the line's middle, which changes service on the hour; stops near the ends change up to
  half a run early or late. A loop's middle is half a lap from its first stop (the same figure), and
  both its runs leave A together. A period followed by no service (the evening before a line's night
  without trains, a peak before an empty midday) departs until its own end, so the line's last
  trains are not pulled early; it runs half a run longer at its level. Trains still run end to end.
  A whole-line fleet's reverse departures are offset by the forward trip so
  each train turns straight back. When the pattern changes (a new period, or an edit) a new
  departure closer than half a headway after the old pattern's last one is left out, so trains
  never leave a terminal in pairs.
- **Trains join and leave at the ends** (T-206, notes/T-206.md). An edit that changes a running
  line's departures (its train count, a whole-line fleet's headway after track or congestion
  changes, its turnaround) takes effect at one game time (the edit's, a moment later for a train
  count, below): departures before it keep the
  old pattern (an *era*, `Service::eras`), so every train already running carries on where it was;
  departures after it follow the new one. A train running when its profile changed is drawn from
  where it was (`Service::shifts`: at the same place, or at the same stop with the dwell it had
  left). At each terminal trains are handed over (`Service::hand_over`): a train that has finished
  its run and turnaround takes the next departure from there if it comes within that departure's
  headway (an hour at most), standing at the platform until then; a departure no train waits for
  is a train entering service there; a train that finds another already waiting, or no departure
  soon enough, leaves service at the end of its turnaround. So an added train enters at a
  terminal, a removed one finishes its trip and leaves at a terminal (on a loop, a train finishing
  a run takes that run's next departure from A, standing there as the timing hold, and trains enter
  and leave service at A only), and headways even out as
  trains turn; the same happens at every change of period. A train that had left service before
  a change never comes back. In steady service with a whole-line fleet every train is ready at
  exactly its next departure and nothing stands longer. Each drawn trip names the departure its
  train takes next (`Trip::next`, the fourth number per trip of `TrackApi.trips`, NaN when it
  leaves service), so a selected train is followed onto its next trip (T-245). Eras are dropped
  once no train of theirs can be drawn (an hour plus the longest trip and two headways after the
  change); until then a save carries them and the shifts (T-1071, [architecture.md](architecture.md)
  11), so a game saved mid-change loads with its trains where they were. Deterministic: the drawn
  trips are a function of the service, its eras and the hour asked for, and every window agrees.
- **Changing a train count** (T-229, notes/T-229.md). The stepper shows the new count at
  once (`game/world.ts` `pendingTrains`), and clicks made while an edit is in flight are folded
  into one edit sent when it lands. The worker sends a service-only state for it: no geometry
  moved, so it skips rebuilding the renderer's buffers and the main thread swaps in only the new
  run profiles (`renderer.setService`). A count or headway edit takes effect a moment ahead of the
  clock (T-1071, notes/T-1071.md): the clock worker's lead is 0.25 s plus twice what its last such
  edit took it, at most 2 s, of wall time at the clock's rate (`serviceLeadS`, `TrackApi.
  set_edit_lead`; nothing when paused), and the state carries that game time (`effectiveAt`). The
  renderer stages the state (`renderer.stage`) and puts it in, with the trips of its version, at
  the first frame at or past that time, so the old and new service draw every train at the same
  place when they swap, however fast the clock runs. The edit's charge, fleet and panel figures change at once.
  Any other state puts in the staged ones first.
- **Trip times on the stops list**: each stop shows the time from the start of the line, next to
  the station (e.g. Grand Central 1m23s, Long Island City 2m59s), and the panel shows the whole
  round trip.

### 6.4 Costs and money

Money is present, not tight, as in NIMBY Rails: construction, train purchase, operating cost per
train-km and car-km, a player-set fare curve by distance. One price list worldwide. Track costs a **base of
US$90M per route-km of double track** (single track 0.6x, high-speed track 1.5x, T-817; ramps at
the dearer level) times a
multiplier per level (Anita, 2026-10-08; ground and +1 raised to 0.5 and 1.0 for the building
knockouts and road crossings that are not computed, T-1068). The reference is a cheap-country metro such as Korea's
or Spain's, about US$100-150M a tunnelled km with its stations (Anita, 2026-10-21, T-805):

| Level | +3 | +2 | +1 | 0 | -1 | -2 | -3 |
|---|---|---|---|---|---|---|---|
| Multiplier | 1.8 | 1.3 | 1.0 | 0.5 | 1.0 | 1.1 | 1.2 |
| US$M per km | 162 | 117 | 90 | 45 | 90 | 99 | 108 |

In a **sandbox** game [T-728] every price, fare and cost is the same, but no edit is refused for
money: it pays, and the cash may go below zero (the cash is not shown; docs/ui-panels.md).

- **Over or under water: 2x** the level's price. **Level 0 cannot be built over water**; a
  ground-level segment drawn across it becomes a bridge (6.1). Water comes from a mask in the
  pack (T-030), and outside every pack's mask from the world's coarse mask (T-1285).
- **Terrain** (T-1270, Anita 2026-10-25, design notes/T-1270.md): each zone's ground is in four
  classes, and track at elevation -1 to +3 costs the class's factor more; bored tunnels (-2, -3)
  cost the same anywhere, so in the mountain class a -2 tunnel (US$99M a km) is cheaper than track
  at ground level (US$112.5M) or a viaduct (US$225M). Over water the water's 2x stands alone. A ramp
  is charged at the dearer of its two levels with the factor counted. Stations, junctions,
  crossings and upkeep take no factor; nothing is refused for steep ground; grades do not slow trains.

  | Class | Flat | Rolling | Hilly | Mountain |
  |---|---|---|---|---|
  | Mean slope | under 2° | 2-5° | 5-10° | 10° and over |
  | Factor (`TERRAIN_FACTOR`) | 1 | 1.25 | 1.6 | 2.5 |

  The classes come from GEDTM30 (OpenGeoHub's bare-earth terrain model at 240 m, CC BY 4.0) by
  `pipeline/zone_terrain.py`: the mean slope over each cell of a world lattice of 1/240 degree
  (about 460 m north-south), then a 3 x 3 majority filter, one file per zone
  (`data/terrain/<id>.terrain`, a median of 13 KB, 56 MB for all 1,770 zones; `cost::ClassGrid`). The clock worker fetches a zone's
  file with its water mask (zones.md 6), but no preview or edit waits for it: track drawn before it
  lands is priced on flat ground and priced again when it lands, the blueprint quote following;
  Construct waits for the files under the blueprint, so nothing is paid on a guess. Each edge keeps
  its spans of rough ground (`Network::edge_terrain`, sampled every 100 m, `TERRAIN_STEP`); a file
  landing reprices only the track inside its box. The track tool's tag gives the classes a segment
  crosses ("Mountain 2.5x"), its multiplier includes the factor, and the blueprint breakdown has a
  Terrain row per class with what the ground adds.
- The Build tab shows only the multiplier and price of the level currently selected, not the
  whole table.
- Stations (200 m, two platform tracks), US$M: a fixed 50 that every station has wherever it is
  (platforms, halls, systems), plus a part by level, 25 at grade, 25 + 15 a level up, 50 + 25 a
  level down: 75 at grade, 90/105/120 at +1/+2/+3, 125/150/175 at -1/-2/-3 (T-1267). So a ground
  station costs half a -2 one, where ground track costs 0.45 of -2 track. Scaled by
  `0.4 + 0.6 x length / 200`.
- Junction: 0.25 km of track at its level; a flyover adds 0.6 km one level away; flat
  crossing 0.1 km (cheap to build, dear in delay). Station upgrades (6.1, T-1281): a terminal
  1 km of track at the station's level (US$45M at grade, US$99M at -2, 2x over water), park and
  ride US$20M, waiting room US$5M, a passage US$10M plus US$90M a km. A crossover: two junctions, all told (6.1,
  T-338); the blueprint breakdown counts it as two junction switches.
- **Blueprint, then construct** (Anita, 2026-10-09). New track, stations and junctions are drawn
  as a **blueprint**: free, freely edited and undone, drawn faded, shown with its total cost. The
  player pays when they click **Construct**, which builds the blueprint at once if the money is
  there: all of it, what one line still lacks (the line panel shows its price), one stretch of
  track, or one station. Only constructed track carries trains. Lines can be planned over
  blueprint track: they show their stop times, and run (load capacity, have trains, count for
  demand) once every edge and stop they use is constructed, so a line drawn over constructed
  track runs as soon as it has trains; Construct with nothing left to build is refused with the
  issue `NothingToConstruct` (T-849). A junction is paid when its third
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
- **Trains** (T-028): a 20 m car costs US$2.5M, a high-speed car US$3.75M (T-817). The network owns cars of
  each kind apart, and spares of one kind never serve the other: a line switched to high-speed
  trains buys its cars beyond the spare high-speed ones, and its ordinary cars stay as spares (and
  the same the other way). The running lines need
  their busiest schedule's trains times their cars. Any edit that makes them need more than are
  owned (a construct that starts a line, a larger allocation, a headway that needs more trains,
  longer trains, or an undo) buys the difference with it, and is refused with nothing changed if
  construction plus trains is more than the cash. Cars are never sold or refunded: those a line no
  longer needs stay in the depot as spares and serve the next line that grows, any line's (T-631).
  Loaded fleets count as already paid. The Money tab lists cars owned, needed and spare (and the
  high-speed cars owned and spare, once there are any); the line
  panel's train buttons say in their hover what a step buys or frees; the blueprint quote and a
  planned line's Construct button include the cars.
- **Running cost** US$4 per train-km plus US$2 per car-km (trains an hour each way x line length,
  times 4 + 2 x cars: a 4-car train US$12 a train-km, 9 cars US$22, 12 cars US$28; under real
  figures so fares can cover it, T-805, T-807, real figures in research/running-costs.md;
  `RUN_COST_TRAIN_KM` and `RUN_COST_CAR_KM`; high-speed trains US$5 a train-km plus US$3 a car-km,
  `HS_RUN_COST_*`, T-817, T-867).
- **Upkeep** (T-867, notes/T-867.md): constructed track and stations cost money every game day
  whether trains run or not, draining evenly through the day: US$30k a route-km of double track
  times its price factor (`track_factor`: high-speed 1.5x, single 0.6x; quad is two double-track
  edges, so twice), US$50k a constructed station at any elevation; crossovers, junctions and
  flyovers nothing on top; blueprint nothing (`UPKEEP_TRACK_KM`, `UPKEEP_STATION`, real US$300 and
  500 times `ECONOMY`; `TrackApi::upkeep`). A line's share (`TrackApi::line_upkeep`): each edge's
  upkeep split between the running lines on it by their trains a weekday, each station's between
  the running lines stopping there; what no running line uses counts only for the network. Its
  profit (line panel, Lines tab) is fares less its running cost and that share. By region, each
  edge's upkeep counts where its midpoint is and each station's in its own region
  (`upkeepShares`, game/regionMoney.ts). The ledgers keep it as `upkeep` beside `running`. Shown
  in the Money tab's Upkeep row, an Upkeep row a day in the station panel (constructed stations)
  and the track inspector (not on crossovers), and "$X / day upkeep" under the Build tab's
  blueprint total (the pieces switched off left out); each hover states the rule.
- Running costs, upkeep and **fares**
  are settled by the clock worker into the cash and a ledger per day (30 kept) and per hour (48
  kept, T-447) as the game runs, not once an hour (T-284). The ledgers hold one entry per day or
  hour, in order; a load merges any day or hour an older save lists twice. Money time never goes
  back: a clock set behind the time already settled adds nothing until it passes it, and a charge
  meanwhile lands on the latest day. Both threads time the clock on `performance.timeOrigin +
  performance.now()`, never `Date.now()`, which drifts from it (T-488). Each clock message carries
  the load it follows, and the worker ignores one sent before the main thread saw the latest load,
  which still holds the previous game's time (T-715). A load settles again from the save's clock, so
  ledger entries after it are dropped and their net returned to the cash (the running cost worked
  out again on the loaded network at the rate the kept whole days show, since the day ledger keeps
  30 days); edits' charges in them stay, on the clock's day. After a load the main thread sends the
  income, region shares and points again even when they match the last game's. **How much each
  game hour gets** (T-736, `app/src/game/moneyTiming.ts`): demand gives fares and rides per hour of
  each period and kind of day, the timetable a running cost per hour of each level. An hour's rides
  are the period's spread over its hours by the boarding account's arrival shape (6.6: tent, falling
  or valley, at the busiest-hour factor of that kind of day), counted when riders set out; its fares
  and points follow the same shape but land as the ride ends, spread over a trailing window as long
  as the network's typical one-way run (each running line's one-way run at medium, weighted by its
  car-km; at most 2 hours); its running cost is the timetable's car-km spread over the same window
  moved half its length earlier, as trains leave half a run before their period (T-835) and keep
  running into the next; a period followed by no service adds its line's half run at its level a
  day, spread over the period (`runTail` in the clock worker). So no period or day
  change is a cliff, a ride late on Friday pays on Saturday, and any run of whole days with the same
  tables keeps its totals. **Within the hour**, its fares are paid as that hour's trains pick up
  riders (the boarding account's trains leaving platforms, 6.6, weighted by the riders each takes;
  evenly in an hour without one), its running cost drains evenly, and both are paid in full by the
  hour's end. The worker settles and sends the cash and today's
  and this hour's ledger four times a wall second while the clock runs; the top bar shows the cash to the dollar,
  and only it and the Money tab's cash, Today column and History re-render. The fare for a trip
  follows one curve of trip length for the network, set in the Money tab [T-883, T-1112, T-1143,
  `app/src/game/fareCurve.ts`]: straight stretches through the player's three nodes, carried on
  past the last at its slope. The first node is the base fare at 0 km; each later one is farther and
  never cheaper, so no fare is below the base or falls with distance. **Fares by country**
  (T-1277, `app/src/game/income.ts`): the curve is set for India, the baseline, and each country
  charges it times its fare scale: by default (its GDP per head at PPP over India's)^0.5 in tenths
  (the US 2.8, the UK 2.3, China 1.6), or the player's own, 0.1 to 9.9 (`Fares.scales`, saved
  with the fares; only the player's are kept). The curve starts as $0.5357 + $0.03571 a km (nodes at 0,
  20 and 100 km), so New York starts at $1.50 + $0.10. A save from before carries no
  `baseline` and its curve, New York's fares, is divided by the home country's scale as it loads.
  An older save's two nodes, or a base and rate, load as the three nodes on their line (one below
  $0 at 0 km starting at $0 instead), any other curve as the line through its first and last node.
  The curve's fares are kept to a hundredth of a cent, so a fare typed where the scale is not 1
  reads back as typed. A trip that changes lines pays it once on its whole length, at the scale of
  the country it starts in: each zone's populations get the curve times their country's scale, and
  the regional market the curve itself. Fare income takes each stretch at its starting stop's
  country's scale and the base at the boarding stop's (`stopScales`, game/money.ts), and so does
  the split between regions. In
  demand, route search charges the first stretch as a base and a rate per km on each ride
  (`set_fares`), and each station pair's cost adds what the rest of the curve adds at the expected
  km of its strategy (the riders' split over lines by their departures; `set_fare_curve`,
  `kernel::trip_km`). The assignment puts that extra, the trips times the curve's extra at their
  pair's km, on the route nodes where they first board, split as they split (`Loads::rn_fare_x`,
  each population's `fare_extra`); the trip explorer shows each route's extra on its rides by km.
  Fare income is the first stretch's base on each of demand's trips and its rate on their
  passenger-km per period, plus demand's extra on the line it is counted at (`LineDemand.fareX`,
  `app/src/game/money.ts`; on a weekend the extra follows each route node's boardings), so demand
  and income charge the same trips the same fares. Changing the fare
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
  period's share of passenger-km, each hour's as above (when riders set out), counted within the
  hour as its fares are paid, and divided between regions by the region of each ride's boarding stop.
- **Points** (T-684, T-694, `app/src/game/points.ts`): a measure per region of how good its
  network is: the points it earned in the past 24 whole game hours (fewer while the hour ledger is
  younger), recomputed once a game hour; never a running total. 20 a rail trip and 1 a
  passenger-km, not scaled by the economy, unlocking nothing yet. A trip is a rider's whole rail
  journey (changes of line and walks between stations within it included), counted in the region
  it starts in: each region's rail trips a day are the solve's (its own demand's, plus half of each
  regional corridor's at each end, all of one within a place), spread over the boardings at the
  region's stations, so a region's day is exact and the hour follows its boardings (a region whose
  riders all board at another region's stations scores no trips). Passenger-km are each line's
  period loads times its stretches' lengths, carried by the period's boardings onto the line
  (each boarding the line's average km), so they count where riders get on. The main thread sends
  each line's points per game hour by region, kind of day and period; the worker settles them as
  fares land (each line at its local period), into a `points` figure in every day and hour entry
  (saved with the ledgers; older entries read as 0). A region's points are the sum of its last 24
  closed hour entries; the Cities list shows them, the top bar's city name in its hover.
- **Stars** (T-685): a region earns a star the first time a final solve's train share for it, on
  the average day (the top bar's goal number as it shows by default; never the weekday or weekend
  view, so switching the day shown earns nothing), reaches 10%. The header keeps the game second it was
  earned (`stars`), and it stays if the share falls. Drawn stamps come later.
- **Starting money $6B** (T-805): about one real first line, a 20 km line at -2 with 18 stations
  plus its trains (about $4.2B), with room for a short second one.
- **Economy at 100x** (Anita, 2026-10-09, as Subway Builder does; T-081): fares and running costs
  per game day count 100 times the real figures (`ECONOMY` in `sim/src/track/params.rs`, the one
  constant; running costs scale in the track model, fares in `app/src/game/money.ts`), so a game
  day earns and spends what a hundred real days would. The fare the player sets and the US$4 a
  train-km plus US$2 a car-km shown (Money tab, line panel and train length hovers) are the figures
  before that scaling; upkeep is shown as the game day's figure (US$30k, real US$300); construction
  and train prices are not scaled.

### 6.5 Edits

The clock worker owns the network. The main thread sends edit operations (each with an inverse,
for undo); the worker answers with the new state, then the edit's answer (or the reasons it was
refused). The alignment being drawn is previewed by the worker as a trial edit, applied, measured
and rolled back, so the preview knows everything an edit would refuse (timings in notes/T-025.md). The
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
which may show a preview. Edit timings: notes/T-050.md. A save holds PIs, nodes, stations and
lines; its format is in [architecture.md](architecture.md) 11.

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
- **Arrivals**: the riders coming to a stop in a period (T-1255: demand's `arrive`, sent with
  `TrackApi.set_arrivals` after `set_riders`: where demand's queue turned riders away there,
  everyone who would ride if the trains had room, the free-flow pass's boardings; elsewhere, and
  without it, the boardings, so the account boards about demand's riders) arrive spread over it by
  a fixed shape, scaled so the
  busiest clock hour is the busiest-hour factor times the mean hour: a tent (rising to the middle,
  falling away) for the two peaks and midday, falling from its start for the evening, highest at
  both ends for the night. The period's total is kept exactly. Riders arrive only as demand's
  service span lets them (T-842b, docs/demand-kernel.md step 4): the shape at each moment is
  multiplied by the share of riders whose tolerance (30 minutes to 6 hours, log-uniform) covers the
  wait for the stop's next departure in a steady day, and scaled so the period's total is the same.
  So nobody arrives hours before a peak-only line's next train or overnight before the first; a
  stop with a train every half hour or better arrives by the plain shape exactly, and a period with no
  train due within 6 hours at all keeps the plain shape (demand sends it almost nobody).
- **Each train, in the order it leaves each stop**: the line's share alighting there gets off
  (demand's alightings over the riders it has arriving, per period; everyone at the last stop), then
  it takes whoever is waiting up to its crush load (160 a car). The rest wait for the next train of
  the same line and direction. Lines sharing stops keep separate queues: demand has already split
  the riders between them by frequency (4.4's common lines), and a rider left behind does not
  switch to another line. On a circle (T-798) the last stop lets off only its share; the riders
  staying aboard join the first stop's queue when their train arrives there, ahead of later
  arrivals, so they leave on that run's next departure, normally the same train after its hold (a
  second pass of the account). In the books they leave and board again there.
- **Riders go another way** (T-1067, T-1255; `WAIT_FOR_ROOM_S`, `GIVE_UP_S`, `WAITING_CAP`): a
  rider comes and waits only for a train with room within 30 minutes. As each train leaves, one
  platform side (a direction of a line at a stop) holds whoever this train and every train due in
  the next 30 minutes would take (this train's room each; the gap since the last train stands for
  the headway), at most 5,000: whoever comes when it is full is turned away, the latest arrivals
  first. Demand holds its riders to the same queue (docs/demand-kernel.md step 7), so on a line
  short of trains the platforms fill and every peak train leaves full, and the riders beyond those
  who ride (the free-flow pass's, at the stops where demand turned riders away) are the ones
  turned away. A rider whom full trains have still kept more
  than 90 minutes past the first train they came for leaves too (as a train leaves, the oldest
  first). None of these boards, so none counts in the line's boardings (`RunAccount::gone`).
  Fares are demand's riders, already held to the queue; the account's boardings differ from them
  a little and do not change the money (T-1069). A drawn crowd between two trains stops at 5,000 too.
- **Local time** (T-309): a line's account runs on its own local time (`LineAccount::shift`): its
  hours, busiest hours, kind of day and "today" from its local midnight. The money settles each
  local time's lines by their own period and day (`localGroups` in the clock worker).
- **A day at a time**: worked out for a day of departures from 03:00, from empty platforms; a
  direction where anyone is still waiting when that day ends is run again from a day earlier, so
  the queue carries round. Departures are the drawn trips to the bit (`Service::run_deps`). While
  a line has eras (T-206) its day starts six hours before its latest change instead, so the
  account holds the departures actually drawn around the change: a train that has not entered
  service carries nobody, and trains that left before the change keep their loads. When a
  line's trains start within that day (a change from a pattern with no trains: a new line, or
  trains set on one that had none; T-680, `RunAccount::open`), its platforms start empty and
  riders arrive at the arrival rate, each stop from one headway before the first train to reach
  it (T-848, `RunAccount::stop_open`): trains enter only at the ends, so a stop along a long line
  sees its first train up to a run after the opening, and its riders come for that train as on
  any day. Nobody who would have arrived earlier waits (they went another way). The hour's crowds,
  "today" and the riders arriving in the hour start there too: departures earlier by time of day
  are the account's day wrapping round, not trains that ran. So the opening day's account counts
  fewer trains (none in the hours before the opening) but no rider waits for a train that could
  not yet be there. Nothing is saved: it follows from the service and the riders. Each line's inputs are hashed, so an edit
  works out again only the lines it changed; new riders redo every line, a few lines at a time
  between the clock worker's other messages (3 a step).
- **Each game hour** the worker sends, with the hour's trips, each trip's riders per stop-to-stop
  stretch and each platform's departures with the riders each left behind and when each train
  stopped there (its stand before it leaves, the closing stand at a first stop, never before the
  train ahead left; T-1123), its steady arrival rate, its pulses of riders changing lines (below),
  the hour's riders left behind and longest wait, and the stretches at capacity (below). The
  renderer reads the train's load for the stretch it is drawn on (empty once at its last
  stop; packed to crush, drawn full with a red edge, T-207, T-218), and a platform's crowd at any
  moment is what the last train to stop there left plus, once it has left, who has arrived since
  (its riders get on as it stops), drawn as the platform filling up
  wherever platforms are drawn (zoom 14 and in, T-189, T-204, T-219, T-553; [ui-map.md](ui-map.md)).
- **Riders changing lines arrive with their train** (T-554, `Boarding::pulses`, notes/T-554.md):
  in the drawn crowd, not the account. Demand's transfers a day between platforms (the solve
  shown, every population; `platformTransfers`) give each platform the share of its boardings
  that change there from each other platform. Each clock hour, that share of the hour's arrivals
  is split over the feeding platform's trains reaching it in the hour by the riders each has aboard
  arriving, and reaches the platform a transfer walk after the train stops (T-1124; demand's: the
  straight line between the two stations x 1.3 at 1.2 m/s; nothing at the same station). Each
  pulse carries the station it walks from and its walk, which the map draws as a bubble walking
  between the two (ui-map.md "Platform crowds"). The
  rest of the hour's arrivals, and the share of a feeding platform whose trains bring none or
  under one rider each that hour, arrive steadily. Each platform's arrivals in the hour are the account's; the
  account's loads, departures and riders left behind do not read the pulses, so they and
  demand's crowding feedback stay one rule. The fares' timing within the hour (T-284) counts
  the pulses with the steady arrivals.
- **Stretches at capacity** (T-1125, T-1152, T-1160; `RunAccount::full_shares`, `stop_waits`): per
  stretch between two stops and direction, the share of the day's trains leaving it packed to
  crush, and the average extra wait for a train with room of the day's riders arriving at its first
  stop (rider-seconds waited after full trains, each left behind waiting the gap to the next
  departure, over riders arriving; the line panel's crowding also counts the riders turned away,
  T-1255, and this does not). Worked out once with
  each account; the account's hour lists the stretches where the longer wait of the two directions
  is at least `FULL_WAIT_MIN_S` (12 s, where the crowding colour leaves green), with the higher
  share and that wait, and the traffic view marks them by the wait ([ui-map.md](ui-map.md) "Track
  display").
- **Room on a platform** (T-219, `platformRoom` in `game/boarding.ts`): what counts as a full
  platform for drawing. Each direction's riders wait on one side, 2.5 m of usable width (half a
  6 m island platform less the edge strip kept clear) along the platform's length, at 0.3 m² a
  rider (why: DECISIONS, T-219). A 150 m platform holds 1,250 a side. Only the drawing uses it: the
  account lets a crowd grow past it, up to `WAITING_CAP` (above).
- **Cost and how full trains get**: measured on the real New York save in notes/T-164.md,
  notes/T-554.md and notes/T-207.md.
- **What the panels show** (T-230): every train-fullness figure is the account's: the gauge on
  trains, the train tag and inspector, the line inspector's "Trains today" (`Boarding::line_summary`:
  fullest train, trains leaving full and when, riders left behind a day, longest wait), the Lines
  tab's train count and the schedule's crowding dots (riders' extra wait after full trains
  against their ride, `Boarding::crowding`, T-491; each rider counted by the fare they pay, the
  base plus the per-km rate times the km a boarder at that stop rides on average with every stop
  letting off its share, `ride_km`, T-818) and the platforms. Demand's busiest-hour average train
  is not shown: riders arrive fastest mid-peak, so a line it puts under 100% can still leave riders
  behind (notes/T-230.md).

Riders left behind feed back into demand (T-188): each crowding round runs this rule on the
period's own trains and adds the average extra wait at each line, direction and stop to that
boarding's cost (docs/demand-kernel.md step 7), so a line that leaves people behind loses riders to
other routes or the car, and after the rounds no line keeps more riders than its trains hold at
crush in the busiest hour (T-1070). The account itself is not read back: the clock worker's day and demand's
periods give the same rule, but demand's is worked out inside the solve, so it settles with the
crowding rounds. There are no signals (SPEC 2.1).
