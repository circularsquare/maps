[Back to the spec](../SPEC.md)

## 6. Building and running

Full design, numbers and sources: `notes/T-008.md`.

### 6.1 Track and stations

#### Geometry and drawing

- **One track type**, anywhere: over roads at normal cost, over water and rough ground dearer
  (6.4). No buildings to demolish, no yards, no depots. Ground is flat in shape: levels are heights
  above or below the local ground, and the terrain only prices track (6.4, T-1270).
- **Geometry: straights and circular arcs**, tangent-continuous, no transition curves or splines
  (exact lengths, one speed per curve, parallel tracks stay arcs). **Track is stored as points of
  intersection (PIs)**: it runs straight between them and cuts the corner at each with a circular
  curve (Anita, 2026-10-09; T-092). A PI's radius is auto unless set: the largest that fits half of
  each adjacent leg (the whole leg next to an end), capped at what the top speed needs (1,796 m). A
  set radius is the widest that corner may take (T-1895): where its legs have no room for it, it
  takes the room they have (all of a leg it shares with an end, what an auto corner leaves, or a
  share in proportion to their own between two set corners), never a refusal. A curve under the
  minimum radius is refused ("The curve is too tight"). A route starting or ending
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
  roughest terrain class it crosses where it is charged (T-1270, T-1394) and water's +2 where it
  crosses water (T-1887), or why it
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

#### Pieces, reshaping and moving

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
  unchanged. Pieces that meet at a slight angle (up to the heading tolerance each way: a click
  almost straight ahead of a track end draws a straight piece that far off) join too: the joint is dropped when it is within 1 m of the straight
  between the corners either side, moving the track there by at most that, and otherwise stays as
  a corner (T-1964). A joint between two different levels stays as a corner on the joined edge, since a
  ramp is centred between corners; a cut through a leg whose ramp lies on a straight puts a corner
  at each end of the ramp on its side (T-622), so a station there does not move the ramp. Two
  constructed pieces join as two blueprint ones do (T-624): the joined edge stays constructed, with
  no charge and no refund, and lines, trains, levels and curve radii carry over unchanged. Joints that cannot be joined
  (a different track count, high speed beside ordinary, constructed beside blueprint, blueprint switched on beside off) stay, and a platform that
  reaches one is refused as "the end of a piece of track"; a station clicked within 5 m of one
  goes on the joint itself (T-1964). A platform may not reach a crossing
  with track on the same level (a flat crossing, 6.2), within 1 m of its ends, nor lie on a ramp
  (a split on a ramp is refused as the platform's ramp).
- **Reshaping blueprint track** (T-064, T-092, T-071): a selected stretch of blueprint track shows its PIs
  as squares on the map; dragging one previews the new shape with each curve's radius and speed
  (or why it cannot be built), letting go applies it as one undoable edit, Escape puts it back. The
  track inspector lists the corners with each curve's speed and a radius stepper (100 to 1,796 m,
  then auto), a level stepper and Remove for each corner. Add corner inserts an auto-radius
  point midway along the selected interval between an end and a corner, or two corners; the
  new square can then be dragged. Removing the last corner restores a straight stretch.
  Insertion, removal and per-corner levels each use the existing checked, undoable shape edit.
  The inspector can also put all corners on one level; its ends keep theirs, with ramps. A radius
  the stepper sets wider than the corner has room for is stored as the widest that fits. A set
  radius that a drag leaves no room for stays set and the fit narrows it (T-1898; splitting track for a station or a
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
  **Drags keep the design speed** (T-1898): in any reshape or node drag a set radius stays set,
  the most its corner may take (above), so a 320 km/h curve bent a little keeps nearly 320 and
  slows only as far as its legs make it, and is wide again once they have room. A corner the edit
  moves or adds (a dragged PI, a bent curve's PI, the PI moving with a node, a new lead-in) with an
  auto radius is laid out for the Build tab's target max speed, as a drawn corner is (`design_radii`;
  each of these edits carries `design`). At 160 nothing is set and corners stay auto.
  **Bending a curve by its middle** (T-1869): dragging a blueprint curve's middle moves that
  curve's PI so the curve's middle lands under the pointer, its radius rule unchanged (auto stays
  auto, a set radius stays set and narrows to fit): a reshape of the edge as dragging the
  PI is, the PI found by stepping it by the preview's miss until the middle is within 0.3 m. Either
  end of a blueprint platform drags the station node by the pointer's move.

#### Placing stations, curve speed and levels

- **Placing stations**: click on track (it is split there), or a loose end's node for an end
  station when the pointer is within half a platform of it; the station tool takes nothing else
  (T-1820, T-1870; ui-building.md "Room along the track"). The sim can still place one on open ground, where the station
  brings its own track [T-794, `TrackApi::add_free_station`], no tool calls it: a straight blueprint piece at the
  selected level, double or quad as the Build tab is set, the platform plus 10 m past each end
  long, with the station at its middle (on quad on the drawn pair only, T-1687), one undoable edit
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
  was clicked; a ramp is centred between vertices of different levels. A ramp longer than its leg
  spreads into the legs either side at its two levels, as near its centre as they allow, so it
  starts sooner and a corner inside it is part way up (T-1895); only when those legs together are
  too short is it refused ("Not enough room to change elevation"). A piece is never cut where a
  ramp is. Nodes and platforms sit at whole levels. Tracks cross freely
  where their heights differ by a level or more; at the same level they make a flat crossing; in
  between is invalid.

#### Track settings

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
  8 m smaller on the inside of the turn, 8 m larger outside; where that inside radius would be
  under 100 m on a drawn curve of 100-108 m it is 100 m instead if that brings the curve's middle
  at most `QUAD_EASE_TOL` (0.5 m) nearer the drawn track, a slight turn, and is otherwise too
  tight for quad, T-1678; a corner whose concentric radius would overrun its legs, as the capped curve of a
  nearly straight segment can once the millimetre grid tilts its metre-long legs, takes the auto radius
  instead, within millimetres of concentric, T-708). Lines change pairs at the ends of the quad stretch, where the two pairs meet as
  a junction (a fork with nothing behind it if the end is new track), and wherever the player has
  joined the pairs (crossing track drawn between them, or a crossover a save holds). Both pairs are
  ordinary double-track edges, cut into pieces per corner (T-244), priced, deleted and drawn like
  any track: quad costs two double tracks, plus junctions at ends on existing track. Drawn through
  points, the pairs end apart instead (above). A station placed on quad track (on track or on a
  node) goes on the pair clicked only (T-1687); a second click on the other pair adds its station
  there. A station placed level with one on the pair beside (`twin_at`; its node within half that
  platform of the spot beside) takes that station's name, so the two are one complex with one label.
- **No automatic crossovers** (T-1688): nothing the player does adds a crossover by itself, not
  drawing (quad included), placing stations or doubling up. The player draws crossing track between
  the pairs where lines should change over. A line whose stops lie on two pairs with no way between
  them is refused like any stops no track joins in order (`LinePath`: "No track joins these
  stations in this order."), unless a longer way round exists, which it then takes. Crossovers in
  saves stay and work.
- **High-speed track** (T-817, design notes/T-817.md): the third setting, always double track
  (4 m apart; there is no high-speed quad, so four tracks are HSR and double drawn side by side).
  An edge flag (`EdgeData::hsr`, `TrackApi::set_route_hsr`), so a station placed on open ground
  with it set lays high-speed track too. It costs `HSR_COST` (1.5x) double track's price at the
  same level (6.4); stations, junctions and flyovers on it cost as on double track. It meets
  ordinary track at plain nodes like any two pieces; a split keeps the flag on both halves, and
  two pieces join under a platform only when both are high speed or both are not (`unjoin_near`).
  Blueprint double track switches to high speed or back, shape and level kept, with the Select
  tool's bar (T-1898, `TrackApi::set_edges_hsr`); single track, crossovers and constructed track
  stay as they are. It cannot be doubled up (`HighSpeed`). Its cap is 320 km/h, which only a line's high-speed
  trains use; ordinary trains run on it at 160. The track inspector shows the limit.
- **Doubling up** (T-708, `TrackApi::double_up`): double track already placed, blueprint or
  built, becomes quad with one undoable edit that adds the express pair as blueprint (built track
  too: Construct pays for the pair as for any blueprint). It covers the run the selected piece is
  on: through plain joints and stations into more double track, stopping each way at a junction,
  a track end, single track, a crossover or track that has a pair beside it already, and short of
  a piece with a curve the pair cannot follow (under 100 m beside it, quad's rule above; T-1678),
  where the pair has its own loose end; the selected piece having one is refused
  (`RadiusTooSmall`). The pair runs 8 m to the left of the selected piece's direction, or to the
  right where the left is refused (crossing other track, water) or would stop short where the right would not, every
  corner concentric, one piece beside each piece with a joint
  beside each joint. The pair passes the run's stations with a plain joint beside each and gets no
  stations of its own (T-1689): the player clicks stations onto it where wanted (T-1687), and no
  crossover goes in (above). At a junction end the pair swings in to meet the junction (the T-255 taper), so lines
  can change pairs there; at any other end it has its own loose end beside the drawn pair's, or
  carries on from a loose end of the pair beside that is there already (quad drawn earlier). Refused
  on single track (`Tracks`), high-speed track (`HighSpeed`), a crossover, and track with a pair
  beside it (`Alongside`); a run stops at high-speed track as at single track.

#### Routes beside track, crossovers and junctions

- **A route beside track** (T-1685, T-1721, `TrackApi::set_route_beside(_, to_edge)`, used by the
  next `preview_route` or `add_route` only): a route with no PIs whose end is open ground beside the
  run of track through `to_edge` (walked through plain joints and stations as doubling up walks, any
  track count, at most 400 pieces) ends on the line 12 m (`BESIDE_SPACING`, T-1895) from that run
  and runs along it: `express_verts` on the run's track as fitted, cut where the route starts and
  ends (inside a curve, a corner where the tangents at the cut part's ends cross, with the curve's
  radius), every corner concentric, with T-1678's easing of inside corners. Distances and sides
  are taken from the fitted track, curves and all. The line is on the start's side, so the route
  never crosses the run (T-1895), except where the start is on the run itself or is open ground
  within three spacings of it, which take the end's side. A start on the line (a node within half
  the spacing of it, open ground within three spacings of the track) runs along it from the start,
  an open-ground start moving onto it and no lead-ins at the ends. Any other start comes in: its
  route keeps its lead-in and turns, at an auto-radius corner, onto the line at the roundest of 17
  places (T-1781): from where it has gone along the run as far as the start is from the run (at
  most half the way to the end) out to 90% of the way, the one whose fitted corner is widest (to
  the design radius, then the nearest), the line's own corners still fitting and no part of it
  crossing the run. A route that would still cross the run (none of the places qualifies, or a
  lead-in does) is the plain route instead. Every PI is at the end's level
  (the run's ramps are not copied, T-1722); the route ramps over what it crosses as any route does.
  Otherwise it is an ordinary route: the same checks, its own loose ends, no crossovers. The
  drawing tool's side is [ui-building.md](ui-building.md) "Drawing track".
- **Crossovers** (T-338, design notes/T-334.md): a crossover joins a track to the track running
  beside it. The game never places them (T-1688, above) and the player has no tool for them
  (T-733): they come only from saves, which keep theirs, and from checks. `TrackApi::add_crossover` finds the nearest other track whose centre line is 6-15 m
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
  there and the branch ramps to the level selected. On quad track only the pair clicked branches;
  high-speed track stays high speed on both halves. It is part of
  the route edit, so the ghost, the click, the real-network snap and the edit API agree, in one
  undo step.

#### Water, bridges and ramps

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
  radius, one PI each, and every PI of the segment gets its radius set. The ramps are laid one after
  another along the segment, each as close to its water as fits: one that a curve or a vertex keeps
  off its place moves back in steps of 25 m, up to 400 m, on its own. Where the ramp down after one
  crossing and the ramp up before the next have no room between them, the bridge stays up over the
  land between, priced as any viaduct [T-2414]. Quad
  track bridges each pair on its own. One edit and one undo step; the preview (`preview_route`)
  and `add_route` share the code, so the ghost shows the bridge, its price and the elevation
  reached (`0 +1` in the tag). Another level is left as drawn. A route through clicked corners (the
  edit API's `pis`, as the real-network snap sends them, T-727) is bridged the same way when every
  corner is at ground level, the bridge's vertices cut into pieces with the corners [T-667].
  Through more than 16 vertices (a snapped real route) each stretch of water is bridged on its
  own, between the corners 4 km of chainage either side, those corners keeping the radii the
  whole route gives them (water read every 3.1 m, a quarter of the usual step, since each piece
  is checked with its own samples); where that finds no room the stretch is raised whole from a
  corner before the water to one after it, the ramps starting as late and ending as soon as they
  can while the water and 14.5 m either side stay up, spread over as many short legs as they need
  (T-1895, above); with no room even so (water just past an end) the route is refused as below
  [T-727d, T-1187; `bridge::bridge_long`].
  A new end (a free click, not track already there) over water, or on land with no room for its
  ramp before the water, stays up at level 1, with no ramp on its side, so a segment may end or
  start mid-river or on an island with one ramp, or none; quad track's express pair, sharing that
  end, must stay up there too [T-1269, T-2414]. A segment drawn at ground level from or to an end
  already at level 1 is bridged the same way: it stays up to the water, or comes down at once
  where there is room (400 m) to go up again before it [T-1269]. A segment from or to a dead end
  of blueprint track at ground level (no station; not on quad) with no room for its ramp puts that
  end a level up in the same edit: the end's piece of track ramps up to it as close to it as fits,
  what that piece has up already staying up, so a bridge just before the end carries on to it
  (`raise_end`) [T-2414]. With no room otherwise (water within 215 m of a station, a junction,
  track the segment branches off or built track at its start or end, or a curve that leaves no
  vertex room) the edit is refused, naming the end short of room where there is one:
  `BridgeNoRoomStation` "Not enough space for a bridge ramp between the station and the water",
  `BridgeNoRoomJunction` "... between the junction and the water", `BridgeNoRoomBuilt` "...
  between the built track and the water", else `BridgeNoRoom` "Not enough space for bridge ramps"
  (T-1852, T-2414)
- **Ramping over track** (T-1615, `draw.rs` `shaped`, `bridge::ramp_over`): with the Build tab's
  ramp switch on (the default; `set_route_ramp`), a route that would cross other track flat (both
  at the route's level, the other not on a ramp, more than `SHARED_NODE_CLEAR` 25 m from the
  route's ends) goes a level over it, or under it from -1 down. It is the bridge above with the
  crossings as the stretches to clear (each a point, 14.5 m clear on either side), seen with the
  route's level as 0 and the way taken as up: the same 200 m ramps, vertex placement, 25 m shifts
  and sharing of one rise between crossings less than 429 m apart. A crossing with a station
  within 400 m of it (any track's) is a stretch half that station's platform plus 10 m each way
  instead, so the bridge keeps a level stretch for a station of the new track there; where its
  ramps have no room that way, the point is tried (T-1708). At level 0 going up the water
  is bridged in the same pass. The route tries the preferred way, then the other, then a flat
  crossing as with the switch off: a way is skipped where the ramps have no room, where it would
  pass +3 or -3, where it would make a flat crossing at the new level, or where the edit's checks
  fail (a ramp across other track, ground over water). Quad's express pair ramps on its own the
  same way. The drawing tool sends the switch with each route (`ramp`); an edit API route
  without it crosses flat, a station placed on open ground never ramps, and saved track is never
  reshaped. The preview and the edit share the code.

#### Stations

- **Stations** are nodes on a route, platform length 60-400 m in 20 m steps, level and free of
  other nodes along it; curved platforms are fine. Through a station the platform is centred on
  the node; at a dead end it runs from the node along the track and trains stop at its middle
  (`service.rs`). Track drawn on from a dead-end station leaves the platform where it is: the
  station's node moves back to the platform's middle, keeping its id, and the new track starts
  from a plain node 2 m past the old end, so the platform clears it (T-1667,
  `end_station_shift`; where that split is refused the station centres on the old end as
  before). Removing the piece on one side of a through station keeps the track to its
  platform's end (half the platform and 4 m past the node; both stations' when the piece runs
  between two), so the station stays where it was rather than turning into a dead end half a
  platform back (T-1161, `station_trim_ops`). Wherever else removal (one piece or a pick of
  several) takes track from a station that stays, the station keeps a straight piece for each end
  it loses, along that end's old direction from the node, its platform reach plus 10 m long (less
  where the removed piece was shorter, never short of the platform), of the removed piece's kind,
  level and state, so a station is never off track and keeps its place and orientation (T-1699,
  `keep_under_stations`). Removing the station too takes all of it. A removal is never refused for
  the track it keeps or joins (T-1895): where that track would be refused (a curve, a crossing), the
  removal goes without it, the trim, the kept pieces and the joins at removed stations left out.
  The map draws it there, at its true length and bent as the track is, from
  zoom 14 ([ui-map.md](ui-map.md) "Stations"). Platform length caps train length (20 m cars).
  Platform tracks = the route's tracks. Riders change between stations up to 300 m apart on foot,
  and platforms within 100 m of each other count as one station (4.4). Stations whose platforms
  come within 50 m form a station complex, which a click on a station selects as one (T-249) and
  the station ridership circles show as one (8, T-146, T-1449); the player can join any two stations up to 1 km
  apart into one complex by a **passage** and remove it (T-197, T-1281; saved with the network as
  `Network::links`, not in the undo history; links from older saves are passages, never charged).
  Riders change between any two stations of a complex on foot, however far apart (4.4, T-161),
  through a passage along the straight line at 2 m/s. A **station building** joins stations too
  (below).
- **Station buildings** (T-2217, `track/building.rs`, `wasm/buildings.rs`, `game/buildings.ts`;
  design notes/T-2208.md 3): an outline of 3 to 24 corners the player draws round stations. Its
  members are every station whose platform touches it, at any level, constructed or not: the
  platform's centre line (as the complexes see it) inside the outline or within 8 m of an edge
  (half a drawn box's width). Worked out again whenever stations or outlines change, never saved;
  a building joins its members into one complex as a passage would, at the plain change pace, and
  a change inside it counts 1 of its fixed 2 minutes, the walk whole, and a wait there counts as
  in a waiting room (T-2220, T-2276, demand-kernel.md 4.4).
  US$2,000 a m² of floor, paid at once, no upkeep, counted in the built value; a new outline pays
  for the floor the old one did not cover and refunds nothing for floor given up; removing it
  refunds nothing. Not in the undo history (`Network::buildings`, saved in a trailing `BLD1` block
  after the ferries' block, save level 4, only when there are any, so a save without buildings is as before). Refused
  (`Building*` issues): fewer than 3 or more than 24 corners or two in one place
  (`BuildingCorners`), edges that cross (`BuildingEdgesCross`), two corners more than 1 km apart
  (`BuildingTooWide`), under 1,000 m² (`BuildingTooSmall`), more than 1 m² over another building
  (`BuildingOverlap`), no constructed station touching it (`BuildingNoStation`). Over water and
  other track is fine. A station removed leaves its building standing, empty. The clock worker's
  edit ops `addBuilding`, `reshapeBuilding`, `renameBuilding` and `removeBuilding` (T-2218) are
  handled as a passage is: the charge (`building_charge`) booked as construction that day and hour,
  in the region of the outline's middle; the `buildingQuote` query gives a drawn outline's floor,
  price, stations touching and refusal. The tool and panel: ui-inspectors.md "Station buildings".

#### Station upgrades

- **Station upgrades** (T-1281, `track/upgrades.rs`, notes/T-1281.md), bought from the station
  panel's row of icons like a flyover: into the blueprint with a blueprint station (built with
  it, in the quote as "Station upgrades"), paid at once on a constructed one; removing one is
  final and refunds nothing. Saved as bits 4 to 7 of the station's node flags (old readers ignore
  them, and drop bit 7, passing tracks), on `StationData::upgrades`, so undo, renames and platform
  changes keep them. No upkeep.
  - **Terminal** (turnback sidings): a line turning there holds its platforms only for the dwell
    and turns in two sidings past the platform (6.2): 30 an hour at the default turnaround. The
    map draws the two sidings between the running tracks past each platform end with track
    beyond, from where the tracks part (zoom 14), and the box's two ends as thick dark bars
    reaching a little past its sides, like buffer stops (T-1416).
  - **Passing tracks** (T-2231, notes/T-2228.md; bit 8): a second platform track each way, the
    outer one for trains that stop, the inner one for trains that pass and a second train stopping.
    Slower lines that stop there wait for faster ones to pass (6.2, "Behind slower trains"). Priced
    as the platform plus 0.4 km of double track at the station's level, 1.5 times on high-speed
    track, twice over water (`upgrades::price`): US$22.5M for a 200 m ground station, US$45M for a
    400 m high-speed one. Works on double track only. `TrackApi::passing_spots(line)`: the
    stations where a line's trains get passed or would pass a slower line's, without them yet.
  - **Park and ride**: a ride to or from the station costs 10 perceived minutes less ([demand-mode.md](demand-mode.md),
    `kernel::ride_fixed`); a P after the station's name. Drawn as a car park beside the complex (below).
  - **Waiting room**: a wait for a train there counts at 0.8 of its minutes in route and mode
    choice (`kernel::wait_factor`, [demand-mode.md](demand-mode.md)). Drawn on one box of the
    complex as an annex as deep as the box is wide, with that platform's waiting riders (its fuller
    side) filling it from the middle out.
  - **Shared by a complex** (T-1528, notes/T-1528.md): each stays a bit on each station, which is
    all demand reads. The complex view builds park and ride or the waiting room at every station
    of the complex in one edit, at one station's price each; a station joining the complex later
    has none until the player builds it there, one leaving takes its own, and a complex split in
    two leaves each part the stations it had. However many of a complex's stations have one, it
    is drawn once.
  - **How they are drawn** (`map/facilityPlace.ts`, placed in the clock worker): from zoom 14.
    The waiting room joins one box's side at its middle, one outline and one white round box and
    annex: of the stations with one and both sides of each, the spot with the fewest other
    platforms, tracks, car parks and the name label's patch nearby, else outside the platform's
    bend. The car park is a lot of its own outside the boxes (`map/parkingLots.ts`), drawn under
    the track and stations, true to size: a 6 m aisle with a row of 2.5 by 5 m spaces either side
    (a bay, 16 m), one bay for each station with park and ride up to three, as long as the
    longest such platform within 80-300 m, rounded to whole spaces, along a platform of the
    complex. Its spot is the first found, nearest first (near edge 20, 30, 42, 56 or 72 m from the
    platform's centre line, past a box drawn at zoom 14; along that platform's middle, then a
    quarter and a half of it either way; each platform of the complex, those with park and ride
    first; the side away from the name label first) that comes within 18 m of no platform's centre
    line, 6 m of no track and 4 m of no waiting room or other lot, and misses the label's patch
    (below the complex for an east-west platform, to its right otherwise, as the names are
    placed); failing that one bay, then 0.6 of the length (at least 60 m); failing all, the spot
    with the least in the way. The surface and its lines are a little off the land's colour
    (`lot`, `lotLine` in shadeTable.ts); the spaces' lines fade in once they are 2.5-4.5 px
    apart (about zoom 16). Its hover names park and ride, the station and how many stations
    share it, and a click selects that station's complex. A terminal's hover is on its buffer
    stop: a dark bar only at the end of its platform where the track ends (a station with one
    track leaving it, flag 512, T-1527); a terminal with track on both sides has none. Station
    flags 32 (the room drawn on this box), 64 (its side), 128 (park and ride), 256 and 512.
  - **Passage**: the link above, priced US$10M plus US$90M a km of the straight line between the
    two stations (`upgrades::passage_price`); paid at once between constructed stations, else
    built and paid with the second of them (`Network::passages`, quote row "Passages").

#### Train length

- **Train length** (T-733, `TrackApi::set_line_cars`, `build_line` in `service.rs`): each line
  stores its cars, 0 for the most its shortest platform fits; a new line starts at the length last
  set on any line in the game (5 cars until one is set, `NEW_LINE_CARS`, so a crowded line is the player's cue to
  lengthen it, Anita T-2271; the save keeps it, `newCars` in its header, and a save from before T-1094 starts at 5 too) [T-1094]; a set length
  runs as set, at most that cap. A platform later shortened below it caps the trains while the
  setting stays, so lengthening the platform brings the set length back. Riders a train holds
  (160 a car at its rated load, 100%, and 240 at crush, 150%, T-2186), the cars the fleet needs and the running cost (US$4 a train-km plus US$2 a car-km) follow the cars
  run. One undoable edit, buying any cars it needs as other fleet growth does (6.4).

### 6.2 Capacity and junctions

**The graph.** Nodes (end, junction, station; the kind follows from how many edge ends meet
there) and edges (one alignment between two nodes: its PIs with radius and level, and its track
count). **Flat crossings are not nodes**: where two edges cross at the same level (two edges of an imported real network, `EdgeData.sep`, never make one, T-1751), the crossing is
found from their geometry, costs money and loads capacity, but splits nothing, so dragging a PI
across track never rewrites the graph or line paths. Track a full level (8 m) or more apart passes
freely; nearer than that the crossing is refused (`Crossing`, "Tracks cannot cross on a ramp unless
they are a full level apart."). Since flat track sits at whole levels, a refusal always has a ramp
at the crossing: a ramp passing track less than a level from it, or two tracks at one level within
1 m of a ramp's foot (`Network::find_crossings`, `CrossKind::Bad`) [T-2226]. Each edge gives one directed track per
direction (shared on single track): `(edge, direction)` is the segment in SPEC 2.1's keyframes.
At a node, every edge end leaves along the node's heading or its reverse; a move is any pair of
ends on opposite sides. A line's path turns back (arrives and leaves on the same side) only at a
station it stops at or at a bare track end, one with no station (T-797, 6.3). In a **junction**, two moves conflict when their paths cross at the same
level (track positions in opposite order on the two sides); merges and diverges do not, as track
capacity counts them. A plain double-track Y has one crossing pair. A **flyover** (a flying
junction) removes it, for its price (6.1, 6.4). A **flat crossing** makes every move on one
route conflict with every move on the other; one with no train on its other route (blueprint, or
constructed track no line runs on) holds nothing. A node where only two edge ends carry running
lines is plain track for capacity: a branch no train uses yet does not cut the section through it
(T-1805).

#### Resources and utilisation

**Resources and utilisation.** Everything shared has an occupancy `s` per train, and per period
`rho = sum(trains per hour x s) / 3600` from the scheduled frequencies:

- track section (a run with the same set of lines), per direction: `s = 70 s + v / 0.6 m/s²`
  (v the top speed reached on it), so at most 41 trains/h at 40 km/h, 34 at 80, 29 at 120, 25 at
  160, 16 at 320 (T-517)
- single-track section between passing points: run time + 60 s, counting both directions
- platform track: dwell + 60 s per stopping train, 40 s per passing one, turnaround + 60 s
  for a short-turn ending at a through station (15 an hour at the default 3 minutes). A
  single-track platform shares both directions. With **passing tracks** (T-2231) a stopping train
  holds one of two platform tracks, (dwell + 60 s) / 2 pooled, and a train that does not stop
  holds none.
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

#### Delay and slower trains

**Delay per train** (seconds), Kingman's queueing formula with a small variability term because
trains are timetabled:

    mix = 1 - sum over (line, direction) streams of (share of trains)²
    W   = 0.2 x mix x s x rho / (1 - rho)        rho <= 0.9; straight-line continuation above
        + 1800 x (1 - 1/rho)                     when rho > 1 (the queue builds for an hour)

One line alone has no queueing delay (its trains are evenly spaced) until it is over capacity. A
90 s junction shared by two equal lines: 27 s at 75%, 81 s at 90%, ~3 min at 100%, ~7 min at 110%.

**Behind slower trains** (T-1924, notes/T-1900.md N1; Anita 2026-11-02). A faster line on a stretch
of track it shares with a slower one in the same direction catches it up and follows it, unless the
slower one stops at a station with passing tracks inside the stretch (below). A shared stretch is a run of path entries both
lines take one after the other (same edge, same direction), cut where either turns back or they part.
Quad track is two edges, so lines on its two pairs share nothing and an express on its own pair keeps
its speed. Over the stretch the slower line takes `D` longer than the faster one from their free
profiles (leaving the entry; the faster one reaching the exit, the slower one leaving it, or reaching
it where either run ends there), with headway `H` and the spacing a train behind it keeps `s` (the
section occupancy above at its top speed there). **Timetabled together** (T-2419, notes/T-2413.md):
behind one slower line and no other, the timetable sends each faster train out as late as it can,
`s` before the next slower train, at the headways the two keep (`overtake::pattern_tph`). That one
closes the gap `H - 2s` to the slower train ahead before it catches it, so it loses
`max(0, D + 2s - H)`; where the faster line runs more often, each of a slower headway's faster trains
leaves a faster headway `h` earlier than the next and loses `h` more, the last one in the headway
counting by its share so the loss runs smoothly with the rates (`capacity::follow_best`). So a fast
line catches a slow one only when it gains more than the gap: 180 km of high-speed track with
ordinary trains at all seven stations every 30 min and high-speed ones at three every 30 min, the
high-speed train loses 11.7 min of the 36.9 it would gain; every 10 min each, 31.7 min (each besides
the queue on the shared track, a minute or two). Two lines that
differ by less than 30 s (`FOLLOW_TIMED_MIN_S`) are one pattern. **Behind several slower lines** on one
stretch, where no phase suits them all, the faster train meets them at a random place in their
headway, so its loss on average is `D - (H - s)/2` when `D >= H - s`, else `D² / 2(H - s)`; slower
lines are taken together from the slowest: for each, the loss behind those at least as slow with
their trains summed; the largest counts (`capacity::follow_loss`). A high-speed train behind
ordinary trains on high-speed track counts the same way. Stretches of one direction with different slower lines that
overlap are one resource (`ResKind::Follow`, wire kind 11, `build_follow`), with no utilisation, so
no map dot, and it neither fills nor caps track. The loss is a hold like the others, so it is in round
trips, trains needed, average speed and demand. It is run off as slower running over the whole
stretch (T-2018): one speed cap from the stretch's start to its end, the same in every hop there,
bisected so the run takes exactly the loss longer (`profile::Hold::span`); the train never stands
between stops for it. A loss too big even at 0.5 m/s (`FOLLOW_CRAWL_MIN_V`) waits its rest at the
stop before the stretch. Drawn, the faster train closes on the one ahead while that one stands at
its stations and is held behind it by the display guard ([ui-map.md](ui-map.md) "Never through the train ahead"), so
it trails it. Worked example (20 km, a station every
km, 10 locals an hour, 6 expresses from the second station to the end, double track,
`express_behind_local_takes_about_its_time`): the local takes 33.4 min one way, the express 8.0 min
free and 29.5 min behind it. An express and a local on one double track every 3 minutes each, as
New York's 7 runs (16 km, a station every 800 m, the express at every fourth,
`seven_express_and_local_on_one_double_track`): the express loses more than all of its 16.7 minute
gain behind the local, and the track is 142% full. A line's following losses are built again whenever any edge it runs over is (the
slower line's path and stops change its `D`), and solved again when a line on them is rescheduled.

**Waiting for faster trains** (T-2231, `track/overtake.rs`, notes/T-2228.md 5). Where the slower run
stops at a station with passing tracks strictly inside the stretch (a passing point), the timetable
makes it wait there instead. Per demand level, from the two
lines' headways and free profiles, train by train over one cycle of departures (the shortest whole
number of the slower headway that is a whole number of the faster one, at most 12 of them and 2 hours,
else 2 hours), after a warm-up. The headways are those the timetable runs to (T-2245,
`overtake::pattern_tph`): a line keeping a headway the player set is timetabled at it, as its trains
run it (or up to 0.5% over it, 6.3 "Headway mode"), so two lines set to 10 minutes meet at the same
stations every trip; other lines at their rates. At every place either run stops, each train keeps `s` behind the one ahead (only
`PASS_CLEAR_S`, 60 s, behind a slower train stepping aside at a passing point); at each passing point
a slower train waits for every faster train that would otherwise come within `s` of it before its next
passing point, leaving 60 s after the last of them passes and late enough to be `s` behind it at the
next place either run stops (T-2245), unless the wait would pass its own headway less `s` (it would
block the next one; the faster one then follows it). The faster line's trains are placed at the one of
24 offsets a faster headway that gives the least waiting, loss and time the slower trains are held
behind faster ones an hour (`Plan::outcome`; time held behind is not charged, and is nil at the best
offset on the test line). Averaged over the cycle, the slower line's wait at each passing point is a
hold at that stop (`stop_holds`; `ResKind::Wait`, wire kind 13, `Resource::node` the station, one per
faster run and passing point, with no utilisation, so no map dot, and it neither fills nor caps
track), and the faster line's remaining loss is its following loss (kind 11) for that slower line;
behind several slower lines the largest loss counts, as above. The departures placement (6.3) puts
the faster line's trains at that offset behind the slower one's; the two
directions each take their own best offset, so where one phase cannot give both, the figures can be
better than what the drawn trains do. When the headways are multiples of each other every slower trip
waits at the same stations.
Worked example (180 km of high-speed track, stations every 30 km, high-speed locals at all seven and
expresses at three, passing tracks at the five between, `passing_tracks_let_the_express_by`): free,
49.3 and 38.6 min one way; every 10 minutes each, the express loses 8 s to overtakes (9.3 min
behind the local without them; 3 min with locals every 30) and the local waits 117 s and 277 s at two stations, 7.7 min more in
all; locals every 30 minutes, the express loses nothing and the local 7.1 min. Both still pay the
queue on track two lines share (about a minute at 12 trains an hour). The same with both lines
keeping a 10 minute headway (`passing_tracks_on_headway_lines`): the same waits, 12 locals and 9
expresses.

#### Capacity limits in the clock worker

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
so its rates and run times stay the same to the bit (T-415). On an edit, a line within 0.0001
trains of its fleet is left as it is while the rest of its level steps (T-1787: on shared track
each line moved re-times its neighbours), and the step's 1% probe reads each round trip's rise
from the capacity solve alone, as the change in the line's hold delays, so the run profiles are
worked out once a round instead of twice. A save carries the settled rates to the bit, so a load
settles from them with an edit's settle and changes nothing; a save without them (older ones, or
one changed to another frame) settles in full, every line of an unsettled level stepped. A line a save loads at a frequency
rather than trains takes the whole trains that frequency needs first (architecture.md 11).
The underlying capacity pass is incremental: an edit
rebuilds only the resources around the edges, nodes and lines it touched and re-solves those and
the ones a rescheduled line uses, with the same result as rebuilding everything (T-050; timings
in notes/T-050.md). Each delay is baked into the line's run-time profile per demand level as a hold
at the resource's entry: a slower approach when the delay is shorter than what braking to a stop
and starting again there would cost, otherwise a stop in front of it. So trains visibly crawl or
queue at a tangle.

**Shown** on the map as a round dot on every resource at 75% and up, for the demand level in
force; nothing shows below 75%. It is a gauge (T-1893; before it a clock face whose hand was the
wait): a white disc with a ring whose colour runs from yellow at 75% through orange at 90% to red
at 100%, red when over 100% or full with some line's trains waiting for room (overloaded). Inside,
a three quarter dial and a needle at the utilisation (full scale at 100%, so lower right when full
or over). No number on the dot (T-1772); the wait is in the hover. Hovering it shows three lines: what it is and its
utilisation ("shared track at 81% of capacity", ">100%" when over), its trains an hour against the
most it takes ("20/25 trains per hour"), and the wait per train at the level in force ("at high
demand, slows trains by 1m21s"; no line without a wait). Each resource has a kind the app words by
(`Capacity::wire_kind`, `game/capacityKinds.ts`, the hover's word in brackets): track, single track,
platform, terminus, junction, flat crossing, a line turning on a through station's platform ("trains
turning on the platform": the line ends there and holds the platform for its turnaround plus 60 s;
helped by building a terminal there), track several lines run over ("shared track"), a turn across
the other track at a through station ("trains turning across the track"), the limiter on reversing
partway ("trains reversing"; "Reversing at Nerul Village is full" in the line panel), a
terminal's sidings (T-1281), and a **merge** (T-2185, wire kind 12, "merge"): shared double track
whose lines come onto it from two or more edges at one node (a line turning back there, or starting
there, joins nothing). Its trains wait for it at that node, so its point is the node, where they
stand; other track sits midway along its first edge. On a Y two lines run, the way towards the
main line is a merge and the way out from it plain shared track. A line whose fleet is cut shows an
alert under its schedule naming the full resource ("Turning on the platform at Nerul Village is
full") and, per demand level, how many of its trains are held out of service (they wait at the
ends of the line instead of running), with the resource's limit in trains an hour and what helps on
hover (`TrackApi.line_held`). Dots sit over the
network, under station names and the demand views' bubbles (T-099). Like station names, they hide
below zoom 12; each sits on its resource's point, and where two would touch the larger delay wins,
then the higher utilisation (T-102); names never move or hide them ([ui-map.md](ui-map.md)
"Station names and capacity tags", T-1772). A junction's dot, like the junction itself, opens the
junction inspector.
The junction inspector (T-031, [ui-building.md](ui-building.md)) lists the junction's moves (from
one station or direction to another), with the lines on each, trains an hour at the level in
force, the wait per train, which moves each crosses, and the busiest crossing's utilisation. The
line panel's round-trip split, waiting at busy track among it: [ui-inspectors.md](ui-inspectors.md).

### 6.3 Trains, lines, schedules

#### Trains and lines

- **Two train kinds, chosen per line** (T-817; Anita 2026-10-21): **normal** (top speed 160
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

#### Line groups

- **Line groups** (T-1791; Anita 2026-10-29): a grouping for the player's eyes only, each line
  still its own service. A group is a number on each line (`LineData::group`, 0 for none) with
  the group's name beside it (`group_name`, empty for its oldest line's name, the same on every
  line of it), so a copy, an undo and a delete carry it. `TrackApi.set_line_group(lines, group)`
  moves lines in or out (0) as one undoable edit, a joining line taking the group's name; a group
  left with one line stops being one, in the same edit, and so on deleting a line.
  `set_line_group_name` renames every line of it in one edit. Copy line puts the copy in its
  source's group, making a new one of the two when the source had none (one edit). The service
  never reads the group: a group edit rebuilds the lines' service like a colour change, wakes no
  demand and shows no pending label. Saved in a trailing `LGR1` block (architecture.md, the
  save format); a save without it loads every line ungrouped. The group's bullet set by hand
  (`group_look`, text and colour, T-1813) rides beside the name the same way, `set_line_group_look`
  one undoable edit, saved in `LGL1`. The Lines and Stations tabs' hand order is a rank per line
  and per node on the network (`set_list_order`, `list_order`), outside the undo history, saved in
  `LOR1`; a deleted line's rank stays with its id, so a new line reusing it takes its place.
  The group view edits stops without moving the line (`wasm/lines.rs`): `drop_stop(line, i)`
  takes a stop out and keeps the path (`set_stops` finds the route again, which can move a local
  onto express track); an end stop's removal cuts the path back to the new end. `end_line_at(line,
  i)` drops the stops past `i` toward the nearer end (the later end when level) the same way.
  `stop_passed(line, station)` makes a station the path passes a stop, path untouched. Each is
  one ordinary service edit and one undo. `drop_stop` and `end_line_at` are refused where a train
  turns back at the stop, at a loop's ends, below two stops, and for a bus line.
  A line with one stop or none is inactive: no route, no trains, nothing refused (T-1242).
  **Reversing partway** (T-797, notes/T-797.md): a line's path may turn back at a stop or at a
  bare track end (a node with one edge end and no station, such as a headshunt), nowhere else.
  Nothing stores it: a reversal is two path entries arriving at and leaving a node on the same
  side, and stops are matched to the path so every reversal not at a bare end falls on one. The
  route search may reverse at a hop's first stop, or at a bare end on the way, for 600 s of search
  time each (`REVERSAL_SEARCH_S`), and the stop insertion counts each reversal as twice the
  route's length, at least 5 km (`order_cost`, `REVERSAL_INSERT_M`, T-1899), so a route without one
  wins unless it is over three times as long. A reversing train
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
  the reversal penalty (append wins ties); an open line whose edited route reverses is tried again
  stop by stop and takes a straighter order where one exists (`straighten`, T-1899). Copying preserves stops, exact path, dwell, turnaround, cars,
  colour and custom badge as independent inputs, and starts with no trains (counts at 0, or a
  headway line's headways at none) until the player sets them, so a copy never doubles a
  service (T-1161); edits and removal are undoable.

#### Scheduling

- **Scheduling** (T-104, Anita): every line sets a whole number of trains on the entire line,
  both directions together, for **high, medium and low demand** (the column header is "Demand").
  High runs 6-20 (the morning peak, midday and evening peak), medium 20-24, low 0-6
  (`PERIOD_LEVEL`; Anita, 2026-10-27, T-1536) on every kind of day, weekends and holidays too
  (T-2335, [demand-timing.md](demand-timing.md) "Kinds of day"). New rail lines start at 0 / 0 / 0 and each level is
  set on its own, so a line given trains at high alone stops at 20:00 (T-2123; the tutorial's last
  steps ask for trains at every level); bus lines start with buses at every level; zero means no new departures. The panel shows the integer allocation and resulting headway.
  Frequency is derived from the capacity-adjusted round trip, with a bounded solve on edits (at
  most 80 rounds, ending at the first round that changes no rate or round trip, T-747; a line
  within 0.05 trains of its fleet whose shortfall has changed sign three times without halving,
  as where its round trip jumps when a hold comes and goes, is settled on the side with trains to
  spare and stays so while its rate and round trip are unchanged, T-1283);
  extending track or changing congestion adjusts the headway while preserving the allocation.
  It never exceeds what the line's track can carry (6.2): trains beyond that wait at the ends.
  A loop's lap is a run and its closing stand; its round trip is twice the longer lap, and its count
  is even, half each way (the stepper moves by 2, headway mode doubles each way's count, an odd
  count from a save or an edit rounds up), so each way's trains keep the headway over their lap.
  Resources used by count schedules keep continuous delays so rounding cannot make the solve
  oscillate. Legacy saves preserve their original frequencies until a schedule is edited.
- **Headway mode** (T-625, `LineData::auto`, `Op::AutoTrains`): instead of a count per level a
  line can keep a headway per level, in seconds (0 is no service; the panel offers 1 minute to 4
  hours, `HEADWAY_MAX_S`: the day's shortest stretch of one demand level is 4 hours and the
  timetable lays at least one departure in each, so a longer headway would run more trains than it
  asks for; T-2519). Its counts are then derived: the fewest trains with round trip over headway, the round
  trip being the capacity-adjusted one with running, dwell, turnaround and waiting (`auto_trains`;
  the real headway may run 0.5% over the chosen one, so a few seconds of drift in the round trip
  buy no train). **The headway set is the headway run** (T-2607, `Network::running_trains`): where
  the fleet's round trip leaves time over at that headway (one train an hour on a 38 minute round
  trip, say), its trains wait it out at the ends, half at each, standing at the terminal platform
  (6.3 "Departures"), so the departures, the drawn trains, demand's rates and the running cost are
  the headway set's; the waits are not counted as terminal occupancy (6.2). Only a fleet short of
  the headway (the 0.5% above, or held by the track below) runs at its round trip over its trains.
  At the track's limit the count is clamped (T-679, `clamp_auto`): when the headway
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
- **Departures** (`Service::run_deps`): per stretch of one level (consecutive periods at the same
  level: high demand's 6-20 is one stretch, `level_stretches`, T-1536; every kind of day the same,
  T-2335) of the line's local time (T-309: its first
  stop's region's, `Lattice::shift`; a New York to Toronto line is on one offset, a London to
  Paris line would run on London's), the first departure half a one-way run before the stretch's
  start and then one a headway, until half a run before its end (T-835, `Lattice::lead`: the mean
  of the two runs at medium, at most 2 hours). So a train belongs to the period in force when it
  reaches the line's middle, which changes service on the hour; stops near the ends change up to
  half a run early or late. A loop's middle is half a lap from its first stop (the same figure), and
  both its runs leave A together. A stretch followed by no service (the evening before a line's night
  without trains, high demand before an empty evening) departs until its own end, so the line's last
  trains are not pulled early; it runs half a run longer at its level. Trains still run end to end.
  A whole-line fleet's reverse departures are offset by the forward trip so
  each train turns straight back; a headway line's by the forward trip and half the time its trains
  wait out (Headway mode above), so they stand that long at each end (T-2607). **Spaced from other lines** (T-1925, notes/T-1900.md D1): every
  departure of a level, both runs, leaves a phase after those times, under a headway
  (`Network::line_phase`, `Lattice::phase`), picked so the line's trains and those of each line it
  shares track, a platform or a flat junction move with reach the first place they meet as far apart
  as they can (`spacing::place`: per train of an hour, how far it falls short of half the smaller
  headway from the other line's nearest there, squared, summed; 24 phases tried a headway). A pair
timetabled together (6.2, "Behind slower trains", T-2419) is placed the other way: each of the
faster line's trains as near as it can be to its place after one of the slower line's (the share of
half the smaller headway it is off, squared): `s` before the next slower train, or for a pair that
passes at passing stations (T-2231) its overtake plan's offset. Greedy on
  edits: only the lines an edit changes directly are placed, in id order, against the others as they
  run, and a line keeps its phase unless another cuts that sum by more than 10%; so two lines on one
  track at one headway leave half a headway apart, and an edit elsewhere moves no one. Outside the undo
  history (a refused or trial edit puts the phases back), saved in a trailing `LPH1` block and in the
  eras (`ERA2`); a save without them has every line placed once, in id order, on load. Riders' waits
  are by headway, so demand does not move; it cannot keep an express from catching a local far down a
  long stretch (its slower running above and the display guard keep it trailing). When the pattern changes (a new period, or an edit) a new
  departure closer than half a headway after the old pattern's last one is left out, so trains
  never leave a terminal in pairs.

#### Trains on a running line

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
  soon enough, leaves service at the end of its turnaround. A train standing at a terminal when a
  change makes its run quicker (its turnaround then ends before the change) waits from the change
  instead, for up to one and a half new headways, and leaves service no earlier than the change
  (T-2429). So an added train enters at a
  terminal, a removed one finishes its trip and leaves at a terminal (on a loop, a train finishing
  a run takes that run's next departure from A, standing there as the timing hold, and trains enter
  and leave service at A only), and headways even out as
  trains turn; the same happens at every change of period. A train that had left service before
  a change never comes back. In steady service with a fleet set by count every train is ready at
  exactly its next departure and nothing stands longer; a headway line's stand their share of the
  time over at each end. Each drawn trip names the departure its
  train takes next (`Trip::next`, the fourth number per trip of `TrackApi.trips`, NaN when it
  leaves service), so a selected train is followed onto its next trip (T-245). Eras are dropped
  once no train of theirs can be drawn (an hour plus the longest trip and two headways after the
  change); until then a save carries them and the shifts (T-1071, [architecture.md](architecture.md)
  11), so a game saved mid-change loads with its trains where they were. Deterministic: the drawn
  trips are a function of the service, its eras and the hour asked for, and every window agrees.
- **Changing a train count** (T-229, notes/T-229.md). The stepper shows the new count at
  once (`game/world.ts` `pendingTrains`), and clicks made while an edit is in flight are folded
  into one edit sent when it lands. The headway stepper (`pendingHeadway`) and the train length
  stepper (`pendingCars`, T-1526) work the same way, each click counting from the length shown. The worker sends a service-only state for it: no geometry
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

#### Construction prices

Money is present, not tight, as in NIMBY Rails: construction, train purchase, operating cost per
train-km and car-km, a player-set fare curve by distance. One price list worldwide. Track costs a **base of
US$75M per route-km of double track** (single track 0.6x, high-speed track 1.5x, T-817; ramps at
the dearer level) times a
multiplier per level (Anita, 2026-11-02, T-1928, T-2017, notes/T-1897.md: round numbers, +1 and -1 at 1.0).
Ground is the cheap level, as open country is; in a city it pays for the roads it crosses (below). A
bored tunnel is near a cheap-country metro such as Korea's or Spain's, about
US$100-150M a tunnelled km with its stations (T-805):

| Level | +3 | +2 | +1 | 0 | -1 | -2 | -3 |
|---|---|---|---|---|---|---|---|
| Multiplier | 2.0 | 1.5 | 1.0 | 0.5 | 1.0 | 1.6 | 2.0 |
| US$M per km | 150 | 112.5 | 75 | 37.5 | 75 | 120 | 150 |

In a **sandbox** game [T-728] every price, fare and cost is the same, but no edit is refused for
money: it pays, and the cash may go below zero (the cash is not shown; docs/ui-panels.md).

- **Over or under water: track's multiplier +2.5** at any level (T-1890, T-1928: a +1 bridge is
  3.5x, a -2 tunnel 4.1x); stations, junctions and crossings over water cost 2x. **Level 0 cannot be built
  over water**; a ground-level segment drawn across it becomes a bridge (6.1). Water comes from a
  mask in the pack (T-030), and outside every pack's mask from the world's coarse mask (T-1285).
- **Long water** (T-1930, Anita 2026-11-02): a viaduct (+1 to +3, the bridge a ground segment
  becomes included) or a cut and cover tunnel (-1) is refused over more than 30 km of unbroken
  water (`WATER_SPAN_MAX`, `Issue::WaterTooLong`; Hangzhou Bay's rail bridge is 29 km), counted
  along the track and across edges that meet at a node in the water. Bored tunnels (-2, -3) have no
  limit, and a bored stretch ends the count. No depth is known, so there is no depth rule. Track
  already built is not refused again, but an edit touching it is checked.
- **Terrain** (T-1270, Anita 2026-10-25, design notes/T-1270.md; added, not multiplied, since
  T-1890): each zone's ground is in four classes, and track at elevation -1 to +3 adds the class's
  surcharge to its level's multiplier; bored tunnels (-2, -3) cost the same anywhere. So in
  mountains a tunnel is the cheapest way through, then ground track, then a viaduct. Over water
  only the water's +2.5 counts. A ramp is charged at the dearer of its two levels with the surcharge
  counted. Stations, junctions, crossings and upkeep take no surcharge; nothing is refused for
  steep ground; grades do not slow trains.

  | Class | Flat | Hills (low) | Hills (high) | Mountains |
  |---|---|---|---|---|
  | Mean 30 m slope over a cell | under 2.5° | 2.5-5.5° | 5.5-11° | 11° and over |
  | Ground round it (the cap) | under 1.25° | 1.25-2.5° | 2.5-5° | 5° and over |
  | Surcharge at -1 to +3 (`TERRAIN_SURCHARGE`) | +0 | +0.5 | +1.2 | +3 |
  | US$M a km at ground level (0.5 +) | 37.5 | 75 | 127.5 | 262.5 |
  | US$M a km on a +1 viaduct (1.0 +) | 75 | 112.5 | 165 | 300 |
  | US$M a km in a -2 tunnel (1.6) | 120 | 120 | 120 | 120 |

  The classes come from GEDTM30 (OpenGeoHub's bare-earth terrain model at 30 m, CC BY 4.0, T-1854)
  by `pipeline/zone_terrain.py` from the intermediate in `data/work/terrain-fine/`
  (`terrain_fine.py`): the mean slope of the 30 m pixels over each cell of a world lattice of 1/960
  degree (about 115 m north-south), classed, a 3 x 3 majority filter (drops single-cell specks, keeps
  a valley floor two cells wide), then capped at the class of the ground round it: the slope between
  the 1/240 degree mean elevations, about 920 m apart (second row), since 30 m bare earth still
  carries towers and raised roads that would read as hills in dense centres. Before any of that, tower
  fields are cut out of the 30 m surface (`terrain_fine.tower_ground`, T-1881): built-up land (GHSL
  built volume at 1 km, CC BY 4.0, at least 5 m of building over the cell's area) whose 30 m surface
  is rough (the rms about a 60 m smoothing, over about 150 m: from nothing at 5.5 m to all at 8.5 m),
  on ground whose low envelope (an opening of 600 m half-width, smoothed) slopes under 3-6°, takes
  that envelope's height. Midtown Manhattan stood 50-100 m above its streets in GEDTM30 and Chicago's
  Loop the same; built hillsides (San Francisco, Seoul, Wellington) are smooth or steep and keep
  their ground. One gzip file per zone
  (`data/terrain/<id>.terrain`, a median of 43 KB, 299 MB for all 1,988 zones, the largest 5.3 MB;
  `cost::ClassGrid`, two bits a cell in memory, at most about 23 MB). The clock worker fetches a zone's
  file with its water mask (zones.md 6), but no preview or edit waits for it: track drawn before it
  lands is priced on flat ground and priced again when it lands, the blueprint quote following;
  Construct waits for the files, water masks and world water under the blueprint, and is refused
  ("The terrain here did not load. Try again.") while one of them could not be fetched, so nothing is
  paid on a guess (T-2613; a 404 counts as no file, flat). A download gives up after 60 s; a failed one
  still needed by the view or the blueprint is fetched again 2 s later, doubling to 30 s (the status tag
  says so, ui-panels.md). Each edge keeps
  its spans of rough ground (`Network::edge_terrain`, sampled every 25 m, `TERRAIN_STEP`); a file
  landing reprices only the track inside its box. The track tool's tag gives the roughest class a
  segment crosses where it is charged with its surcharge ("Mountains +3.0x", T-1394, T-1890), and
  "Water +2.5x" when it crosses water (T-1887); its multiplier includes both, and the blueprint breakdown has a
  Terrain row per class with what the ground adds.
- **Road crossings** (T-1929, Anita 2026-11-02, notes/T-1929.md): track adds **US$6M for each road
  it crosses at level 0** (`ROAD_CROSSING_COST`; its height under half a level, so the low half of
  a ramp too), standing for the knockdowns and road works of a city. Roads are the basemap's zoom
  14 lines (OpenMapTiles `transportation`): trunk, primary, secondary, tertiary, minor and busway
  count 1, a motorway 3; service ways, tracks, paths, and any road on a bridge or in a tunnel count
  none. A divided road is two lines, so it counts twice. So ground track through a dense centre
  (11-13 crossings a km) costs about US$100-115M a km, more than a +1 viaduct, and in open country
  (under 1 a km) about US$40M. The clock worker fetches the tiles from the basemap's own server (the URL template the page
  sends, `roadFeed.ts`, decoded by `map/roadTiles.ts`) under each route preview and under the
  blueprint after a load, and puts each in the track model (`TrackApi::add_road_tile`,
  `track/roads.rs`: in Web Mercator tile units, indexed 32 x 32 cells a tile, never dropped in a
  session). Like terrain, each edge keeps its crossings' chainages (`Network::edge_roads`, the
  alignment sampled every 20 m, `ROAD_STEP`; none on a crossover or on track wholly half a level
  or more off the ground, which can never be charged). Each crossing is found in one tile, so a
  tile landing adds its own to the edges near it (found in the 1 km edge grid) and changes nothing
  else; a split, a reshape or a load works them out from the tiles in hand, nothing is saved.
  No preview waits for tiles: one priced before they land is asked again when they do, and
  Construct waits for the tiles under the blueprint and is refused ("The roads here did not load.
  Try again.") while one cannot be fetched, so nothing is paid on a guess (T-2613; a 404 or 204 tile
  has no roads). The blueprint quote has one "Road crossings" row, kind 7,
  the weighted count and price.
- The Build tab shows only the multiplier and price of the level currently selected, not the
  whole table.
- Stations (200 m, two platform tracks), US$M: a fixed 50 that every station has wherever it is
  (platforms, halls, systems), plus a part by level, 25 at grade, 25 + 15 a level up, 50 + 25 a
  level down: 75 at grade, 90/105/120 at +1/+2/+3, 125/150/175 at -1/-2/-3 (T-1267). So a ground
  station costs half a -2 one, where ground track costs under a third of -2 track. Scaled by
  `0.4 + 0.6 x length / 200`.
- Junction: 0.25 km of track at its level; a flyover adds 0.6 km one level away; flat
  crossing 0.1 km (cheap to build, dear in delay). Station upgrades (6.1, T-1281): a terminal
  1 km of track at the station's level (US$37.5M at grade, US$120M at -2, 2x over water), passing
  tracks the platform plus 0.4 km of it, 1.5x on high-speed track (US$22.5M for a 200 m station at
  grade, US$72M at -2; T-2231), park and ride US$20M, waiting room US$5M, a passage US$10M plus US$90M a km, a station building US$2,000 a m² of floor (T-2217). A crossover: two junctions, all told (6.1,
  T-338); the blueprint breakdown counts it as two junction switches.

#### Blueprint, trains and running costs

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
- **Removing constructed track refunds nothing**, and the game asks before doing it (T-096). Undo applies to blueprint edits, lines,
  schedules and removing constructed track or stations (T-1649: the undo puts back what was
  removed, free, charging only trains it needs). Constructing, making a constructed junction flying,
  lengthening a constructed platform and removing a constructed bus route are final (paid at once, or
  with no track to put back): they clear the undo history, so undo never builds anything for free.
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
- **Running cost** US$4 per train-km plus US$2 per car-km (trains an hour each way x line length at
  the level running that hour,
  times 4 + 2 x cars: a 4-car train US$12 a train-km, 9 cars US$22, 12 cars US$28; under real
  figures so fares can cover it, T-805, T-807, real figures in research/running-costs.md;
  `RUN_COST_TRAIN_KM` and `RUN_COST_CAR_KM`; high-speed trains US$5 a train-km plus US$3 a car-km,
  `HS_RUN_COST_*`, T-817, T-867). **Running costs follow the country** (T-2669, Anita 2026-11-06,
  notes/T-2668.md option N): these are US prices; a country's are them times (its GDP per head over
  the US's)^0.5 to a hundredth (`runScale`, game/income.ts: Mexico 0.39x, India 0.17x, Japan 0.63x;
  1 outside every zone), since running trains is mostly wages. A line pays the mean of its km's
  countries, each stretch between stops half to each stop's (its `run` shares, the same split as the
  region ledgers; `runFactorOf`, `lineRunFactor` in game/regionMoney.ts), and so does a bus or ferry
  line. The clock worker gives the track model each line's factor (`set_line_run_factors`, kept off the
  save and cleared on a load, then sent again), which drains the cash by it; the line panel, Lines
  tab, track hover and edit note take the same. Upkeep, construction and trains stay at world prices.
  The Money tab's Running trains hover adds "Running costs in Mexico: 0.39x" for the country it shows,
  and the line panel's train length hover "Running costs on this line: 0.39x", where not 1.
- **Upkeep** (T-867, notes/T-867.md): constructed track and stations cost money every game day
  whether trains run or not, draining evenly through the day: a route-km of double track US$35k
  on the ground, US$50k on a viaduct (+1 to +3), US$100k in cut and cover (-1) and US$130k in a
  bored tunnel (-2, -3) (T-1931; figures Anita's, T-2619, 2026-11-06, after research/upkeep-france.md),
  so a tunnel's gains are mountains and long water, a
  viaduct's cheaper upkeep. An edge pays the dearest structure between its lowest and highest level
  (a ramp into a tunnel makes the whole edge a tunnel's rate; a -1 to -2 edge pays bored). Times
  its price factor (`track_factor`: high-speed 1.5x, single 0.6x; quad is two double-track edges,
  so twice). A constructed station pays US$200k underground (its level -1 to -3, where its price takes
  the dearer step a level, 6.4 "Stations") and US$50k at grade or above (T-2619); crossovers, junctions and flyovers
  nothing on top; blueprint nothing (`UPKEEP_GROUND_KM`, `UPKEEP_VIADUCT_KM`, `UPKEEP_CUT_KM`,
  `UPKEEP_BORED_KM`, `UPKEEP_STATION`, `UPKEEP_STATION_UNDER`, real US$350, 500, 1,000, 1,300, 500 and
  2,000 times `ECONOMY`; `cost::track_upkeep_km`, `cost::station_upkeep`, `TrackApi::upkeep`; the app
  reads the two levels from `edges_info` and a station's from its own level). A line's share (`TrackApi::line_upkeep`): each edge's
  upkeep split between the running lines on it by their trains a weekday, each station's between
  the running lines stopping there; what no running line uses counts only for the network. Its
  profit (line panel, Lines tab) is fares less its running cost and that share. By region, each
  edge's upkeep counts where its midpoint is and each station's in its own region
  (`upkeepShares`, game/regionMoney.ts). The ledgers keep it as `upkeep` beside `running`. Shown
  in the Money tab's Upkeep row, an Upkeep row a day in the station panel (constructed stations)
  and the track inspector (not on crossovers), and "$X / day upkeep" under the Build tab's
  blueprint total (the pieces switched off left out); each hover states the rule. The panels' upkeep
  figures (station, track and the Build tab's) are in the alert red, as a cost [T-1893].

#### Fares and the ledgers

- Running costs, upkeep and **fares**
  are settled by the clock worker into the cash and a ledger per day (every day kept, T-2143) and per hour (48
  kept, T-447) as the game runs, not once an hour (T-284). The ledgers hold one entry per day or
  hour, in order; a day runs midnight to midnight in the start zone's local time, an hour is the
  clock's ([world.md](world.md) "Local time", T-2271); a load merges any day or hour an older save lists twice. Money time never goes
  back: a clock set behind the time already settled adds nothing until it passes it, and a charge
  meanwhile lands on the latest day. Both threads time the clock on `performance.timeOrigin +
  performance.now()`, never `Date.now()`, which drifts from it (T-488). Each clock message carries
  the load it follows, and the worker ignores one sent before the main thread saw the latest load,
  which still holds the previous game's time (T-715). A load settles again from the save's clock, so
  ledger entries after it are dropped and their net returned to the cash (the running cost worked
  out again on the loaded network at the rate the kept whole days show); edits' charges in them stay, on the clock's day. After a load the main thread sends the
  income, region shares and points again even when they match the last game's. **How much each
  game hour gets** (T-736, `app/src/game/moneyTiming.ts`): demand gives fares and rides per hour of
  each period and kind of day, the timetable a running cost per hour of each level, the same every
  day (`costParts` in the clock worker). An hour's rides
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
  (T-1277, T-1906, `app/src/game/income.ts`): each country charges the curve times its fare scale,
  by default its `incomeScale`: since T-2670 (Anita, 2026-11-06, notes/T-2668.md option N; T-1521 had
  India's income to the 0.7) the US's 11.6 times (its GDP per head at market rates over the US's)^0.5 in
  tenths (the UK 9.3, Japan 7.3, Mexico and China 4.6, India 2.0, Nigeria 1.4). The power is the minutes
  a dollar's too (docs/demand-mode.md), so a default fare weighs the same with riders everywhere and
  fares follow income more gently than T-1521's: a US start's Japan $1.55, Mexico $1.00, India $0.45.
  The US keeps 11.6, so every saved curve charges the US as before; a save's start key from before
  (`Fares.income` absent: set under 0.7) becomes its country's scale now (`startFromOld`; a US start's
  11.6 stays), and the fares carry `income: 0.5` from then on. A new game's curve is its start country's fares and
  it stores that country's income scale as `scales["@"]` (`START`), so a country's default is its
  income scale over that one (1 where it started). There the player's own scales are kept as multiples
  of the country's default; a save from before T-1906 has no `"@"`, its curve in India's units and its
  scales absolute (`scales["*"]`, `EVERY`, in a few saves of 2026-11-02, is one default for every
  country). **The Money tab shows each country's fares as a multiple of the reference country's**
  (T-2679, Anita): the country of the first built station (the lowest node id among constructed
  stations), kept in `Fares.ref` as soon as one is built, until the player makes another country the
  reference; with nothing built, the start zone's country, else the home city's. The reference reads a
  plain 1.00x; any other country its scale over the reference's (`shownScale`), so in a US-started game
  Mexico reads 0.40x and Japan 0.63x until set. **A multiple the player sets is kept to 0.01x** and
  stored so it charges exactly that many times the reference's scale (`storedScale`, T-2021), in fares
  charged, income and the demand sent to the sim alike. Making another country the reference changes
  only `ref`: no stored scale, fare or demand input changes (demand's network and the save's demand
  cache key leave `ref` out), so with Mexico the reference Mexico reads 1.00x and the US 2.52x, every
  fare the same to the cent. A reference country whose own scale was set before keeps it (its row and
  the curve show what it charges). A save without `ref` has each own scale rounded to 0.01x of the
  country's default as it loads (`tidyOwn`), so it charges what it did; from then on scales are kept as
  set (`keepOwn`). A new game's curve is set once from the country it starts in (`startNodes` in
  `game/fareDefault.ts`, T-1906): `START_NODES` (nodes at 0, 20 and 100 km at $0.0862, $0.2241 and
  $0.7757, one straight line, T-1670) times the factor giving the US $2.50, times the country's
  `incomeScale`, each fare rounded: the US $2.50, $6.50 and $22.50, Japan $1.55, $4.10 and $14. While
  that curve is untouched, a country without the player's scale charges its own round start curve
  (`roundedDefault`: a US start's Japan $1.55, not $1.5733).
  In a save from before, while a country's scale and the curve (the old start's) are untouched, its
  three fares are rounded (`game/fareDefault.ts`, T-1662). A save keeps its whole curve and is marked
  `basis: "usd"`; one from before T-1521 has its curve multiplied by `oldCurveFactor` and each
  scale the player set kept at its factor over the income's, so its home country's fares weigh
  exactly as before and the rest within the scales' rounding (`game/income.ts`). A save from before T-1277 carries no
  `baseline` and its curve, New York's fares, is divided by the home country's old scale as it loads.
  A save from before T-1521, or one with no nodes, takes its base and rate from its tidied (rounded)
  nodes as it loads, the same as the slot writes next, so its first reload's network is every later one's
  and the demand cache hits on it (`loadFares`, T-2559).
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
  pair's km, on the stretches they ride, each by its km, split as they split (`Loads::rn_fare_x`,
  each population's `fare_extra`, T-1516); the trip explorer shows each route's extra on its rides
  by km. So a trip that changes lines pays each line the base share of its boarding and the curve's
  fare a km on that line's km, and no line's fares go below zero. **High-speed fares** [T-1626]: a
  line with high-speed trains charges its rate and its share of the curve's extra times the
  high-speed multiplier (`Fares.hsr`, default `HSR_FARE` 1.5, 0 to 5, saved in the header; a save
  without one charges 1.5); the base stays once a trip; a ferry line likewise with the Ferry multiplier
  (`Fares.ferry`, default `FERRY_FARE` 1.00, `lineFareMul`, T-2195). So a trip pays base + (curve(km) - base) x
  (1 + (m - 1) x its share of km on high-speed lines). Demand takes it as each route node's
  multiplier (`set_fare_mul`, `Graph::rn_fare_mul`): ride edges charge their rate times it, a
  station pair's curve extra is times the strategy's expected high-speed-weighted km over its km,
  and the assignment spreads the extra over each km times its multiplier; the market's income
  groups weigh the same fares (`IntercityApi::fare_col`). Fare income takes the rate times it on
  that line's passenger-km. **Transfer fare** [T-1801]: `Fares.transfer` (curve units, absent for
  $0, saved in the header; an older save loads with none) is paid on each boarding after a trip's
  first, times the trip's country's scale, in route choice and mode choice (docs/demand-mode.md)
  and in fare income on the line boarded after the change (`fare_extra`, so inside `LineDemand.fareX`);
  the regional market's trips change free. Fare income is the first stretch's base on
  each of demand's trips and its rate on their passenger-km per period, plus demand's extra on the
  line it is counted at (`LineDemand.fareX`, `app/src/game/money.ts`; on a weekend the extra follows
  each stretch's riders), so demand and income charge the same trips the same fares. **The base's
  share between lines** [T-2104, `baseTrips`]: the network's base fares a day (every boarding times
  rail trips over boardings, each at its boarding stop's fare scale) go half by each line's boardings
  and half by its passenger-km, and no line takes more than one base a boarding, the excess going to
  the others by the same weights; buses and trains alike. So a short leg (a feeder bus) takes less
  than an even share of the base and a long line more, and the network's fares are the same as an
  even share per boarding would give. Each line's base goes over its periods by passenger-km. Changing the fare
  recalculates demand: higher prices can lose riders (T-067). Running costs can take the cash below zero; edits that cost nothing still
  work then. notes/T-028.md.

#### Regions, points and stars

- **Money by region** (T-531, `app/src/game/regionMoney.ts`): beside the network's, the worker
  keeps both ledgers per region, a region being a zone as `stationZone` places it (the Lines and
  Stations tabs' place), and "" for anything outside every zone, so the regions add up to the
  network. Settling divides each running line's hour: fares by the line's day of riders, the base
  part by the region of each boarding stop and the per-km part by passenger-km, each stretch
  between two stops half to each stop's region; running cost by route km, split the same way.
  A paid edit's construction goes to regions by the cost of the track it newly built in each
  (track midpoint's region; with no track, its new stations' regions, evenly; with neither, a station
  building's by the region of its outline's middle, T-2218); its new trains by the
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

#### Starting money, loan and the economy's scale

- **Starting money $10B** (T-805, Anita T-2271): about two real first lines, each a 20 km line at -2
  with 18 stations plus its trains (about $4.2B).
- **A fare edit pays when its riders show** (T-2266, `game/fareInEffect.ts`): the fare controls
  change at once, but a line's income keeps the fares its riders were solved with (`solvedFares`)
  until riders solved with the new ones show, whether the change raised or cut them; a solve
  overtaken by a later edit never switches them. Nothing of this is saved: a loaded game solves its
  riders with its saved fares.
- **The loan** (T-1661, T-2022, T-1906, `game/loan.ts`): one at a time, of $1B, $5B or $10B, at the
  bottom of the Money tab. It pays itself back in 60 equal daily payments, the same for every size:
  an annuity at 1% interest a game day on what is left ($22.2M a day for $1B, $1.33B in all),
  drained evenly through each day like upkeep and shown in the ledger and history as "Loan". A
  payment goes whatever the cash, which can go below zero as with upkeep. Repay pays off what is
  left (the payments still due less the interest they would carry) once the cash covers it; after
  the last payment or a repay another can be taken. A sandbox has none. The loan (its amount,
  payment and days left) is saved with the game; a save from before T-2022 starts its loan's 60
  days from the load. A loan is refused while its payment a day is more than the profit a day
  (fares minus running costs and upkeep) averaged over the last 7 whole game days, fewer in a
  younger game; before a whole day with fares, only the $1B loan can be taken.
- **Economy at 100x** (Anita, 2026-10-09, as Subway Builder does; T-081): fares and running costs
  per game day count 100 times the real figures (`ECONOMY` in `sim/src/track/params.rs`, the one
  constant; running costs scale in the track model, fares in `app/src/game/money.ts`), so a game
  day earns and spends what a hundred real days would. The fare the player sets and the US$4 a
  train-km plus US$2 a car-km shown (Money tab, line panel and train length hovers) are the figures
  before that scaling; upkeep is shown as the game day's figure (US$35k on the ground, real US$350); construction
  and train prices are not scaled.

### 6.5 Edits

The clock worker owns the network. The main thread sends edit operations (each with an inverse,
for undo); the worker answers with the new state, then the edit's answer (or the reasons it was
refused). The alignment being drawn is previewed by the worker as a trial edit, applied, measured
and rolled back, so the preview knows everything an edit would refuse (timings in notes/T-025.md);
the trial leaves the services alone (`TrackWorld::trial`, T-1779), as no preview shows a train's
run. A blueprint route that changes no line for demand sends no new trips or boarding hour (T-1780). The
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
track no line uses, or building stations no line stops at yet (T-2410), never wakes demand. Demand starts after edits pause (~0.5 s); a newer edit waits
for a running edit's solve to finish (T-1734, docs/demand-runtime.md) and cuts a load's solve short. An edit is refused only for problems it adds (radius, ramps, crossings,
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
  it takes whoever is waiting up to its crush load (240 a car, 150% of the rated 160, T-2186). The rest wait for the next train of
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
  between the clock worker's other messages (3 a step). New transfers (T-554) while that runs wait
  for its last step to send the hour, and the same transfers again send nothing; after the riders
  or transfers, the next hour's slice goes a task after this hour's (T-1810).
- **Each game hour** the worker sends, with the hour's trips, each trip's riders per stop-to-stop
  stretch and each platform's departures with the riders each left behind and when each train
  stopped there (its stand before it leaves, the closing stand at a first stop, never before the
  train ahead left; T-1123), its steady arrival rate, its pulses of riders changing lines (below),
  the hour's riders left behind and longest wait, and the stretches at capacity (below). The
  renderer reads the train's load for the stretch it is drawn on (past the rated load a second
  band, packed to crush with a red edge, T-207, T-218, T-2186), and a platform's crowd at any moment is what the last train to stop
  there left plus, once it has left, who has arrived since. While a train stands at a stop its
  riders move aboard evenly over the stand (T-1785): the platform falls linearly from the crowd it
  found to the one it leaves, and the train's load goes linearly from the stretch it came in on to
  the one it leaves on (held at the incoming until it stops). At its last stop everyone gets off as
  it arrives, and through its stand there the train fills linearly to its next trip's first
  stretch (`trainStands` in game/boarding.ts: the stand is the platform's, from when the train
  stopped to its departure). Only the drawing does this: the account, its boardings and the fares
  are unchanged. The crowd is drawn as the platform filling up
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
- **Crowded stretches** (T-2236; `RunAccount::hour_loads`, `Boarding::stretch_hours`, `period_peaks`): per
  stretch between two stops, direction and hour of the day, the mean riders aboard the trains leaving its
  first stop in that hour; as a share of the line's rated load (`LineSvc::rated`, 160 a car; a bus's or
  boat's is its crush), the fuller direction's. Worked out once with each account; the account's hour lists
  the stretches whose busiest hour is at `CROWDED_FROM` (1.005, past 100% as the hover rounds it) or more,
  with that load and hour, and the traffic view marks them by the load ([ui-map.md](ui-map.md) "Track
  display"); `line_periods` ends with each period's busiest hour per stretch, and `Boarding::window` with the
  period now's on the fullest stretch (the line panel's crowding dots and the Lines tab's train count).
- **Room on a platform** (T-219, `platformRoom` in `game/boarding.ts`): what counts as a full
  platform for drawing. Each direction's riders wait on one side, 2.5 m of usable width (half a
  6 m island platform less the edge strip kept clear) along the platform's length, at 0.3 m² a
  rider (why: DECISIONS-to-2026-10-19, T-219). A 150 m platform holds 1,250 a side. Only the drawing uses it: the
  account lets a crowd grow past it, up to `WAITING_CAP` (above).
- **Cost and how full trains get**: measured on the real New York save in notes/T-164.md,
  notes/T-554.md and notes/T-207.md.
- **What the panels show** (T-230): every train-fullness figure is the account's: the gauge on
  trains, the train tag and inspector, the line inspector's "Trains today" (`Boarding::line_summary`:
  fullest train, trains leaving full and when, riders left behind a day, longest wait), the Lines
  tab's train count and the schedule's crowding dots (the busiest hour's load past the rated load,
  `period_peaks`, T-2236) and the platforms. Demand's busiest-hour average train
  is not shown: riders arrive fastest mid-peak, so a line it puts under 100% can still leave riders
  behind (notes/T-230.md).

Riders left behind feed back into demand (T-188): each crowding round runs this rule on the
period's own trains and adds the average extra wait at each line, direction and stop to that
boarding's cost (docs/demand-kernel.md step 7), so a line that leaves people behind loses riders to
other routes or the car, and after the rounds no line keeps more riders than its trains hold at
crush in the busiest hour (T-1070). The account itself is not read back: the clock worker's day and demand's
periods give the same rule, but demand's is worked out inside the solve, so it settles with the
crowding rounds. There are no signals (SPEC 2.1).

### 6.7 Bus lines

A bus line (T-1464, T-1465; design notes/T-1445.md) is a line with no track: the player clicks points
and between two clicks the route follows the basemap's roads, motorways and their bridges and tunnels
included (T-2469); the app refuses a point they do not reach (T-2451, T-2467). It shares lines' ids, schedules (buses or a headway per
demand level), trips, the boarding account and the fare curve; a bus is one 12 m vehicle.

- **The route** is points, each with two bits: a corner (the player clicked it) and a road stretch
  (the stretch ending at it follows roads). The app lays the roads (ui-building.md, "Drawing a bus
  line"); the sim takes the points as they come. At most 8,000 points.
- **Road length**: a road stretch counts at its drawn ground length, a straight one (from a save made
  before T-2451, or the edit API) at 1.2 times it (`bus::ROAD_FACTOR`). It sets the run time, the price and the running cost; the bus is drawn along
  the route. Stops, crossings and each stop's stretch of route (`cover`) are in road metres.
- **Stops place themselves** (`bus::plan`), never saved: every train station within 200 m of the
  route (two within 100 m of each other along it: the nearer to the route), both ends (an end within
  200 m of a station is that station), a stop where it crosses another bus route, and about one
  every 2 km of road between (`k = ceil(gap / 2 km) - 1`, at least one between two stations 800 m
  or more apart), each on the grid of street stops every 400 m, which are drawn only. A constructed
  route stops at constructed stations and crosses constructed routes; a planned one previews all.
  Any edit to a station or a line places every bus line's stops again.
- **Run**: door to door (street stops and traffic included) at 70% of the car's speed on local streets
  where the bus is, at most 18 km/h (`streets.rs`, T-1577): the car's speed is demand's, 44 km/h in
  open country falling to 16.5 where homes and jobs are dense (half way at 5,000 a km² in the 2 km zone,
  demand-mode.md), so a bus does 18 outside the dense places and 11.5 in the densest. Each 100 m of road
  has its own speed and a hop between stops takes the time of the streets it crosses (`bus::Pace`). The
  street speeds come from the packs (below); a route in a zone with none runs at 18. The line's dwell
  stands at train stations, the line's turnaround at the ends (120 s for a new bus line). At each
  street stop (the drawn ones, `plan.street`) the bus stands 20 s (`bus::STREET_DWELL_S`, T-1798), taken out
  of the hop's time (at most 40% of it), so the bus runs faster between stands and arrives when it
  did before: run times and demand do not see the stands. No accelerating, no capacity, no holds.
- **Riders**: 80, rated and at crush alike (demand: 0.5 of a car's seats and rated 160, 22 seats;
  a bus takes nobody past its 80, T-2186); the boarding account keeps no
  platform crowd at its stops that are no station.
- **Street speeds**: a zone's pack gives its cells' people and jobs, the zone of each cell and the zones'
  squares and land, which the clock worker fetches as ranges of the `.bin` with the zone's mask (and
  for the zones a bus route is drawn or loaded in) and `TrackApi::add_pack_streets` turns into a grid of
  the car's local street speed per zone. A game's load waits for the zones under its bus routes, so its
  network comes from the save alone and a reload finds its demand cache (T-2499); otherwise nothing waits
  for them, bus lines are timed again when one lands. Not saved, carried over a load like the terrain.
  Every Range fetch takes the bytes it asked for whether the answer is a 206, the whole file or a 200
  holding the range alone (`workers/rangeFetch.ts`; Chrome's cache gives the last while the pack
  downloads whole for demand).
- **Money**: US$1M a road-km, paid on Construct with US$0.6M a bus, a fleet of its own with spares
  shared between bus lines; running US$6 a bus-km and upkeep US$30 a road-km a day (real figures,
  `ECONOMY` times in the game). Removing or redrawing a constructed route is final and refunds
  nothing. A new bus line starts at a bus every 8, 10 and 20 minutes by level. A copy is planned.
- **Saves**: a trailing `LBS2` block holds each bus line's route (points and their bits) and whether it
  is constructed; an `LBS1` block (T-1464) loads with every stretch straight.

### 6.8 Ferry routes

The router for ferry lines (T-2192; design notes/T-2177.md, which also holds the format the sim takes,
"F1 -> F2 route format"); the lines themselves are 6.9, the player's tool T-2194. All in the app (`ferry/route.ts`, in `workers/ferryWater.worker.ts`); the sim never
searches water.

- **The water** is the basemap's own: the `water` layer of its zoom 12 tiles, fetched from its tile server
  as the bus tool fetches roads, filled at cell centres (no swimming pools, water in tunnels or intermittent
  water). OSM's water stops at many canal bridges: along a `transportation` line marked as a bridge, land
  with water both ways across the line, 60 m or less in all, counts as water (a viaduct along a shore has
  water on one side only and stays land).
- **Ports**: a click within 200 m of the shore, on land or on the water (T-2468), gives a landing on the
  shore (where the line from the click to the nearest cell of the other kind changes kind) and a berth,
  the nearest water with 6 m of room within 300 m of it, where routes start and end. No shore within
  200 m: refused (`far-from-water` on land, `on-water` on the water).
- **A leg** between two berths is searched on a raster of its box (the two ends plus 35% of their distance,
  at least 800 m round, then 3 km or their distance round if the waters do not meet), its cell the finest
  of 2.5 m or more that keeps the raster under 1.5M cells. A cell is sailable if its centre is 5 m or more
  from land's (a channel of 10 m and a cell passes all along). A* over 16 neighbours; a metre costs
  1 + 2 (1 - room / 15 m)² where the room is under 15 m (narrow water keeps a boat to its middle), plus
  the 400 m preference: 3 (1 - room / 400 m)² times a weight, fading out within 300 m of either end. First
  the plain path (weight 0), then weights 3, 1 and 0.3, the first whose path is at most 1.3 times the plain
  one's taken, else the plain one.
- **Long legs**: where that cell would be over 5 m, the leg is planned on the coarse raster (its tiles at
  zoom 11 or 10 where zoom 12 would take over 80) and each end's first 700 m or so searched on a 2.5 m raster
  of its own, the coarse plan starting and ending where those reach, at their distance by water. Legs over
  150 km are refused (`too-long`); waters that do not meet, `no-water-path`.
- **Shape**: the cells are simplified (Douglas-Peucker at 25 m, or a quarter of the room, never off sailable
  water), then every corner but a port's gets the widest arc that fits its two sides and stays sailable, at
  most 300 m. Where pieces of a leg meet, two corners either side take the turn so the route passes straight.
- **Speeds**, one a point: 32 km/h 200 m or more from land (T-2450), 10 km/h at 20 m or less, straight between, at
  most √(0.3 m/s² x radius) on a bend; points every 25 m or so where the speed changes. Acceleration (0.15
  m/s²) and stops are the sim's.
- **Cost**: 0.2 to 1 s a leg in the browser once its tiles are in (`checks/t2192-ferry-routes.mjs`). Tiles,
  each leg's search and its rasters are kept, so moving one port searches only its two legs.

### 6.9 Ferry lines

A ferry line (T-2193; design notes/T-2177.md) is a line on water between ports, with no track. The app
searches the route on the basemap's water (T-2192) and hands the sim its points; the sim never sees the
water. The player's tool and panel are ui-building.md "Drawing a ferry line" and ui-inspectors.md
(T-2194), demand's side docs/demand-kernel.md step 4, "Ferries" (T-2195: a line like any other, its
ports stations, a boat two cars). Code: `track/ferry.rs`, exports `wasm/ferry.rs`.

- **On the map** (T-2194, `map/busLayer.ts`, whose rows a ferry line shares: the app's line row for a
  ferry is a bus row with `ferry` set): the route dashed in the line's colour on a continuous casing,
  half a rail line's width, under the track; a solid stroke from each port's landing to its berth; no
  corridor. Ports draw as stations. Boats are pointed hulls 35 m by 8 m in the line's colour, run 0's
  bow ahead (`BOAT_CAR`, network.ts). A ferry line's bullet is a pill.
- **The boat fleet** is in the game header (`fleetFerry`, absent for none), as `fleetBus`.

- **Ports** are stations of their own kind (`Network::is_port`): a node with no track, at level 0,
  with a name, walking access, the boarding account's waiting crowd, passages to stations and bus
  stops like any station (a bus route within 200 m stops there). Two points: the **landing** on the
  shore (the node, where riders walk to) and the **berth** in the water (`StationData::berth`, within
  320 m of it), where routes start, stop and end. No platform, track or upgrades (refused as
  `NotAtPort`; a berth out of reach as `PortBerth`). Any number of ferry lines share a port; a port a
  line stops at cannot be removed.
- **The route** is points in the network frame, each with a kind (0 route, 1 a port's berth, 2 a
  waypoint passed without stopping) and the most the boat may do there, m/s (kept on a mm/s grid,
  0.25 to 8.889). Its stops are the line's ports, in order, one per kind 1 point, each within 50 m of
  that port's berth; the first and last points are ports. At most 8,000 points, at least 100 m.
- **Run**: between two points the boat runs at most the lower of their two speeds; it accelerates and
  brakes at 0.15 m/s² (`ferry::BOAT_A`, 59 s and 263 m to reach 32 km/h), stops at every port, stands
  the line's dwell at the ports between its ends and the turnaround at the ends. Out and back like a
  bus line. No capacity holds. A boat is two train cars: 320 riders at the rated load, 480 at crush
(`BOAT_CRUSH`; demand's 2 cars, T-2195).
- **Money**: a port US$10M, paid on Construct (a constructed line redrawn to a new port pays for it
  then), upkeep US$100 a day (US$10k a game day); boats US$8M, a fleet of their own (the fourth,
  `fleet_ferry`) with spares shared between ferry lines; running US$40 a boat-km (real, `ECONOMY`
  times in the game). The route is free, and redrawing it is free; removing a ferry line leaves its
  ports and is undoable. The Construct quote lists ports as kind 12, boats as kind 13 and the route's
  metres (free) as kind 14. A new ferry line starts at a boat every 20 and 30 minutes at high and
  medium demand and none at low (T-2273; `NEW_FERRY_HEADWAYS` in the clock worker), 60 s dwell, 300 s
  turnaround (`ferry::FERRY_DWELL_S`, `FERRY_TURNAROUND_S`, which the app passes).
- **Saves**: a trailing `LFR2` block (save level 5) holds the ports' berths (as mm deltas from their
  landings) and each ferry line's route (points, kinds, speeds) and whether it is constructed. A save
  without it has no ports or ferries and loads as before. An `LFR1` block (level 3) is the same with
  its speeds searched at 25 km/h in open water: each speed over 10 km/h loads at the same share of the
  way from 10 to 32 (`ferry::v_from_lfr1`; a bend's limit rises with it, T-2450).
