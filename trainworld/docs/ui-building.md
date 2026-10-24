[Back to the spec](../SPEC.md) · Interface: [panels](ui-panels.md) · [inspectors](ui-inspectors.md) ·
[the network on the map](ui-map.md) · [demand views](ui-demand-views.md) · **building**

# 8. Interface: building

The rules behind these tools (geometry, levels, stations, costs, edits) are in
[track-and-service.md](track-and-service.md); this file is how the player works them.

## The Build tab

[T-100; `ui/tabs.tsx`, `ui/Blueprint.tsx`; tools in `map/tools.ts`]

- **Tools**: draw track (key 1), station (2), delete (3), select (5, [below](#the-select-tool)), in a
  2 by 2 grid [T-1128]; then the elevation (seven buttons
  tinted in their Elevation colours, the chosen one solid with a dark outline, a light one in the dark theme, T-478, T-692) with its
  multiplier and price per km (with High speed picked, high-speed track's: 1.5x, so 0.45x at
  ground), Double, Quad or High speed track [T-255, T-817; no single track, T-733],
  the target max speed (a row under the track choice: a stepper through 160, 200, 250 and 320 km/h, 160
  by default; higher lays the curves of newly drawn track wider, [track-and-service.md](track-and-service.md)
  6.1; its hover: "Long track pieces are laid out straight enough for this speed") [T-817, T-1127]; picking High
  speed sets it to 320 (it can still be stepped down) and picking Double or Quad again puts back
  the speed from before [T-817], and the platform
  length for new stations (the next row, no heading; the label's hover says it is for new stations). There is no crossover tool: the game places
  crossovers past quad stations [T-733, [track-and-service.md](track-and-service.md) 6.1].
  Everything the player reads says elevation, never level (code keeps `level`) [T-526].
  Beside keep placing, the **snap switch** (a magnet, "Snap track to the real network") [T-727]: while
  it is on and the track tool is in hand, the real network is drawn at full strength
  ([ui-map.md](ui-map.md) "Real network"); what it does to drawing is [below](#snapping-to-the-real-network).
  The elevation, the double, quad or high speed choice, the target max speed, the keep placing switch and the snap switch are remembered per
  browser (localStorage with the other panel settings, not in the save) [T-468]; keep placing is on
  by default [T-942].
  The number keys work from any tab and open the Build tab; the key of the tool already on does
  nothing. Each button shows its key. Three letter keys work from any tab [T-670, `ACTION_KEYS` in
  `ui/tabs.tsx`]: C constructs all blueprints (the same as "Construct blueprints" below, nothing when
  there is nothing to build), N makes a new line and L starts or finishes editing the selected line's
  stops; none fires while a text field has focus or a question is open. There is no button for plain selecting: clicking the active tool again or
  leaving it with Esc returns to selecting. The trip tool (key T) is in
  [ui-inspectors.md](ui-inspectors.md).
- **At the bottom of the pane**, fixed while the settings above scroll: the blueprint total, undo
  and redo, and "Construct blueprints", which builds every blueprint that is not disabled
  ([below](#the-select-tool)). The total is red when it is more than the cash on hand [T-527]. It includes the new
  train cars that constructing starts, after spare cars. An expandable cost breakdown shows track
  by priced level, single, double or high-speed, land or water and ramps (charged at the dearer end), stations,
  junction switches, flyovers and flat crossings, with lengths, counts and prices from the track
  model. The fleet quote previews the constructed service, so lines started together include their
  resulting delays, without changing the game.
- **Undo**: Ctrl+Z; redo Ctrl+Y or Ctrl+Shift+Z. What undo covers: [track-and-service.md](track-and-service.md)
  6.4.
- **Esc** undoes one thing at a time: a question waiting is answered Keep, a drag goes back, a
  half-drawn route is dropped (the tool stays on) with any ghost or real route still being worked
  out for it [T-1039], then the tool is left for selecting, then the selection clears. Leaving the
  track tool any other way (another tool, a game loaded) drops the route too.
- Tool hints and a refused edit's reason show at the map's bottom left ([ui-panels.md](ui-panels.md) "The dock").

## Drawing track

[T-023, T-085, T-092, T-244, T-311, T-322, T-418, T-422, T-442]

Track is drawn through points, double and quad alike [T-418, T-422, T-423]. The geometry is in [track-and-service.md](track-and-service.md) 6.1.

- **Clicks**: the first click is the start, the second places a segment ending at it at once (one
  undo step), the track passing through the click, at the level selected. A click with Shift held
  places the segment and starts the next one at its end; without Shift the tool is ready for a new
  start. The **keep placing** button (an icon beside the Double and Quad choice) does the
  same for every click until it is switched off; it is on by default, and each press of Caps Lock
  flips it, whatever the lock's own light shows (the button shows the switch) [T-445, T-942]. A double click, Escape or Enter ends the run (the first click of
  the double places the last segment); the button stays on. A click on the start (within 6 px, or
  on its node) does the same at any later time: it ends the run, or drops a start that has no
  segment yet [T-707]; a segment never ends where it starts, so no loop is lost. To carry on later, click the loose end
  (on quad, either pair's). Backspace while chaining undoes the last segment and draws on from its
  start. A segment may turn
  up to 350 degrees, so a loop or hairpin is one click [T-442]. A click whose segment cannot be
  built (a curve under 100 m, a click straight behind the start, water, too little room for a
  ramp) is refused by the edit, its reason at the bottom left; the ghost was red already. A
  free-drawn segment reaches 500 km at most, straight line between its ends: past that no ghost is
  drawn, the tag says so and a click is refused, so a run left open in one city never lays track to
  the next one the player pans to (real routes, T-727, are not capped) [T-1020]. While
  placing, the circle the segment's curve lies on is drawn faint grey (red when refused)
  [`map/curveCircle.ts`]; none for a straight segment. A click beyond the 1,796 m circle gets that
  circle's curve then a straight to the click [T-471].
- **Curve labels**: while drawing, the start shows as a square and the curve its speed limit, on
  the speed colour at half strength, no radius [T-417, T-530] (one label for a curve split
  into several corners); the same labels show while a selected blueprint stretch is reshaped or a
  node dragged (`setCurveLabel`, `map/tools.ts`). They show only for curves on screen and never
  over one another (64 by 22 px), at most 300, so a long snapped route zoomed out keeps a few
  [T-727d].
- **Loose ends** [T-416; `map/endMarkers.ts`]: while the track or station tool is on, a small white
  circle with a dark border marks every loose end (a node with one piece of track and no
  station), blueprint or built. A track click on one starts or continues the route from it; the
  station tool's pointer snaps to it, so the station goes at the line's end, its platform running
  back along the track through the joints of a drawn route as a platform on a piece does [T-860].
  The markers take no presses, and show from zoom 9.
- **The tag by the cursor** [T-411, T-526] has two lines: the segment's length, average cost
  multiplier and price ("1.5 km, cost 1.33x, $12.8M"), then the elevations the segment passes
  and its lowest speed limit with the track tag's icons and colours ("↕ 0 +2  » 67 km/h": where
  it starts, a bridge's top or a dip's bottom, where it ends, each once [T-707]),
  `map/trackTagRows.ts`; [ui-map.md](ui-map.md) "Hover tags on the network"). The multiplier sits
  on yellow above 1x, deeper up to 2.5x, plain at 1x and below. It is the level's multiplier
  (6.4: 0.5x at ground, 1.0x at +1) and water's, averaged over the track priced, both pairs on
  quad, whose price is two double tracks' [T-711]. Where the segment could not be
  built ending at the cursor, the reason comes first, in red, and the cost line is left out. The
  build tools' tags are drawn over the bubbles' tags; the track tag shows above the pointer when
  it is on existing track.
- **Shift**: with a tool on, MapLibre's shift-drag zoom is off so a shift-click reaches the tool.
  In the select tool with demand bubbles on, Shift+drag draws a bubble box instead [T-865].
- **The ghost** is drawn as blueprint track (dark blue rails round a light blue middle) with a thin
  outline outside the rails (black; the map's ink, near white, in the dark theme; `ghost` in
  `game/shadeTable.ts`) [T-1054]. The middle is coloured
  by the Elevation or Max speed view from the preview's fitted geometry; red when invalid. It is never
  thinner than 4 px across its rails with 1 px of outline either side, so a route hundreds of km long
  shows in full zoomed out [T-1038]. The same holds
  for every placement ghost: a dragged corner or node, a moved pick, a crossover, a station's own
  piece of track.
- **Waiting** [T-1039]: when a click or the newest ghost has waited on the clock worker more than
  300 ms, a small spinner leads the tag.
- **Placed blueprint track** is drawn as blue rails ([ui-map.md](ui-map.md) "Lines and track").
- A segment is one piece (each pair's on quad); the delete tool, Construct and the track
  inspector act on one piece [T-244].

## Snapping to the real network

[T-727d; `map/realSnap.ts`, `workers/realRoute.worker.ts`; design notes/T-727.md] With the snap
switch on, the track tool follows the real network the layer draws.

- **The ring**: within 12 px of a drawn real line (the layer's own rendered features), the ring
  sits on the line where a click would snap. A first click on open ground there starts on the line;
  a first click on the player's own track starts there as always, and the route begins from the
  real track under the click.
- **The route**: with the start within reach of real track, the pointer resting 120 ms near real
  track asks the router worker for the route. It is the cheapest over the graph files
  (`packs/realgraph/<cc>.rgr`: the drawn track, and short undrawn track joining it at two points or
  more, at most 3 km a group, so a tunnel portal or station throat no line runs over leaves no gap
  [T-816]), every chain within the snap radius (plus 20 m) a candidate at
  either click, its distance from the click counted twice; where the two clicks' candidates share
  no family (one on the metro under a main-line station, the other on rail), each click also takes
  the other's family within 400 m [T-923]; no turn sharper than 60 degrees where
  two chains meet; the other family (metro or rail) than the first click's nearest track at three
  times its length, so a subway click follows the subway and an intercity route keeps off it;
  high-speed track (flag 8 in the graph) at 0.8 times its length, so London to Paris keeps to HS1
  and the LGV by Lille (493 km) over the shorter classic line by Amiens [T-838]; otherwise a metre
  costs a metre. The length shown and built is the real one. A long way round is allowed: the
  player sees it in the ghost before clicking [T-839]. A
  region's graph is fetched when the view, a click or the search first reaches it (a border
  junction loads the neighbour's file, joined by OSM node id). The search stops after 2 s or 3
  million states. The real line is drawn as the ghost, then the corners fitted to it at 25 m
  (`TrackApi.trace_start`, run in the router worker's own track model in steps of a few ms, with a
  look at its messages every 25 ms so a newer rest, or the pointer leaving real track, drops a search
  or fit still running) are previewed by the clock worker and replace it with the usual tag. A search
  or fit over 300 ms shows the small spinner in the tag. Changing the elevation or double and quad fits the route again.
- **The second click** places the route as one segment: one edit, one undo step, Shift or keep
  placing carrying on from its end. A click before the route is ready waits for it. It ends on the
  player's own track when the pointer is on it, else at the real line's snap point.
- **What the start reaches** [T-839]: from the first click until the segment is placed or dropped
  (Escape, a click on the start, the snap switched off), the real track a route from the start
  could run over is drawn over the layer in the layer's colours at 0.9 opacity and a little wider,
  and the rest of the layer drops from 0.4 to 0.25. Reachable is the search's own moves without
  its costs: every chain from the start's candidates on through junctions with no turn over 60
  degrees, either family, the whole joined graph with no distance limit. The router worker works it
  out once per start and sends the chains crossing the view and a quarter of it around, vertices
  1.5 px apart and chains under that left out, again on each move or zoom; a region with no graph,
  or one not yet loaded (the regions at the start and up to four in view are loaded first),
  counts as unreachable. Timings: notes/T-839.md.
- **Nothing joins them** (two networks that never meet, or the search ran out): a red straight
  ghost and "No real track joins these points" in the tag; the click places nothing and says so.
- **Away from real track** the tool draws an ordinary segment.
- **Where no graph exists** [T-913] the tool places track as with the switch off: no ring, no
  search, no reach. The router worker sends the page the outlines (noritetsu's `regions.json`) of
  the regions listed in `packs/realgraph/index.json`; a point inside one, or within 0.05 degrees of
  one (outlines are coarse at coasts and borders), is covered. Until that answer comes nothing is.
  Every noritetsu region has a graph (118) but Jordan and Mauritius, which have no drawn track.

## Placing stations

[T-065, T-311, T-312]

- **The ghost** is the platform's full box at its true length at every zoom: blue (`#2f6fe0` fill,
  `#1c4fb8` outline) where a click would place it and red (`#e03c3c`, `#b91f1f`) where it would be
  refused, fixed colours with a thin white halo; a refused spot shows the reason in the tag. On
  quad track the twin's platform shows too [T-708; `pairBeside`, `map/stationGeometry.ts`]: on
  the nearest track 4-12 m beside the spot running the same way, level with it, as the track
  model places the twin.
  **First frame** [T-695]: the page works out at once, from what it already has, where the click
  would go and whether it would be taken (`likelyFirst`, `layoutIssues` in `map/stationSnap.ts`),
  and shows that: a node with a station or a junction is refused; a spot is refused when the
  platform (half its length each way, all of it from a dead end node) and a station's reach beyond
  come within 1 m of the next station, junction or track end, a drawn route's plain joints within
  reach joined through when the pieces match (built or blueprint, tracks, not a crossover), as the
  track model joins them; or when a flat crossing (the map's marks) is under the platform. Then the
  clock worker tries the real edit
  (`TrackApi.preview_station`) and its answer replaces the guess; a refused spot stays where the
  guess put it. The worker's last answer for each track is kept for the network and platform
  length it was given in: a refusal the layout cannot explain (water, a ramp, a crossing that is
  not flat, a curve, cash) stands for spots on that track within 16 px of where it was tried, and
  that track is not asked again there, so while the pointer moves each answer gets further down
  the list. Where 7 or more candidates are under the pointer (a bundle of overlapping tracks) and
  the worker has not yet confirmed the guess, the ghost stays as it was if it was within 10 px, else shows nothing, until the answer. The click
  tries the ghost's spot first, so the ghost and the click cannot disagree.
- **What is under the pointer** [T-469, T-671]: the ghost and the click take the first spot the
  track model would accept (the same trial edit as above), tried in this order: the nodes within
  12 px (nearest first, at most 3), then the tracks within 4 px of the nearest, then the rest within
  10 px (at most 5 tracks). Among the near tracks the order is: one where the station would not meet
  a station (none within the two platforms' reaches along the track, up to a junction), blueprint
  over built, the track the last station of the session went on (from its node along the track to
  the next junctions), then the nearest (`map/stationSnap.ts`). A spot that would be refused never
  wins over one that would be accepted; when none is accepted the ghost shows the first one's
  reason.
- **The end of the track** [T-671]: a spot refused only because the platform would run off the end
  of its track (`PlatformTooLong`, `PlatformJoint`) is tried again at the node at the nearer end of
  that track, as when the pointer is on the end node. Any other refusal (water, a crossing, a
  station) stays and says why.
- **Placement help** [T-312, T-412]: with the pointer snapped to track the tag says how far the
  spot is, in a straight line, from the station placed last in this session ("1.2 km from W 46th
  St"; nothing before the first), then roughly how many local trips there are within 1 km
  (1 mile in US units): "12k trips within 1km", two significant figures, "Under 50" below that.
  The count is the day shown. It is the Homes and Destinations bubbles' legs (all purposes) over the cells whose centres
  are inside the circle, from the zone in view only; a trip with both ends inside counts twice.
  A refused spot shows only the reason. Two thin rings at 1 km and
  2 km of ground distance, each labelled at its top, follow the spot under the ghost
  (`stationRings` in `map/tools.ts`, an SVG layer, fixed grey ink at 35%).
- **On open ground** [T-794, T-834, T-907, T-917; `freeSpan`, `freePiece`, `freeGhost` in `map/tools.ts`]:
  where the pointer is not within reach of track or a node (above), a station is placed like track,
  in two clicks: it lays a straight blueprint piece of track one platform long (plus 10 m each end)
  at the selected elevation, double, quad or high speed as set, with the station at its middle, one
  undoable edit (rules: [track-and-service.md](track-and-service.md) 6.1). The first click is one
  end of the platform. Before it there is no ghost, only the snap ring at the pointer. After it the
  platform, at the set platform length, runs from that end towards the pointer, and the second
  click (anywhere, track or not) sets that direction and places it. Within 8 px of the first end
  it keeps the heading of the last station placed this way (east at first). Esc after the first
  click drops it and the tool stays on. The ghost
  is the piece's track drawn as the track tool's (red when refused) with the platform box on it
  (and the twin's on quad), the 1 and 2 km rings round its middle, and the tag (by the pointer) has
  the lengths and the price, "200 m platform and 220 m of track, $16M", over the placement help
  lines for the middle. The clock worker tries the
  same edit and rolls it back (`TrackApi.preview_free_station`), so the ghost and the click agree;
  a refused spot shows the reason instead.
- How a new station is named: [track-and-service.md](track-and-service.md) 6.1. A double click on
  the name in its inspector renames it.

## Crossovers

[T-338, T-733; rules in [track-and-service.md](track-and-service.md) 6.1] The player does not place
crossovers. A station placed on quad track, or doubling up through stations, adds one past each
station in the same edit; the station ghost does not draw it, and it shows on the map as blueprint
once placed, its price in the blueprint total and in the double-up button's price. A crossover is
selected, constructed and removed whole like other track (below). The clock worker's
`previewCrossover` and `crossover` edit are for checks only.

## Deleting

[T-096, T-244, T-338; `game/remove.ts`] Hovering shows what a click removes: a stretch of track
drawn red (one piece of a drawn route, or "the crossover", both diagonals drawn red since it goes whole), a station or a
flyover ringed, its name by the cursor. Blueprint is
removed at once (undoable); anything constructed only after a yes in the confirmation dialog
("Remove 1.2 km of constructed track? Nothing is refunded." with Remove and Keep; Esc is Keep). The
inspectors' Remove buttons and the Select tool's Remove ask the same way. A removal that changes a
line asks too, built or not [T-859]: a row per line under the question, its bullet then what
happens ("subway 4 will lose its stop at Nevins St and stop running until you fix its route."),
from the track model's trial of the removal (`TrackApi.removal_lines`). Lines are removed from the
line panel.

## The Select tool

[T-333, T-815, T-829; `map/blueprintBox.ts`, `game/blueprintPick.ts`, `ui/Blueprint.tsx`
`BlueprintBar`] The Select tool (key 5, a dashed box icon) picks several pieces at once to disable,
re-level or remove them. A pick is of one kind (Anita, T-829): one that holds any blueprint takes
only blueprint, track and stations; one with no blueprint takes constructed track and constructed
stations. Disabled blueprint is left out by Construct and the blueprint total, and drawn grey
([ui-map.md](ui-map.md) "Lines and track"). It is a tool of its own because a plain drag on the map
is the pan and Shift+drag is MapLibre's zoom box, so a box needs a mode.

- **Dragging the pick** [T-475]: a left drag that starts on picked track or within 9 px of one
  of its nodes (a grab cursor there) moves every movable node of the pick (the ends of picked
  track and picked stations; not a crossover's ends) by the same offset, previewed as a node drag
  is. Track between two moved nodes keeps its shape; track to a node that stays bends as in a node
  drag (`TrackApi.move_nodes`). One undoable edit. A press there without a drag is a click.
- **Left drag** elsewhere draws a box. It picks the track it touches and the stations inside it
  (stations only from zoom 9, where they are drawn; a junction goes with its track, never on its
  own), blueprint if the box holds any, else constructed; the pick follows the box live.
  **Shift+drag adds** to the pick, and the sum keeps one kind by the same rule (constructed pieces
  drop out once any blueprint is in). A click picks the piece under the pointer (a station first,
  blueprint over constructed), Shift+click adds it or takes it out again, a click on nothing clears
  the pick. A piece that is constructed or removed leaves a blueprint pick.
- **Right or middle drag moves the map** while the tool is on; the wheel and arrow keys work as
  always. The map's own left drag is off for the tool's time.
- **The selection bar** [T-353; `BlueprintBar` in `ui/Blueprint.tsx`] sits at the top centre of the
  map while the tool has a selection (on the demand buttons' row on a map column over 1350 px, so it
  clears the status tag beside them [T-349]; one row lower below that, lower again below 680 px; it
  appears and goes in one step) is the pick's inspector: "N selected" (its hover says blueprint or
  constructed and lists track and stations), the picked track's total length, and the actions.
  For blueprint: the seven elevation buttons tinted as in the Build tab (the one all the picked
  track is on solid), then Disable and Enable as buttons with icon and word (keys D and E in their
  hovers), each greyed when it would change nothing. For either kind: Remove (key Delete) and
  Clear. The Build tab shows only "N disabled" while any blueprint is disabled, whatever the tool.
  Disabling and enabling are one undoable edit each (Ctrl+Z). Esc clears the pick, then leaves the
  tool; leaving also clears it. The bar is about 580 px wide with the elevation buttons and keeps
  260 px clear each side of the map, wrapping onto more rows where the map is narrower [T-841].
- **Elevation** [T-815; `TrackApi.set_edges_level`]: a button puts every picked blueprint stretch
  on that elevation, one undoable edit: each piece's corners as the track inspector's "Elevation of
  the stretch" does for one, and every node all of whose track is picked (a joint, a loose end, a
  junction, a blueprint station with its platform), so a picked run has no ramps at its joints. A
  node with other track keeps its elevation and the picked piece ramps to it. Pieces that cannot
  change are left as they are: a crossover, a piece with no corners whose two ends cannot both
  move, and pieces the track model refuses there (a ramp with no room, a crossing between levels)
  or that would make a flat crossing not there before; those are tried group by group (pieces
  joined through moving nodes), then a refused group's pieces with corners one by one with their
  ends kept. The message says how many changed ("9 of 12 pieces set to elevation +1. 3 could not
  change."), or that none could.
- **Remove** [T-829; `TrackApi.delete_pieces`]: everything picked goes in one edit, with the nodes
  left bare and a crossover whole. Blueprint goes at once (one undo); constructed asks first, as
  for one piece ("Remove 3.1 km of constructed track and 2 constructed stations? Nothing is
  refunded."), and is final.
- Disabled is a state on the piece, saved with the game ([track-and-service.md](track-and-service.md)
  6.4). A selected disabled blueprint track says "Disabled" under "Blueprint" in its inspector
  (hover: Construct leaves it out until it is enabled) [T-821]. The piece's own Construct button (track and station inspectors) and a line's Construct build
  it anyway, which clears the state.

## Reshaping blueprint

[T-064, T-071, T-092, T-093; `map/edgeEdit.ts`, `map/nodeDrag.ts`; rules in
[track-and-service.md](track-and-service.md) 6.1]

- With the select tool, a selected blueprint stretch shows its corners as squares (not a
  crossover's, which keeps its shape [T-338]); any blueprint track end, junction or station can be
  dragged, except a node at a crossover's end. The cursor is a grab cursor over a node or square
  that can, and a grabbing one anywhere while dragging [T-474]. A draggable node takes the drag
  over a demand bubble. A click on a node's square (drawn over the bubbles) acts on the point, never
  the bubble [T-486]; a blueprint station has no square, so a click on one under a bubble selects
  the bubble.
  With the select tool every blueprint track end, corner and junction without a station shows the
  same white square as a corner [T-416; `map/endMarkers.ts`], so draggable points can be seen.
  A drag previews the new shape with each curve's speed (or why not); letting go is one
  undoable edit, Escape puts it back.
- **The track inspector** lists the corners with each curve's speed, a radius stepper (100 to 1,500
  m, then auto), an elevation stepper and Remove; Add corner inserts one midway along the selected
  interval; "Elevation of the stretch" puts the piece on one elevation: all its corners, or for a
  straight piece with none its ends, as the selection bar does for one piece (`edgesLevel`; a
  notice says when it cannot change) [T-841]. It has Construct
  and the elevation buttons for its piece [T-244], and Remove. On double track (blueprint or
  built) the title carries **Double up** [T-708; `DoubleUp` in `ui/inspector/others.tsx`], an
  icon of two rails and a plus: a click adds the express pair beside its run as blueprint
  ([track-and-service.md](track-and-service.md) 6.1 "Doubling up"). Its hover gives the price, or
  why not, greyed (a pair beside already). The title names the track: Double track, Single track
  or High-speed track [T-817]; high-speed track has no Double up. A **Speed limit** row above
  Slowest curve gives the track's cap, 160 km/h or 320 on high-speed track, its hover saying which
  trains run up to what [T-817]; the slowest curve on high-speed track counts up to 320.
- **On a crossover's diagonal** [T-391] the track inspector is titled "Crossover" and drops the
  corners and elevation controls; it shows the slowest curve, the price of both diagonals, Construct
  (builds both) and Remove crossover.
- **The junction inspector** [T-031, T-080] draws the junction as a small diagram with its moves,
  says in plain words that it is flat, and offers "Build a flyover" (a button under its header) with its price and the waiting
  it saves ([track-and-service.md](track-and-service.md) 6.2), except at a crossover's end, where
  the flyover offer is left out [T-391]. A junction's capacity tag opens it.

## Making and editing lines

[T-109 to T-112, T-233, T-237, T-239; `map/stopMarkers.ts`] A line is made from the Lines tab
(New line, name it, click stations; [ui-inspectors.md](ui-inspectors.md)) and its stops edited from
the line panel (Edit stops). Map clicks test every insertion position and take the shortest valid
route, keeping the existing stop order; removing a stop leaves the others in order.

**Stop editing mode** (Edit stops, New line, Copy line):

- The map hides the demand views (bubbles, inter-city bubbles, the trip route,
  capacity tags, traffic widths and station circles) and draws the other lines and their trains at
  30%; the edited line's trains stay full.
- A round toggle marker on every station, over everything: a stop is a disc in the line's colour
  inside a white ring inside a dark one; other stations are hollow. Stations a stop can be added at
  without changing the line's path (on track the route passes over, or past either end on track
  the line can carry on over, T-237) are drawn on top at 80% of a stop's size, white inside a
  plain grey ring with none of the line's colour (T-476); stations reachable
  only by changing the route are 65% and grey; those reachable only by the line's trains reversing
  at a stop or a bare track end where they did not before (T-797) are the same, with a dashed ring;
  unreachable ones (no track, or only by turning back at a station passed or a junction) the same
  size and greyer. In the dark theme the greys are dark discs with lighter rings that fade
  into the land, the unreachable ones dimmest (CSS tokens `--mk-*`, T-543). The tiers come from the clock worker trying each station with the same
  insertion search a click makes (`TrackApi::stop_tiers`), so a marker always says what its click
  does; until the answer lands an estimate from the track graph shows on entering, and after an edit
  the last answer stands with only the stops changed (T-648). A click on a marker is one edit
  (`insertStop`): the worker fits the station in where `TrackApi::insert_stop` puts it, among the
  stops it holds then.
- Hover says "Click to add a stop", "Click to add a stop and change the route", "Click to add a
  stop. Trains reverse at <station>" (`stop_tiers` gives tier 4 with the station; "at the end of
  the track" for a bare track end), "Click to remove
  the stop" or "No track joins it to this line", under the station's name; a click on the last says
  why in its own tag for four seconds.
- **Loops** (T-798): the first stop's marker says "Click to close the loop" (tier 5) while the line
  has two or more stops and track leads round from the last stop back to it without the train going
  back the way it came; the click appends the first stop (`setStops`). On a loop it says "Click to
  open the loop" (tier 6) and the click takes the closing stop off. The closing stop's cross in the
  stop list also opens it ("Open the loop"). New stops on a loop go between its first stop and the
  closing one (`best_insert`). The markers take no pointer events (the map's pick handles clicks), so the wheel
  and dragging work over them.
- A panel at the map's lower left shows "Editing stops on <line>: N stops" and Done; Enter and
  Escape also leave. Red removal crosses show in the stop list while editing.
- **Reversals** (T-797): once a stop makes the trains reverse, the map's large reversal icon
  ([ui-map.md](ui-map.md)) appears over that station, or the bare track end they reverse at, at
  once, so an unintended reversal shows; the stop list marks a reversing stop with a small copy of
  the icon (hover: the stand and the limit of 12 an hour a side; a bare end is no stop and has no
  row), and the round trip gains a "Reversing partway" row counting both.
- **Stations passed** [T-646, `TrackApi::passed_stations`]: while editing, the stop list shows,
  between two stops whose stretch of the line's own path runs through other stations, a row
  "+ 2 stations passed" (its hover lists them: "Add Foo, Bar and Baz") [T-814]; with stations passed in more than one gap, "Add
  all 5 stations the line passes" heads the list. A click shows those stations in the list as
  faint "+ Name" rows, in path order, with Add and Cancel above the list, where "Add all" was [T-855]; Add makes them stops in
  one undoable edit. Only stations on the line's path count: a quad twin on the pair beside, or a
  station on a branch, is not offered. The passed stations come in the clock state with each line
  (`ClockLine.passed`), so the list never shows new stops with the old gaps (T-648).
