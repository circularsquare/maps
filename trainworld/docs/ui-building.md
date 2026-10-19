[Back to the spec](../SPEC.md) · Interface: [panels](ui-panels.md) · [inspectors](ui-inspectors.md) ·
[the network on the map](ui-map.md) · [demand views](ui-demand-views.md) · **building**

# 8. Interface: building

The rules behind these tools (geometry, levels, stations, costs, edits) are in
[track-and-service.md](track-and-service.md); this file is how the player works them.

## The Build tab

[T-100; `ui/tabs.tsx`, `ui/Blueprint.tsx`; tools in `map/tools.ts`]

- **Tools**: draw track (key 1), station (2), delete (3), select blueprints (5, below), three
  buttons on the first row and one on the second; then the elevation (seven buttons
  tinted in their Elevation colours, the chosen one solid with a dark outline, a light one in the dark theme, T-478, T-692) with its
  multiplier and price per km, double or quad track [T-255; single track not offered since T-733],
  and the platform length for new stations (a row under the track choice, no heading; the label's
  hover says it is for new stations). There is no crossover tool: the game places
  crossovers past quad stations [T-733, [track-and-service.md](track-and-service.md) 6.1].
  Everything the player reads says elevation, never level (code keeps `level`) [T-526].
  Beside keep placing, the **snap switch** (a magnet, "Snap track to the real network") [T-727]: while
  it is on and the track tool is in hand, the real network is drawn at full strength
  ([ui-map.md](ui-map.md) "Real network"); the snapping itself is T-727d (notes/T-727.md).
  The elevation, the double or quad choice, the keep placing switch and the snap switch are remembered per
  browser; a remembered single (from before T-733) reads as double (localStorage with the other panel settings, not in the save) [T-468]. Caps Lock only
  changes the switch when its state changes after the page loads, so a stored setting wins over a
  Caps Lock left on or off [T-487].
  The number keys work from any tab and open the Build tab; the key of the tool already on does
  nothing. Each button shows its key. Three letter keys work from any tab [T-670, `ACTION_KEYS` in
  `ui/tabs.tsx`]: C constructs all blueprints (the same as "Construct blueprints" below, nothing when
  there is nothing to build), N makes a new line and L starts or finishes editing the selected line's
  stops; none fires while a text field has focus or a question is open. There is no button for plain selecting: clicking the active tool again or
  leaving it with Esc returns to selecting. The trip tool (key T) is in
  [ui-inspectors.md](ui-inspectors.md).
- **At the bottom of the pane**, fixed while the settings above scroll: the blueprint total, undo
  and redo, and "Construct blueprints", which builds every blueprint that is not disabled
  ([below](#disabling-blueprint)). The total is red when it is more than the cash on hand [T-527]. It includes the new
  train cars that constructing starts, after spare cars. An expandable cost breakdown shows track
  by priced level, single or double, land or water and ramps (charged at the dearer end), stations,
  junction switches, flyovers and flat crossings, with lengths, counts and prices from the track
  model. The fleet quote previews the constructed service, so lines started together include their
  resulting delays, without changing the game.
- **Undo**: Ctrl+Z; redo Ctrl+Y or Ctrl+Shift+Z. What undo covers: [track-and-service.md](track-and-service.md)
  6.4.
- **Esc** undoes one thing at a time: a question waiting is answered Keep, a drag goes back, a
  half-drawn route is dropped (the tool stays on), then the tool is left for selecting, then the
  selection clears.
- Tool hints and a refused edit's reason show at the map's bottom left.

## Drawing track

[T-023, T-085, T-092, T-244, T-311, T-322, T-418, T-422, T-442]

Track is drawn through points, double and quad alike [T-418, T-423; Corners retired with
T-422]. The geometry is in [track-and-service.md](track-and-service.md) 6.1.

- **Clicks**: the first click is the start, the second places a segment ending at it at once (one
  undo step), the track passing through the click, at the level selected. A click with Shift held
  places the segment and starts the next one at its end; without Shift the tool is ready for a new
  start. The **keep placing** button (an icon beside the Double and Quad choice) does the
  same for every click until it is switched off, and so does Caps Lock, which is the same switch
  (the button shows it) [T-445]. A double click, Escape or Enter ends the run (the first click of
  the double places the last segment); the button stays on. A click on the start (within 6 px, or
  on its node) does the same at any later time: it ends the run, or drops a start that has no
  segment yet [T-707]; a segment never ends where it starts, so no loop is lost. To carry on later, click the loose end
  (on quad, either pair's). Backspace while chaining undoes the last segment and draws on from its
  start. A segment may turn
  up to 350 degrees, so a loop or hairpin is one click [T-442]. A click whose segment cannot be
  built (a curve under 100 m, a click straight behind the start, water, too little room for a
  ramp) is refused by the edit, its reason at the bottom left; the ghost was red already. While
  placing, the circle the segment's curve lies on is drawn faint grey (red when refused)
  [`map/curveCircle.ts`]; none for a straight segment. A click beyond the 1,796 m circle gets that
  circle's curve then a straight to the click [T-471].
- **Curve labels**: while drawing, the start shows as a square and the curve its speed limit, on
  the speed colour at half strength [T-417; no radius since T-530] (one label for a curve split
  into several corners); the same labels show while a selected blueprint stretch is reshaped or a
  node dragged (`setCurveLabel`, `map/tools.ts`).
- **Loose ends** [T-416; `map/endMarkers.ts`]: while the track or station tool is on, a small white
  circle with a dark border marks every loose end (a node with one piece of track and no
  station), blueprint or built. A track click on one starts or continues the route from it; the
  station tool's pointer snaps to it, so the station goes at the line's end. The markers take no
  presses, and show from zoom 9.
- **The tag by the cursor** [T-411, T-526] has two lines: the segment's length, average cost
  multiplier and price ("1.5 km, cost 1.33x, $12.8M"), then the elevations the segment passes
  and its lowest speed limit with the track tag's icons and colours ("↕ 0 +2  » 67 km/h": where
  it starts, a bridge's top or a dip's bottom, where it ends, each once [T-707]),
  `map/trackTagRows.ts`; [ui-map.md](ui-map.md) "Hover tags on the network"). The multiplier sits
  on yellow above 1x, deeper up to 2.5x, plain at 1x and below. It is the level's multiplier
  (6.4: 0.3x at ground, 0.8x at +1) and water's, averaged over the track priced, both pairs on
  quad, whose price is two double tracks' [T-711]. Where the segment could not be
  built ending at the cursor, the reason comes first, in red, and the cost line is left out. The
  build tools' tags are drawn over the bubbles' tags; the track tag shows above the pointer when
  it is on existing track.
- **Shift**: with a tool on, MapLibre's shift-drag zoom is off so a shift-click reaches the tool.
- **The ghost** is solid with a dark casing: white in the Line and traffic views, coloured by the
  Elevation or Max speed view from the preview's fitted geometry; red when invalid.
- **Placed blueprint track** is drawn as blue rails ([ui-map.md](ui-map.md) "Lines and track").
- A segment is one piece (each pair's on quad); the delete tool, Construct and the track
  inspector act on one piece [T-244].

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
  the list. Where 7 or more candidates are under the pointer and the worker has not yet confirmed
  the guess (a bundle of overlapping tracks, where the guess was often wrong on the real save), the
  ghost stays as it was if it was within 10 px, else shows nothing, until the answer. The click
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
  St"; nothing before the first), then roughly how many local trips a day start or end within 800 m
  (half a mile in US units): "About 12k trips a day start or end within 800 m", two significant
  figures, "Under 50" below that. 800 m is a 10-minute walk at the model's 6 km/h and 1.3 detour.
  The count is the Homes and Destinations bubbles' legs (all purposes) over the cells whose centres
  are inside the circle, from the zone in view only; a trip with both ends inside counts twice.
  A refused spot shows only the reason. Two thin rings at 1 km and
  2 km of ground distance, each labelled at its top, follow the spot under the ghost
  (`stationRings` in `map/tools.ts`, an SVG layer, fixed grey ink at 35%).
- How a new station is named: [track-and-service.md](track-and-service.md) 6.1. A double click on
  the name in its inspector renames it.

## Crossovers

[T-338, T-733; rules in [track-and-service.md](track-and-service.md) 6.1] The player does not place
crossovers. A station placed on quad track, or doubling up through stations, adds one past each
station in the same edit; the station ghost does not draw it, and it shows on the map as blueprint
once placed, its price in the blueprint total and in the double-up button's price. A crossover is
selected, constructed and removed whole like other track (below). The clock worker still answers
`previewCrossover` and the `crossover` edit, for checks.

## Deleting

[T-096, T-244, T-338; `game/remove.ts`] Hovering shows what a click removes: a stretch of track
drawn red (one piece of a drawn route, or "the crossover", both diagonals drawn red since it goes whole), a station or a
flyover ringed, its name by the cursor. Blueprint is
removed at once (undoable); anything constructed only after a yes in the confirmation dialog
("Remove 1.2 km of constructed track? Nothing is refunded." with Remove and Keep; Esc is Keep). The
inspectors' Remove buttons ask the same way. Lines are removed from the line panel.

## Disabling blueprint

[T-333; `map/blueprintBox.ts`, `game/blueprintPick.ts`, `ui/Blueprint.tsx` `BlueprintPick`] The
Select tool (key 5, a dashed box icon) picks blueprint track and blueprint stations so they can be
disabled: Construct and the blueprint total leave a disabled piece out, and it is drawn grey
([ui-map.md](ui-map.md) "Lines and track"). It is a tool of its own because a plain drag on the map
is the pan and Shift+drag is MapLibre's zoom box, so a box needs a mode.

- **Dragging the pick** [T-475]: a left drag that starts on picked track or within 9 px of one
  of its nodes (a grab cursor there) moves every movable node of the pick (the ends of picked
  track and picked stations; not a crossover's ends) by the same offset, previewed as a node drag
  is. Track between two moved nodes keeps its shape; track to a node that stays bends as in a node
  drag (`TrackApi.move_nodes`). One undoable edit. A press there without a drag is a click.
- **Left drag** elsewhere draws a box. It picks the blueprint track it touches and the blueprint stations
  inside it (stations only from zoom 9, where they are drawn); the pick follows the box live.
  **Shift+drag adds** to the pick. A click picks the piece under the pointer, Shift+click adds it or
  takes it out again, a click on nothing clears the pick. Constructed track and stations are never
  picked.
- **Right or middle drag moves the map** while the tool is on; the wheel and arrow keys work as
  always. The map's own left drag is off for the tool's time.
- **The selection bar** [T-353; `BlueprintBar` in `ui/Blueprint.tsx`] sits at the top centre of the
  map while the tool has a selection (on the demand buttons' row on a map column over 1350 px, so it
  clears the status tag beside them [T-349]; one row lower below that, lower again below 680 px; it
  appears and goes in one step): "N selected" (its hover lists track and stations), Disable and
  Enable as buttons with icon and word (keys D and E in their hovers), each greyed when it would
  change nothing, and Clear. The Build tab shows only "N disabled" while any blueprint is disabled,
  whatever the tool. Disabling and enabling are one undoable edit each (Ctrl+Z). Esc clears the
  pick, then leaves the tool; leaving also clears it.
- Disabled is a state on the piece, saved with the game ([track-and-service.md](track-and-service.md)
  6.4). The piece's own Construct button (track and station inspectors) and a line's Construct build
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
  interval; it can put all corners at one elevation ("Elevation of the stretch"). It has Construct
  and the elevation buttons for its piece [T-244], and Remove. On double track (blueprint or
  built) the title carries **Double up** [T-708; `DoubleUp` in `ui/inspector/others.tsx`], an
  icon of two rails and a plus: a click adds the express pair beside its run as blueprint
  ([track-and-service.md](track-and-service.md) 6.1 "Doubling up"). Its hover gives the price, or
  why not, greyed (a pair beside already).
- **On a crossover's diagonal** [T-391] the track inspector is titled "Crossover" and drops the
  corners and elevation controls; it shows the slowest curve, the price of both diagonals, Construct
  (builds both) and Remove crossover.
- **The junction inspector** [T-031, T-080] draws the junction as a small diagram with its moves,
  says in plain words that it is flat, and offers "Build a flyover" with its price and the waiting
  it saves ([track-and-service.md](track-and-service.md) 6.2), except at a crossover's end, where
  the flyover offer is left out [T-391]. A junction's capacity tag opens it.

## Making and editing lines

[T-109 to T-112, T-233, T-237, T-239; `map/stopMarkers.ts`] A line is made from the Lines tab
(New line, name it, click stations; [ui-inspectors.md](ui-inspectors.md)) and its stops edited from
the line panel (Edit stops). Map clicks test every insertion position and take the shortest valid
route, keeping the existing stop order; removing a stop leaves the others in order.

**Stop editing mode** (Edit stops, New line, Copy line):

- The map hides the demand views (bubbles, inter-region bubbles, the trip route,
  capacity tags, traffic widths and station circles) and draws the other lines and their trains at
  30%; the edited line's trains stay full.
- A round toggle marker on every station, over everything: a stop is a disc in the line's colour
  inside a white ring inside a dark one; other stations are hollow. Stations a stop can be added at
  without changing the line's path (on track the route passes over, or past either end on track
  the line can carry on over, T-237) are drawn on top at 80% of a stop's size, white inside a
  plain grey ring with none of the line's colour (T-476); stations reachable
  only by changing the route are 65% and grey; unreachable ones (no track, or only by turning back)
  the same size and greyer. In the dark theme the greys are dark discs with lighter rings that fade
  into the land, the unreachable ones dimmest (CSS tokens `--mk-*`, T-543). The tiers come from the clock worker trying each station with the same
  insertion search a click makes (`TrackApi::stop_tiers`), so a marker always says what its click
  does; until the answer lands an estimate from the track graph shows.
- Hover says "Click to add a stop", "Click to add a stop and change the route", "Click to remove the
  stop" or "No track joins it to this line", under the station's name; a click on the last says
  why in its own tag for four seconds. The markers take no pointer events (the map's pick handles clicks), so the wheel
  and dragging work over them.
- A panel at the map's lower left shows "Editing stops on <line>: N stops" and Done; Enter and
  Escape also leave. Red removal crosses show in the stop list while editing.
- **Stations passed** [T-646, `TrackApi::passed_stations`]: while editing, the stop list shows,
  between two stops whose stretch of the line's own path runs through other stations, a row
  "+ 2 stations passed" (its hover names them); with stations passed in more than one gap, "Add
  all 5 stations the line passes" heads the list. A click shows those stations in the list as
  faint "+ Name" rows, in path order, with Add and Cancel under the list; Add makes them stops in
  one undoable edit. Only stations on the line's path count: a quad twin on the pair beside, or a
  station on a branch, is not offered.
