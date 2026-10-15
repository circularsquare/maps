[Back to the spec](../SPEC.md) · Interface: [panels](ui-panels.md) · [inspectors](ui-inspectors.md) ·
[the network on the map](ui-map.md) · [demand views](ui-demand-views.md) · **building**

# 8. Interface: building

The rules behind these tools (geometry, levels, stations, costs, edits) are in
[track-and-service.md](track-and-service.md); this file is how the player works them.

## The Build tab

[T-100; `ui/tabs.tsx`, `ui/Blueprint.tsx`; tools in `map/tools.ts`]

- **Tools**: draw track (key 1), station (2), crossover (6, below), delete (3), select blueprints
  (5, below), three buttons on the first row and two on the second; then the level with its
  multiplier and price per km, single, double or quad track [T-255], and the platform length for
  new stations.
  The number keys work from any tab and open the Build tab; the key of the tool already on does
  nothing. Each button shows its key. There is no button for plain selecting: clicking the active tool again or
  leaving it with Esc returns to selecting. The trip tool (key 4) is in
  [ui-inspectors.md](ui-inspectors.md).
- **At the bottom of the pane**, fixed while the settings above scroll: the blueprint total, undo
  and redo, and "Construct blueprints", which builds every blueprint that is not disabled
  ([below](#disabling-blueprint)). The total includes the new
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

[T-023, T-085, T-092, T-244, T-311, T-322]

- Clicks place corners at the level selected (rules in [track-and-service.md](track-and-service.md)
  6.1). While drawing, the clicked points show as squares and each curve its radius and speed
  limit. **The tag by the cursor** [T-411] has two lines: the stretch's length, average cost
  multiplier, price and tightest curve ("1.5 km, cost 1.33x, $25.5M, 401 m curve"; no curve, no
  radius), then the level being placed and the route's lowest speed limit ("Level +2, max speed
  67 km/h"). The multiplier sits on yellow above 1x, deeper up to 2.5x, plain at 1x and below; the
  level on its Height colour and the speed on its Max speed colour (`speedColour`,
  `game/palette.ts`), each at half strength behind dark text. Where the route could not be built
  ending at the cursor, the reason comes first, in red, and the cost line is left out.
- **A refused click** [T-410]: a click that adds a corner where the route, if it ended there,
  could not be built (a curve under 100 m, ground track over water, too little room for a ramp) is
  refused with the reason at the bottom left, and the tag adds "Shift-click to place it anyway".
  The answer is the hover's preview for that spot, or one asked for the click when the pointer's
  answer is for another spot; a shift-click places the corner without asking. Clicks that finish on
  track or a node, and the first click, are not checked here (the edit itself refuses those). With
  a tool on, MapLibre's shift-drag zoom is off so the shift-click reaches the tool.
- **The ghost** is solid with a dark casing: white in the Line and traffic views, coloured by the
  Height or Max speed view from the preview's fitted geometry; red when invalid.
- **Placed blueprint track** is drawn as blue rails ([ui-map.md](ui-map.md) "Lines and track").
- A finished route is one piece per corner; the delete tool, Construct and the track inspector act
  on one piece [T-244].

## Placing stations

[T-065, T-311, T-312]

- **The ghost** is the platform's full box at its true length at every zoom: blue (`#2f6fe0` fill,
  `#1c4fb8` outline) where a click would place it and red (`#e03c3c`, `#b91f1f`) where it would be
  refused, fixed colours with a thin white halo; a refused spot shows the reason in the tag. The
  answer comes from the clock worker trying the real edit (`TrackApi.preview_station`; one request
  in flight, asked when the spot, platform length or network changes, the last answer kept
  meanwhile), so the ghost and the click cannot disagree.
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

## Placing crossovers

[T-338; `map/tools.ts`, rules in [track-and-service.md](track-and-service.md) 6.1] The crossover
tool (key 6, an X between two rails) snaps to track only, never to a node. With the pointer on
track the ghost is the crossover's X as the click would place it, drawn like the track tool's
ghost (red where it would be refused), and the tag says its price ("$49.5M") or why not ("A
crossover needs other track alongside within 15 m, with 60 m of plain track each way."). The
answer comes from the clock worker trying the real edit (`TrackApi.preview_crossover`, one request
in flight). A click places it as blueprint, one undoable edit; a click off track says to click on
track beside other track.

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

- **Left drag** draws a box. It picks the blueprint track it touches and the blueprint stations
  inside it (stations only from zoom 11, where they are drawn); the pick follows the box live.
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
  dragged, except a node at a crossover's end. The cursor is a move cursor over a node that can.
  A drag previews the new shape with each curve's radius and speed (or why not); letting go is one
  undoable edit, Escape puts it back.
- **The track inspector** lists the corners with each curve's speed, a radius stepper (100 to 1,500
  m, then auto), a level stepper and Remove; Add corner inserts one midway along the selected
  interval; it can put all corners on one level. It has Construct and the level buttons for its
  piece [T-244], and Remove.
- **On a crossover's diagonal** [T-391] the track inspector is titled "Crossover" and drops the
  corners and level controls; it shows the slowest curve, the price of both diagonals, Construct
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

**Stop editing mode** (Edit stops, New line, Copy line, Express copy):

- The map hides the demand views (bubbles, catchment, inter-region bubbles, the trip route,
  capacity tags, traffic widths and station circles) and draws the other lines and their trains at
  30%; the edited line's trains stay full.
- A round toggle marker on every station, over everything: a stop is a disc in the line's colour
  inside a white ring inside a dark one; other stations are hollow. Stations a stop can be added at
  without changing the line's path (on track the route passes over, or past either end on track
  the line can carry on over, T-237) are drawn on top at 80% of a stop's size; stations reachable
  only by changing the route are 65% and grey; unreachable ones (no track, or only by turning back)
  the same size and greyer. The tiers come from the clock worker trying each station with the same
  insertion search a click makes (`TrackApi::stop_tiers`), so a marker always says what its click
  does; until the answer lands an estimate from the track graph shows.
- Hover says "Click to add a stop", "Click to add a stop and change the route", "Click to remove the
  stop" or "No track joins it to this line", under the station's name; a click on the last says
  why in its own tag for four seconds. The markers take no pointer events (the map's pick handles clicks), so the wheel
  and dragging work over them.
- A panel at the map's lower left shows "Editing stops on <line>: N stops" and Done; Enter and
  Escape also leave. Red removal crosses show in the stop list while editing.
