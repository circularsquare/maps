[Back to the spec](../SPEC.md) · Interface: [panels](ui-panels.md) · [inspectors](ui-inspectors.md) ·
[the network on the map](ui-map.md) · [demand views](ui-demand-views.md) · **building**

# 8. Interface: building

The rules behind these tools (geometry, levels, stations, costs, edits) are in
[track-and-service.md](track-and-service.md); this file is how the player works them.

## The Build tab

[T-100; `ui/BuildTab.tsx`, `ui/Blueprint.tsx`; tools in `map/tools.ts`]

- **Tools**: draw track (key 1), station (2), delete (3), select (4, [below](#the-select-tool)), in a
  2 by 2 grid [T-1128], and under them Bus (U, [below](#drawing-a-bus-line)) [T-1514] and Ferry (O).
  The tools' options are on the map while the tool is on ([below](#the-track-and-station-bars)), so
  the tab holds the tools, the "N disabled" count and the blueprint [T-2282]. There is no terrain
  switch here: Settings > Show > Terrain is the one ([ui-map.md](ui-map.md)) [T-1394, T-2282].
  The tool letters work from any tab and open the Build tab; the key of the tool already on does
  nothing. Each button shows its key. Three more letter keys work from any tab [T-670, `ACTION_KEYS` in
  `ui/toolKeys.ts`]: C constructs all blueprints (the same as "Construct blueprints" below, nothing when
  there is nothing to build), N makes a new line and S starts or finishes editing the selected line's
  stops; none fires while a text field has focus or a question is open. Full key list: Keys and focus in
  [ui-panels.md](ui-panels.md#keys-and-focus). There is no button for plain selecting: clicking the active tool again,
  leaving it with Esc or the descriptor's cross returns to selecting. The trip tool (key T) is in
  [ui-inspectors.md](ui-inspectors.md).
- **The blueprint**, under the tools and taking the rest of the pane: the blueprint total, undo
  and redo, and "Construct blueprints", which builds every blueprint that is not disabled
  ([below](#the-select-tool)). The total is red when it is more than the cash on hand [T-527]. It includes the new
  train cars that constructing starts, after spare cars.
  Under the total and its upkeep line, above Construct, is the **Cost breakdown** [T-2269], collapsed
  by default (it opens and closes with its heading, 13 px and semibold over rows of 12 px with the
  parts' names in medium weight, T-2339; past the pane's height the breakdown scrolls within it).
  Open, it lists the blueprint by **part**: track joined by
  blueprint track (crossovers included), biggest price first, each a heading row named by the first
  and last stations met walking it from an end ("Alpha St to Charlie Sq"; one station's name; "Track"
  with none). A part is that one line until its chevron opens it [T-2339]; each part opens and
  folds on its own, all folded at first, and stays open across edits while the breakdown is open. A
  part with more than one row has the chevron, and open it shows its rows in track order: each blueprint
  station ("Alpha St", its upgrades and passages in its price) and each **stretch**, the track
  between two stations, junctions or track ends, named by its ends ("Alpha St to Bravo Av",
  "Alpha St to a junction", "Bravo Av to a track end", "Bravo Av to built track"; an end within
  250 m of a station takes the station's name, since track drawn from a station starts at its
  platform's end), with its length and price (its flat crossings in it); then one "2 junctions"
  row for the part's junction switches, flyovers and crossovers. Quad track's twin rows (the same
  names, ends within 80 m) are one row. A blueprint station on built track only is a part of its
  own. Prices come from the track model piece by piece (`blueprint_piece_costs`, kind, two ids,
  US$M and whether the quote counts it; `game/blueprintStretches.ts` groups them); the parts' counted
  prices plus the rows below add to the total. Each heading, station and stretch row has a checkbox:
  unticking switches its pieces off (the part's checkbox, every piece of the part; one undoable
  edit), which is the Select tool's Disable, so the pieces are drawn grey, left out of the total
  and of Construct, and saved switched off [T-333]; a row all switched off is grey and shows its
  full price, a row partly off shows what is still counted. Hovers: "Disable this stretch" /
  "Enable this stretch" (and "this station", "all joined track" on a heading), a stretch's level
  ("Viaduct (+1)", "Levels −1 to +1") and its flat crossings, a heading's stations and length.
  The pointer over a row highlights its track on the map with the selection's ink band (a station,
  the track at it; `bpHover`); a click on a row moves the camera to fit it (`locateNodes`). After
  the parts come the trains, bus routes, buses, ports, boats and ferry routes as kind rows, then a
  collapsed **By kind**, the whole quote by kind as below. The pieces are asked of the clock worker
  only while the breakdown is open. By kind shows track
  by priced level, single, double or high-speed, land or water and ramps (charged at the dearer end), stations,
  junction switches, flyovers and flat crossings, with lengths, counts and prices from the track
  model. Each row reads one line, "3.7 km viaduct (+2)" and its price, and its hover is the price worked out, one item a line: "Base cost $90M per km", then "Length × 3.7 km", "Viaduct × 1.3", "Over water × 2", "Hills × 1.25", "Single track × 0.6" or "High-speed track × 1.5" for each factor that is not 1, and "= $430M" [T-1839]. Track rows are split by level, track type, water and terrain class (the sixth quote field, `blueprint_cost_items`; ramps are not told apart, being charged at the dearer end; there is no separate terrain row); stations, junctions, flyovers, crossings, cars, buses, passages and upgrades keep their own rows with the same kind of hover, and the roads crossed at ground level are one "Road crossings" row, "12 road crossings" (hover "Base cost $10M per road crossing", "Road crossings × 12", "= $72M") [T-1929, T-2017]. The fleet quote previews the constructed service, so lines started together include their
  resulting delays, without changing the game. The quote, and a planned line's Construct price in
  its panel, are asked of the clock worker only while a panel shows them, 250 ms after the last
  edit (`game/quotes.ts`, T-1171); a state carries neither.
- **Undo**: Ctrl+Z; redo Ctrl+Y or Ctrl+Shift+Z. What undo covers: [track-and-service.md](track-and-service.md)
  6.4.
- **Esc** undoes one thing at a time: a question waiting is answered Keep, a drag goes back, a
  half-drawn route is dropped (the tool stays on) with any ghost or real route still being worked
  out for it [T-1039], then the tool is left for selecting, then the selection clears. Leaving the
  track tool any other way (another tool, a game loaded) drops the route too.
- A refused edit's reason and what an edit did ("Blueprint added: ...") show at the map's bottom
  left, under the mode's descriptor ([ui-panels.md](ui-panels.md) "The dock"). The build tools (track, station, delete, select)
  show no instructions there (Anita, T-1159); stop editing and the trip tool keep their one-line hints.

## The track and station bars

[T-2282; `ui/ToolBar.tsx`; design notes/T-2267.md] The track and station tools' options are in a bar
on the map while the tool is on, in the selection bar's box and place: top centre, lower on narrower
maps as the selection bar goes (`.blueprint-bar` rules), and on a phone the map's bottom edge. Only one
of the three bars shows at a time; the other modes have none. The track and station bars have a 2 px top
edge in their mode's colour (`CLICK_MODE_COLOURS`, as the aura's) [T-2446]. Their text is 12 px, as
the map's other controls', and the level buttons' numbers 15 px (also in the phone's pop-up) [T-2522, T-2600].

- **The track bar**, two rows. First "Level" (muted), the seven level buttons, the size they had in
  the Build tab, tinted in their Level colours, the chosen one solid with a dark outline, a light one
  in the dark theme [T-478, T-692]; hover "Ground level (Shift+4)" and so on, then "0.5x cost, $37.5M
  per km", that level's multiplier and price as the price gives them [T-2271], then "0.7x upkeep",
  that level's upkeep for the picked track type as a multiple of a viaduct's, the level whose cost is
  1.0x too: 0.7x on the ground, 1.0x on a viaduct, 2.0x in cut and cover, 2.6x bored, times 1.5 on
  high-speed track to two decimals as the cost (`upkeepMult`, T-2676, Anita 2026-11-06: "relative
  multiples are probably just easier to understand"; T-2006 had dollars a km a day) (the three lines
  are `levelTip` in `ui/trackPrice.ts`; the price's chip hover in the track bar ends with the same
  upkeep line). The Money tab's Upkeep hover gives a viaduct's dollars a km a day and the other levels
  as these multiples, and so do the blueprint's and the track inspector's upkeep hovers (`upkeepLines`,
  `upkeepRule`); totals stay in dollars. Shift+1 to Shift+7 pick the drawing level from
  any tab, by key code, 4 being 0 [T-1701, T-1834], except that while the selection bar shows a blueprint pick
  with its level buttons they put the pick at that level, as the buttons do, and leave the drawing level alone [T-2285]; every segment's open-ground end takes the level
  picked (`Tools.freeLevel`). Then the **ramp switch** (track humping over a line seen side on; hover
  "Automatically try to ramp over crossed tracks at the same level."; on by default) [T-1615,
  [track-and-service.md](track-and-service.md) 6.1 "Ramping over track"], then the price: the picked
  level's price per km and multiplier, "$99M per km (1.1x cost)" (with High speed picked,
  high-speed track's: 1.5x, so 0.75x at ground), the multiplier on the track tool's tag colours:
  light yellow at 1x [T-2271], yellow growing above it, light green growing below it to 0.5 strong
  at 0.5x [T-2744] (`costTint` in
  `map/trackTagRows.ts`); its hover names the level [T-1159] (with ", high-speed track" after it)
  and gives the same upkeep line as the level buttons [T-2006].
  The second row: **Regular** or **High speed** track [T-255, T-817], the
  selection bar's words, then three icon buttons, **Single track** (one line), **Double track** (two lines)
  and **Quad track** (four lines) [T-2931, T-3093]. Single is 0.6x the cost and the single-track upkeep shown;
  quad is two double tracks' price (2x the cost and upkeep shown). Single and quad are never high speed
  (picking either switches to Regular) and picking one clears the other. Tram has no count. A browser remembers the
  choice (`build.tracks`). Double up (the track inspector's and the selection bar's) still adds the pair beside a run.
  Then the **keep placing** switch, the **snap switch** (a magnet, "Snap to real world network") [T-727]:
  while it is on and the track tool is in hand, the real network is drawn at full strength
  ([ui-map.md](ui-map.md) "Real network"); what it does to drawing is [below](#snapping-to-the-real-network);
  and the **beside switch** (two curving tracks; hover "Snap beside track", then "Hover over the blue
  region to draw track parallel to another track" [T-1811]), off by default [T-1685, T-1756]: what it
  does is under [Drawing track](#drawing-track). Last **Target max speed** (hover "Attempt to draw
  track straight enough to reach this speed."), a stepper through 160, 200, 250 and 320 km/h, 160 by
  default, its box fitting "320 km/h"; higher lays the curves of newly drawn track wider,
  [track-and-service.md](track-and-service.md) 6.1 [T-817, T-1127]; picking High speed sets it to 320
  (it can still be stepped down) and picking Regular again puts back the speed from before [T-817].
  There is no crossover tool, and the game adds none: the player draws crossing track [T-733, T-1688,
  [track-and-service.md](track-and-service.md) 6.1]. Everything the player reads says level for the
  track's seven steps, never elevation ("Level", "Level +1", "Levels go from −3 to +3."; "ground
  level" is one of them) [T-2226, reversing T-526].
- **The station bar**, one row: **Platform length** and its stepper, for new stations. The station
  tool reads no other option. Tram stops and the other stations each keep their own length (tram 2
  cars to start, regular and high speed 10 cars, shared by those two); the bar shows the length of
  the kind under the pointer, else of the track type picked [T-3015].
- The level, the ramp switch, the track type, the target max speed, the keep placing switch, the
  snap and beside switches and both platform lengths are remembered per browser (localStorage with the other panel settings,
  not in the save) [T-468]; keep placing is on by default [T-942]. A browser with nothing remembered
  starts at +1, the first visit's level (ground track bridges water and ramps over other track, which
  brings refusals a new player has no reason to expect) [T-1822].
- **On a phone** (at most 600 px wide and at least 560 px tall) the bar is one row of 36 px buttons on
  the map's bottom edge, 10 px in, that fits 390 px (scrolling sideways when narrower): the chosen
  level as one button in its colour, the track type as a button showing "Regular" or "High speed",
  the ramp switch, the three switches, and the target speed as a button showing "160"; a tap on the
  level, type or speed button opens its choices above the bar (the selection bar's pattern, T-1263),
  and a pick closes them. The price is not shown there. The tool notes sit right above the bar and
  give way while a choice is open.

## Click modes: the aura and the descriptor

[T-2284; `MapToolBar` in `ui/ToolBar.tsx`, `ui/ToolNotes.tsx`, `CLICK_MODE_COLOURS` in
`game/palette.ts`; design notes/T-2267.md 3.7] Each click mode but plain selecting has its own
colour: track `#0a819d`, station `#8d71b1`, delete `#d63f73`, select tool `#9c7b31`, bus `#a09e2d`,
ferry `#16716f`, line stops the edited line's own colour (kept legible on the shade, `legibleOn`; `#d777ed`
while no line is picked yet), trip explorer `#268769`, station building `#a15b7b`, passage
`#b0866a`, the same in both shades (chosen 11 or more CIEDE2000 from every colour the game draws).

- **The aura**: the mode's colour round the inside of the map, full at the edge and fading to nothing
  8 px in, still, taking no pointer events; none while plain selecting, so an aura means a click does
  something other than open an inspector. Nothing at the map's edges sits within it: the tool notes,
  the attribution and the phone bars are 10 px in, the other map controls were already.
- **The descriptor** is the first line of the tool notes: a 10 px square in the mode's colour, then
  what the player is doing: "Placing track", "Placing stations", "Deleting", "Selecting track and
  stations", "Exploring a trip" (its hint under it), "New passage from Bedford Av", each with a cross
  after it to leave the mode (hover "Stop (Esc)", for a phone, which has no Esc). The bus, ferry,
  stop editing and station building tools' rows are their descriptors, the square before their words
  ("New bus line: 3 points" with Cancel and Make line, and so on). Esc places a new bus or ferry line of
  two points or more, as it ends a chain of track; on an edit of a built one it drops the edit (T-2924).

## Building by touch

[T-1260; `map/touch.ts`, `Tools.tap` in `map/tools.ts`, `ui/TouchBar.tsx`; design notes in
DECISIONS.md] A finger has no hover, so on a touch screen a tap with a build tool shows what a click
there would do, and a second tap on the same spot (within 24 px) or the control bar's confirm does it.
The mouse keeps the click flow below unchanged; whichever was pressed last on the page decides (a
mouse moving again takes it back).

- **Track**: the first tap is the start, set at once as a click sets it. The next tap is the end:
  the ghost, its curve labels and the tag show, nothing is placed. Another tap elsewhere moves the
  end there, and the end can be dragged by its handle (a 44 px ring round the point); a tap on the
  end or the handle, or confirm, places the segment, and with keep placing the next starts at its
  end. A tap on the start with no end waiting ends the run, as a click on it does.
- **Stations**: on track, a tap shows the platform ghost and its tag, a second tap or confirm
  places it. A tap on open ground is refused at once (no second tap, as a click). After
  it is placed no ghost shows there until the finger moves or taps elsewhere [T-1756].
- **Delete**: a tap shows what would go (drawn red, its name in the tag); confirm or a second tap
  removes it (built things still ask), cancel drops it.
- **Line stops** are toggled by a tap at once, as clicks are; the hints say "tap" and leave out Enter.
- **The control bar** sits beside the point being placed, 44 px from it so the finger does not
  cover it, kept 8 px inside the map: past a waiting end on the side away from the start (the
  ghost lies between), and at a run's new start behind it, where the last segment came from, since
  the next tap is usually ahead. The tag sits beyond the bar. Its buttons, 44 px square with hovers:
  confirm (a green tick, greyed while the ghost is refused), take back (yellow; with a segment being
  drawn it drops the start and the waiting end both, so the next taps are a new start and end and
  the tool stays on [T-1703]; with none in progress it takes back the last segment as Backspace
  does) and stop (a red cross; the run ends and its track stays, or the placement is dropped; the
  tool stays on). The level and track type are the track bar's [T-1706, T-2282].
- **Bus points** drag with a finger and are deleted by holding one (Drawing a bus line, T-1517).
- **Pinch and pan** work as always; a pan or pinch never places or moves a point. A tap that stops
  a fling is no tap (MapLibre's own rule).
- **Select tool**: a one-finger drag pans; a tap adds the piece under it to the pick or takes it
  out (Shift+click's rule, since a finger has no Shift); a press held 450 ms without moving more than
  8 px then dragged draws a box that adds to the pick; a press on the pick drags it, as with the
  mouse. Clear in the selection bar empties the pick.

## Drawing track

[T-023, T-085, T-092, T-244, T-311, T-322, T-418, T-422, T-442]

Track is drawn through points, double and quad alike [T-418, T-422, T-423]. The geometry is in [track-and-service.md](track-and-service.md) 6.1.

- **Drawing beside track** [T-1685, T-1721]: with the beside switch on and the track tool in hand, a
  faint blue band 40 px either side of all track, blueprint or built, marks where drawing snaps beside
  it. A start clicked in the band on open ground (further than the 8 px that joins track) goes on
  the line 12 m (`BESIDE_SPACING`) from that track, on the click's side. From any start (that one, a
  track end, the end of the last segment), while the pointer is on open ground in the band, the
  segment ends on the line beside the pointer and runs along it, curves concentric with the
  track's: a start on the line runs along it all the way; any other start comes in, turning onto
  the line (a curve under the usual rules) somewhere past where it has gone along the track as far
  as the start is from it. The line is on the start's side, so the segment never crosses the track
  it follows [T-1895]: a chain keeps to its side when the pointer strays across. Only a first
  click's start near the line moves across with the pointer. Leaving the band, or onto track or a
  node, gives the plain segment. The segment is ordinary track: the same checks refuse it, it ramps
  over what it crosses by itself (the track's own ramps are not copied), joins nothing at its end
  and adds no crossovers.
- **The point by the pointer** [T-1721]: a red dot where a click branches off a piece of track, a
  yellow dot where it joins at a node (a track end, a junction, a station), a blue ring where the
  segment will end beside track.

- **Clicks**: the first click is the start, the second places a segment ending at it at once (one
  undo step), the track passing through the click, at the level selected. A click with Shift held
  places the segment and starts the next one at its end; without Shift the tool is ready for a new
  start. The **keep placing** button (an icon beside the Regular and High speed choice) does the
  same for every click until it is switched off; it is on by default, and each press of Caps Lock
  flips it, whatever the lock's own light shows (the button shows the switch) [T-445, T-942]. A
  segment that joins track (a loose end or other node, or the middle of a piece of track or
  blueprint) ends the run whatever keep placing says; Shift held on that click still carries on from
  the join [T-1811]. A double click, Escape or Enter ends the run (the first click of
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
  [`map/curveCircle.ts`]; none for a straight segment. A dotted line as faint as the circle runs on 120
  px from the segment's end the way it arrives there, where a straight drawn on from that end would
  go; a straight segment has one too [T-1895], except one from a start on open ground, whose
  line on would only continue it [T-2008]. A click beyond the 1,796 m circle gets that
  circle's curve then a straight to the click [T-471].
- **Tangent on a node drag** [T-2765; `drawTangent` in `map/curveCircle.ts`]: dragging a track end with no other track
  at it draws the same dotted line on from the end as the track tool does; none for a junction or a platform end.
- **What a click joins** [T-1895]: within 12 px of a node (a track end, a junction, a station) a
  click joins at it; within 8 px of a piece of track it branches off it (the station tool reaches
  15 px for track); further out it is open ground.
- **From a station** [T-1155]: a click on a through station, its node or its platform, starts
  the track 20 m past the platform's end on the side the pointer is, as a junction there
  (stations take no junction; [track-and-service.md](track-and-service.md) 6.1); a segment ending
  on a station joins the same way. Before the click the ghost starts there, with the start
  square and its junction mark. A station at the end of its track is extended.
- **Curve labels**: while drawing, the start shows as a square and the curve its speed limit, on
  the speed colour at half strength, no radius [T-417, T-530] (one label for a curve split
  into several corners); the same labels show while a selected blueprint stretch is reshaped or a
  node dragged (`placeCurveLabels`, `map/trackTagRows.ts`). They show only for curves on screen and never
  over one another (64 by 22 px), at most 300, so a long snapped route zoomed out keeps a few
  [T-727d].
- **Loose ends** [T-416, T-1485, T-1522, T-1534; `map/endMarkers.ts`]: while the track or station tool is on, a
  small hollow ring in the accent colour marks each loose end (a node with one piece of track and
  no station) within 160 px of the pointer, and only where the track is visibly drawn: built track
  and active blueprints, not a switched-off blueprint, and not a piece under 4 px long on screen at
  the current zoom (a screen with no hovering pointer shows them all). None show while the pointer is off
  the map, which includes over the view tabs or any other control laid on it. A track click on one starts or continues the route from it; with the
  station tool a spot within half a platform of one puts the station at the line's end, its platform running
  back along the track through the joints of a drawn route as a platform on a piece does [T-860, T-1870].
  The markers take no presses, and show from zoom 9. The track tool also shows the select tool's
  squares at every draggable blueprint point ([below](#reshaping-blueprint)); a blueprint loose end
  has its square in the accent colour in place of the ring, which stays for built loose ends [T-1811].
- **The tag by the cursor** [T-411, T-526] has two lines: the segment's length, average cost
  multiplier and price ("1.5 km, cost 1.33x, $12.8M"), then the elevations the segment passes
  and its lowest speed limit with the track tag's icons and colours ("↕ 0 +2  » 67 km/h": where
  it starts, a bridge's top or a dip's bottom, where it ends, each once [T-707]),
  `map/trackTagRows.ts`; [ui-map.md](ui-map.md) "Hover tags on the network"). The multiplier sits
  on `costTint`'s colours, as the Build tab's price: light yellow at 1x, deeper up to 2.5x, light
  green below 1x [T-2271]. It is the level's multiplier
  (6.4: 0.5x at ground, 1.0x at +1) plus terrain's and water's surcharges, averaged over the track
  priced, both pairs on quad, whose price is two double tracks' [T-711]. Where the segment could not be
  built ending at the cursor, the reason comes first, in red, and the cost line is left out. The
  build tools' tags are drawn over the bubbles' tags; the track tag shows above the pointer when
  it is on existing track. A third line names the roughest terrain class charged with its
  surcharge ("Mountains +3.0x", "Hills +1.2x", "Hills +0.5x") and "Water +2.5x" when the segment
  crosses water, and the roads it crosses at ground level with what they add ("12 crossings
  +$72M" beside the road icon, 6.4) [T-2226], each with an icon. The crossings chip is coloured by
  what they make cheapest for the segment's stretch at ground level (its length from the ghost's
  heights, the route's mean terrain surcharge): tan where a viaduct would cost less (cut and cover
  costs the same), light red-brown where a bored tunnel would, else at grade, the tan thinned toward
  the tag's background by how far the crossings fall short of the cheaper one's extra price [T-2743]; a chip's label and value stay on one line, and when the row is too wide whole chips move to the next line [T-1394, T-1887, T-1890, T-1929, T-2017]. The cost line's price includes
  the road crossings; its multiplier does not. Road tiles landing after the ghost was priced ask it
  again.
- **Shift**: with a tool on, MapLibre's shift-drag zoom is off so a shift-click reaches the tool.
  In the select tool with demand bubbles on, Shift+drag draws a bubble box instead [T-865].
- **Ctrl** (Cmd on a Mac, where Ctrl+click is a right click) [T-1887]: held with the track tool,
  the ghost and clicks snap to no track, node, station or spot beside track: the start and end go
  where the pointer is, as on open ground, so nothing joins or branches; crossing track is then
  what it is for any free segment (a crossing at the same level, or refused). The real-line ring
  [T-727d] still applies. Pressing or letting go of the key redraws the ghost at once. Ctrl had no
  meaning in the track tool before (only Ctrl+Z, Y, S, K and F, and no node drag starts with Ctrl held).
- **The ghost** is drawn as blueprint track (dark blue rails round a light blue middle) with a thin
  outline outside the rails (black; the map's ink, near white, in the dark theme; `ghost` in
  `game/shadeTable.ts`) [T-1054]. The middle is coloured
  by the Level or Speed view (the limit) from the preview's fitted geometry; red when invalid. It is never
  thinner than 4 px across its rails with 1 px of outline either side, so a route hundreds of km long
  shows in full zoomed out [T-1038]. The same holds
  for every placement ghost: a dragged corner or node, a moved pick, a crossover, a station's own
  piece of track.
- **Waiting** [T-1039]: when a click or the newest ghost has waited on the clock worker more than
  300 ms, a small spinner leads the tag.
- **Placed blueprint track** is drawn as blue rails ([ui-map.md](ui-map.md) "Lines and track").
- A segment is one piece (each pair's on quad); the delete tool, Construct and the track
  inspector act on one piece [T-244].

## Drawing a bus line

[T-1464, T-1465, T-1514; `map/busDraw.ts`] The Build tab's Bus button (under the four tools, beside
Ferry, key U), the Lines tab's bus button beside New line, or U from any tab starts the bus tool for a
new line; the Build tab's button again, while it is on, leaves it. Each click adds a
point. Between two points the route takes the shortest path over the basemap's zoom 14 roads
(`busRoads.ts`; trunk, primary, secondary, tertiary, minor and bus-only roads, and motorways, so the
long bridges and tunnels most cities carry on them (T-2469); any direction, length as it is, no turn
rules; service ways, tracks and paths are left out, except that a plain service road (no driveway, alley or parking aisle tag, open to all, at ground level) at most 60 m long with each end within 3 m of a street can be driven, and is never landed on (T-2882)), at most 3 times the straight line or 1 km more
than it, whichever is longer. Roads join where a road's vertex comes within about 1 m of another road
at its level, or where two ground-level roads cross, since the tiles drop the junction's vertex from a road
running straight through it (T-2873; a motorway joins another motorway, and its own ends join any road).
A click within 100 m of a road lands on the nearest one, a motorway
counting 3 times as far and 30 m more, so a click by a street lands on the street; where that road
has no path from the active end, it lands on the nearest other road within 100 m that has one, each
metre further from the click counting as 3 m of road (T-2467). The point stays where it landed.

A route never runs straight off the roads (T-2451, T-2467). A click, the first included, is refused
with a message at the map's bottom left where no road is within 100 m ("Too far from a road. Click
within 100 m of one."), where no road within 100 m has a path from the active end ("No road path
found."), or where the two are too far apart to search (over about 100 zoom 14 tiles: "Too far between
points. Add one on the way."); the tag over the pointer says which before the click, and the leg to the
pointer is dashed red. A click whose roads are still loading waits for them (a few seconds at most,
else "The roads here did not load. Try again."), and meanwhile the leg to the pointer is not drawn: a
leg shows only along its roads in the line's colour, or dashed red. A drag that drops a point where the
roads do not reach puts it back with the same message, and a right click that would leave its
neighbours with no road path between them is refused ("No road path found without this point."). A
straight stretch from an older save that finds no road is drawn dashed red, the tag leads with why, and
finishing is refused ("No road path found for part of the route. Move or delete a point beside it.").
A point on one
carriageway of a divided road (the basemap draws the two as separate lines) may cross to the roughly
parallel road 4 to 40 m across before the path follows a road, the hop counted at its length, so the
route turns there instead of running to the next junction and back (T-1512; `across` in `roadGraph.ts`). The roads come from
the basemap's own tile server at zoom 14, whatever the map's zoom. While drawing,
the route shows in the line's colour with the stops the sim would place (ends as small rounded
squares, a ring round each train station it meets, small rings between and the street stops), and
the tag by the pointer gives the road length and the price. What those stops will connect with is lit
as hovered things are (T-2993, `game/busLinks.ts` `routeMeets`): each train station it stops at, with a
built line's dashed transfer line from the stop to it, and each other bus line it will change with, one
it crosses (T-2952) or with a stop within 100 m of one of its stops (T-2992), drawn a little wider and
lighter; at most 16 of each.

- **Points**: each clicked point shows as a handle in the line's colour (only while the tool is
  on): a dot with a ring of the map's land colour, larger under the pointer; the **active end**, where
  new points join, is a ring instead. With two points a click on the other end makes it the active end,
  so a route grows from either end; with three or more it closes a **loop** (below). A press on any point dragged more than 3 px moves it (the map does not
  pan), and the legs either side are searched again on every move; Escape during the drag puts it
  back. A right click on a point deletes it, its neighbours joined by a new leg; a right click
  elsewhere or Backspace takes back the active end. Over a point the tag says "Drag to move" and
  "Right click to delete", led by what a click there does, and the network's hover is left out.
  With a finger (T-1517): a press within 18 px of a point drags it as the mouse does, the map does
  not pan, and a second finger puts it back; a press held still for 500 ms (moving up to 8 px)
  deletes it (the phone buzzes) and the lift adds nothing. A press that does not move and is lifted
  sooner is a tap, as in Building by touch.
- **Loop** (T-2899): with three points or more, a click on the first point (the tail active) or on the last
  (the head active) joins the other end to it by road, adds a copy of the first point at the end, and makes
  the line or saves the route as a click on the active end does; the tag over that point says "Click to
  close the loop". A road that does not reach is refused as a click's is. A closed route has no active end
  and no leg follows the pointer: a click on empty ground says "The loop is closed. Right click its first
  point to open it."; a click on the first point (also the last) finishes; dragging it moves both and
  searches the legs into and out of it; a right click on it or Backspace takes the closing point off and
  opens the route, and its tag says "Right click to open the loop". Taking out a point of a ring of three
  leaves an open route of two. The tag's stop count leaves the closing stop out. A closed route is a loop line (track-and-service.md
  6.7); Edit route on one reopens it closed.
- **Finishing** makes the line (selects it and asks its name) or saves the edited route: a click on
  the active end, a double click (its first click places the last point), Enter, or the button in
  the tool's bar at the map's bottom left ("Make line", "Save route"; beside it Cancel, as Escape).
  **Escape** drops a new route's points (a second Escape leaves the tool), and leaves an edit with
  the route as it was. A click on the active end of a single point does nothing.
- **Edit route** in the bus panel reopens the tool on the line's route with its clicked points and
  the road stretches between them as they were (a straight one from an older save is searched again,
  red if it still finds no road), the last point active; the same button, now **Save
  route**, saves it. Finishing an edit that changed nothing closes it untouched; otherwise it redraws
  the route (final for a constructed route, which pays any longer road). A new bus line is planned
  until Construct. While the tool is open on a line, that line's current route is drawn muted (the map
  ink at half strength, no stops) under the draft, so the draft reads as the new one (T-3084;
  `BusLayer.setMuted`). A press on one of the selected bus line's clicked points or stops that moves
  3 px with no tool on opens Edit route and carries on as a drag of the draft (a stop becomes a point
  of the route where it stands); the game's route changes only when the edit is saved, as one undo step
  (none for a constructed route). Ctrl+Z and Ctrl+Y (or Ctrl+Shift+Z) in the tool step the draft's
  points back and forward and leave the game's own undo alone.
- **By touch**: a tap previews, a second tap or the check places it; a tap on the active end with
  nothing waiting finishes, on the other end makes it active (or closes the loop, with three points). Dragging and deleting points take a
  mouse.
- **Stop spacing** (T-2885, T-2937; `StopsHead` in `ui/inspector/BusInspector.tsx`): the bus panel's Stops heading
  carries two buttons, Auto and Manual, as the schedule's Buses and Headway pair does; not on a ferry. Under Auto
  (planned or constructed line) a Stop spacing row follows with a stepper from 400 m to 100 km. Its first step,
  400 m, is the automatic placing (track-and-service.md 6.7, T-2947): the stops are about that far apart and
  every station and crossing is one. A click on the value takes a typed distance in the shown unit (anything
  under 400 m is 400). Under Manual there is no row.
  Each step is one undoable edit, the map's stops and the list following at once, and the route tool's
  preview uses the line's stops.
- **Manual** (T-2891, T-2939; `map/busStops.ts`): choosing it opens stop editing, the train mode's look and its
  keys (S or the panel's Edit stops button, shown while the line is manual; Enter or Stop editing ends it).
  The first time it starts from the route's two ends. A stop is the train editor's disc, and a hollow marker
  follows the pointer along the route. Stops go only on the route (T-2982): stations have no marker and take no
  click. Within 200 m of where the route passes nearest a station the route can stop at (`STATION_REACH_M`) the
  marker snaps to that point and the tag names the station; a click there adds the station's stop, on the route.
  Away from stations, within 100 m of where the route passes nearest another bus line's stop that is within
  100 m of the route (`BUS_XFER_M`, demand's station merge; T-2992), the marker snaps there and the tag names
  that line, so the stop is one place with the other line's.
  A click on the route adds a stop, a click on a stop takes it out, each one edit and one undo; the
  two ends stay. Choosing Auto (or any spacing) leaves Manual but the sim keeps the placed stops for the
  session (`BusRoute::hand_kept`, not saved): choosing Manual again gives them back, and a redraw in between
  keeps those within 200 m of the new route. A line's two end stops are drawn like its other stops in Auto
  and Manual alike (T-2938).

## Drawing a ferry line

[T-2194; `map/ferryDraw.ts`; design notes/T-2177.md 3.2] The Build tab's Ferry button (beside Bus,
each half the row, key O), the Lines tab's ferry button beside the bus one, or O from any tab starts
the ferry tool for a new line. The player places **ports only**; between two consecutive ports the
route is found over the basemap's water (track-and-service.md 6.8), never drawn by hand.

- **A click** within 200 m of the shore, on land or on the water, puts a new port on the nearest shore
  (T-2468): its landing where the line from the click to the nearest cell of the other kind (water from
  land, land from water) crosses the shore, its berth in the water where boats stop, named as a new
  station is. Before the click the pointer shows that port where it would go, a circle on the landing
  and a line to the berth in `--ferry-ghost` (dark blue in light, light blue in dark). A click on a port
  the network has (within 10 px or 30 m), or one whose landing would fall within 30 m of such a port's,
  stops there, so lines share ports. A click on water more than 200 m from land is refused ("Place
  ports on land by the water"), and on land more than 200 m from water ("Too far from water"); the tag
  over the pointer says which before the click, "Place port" or "Stop at <port>" where a click works.
  The tag stays up as the pointer moves, as the track tool's does: while the spot under a new position
  is asked it keeps the last answer, so only its text changes (T-2273). From the active end's berth a
  dashed leg in `--ferry-ghost` runs to the berth the next click would join.
- **While drawing** each port is a circle on its landing with a short line to its berth, the active
  end larger; the route between is dashed in the line's colour, grey and straight while it is
  searched, red and straight where no water joins two ports. The tag gives the route's length and the
  price of the ports not yet constructed (US$10M each; the route is free), and how many ports; a leg
  with no route says "No water path found", and the line cannot be made until a port is taken out.
- **Ports as handles**: new ports join at the active end; with two ports a click on the other end makes
  it the active one; a right click on a port or Backspace takes one out. No dragging: a port is moved by
  taking it out and placing another.
- **Loop** (T-2899): with three ports or more, a click on the first port (the tail active) or the last
  (the head active) puts the first port again at the end and makes the line, as for a bus line; the tag
  says "Click to close the loop". A new port closing the loop is sent as `again`, the one placed first
  (`FerryPortIn`), so it is added once. A closed line takes no new port (a click says "The loop is
  closed. Right click its first port to open it."), a click on the first port finishes, and a right click
  on it or Backspace takes the closing port off ("Right click to open the loop"), and taking out a port
  of a ring of three leaves an open line of two. The tag's port count and the price count the first port
  once. In the panel's port list a loop keeps its order (no dragging); taking out its first or last port
  opens it, and one of a ring of three leaves a line of two.
- **Finishing** as the bus tool: a click on the active end, a double click, Enter, or Make line (Save
  route) in the tool's bar; fewer than two ports is refused, and so is a leg with no route ("No water
  path found between A and B."). New ports are added with the line as blueprint ports. Escape drops a
  new line's ports, a second leaves.
- **Edit route** in the ferry panel reopens the tool on the line's ports and its saved route, which is
  searched again only once a port changes. Redrawing is free; a constructed line pays for its new
  ports at once.
- **By touch**: a tap places a port at once (a port is a place, not a point to adjust); the check
  finishes.

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
  or fit over 300 ms shows the small spinner in the tag. The fit lays the curves out for the
  target max speed, as drawn track is (with High speed picked, 320 km/h) [T-1290]; changing the
  elevation, the track or the target max speed fits the route again.
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

- **A click leaves the ghost standing** [T-2763; `addPending` in `map/tools.ts`] where it was clicked, until
  the clock worker's answer puts the station in the world (0.3 to 3 s on the real save, the lines through a
  split piece drawing again); a refusal takes it away.
- **The ghost** is the platform's full box at its true length at every zoom: blue (`#2f6fe0` fill,
  `#1c4fb8` outline) where a click would place it and red (`#e03c3c`, `#b91f1f`) where it would be
  refused, fixed colours with a thin white halo; a refused spot shows the reason in the tag. On
  quad track it is on the pair under the pointer only, as the station will be [T-1687]. With the
  pointer off the map, no ghost, ring or tag shows for the station, track and delete tools (a
  run's segment stays) [T-1777].
  **First frame** [T-695]: the page works out at once, from what it already has, where the click
  would go and whether it would be taken (`likelyFirst`, `layoutIssues` in `map/stationSnap.ts`),
  and shows that: a spot is refused when the
  platform (half its length each way, all of it from a dead end node) and a station's reach beyond
  come within 1 m of the next station, junction or track end, a drawn route's plain joints within
  reach joined through when the pieces match (built or blueprint, tracks, high speed, not a crossover), as the
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
- **What is under the pointer** [T-469, T-671, T-1870]: the ghost and the click take the first spot the
  track model would accept (the same trial edit as above), each at the point on its track nearest
  the pointer, tried in this order: the tracks within 4 px of the nearest, then the rest within
  15 px (at most 5 tracks) [T-1820: 1.5 times the earlier 10; `SNAP_EDGE_PX` in `map/tools.ts`].
  The station tool never snaps to a node: the track tool's node radius (18 px then; 12 px since T-1895, `TRACK_NODE_PX`) covered
  a long stretch of track zoomed out, pulling stations onto joints [T-1870]. Among the near tracks the order is: one where the station would not meet
  a station (none within the two platforms' reaches along the track, up to a junction), blueprint
  over built, the track the last station of the session went on (from its node along the track to
  the next junctions), then the nearest (`map/stationSnap.ts`). A spot that would be refused never
  wins over one that would be accepted; when none is accepted the ghost shows the first one's
  reason.
- **Room along the track** [T-671, T-1870; `roomNear` in `map/stationSnap.ts`]: a spot refused only
  for room (the platform would reach a loose end, `PlatformTooLong`; another station's platform,
  `PlatformStation`; a junction, `PlatformJunction`; a joint that is not joined, `PlatformJoint`; or
  the split would be within 5 m of a node, `SplitPoint`) on one side only is tried again at the
  nearest spot with room: at a loose end, the end node (an end station); otherwise the spot moved
  away along the track, through plain joints, until the platform is 2 m clear (the track model asks
  1 m), or for a joint that is not joined, onto the joint itself when that is nearer (each tried in
  turn, nearest first). It moves at most half the platform and 2 m, so a pointer on another station's platform
  stays refused ("There is already a station here."), while one anywhere off it, or at a junction,
  gets the nearest place that fits. Short of room at both ends, or any other refusal (water, a
  crossing), stays and says why; of the room refusals the tag names one, in the order station,
  junction, joint, track end (`describeIssues`).
- **Placement help** [T-312, T-412]: with the pointer snapped to track the tag says how far the
  spot is, in a straight line, from the station placed last in this session ("1.2 km from W 46th
  St"; nothing before the first), then roughly how many local trips there are within 1 km
  (1 mile in US units): "12k trips within 1km", two significant figures, "Under 50" below that,
  over a thin bar of its size on a log scale, empty at 1,000 and full at 1M, ticks at 10k and 100k
  (`NEAR_BAR_*` in `map/tools.ts`) [T-2271]. The count is the day shown. It is the Homes and Destinations bubbles' legs (all purposes) over the cells whose centres
  are inside the circle, from the zone in view only; a trip with both ends inside counts twice.
  The legs are asked for when the station tool is picked and again whenever the demand shown
  changes, so the count is there when the pointer reaches track; while a newer solve's legs are
  on their way the last ones received stay in use, and a failed ask is repeated 2 s later [T-2430].
  A refused spot shows only the reason. Two thin rings at 1 km and
  2 km of ground distance (0.5 and 1 km for a tram stop, T-3001), each labelled at its top, follow the spot under the ghost
  (`stationRings` in `map/tools.ts`, an SVG layer, fixed grey ink at 35%).
- **Only on track** [T-1820]: with the pointer further than the snap radii from track and nodes, the
  ghost is a refusal: no platform, no rings, the tag "Place stations on track" in red by the pointer.
  A click there places nothing and says the same in the notice. The game has no station on open
  ground any more; a save's old ones stay. The sim's `add_free_station` and the worker's free
  `StationAt` stay for other callers, and the page no longer calls them.
- How a new station is named: [track-and-service.md](track-and-service.md) 6.1. A double click on
  the name in its inspector renames it.

## Crossovers

[T-338, T-733, T-1688; rules in [track-and-service.md](track-and-service.md) 6.1] The game adds no
crossovers and there is no crossover tool: the player draws crossing track between two pairs where
lines should change over. A crossover a save holds is selected, constructed and removed whole like
other track (below). The clock worker's `previewCrossover` and `crossover` edit are for checks only.

## Deleting

[T-096, T-244, T-338; `game/remove.ts`] Hovering shows what a click removes: a stretch of track
drawn red (one piece of a drawn route, or "the crossover", both diagonals drawn red since it goes whole), a station or a
flyover ringed, its name by the cursor. The red goes with the pointer leaving the map or the tool
changing by any route, and a delete's late reply does not draw it again under another tool [T-1872;
`clearDelHover` in `map/tools.ts`]. The track under a station takes the station [T-2412; `stationUnder` in
`map/tools.ts`]: the pointer snapped to its node, or on its platform's stretch of track (half the platform
and 4 m either side, all of a terminus's), rings and removes the station, with its own question, and the track stays; not
zoomed out where stations are hidden. A station's short end piece [T-2797; `stationStub` in `game/remove.ts`, `holds_station` in the track
model], from the station to a bare end and at most 11 m past the platform, also takes the station under the delete tool, bare end included, since
removing that piece would only lay it back; zoomed out, or from the track inspector or the Select tool's Remove with only such pieces picked, it is refused
("A station needs this track. Remove the station first."). **Keys** [T-2797]: with the Select tool's pick, Delete or Backspace removes it; with track or a station selected and no tool on,
Delete or Backspace is its Remove button; with nothing selected, Delete picks the delete tool; while drawing, Delete takes the last segment back like Backspace. A piece beside a through station is drawn red, and named by its
length, only past the platform's end, which is all that goes ([track-and-service.md](track-and-service.md)
6.1, T-1161; `stationTrim` in `game/remove.ts`). Blueprint is
removed at once (undoable; [T-2763] a piece no line runs over and no station touches leaves the screen at the click, `hideEdge` in `map/renderer.ts`, and the line-effects question to the clock worker is skipped; the model follows when the worker answers); anything constructed only after a yes in the confirmation dialog
("Remove 1.2 km of built track? Nothing is refunded." with Remove and Keep; Esc is Keep). The
inspectors' Remove buttons and the Select tool's Remove ask the same way. **A big delete asks
too** [T-1649; `isBigDelete`, `removeQuestion` in `game/remove.ts`]: more than 50 km of track, or a
tenth of the network's, whichever is less (never under 5 km), blueprint or built, in the same
dialog, worded "Delete 4,499 km of built track? Nothing is refunded." with "Delete track" and "Keep
it"; the lines it breaks are listed under it, the first five then "and 104 more lines". Removing
built track and stations is undoable (one undo brings track, stations and lines back, free);
the question for lines' effects runs a trial removal that is rolled back whole, whatever problems
the saved track already had (`TrackWorld::restore`). A removal that changes a
line asks too, built or not [T-859]: a row per line under the question, its bullet then what
happens ("subway 4 will lose its stop at Nevins St and stop running until you fix its route."),
from the track model's trial of the removal (`TrackApi.removal_lines`). Train lines are removed from the
line panel. A bus route under the pointer, where no track is, is removable too [T-1513]: the tag says
"Remove Bus 3", a planned one goes at once with its line (undoable), a constructed one asks first
("Remove Bus 3? Nothing is refunded."). A removed station's two pieces of track become one where they simply join
([track-and-service.md](track-and-service.md) "Lines", T-1299).

## The Select tool

[T-333, T-815, T-829; `map/blueprintBox.ts`, `game/blueprintPick.ts`, `ui/Blueprint.tsx`
`BlueprintBar`] The Select tool (a dashed box icon) picks several pieces at once to disable,
re-level or remove them. A pick is of one kind (Anita, T-829): one that holds any blueprint takes
only blueprint, track and stations; one with no blueprint takes constructed track and constructed
stations. Disabled blueprint is left out by Construct and the blueprint total, and drawn grey
([ui-map.md](ui-map.md) "Lines and track"). It is a tool of its own because a plain drag on the map
is the pan and Shift+drag is MapLibre's zoom box, so a box needs a mode.

- **Dragging the pick** [T-475]: a left drag that starts on picked track or within 9 px of one
  of its nodes (a grab cursor there) moves every movable node of the pick by the same offset,
  previewed as a node drag is: the nodes a box held inside it (T-1898, Anita: "only move the
  waypoints that are in the selection box"; `BpPicked.nodes`), and for pieces picked by a click,
  a double click or Shift+click both their ends; picked stations; never a crossover's ends. A
  piece with one end outside the pick stretches to follow. Track between two moved nodes keeps its shape; track to a node that stays bends as in a node
  drag (`TrackApi.move_nodes`). One undoable edit. A press there without a drag is a click.
  While the tool has a pick, every movable node of it has the white square of a draggable point
  (from zoom 9), and the squares move with a drag [T-1869]; the drag shows each curve's speed
  label on the previewed track and, where it cannot be moved, why, as a node drag does.
- **Left drag** elsewhere draws a box. It picks the track with both its ends inside it (T-2019,
  Anita: only "track segments that have both their endpoints in the selection box"; a piece the box
  only cuts across is not picked or highlighted, so the pick is what a drag moves) and the stations
  inside it (stations only from zoom 9, where they are drawn; a junction goes with its track, never
  on its own), blueprint if the box holds any, else constructed; the pick follows the box live.
  **Shift+drag adds** to the pick, and the sum keeps one kind by the same rule (constructed pieces
  drop out once any blueprint is in). A click picks the piece under the pointer (a station first,
  blueprint over constructed), a click on nothing clears the pick. A piece that is constructed or
  removed leaves a blueprint pick.
- **Runs** [T-1680; `runOf`, `pathTo` in `game/blueprintPick.ts`]: a double click on track picks
  its run, every piece out each way to the next junction or track end (through stations and
  joints where only two pieces meet), stopping where built track meets blueprint; with Shift held
  it adds the run. Shift+click on unpicked track adds it and the shortest way there along track
  of the pick's kind (by length, from any picked piece's end); with no such way it adds the piece
  alone. Shift+click on a picked piece or station takes it out; Shift+click on a station adds it.
  The tool's hover says so. A finger's tap adds or takes out one piece as before.
- **Right or middle drag moves the map** while the tool is on; the wheel and arrow keys work as
  always. The map's own left drag is off for the tool's time.
- **The selection bar** [T-353; `BlueprintBar` in `ui/Blueprint.tsx`] sits at the top centre of the
  map while the tool has a selection, and while plain selecting has a blueprint piece clicked
  [T-2283]: the click opens the track inspector (which has no level buttons) and makes that one
  piece the pick, so the bar acts on it; any other selection, Esc or Clear drops it, and built
  track clicked gets no bar (on the demand buttons' row on a map column over 1350 px, so it
  clears the status tag beside them [T-349]; one row lower below that, lower again below 680 px; it
  appears and goes in one step) is the pick's inspector: "N selected" (its hover says blueprint or
  constructed and lists track and stations), the picked track's total length, and the actions.
  For blueprint: the seven level buttons tinted as in the track bar (the one all the picked
  track is on solid), **Regular**, **High speed** and **Tram** while the pick holds blueprint double track
  [T-1898, T-3108; `kindPicked`, `TrackApi::set_edges_kind`] (the kind all of it is solid, none when
  mixed; High speed's hover gives its 1.5x price): every picked double piece to that kind in one
  undoable edit, free, by the rules of Rebuild below: rail and tram change as whole joined stretches (they never
  join), stations are cut to a tram stop or back and lines become tram or train lines with the new kind's dwell and
  turnaround. Single track, crossovers and pieces the track model refuses (a stretch joined to track outside the
  pick, a curve too tight for trains) stay as they are and are counted in the message as for
  elevation ("3 of 5 pieces set to high-speed track. 2 could not change."). A level or kind picked here
  (its key too) becomes the track tool's as well, as if picked in the track bar (High speed with its top design
  speed, Regular with the speed before it), so the next track placed is laid the same way [T-2474]; a level
  no picked piece could take leaves the tool as it was. Then Disable and Enable as buttons with icon and word (one key, E, enables when any picked piece is off and
  disables when all are on, T-2338; Enable's hover names it, and Disable's while all are on), each greyed when it would change nothing. For either kind: **Double up** while the pick
  holds double track (the track inspector's icon) [T-1680; `doubleUpPieces`, `doubleUpPicked`]: one
  `TrackApi::double_up` for each run of picked double track (pieces joined where only two meet),
  one undoable edit a run, so it covers each run whole, junction to junction, even where only part
  is picked; its hover gives the summed price and the runs' count, or the first refusal, greyed
  when none can; then Remove (key Delete or Backspace) and Clear. E and Delete act on the pick with the Select tool
  only, so while plain selecting the hovers name no key but Clear's Esc [T-2397]. The Build tab shows only "N disabled" while any blueprint is disabled, whatever the tool,
  and beside the blueprint total, in small muted type, what the disabled pieces would cost ("$1.54B
  disabled"; hover: the pieces and the exact sum, left out of the total, then "Double click to enable
  them all", which switches every disabled piece on in one undoable edit) [T-1296, T-1420]: the clock's
  `blueprintCost` (everything unbuilt, disabled pieces, their junctions and crossings in it) less the
  quote's rows other than new cars.
  Disabling and enabling are one undoable edit each (Ctrl+Z). Esc clears the pick, then leaves the
  tool; leaving also clears it. The bar is about 580 px wide with the elevation buttons and keeps
  260 px clear each side of the map, wrapping onto more rows where the map is narrower [T-841]. On
  a phone ([ui-panels.md](ui-panels.md) "Narrow screens") it sits on the map's bottom edge instead,
  the map's whole width in one row that fits 390 px (it scrolls sideways only below about 360 px),
  every button 36 px at least: the tool's icon before the count, one elevation button showing the
  pick's level that opens the seven above the bar, and the four actions as icons with their hovers.
  It sits under the tool notes that share the corner [T-1263].
- **Level** [T-815; `TrackApi.set_edges_level`]: a button puts every picked blueprint stretch
  on that level, one undoable edit: each piece's corners as the track inspector's "Level" buttons
  do for one, and every node all of whose track is picked (a joint, a loose end, a
  junction, a blueprint station with its platform), so a picked run has no ramps at its joints. A
  node with other track keeps its elevation and the picked piece ramps to it. Pieces that cannot
  change are left as they are: a crossover, a piece with no corners whose two ends cannot both
  move, and pieces the track model refuses there (a ramp with no room, a crossing between levels)
  or that would make a flat crossing not there before; those are tried group by group (pieces
  joined through moving nodes), then a refused group's pieces with corners one by one with their
  ends kept. The message says how many changed ("9 of 12 pieces set to level +1. 3 could not
  change."), or that none could. The phone bar's one button's hover is the level's name ("Viaduct"),
  or "Level of the selected track: mixed".
- **Rebuild** [T-3059; `game/rebuild.ts`, `TrackApi::rebuild_quote` and `rebuild_pieces` in
  `sim/src/track/wasm/rebuild.rs`; the preview of the new shape is T-3060]: a pick of built track and
  stations has the same level buttons and Regular / High speed as a blueprint pick, plus **Tram** (T-3088, the
  track tool's word), but they only choose (the chosen ones are solid; nothing is chosen at first), and a
  **Rebuild** button (wrench icon and word) spends: it is greyed until a choice is made. The track and the
  stations keep their ids, so every line keeps its stops and path, stations keep their names and upgrades, and
  nothing is deleted. The level is set as for blueprint (corners, and every node all of whose track is picked, a
  picked station's node and platform edges with them; a node with unpicked track keeps its level and the piece
  ramps to it), and the track type for double track only; rail made tram takes its stations (cut to 80 m tram
  stops) and lines (now tram lines) with it, and tram made rail the other way. A piece the track model refuses
  (no room to ramp to unpicked track, a station left out of the pick, a crossing, a curve too tight for trains,
  tram track meeting rail) is **refused**: the quote names it, the map outlines it in red over the pick's band,
  and the hover and the question count it ("3 of 12 pieces can't change"); it stays as it was, and the rest
  ramps to it (T-3087). Pieces that stay only because they cannot change family alone are counted but not drawn
  red. When nothing can change, Rebuild asks nothing and says why (the track model's refusal, such as "Tram track
  and train track can't join. They can cross."). The price is the full price of the changed track, and of the
  stations whose level or type changed, built new at the new settings, with nothing back for what was there
  (Anita, 2026-11-10: "full new price, that feels more fair"); trains the longer round trips need are bought as
  for any edit. The hover shows the length and what it becomes, then the price; the question
  ("Rebuild 3.1 km of built track and 2 built stations to level +1 for $412M? Nothing is refunded for what
  is there now.", Keep or Rebuild) comes before anything is spent. The edit is final and clears undo, like
  Construct, and waits for terrain, water and road tiles as Construct does. The notice says how many
  pieces were rebuilt ("9 of 12 pieces rebuilt to level +1. 3 could not change.").
- **Remove** [T-829; `TrackApi.delete_pieces`]: everything picked goes in one edit, with the nodes
  left bare and a crossover whole. Blueprint goes at once (one undo) unless it is big (above);
  constructed asks first, as for one piece ("Remove 3.1 km of built track and 2 built stations?
  Nothing is refunded."), and one undo brings it back.
- Disabled is a state on the piece, saved with the game ([track-and-service.md](track-and-service.md)
  6.4). A selected disabled blueprint track says "Disabled" under "Blueprint" in its inspector
  (hover: "Will not be constructed unless enabled") [T-821, T-1242]. The Build tab's "N disabled" hovers which pieces will not be constructed. The piece's own Construct button (track and station inspectors) and a line's Construct build
  it anyway, which clears the state.

## Reshaping blueprint

[T-064, T-071, T-092, T-093; `map/edgeEdit.ts`, `map/nodeDrag.ts`; rules in
[track-and-service.md](track-and-service.md) 6.1]

- With the select tool, a selected blueprint stretch shows its corners as small hollow diamonds [T-2977] (not a
  crossover's, which keeps its shape [T-338]); any blueprint track end, junction or station can be
  dragged, except a node at a crossover's end. The cursor is a grab cursor over a node or diamond
  that can, and a grabbing one anywhere while dragging [T-474]. A draggable node takes the drag
  over a demand bubble. A click on a node's square (drawn over the bubbles) acts on the point, never
  the bubble [T-486]; a click on a blueprint station under a bubble selects the station, as for any
  station [T-1515].
  With the select tool every blueprint track end, corner and junction without a station shows the
  same white square as a node [T-416; `map/endMarkers.ts`], so draggable points can be seen; a
  junction with a fork badge (flat junction marks on) has no square, the badge is its handle [T-663].
  A drag previews the new shape with each curve's speed (or why not); letting go is one
  undoable edit, Escape puts it back.
- **Adding a point** [T-2951; `addPoint` in `map/edgeEdit.ts`, `TrackApi.add_point`]: with the
  select tool, a double click on blueprint track (a line drawn on it included) puts a new point on
  the track there, a plain node like a drawn segment's end, which then drags as any node does: the
  track bends through it, keeping its heading (a drag of the point between two drawn segments
  works the same way). The shape does not change until it is dragged; the piece is now two. One
  undo step, and the drag is another. On a node, a platform's end or a station it does nothing;
  on constructed track it is refused ("Built track cannot be moved. Remove it and draw it again."),
  as are a point within 5 m of a piece's end or on a ramp, and one inside a platform. A mouse's
  double click never zooms the map; a finger's double tap still zooms with the select tool, and a
  finger adds a point with the track inspector's Add point (below). The track tool's double click
  ends a run as before, and the Select tool's picks a run.
- **Platform ends** [T-1869; `map/dragHandles.ts`, `map/nodeDrag.ts`]: with the select or track
  tool, from zoom 14, where platforms are drawn, both ends of a blueprint station's platform have a
  square. Dragging one moves the station as dragging its node does, by the pointer's move, so the
  end stays near the pointer (the platform then follows its track). It previews, commits and
  cancels as a node drag does, and takes a press over a demand bubble [T-486]. A node within reach
  comes first. Any node drag keeps the design speed (a set radius stays set and an auto one is
  laid out for the target max speed, T-1898; [track-and-service.md](track-and-service.md) 6.1);
  curve speed labels show on curves under 5,000 m, and on high-speed track on any curve holding
  trains under 320, its speeds counted up to 320 there.
- **With the track tool** [T-734, T-1811] the same node drag works, its squares shown: a press on a
  blueprint node that moves 3 px or more drags it under the select tool's rules, and a press that
  moves less is the click that places or branches track as always (no hold time). Constructed nodes
  are not grabbed. A route being drawn
  from the dragged node is dropped when the drag lets go; a route from another point carries on.
- **The track inspector** of a blueprint stretch has Construct, "Remove blueprint" (built track
  says "Remove track") and the lines here [T-1669]. Its title carries **Add point** [T-2951], an
  icon of track with a square on it and a plus, on blueprint track that is not a crossover: a point
  halfway along the piece, as a double click adds one (hover "Add a point halfway along"). It has no corner list and
  no curve radius: corners are dragged on the map. Its level is set in the selection bar, which the
  click on the piece opens too (`edgesLevel`; a notice says when it cannot change) [T-841, T-244,
  T-2283]. On double track (blueprint or
  built) the title carries **Double up** [T-708; `DoubleUp` in `ui/inspector/TrackInspector.tsx`], an
  icon of two rails and a plus: a click adds the express pair beside its run as blueprint
  ([track-and-service.md](track-and-service.md) 6.1 "Doubling up"). Its hover gives the price, or
  why not, greyed (a pair beside already). The title names the track: Double track, Single track
  or High-speed track [T-817]; high-speed track has no Double up. A **Speed limit** row above
  Slowest curve gives the track's cap, 160 km/h or 320 on high-speed track, its hover saying which
  trains run up to what [T-817]; the slowest curve on high-speed track counts up to 320.
- **On a crossover's diagonal** [T-391] the track inspector is titled "Crossover" and drops the
  elevation controls; it shows the slowest curve, the price of both diagonals, Construct
  (builds both) and Remove crossover.
- **The junction inspector** [T-031, T-080] draws the junction as a small diagram with its moves,
  says in plain words that it is flat, and offers "Construct a flyover" (built junction) or "Add a flyover" (blueprint), a button under its header, with its price and the waiting
  it saves ([track-and-service.md](track-and-service.md) 6.2), except at a crossover's end, where
  the flyover offer is left out [T-391]. A junction's capacity tag opens it.

## Making and editing lines

[T-109 to T-112, T-233, T-237, T-239; `map/stopMarkers.ts`] A line is made from the Lines tab
(New line, name it, click stations; [ui-inspectors.md](ui-inspectors.md)) and its stops edited from
the line panel (Edit stops). Map clicks test every insertion position and take the cheapest valid
route, keeping the existing stop order: its length, plus per reversal twice that length (at least
5 km, `order_cost`), so stops that lie along one way through the track run straight along it unless
that way is over three times as long (T-1899). Every stop edit of an open line whose route then
reverses also tries its stops again one at a time in their order, as clicks would add them, and
takes that order when it reverses fewer times and costs less (`straighten`), unless the player set
the line's order by hand (below); removing a stop otherwise leaves the others in order. Where several
positions give the same path (a station on a stretch the line runs out and back over, past a
reversal), the stop goes where the stops lie closest together along the path (least sum of
distances between every two stops), so it is served on the pass that leads on to the rest of the
line, whatever order the stations were clicked in (T-1674); a full tie goes to the later position.

**Stop order by hand** (T-2895; `LineInspector.tsx`, `TrackApi::order_stops`): while a line's stops are
edited, a press on a stop's row dragged more than 4 px moves the stop to the gap the drop line marks,
in the plain list and in related lines alike (the pointer is a grab hand; not by touch; Escape puts it
back). A loop's first and closing rows stay, and nothing drops outside them. The drop is one edit
(`orderStops`) and one undo: the route is found again for that order, trains reversing wherever it
needs, with no question asked, and the line then keeps its order (`LineData::hand`): no stop edit
straightens it, and a map click still fits a new stop in where it costs least without moving the
others. An order with no route is refused, the list as it was, with "No track joins {a} and {b}." or
"Trains can't get from {a} to {b} without turning back at a junction." for the first two stops in a row
with no route between them (`order_fault`). While such a line's stops are edited the Stops heading has
a **Sort stops** button (hover "Put the stops in the order with the shortest route"; `sortStops`,
`sort_stops`): the stops, each station once, go in again one at a time as clicks would add them, that
order is taken when its route costs less, and the line follows the game's order again; one undo.

A **new line** (T-1899) is one from New line until its stop editing first ends (Done, Enter, Escape,
another tool or line). While it is new, each stop added or removed on the map sets its trains: high
speed when more than half its route's km is high-speed track, ordinary otherwise (a track's New line
button does the same for the two stops it starts with). The type is set before the stops in the undo
history, so an undo takes the stop back first. Setting the type by hand ends this for that line; a
line edited later keeps its type. Where markers overlap under a click, the line's stops win, then
stations the line's path reaches as it is, then the rest it can reach; on a new line, a station no
running line's trains stop at then wins over one they do; then the nearest.

**Stop editing mode** (Edit stops, New line, Copy line):

- The demand bubbles (local, inter-city, the airport and attraction badges) stay, at 35% and beneath
  the network, with no hover tags [T-2271]; the trip route, capacity tags, traffic widths
  and station circles go; the other lines and their trains are drawn at 30%; the edited line's trains
  stay full.
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
- **The gap** (T-1896; `map/gapFlash.ts`, `TrackApi::stop_gap`): that click also draws where the
  track is missing for about five seconds (4.2 s full, then a 0.8 s fade), in the trip explorer's
  look for a straight line (a dotted ink line on a casing band, an open ring at each end). The two
  sides are the track the line router can reach from the line's stops and from the station
  (`reach_from`, the same search that marks the station unreachable). The gap runs from the loose
  end (a node with one track end or none) of either side nearest the other side's track to that
  nearest point of track, so a stretch built up to a line but not joined shows from its end to the
  line; a side with no loose ends tries its nodes. Nothing is drawn when the two sides share track
  (unreachable only by turning back). If either end is outside the view (less 45 px), the camera
  fits the gap with 90 px of room, never zooming in.
- **Loops** (T-798): the first stop's marker says "Click to close the loop" (tier 5) while the line
  has two or more stops and track leads round from the last stop back to it without the train going
  back the way it came or back over the line's own track (a ring, a circle; T-3021); the click appends
  the first stop (`setStops`). On a loop it says "Click to open the loop" (tier 6) and the click opens
  it (`openLoop`: a circle's closing stop off, a turned line's way back). The closing stop's cross in
  the stop list also opens it ("Open the loop"). New stops on a loop go between its first stop and the
  closing one (`best_insert`); on a turned line's stem, on both passes.
- **Loops at an end** (T-3021; track-and-service.md 6.3 "Turning round a loop at an end"): the stop
  before a loop of track at an end of the line (the last stop at or before where the loop rejoins
  the line; the end stop itself when the loop has no stop of its line) says "Click to loop back here"
  (tier 7) and the click turns the line round it, one way (`turnEnd`). On a line so turned the same
  stop says "Click to stop looping back here" (tier 8) and the click opens it (`openLoop`). The stop
  list shows a turned line one row per station (`game/turnedLine.ts`, T-3025; ui-inspectors.md): a
  cross takes a station out on every pass, the turn's row has a button that opens the loop, and rows
  do not drag. Adding passed stations on a turned line goes through `insertStop`, as a map click does (T-3039). A one-way stop's marker is filled on the half to the right of the way the train goes (T-3041). A hand-ordered line's rows stay one per pass while its stops are edited. The markers take no pointer events (the map's pick handles clicks), so the wheel
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
  "+ 2 stations passed" (its hover lists them: "Add Foo, Bar and Baz") [T-814]. A click on it
  puts Add and Cancel in its place, Add stretched wide so it is under the pointer, with the stations
  listed below as faint "+ Name" rows in path order; Add makes them stops in one undoable edit. A
  double click on the gap's row adds them at once (the second click lands on Add, or, if the panel
  has not redrawn yet, the old row's dblclick adds; the hover says "Double-click to add them at
  once") [T-1297, T-1489]. **A** adds every station the open line panel's
  line passes, whether or not its stops are being edited; there is no "Add all" row (T-1297's was
  clutter, T-1417). The row and Add count every passed station, those another line stops at
  included (T-3007). Only stations on the line's path count: a quad twin on the pair beside, or a
  station on a branch, is not offered. The passed stations come in the clock state with each line
  (`ClockLine.passed`), so the list never shows new stops with the old gaps (T-648).
