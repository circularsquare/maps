[Back to the spec](../SPEC.md) · Interface: [panels](ui-panels.md) · [inspectors](ui-inspectors.md) ·
**the network on the map** · [demand views](ui-demand-views.md) · [building](ui-building.md)

# 8. Interface: the network on the map

Track, stations and trains are drawn by our own WebGL2 renderer on a canvas over the basemap
(`map/renderer.ts`; why and how in [architecture.md](architecture.md) 7). The basemap is
Positron, repainted faint (`map/basemap.ts`), so the network reads first.

## Zoom bands

- **Below zoom 11** (more than a city such as Manhattan with its neighbours, about 30 km) only the
  lines are drawn [T-205, `STATIONS_ZOOM`]. Stations and trains are neither drawn nor picked, so a
  click or hover finds the line. A selected station or complex is still drawn and clickable.
  Demand drawings and capacity tags are unchanged.
- **Zoom 11 to 15**: every station is its own dot; trains are pills.
- **From zoom 15** (`STATION_BOX_ZOOM`): a station is its platform, trains show their cars, and
  platform crowds are drawn.
- Station names and capacity tags hide below zoom 12 [T-101].

## Lines and track

- **Widths** [T-191, T-205]: a line is 0.85 px at zoom 9, 3.9 at 12 and 6.6 at 15 (linear between,
  held outside); bare track 0.4, 1.8 and 3; a white casing 0.6-2.4 px wider, drawn as a fixed margin
  under the stroke and not part of a traffic width [T-267]; a selected line or track 0.5-3 px wider
  still. A stroke thinner than a device pixel is drawn a pixel wide and that much fainter.
- **Lines sharing track are drawn over each other** [T-190]: interlining is not represented; the
  line with the highest id is on top and wins a click there.
- **Track display** [T-105; Settings], four choices, remembered per browser:
  - **Line**: uniform width in the line colours.
  - **Line + traffic thickness** (the default): width by riders over the day shown
    ([ui-demand-views.md](ui-demand-views.md) "Line width by riders").
  - **Height**: the track coloured by level, warm above ground, green at ground, blue-green to
    indigo below: +3 #c2351c, +2 #ee7a1a, +1 #f0b928, 0 #58b04a, -1 #16a39a, -2 #2f6fd0, -3 #4a2c9a
    (`LEVEL_COLOURS`, T-413, T-478; neighbours at least 30 apart in CIELAB). The same table colours
    the Build tab's level buttons and the Settings key.
  - **Max speed**: the geometry's allowed speed at each section, curve limits included, capped at
    160 km/h (not a train's speed), keyed 40/80/120/160 km/h from warm red through ochre and green
    to blue (`SPEED_STOPS`, the stroke shader blends the same stops).
  In Height and Max speed the width is uniform and trains keep their line colours.
- **Blueprint track** [T-311, T-322, T-331, T-333] is drawn solid and opaque as two rails in every
  track view: a light blue middle (#a9cdf3) between dark blue rails (#2f62b3). The whole width is 75%
  of what T-322 drew (a line plus a rail of 16% of a line width, at least 1 px, either side), the
  rails never under 1 px: 6.5 px at zoom 15 (4.5 middle), 4.4 at zoom 12. A line planned along it
  fills the middle, not a full line width. **Switched off** [T-333]: grey rails (#9aa3ae) round a
  nearly white middle (#f2f4f7), and a station at 30% instead of 60%; in Height and Max speed the
  colour is mixed 80% towards white instead of 45%. **Picked** with the blueprint tool: an ink band
  3 px beyond the rails, drawn under the track, and the stations' selection outline. All rails are drawn before any middle, as the casing is, so curves
  and joints join cleanly. In Height and Max speed the middle keeps the view's colour mixed 45%
  towards white; a selected blueprint's middle is the selection ink mixed the same way. A line
  planned along blueprint track fills the middle in its colour mixed 45% towards white. Lines that
  do not run stay see-through at 55%.
- **Line colours** [T-223, `LINE_PALETTE` in `game/palette.ts`]: a normal, fairly saturated transit
  palette, thirteen defaults: red, blue, green, orange, purple, yellow, brown, teal, gray,
  yellow-green, pink, magenta and light blue, every pair at least 19 apart in CIEDE2000; chips on
  the light ones take dark letters. A new line takes the first default no line uses.

## Stations

- **Dots, zoom 11 to 15** [T-189, T-313, T-322; `map/stationGeometry.ts`]: white with a dark rim at
  the platform's middle. Radius 4.3/5/7.4 px at zoom 11/12/15 for an ordinary station, 5.6/6.6/9.5
  px for one served by more than one line (`stationRadii`), never narrower than the lines through
  it, the rim 1.5 px at zoom 13. A complex's dots are one shape: every dark disc is drawn first,
  then every white, so overlapping dots have one outline. Dots that do not touch are joined by a
  **connector**: a straight band centre to centre with the dots' rim along both sides and white 30%
  of an ordinary dot's white radius either side of its middle, drawn in the same passes, so dots
  and connectors are one white shape. Connectors join the stations the complex distance rule groups
  (`dotGroups` in `game/complexView.ts`) by the shortest set of joins, and each player link between
  two groups [T-225]. A connector is faded when either station is a blueprint.
- **Each dot is its own station** [T-249, T-305]: a click picks the dot whose centre is nearest
  among those under the pointer and selects its complex; the delete and line tools take that
  station, and its tag lists its own lines.
- **Crowd tint** [T-228]: with platform crowds on, a dot's white is tinted towards the crowd grey by
  its fullest platform side's share of the room (at most 45% of the way, at full), or a pale red (40%
  of the way to the over-full red) while a train at any of its platforms left riders behind.
  Connectors stay white.
- **Platforms, from zoom 15** [T-204, T-254, T-239]: a band along its track as long as the platform,
  following the track's centre line sampled at 13 points (`platformShape`, `PLATFORM_POINTS`;
  straight on past an edge that stops short), flat-ended, centred on the station node, or at a dead
  end running from the node along the track. Its width is fixed on the ground like its length: a
  quarter narrower than a plain line and its rim at zoom 15 (9 px there, about 16 m at New York's
  latitude), so its shape never changes with zoom and it is never shorter than it is wide. The same
  width in every mode: a wide traffic line can be wider than its stations. Everything that draws or
  tests a platform (box, edges, crowd fill, track stencil, picking, placement ghost, selection
  outline, the cut-out in the bubbles) uses that one line (`polyFrame` in the shaders,
  `stationShape`, `stationPath`). Blueprint stations are faded.
- **Overlapping platforms are one shape** [T-217]: every dark shape first, then every white inside
  its rim, so one outline runs round the whole (a faded blueprint draws only its rim).
- **Inside the white** [T-236], over the crowd fill: the track and line strokes run through the
  box (drawn where a stencil marks the white, no wider than 30% of it, without casing), then each
  platform's two long edges as thin lines in the platform blue (`PLATFORM_BLUE` #8085f0), 14% of the
  white's half width thick (at least 1 px), 15% of it in from the rim, coming in between zoom 15
  and 15.5.
- **A player link's bridge** [T-197]: a straight white band 3 px wide at zoom 15 (fixed on the
  ground) with the 2 px rim along both sides, between the nearest points of the two platforms'
  centre lines, at least 6 px inside each platform's ends; overlapping platforms get one inside
  both. Members of a complex that do not touch stay separate shapes.
- **Selected** [T-163, T-189]: the station keeps its size and gets a 2 px dark outline and a soft
  dark glow 6 px beyond, round the union of the selected shapes and bridges, drawn under them;
  selected stations are drawn after the others and the demand bubbles are cut away over them. In
  complex view every member, bridge and connector is drawn selected. A click on a selected station
  picks it even over bubbles.

**Station complexes** [T-146, `game/complexes.ts`]: stations whose platforms come within 50 m (the
straight segment of the platform's length along the track heading), or whose centres are within
100 m (demand's merge distance), are linked, whatever their levels; a complex is everything linked
directly or through others, plus the player's links [T-197]. Worked out whenever stations or links
change; only links are saved. Riders change between any two stations of a complex on foot
[T-161, [demand.md](demand.md) 4.4].

## Trains

- **Shape** [T-103, T-107, T-189, T-199]: not drawn below zoom 11; pills from 11, separate 20 m cars
  from 15, in a darker shade of the line colour with a white rim blended over a pixel, antialiased.
  Width 3.75 px at zoom 9, 6 at 12, 9 at 15 (linear between, held outside; true width only from
  about zoom 19), shared by pills and cars so the switch never shrinks a train. A pill is never
  shorter than 11, 17.6 and 26.4 px at those zooms and bends with its track along the circle
  through its ends and middle. Each car is turned along the chord between its ends, so the consist
  snakes round curves and stays on the route at terminals (notes/T-047.md). One instanced draw for
  all trains, positions from keyframes on the GPU. A click on any car selects the train; one drawn
  over a station selects the train.
- **How full** [T-103, T-207, T-218, T-230]: the core fills from its back like a gauge across the
  whole consist, in the dark shade of the line colour (48% of the line RGB) over an 18% white tint.
  Half full is every seat taken, full is packed to crush, from the boarding account
  ([track-and-service.md](track-and-service.md) 6.6): this train's riders on the stretch it is
  drawn on, empty at its last stop. A train at crush has a red rim 3.9 px wide (the white rim is
  1.7 px).
- **No figures yet** [T-230, T-324]: a running line demand has no riders for (just after an edit
  changed its stops) is drawn striped in the line colour and white, never solid, and its tag says
  "Calculating demand".
- **Selected** [T-242]: drawn over every other train, unchanged, inside a halo round the whole
  consist: a dark outline, a white band, a dark ring of the train's shape 6 px out and a soft glow,
  so it reads even at zoom 11-13. A click tries the selected train first. It is followed onto its
  next trip [T-245, [ui-inspectors.md](ui-inspectors.md)].

## Platform crowds

[T-164, T-219, T-236] From zoom 15 the white of each platform box fills with a dark warm grey
(#77726d, kept 1.5 px clear of the rim or the blue edge) as riders wait. Each direction of travel
has its half of the box along its length; a platform whose trains all use one side (a terminal)
fills as one. A side fills from the middle outwards by its riders' share of the room
(`platformRoom`, [track-and-service.md](track-and-service.md) 6.6: 1,250 a side on 150 m), full to
both ends at that or more. A side is red instead of grey while the last train on it left anyone
behind. A standing train covers the box's middle; from zoom 16 the sides show beside it. The crowd
grows between trains at the arrival rate and drops when one leaves, worked out on the main thread
at most once a game second, only while drawn.

## Station names and capacity tags

- **Names** [T-032, T-074, T-101, T-169; `map/stationLabels.ts`]: DOM labels in the UI font above
  the overlay. Hidden below zoom 12; above it padded collision boxes keep them apart (48 x 28 px
  until zoom 14, then 20 x 12), transfers first, then constructed stations, then lines. A
  complex's stations sharing a name have one label at their mean position, oriented like the
  member with the most lines. They beat basemap place names through an invisible symbol layer of
  the same names that MapLibre's collision honours. A flat pan moves their shared compositor
  layer; placement and the greedy overlap pass run on other camera changes, edits, font loading,
  showing labels again and after movement ends. A 256 px margin prepares names entering the
  screen, and a pan re-places them after 128 px of travel (notes/T-074.md).
- **Capacity tags** [T-184, T-224, T-256; `map/capacityMarkers.ts`; what they mean in
  [track-and-service.md](track-and-service.md) 6.2]: placed after names, which never move for a
  tag. Each takes the first of six spots round its resource clear of names (3 px) and earlier tags
  (above right, above left, below right, below left, right, left, 5-6 px out), else hides. While a
  tool other than select is on they take no pointer events, so a press reaches what is under them.
  With select, a tag that opens nothing of its own (not a junction) selects what is under it, its
  station first.

## Hover tags on the network

- **Track** [T-157, `map/trackHover.ts`]: with the select tool, resting on a line or track for 0.1 s
  shows a tag by the pointer saying what the track display draws there. Line: the name. Traffic:
  the name and "41,200 forecast riders a day in both directions" (or "No forecast riders on this
  stretch") on that stop-to-stop stretch for the day shown (counts to the nearest 100 above
  1,000), with a third line under the purpose filter
  ("Shopping and errands trips only"). Whatever the display [T-473], the tag also carries the level
  ("Tunnel at level −2", "Ground level") and "Max speed 120 km/h" at that point, and on bare track
  the lines using it ("Lines: subway 5, subway 7", or "No line runs here"). A planned line adds
  "Not running yet", a broken one "Not running"; before the first solve the traffic view says
  "Calculating demand". The tag follows the pointer along the track; leaving, pressing, dragging, zooming or
  picking a tool hides it. It follows the pick order, so bubbles, capacity tags and handles above
  the network keep it hidden; junctions say nothing.
- **Trains and stations** [T-194, T-305, T-313; `map/networkTips.ts`]: the same tag (wrapping at 320
  px). A train: "Line 1 train", "340 riders aboard" and "352 seats, 1,280 packed full". A station:
  its name, then its lines by symbol or by name without a leading mode word ("subway 6X" is "6X"),
  each once (a station drawn as its own dot or box lists its own lines); with demand circles on,
  "41,200 boardings and alightings a day"; where crowds are drawn a line per line and direction
  with anyone waiting ("120 waiting for subway 1 to Van Cortlandt Park-242 St", "after a full train
  left some behind" while red) or "Nobody waiting on the platforms", then the fill ("Platform 21%
  full on one side and 4% on the other (room for 1,250 a side)"). On a dot, the crowd lines are its
  tint's numbers ("1,257 waiting", "Fullest platform 45% full", "A full train left some behind"). A
  demand circle away from any station gives its busiest station's name and count. Refreshed each
  game minute while shown.

## Picking

Picking follows drawing order [`map/pick.ts`]: demand discs above trains, then stations, lines,
junctions and track; within a layer the last drawn wins; hidden layers do not intercept. Network
hits reach 7 px from the centre line, or half the drawn width where a line is wider [T-227].
Clicking the line already selected selects the track under it. Clicking a selection's far end or a
station's catchment keeps the parent selection. Selected editing handles stay draggable above the
demand view.

## Zone outlines

[T-131, T-139, T-310; `map/cityBoundaries.ts`, `map/boundaryGeometry.ts`] Each registered pack's
footprint (its dissolved county boundary) has a thin solid outline at regional zooms, fading
between zoom 8.5 and 10.5. The focused zone keeps a thin boundary at closer zooms (darker outside
stroke, lighter inside) and a subtle shade darkens the basemap outside its exact footprint,
enclaves included. These are demand areas, not municipal limits. They sit under labels and game
overlays, never take clicks, and survive basemap style changes. A change of focus swaps the shade
in one step, never through an empty frame; the footprints are ready from startup.
