[Back to the spec](../SPEC.md) · Interface: [panels](ui-panels.md) · [inspectors](ui-inspectors.md) ·
**the network on the map** · [demand views](ui-demand-views.md) · [building](ui-building.md)

# 8. Interface: the network on the map

Track, stations and trains are drawn by our own WebGL2 renderer on a canvas over the basemap
(`map/renderer.ts`; why and how in [architecture.md](architecture.md) 7). The basemap is
Positron, repainted (`map/basemap.ts`) so the network reads first: warm paper land, soft teal
water, sage parks and woods, place names faint grey. Roads have three weights (motorways, main
roads, everything else), one colour and no casings; the smallest classes appear only from zoom 11
(streets) and 14 (paths); minor road names, road shields and the basemap's railways are hidden.
Positron has no shop or food icons, so the airport symbols are the only icons left. The dark theme
is the same three hues deep and muted (dark warm brown-grey land, deep teal, dark sage), by paint
properties only, so switching theme reloads no style or tiles [T-039, T-676]. The light and dark
map colours (land, water, park, road, labels, the casing, the map ink, a switched-off blueprint,
the shade outside the focused city, the save thumbnail's track) are one table,
`game/shadeTable.ts`, whose named palette values at the top are the ones to retune by hand (read
through `game/shades.ts`) [T-543].

## Real network (reference layer)

Settings > Show > **Real network**, on by default: faint lines (15% opaque, T-727) where railways run
in the real world, for reference, view only; 40% (T-697) while the track tool snaps to them (the
Build tab's snap switch, [ui-building.md](ui-building.md)), which also shows them with the setting off (`map/realNetwork.ts`, T-682). They sit above the basemap's
roads and under its labels, so under our own network (which is drawn on the overlay canvas above
everything). Three classes by colour and width, in both themes (`realMain`, `realOther`, `realMetro`
in `game/shadeTable.ts`): main line, other passenger rail, metro. Trams, light rail, monorails,
funiculars, narrow gauge, heritage lines and track no line runs over are not drawn. Lines appear
from zoom 4; each region's tiles are added when the view first touches it, nearest first, and
nothing is fetched while it is hidden. The lines are noritetsu's per-region tiles (OSM and
the national registers, one `track` layer); the tiles carry no line type or name at low zoom, so
there is no high-speed class and no hover yet.

## Zoom bands

- **Below zoom 9** (about 160 km across, a whole region) only the lines are drawn [T-205, T-542,
  `STATIONS_ZOOM`]. Stations and trains are neither drawn nor picked, so a click or hover finds the
  line. A selected station or complex is still drawn and clickable. Demand drawings and capacity
  tags are unchanged.
- **Zoom 9 to 10**: every station is its own dot, drawn and picked; trains are not.
- **Zoom 10 to 14** [`TRAINS_ZOOM`]: stations are dots; trains are pills, drawn and picked from 10.
- **From zoom 14** (`STATION_BOX_ZOOM`, T-553): a station is its platform, trains show their cars, and
  platform crowds are drawn.
- **About zoom 17** (T-658; by 4 m on screen, so the same in every city: 16.7 to 17.2 in New York):
  double track opens into its two tracks ("Lines and track").
- Station names and capacity tags hide below zoom 12 [T-101].

## Lines and track

- **Widths** [T-191, T-205]: a line is 0.85 px at zoom 9, 3.9 at 12 and 6.6 at 15 (linear between,
  held outside); bare track 0.4, 1.8 and 3; a casing 0.6-2.4 px wider (white, near-black #141517 in
  the dark theme [T-039]; line colours are the same in both), drawn as a fixed margin
  under the stroke and not part of a traffic width [T-267]; a selected line or track 0.5-3 px wider
  still. A stroke thinner than a device pixel is drawn a pixel wide and that much fainter. Line and
  track ends, and the joints between a stroke's segments, are round, casing included [T-673].
- **Double track as its two tracks, close in** [T-658, `laneState` in `map/renderer.ts`, design
  notes/T-658.md]: drawing only; the track stays one piece. Where 4 m on the ground is 4 px on
  screen, a double track's stroke starts growing to its two tracks' width (4 m plus a track's
  2.4 m), which it reaches at 7 px; from there to 10 px a stripe opens down its middle, widening
  to 1.6 m while the outline stays the same, each track half of what is left, until the tracks'
  centres are 4 m apart and each is 2.4 m wide; from 10 px on everything is to scale, except that
  from 10 to 16 px each track narrows to 1.92 m (80%; `LANE_NARROW`, T-713) so the riders waiting
  behind show, the pair and its band narrowing with it. A station on double track then has a
  platform each side ("Stations").
- **Turn-back crossovers** [T-658d, `turnbackStrokes` in `map/network.ts`]: close in only, where
  a line ends at a double-track station, the crossover the game assumes there (6.2) is drawn in
  bare-track grey over the line's two tracks, 50 m long: a scissors (both diagonals) just outside
  a dead-end terminus's platform, and at a through station one diagonal just outside the
  platform on the side the line arrives from, from the arrival track to the departure track. Not
  picked, no hover of its own.
  Trains keep right (T-319): each line's run 0 is drawn on the right-hand track of its direction
  and run 1 on the left. Under a line's two tracks is a band as wide as the pair or its traffic
  width, whichever is wider, in the line colour mixed 60% towards the casing, over one casing
  band (Anita, 2026-10-19: the tracks over the traffic width). Bare track is two grey tracks as
  much thinner than 2.4 m as it is thinner than a line; blueprint track is two blueprint tracks,
  each its own rails and middle. Single track stays one track on the centre line, 2.4 m wide;
  double track narrows into it over 40 m where the two meet at a plain node (`lanePoints` in
  `map/network.ts`); at a junction a single-track branch leaves from the centre line. Quad
  track is four tracks 4 m apart; a crossover between double tracks is two double tracks in an
  X. Clicks and the hover reach half the pair's width from the centre line once it has grown.
- **High-speed track** [T-817, `F_HSR` in `map/network.ts`, the stroke shaders in
  `map/renderer.ts`]: double track's width and colours with a thin centre line along the stroke,
  18% of its width (at least 0.6 px): the casing colour on built track (light, dark in the dark
  theme), the rail colour on blueprint, the ink on the drawing ghost; lines over it carry it too.
  Far out, where the stroke is a pixel or two, it shows as a paler built stroke and a darker
  blueprint one. Once the two tracks part, the line runs between them in the track's or the
  line's own colour.
- **Lines sharing track are drawn over each other** [T-190, T-533], each of the two tracks too
  close in: interlining is not represented.
  One order for all lines: a line with no trains (fleet zero) goes under every line that runs
  trains; within each group the shorter route (end to end, one direction) is on top, ties to the
  higher id. The line on top wins a click. It is rebuilt when lines or fleets change.
- **Track display** [T-105; on the map under the demand buttons since T-703], four choices, remembered per browser:
  - **Line**: uniform width in the line colours.
  - **Line + traffic thickness** (the default): width by riders over the day shown
    ([ui-demand-views.md](ui-demand-views.md) "Line width by riders"). A line on track at
    capacity [T-819] (an edge of a track or single-track section, or either track of a flat
    crossing, at `FULL_RHO` for the demand level in force, as the capacity tags take it;
    `TrackApi.full_track`; platforms, termini and junctions keep only their tags) has a red band 1.5 px each side
    outside its casing (#c0392b, #f0584f in the dark theme; `fullTrack` in `game/shadeTable.ts`),
    drawn in a pass of its own before the lines' casings, so a red line keeps its casing between
    and a neighbouring line's casing covers the band where they touch; the hover adds "Track at capacity at high
    demand" in red.
  - **Elevation** (`height` in code): the track coloured by elevation (a level in code), blue-green to indigo above ground (open sky), green
    at ground, warm below (deep): +3 #4a2c9a, +2 #2f6fd0, +1 #16a39a, 0 #58b04a, -1 #f0b928, -2 #ee7a1a,
    -3 #c2351c (`LEVEL_COLOURS`, T-413, T-478, T-480; neighbours at least 30 apart in CIELAB). The same table colours
    the Build tab's elevation buttons, the Settings key and the elevation in the track tags.
  - **Max speed**: the geometry's allowed speed at each section, curve limits included (not a
    train's speed): 160 km/h on ordinary track, up to 320 on high-speed track. Keyed 40/80/120/160/240/320 km/h
    from warm red through ochre and green to blue, then indigo (#5b4fc9) and violet (#a03fc2)
    (`SPEED_STOPS`; the stroke shader is built from the same stops, and the key beside the display
    buttons lists them).
  In Elevation and Max speed the width is uniform and trains keep their line colours.
- **Blueprint track** [T-311, T-322, T-331, T-333] is drawn solid and opaque as two rails in every
  track view: a light blue middle (#a9cdf3) between dark blue rails (#2f62b3). The whole width is 75%
  of what T-322 drew (a line plus a rail of 16% of a line width, at least 1 px, either side), the
  rails never under 1 px: 6.5 px at zoom 15 (4.5 middle), 4.4 at zoom 12. A line planned along it
  fills the middle, not a full line width. **Switched off** [T-333, T-543]: grey rails (#9aa3ae) round a
  nearly white middle (#f2f4f7) in the light theme, a dim middle (#383e46) between lighter grey rails
  (#6c747e) in the dark one, and a station at 30% instead of 60%; in Elevation and Max speed, and for
  a line planned along it, the colour is mixed 80% towards white (towards #383e46 in the dark theme)
  instead of 45% towards white. **Picked** with the blueprint tool: an ink band
  3 px beyond the rails, drawn under the track, and the stations' selection outline. All rails are drawn before any middle, as the casing is, so curves
  and joints join cleanly. Blueprint track and the lines planned on it are drawn after built track
  and its lines [T-695], so where they overlap the blueprint is on top and wins the click, as the
  station tool prefers it; the order of lines sharing track (above) holds within each. In Elevation and Max speed the middle keeps the view's colour mixed 45%
  towards white; a selected blueprint's middle is the selection ink mixed the same way. A line
  planned along blueprint track fills the middle in its colour mixed 45% towards white. Lines that
  do not run stay see-through at 55%.
- **Line colours** [T-223, `LINE_PALETTE` in `game/palette.ts`]: a normal, fairly saturated transit
  palette, sixteen defaults: red, blue, green, orange, purple, yellow, brown, teal, gray,
  yellow-green, pink, magenta, light blue, olive, coral and lavender, every pair at least 19 apart in
  CIEDE2000 and at least 17 from the land, water and parks of both themes; chips on
  the light ones take dark letters. A new line takes a random default no line in its city uses [T-856].
- **Marks on the track** [T-629, T-655; `map/gradeMarks.ts`, data in `map/markData.ts`], drawn by the
  renderer over the track and under stations and trains, from zoom 9 (blueprint and drawing) or 11
  (built):
  - A **flat crossing** (an X) and a **flat junction** (a fork, a node with three or more track ends
    and no station, not a flyover) in a round badge with a thin ring. On blueprint track and in the
    drawing tool's preview (the crossings and junctions the new track would make, from the track
    model) the badge is big: radius 7 px at zoom 9, 11 at 12, 15 at 15. On built track it is small
    and quiet: 5 px at zoom 11, 6.5 at 13, 8.5 at 15, at 85% opacity. Its ring and glyph are in the
    Elevation colour of the level the meeting is on [T-673]. A junction is built once three of its ends are, a crossing
    once both tracks are. A crossing the model refuses (under one elevation step apart, or on a
    ramp) is a red X.
  - A **ramp** (a wedge) where track climbs from level 0 and a **tunnel portal** (a half disc) where it
    drops from level 0, at the ramp's level 0 end, pointing along the track, filled in the Elevation
    colour of the level it reaches, from zoom 11 at 3.5 px radius, 5 at 13, 7 at 15. A ramp that
    starts below ground or above level 1 (1 to 2, -1 to -2) has none.
  The fill is the casing's. The Settings checkbox "Flat junctions and crossings" (on by default,
  saved with the display settings) hides the crossings and junctions and their hover; ramps,
  portals and refused crossings stay [T-673].

## Stations

- **Dots, zoom 9 to 14** [T-189, T-313, T-322, T-542; `map/stationGeometry.ts`]: white with a dark rim (in both themes, T-039) at
  the platform's middle. Radius 1.7/4.3/5/7.4 px at zoom 9/11/12/15 for an ordinary station, 2.4/5.6/6.6/9.5
  px for one served by more than one line (`stationRadii`; linear between, held outside), never narrower than the lines through
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
- **Crowd pie** [T-228, T-553]: with platform crowds on, a dot's white fills like a pie chart,
  clockwise from 12 o'clock on the screen, with the crowd grey (#77726d) in proportion to its fullest
  platform side's share of the room (full circle at full), or the over-full red once that side is
  full [T-559]. The rim stays; each dot of a complex is its own pie, and the
  connectors stay white. The same colours in both themes.
- **Platforms, from zoom 14** [T-204, T-254, T-239, T-553]: a band along its track as long as the
  platform, following the track's centre line sampled at 13 points (`platformShape`,
  `PLATFORM_POINTS`; straight on past an edge that stops short), flat-ended, centred on the station
  node, or at a dead end running from the node along the track. Its width is fixed on the ground
  like its length from zoom 15 (`BOX_WIDTH_ZOOM`): 1.125 times a plain line and its rim
  there (11.25 px, about 24 m at New York's latitude; `BOX_NARROW`, 50% wider since T-652), so its shape never changes with zoom and it is
  never shorter than it is wide. Further out, to zoom 14, it keeps that 11.25 px on screen (white 7.25 px
  across) while its length keeps shrinking with the ground, about 40-55 px at zoom 14. The same
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
- **A platform each side, close in** [T-658c]: on a double-track station (flag 16 in the station
  buffer), as the two tracks part ("Lines and track") the blue edges move in beside them, 0.4 m
  wide and 1 px outside each track, and each direction's riders stand on their own platform,
  from just outside its blue edge out to 1.5 px inside the rim, instead of filling half the box:
  the + side's trains travel along the platform axis and keep right, so their track and their
  platform are on the + side. A single-track station is unchanged.
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
[T-161, [demand-kernel.md](demand-kernel.md) 4.4].

## Trains

- **Shape** [T-103, T-107, T-189, T-199]: not drawn below zoom 10 [T-542, `TRAINS_ZOOM`]; pills from 10, separate 20 m cars
  from 14 (`STATION_BOX_ZOOM`), in a darker shade of the line colour with a white rim blended over a pixel, antialiased.
  Width 3 px at zoom 10, 5.25 at 11, 6 at 12, 9 at 15 (linear between, held outside; true width only from
  about zoom 19), shared by pills and cars so the switch never shrinks a train. A pill is never
  shorter than 11, 17.6 and 26.4 px at those zooms and bends with its track along the circle
  through its ends and middle. Each car is turned along the chord between its ends, so the consist
  snakes round curves and stays on the route at terminals (notes/T-047.md). One instanced draw for
  all trains, positions from keyframes on the GPU. A click on any car selects the train; one drawn
  over a station selects the train.
- **On its own track** [T-658b]: as double track parts into two (about zoom 17, "Lines and
  track"), a train moves from the centre line to the right-hand track of its direction of travel
  (2 m out once apart) and narrows from its pill width to its true 3 m, so two trains passing go
  by side by side without touching. Where double track narrows to single the train drifts back
  to the centre over the taper (`sampleLane`, one value a 50 m sample). A train turning back
  stands on its arrival track and steps across to the other as its next trip starts. Clicks
  find it where it is drawn (`laneShiftM`).
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
  consist: an outline in the map ink (dark, light in the dark theme), a band in the casing's colour
  (white, near-black in the dark theme), a ring of the train's shape 6 px out in the ink and a soft
  glow, so it reads even at zoom 11-13 [T-543]. A click tries the selected train first. It is followed onto its
  next trip [T-245, [ui-inspectors.md](ui-inspectors.md)].

## Platform crowds

[T-164, T-219, T-236, T-553] From zoom 14 the white of each platform box fills with a dark warm grey
(#77726d, kept 1.5 px clear of the rim or the blue edge) as riders wait. Each direction of travel
has its half of the box along its length (close in on double track, its own platform beside its
track, "Stations", T-658c); a platform whose trains all use one side (a terminal)
fills as one. A side fills from the middle outwards by its riders' share of the room
(`platformRoom`, [track-and-service.md](track-and-service.md) 6.6: 1,250 a side on 150 m), full to
both ends at that or more. A side is red instead of grey once it is full [T-559]; the red stays inside the box like the grey (T-713). A standing train covers the box's middle; from zoom 16 the sides show beside it. The crowd
grows between trains at the steady arrival rate, jumps a transfer walk after another line's train
leaves with riders changing to it [T-554], and drops when one of its own leaves, worked out on the
main thread at most once a game second, only while drawn.

## Life at stations

[T-686; `map/stationLife.ts`, `stationSound.ts`, `faces.ts`] From zoom 14 (`STATION_BOX_ZOOM`), with
the clock running, a train pulling in at a built station in view is found by its stop offset passing
the station's between two samples (eight a second; only the lines stopping in view are read). Below
zoom 14, or with Faces and Sound both off, no timer runs and there is no canvas. The riders come
from the boarding account: those who get on are the platform's crowd when the train arrives plus
arrivals until it leaves, less those left behind; those who get off are the load carried in plus
those who got on, less the load carried on.

- **Faces**: one per 20 riders getting off (rounded at random), at most 4 per train, 5 per platform
  and 36 in all. Each rises 30 px from a random point along the platform over 2 s and fades, 14 px
  across at zoom 14 to 20 px at 17. Happy (pale yellow, blushing); sad (blue, brows raised at the
  inner ends, a tear, a frown) when the train before left 20 or more riders behind (`CROWDED_LEFT`);
  tired (drained yellow, heavy lids, a flat mouth) when the riders waiting have waited 10 minutes
  or more on average (`TIRED_WAIT_S`) [T-696, T-802]. The average counts the newcomers at half the
  time since the last train left and the riders it left behind at that plus half the gap before it;
  the hour's first arrival at a platform has no figure and is happy. Sad wins over tired; no face
  is red. Three sprites on a 2D canvas over the network; a frame loop
  runs only while a face is alive. Not in the demand views' way: the bubbles draw over them.
- **Sound** (WebAudio, no files): one soft sine note as the train pulls in, the line's own, one of the 12 semitones from C5
  [T-706]: a city's first five lines (a line's city is where its first stop is) take C, E, G, D, A in
  that order, later lines a semitone from their number. The note is chosen once, when the arrival
  code first meets the line, and kept in the save (`notes` in the header: note and city, per line id,
  kept after the line is deleted), so deleting a line changes nobody else's; a line's rank is how
  many lines were given a note in its city before it. Older saves get theirs on load, in line
  number order. A line with no stops or in no city is not stored. A platform with sad or tired faces sounds a minor third lower, softer, sagging [T-696]; then pops (a short
  falling blip) as riders get off (higher) and on (lower), 1 to 4 each by the rider count. Silent
  until the first click or key. Volume is the setting times a zoom gain (a quarter at 14, full from
  17), shared down by the arrivals sounding in the last second; a second one sounding together gets
  the note only, and none beyond 3; panned by where the station is on screen. At the second speed
  only the note and two pops; none at the third.
- Both are on by default and set in Settings (Stations up close).

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
- **Reversal icons** [T-797; `map/reversalMarks.ts`]: over every station where a line's trains
  reverse partway (`LineStats.reversals`, from `TrackApi::line_reversals`) and every bare track end
  they reverse at (`LineStats.endTurns`, `TrackApi::line_end_turns`), a 30 px amber disc with
  a dark U-turn arrow, floating on a short stem just above the place so the stop editing marker
  on it stays clickable. It keeps its size at every zoom and never hides for names or zoom, so a
  reversal, most often an accident, shows from far out; it shows during stop editing too. Hover
  (a game tag) names the station ("Track end" at a bare end), each line "<line> reverses here", and
  that 12 trains an hour at most reverse there (from each side, at a station). With select, a click selects the first of those lines (not
  while stops are edited); with another tool it takes no pointer events.

## Hover tags on the network

- **Track** [T-157, T-526, `map/trackHover.ts`]: with the select tool or a build tool, the pointer on
  a line or track shows a tag at once saying what the track display draws there. Line: the name. Traffic:
  the name and "41,200 riders per day" (or "0 riders per day") on that stop-to-stop stretch, both
  directions together, for the day shown (counts to the nearest 100 above 1,000), with a third line under the purpose filter
  ("Shopping and errands trips only"), and on track at capacity "Track at capacity at high demand" in red [T-819]. Whatever the display [T-473], the tag also carries a row
  for the elevation at that point (an up and down arrow, then "+1" on its Elevation colour) and one
  for the speed limit (two right chevrons, then "120 km/h" on its Max speed colour), each colour at
  half strength behind dark text (`map/trackTagRows.ts`), and last every line using the track as
  line bullets, in the save's order, wrapping, over a line as over bare track (the markup and CSS of
  the `Chip` in `ui/widgets.tsx`, built as plain DOM by `lineBullets`; no row when none runs
  there) [T-532, T-559]. A planned line adds
  "Not running yet", a broken one "Not running"; before the first solve the traffic view says
  "Calculating demand". The tag follows the pointer along the track; leaving, pressing, dragging or
  zooming hides it. It follows the pick order: with the select tool, bubbles, capacity tags and
  handles above the network keep it hidden; with a build tool it sits above the pointer, clear of
  the tool's own tag below, and leaves out the station demand circles. A mark on the track names itself
  ("Flat crossing" or "Flat junction" with the elevation, "Ramp up to +1", "Tunnel portal down to −1",
  "Blueprint" on blueprint track), over the track's own tag; a junction without a mark says nothing. The
  stop editor and the trip tool have no track tag. The track tag and the build tools' tags are
  drawn over the bubbles' tags.
- **Trains and stations** [T-194, T-305, T-313; `map/networkTips.ts`]: the same tag (wrapping at 320
  px). A train: "Line 1 train", "340 riders aboard" and "352 seats, 1,280 full". A station:
  its name, then its lines as bullets, each once (a station drawn as its own dot or box lists its
  own lines) [T-559]; with demand circles on, "41,200 entries and transfers per day" (or entries, or transfers, as the station ridership choice says) [T-722]; where crowds
  are drawn a row per line and direction with anyone waiting (the line's bullet, "to Van Cortlandt
  Park-242 St", "120 waiting"), the count in the alert red while that platform side is full, or
  "Nobody waiting on the platforms". On a dot, the crowd lines are its pie's numbers ("1,257
  waiting", red when full, then "Fullest platform 45% full"). A
  demand circle away from any station gives its busiest station's name and count. Refreshed each
  game minute while shown.
- **What else is here** [T-732, `map/hoverRest.ts`]: with the select tool, when the pointer rests
  250 ms on the map, a list under the hover tag names what else is under it, at most one demand
  bubble (its count line, behind a small circle), one station (its name, behind a small platform
  box) and three lines (bullet and name, in pick order), the thing the tag already describes
  counted in those limits and left out; nothing else under the pointer, no list. Clicking an entry
  selects it (a bubble beneath the selected station selects that bubble). The pointer may cross
  from where it rested to the list, inside the hull of that point and the tag with its list,
  without either moving; leaving that area, pressing, scrolling, a camera move or an edit to the
  network (track, which lines run where, a station placed or renamed, a line renamed) drops the
  list; demand and the clock's figures changing do not [T-753].
  The query runs once per rest, never per move. Pointing at an entry lights that thing on the map
  (below) in place of what the pointer rests on [T-741].
- **Lightening** [T-730, `map/hovered.ts`]: with the select tool, what a hover tag describes is
  drawn lighter at once, mixed toward white in both themes: a demand bubble 30% (the disc whose tag
  shows, any disc layer), a line 30% along its whole length, bare track that edge, and a station's
  dark rim 60% over its whole complex (with the bridges and connectors between its members). A
  train, a mark, an airport badge and the station demand circles light nothing; build tools light
  nothing. It is a uniform in the shaders: a change redraws once and rebuilds nothing.

## Picking

Clicks reach only what is drawn on top [T-486]: a click never acts on something drawn beneath
another clickable thing. Picking follows drawing order [`map/pick.ts`, `main.ts` `selectAt`], top
first: editing handles (a selected stretch's corner squares, which take their own presses, and the
select tool's squares on blueprint points, which pass over the bubbles, their hover tags and the
trains beneath, `map/endMarkers.ts`); the selected station (cut out of the bubbles); demand
discs (commuter bubbles over longer-trip discs); capacity tags (under
the discs; one that opens nothing selects what is under it); trains (the selected one first), then
stations, lines, junctions and track; within a layer the last drawn wins; hidden layers do not
intercept. Station names, airport badges and the stop editor's markers take no clicks: a click on
one reaches what is under it (a badge sits on its airport's destination bubble; the stop markers' station
is picked by the stop tool). Network
hits reach 7 px from the centre line, or half the drawn width where a line is wider [T-227].
Demand discs
drawn under 8 px in radius are hit within 8 px of their centre [T-566, `MIN_PICK_PX` in
`map/demandHover.ts`; the hover tag follows the same test], unless a disc drawn above them holds
the pointer; where those areas overlap the nearest centre wins.
Clicking the line already selected selects the track under it; a click on track several lines run
over (one edge, more than one line on it) selects the track whose inspector lists those lines to
choose from, in every track display [T-654, `pick`'s `chooseShared`]. Clicking a selection's longer-trip discs keeps the
parent selection. Selected editing handles stay draggable above the
demand view.

**While building** [T-526]: with a build tool on (track, station, delete, blueprint),
the demand views (bubbles, inter-region bubbles, airport badges and their labels) are drawn
beneath the network, blueprints and stations, since the tool is about track. The tools' clicks
never pick a bubble, and a bubble's tag shows only where no network and no tool tag is under the
pointer.

## Zone outlines

[T-131, T-139, T-310, T-800; `map/cityBoundaries.ts`, `map/boundaryGeometry.ts`,
`map/coarseOutlines.ts`] Each registered pack's footprint (its dissolved county boundary) has a thin
solid outline (0.7 px, 36% opacity) at regional zooms, fading between zoom 8.5 and 10.5. Every
zone has one from the start: `packs/outlines.json` (pipeline/zone_outlines.py: all registered
zones, simplified to 300 m, 0.65 MB gzipped) is read at startup and drawn at once, and a zone's own
boundary replaces its coarse outline when it loads for a view between zoom 5 and 10.5. A border two
zones share is drawn once, as a line of the first zone loaded, so every edge has the same weight
(`OutlineEdges`; the coarse file keeps shared runs identical in both zones). The focused zone keeps
a thin boundary at closer zooms (darker outside stroke at 36%, lighter inside at 14%), and draws
the borders it shares itself, so a neighbour's outline leaves those out. A subtle shade darkens the basemap outside its exact footprint,
enclaves included. These are demand areas, not municipal limits. They sit under labels and game
overlays, never take clicks, and survive basemap style changes. A change of focus swaps the shade
in one step, never through an empty frame; the footprints are ready from startup.
