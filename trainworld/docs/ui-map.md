[Back to the spec](../SPEC.md) · Interface: [panels](ui-panels.md) · [inspectors](ui-inspectors.md) ·
**the network on the map** · [demand views](ui-demand-views.md) · [building](ui-building.md)

# 8. Interface: the network on the map

Track, stations and trains are drawn by our own WebGL2 renderer on a canvas over the basemap
(`map/renderer.ts`; why and how in [architecture.md](architecture.md) 7). The basemap is
Positron, repainted (`map/basemap.ts`) so the network reads first: warm paper land, soft teal
water, sage parks and woods, place names faint grey. Roads have three weights (motorways, main
roads, everything else), one colour and no casings; the smallest classes appear only from zoom 11
(streets) and 14 (paths); minor road names, road shields and the basemap's railways are hidden.
Positron has no shop or food icons, so the airport symbols are the only icons left: the black dot it draws beside a city, town or village name is removed (the name stays, centred on the place) because it reads as a gameplay dot [T-978]; the start menu's map hides those names too [T-974].
Basemap labels are plain one-colour text with no halo, the palette's label grey a little darker
(light) or lighter (dark) so it holds over water and shade [T-1692]. Fewer show at once: suburbs,
islands and villages start at zoom 12, towns at 9, small rivers at 12, river and strait names at 8,
main road names at 13.5, airports at 12, and every label keeps 10 px clear (28 px round city and
town names), so of two close cities the larger wins. The dark theme
is the same three hues deep and muted (dark warm brown-grey land, deep teal, dark sage), by paint
properties only, so switching theme reloads no style or tiles [T-039, T-676]. The light and dark
map colours (land, water, park, road, labels, the casing, the map ink, a switched-off blueprint,
the shade outside the focused city, the save thumbnail's track) are one table,
`game/shadeTable.ts`, whose named palette values at the top are the ones to retune by hand (read
through `game/shades.ts`) [T-543].

**Satellite** [T-3249]: Settings > Map has a Map | Satellite switch beside Flat / Globe, off by default and remembered with the other map settings; its hover says the imagery is low resolution outside Japan and the US. On, three raster layers (`map/basemap.ts`) sit over the drawn map's fills and minor roads: Sentinel-2 cloudless 2024 from EOX everywhere (10 m, soft past zoom 14), GSI's photos over Japan (to zoom 18) and USGS NAIP over the contiguous US (to zoom 16) on top of it, each fetching only inside its box, each source's credit in the map's attribution control. USGS answers blank tiles for Canada and Mexico inside its box; a small protocol (`twusgs`) turns those into 404s so the Sentinel-2 layer shows. The drawn map stays under the imagery at its own 0% look in the panels' theme (fills, water, buildings, minor roads, names following the slider as in Map mode); motorways and main roads (26% and 18% white) and boundaries (white 30%) are drawn over the imagery. The Base map slider is the imagery's opacity over that map, untouched in colour (no desaturation): 0% looks like Map mode at 0%, 100% is the imagery alone, and between them it fades smoothly [T-3267, T-3278]. From 50% up (`SAT_INK_FLIP`) roads become white and names light (#f2f2f2) with a dark halo (the no-halo rule gives way over imagery); below it they keep the drawn map's own paint. The map's own ink (casings, real network, station circles, bubbles, the focus shade) uses the dark theme's colours while the imagery shows at 50% or more, and the theme's own below that (`mapShade` in `game/ui.ts`); the dark theme always uses dark ink. Panels, save pictures and Community thumbnails follow the theme as before and never include the base map. The vector tiles still load: labels and the road feeds read them.

## Real network (reference layer)

Settings > **Real world network**, a choice of None, Passenger (the default) and All [T-3289; a stored on is Passenger]: faint lines (15% opaque in the light theme, T-727;
30% in the dark one, in lighter colours, T-1291) where railways run in the real world, for
reference, view only, whatever the Base map slider says [T-1291]; 40% (T-697) while the track tool snaps to them (the
Build tab's snap switch, [ui-building.md](ui-building.md)), which also shows them with the setting off (`map/realNetwork.ts`, T-682). They sit above the basemap's
roads and under its labels, so under our own network (which is drawn on the overlay canvas above
everything). Three classes by colour and width, in both themes (`realMain`, `realOther`, `realMetro`
in `game/shadeTable.ts`): main line, other passenger rail, metro. Light rail is drawn as other rail: the tiles carry no usage, so
S-Bahn and S-tog track come with Stadtbahn, US light rail, the DLR and Metrolink [T-3285]. Trams, monorails,
funiculars, narrow gauge, heritage lines and track no line runs over are drawn only under All, which shows every feature the tiles carry; yards, sidings and freight-only track are not in the tiles (dropped when they are built), so there is no freight. Snapping goes only to passenger track, so while it is on the layer shows only that, whatever the setting says. Lines appear
from zoom 6 [T-2748: from zoom 4 a view of Europe drew 41 regions' tiles, most of a frame's cost on
the map a new game picks a city on]; each region's tiles are added when the view first touches it, nearest first, and
nothing is fetched while it is hidden. The lines are noritetsu's per-region tiles (OSM and
the national registers, one `track` layer); the tiles carry no line type or name at low zoom, so
there is no high-speed class and no hover yet.

## Terrain (hill shading)

**Terrain**, off by default [T-1270], switched by the mountain button beside the Build tab's price per km
(T-1394) or Settings > Show: MapLibre's hillshade over the land cover and
under roads, from the Terrain Tiles on AWS Open Data (Terrarium PNG, 256 px, fetched to zoom 12 and
stretched past it; `map/basemap.ts`). The water is drawn again over it (`water_over_shade`), so the
sea floor's relief stays hidden. Shadow and highlight follow the theme (light: warm brown shadow,
white highlight, exaggeration 0.35; dark: black shadow, 0.45) and the base map contrast slider mutes
them with the rest. **The shade never shows on water** [T-1552, T-2628]: a terrain tile is held back (a
`twdem://` protocol in `basemap.ts`) until every basemap vector tile over its ground in view, at the zoom
the basemap draws now, has loaded, so the shade comes with the water and land however slow they are.
There is no time limit; it gives up only when the basemap has stopped loading (a vector tile that failed
leaves bare ground, water or not). So the sea floor, lake beds and river channels in the tiles never
show: by the time a tile is drawn, `water_over_shade` covers them. Pricing never reads the shade. **Flat ground is drawn flat** [T-1881]: the Terrain Tiles are a surface model in
much of the world (SRTM in Asia: towers, raised roads and radar speckle read as hills in dense
centres), so from tile zoom 10 (`FLAT_MIN_Z`) a tile that meets loaded terrain blocks goes to a worker
(`map/demFlat.ts`, `workers/demFlat.worker.ts`) and comes back with its heights over flat-class ground
blurred (sigma 400 m, the 3 x 3 tiles around it giving the blur its reach), blended by how flat the
ground is: each class cell the share of flat cells within two cells of it, bilinear between cell
centres, so a lone hilly cell in flat ground is mostly drawn flat. What the shade shows as relief is
then what the track pays for. A tile waits up to 1.5 s for the terrain blocks under it (only blocks with a
file, so open sea does not wait) and is drawn again when a late block comes; a tile where no block is loaded is
drawn as it comes. `?rawdem=1` draws every tile as it comes. Off, no layer draws and no tile is fetched. Its hover in Settings gives the
terrain price factors (track-and-service.md 6.4). Mapterhorn's hosted tiles were left aside: no terms
for use by an app are stated; for a published game the fallback is our own Terrarium tiles on R2
cut from GEDTM30 (notes/T-1270.md).

**The class tint** [T-1270 stage 3]: with Terrain on and a build tool on (track, station, delete,
blueprint), hills (both classes) and mountains get a faint brown wash, stronger by class (light:
brown at 8%, 15%, 24%; dark: a lighter tan at 6%, 11%, 18%); flat ground is bare. It is drawn from
the terrain blocks the clock worker sends the page (track-and-service.md 6.4), one "terrain" message a block
(`game/terrain.ts`), so it shows exactly what is charged, and nothing where no block is loaded. `map/terrainTint.ts` draws
one image over the view and half its size again around it, rows even in mercator, each hex (T-2890,
T-3372: the hexes track is charged by, track-and-service.md 6.4) the class of the lattice cell at its centre
(nearest, no smoothing; halved in size once a hex is under an image pixel), redrawn when the camera stops; above the hillshade, under the
water, roads and labels; hidden below zoom 9, where a view is wider than the blocks the view brings in. The Terrain setting turns the hillshade and `water_over_shade` on and off together, so the tint stays off the water there too. Washed ground
has no hover of its own; the track tool's tag names the class and its factor [T-1797].

## Zoom bands

- **Below zoom 9** (about 160 km across, a whole region) only the lines are drawn [T-205, T-542,
  `STATIONS_ZOOM`]. Stations and trains are neither drawn nor picked, so a click or hover finds the
  line. A selected station or complex is still drawn and clickable. The station ridership circles
  and reversal icons hide with the stations, hover and all [T-1274, T-1772]; the other demand
  drawings are unchanged.
- **Zoom 9 to 10**: every station is its own dot, drawn and picked; trains are not.
- **Zoom 10 to 14** [`TRAINS_ZOOM`]: stations are dots; trains are pills, drawn and picked from 10.
- **From zoom 14** (`STATION_BOX_ZOOM`, T-553): a station is its platform, trains show their cars, and
  platform crowds are drawn.
- **About zoom 17** (T-658; set by 4 m on screen, so the same in every city):
  double track opens into its two tracks ("Lines and track").
- Station names and capacity dots hide below zoom 12 [T-101]; reversal icons hide with the
  stations below zoom 9 [T-1772].
- **Zoom limit** [T-3368; `map/mapView.ts`]: the map zooms to 22, but after a finger press (`touchInput`) to 20, since past that a track drawn to scale is wider than a phone screen.

## Lines and track

### Widths and double track

- **Widths** [T-191, T-205, T-1821]: a line is 1.1 px at zoom 9, 5.1 at 12 and 8.6 at 15 (linear
  between, held outside; `LINE_PX`: 0.85, 3.9 and 6.6 times `LINE_VIEW_WIDTH` 1.3; the Traffic view
  scales its widths by 0.85 / 1.3, so they stay where they were); bare track 0.4, 1.8 and 3; a casing 0.6-2.4 px wider (white, near-black #141517 in
  the dark theme [T-039]; line colours are the same in both), drawn as a fixed margin
  under the stroke and not part of a traffic width [T-267]; a selected line or track 0.5-3 px wider
  still, and a selected line or group [T-1812, T-2058] draws every other line muted: solid, 45% of
  its width inside its usual casing, its colour mixed halfway to the casing's (Anita: "thin and
  faded"; solid, so overlapping muted lines do not build up), and their trains at 55% (30%
  while stops are edited or the pointer is on a column of the group view; `?perfOff=muteThin` draws
  the lines at those strengths instead, as before T-2058), all of a selected
  group's lines drawn selected together (`selLines`, at most 16; bus lines take the one line only). A stroke thinner than a device pixel is drawn a pixel wide and that much fainter. Line and
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
- **One-way legs from their own track** [T-3117, `legShifts` in `map/network.ts`]: where one-way
  single track meets double track at a junction (a flyover's three), its end is drawn 2 m off its
  centre line, away from the nearest other track leaving that node its way, easing back over 60 m,
  so each leg carries on from its own track of the double. Its one-way arrows keep clear of
  junctions as of stations, and of every mark on the track (ramps, flat crossings and junctions,
  unusable one-way track, T-3189): one that would sit on one is skipped. A line is drawn on the track its return run takes too, where that is
  not its path backward (a flyover's other leg): stroked in its colour, in its stretches, and
  picked out with it when selected. That run's trains are drawn along it, as the sim runs them [T-3250]:
  the clock worker samples the run on its own (`run_polyline` run 1) into a second half of the line
  table (row lines + line, `backSlotOf`), and the train shader, `tripAt`, picking and the train guard
  place them along that row as a run 0 of its own.
- **Turn-back crossovers** [T-658d, `turnbackStrokes` in `map/network.ts`]: close in only, where
  a line ends at a double-track station, the crossover the game assumes there (6.2) is drawn in
  bare-track grey over the line's two tracks, 50 m long: a scissors (both diagonals) just outside
  a dead-end terminus's platform, and at a through station one diagonal just outside the
  platform on the side the line arrives from, from the arrival track to the departure track. Not
  picked, no hover of its own.
- **Single-lead crossovers** [T-3161, `singleLeadStrokes` in `map/network.ts`]: close in only, where
  single track that runs both ways joins one track of double track at a junction, the link the move
  table gives it to the double's other track is drawn as the turn-back crossovers are: one diagonal
  40 m long along the double leaving the node on the other side, from the joined track at the node
  to the other track, faded unless both pieces are built. It shows where the flat-junction badge's
  crossing is. One-way single track draws none.
- **Passing tracks** [T-2233, `passingTracks` in `map/network.ts`, `passingState` in
  `map/renderer.ts`]: from zoom 14, a station with passing tracks on double track has an outer
  track either side in bare-track grey, along the platform and bowing back onto the running track
  over 100 m past each end (eased both ends), through the box and outside it, faded on a blueprint
  station. The bow follows the track on through plain nodes; where a junction, a track end or
  another passing station's outer track comes sooner, it is shortened to end there (at least 10 m;
  half the room each where two passing stations face each other) [T-2248, `passLead`]. The outer tracks are 6 m from the centre line on the ground, or further out where trains
  are drawn wider than they are: a train's drawn width and 1.5 px from it, so a train on one never
  touches one passing (10.5 px at zoom 15). The box is at least wide enough for a train on each outer
  track (about 36 px at zoom 15 against 11.25 for others); clicks, the selection outline and the
  rest take the same width. Not picked, no hover of their own.
- **Close in, trains keep to their side** [T-319, T-3297]: each line's run 0 is drawn on the track on the
  game's running side of its direction (the right, or the left where trains keep left) and run 1 on the other. Under a line's two tracks is a band as wide as the pair or its traffic
  width, whichever is wider, in the line colour mixed 60% towards the casing, over one casing
  band (Anita, 2026-10-19: the tracks over the traffic width). Bare track is two grey tracks as
  much thinner than 2.4 m as it is thinner than a line; blueprint track is two blueprint tracks,
  each its own rails and middle. Single track stays one track on the centre line, 2.4 m wide;
  double track narrows into it over 40 m where the two meet at a plain node (`lanePoints` in
  `map/network.ts`); at a junction a single-track branch leaves from the centre line. Through a
  station, two-way single track splits into two tracks as double track's are [T-3212,
  `singleSplitEnds` in `map/terminalBays.ts`]: two along the platform (half of it each side of a
  through station, all of a dead end's), easing back to one over 60 m past each end, carried on
  through plain nodes. It shows only once double track's two tracks part (from 4 m at 7 px), opening out of single track's width; further
  out the track and its lines keep single track's width through the station (`F_SPLIT`; Anita,
  2026-11-11: no thickening before the tracks show). Trains of opposite directions meet there (the capacity's passing points),
  and stand on the two tracks; the station has a platform each side, as on double track. A split
  that reaches double track within its 60 m stays two tracks to it, and double track meeting
  single at such a station does not narrow; one that reaches a junction first eases over what
  room there is. One-way single track, and single track joining one track of a double at the
  station, stay one track. Quad
  track is four tracks 4 m apart; a crossover between double tracks is two double tracks in an
  X. Clicks and the hover reach half the pair's width from the centre line once it has grown.

### High-speed track and lines sharing track

- **High-speed track** [T-817, `F_HSR` in `map/network.ts`, the stroke shaders in
  `map/renderer.ts`]: double track's colours with a thin centre line along the stroke, 18% of its
  width: the casing colour on built track (light, dark in the dark theme), the rail colour on
  blueprint and on the drawing ghost (at least 0.6 px there); lines of high-speed trains over it carry
  it too, a line of ordinary trains is drawn as an ordinary line there [T-1494]. Built
  high-speed track and its lines are wider while drawn as one stroke, 1.4 times at zoom 9 and
  below, 1.25 at 11, as wide from 13 (`HSR_WIDTH`), and their centre line comes in only as the
  stroke widens from 3 to 5 px (`HSR_STRIPE_PX`), so far out they are a solid stroke a little
  heavier than ordinary track [T-1097]. Once the two tracks part, the line runs between them in the
  track's or the line's own colour.
- **Lines sharing track side by side** [T-1799, `map/lineBands.ts`, design notes/T-1799.md], only
  with the Settings checkbox "Lines side by side" on (on by default, T-2346, a stored choice from before settings version 12
  dropped once since the old default was stored with the rest; `display.lineBands`,
  set on `renderer.lineBands` in main.ts, redrawn at once). In the Line view: on an edge with two or more lines each line is a band 80% of a line's width in its
  colour less 1 px (`BAND_GAP_PX`, T-2126: the casing's colour, white in the light theme and dark in
  the dark one, shows between bands at every zoom), its sides and round caps smoothed on a sharp
  ramp (1.6 x the distance to the edge, minus 0.5 px) so the gap keeps its width at every angle and
  a gap falling between two pixel columns still leaves them 70% casing, and the caps no fatter than
  the sides, so curves made of short segments do not pinch it (T-2183, T-2223, `v_aa`); one casing
  around them all (every casing is drawn before any colour). A band is shifted along its own
  segment's normal, so line and track strokes keep no segment under 8 plane m (`MIN_SEG_M` in
  network.ts, T-2258) unless it is all of its edge: a point closer than that to the last one kept
  is dropped, the edge's last point moving the segment before it.
  Order on each edge, left to right: of two lines, the one turning more to the left where they
  part, looking ahead in the lower-numbered line's direction, else looking back, else the lower
  number on the left of that line's travel; an edge's order is by how many of the others each line
  is left of, which is the pairwise order wherever that is one. So bands cross only at a node where
  the two routes part or meet, or where three or more lines' partings go around in a circle (some
  pair has to cross). **Tracks side by side** (corridors, T-1799 stage 3, T-1848; `map/corridors.ts`)
  are one ribbon too, decided per place, not per edge: every edge with lines is sampled every 25 m
  (`SAMPLE_M`), and a sample records the other edges running the same way (within about 25 degrees)
  within 20 m (`CORRIDOR_M`; a four-track trunk is two double tracks 8 m apart, T-1305;
  player-built double tracks beside each other count too), out to 30 m once within 20 m along the
  way (`HOLD`), and not the next stretch of its own track where the two meet at a node; each
  record is mirrored, so both tracks see each other at the same places. The ribbon at a place is
  every track linked there, track to track (a line in it once), unless it is over 70 m across (a
  throat or yard, `SPAN_M`) or over 32 tracks, when each track is left as it is. Its tracks' order
  is decided once for the whole network, pair by pair (T-2058, so two lines never swap sides where
  their tracks do not): of two tracks, by their lowest numbered lines, the one whose track is on the
  left over all the places the two are beside each other (offsets summed along the way, in the lower
  numbered line's travel), and where that averages under 6 m (`COINCIDE_M`: single tracks, or the real
  New York save's lines each on its own edge) the lower number on the left of its own travel; a pair
  never beside each other directly takes its order on the ground there; the ribbon's order is by how
  many of the others each track is left of; each edge's own lines in its order. The ribbon is centred on its tracks' middle at that
  place at every zoom: a band's shift is its slot in line widths plus the metres from its track to
  the ribbon's middle, which the shaders draw at the zoom in view (T-2132; `a_band_m`, trains and
  picking alike). **Along each line** the ribbon it is in changes only where the lines
  beside it change for at least 200 m (`CORRIDOR_MIN_M`, a platform's length: a neighbour splaying
  round a platform or touching at a junction is smoothed over), unless a line on the line's own
  edge would go missing from the ribbon (lines on one edge are always side by side); and a ribbon
  left and rejoined with the same lines is held across a gap of up to 2 km while the line's track
  stays within 50 m of a track of each other line in it (`HOLD_RUN_M`, `HOLD_M`; T-2132, so two
  pairs of tracks 20-50 m apart do not close up and spread again zoomed out; `?perfOff=bandsHold`). **Every change
  of a band's shift is a long S-curve** (T-2058, Anita: lines "never like suddenly turn or translate
  if they don't suddenly turn in the actual track"): a smoothstep 250 m long for ribbons of up to two
  lines and 50 m more per line beyond, at most 800 m (`SMOOTH_M`, `SMOOTH_PER_LINE_M`,
  `SMOOTH_MAX_M`), every line of the two ribbons there taking the same length so a ribbon moves as
  one; changes closer than that are added together into one curve (each knot a mix of the pieces
  around it, the Traffic view's widths mixed by the same weights). Where it sits: a change at a node
  between ribbons of different sizes starts at the node on the side with fewer lines, counted on the
  line's own edges first and then in the ribbons (lines part leaving side by side and close up as
  they curve out, merge the other way; T-1799 stage 4), except that where the ribbon on that side
  lasts less than the curve's length (a loop track beside a trunk) and the line's own edges keep
  their lines, it ends at the node on the side with more, so the line is on its own track all along
  the short stretch (T-2060); a ribbon change along the way where the line's track comes beside
  tracks it joins at a node within half a curve's length (or leaves one edge just before) is placed
  as at that node, so the merging lines are in their places where the tracks meet (T-2059); any
  other change is centred on it. Each line works this out on its own track, where the same change
  can be at a node on one track and along the way on the one beside, so the changes of one ribbon
  (by its lines) within 100 m of each other then take one placement (T-2140, `ALIGN_M`; Shinjuku's
  Traffic view bulged where they did not): that of the line where a line joins its track, else
  where its change is at a node, else the lowest numbered, its curve's middle at the nearest point
  of each other line's track (within 80 m, `ALIGN_NEAR_M`), else as far from that line's own change
  into or out of the ribbon (`?perfOff=bandsAlign`). A line alone keeps its full
  width; a line's band is drawn by knots (`BandProfile`, `bandPoints` cutting the line's segments
  at each); `?perfOff=bandsSmooth` gives T-1848's eases (80 m along the track, 30 m at a node) and
  per-ribbon order, `?perfOff=bandsOrder` the S-curves with per-ribbon order, `?perfOff=bandsPlace`
  T-2058's placement without those two exceptions. Zoomed out the bands fold into one stroke as a line narrows from
  1.75 to 1.25 px (`BAND_FOLD_PX`, about zoom 10 to 9.5), and close in they close up onto the
  centre line as double track grows to its two tracks (from about zoom 16.5), where the lines are
  drawn over each other on the real tracks again. In the **Traffic view** (stage 2) the same ribbons,
  order and knots, each band as wide as its own line's riders (`trafficBands` in `map/lineBands.ts`,
  from `buildLayers`, with each knot's ribbon from `NetworkBuffers.corridor`), so a trunk is the sum
  of its lines (segments between knots that differ are `F_EASE`). A station's box
  or dot stays as wide as for one line (T-1821); through the box (T-236) the bands take the Line
  view's slots, narrowed to fit; a ribbon of many tracks runs past a box on one of them as it is (a
  box holds only the middle bands). A click or hover takes the band under the pointer, else the nearest within the slop; trains
  run on their line's band (`sampleBand`, or the Traffic view's; `tripAt`'s `bandM`). `CORRIDORS` in
  `map/lineBands.ts` turns tracks side by side off (only lines on one edge then). Not yet: the
  Elevation and Speed views are drawn as below; `?perfOff=bands` turns the bands off whatever Settings says.
- **Lines sharing track drawn over each other** [T-190, T-533] where they are not side by side
  (the setting off, zoomed out, close in, the other views): one order for all lines: a line with no trains (fleet
  zero) goes under every line that runs trains; within each group the shorter route (end to end,
  one direction) is on top, ties to the higher id. The line on top wins a click. It is rebuilt when
  lines or fleets change.

### Track display

- **Track display** [T-105, T-703; on the map under the demand buttons], five choices, remembered per browser:
  - **Line** (the default [T-1101]; a stored Traffic from before settings version 10 is dropped
    once, T-1244): uniform width in the line colours.
  - **Traffic**: line colours, width by riders over the day shown
    ([ui-demand-views.md](ui-demand-views.md) "Line width by riders"). A line on track at
    capacity [T-819] (an edge of a track or single-track section, or either track of a flat
    crossing, at `FULL_RHO` for the demand level in force, as the capacity tags take it;
    `TrackApi.full_track`; platforms, termini and junctions keep only their tags) has a red band
    outside its casing, each side 0.875 px at most plus 7.5% of the line's own width (a thin local line
    and a wide traffic line alike show it, T-1663, T-1676), in dashes a 16 px period, 60% of it drawn (`u_full_dash`), measured along the line's run of segments on one edge so every dash has one length round curves (T-2184, `a_full.y`); #c0392b, #f0584f in the dark theme; `fullTrack` in
    `game/shadeTable.ts`), drawn in a pass of its own before the lines' casings, so a red line keeps
    its casing between and a neighbouring line's casing covers the band where they touch, with
    0.5 px of the casing's colour outside the band (a pass under it, `u_full_out`; T-1284) so the red
    reads as a stripe on a red or orange line; the fixed part of band and outline scales with zoom
    (30% at zoom 9, 60% at 12, full from 15; `fullScale` in `map/renderer.ts`, T-1549), and the
    line's body is darker by 14% (`FULL_DARKEN`). A crowded stretch between two stops [T-2236] (its
    busiest hour's trains, in the fuller direction, carry on average past 100% of their rated load, 160
    a car, as the hover rounds it: `CROWDED_FROM` 1.005 in boarding.rs; the trains leaving the
    stretch's first stop in each hour of the day, from the boarding account of the hour's kind of
    day, track-and-service.md 6.6) has the same mark as strong as that load: 50% of full strength
    just past 100% plus the rest in proportion, all of it at crush's 150% (`FULL_FLOOR`,
    `CROWD_LOAD_FROM`, `CROWD_LOAD_STRONG` in `map/rendererShaders.ts`), the band mixed from the
    casing's colour towards the red and the darkening scaled alike. Riders waiting for a train with
    room (trains at crush) are not a map mark: the station panel's platforms and the trouble list
    say them. Where lines sit side by side
    the crowding marks only that line's band, with the same red dashes along both its sides, outside its
    colour and as wide as a line's own, in their rim of the casing's colour, drawn in the same stroke as
    the line's colour and so in that line's order: a line above it covers them, and they lie over only
    the lines beneath (T-2184; T-2347, Anita: the dashes stay as they are, but in the line's draw order),
    and the band darkens; track at
    capacity keeps its red band, which goes around the whole ribbon (T-1799 stage 2). The hover adds "Track at capacity at high
    demand" in red, or "Peak load 132% at 08:00" (that busiest hour, the line's local time) with the
    load on its crowding colour, the line panel's crowding dots' (`crowdColour` in `game/palette.ts`). The Settings switch "Crowded trains" (on by default) shows or hides both marks
    and their hover lines.
  - **Level** (`height` in code; hover "Show track level colors"): the track coloured by level, blue-green to indigo above ground (open sky), green
    at ground, warm below (deep): +3 #4a2c9a, +2 #2f6fd0, +1 #16a39a, 0 #58b04a, -1 #f0b928, -2 #ee7a1a,
    -3 #c2351c (`LEVEL_COLOURS`, T-413, T-478, T-480; neighbours at least 30 apart in CIELAB). The same table colours
    the track bar's level buttons, the Settings key and the level in the track tags.
  - **Speed** [T-1271]: built track a running line with trains uses is coloured by the trains'
    speed there, the rest by its speed limit (the geometry's, curve limits included: 160 km/h on
    ordinary track, up to 320 on high-speed track). The trains' speed at a stroke segment's middle
    is the run profiles' (speed against distance along each run, the medium demand level's; the
    levels differ only in standing time), the fastest of every such line in either direction, so
    shared track shows its fastest service and a station its trains' slowest (`lineTrainSpeeds` in
    `map/network.ts`). Keyed 1/20/40/60/80/110/160/200/260/320 km/h
    from maroon through red, orange, yellow (#d9b92e), yellow-green (#a3c23c) and green (#4aa850) to blue-green (#4e9d95), blue (#397bc1), indigo (#5b4fc9) and violet (#a03fc2), one ramp for trains, trams, buses and ferries [T-3013]
    (`SPEED_STOPS`; the stroke shader is built from the same stops); under 40 is the red. No key
    sits beside the display buttons for the speed or the elevation colours (T-1499): the track hover
    says the speed and the elevation, and which speed it shows ("Track" under "Hover tags on the
    network"). A running bus or ferry line's route [T-2452] is coloured by its own vehicles' speed
    on the same ramp (the faster of its two runs' profiles at each route segment's middle, placed between its
    stops by distance), the hover's chip likewise.
  - **Profit** [T-2420, T-2440, design notes/T-2411.md; hover "Show track profit colors"]: built track coloured by
    how much of its costs its fares cover on the day shown (fares over costs, a ratio, not their difference),
    worked out only while the view is on (`game/stretchMoney.ts`). A stretch is the track between two stations,
    junctions or track ends. Each running rail line's money goes onto the edges of its path: its fares, per-km
    and base alike, by the paid passenger-km on each stretch between its stops, both directions (the line's form
    of spreading each trip's fare over the km it rides); its running trains' cost a day by length; its share of
    each edge's upkeep (by trains a day, as the line's upkeep share is split) and of each station's, half to the
    stretch on each side. Summed over the lines on each edge, a line's edges come to its fares that day and its
    upkeep share exactly (`checks/t2411-profit-map.mjs`). Track no running line uses has its own upkeep, so it
    reads 0. Construction and trains bought are left out, as in the Cities tab. A running bus or ferry line
    [T-2452] has no track: each stretch between its stops is coloured by its own money by the same rule (fares
    by paid passenger-km, its bus-km or boat-km cost by length, a bus route's upkeep by length, a ferry's
    share of each port's upkeep half to each side; a bus pays no share of the stations it stops at, as
    `line_upkeep` has it), so its stretches sum to its fares and upkeep share too. Colours
    (`PROFIT_STOPS`, blended in the stroke shader, the same in both themes): 0 #b2182b, 0.5 #e9853b, 0.95
    #f1d54a (yellow), 1.05 #96d58f (pale green), 1.6 #8ecae6, 3 and over #2166ac; blue past green so red-green
    colour blindness still reads it. Plain track is drawn where there are no figures: blueprint, and any stretch
    a running line is on that demand has no figures for yet (`lineFiguresIn`: the game loading, a new line not
    yet solved). No key (as T-1499): the hover gives the numbers.
  In Elevation, Speed and Profit the width is uniform and trains keep their line colours. Bus and ferry
  lines [T-2452] are drawn in every display but Elevation, keeping their own width, casing and dashes; in
  Speed and Profit only their colour is the display's, the plain track grey (#77746f) where a running line
  has no figure (no vehicles, the game loading) and on a ferry's landing strokes; a planned one keeps its
  blueprint look.

### Blueprint track, line colours and marks

- **Blueprint track** [T-311, T-322, T-331, T-333] is drawn solid and opaque as two rails in every
  track view: a light blue middle (#a9cdf3) between dark blue rails (#2f62b3). The whole width is 75%
  (`BP_NARROW`) of a line plus a rail of 16% of a line width (at least 1 px) either side, the rails
  never under 1 px: 6.5 px at zoom 15 (4.5 middle), 4.4 at zoom 12. A line planned along it fills
  the middle (not a full line width) in its colour mixed 45% towards white; in Elevation and
  Speed (the limit: no trains run on a blueprint) the middle is the view's colour mixed the same, and a selected blueprint's middle the
  selection ink mixed so. Lines that do not run stay see-through at 55%. **Switched off** [T-333,
  T-543]: grey rails (#9aa3ae) round a nearly white middle (#f2f4f7) in the light theme, a dim
  middle (#383e46) between lighter grey rails (#6c747e) in the dark one, and a station at 30%
  instead of 60%; a planned line's colour and the Elevation and Speed colours are mixed 80%
  towards white (towards #383e46 in the dark theme) in place of 45%. **Picked** with the Select
  tool, blueprint or built, and the track inspector's piece: a selected station's look along the
  track, a 2 px map-ink outline 3 px out from the track (or its lines, whichever is wider) with the
  casing's colour between, drawn under the track; it reads apart from a selected line's wider
  stroke [T-1679]. Picked stations get the stations' selection outline. All rails are drawn before any middle, as the casing is, so curves and joints join
  cleanly. Blueprint track and the lines planned on it are drawn after built track and its lines
  [T-695], so where they overlap the blueprint is on top and wins the click, as the station tool
  prefers it; the order of lines sharing track (above) holds within each.
- **Line colours** [T-223, `LINE_PALETTE` in `game/palette.ts`]: a normal, fairly saturated transit
  palette, sixteen defaults: red, blue, green, orange, purple, yellow, brown, teal, gray,
  yellow-green, pink, magenta, light blue, olive, coral and lavender, every pair at least 19 apart in
  CIEDE2000 and at least 17 from the land, water and parks of both themes; chips on
  the light ones take dark letters. A new line takes the default farthest in OKLab from the
  colours of the lines near where it starts (its first stop, else the middle of the map): those
  with a stop within 15 km, topped up with the next nearest to six, ties at random
  (`nextFreeColour` in `game/lineActions.ts`, T-1278). The first six (red, blue, green, orange,
  purple, yellow) are the plain ones: a city's first four lines take only these, and later lines
  count their distance 1.3 times, so they win unless clearly taken nearby (T-1894). While a line is
  still the new one (stop editing from New line), each stop edit checks the lines beside it: those on
  a track edge or station it shares, or with a stop within 300 m of one of its stops. If its colour,
  still the one the game picked, is under 19 apart in CIEDE2000 (the palette's own spacing) from one
  of them, it takes the default farthest from the colours beside it, ties at random, unless none is
  farther than it already is. A colour the player sets is never changed, nor an older line's
  (`colourPick.ts`, `lineColour.ts`, T-2291). A pick counts as used the lines with no stops yet and the last four
  colours it handed out, so lines made back to back before the game has the earlier ones still differ [T-2764].
- **Marks on the track** [T-629, T-655, T-1871; `map/gradeMarks.ts`, data in `map/markData.ts`], drawn by the
  renderer over the track and under stations and trains, from zoom 9 (blueprint and drawing) or 11
  (built):
  - A **flat crossing** (an X) and a **flat junction** (a fork, a node with three or more track ends
    and no station, not a flyover, where two moves through it can cross at its level by the rule of
    track-and-service.md 6.2: T-3103, `capacity::junction_crosses`, so no badge where every move merges or diverges) in a round badge with a thin ring. On blueprint track and in the
    drawing tool's preview (the crossings and junctions the new track would make, from the track
    model) the badge is big: radius 7 px at zoom 9, 11 at 12, 15 at 15. On built track it is small
    and quiet from zoom 11, at 85% opacity: 7.5 px at zoom 14, following the square root of the
    map's scale as the ridership circles do (halved two zoom levels out), held between 4.5 and 9.5
    px (`quietBadgeRadius`, T-1772). Its ring and glyph are in the
    Elevation colour of the level the meeting is on [T-673]. A junction is built once three of its ends are, a crossing
    once both tracks are. A crossing the model refuses (on a ramp, the tracks less than a level
    apart; [track-and-service.md](track-and-service.md) "The graph") is a red X, hover "Tracks cross
    on a ramp less than a level apart" [T-2226].
  - **One-way track no train can use** [T-3164, `Network::node_unusable`]: at a node with no station
    and two or more track ends, one-way single track pointing in with no move on, or pointing out
    with no move into it, gets a red badge with two arrows, pointing at each other or apart, sized
    as the junction badge (big on blueprint, a node counting as built once all its ends are). Hover
    "Unusable converging one-way track" or "Unusable diverging one-way track"; where a node has both,
    converging. Bare ends (a stub being drawn) and stations (a line may end there) get none. Not
    hidden by the "Flat junctions and crossings" setting.
  - A **ramp** (a trapezoid, its narrow end a quarter of its wide end, so it does not read as a
    one-way arrow, T-3189) for every change of level along track, 0 to +3, 0 to -3, 1 to 2, -1 to -2
    and the rest. One mark per ramp (a stretch where the height keeps rising or keeps falling), at its
    middle, pointing along the track toward the end that is farther from the ground, filled in the
    Elevation colour of that level (+1 teal, +2 blue, +3 indigo, -1 yellow, -2 orange, -3 red-orange),
    from zoom 11 at 3.5 px radius, 5 at 13, 7 at 15. A ramp that crosses the ground (-1 to +1) gets
    one mark on each side. The hover says "Ramp up to +2" or "Ramp down to -2" (the level the colour
    shows), then "Blueprint" on blueprint track.
  The fill is the casing's. The Settings checkbox "Flat junctions and crossings" (on by default,
  saved with the display settings) hides the crossings and junctions and their hover; ramps
  and refused crossings stay [T-673].

## Bus lines

A bus line (T-1464) is drawn by its own pass under rail track: 0.45 of a rail line's width in its
colour on a casing (white in light, dark in the dark theme), a planned one as rail blueprints are.
In the Traffic view a running bus or ferry line is drawn as wide as a rail line with as many riders,
stretch by stretch between the stops demand counts (a ferry's ports), on the rail lines' scale
(`demandLayers.ts` `busWidths`, T-2274).
Street stops are small rings from zoom 13, ends small rounded squares from zoom 11 (T-2950). A stop at a train
station is drawn like any other, ring or end square, on the route where it passes nearest the station, never on
the station's centre (T-2982), in a pass after the stations so it reads over the station's mark where the two
meet (`BusLayer.drawStationStops`, T-2948). From zoom 13 a thin dashed line runs from such a stop to its
station (1.5 px, the map's ink at 50%, the transfer; `FLAG_TRANSFER`), drawn under the track and never
picked; the same line joins two bus lines' stops within 100 m of each other, one place for demand (T-2992,
`game/busLinks.ts`; none between two on one spot). A bus or ferry loop run one way has small arrowheads in its colour from zoom 13, pointing the way
its vehicles go and sticking out both sides of the route, twice as wide as they are long (their casing only
off the route), none in the Speed and Profit displays (T-2991, `oneWayMarks.ts`). Its street rings are half filled, the half on the right of the way the buses go, where the sim serves the stop one way (`line_stop_ways`, T-3147), and the bus stop hover has the line's chevron after its bullet. In stop editing, the hover over a stop served one way has "One way stop" above "Click to remove the stop" (T-3211). A train loop run one way
has a tiny white chevron on the line instead, on the edges its path runs once (a balloon's tail runs both
ways), from zoom 12 where the line or its band is at least 4.5 px wide, in the Line view only. Both are
placed every 100 m along the route (bus and ferry arrows 125 m); at each zoom every 2^n-th is drawn so they are at least 150 px apart (arrows 190 px),
and none closer to a stop than its mark (14 px from a station). Selecting the line draws its 300 m
corridor as a 1 px outline in the line's colour at 75% (the outline of the union of its segments, made
with the stencil; `busLayer.ts`, T-1490); hovering draws nothing, and nothing is shaded. Hovering the route gives a rail line's tag: the line's
badge and name, "Bus line", then "Not running yet" while planned or, in any track display [T-1907],
the riders on the stretch between the stops under the pointer (a ferry's between the ports, T-2274); in Speed then
"Fastest bus 16 km/h" ("Fastest boat" on a ferry) on its colour, or "No buses run here" ("No boats"), and in
Profit before the riders the track's money rows for that stretch with "Running buses" ("Running boats") in
place of "Running trains", or "No buses run here" while none run [T-2452]; "Bus stop" or "End of the
line" on a stop. A click selects the line; rail wins where both are under the pointer. A click on a stop where
two or more bus lines have a stop (marks drawn on top of each other) selects the stop instead (T-2949): its
panel names it (the nearest station's, else "Bus stop") and lists the lines that stop there, a click on one
selecting that line; the map draws nothing selected. A bus is one
short narrow car, one rigid straight body that points along the route (never bending or turning sideways in a corner), drawn at its minimum screen size from a distance (0.45 of a train's length and 0.69 of its width, T-2957) and 1.2 times
its real size close up, and standing at every stop, street stops included (T-1798). A bus line's badge is a rounded square everywhere (corner 18% of its side, the
map's end squares too), a rail line's a circle, unless the player picks another shape for the line or its mode
([ui-inspectors.md](ui-inspectors.md) "Line shapes", T-3319). A new bus line is named "Bus n" (the next free
number) and shows n as its symbol.
## Stations

- **Dots, zoom 9 to 14** [T-189, T-313, T-322, T-542; `map/stationGeometry.ts`]: white with a dark rim (in both themes, T-039; the rim, a terminal's buffer stop and the selection outline are the map's dark grey in light and a warm grey #8f8b85 in dark, `stationInk` in `game/shadeTable.ts`, T-1432) at
  the platform's middle. Radius 1.7/4.3/5/7.4 px at zoom 9/11/12/15 for an ordinary station, 2.4/5.6/6.6/9.5
  px for one served by more than one line (`stationRadii`; linear between, held outside), never narrower than the lines through
  it, the rim 1.5 px at zoom 13. A complex's dots are one shape: every dark disc is drawn first,
  then every white, so overlapping dots have one outline. Dots that do not touch are joined by a
  **connector**: a straight band centre to centre with the dots' rim along both sides and white 30%
  of an ordinary dot's white radius either side of its middle, drawn in the same passes, so dots
  and connectors are one white shape. Connectors join the stations the complex distance rule groups
  (`dotGroups` in `game/complexView.ts`) by the shortest set of joins, and each player link between
  two groups [T-225]. A connector is blue like a blueprint station when either station is a blueprint.
- **Each dot is its own station** [T-249, T-305]: a click picks the dot whose centre is nearest
  among those under the pointer and selects its complex; the delete and line tools take that
  station, and its tag lists its own lines.
- **Crowd pie** [T-228, T-553]: with platform crowds on, a dot's white fills like a pie chart,
  clockwise from 12 o'clock on the screen, with the crowd grey (#a4a19d) in proportion to its fullest
  platform side's share of the room (full circle at full), or the over-full red once that side is
  full [T-559]. The rim stays; each dot of a complex is its own pie, and the
  connectors stay white. The same colours in both themes.
- **Platforms, from zoom 14** [T-204, T-254, T-239, T-553]: a band along its track as long as the
  platform, following the track's centre line sampled at 13 points (`platformShape`,
  `PLATFORM_POINTS`; straight on past an edge that stops short), flat-ended, centred on the station
  node, or at a dead end running from the node along the track. Its width is fixed on the ground
  like its length from zoom 15 (`BOX_WIDTH_ZOOM`): 1.125 times a plain line and its rim
  there (11.25 px, about 24 m at New York's latitude; `BOX_NARROW`, T-652), so its shape never changes with zoom and it is
  never shorter than it is wide. Further out, to zoom 14, it keeps that 11.25 px on screen (white 7.25 px
  across) while its length keeps shrinking with the ground, about 40-55 px at zoom 14. The same
  width in every mode: a wide traffic line or a ribbon of lines side by side can be wider than its
  stations (T-1821). Everything that draws or
  tests a platform (box, crowd fill, track stencil, picking, placement ghost, selection
  outline, the cut-out in the bubbles) uses that one line (`polyFrame` in the shaders,
  `stationShape`, `stationPath`). Blueprint stations are drawn as blueprint track is (T-2930): a light blue
  fill (#a9cdf3) in a dark blue rim (#2f62b3), so they read as blueprint beside white built ones.
- **Overlapping platforms are one shape** [T-217]: every dark shape first, then every white inside
  its rim, so one outline runs around the whole (a blueprint's rim and fill are its blue).
- **Inside the white** [T-236], over the crowd fill: the track and line strokes run through the
  box (drawn where a stencil marks the white, no wider than 30% of it, without casing). No platform
  edges are drawn [T-1121].
- **A platform each side, close in** [T-658c]: on a double-track station, or one where single track splits (T-3212; flag 16 in the station
  buffer), as the two tracks part ("Lines and track") each direction's riders stand on their own
  platform, from the outer edge of its track out to the rim [T-1121], instead of filling half the box:
  the + side is the right of the platform axis, so with trains keeping right those travelling along the
  axis stand there, and with trains keeping left those travelling against it (T-3297). At a station with passing tracks the platform is an island between
  the running track and the outer track (where trains on both stop, T-2248), so its riders stand
  from the running track's outer edge to the outer track's inner edge [T-2249, `u_pass_in`]; from
  zoom 14 to where the tracks part, from the middle to the outer track. A single-track station is
  unchanged.
- **A player link's bridge** [T-197]: a straight white band 3 px wide at zoom 15 (fixed on the
  ground) with the 2 px rim along both sides, between the nearest points of the two platforms'
  centre lines, at least 6 px inside each platform's ends; overlapping platforms get one inside
  both. **Members of a complex whose boxes do not touch** (centre lines more than 16 m apart,
  `BOX_TOUCH_M`) get the same bridge whether or not a passage was built [T-1859, network.ts
  `complexPassages`]: the shortest joins between platforms that connect the complex, with touching
  boxes and built passages counting as joined, so none is drawn twice. A drawn-only bridge costs
  nothing and walks at the plain change pace (a built passage is still faster, T-1281).
- **Selected** [T-163, T-189]: the station keeps its size and gets a 2 px dark outline and a soft
  dark glow 6 px beyond, round the union of the selected shapes and bridges, drawn under them;
  selected stations are drawn after the others and the demand bubbles are cut away over them. In
  complex view every member, bridge and connector is drawn selected. A click on a station picks it
  even over bubbles [T-1515].

**Station complexes** [T-146, T-1859, `game/complexes.ts`]: stations whose platforms come within
50 m, centre line to centre line, are linked: the drawn box's line (along the track, curved, from
the node at a dead end; `workers/platformLines.ts`) or the straight segment of the platform's length
through the node along the track heading (the older rule, kept so no complex splits). Stations whose
centres are within 100 m (demand's merge distance) are linked too, whatever their levels; a complex is everything linked
directly or through others, plus the player's links [T-197] and every station touching one station
building [T-2217]. Worked out whenever stations or links change; only links and buildings are saved.
Riders change between any two stations of a complex on foot [T-161, [demand-kernel.md](demand-kernel.md) 4.4].
A complex has one name by default [T-3288]: the stations nobody typed a name for all show the name of the oldest
of them (`workers/stateRows.ts` `foldNames`; the model keeps each station's own name), so the map labels it once;
a station with a typed name keeps its own. Where one name stands for the complex (Stations tab row, diagram label)
it is the member's whose name the player typed, else the one with the most lines [T-2683, `game/complexLead.ts`].
A splitting or merging complex changes no typed name; stations that were following a name show the shared name of
their new complex, or their own if they are alone.

**Station buildings on the map** [T-2219, `map/buildingFloors.ts` drawing what `map/buildingNet.ts`
works out in the clock worker, the inside's geometry `game/buildingLayout.ts`; the tool and panel in
[ui-inspectors.md](ui-inspectors.md) "Station buildings"]:
- **One shape**: the floor (#f9f7f3 in both themes, as the boxes are white in both) and its members'
  boxes are one white shape with one rim in the stations' ink, 1.5 station rims wide (2 in the dark
  theme), half of it over the floor. Members are the stations whose drawn platform touches the
  outline; their own rims are not drawn on the floor (its floor marks the stencil), so they are plain
  white bands on it, and a part sticking out keeps its rim and joins the building's. No bridges are
  drawn between two members. Other stations draw over it as their own boxes. The track and lines run
  over the floor as over the land, and through the members' boxes as through any box.
- **Shops**: on the floor clear of the outline (3 m), every platform's box (4 m) and every line (3 m
  past its stroke) and the waiting room, cells on a grid lined up with the first edge, 13 m along
  the platforms by 15 m across with 4 m between, outward from a clear cross-hall 28 m wide across the
  middle of the long side. About 85% of cells have a shop, and 40% of those take the next cell too as
  one long shop; picked by the building and the cell, so they never shuffle. Seven pale colours with a
  rim 16% darker, the same in both themes; from zoom 17 a shop at least 9 px across has a glyph in its
  rim's colour, upright on the screen: a cup, a bag, a bowl or a book. No hover.
- **By zoom**: below 13 nothing (dots and connectors as before). From 13 the floor and rim fade in, the
  floor at 60% until 14; from 14 the whole shape, members' rims off the floor; shops fade in from 15
  to 15.5, glyphs from 16.8 to 17. A building with no station inside is at half strength with no shops.
- **The hall's crowd** (with platform crowds on, from zoom 14): riders past a member's side's room
  stand in the hall, one pool for the building (room: one rider to 2 m² of the floor not under its
  members' platforms, `hallRoom`). They are a stipple of the walkers' grey (2 px dots on a 4 px
  screen grid) on 4 m cells of free floor, as many cells as the hall's share full, nearest the
  platforms spilling first; red when the hall is full. A member's side turns red only when it is past
  its room and the hall is full; zoomed out a member's dot reads its fullest side against its own room
  plus an even share of the hall.
- **The name**: one label for the building, its name, and its stations' own names go. From zoom 14
  it sits in the middle of the cross-hall, dark and bold on the floor in both themes, when the
  building's short side on screen is at least the name's width plus 24 px; otherwise below the floor
  (right of it when the floor is taller than wide on screen). Placed before the stations' names.
- **Selected**: the complex's building gets the selected stations' outline and glow round its rim;
  its members' boxes get none of their own.
- Car parks go outside the outline: the lot search tries beside its edges first, longest edge first,
  and counts any overlap with a building as bad.

## Trains

- **Shape** [T-103, T-107, T-189, T-199]: not drawn below zoom 10 [T-542, `TRAINS_ZOOM`]; pills from 10, separate 20 m cars
  from 14 (`STATION_BOX_ZOOM`), in a darker shade of the line colour with a white rim blended over a pixel, antialiased.
  Width 3 px at zoom 10, 5.25 at 11, 6 at 12, 9 at 15 (linear between, held outside; true width only from
  about zoom 19), shared by pills and cars so the switch never shrinks a train. A pill is never
  shorter than 11, 17.6 and 26.4 px at those zooms and bends with its track along the circle
  through its ends and middle. Each car is turned along the chord between its ends, so the consist
  snakes round curves and stays on the route at terminals (notes/T-047.md). One instanced draw for
  all trains, positions from keyframes on the GPU, less each train's lag (below). A click on any car selects the train; one drawn
  over a station selects the train.
- **On its own track** [T-658b]: as double track parts into two (about zoom 17, "Lines and
  track"), a train moves from the centre line to the right-hand track of its direction of travel
  (2 m out once apart) and narrows from its pill width to its true 3 m, so two trains passing go
  by side by side without touching. Where double track narrows to single, or a station's split joins again, the train drifts back
  to the centre over the taper (`sampleLane`, one value a 50 m sample). A train turning back
  stands on its arrival track and steps across to the other as its next trip starts. Clicks
  find it where it is drawn (`laneShiftM`). A train's two ends are each shifted by the lane and
  band there, and it faces along the line between them, so a train easing sideways with its band
  (lines side by side) or onto its own track turns with it (T-2182).
- **On the outer track at a passing station** [T-2233]: from zoom 14 a train of a line that stops
  at a station with passing tracks (not at its run's ends) moves out to the outer track on its right
  as it comes through the lead, stands there, and comes back over the far lead, following
  the drawn track (`samplePass`: per 50 m sample, metres past the nearest such platform's end).
  Trains of lines running through stay on the running track, and so do those of a faster line
  stopping there beside a slower one [T-2248, `innerStops`]: of two lines that stop at the station
  and share the track through it, the one stopping at fewer of the stations on the stretch they run
  together stops on the running track, so an express calling where a local stands stops beside it.
  Lines stopping at as many stations share the outer track.
- **Never through the train ahead** [T-1965, `map/trainGuard.ts`]: each train is drawn from its
  own line's timetable, which can put a faster train inside a slower one on the same track (an
  edge in one direction). Each frame, for trains in or within 150 px of the view on lines that
  share track with another line, a train's front stays 30 m behind the back of the train ahead on
  its track (their true lengths; zoomed out, lines sharing track are drawn side by side), and a
  train does not move onto a merge while another is on it. Order on a track never changes from one
  frame to the next, but at a station with passing tracks [T-2233]: a train standing on its outer
  track (its back no more than 20% of the lead short of the platform, its front no further than
  past the lead) and a train of a line on the running track there (running through, or stopping
  beside it, T-2248) hold neither of each other, at
  every zoom, so the faster one is drawn passing. Each direction of a line has one phase while the
  timetable's holds assume each direction's own (track-and-service.md 6.2), so in one direction a
  drawn express can catch a local between stations; it is then held behind it to the next station
  where the local stops on passing tracks, and passes there. A held train is drawn late in its own run (`lag` per trip, an attribute on the
  GPU), so it keeps its stops and speeds and waits outside a station where the train ahead stands;
  once clear it makes the lag up at 0.3 s a game second, drawn up to 30% faster than its timetable.
  A train still held when its next trip is due takes that trip late by the rest. At most 20 minutes
  behind (past that it is let through), and a clock jump of over 15 minutes starts every train on
  its timetable. Clicks, hover tags, the train panel and following a selected train use the drawn
  place. `?perfOff=follow` or `renderer.follow` turns it off.
- **Places at a line's end** [T-2980, T-2986, `map/terminalBays.ts`]: where a line ends on double
  track at a dead end with a terminal (T-1281), its trains use both platforms, and so do those at a through
  station with one (track-and-service.md 6.2). Nothing is drawn past a line's last station (T-3053): a
  train stands whole on its platform, its middle at a dead end's platform middle or at a through
  station's stop, over the line's end by up to half its length less half the platform (`endClear`;
  the place distances are sent as `-p - 1` at a through end). A dead end
  without a terminal has one place, its arrival track's platform (`kind` 3): the next train waits
  short of the crossover until the one there has left, and each leaves over the crossover. Lines arriving at one end along the
  same track share its places. Each arriving train takes the place that frees first (its own track
  on a tie); a place is free 60 s after its train leaves. The pick is worked out in the clock
  worker from the timetable, with the hour before the window, so every window draws the same
  (`assignBays`, four numbers per trip sent with the trips; `?perfOff=bays` turns it off). Drawn: a
  train for a dead end's other platform crosses the scissors on the way in, over the 50 m past the
  platform (or past its own length, if longer), and a train leaving from the arrival side's
  platform crosses on the way out; a train ending at a through platform without a terminal leaves
  over the diagonal behind it instead of jumping across. A line ending at a station where single track
  splits [T-3202, T-3212] has two places there, the split's two tracks, the crossover 60 m
  past where they join, and each arriving train takes the place free longest, so the train it meets
  there stands on the other. A train standing at such an end holds no one on its track; one due at a place still
  taken (its train standing there, or leaving and not yet off the platform, but not one standing
  for the single track until it is in) stands short of
  the crossover, 30 m before it, until it is free. One place only (a through platform without a
  terminal): the train ahead holds it, as above. A line with places at its ends is guarded
  even alone on its track (`TrainGuard.placeLines`, T-2986), so a stand longer than its headway at a
  one-place end holds the next train back instead of drawing two there. Clicks and the train panel use the
  drawn place.
- **Taking single track in turn** [T-3202, `map/trainGuard.ts`]: at every zoom, trains of
  opposite directions on two-way single track (the capacity's single-track stretches between
  passing points, `TrackApi.guard_singles`) take a stretch in turn. Each train's time on it comes
  from its timetable and lag, from leaving the stop before it to coming to rest at the stop after
  it (stops within 300 m of its ends). One already on it keeps it; the rest go in the order they
  are due, and one coming the other way stands at its stop until 5 s after the other is in, so on
  a two-station line with two trains one runs while the other waits, about a run time late. Trains
  the same way go together (the train ahead holds them). Every train of a line on single track in
  view is looked at, wherever it is, since the one coming the other way is often far out of the
  view. Lag is made up and capped as above. At a station on single track a train is drawn on its
  own track of the split ("Close in, trains keep right"; its lane is the drawn track's, T-3212), so
  two meeting there stand side by side.
  Drawing only: no figure moves. `?perfOff=singleWait` or `renderer.guard.singles` turns the wait off.
- **Waiting at flat crossings** [T-1966, `map/trainGuard.ts`]: zoomed in to `CONFLICT_ZOOM` (14,
  where trains are drawn as cars at about their true length), trains in or near the view wait for
  each other at flat crossings and at junction moves that cross (the track model's conflicting
  pairs, `TrackApi.guard_conflicts`). The rule is by time: each train's time over a conflict comes
  from its timetable and lag, from its front 20 m short of the point to its back 10 m past it (a
  train standing over it at its run's end or start holds it for the whole stand). One already over
  it keeps it; the rest go in the order they reach it (those already over another conflict
  first), each standing with its front 20 m short until every train on a crossing route ahead of
  it has cleared, then going on late (the same lag as above). A train that will have to wait at a
  conflict before its back clears the one it is at stands short of that one too, so it never stops
  on a crossing. A train first looked at (come into view, zoomed in to, after a clock jump) takes
  its turn by its timetable too, so two found over one point are parted. Trains of a line with
  crossings but no shared track are then also held behind the train ahead on their track, as
  above, so one queues behind a train waiting at a crossing; a train standing out its turn at the
  end of its run holds no one there.
  No wait is longer than 4 minutes (`CONFLICT_WAIT_MAX`), and a train already 8 minutes late
  waits no more: past either it is let over, drawn through whatever is there. Zoomed out further, trains are drawn through each other at
  crossings. The timetable already carries the average wait at each conflict as a hold
  (track-and-service.md 6.2), so the drawn wait changes no run time, boarding or demand.
  `?perfOff=crossWait` or `renderer.guard.crossings` turns it off.
- **How full** [T-103, T-207, T-218, T-230, T-2186]: the core fills from its back like a gauge across the
  whole consist, in the dark shade of the line colour (48% of the line RGB) over an 18% white tint.
  Half full is every seat taken, full is the rated load (160 a car, 100%), from the boarding account
  ([track-and-service.md](track-and-service.md) 6.6): this train's riders on the stretch it is
  drawn on, empty at its last stop. Past 100% a second band, the dark shade 65% of the way to the
  warning red, fills from the back again, whole at crush (240 a car, 150%), where the train has a
  red rim 3.9 px wide (the white rim is 1.7 px). A boat is two cars (480 at crush); a bus a third of a car
  (53 at the rated load, red-rimmed at its crush of 80) [T-2195, T-2919].
- **No figures yet** [T-230, T-324]: a running line demand has no riders for (just after an edit
  changed its stops) is drawn striped in the line colour and white, never solid, and its tag says
  "Calculating demand".
- **Selected** [T-242]: drawn over every other train, unchanged, inside a halo around the whole
  consist: an outline in the map ink (dark, light in the dark theme), a band in the casing's colour
  (white, near-black in the dark theme), a ring of the train's shape 6 px out in the ink and a soft
  glow, so it reads even at zoom 11-13 [T-543]. A click tries the selected train first. It is followed onto its
  next trip [T-245, [ui-inspectors.md](ui-inspectors.md)].

## Platform crowds

[T-164, T-219, T-236, T-553] From zoom 14 the white of each platform box fills with a dark warm grey
(#a4a19d, right up to the rim, T-1121) as riders wait. Each direction of travel
has its half of the box along its length (close in on double track, its own platform beside its
track, "Stations", T-658c; at a passing station the island inside its outer track, T-2249); a platform whose trains all use one side (a terminal)
fills as one. A side fills from the middle outwards by its riders' share of the room
(`platformRoom`, [track-and-service.md](track-and-service.md) 6.6: 1,250 a side on 150 m), full to
both ends at that or more. A side is red instead of grey once it is full [T-559]; the red stays inside the box like the grey (T-713). A standing train covers the box's middle; from zoom 16 the sides show beside it. The crowd
grows between trains at the steady arrival rate, jumps as riders changing from another line's train
reach it [T-554], and while one of its own trains stands there [T-1123] falls evenly over the stand
to the riders that train will leave behind, as the train's gauge fills alike [T-1785], growing again
once it has left; worked out on the main thread at most once a
game second, only while drawn.

**Riders changing trains** [T-1124; `map/transferBubbles.ts`]: with the crowds, from zoom 14, the
riders who change from a train at one station to a line at another station of its complex leave as
it stops, as one bubble per train and platform they go to: a disc of the crowd grey (a lighter
#b8b3ac in the dark theme, `walkers` in game/shadeTable.ts, T-1288) with a rim in the casing's colour, moving in a straight line from the middle of the platform left to the middle
of the one they go to over demand's walk between the two stations (the boarding account's,
track-and-service.md 6.6), and joining that platform's crowd at once on arrival. Its radius is the
platform box's white half width times the square root of its riders over 250 (`BUBBLE_RIDERS`),
held between 0.35 and 1.3 times it. A change at the same station walks nothing and joins at once,
with no bubble. Built once a game hour from the account's slice; each frame only sets the time.
The hover says "40 riders changing to subway A".

## Life at stations

[T-686; `map/stationLife.ts`, `stationSound.ts`, `faces.ts`] From zoom 14 (`STATION_BOX_ZOOM`), with
the clock running, a train pulling in at a built station in view is found by its stop offset passing
the station's between two samples (eight a second; only the lines stopping in view are read). Below
zoom 14, or with Faces and Sound both off, no timer runs and there is no canvas. The riders come
from the boarding account: those who get on are the platform's crowd when the train arrives plus
arrivals until it leaves, less those left behind; those who get off are the load carried in plus
those who got on, less the load carried on.

- **Faces**: one per 20 riders getting off (rounded at random), at most 4 per train, 5 per platform
  and 36 in all. Each rises 30 px over 2 s and fades, 14 px across at zoom 14 to 20 px at 17, from
  its train's own platform as drawn (each line's station is its own box, so a complex's faces stay
  with their lines; the figures are that platform's own) [T-2751]: one of 5 places spread over the
  middle 80% of the box, the least used by faces alive on that side first, 55% of the box's half
  width out from the centre line on the side its riders stand (as the crowd fill's sides), off the
  track. Happy (pale yellow, blushing); sad (blue, brows raised at the
  inner ends, a tear, a frown) when the train before left 20 or more riders behind (`CROWDED_LEFT`);
  tired (drained yellow, heavy lids, a flat mouth) when the riders waiting have waited 10 minutes
  or more on average (`TIRED_WAIT_S`) [T-696, T-802]. The average counts the newcomers at half the
  time since the last train left and the riders it left behind at that plus half the gap before it;
  the hour's first arrival at a platform has no figure and is happy. Sad wins over tired; no face
  is red. A few have a rare look [T-1154], drawn at random when the face appears and kept for its
  life (`pickLook`): expressionless (dash eyes, straight mouth) 4% and asleep (closed eyes, a small
  round mouth, a z) 4% on happy and tired faces; crying (eyes shut, a wailing mouth, two streams of
  tears) 8% and knocked out (cartoon X eyes, a tongue out) 3% on sad ones; a bucket hat, a
  baseball cap or sunglasses 2.5% each on any face (no sunglasses over shut or X'd eyes, and no z
  under a hat); short hair 10% and long hair 10%, in one of four colours (brown, black, auburn,
  honey), on a face with no hat [T-1852]; and a child, drawn at 72% size, 6%. In a country the
  table in `map/faceTones.ts` lists (the station's zone's country), a face is one of six plain skin
  tones instead of the mood's yellow, drawn from the country's weights, tired ones mixed 35% towards
  drained yellow and sad ones 45% towards blue, and its hair colour is drawn from the country's weights (black
  for about 9 in 10 in East, Southeast and South Asia); elsewhere the yellow face and evenly drawn
  hair [T-1907]. Three sprites on a 2D canvas over the network; a frame loop
  runs only while a face is alive. Not in the demand views' way: the bubbles draw over them.
- **Sound** (WebAudio, no files): one soft sine note as the train pulls in, the line's own, one of the 12 semitones from C5
  [T-706]: a city's first five lines (a line's city is where its first stop is) take C, E, G, D, A in
  that order, later lines a semitone from their number. The note is chosen once, when the arrival
  code first meets the line, and kept in the save (`notes` in the header: note and city, per line id,
  kept after the line is deleted), so deleting a line changes nobody else's; a line's rank is how
  many lines were given a note in its city before it. Older saves get theirs on load, in line
  number order. A line with no stops or in no city is not stored. A platform with sad or tired faces sounds a minor third lower, softer, sagging [T-696]; then pops (a short
  falling blip) as riders get off (higher) and on (lower), 1 to 4 each by the rider count. The
  getting-off pops [T-1122] are skipped at a 30% chance (`OFF_SKIP`); otherwise the group is
  shifted by a random up to 4 semitones either way from 640 Hz, each pop up to a semitone more,
  at a level in proportion to the riders getting off: the pops' own at 80 riders
  (`OFF_FULL_RIDERS`), never under a quarter of it nor over 1.25 times it. Each arrival takes one
  of three lengths at random [T-1153]: short at 45% (the note rings 0.6 as long, then one group of
  pops, getting off or getting on, at most 2, straight after it), medium at 35% (both groups, at
  most 2 each), full otherwise. Silent
  until the first click or key: the audio context opens once it is over, with no button held and the
  camera still, when the page is next idle [T-1207] (`map/audioOut.ts`). Volume is the setting times a zoom gain (a quarter at 14, full from
  17), shared down by the arrivals sounding in the last second; a second one sounding together gets
  the note only, and none beyond 3; panned by where the station is on screen. At the second speed
  only the note and two pops; none at the third.
- Both are on by default and set in Settings (Stations up close).
- **Building sounds** [T-1282, T-1413, T-1474, T-1532; `map/buildSound.ts`, voices in `map/buildVoices.ts`,
  recordings in `map/samples.ts`]: under the same Sound switch and volume, with no zoom gain. They
  play at the click itself, before the clock worker answers: `edit()` in `game/world.ts` tells
  `setBeforeEdit` as it sends, and the track and station tools play theirs in their click handlers.
  A click the ghost already shows refused is silent; any other that the worker then refuses has
  sounded. The exception is the construct sound (below), which plays after the worker accepts, via
  `setAfterEdit`, so a construct refused for cash is the knock alone [T-1683]. A train count step the
  cash will not cover is refused before it is sent (`trainShortfall` in `game/money.ts`, the worker's
  rule: the cars the line's busiest level needs beyond the cars owned, at the car price, against the
  cash): the knock alone, nothing changes [T-1797]; so is a headway step (or typed headway) at or
  over the level's shortest headway (`headwayTrains` in `game/world.ts`, from the limits the line's
  panel last read) whose trains the cash will not cover; a shorter headway, a line whose limits are
  unread and a bus line go to the worker, which may then refuse after the pop [T-1800]. One sound per click however many pieces it places (a pasted schedule or a folded train
  count is one sound per click, not per edit). Recordings (`app/public/sounds/`, with
  their leading silence cut, decoded when the audio context opens; peaks at volume 1 in
  `buildSound.ts` `PEAK`): a finger snap on placing track as blueprint (0.21) and, three semitones lower (rate times 0.84,
  `STATION_RATE`, the same four variants around it), placing a station [T-1698];
  a gun magazine insert (`construct.wav`, 0.66 s, 0.15) once the worker has accepted Construct
  blueprints (the button and C), a line's or a piece's construct, and a station upgrade or passage
  built for money; one added only to the blueprint gets the pop; a wood knock (`knock.wav`, 0.16 s, 0.12) on every refusal, that is every
  `say(text, true)` in `game/world.ts` (a train count or headway the cash will not cover (the pop is still played for a headway), a track
  or move that cannot go there, a bus line that cannot be drawn; a refusal for money says what it
  would have bought, "Not enough money to buy trains" for a line edit, "to build this" for track, T-1696); a cork pop, quiet (0.09), on a panel click that changes the world: train
  count, headway or switch to headways, train length, ordinary or high speed, a stop added, moved or
  removed in a line, a station's platform length, a station upgrade or a passage taken out, double
  up, a track's elevation, a junction's flying switch, a line copied or removed, a schedule pasted.
  The snap and the pop each play one of four variants (playback rate 0.94, 0.98, 1.02, 1.06 with
  gains 0.95, 1, 0.85, 0.9; `VARIANTS`), the construct one of three (0.96, 1, 1.04 with gains
  0.95, 1, 0.9; `CONSTRUCT_VARIANTS`), picked at random, never the same one twice in a row
  [T-1658, T-1683]. Synthesised: a soft chime of two sine notes (E5, then A5
  0.16 s later, 0.7 s each) when a whole-game load that held the clock over 30 s ends
  (`loadComplete` in `game/loading.ts`, T-1655). A line's opening day makes no sound. Silent: opening a panel, hovering, tabs, names,
  colours, fares, undo, the remove tools, drags. The same sound twice within a short gap (construct 0.3 s, snap
  0.15, pop 0.05, knock 0.25) plays once. One audio context serves all (`map/audioOut.ts`):
  opened with `latencyHint: "interactive"`, resumed at once, a silent buffer played through it once
  the files are decoded.

## Station names and capacity tags

- **Names** [T-032, T-074, T-101, T-169; `map/stationLabels.ts`]: DOM labels in the UI font above
  the overlay. Hidden below zoom 12; above it padded collision boxes keep them apart (48 x 28 px
  until zoom 14, then 20 x 12), transfers first, then constructed stations, then lines. A
  complex's stations sharing a name have one label at their mean position, oriented like the
  member with the most lines. They beat basemap place names through an invisible symbol layer of
  the same names that MapLibre's collision honours. A flat pan moves their shared compositor
  layer; a zoom, rotation or tilt moves each shown name with its station every frame at the spot
  it last took (T-1621); placement and the greedy overlap pass run at most every 100 ms while the camera moves, and on edits, font loading,
  showing labels again and after movement ends. A 256 px margin prepares names entering the
  screen, and a pan re-places them after 128 px of travel (notes/T-074.md).
- **Opening day** [T-1434; `game/openingDay.ts`]: a line opens when it first runs with trains in
  this page's session (running, and some level's trains above 0), having been in the world before.
  For one game day (86,400 s of the clock, from the moment it opens) the line's Riders cell in the
  Lines tab counts up from 0 to its figure over 2.4 s, eased out, starting when the figure first reads above 0
  with the tab open ([ui-inspectors.md](ui-inspectors.md) Lines tab). Not saved: a loaded game
  shows no opening days, and one in progress ends with the page.
- **Edit note** [T-2125, after T-1478; `game/editNote.ts`, `ui/EditNote.tsx`]: once demand has settled
  after the player's edits, the map's upper left, under the view rows and the demand alert (in
  `.map-top-left`, so it moves up when rows are folded), shows the change in train trips a day and
  in the network's profit a day: "+5,226 trips/day", "−$12k profit/day". Plain 12 px text on
  the map with a halo in the background colour, no box; each row after a 12 px arrow, up in `--good`
  green or down in `--bad` red; the profit row's text is red when it falls. A row whose change rounds
  to 0 (trips to 0, money under $500) is left out, and with both out there is no note. The numbers
  ease up from 0 over 0.9 s, fast at first. It stays 8 s, longer while the pointer is on it, and goes
  at the next edit. Each row's hover gives the before and after for the day shown ("Trips per
  weekday: 812,400 to 817,626"), and when the edits were about exactly one line (the line an edit
  names, or a new line), that line's own too: its riders under the trips ("Riders on 大江戸線:
  1,016,738 to 1,140,493"), its profit under the profit ("大江戸線: $2.03M to −$17.0M").
  Under the rows, the bullets of the lines whose riders (boardings a day) moved most, most first, up
  to 5 and only those that moved by 100 or more [T-2390]: each a small bullet that selects its line
  on a click, its hover the line's name over its before and after ("4" / "812,400 to 790,113 riders
  per weekday").
  - **Estimating** [T-2626; `game/profitEstimate.ts`]: after a trains, headway or cars edit on a
    running line (and no other kind of edit since the last settled state) the note shows at once the
    profit row alone, as an estimate: "Estimated −$5.3M profit/day" in muted text (`--muted`, the arrow
    grey, never red or green), no trips row and no line bullets. The hover gives its two parts for the
    day shown, "Running cost +$9.6M per weekday" over "Fares about +$4.3M per weekday"; a part under
    $500 is left out. Running cost is the clock's running cost an hour against the last settled
    state's, exact. Fares are that state's, plus the wait term: each boarding waits half the headway,
    so a period's riders change by boardings x (exp(e x change in wait) - 1) on each line whose trains
    an hour changed (a line left with no trains loses its riders), e being `DemandView.fareSense.local`
    per minute over the rail trips (the minutes a fare dollar costs are the boardings' mean over the
    lines' countries), and each rider gained or lost pays the network's mean fare a trip. A cars edit
    moves no wait, so its estimate is running cost alone. The estimate is read against the last settled
    state, so edits add up and a line's early local result does not move it. It stays, and follows
    the network, while demand settles, with no 8 s limit; the settled note then replaces it, its profit
    easing from the estimate to the solved figure. With no estimate (any other edit, a figure under
    $500) the note waits as before. When a zone's level of detail changed meanwhile (like with like,
    below) the estimate is the note and goes after 8 s [T-2682]. The line panel and the Lines
    tab keep their "calculating..." figures.
  - **Which edits**: every edit through `edit()` but a line's or group's look, grouping and loans
    (`NEUTRAL` in game/editCap.ts): trains, headways, stops, lines new or removed, cars, high speed,
    track, stations, construction, fares, undo and redo. Edits made while demand settles from earlier
    ones add up: the note is the change since the last settled state.
  - **Settled**: the day shown's demand result is final, demand is not updating, no city is loading,
    every running line's figures are for its stops as they are, and the final shown was solved for a
    network made at or after the world version the last edit came in [T-2658] (`finalWorld`, ui-inspectors.md
    Lines tab), so a final of an earlier edit's network, shown the moment before the held edit is sent,
    is never taken for the edit's; or, when no network came after the edit (it changed nothing demand
    reads), 8 s with no edit and no demand update. Both sides of a note are so settled.
    A switch of the shown day, or 5 minutes without settling, makes none. A network with no line
    running counts as settled with 0 train trips at any moment, the game's load included, so a new
    game's first line has a note however early it is built [T-2682]. While the zone in view is shown
    coarse with its fine figures on the way (`refiningZone`, a zone's first solve) edits wait for
    those, up to a minute.
  - **Like with like** [T-2180]: while edits settle (`editsSettling`) no zone changes its level of
    detail ([demand-runtime.md](demand-runtime.md) "Levels of detail"), and when a zone with train
    trips in both states is shown coarse (`roughZones`) in one and fine in the other (a change already
    under way at the first edit) there is no settled note: coarse and fine figures differ by more than
    a small edit moves. A zone with no train trips on one side (new to the network) is 0 there at any
    level, so it never counts [T-2682].
  - **Figures**: for the day shown (the weekday's for the average day). Train trips [T-2237] are the
    trips taken by rail, each once however many lines it rides (`railTrips`): every zone's view's
    `railTripsPerDay` (the city views' train trips) plus the market's between places (longer trips
    and commutes, `intercity.railTripsPerDay`), summed from the zones because a local or focus
    result puts in a zone's figures and leaves the network view's total as it was. Until T-2237 the
    row was riders, the lines' boardings with transfers: doubling 大江戸線's trains on the real Tokyo
    save put +123,516 riders on it while 54 other lines lost 76,679 (notes/T-2180.md), so a line that
    saves people a change could show riders falling while trips rose; the same edit is +19,797 train
    trips against +46,492 boardings, and its undo the same back. A line's own riders in the
    hover are its `ridersPerDay` (its boardings, the riders fares are paid on; the Lines tab's Riders
    are the boarding account's past 24 hours, which leave out riders full trains turn away, so the
    two can move apart); profit is the day's fare
    income (`incomeByDay`, what the clock pays), minus the running cost (`runningPerHour` by period,
    a Saturday's and Sunday's high hours at the medium level) and the upkeep a day. A line's own profit
    is its fares, its trains' running cost and its share of upkeep, as the Lines tab's Money.
  Hidden with the rest of the interface (F2). Not saved.
- **Passing note** [T-1709, T-2166; `ui/PassingNote.tsx`]: a line of words at the map's upper left,
  under the edit note and in its look (plain text with a halo, no box), for 5 s (the sent feedback's).
- **Capacity dots** [T-184, T-224, T-256, T-1772; `map/capacityMarkers.ts`; what they mean and
  how they look in [track-and-service.md](track-and-service.md) 6.2]: a 16 px dot centred on its
  resource's point, a 1.5 px ring in the casing colour (white, near-black in the dark theme). Dots
  keep 6 px clear of one another below zoom 14 and 3 px from it; the worst delays place first and a
  dot that would crowd one already placed hides. Station names neither move nor hide them: names
  draw over dots (z 2 over 1). While the camera moves, each shown dot follows its resource every
  frame, and which dots show is worked out again at most every 100 ms [T-1614]. While a tool other
  than select is on they take no pointer events, so a press reaches what is under them. With
  select, a dot that opens nothing of its own (not a junction) selects what is under it, its station
  first.
- **Reversal icons** [T-797, T-1581; `map/reversalMarks.ts`]: over every station where a line's trains
  reverse partway (`LineStats.reversals`, from `TrackApi::line_reversals`) and every bare track end
  they reverse at (`LineStats.endTurns`, `TrackApi::line_end_turns`), a badge in the built flat
  junction badge's style: the casing colour inside, a 1 px amber ring and an amber U-turn arrow
  (#c98a00, #e8b020 in the dark theme), at 0.9 opacity, floating on a 1 px stem 6 px long just
  above the place so the stop editing marker on it stays clickable. Its diameter is twice the
  built flat junction badge's radius (`quietBadgeRadius` in `map/stationGeometry.ts`: 7.5 px at
  zoom 14, following the square root of the map's scale as the ridership circles do, held between
  4.5 and 9.5), 1.6 times that while a placing tool is on (track, station, line, blueprint or bus),
  switching at once. It hides with the stations below zoom 9 (`STATIONS_ZOOM`), never for names,
  and shows during stop editing too. Hover
  (a game tag) names the station ("Track end" at a bare end), each line "<line> reverses here", and
  that 12 trains an hour at most reverse there (from each side, at a station). With select, a click selects the first of those lines (not
  while stops are edited); with another tool it takes no pointer events.

## Hover tags on the network

- **Track** [T-157, T-526, `map/trackHover.ts`]: with the select tool or a build tool, the pointer on
  a line or track shows a tag at once saying what the track display draws there. Line: the line's
  bullet, then its name, on one row [T-2007]; under it, in every track view, "41,200 riders per
  day" (or "0 riders per day") on that stop-to-stop stretch, both
  directions together, for the day shown (counts to the nearest 100 above 1,000), with a third line under the purpose filter
  ("Errands trips only"); outside Traffic no riders row until demand answers. Over track with
  lines on it (the tag of Height and Speed, which draw no line strokes) the first row is the
  riders of all those lines on their stretches there together, worded the same [T-1907]. Traffic
  adds "Average load 28%": the stretch's riders (every purpose) over what the
  day's trips hold at the rated load (160 a car), the line panel's Average load for one stretch, the trips counted
  from the boarding account for the day shown (the average day 5 weekdays, a Saturday and a Sunday
  over 7), able to pass 100% up to crush's 150% [T-2186], no row until the account answers or where no trips run [T-2007];
  and on track at capacity "Track at capacity at high demand" in red [T-819]. Whatever the display [T-473], the tag also carries one row
  for the level at that point (an up and down arrow, then "+1" on its Level colour) and
  the speed limit beside it (two right chevrons, then "120 km/h" on its Speed colour) [T-2007], each colour at
  half strength behind dark text, 2 px short at the bottom so rows' colours leave a gap
  (`map/trackTagRows.ts`, `.tag-chip`, T-1495); in the Speed view a first row says what
  the colour there is: "Fastest train 85 km/h" (on its colour) where trains run, "No trains run
  here" where the limit is shown [T-1271]; in the Profit view the first rows are the stretch's
  "Fares cover 64% of costs" (the share on its colour), "Profit −$280k per day" (the day shown's phrase), "−$512k per km",
  "Fares $400k", "Running trains −$590k" and "Upkeep −$90k", negative money red; on bare track "No
  trains run here" and "Upkeep −$40k per day"; "Calculating demand" before the first answer or while a line on it has no figures [T-2420, T-2440]; and last, over bare track only, every line using the track as
  line bullets, in the save's order, wrapping (over a line the tag is that line's alone, its bullet
  heading it [T-2007, T-2387]) (the markup and CSS of
  the `Chip` in `ui/widgets.tsx`, built as plain DOM by `lineBullets`; no row when none runs
  there) [T-532, T-559]. A planned line adds
  "Not running yet", a broken one "Not running"; before the first solve the traffic view says
  "Calculating demand". The tag follows the pointer along the track; leaving, pressing, dragging or
  zooming hides it. It follows the pick order: with the select tool, bubbles, capacity tags and
  handles above the network keep it hidden, except on a line, station or train, which come before
  the bubbles [T-1515], and inside a station ridership circle [T-1448]; with a build tool it sits above the pointer, clear of
  the tool's own tag below, and leaves out the station ridership circles. A mark on the track names itself
  ("Flat crossing" or "Flat junction" with the elevation, "Ramp up to +1", "Tunnel portal down to −1",
  "Blueprint" on blueprint track), over the track's own tag; a junction without a mark says nothing. The
  stop editor and the trip tool have no track tag. The track tag and the build tools' tags are
  drawn over the bubbles' tags.
- **Trains and stations** [T-194, T-305, T-313; `map/networkTips.ts`]: the same tag (wrapping at 320
  px). A train: "Line 1 train", "340 riders aboard" and "352 seats, 1,280 full". A station:
  its name, then its lines as bullets, each once (a station drawn as its own dot or box lists its
  own lines; a line that serves the station one way has a chevron in its colour after its bullet, pointing the way its trains head, turned with the map, T-3111) [T-559]; with ridership circles on, its complex's circle's count rows ([ui-demand-views.md](ui-demand-views.md) "Station ridership circles": under Total the riders, then entries and transfers apart, each after a dot in its colour) [T-722, T-1448]; where crowds
  are drawn a row per line and direction with anyone waiting (the line's bullet, "to Van Cortlandt
  Park-242 St", "120 waiting"), the count in the alert red while that platform side is full, or
  "Nobody waiting on the platforms". On a dot, the crowd lines are its pie's numbers ("1,257
  waiting", red when full, then "Fullest platform 45% full"). A
  ridership circle away from the network gives its busiest station's name and the same rows. Refreshed each
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
  train, a mark, an airport badge and the station ridership circles light nothing; build tools light
  nothing. It is a uniform in the shaders: a change redraws once and rebuilds nothing.

## Picking

Clicks reach only what is drawn on top [T-486]: a click never acts on something drawn beneath
another clickable thing. Picking follows drawing order [`map/pick.ts`, `main.ts` `selectAt`], top
first: editing handles (a selected stretch's corner squares, which take their own presses, and the
select tool's squares on blueprint points, which pass over the bubbles, their hover tags and the
trains beneath, `map/endMarkers.ts`); the selected station (cut out of the bubbles); demand
discs (commuter bubbles over longer-trip discs); capacity tags (under
the discs; one that opens nothing selects what is under it); stations, so a station with a train on
it is picked, not the train [T-1414]; trains (the selected one first), then lines, junctions and track; within a layer the last drawn wins; hidden layers do not
intercept. Station names, airport badges, attraction marks (T-263, docs/ui-demand-views.md
"Attractions") and the stop editor's markers take no clicks: a click on
one reaches what is under it (a badge sits on its airport's destination bubble; the stop markers' station
is picked by the stop tool). Network
hits reach 7 px from the centre line, or half the drawn width where a line is wider [T-227].
Demand discs
drawn under 8 px in radius are hit within 8 px of their centre [T-566, `MIN_PICK_PX` in
`map/demandHover.ts`; the hover tag follows the same test], unless a disc drawn above them holds
the pointer; where those areas overlap the nearest centre wins.
Clicking the line already selected selects the track under it. Where several lines are under the
pointer (drawn over each other, or side by side [T-1799]) a click selects the nearest, and between
lines within 2 px of each other the one with more riders per day on that stretch (more trains an
hour times cars before demand answers); a click with one of them selected selects the next in that
order, and after the last the track, whose inspector lists them all [T-1907, `pick`'s
`chooseShared` and `size`, `lineSizeAt` in `map/trackHover.ts`]. Clicking a selection's longer-trip discs keeps the
parent selection. Selected editing handles stay draggable above the
demand view.

**While building** [T-526]: with a build tool on (track, station, delete, blueprint),
the demand views (bubbles, inter-city bubbles, airport badges and their labels) are drawn
beneath the network, blueprints and stations, since the tool is about track. The tools' clicks
never pick a bubble, and a bubble's tag shows only where no network and no tool tag is under the
pointer.

## Zone outlines

[T-131, T-139, T-310, T-800; `map/cityBoundaries.ts`, `map/boundaryGeometry.ts`,
`map/coarseOutlines.ts`] Each registered pack's footprint (its dissolved county boundary) has a thin
solid outline (0.7 px, 36% opacity) at regional zooms, fading between zoom 8.5 and 10.5. Every
zone has one from the start: `packs/outlines.json` (pipeline/zone_outlines.py: all registered
zones, simplified to 300 m, a coastal zone's with the sea off its coast, zones.md "Sea off the
coast") is read at startup and drawn at once, and a zone's own
boundary replaces its coarse outline when it loads for a view between zoom 5 and 10.5. A border two
zones share is drawn once, as a line of the first zone loaded, so every edge has the same weight
(`OutlineEdges`; the coarse file keeps shared runs identical in both zones). The focused zone keeps
a thin boundary at closer zooms (darker outside stroke at 36%, lighter inside at 14%), and draws
the borders it shares itself, so a neighbour's outline leaves those out. A subtle shade darkens the basemap outside its exact footprint (at every zoom, so the edge of the data always shows; T-2975),
enclaves included, with its sea (the coarse file's `water`) left clear like its land [T-1145]: that
union of footprint and sea is made in a worker (`workers/focusShade.worker.ts`), the shade over the
sea standing until it arrives. These are demand areas, not municipal limits. They sit under labels and game
overlays, never take clicks, and survive basemap style changes. A change of focus swaps the shade
in one step, never through an empty frame [T-904]: a zone near the view has its shade ready at each
pan's end, and a zone focused before its footprint has arrived (a long pan straight onto it) has
the shade over the whole map, itself included, until it comes.

## Realms not shown

[T-975, T-1006; `map/realmOverlay.ts`, `game/realms.ts`; the rules in [zones.md](zones.md) "Realms"]
Everything outside the realms shown (those the game has built in, starts in or the player has shown)
is covered by one shade, a little darker than the outside shade and over it (11% of the outside shade's colour, 22.5% black in the dark theme: together about 18% and 50%), on land and water alike, under
labels and the network, taking no clicks. Its edge is the shown zones' coarse outline, the line the
zone outlines draw, never a coast; nothing of a hidden realm's zones is drawn. Each hidden realm has
one floating circle, 16 px across, filled and outlined like an unopened zone's dot (white, a 1 px medium grey outline, T-2391; in the dark theme toned down to a mid grey #8a857f with a near-black rim, T-1298), at its point (the centre of the zone nearest the
mean of its zones' centres, so on its land), moving with the map and leaving the screen with that
point, never pinned at the edge [T-1142] (hover: the realm's name over "Click the shaded area to show the cities there").
Circles that would overlap give way to the realm with more zones. A click on the circle, or a plain
click anywhere on the realm's land (inside one of its zone outlines, not the sea or dead space), with
the select tool and no Shift, shows the realm in one step: the shade opens over its zones, the circle
goes, and its zones draw and load for the view as any other. A drag does not; neither does a click
that selected something, a click with a track or another tool on, or one while linking stations.
With the select tool the land has a hover in the track tag's place when nothing of the network is
under the pointer: the realm's name over "Click to load" [T-1151]. A view coming to rest after a move,
0.4 s at zoom 10 or closer (`AUTO_SHOW_ZOOM`) with its centre on a hidden realm's land, shows that
realm as a click would; the view a page opens on does not (with no view in the address that is New
York, whatever the game).
Below zoom 2.5 (`CITY_MARKS_MIN_ZOOM`: the whole world) there are no circles, no hover, no click and no
overview city bubbles [T-1046, T-1048, T-1049]. Shade and circles are hidden behind the start menu,
and so are the city bubbles: the start menu's world map is blank. So is the map while a game loads
(`mapBlank` in `game/loading.ts`: from the start menu's pick or a whole-game load until the zone in
view's first figures of that game are in, a cached result, a rough solve's or an estimate [T-1533], or
else until the loading card goes and the clock is released): no shade, circles, zone outlines, outside
shade, loading grey, city bubbles or labels over the basemap; they all arrive in one step, with no fade
[T-1051]. The card and the held clock stay until the final [T-347], or go with a cached result of
the game's own network once that network is in [T-2560].

## Globe

[T-033; `map/gl.ts` `Camera`, `PROJECT_GLSL`; `map/mapView.ts` `followProjection`] Settings > Map
offers **Flat** (the default) or **Globe**, remembered per browser with the other view settings.
Switching changes only the view: the game, demand and the save carry on untouched. Globe is
MapLibre's own "globe" projection: a globe out to zoom 11, blending into the flat map between zoom 11
and 12, and the flat map itself from 12 in, so building close in is exactly as on the flat map.

Everything the flat map draws is on the globe in its place: track and lines, stations, trains,
platform crowds and transfer bubbles, crossing marks, demand bubbles and their lines, the realm
shade (cut into pieces no longer than about 5.6 degrees so it bends with the sphere, with caps over
both poles), zone outlines, the real network, labels, airport badges, realm circles and loading bars.
Sizes on screen are the flat map's at the same zoom (a line's width, a dot, a bubble's radius).
What lies beyond the horizon is not drawn and takes no click or hover; a DOM mark whose place is
there is hidden (`hiddenOnGlobe` in `map/live.ts`; a mark placed through the overlay's projection,
which gives NaN there, goes through `placeAt` in `map/screenPlace.ts`, which hides it, T-1777). Clicks, hovers and the building tools work at
every zoom on the globe: picking projects with the globe's camera and narrows its candidates round
the pointer's own place (`pointerLocal` in `map/pick.ts`).

The map, flat or globe, is always seen from straight above (MapLibre's `maxPitch` 0). A right-button
drag turns it, as does Ctrl with a left drag (MapLibre's own), while Settings > Map's "Rotate map with
right drag" is on (off by default); fingers and the keyboard never turn it. In the blueprint tool and the
bus tool the right button stays theirs (a pan; a step back or removal), so no drag turns the map while
either is on. Turning the switch off eases the map back to north. While the map is turned, a needle
pointing north shows beside the find button (hover "Turn map back to north"); a click eases the map
back to north and the needle goes [T-1907, T-2031; `followRotation` in `map/mapView.ts`,
`NorthButton` in `ui/DemandSelector.tsx`]. Everything drawn over the map follows the turn; names,
tags, faces and other DOM marks stay upright.
