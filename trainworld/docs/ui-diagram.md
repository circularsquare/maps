# The network diagram

The design, with its references and stages: [notes/T-2146.md](../notes/T-2146.md). Built so far: the
graph (T-2159), the layout (T-2160), the view and its drawing (T-2161), names (T-2162), and clicks, the
camera and the picture (T-2163), city blocks (T-2407; notes/T-2407.md), the zones' borders (T-2464;
notes/T-2464.md) and city names far out (T-2438); water is T-2169.

## The button and key M

A button at the top left of the Cities tab, on the row with the leaderboards' trophy and the
achievements' medal at its right (`ui/DiagramButton.tsx`, placed by `AchievementsButton`; Anita,
2026-11-03), and key **M** (`diagram/view.ts`) swap the map for the diagram, and back. Hover: "Show
network diagram (M)", and while the diagram is up "Show map (M)". **Esc** goes back to the map too, once
nothing else is left for it to undo: a question, a drag and the selection come first (`main.ts`, T-2416).
The diagram is a 2D canvas over the map; the dock and top bar stay, and the map's upper left gives way to
the diagram's own row (`DiagramTopLeft`): a cross back to the map (hover "Show map (M)", T-2416), the
save button once a drawing is up, the busy tag, and during the first layout
the demand status's tag: the bunny's train, "Calculating diagram" and a thin loading bar (hover: its
percent) fed by the layout worker's progress (T-2235; A to 0.2 of the layout, B to 0.7, the routing
the rest; the layout 0.85 of the whole, the names the last 0.15; a post each percent).
Drag pans, the wheel zooms about the pointer, drawn once a frame. `?diagram=r2` lays it out with round
2's options.

- **The camera** (T-2163): each time it opens it is centred on the selection (a station's complex; for
  a line or group, of its stops the one nearest the map's centre), else on the complex nearest the
  map's centre; at a pitch of 7 px the first time, as left after. Going back moves the map, keeping its
  zoom, the same way (the selection's node nearest the diagram's middle, else the complex nearest it),
  only if the diagram was panned, zoomed or the selection changed while it was up.
- **Clicks and hovers** are the map's: a pill or a name selects its station (the complex key's
  station, so the complex view), a line selects the line, a press moving under 4 px is a click,
  nothing clears the selection. Picked within 3 px. Hovers show the map's tags: a station's
  (`stationTip`), a line's bullet and name, status, and riders on the stretch under the pointer
  (`dailyRiders`, the stretch from the ribbon's chain index).
- **The focus**: a selected line, train's line or group draws full and the other lines muted as the
  map does (0.45 of the width at half opacity); with a station selected its lines are the full ones and
  its pills outlined in the accent colour, half again as thick. Pills and names of stations off the
  full lines fade to 0.35. A line under the pointer is drawn full over the rest.
- **Build keys** (1 to 4, N, S, C, G, U, O, T) and any build tool picked go back to the map first.
- **Save picture**: an icon button (download) with the hover "Save picture" saves the whole diagram,
  unselected, in the current theme, as "<game name> diagram.png" ("Network diagram.png" without a
  name), the busy tag saying "Saving picture" meanwhile. Its pitch is the largest names level of at
  most 10 px (9.9 px) that keeps it within 16,000 px a side and 120 million px; failing all, the lines
  alone as large as fits.

## The drawing

`diagram/draw.ts`. Each line is one ribbon (casing in the background colour, strokes 5/6.4 of a pitch
wide), corners rounded with radius 6 stroke widths (3 at a station, so a line turning there passes
under its pill) plus the bundle's half width, plus or minus the stroke's offset, so a bundle's bends
are concentric. A slot change on a straight run is a 45 degree jog. Pieces the layout leaves going out of a
node the same way (where it could turn none of them without folding a line) are drawn as one bundle there:
the piece with the most strokes where it is, each other beside it along its whole length, on the side its
stations lie in reality, nearest first (`besideShifts`, T-2405). Any two pieces' legs still drawn along each
other over more than a pitch, nearer than their bundles' half widths together, have the piece with fewer
strokes moved aside until the bundles sit side by side, the way it already lies, else the way its stations
lie (`apart`; up to 4 passes). So no line is drawn under another (`checks/t2405-diagram-hidden.mjs`). A line with high-speed trains (T-817) has a
stripe of the background colour along its middle, 0.34 of its stroke, so it reads as two thin strokes of its
colour (T-2272; notes/T-2272.md for the maps this follows). A stroke that would be under a device px wide
is drawn a px wide at its width's share of its opacity, at least a quarter of it, so no line vanishes far
out; a high-speed line drawn so has no stripe (T-2408).

**Pills.** Each station's pill lies along an axis: of the four octilinear axes through its place, the
one every line stopping there crosses, and of those the one giving the shortest pill. Where none
crosses them all (a line turning at the station, or ending along an axis), the axis may pass up to 2
pitches to one side of the place, at a pitch of pill length a pitch. A line's stop is where its drawn
ribbon crosses the axis; a line ending there has its last run carried on, or cut back, to the axis; a
line running beside the axis within 1.25 pitches lies under the pill's edge. So a pill is always a
straight capsule on its lines, white with the ink outline in both themes, two pitches across; in the dark
theme a thin edge of the map's ink (0.16 of a stroke) runs round the outline, so the
ring shows on the dark background (T-2272). A **local stop** is drawn small, its fill half a stroke across
each side of its axis and its outline 0.24 of a stroke (about 1.2 pitches across): a station one line stops
at, or one a line passes without stopping while the lines stopping are all of one colour; a line's end
never is one (T-2272, `localStops`). Stops
along the axis more than 1.6 pitches apart make separate pills, unless they are within 4 pitches and
no line crosses the axis between them; a line running along the axis further off gets a round stop
on its own ribbon. Pills of one station are joined by a passage, 0.45 of a pill wide. Below a pitch
of 0.625 device px, no pills (T-2272).

The drawing is made in the layout worker and posted with the layout: each line's path as commands,
made a Path2D on the page when first painted.

## Zone borders

`diagram/borders.ts` (T-2464; notes/T-2464.md). The borders are decided in the drawing, the way the Tube map draws
fare zones: a border says which drawn stations are in which zone, not where the ground boundary runs. A station's
zone is the outline holding it (the map's coarse outlines, `packs/outlines.json`), else the nearest within 40 km;
the page sends the outlines of the zones near the network once each, in mercator, with the graph's frame worked out
from the stations' own places. The layout worker makes the borders after the names (and again, alone, for outlines
that arrive after a layout), between blocks as inside them.

- **The region**: on a grid of one pitch, every cell within 5 pitches of a line's casing or a pill; then grown by
  50 pitches and shrunk by 50 again (never inside the 5), so gaps and bays under about 100 pitches close and the
  outline keeps well clear of the lines instead of hugging them (T-2488, T-2491); then every gap
  the network encloses filled. So each zone with stations is a closed outline round its own stations and lines, with
  no holes.
- **The zones**: each cell takes the zone of the pill nearest it through the region (steps of a cell, straight or
  diagonal, from the cells under each pill), so every station is on its own zone's side and a line leaving a zone
  crosses the border between its two stations, near the middle. A station passed and never stopped at has no pill
  and no say.
- **Shape**: the grid's boundaries, each run between the points where borders meet traced once, are fitted with
  runs at 0, 45 and 90 degrees: from each point the farthest one reachable by one run or two (one straight, one
  diagonal) staying within the point's tolerance of every point between, and moving no pill across. The tolerance is
  the point's clearance less 0.3 pitch, at least 0.7: on the region's edge, clearance from the lines and pills, at
  most 3, and outward (away from the lines) up to 10, so the edge fills notches and steps with long runs (T-2491);
  between zones, clearance from those and the region's edge, at most 12. Corners are rounded with a radius of 3
  pitches (less on short runs).
- **Drawn** under everything, the lines' casings cutting it, in the map's outline grey (#73798b) at 0.7 opacity,
  0.2 of a pitch wide; below a device px, a px wide and fainter by its width's share, never below half (the lines'
  rule, T-2408, with a higher floor). One path; a border two zones share is drawn once.

## Water

`diagram/waterReal.ts`, `diagram/water.ts`, `diagram/waterSea.ts` (T-2169, T-2497, T-2516, T-2540; notes/T-2169.md,
notes/T-2497.md, notes/T-2516.md, notes/T-2540.md). Rivers are drawn as bands and the sea (lakes too) as a coast, under everything else, in the theme's
map water mixed 55% of the way from the background (`WATER_SHARE` in draw.ts, in Oklab: nearer the land, so a river or
coast drawn a little wrong shows less). The page names the zones near the network and the worker fetches their water
masks from the packs (the 25 m rasters the track model prices water with), each zone once. Before a first layout the
worker finds the sea on the ground (the coarse and far grids and the land masses below), which the layout keeps open
(The layout, Water); the rivers and the drawing come last, after the borders, posted on their own.

- **On the ground**, near the network only. A coarse grid (200 m, within 15 km of a station) says where the **sea**
  is: water at least 2 km across everywhere within it, and water within 400 m of that counts as the sea's. Each sea
  connected on the ground (four neighbours) is one **body**. A fine grid (40 m) finds **rivers**: water at least three
  cells (about 120 to 160 m) across that is not the sea's, of the rivers some hop's track passes over, each followed
  from the tiles within 3 km of a track out to 12 km from a station (where it runs on past that, or out of the masks,
  its end is **open**). Each river is thinned to its centre line and traced as arcs between the places where rivers
  meet or end; branches ending free under 400 m (or three half widths) and rivers under 600 m in all are dropped. A
  river ending free within 400 m of another is joined to it by a straight arc (a bridge the mask draws as land), and
  one ending free within 1 km of the sea ends at it. Where a river meets the sea, its end takes the body nearest. A
  hop crosses a river where its track crosses the centre line an odd number of times; one hop crossing two arcs within
  600 m of each other (where rivers meet) crosses the wider only.
- **Which rivers are drawn**: only rivers (water connected on the ground) some hop crosses, and each drawn **whole,
  as one band**: its crossed arcs, the arcs joining them along it, the way from them to its nearest mouth, and the way
  to each mouth at least 150 m wide each side of its middle within 25 km (so the East River runs from the harbour to
  Long Island Sound). A river with no mouth runs on to its open end; one with neither (it stops on the ground at both
  ends) is not drawn. A narrow arc at an end of what is drawn (all its crossings under 150 m each side) that crosses
  another line is left out, and an arc left ending free with no crossing goes with it; a river narrow wherever it is
  crossed is drawn only where it crosses no other line, else not at all. So **no band is drawn in pieces**.
- **How wide** (T-2528): a band is the river's real width (its median along each arc) at the drawing's scale round it,
  the scale field: at each place, the nearest 16 stations' own scales (their local maps', drawn over real), a geometric
  mean weighted by exp(-d^2 / 2 R^2), R 1.5 km or the fourth nearest's distance, so it changes smoothly from place to
  place. Smoothed along the river over 1.5 km either side, then at least half the river's widest drawn stretch times
  their real widths' ratio (so a big river narrows gently downstream and a narrow branch stays narrow), between three
  lines' width (3 x 5/6.4 of a pitch) and 18 pitches. So the Hudson and East River beside Manhattan, drawn large, are
  wide, the Hudson up past Yonkers stays a band, and an outskirt river such as the Navesink stays near the least
  width. Where the band would cover a pill, or a line running alongside it (within 30 degrees) that it does not cross,
  it narrows to keep half a pitch off it. It is drawn as runs of one width, in quarter-pitch steps, so **every edge
  of the water runs at 0, 45 or 90 degrees** (Anita, 2026-11-05): a run holds its width at least 4 pitches,
  narrowing early to the least room ahead and widening only by half a pitch or more where that holds as long; a width
  never changes at a bend, and where it changes the step is square (the runs cut square there), while a band's own
  ends, and where rivers meet, are round.
- **Rivers in the drawing**: bands with round ends. Each crossing is a straight
  run square to the hop's bundle, at the real share of the hop (kept within 25 to 75%; moved to 20, 35, 50, 65 or 80%
  where it would otherwise end on another line), from the side the river comes from on the ground, on past the
  outermost casing by half a pitch; a run that crosses every line of another hop the river crosses serves for both.
  Between crossings the band is routed on a one-pitch grid like a line (runs at 0, 45 and 90 degrees, a 45 degree bend
  costing 10 pitches of length and a 90 degree one 30), dearer within a pitch of a casing, near the river's course
  carried into the drawing through the nearest stations' local maps (T-2488's border lines), within 24 pitches of it.
  It may cross other lines: each pitch inside a casing costs 10 pitches of length, inside a hop the river crosses
  elsewhere 60, inside a pill 150, so it crosses a line where going round costs more (T-2516: a band that runs on
  smoothly reads better than one winding through the gaps, and the drawing often puts a river's two banks on the wrong
  sides of a line). Where that way crosses lines, the run is tried the other way round too, and the one crossing
  fewer taken. So **a band crosses each hop that crosses that river once**, save where it has to cross its own hop
  again, and other lines where the drawing leaves no way round.
- **Ends**: a mouth goes into the drawn sea of the body it meets: to the nearest cell of that body's drawn sea, at
  least a pitch past the band's edge from the coast, within 80 pitches of where its place on the ground is carried. A
  mouth at a sea not drawn near (Newark Bay) goes on by water on the ground (the coarse grid's water, the shortest way
  within 30 km) to a sea that is drawn, as its own band (the Kill Van Kull to the harbour); failing both, into whatever
  drawn sea is nearest within those 80 pitches. An open end runs on
  straight, away from the drawing's middle, to 12 pitches past the drawing's box. A free end (the river ends on the
  ground) runs 3 pitches past its last crossing, a wide river's 5. A river whose mouths all find no drawn sea that way is
  not drawn.
- **Connected seas**: pieces of one body drawn apart (each 60 square pitches or more) and not joined by a drawn river
  are joined by a channel band, the shortest way between them on the drawing's grid, routed like a river, where that
  is within 40 pitches.
- Corners are rounded at 2.5 pitches, unless that would cut across a line.
- **The sea everywhere** (T-2540): past the coarse grid, a far grid (1 km a cell, coarser for a big network, at most
  1.5 million cells; 100 km past the coarse grid) holds the masks' water at least three of its cells wide; a place no
  mask covers takes the nearest known cell's (so open sea far from the stations is sea, and land between them land).
- **Land masses**: land on the ground connected round the water that parts it is one land mass. The sea parts land,
  and so does water at least about 600 m wide (three coarse cells) all the way across; rivers narrow upstream, so they
  part nothing, while straits and channels between seas do (the Tsugaru, Kanmon and Seto straits).
- **Islands with no station** (T-2550): a land mass by the sea that no station is on (a station on water takes the land
  within 2 km) is left out, taken as sea everywhere, when its area is under a tenth of the land in the 100 km square
  round its middle (the far grid's land, its own included): so Shodoshima, Okushiri and the small islands of the Seto
  Inland Sea go, while Awaji, Amakusa and Sado stay. Each one kept is drawn once, by the stations within 15 km of the
  station nearest it (by the far grid's steps), on that station's land mass.
- **The sea in the drawing**: on a grid over the drawing (1.5 pitches a cell, coarser for a big drawing, at most a
  million cells; 40 pitches past the network's box), each cell is sea where its four nearest drawn stations, voting by
  nearness, find the sea on the ground at the place each carries the cell's offset back to through its local map. So
  the coast between stations is the real one carried into the drawing, and open sea stays sea however far it lies.
  Past its hops' mean drawn length a station carries the rest of the offset by its region's map (the similarity that
  best fits its land mass's stations within 50 km of the middle of its 10 km square), where that draws the ground
  smaller than its own (T-2550: a city's map carried out over empty land drew it at the city's scale, a spit 180
  pitches long east of Hakodate and Awaji at Kobe's scale). A voter finding land of another land mass with stations on
  it, or an island with no station that is not its own to draw (above), counts it as sea, since those stations draw
  it. So does land it carries the cell to across the sea (a run of 2 km and 30% of the way on the ground) nearest a
  station of the far shore within 8 km that is not among the cell's voters (T-2528: that shore is drawn where its own
  stations are; Connecticut's maps carried the middle of Long Island Sound, drawn wider than the Sound at their scale,
  onto Long Island, drawing it as land with lakes). Every pill and the 2.5 pitches round it is land. Smoothed (a 3 by 3 majority, twice; then opened and closed by 4
  pitches, so inlets and notches under about 8 pitches across go, the land round the pills kept).
- **Straits**: then a cell is sea where it meets a cell nearest another station, and the two nearest stations are on
  different land masses, or the straight way between them on the ground crosses at least 2 km of sea in one run and
  30% of its length; only a pill and half a pitch (and half a cell) round it stays land. So **land masses are drawn
  apart** wherever the drawing leaves room between their stations, and a bay or strait the drawing squeezes (Tokyo Bay)
  stays open as a channel.
- Sea under 60 square pitches (but a strait) and land without a station under 30 dropped, and land without a station
  of any size too unless most of its cells' land votes found an island with no station (T-2550: a piece of a land mass
  drawn again far from its own stations, off another station's map, goes), the coast is drawn the way a
  line is: fitted with runs at 0, 45 and 90 degrees, straying seaward up to 12 pitches but at most 0.35 of the way
  across the sea to the next land straight out from it (within a cell of a strait, half a pitch or 0.4 of a cell), and
  landward up to its clearance from the lines and pills less a pitch (between 1 and 8), never moving a pill across it,
  so it runs straight for long stretches (T-2497) without closing a bay or strait the grid has open. A ring that, with
  another fitted near it, still leaves a station in the sea is fitted again tight. Corners rounded at 4 pitches, filled
  even-odd. **Every station is on land**, even where the coast departs from the real one. Each connected piece of
  drawn sea takes the body most of its cells' votes found.
- Drawn under the zone borders and the lines' casings: the sea filled, the rivers stroked at least a device px wide.

## Names

`diagram/names.ts`, placed in the layout worker after the drawing. A station's name is its complex's:
of its members' names, the one with the most lines stopping, cut at the first "/". The page measures
each name in the interface font at 12 px times the interface scale (the map's station names' size)
and sends the widths with the graph. Drawn in the map's ink (`--map-ink`) with a 3 px halo of the
diagram's background, over everything else.

- **Order**: interchanges (two line colours or more stopping) by lines stopping, then line ends, then
  the rest by lines stopping. A station with no pill (passed, never stopped at) has no name.
- **Places**: eight round each of the station's pills, east first, then west, then the rest, each
  starting 3 px off the pill's outline at the point farthest that way (a local stop's names as round a full
pill, so they clear its own line); east, west, north and south
  also slid to be flush with that point either way. Off the pill's ends cost least (the lines cross
  its sides), then the corners. A name of 12 characters or more may go over two lines, broken at the
  space or hyphen nearest its middle.
- **One side along a run**: a station inside a piece's straight leg prefers the side of the leg with
  fewer other stations' pills within 6 pitches (east, else north, when even).
- **Two distances**: at the pill, or 1.5 pitches further out (over a line beside it). An interchange
  may also go 3 or 4.5 pitches out where lines meet all round it, but never over 32 px out.
- **Kept off**: a line's middle by half a pitch plus a px; other pills and passages (each at its drawn size); names already
  placed (`BoxGrid`, in px); and no name nearer another station's pill than its own. The box tested
  runs from the pill (or the lifted point) to the far side of the name.
- **Zoom levels**: placed afresh for a pitch of 7 px times 2^(k/4), k from -3 to 12 (4.2 to 56 px); a
  level's names are drawn at every zoom from it to the next, none below the first faint level. Since text keeps its
  px size while the drawing scales, a name past its level only shrinks toward its pill, inside the box
  tested, so nothing it was clear of comes nearer. Each level favours the last level's place for a name
  (by 1.2 of cost), so names move little across levels.
- **Faint levels** (T-2272): below those, k from -9 to -4 (1.47 to 3.5 px), the big stations only, drawn
  at 0.6 opacity: of the stations with 3 lines or more stopping, most lines first, each one with no station
  kept before it within 120 px at the level. Their names keep off each other's names only (over lines and
  pills, the halo keeping them legible). The levels proper are placed as if these were not there.

## City names

`diagram/cityNames.ts` (T-2438, T-2521). Below a pitch of 1.47 px no station names are drawn; each city block (The
layout, Blocks) has its name instead: big, in the grey of the map's place names (`placeNameInk` in `map/basemap.ts`,
each theme's own; T-2492) at half opacity, in the station names' font, centred on the middle of the block's box, and
painted under the lines (a callback `paint` runs after the water and borders, before the line casings), so lines and
stations cross it and it has no halo. Made in the layout worker after the borders (names and the box's middle), sized
and placed on every paint.

- **Size**: its width is 0.6 of the block's box on screen, within 1.5 to 7 times the station names' size (18 to 84
  px), so a name grows with the zoom and stays small beside a small block.
- **Overlaps**: biggest block (by stations) first. A name that would meet one already placed (with 6 px to spare)
  shrinks by a tenth at a time toward 1.5 times the station names' size; if it still meets one it is left out. A
  name off the screen blocks none.

- **The name**: the zone (the borders' zone of each station) holding most of the block's stations, ties to a city
  zone, then to more local trips; its shown name, as the map's (`game/zoneNames.ts`). The page
  sends it with the zone's outline. The biggest block of a zone has the zone's name. Each smaller block of the
  zone (T-2447; Japan's prefectures hold several) has its own place: the nearest city or town of the zone's place
  list (`game/placeList.ts`) to the middle of its stations, that no block in view has yet (a village, the 30
  biggest, only when no city or town is left). No name is shown twice. The page fetches a zone's list only when
  such a block exists, after the first drawing, and the worker names again when it arrives; until then (or with
  no list or no place left) the block has no name.
- The saved picture, too big for station names at any level, is the lines alone at the largest scale that fits, with
  the city names under them where the pitch is below 1.47 px.

## The graph

Built in the clock worker after each network state while the diagram is shown, and posted as
`diagramGraph` (`diagram/graph.ts`). Its nodes are station complexes (`complexOfNodes`), each keyed
by its lowest station node id on the lines. Each rail line becomes the chain of complexes it runs
through, first stop to last: its own path's stations, and the stations whose platform is within
35 m of its track and parallel (within about 25 degrees), found from `corridors.ts`' beside table (a
station's own edges' end samples name the edges beside it). A complex met again within 300 m with
one other between is met once. Edges join consecutive complexes of a chain and carry its lines, each with
the stroke it is drawn in there (`stroke`): lines of one colour, both high-speed or both not, share one where
at least half of the shorter one's track on that hop is track both run on (interlining); every other line has
its own (T-2405). Each line carries its colour and whether it runs high-speed trains. Each edge also carries its
track on the ground, as the first line on it runs it between the two complexes, kept to within 15 m (`hop`, for the
water's crossings, T-2169).
Places are metres east and north of the network's middle.

## The layout

In a worker of its own (`workers/diagram.worker.ts`, `diagram/layout.ts`, `diagram/lsqr.ts`), one
graph at a time, the newest waiting one next; the last drawing stays up meanwhile. The layout, drawing
and names use no maths beyond `+ - * /` and square roots (`diagram/fmath.ts` has atan2 and powers from
those), so every browser, and the Node checks, lay a graph out to the same bits.

- **The distortion dial** (T-2556, `DISTORT` in layout.ts, from 0 to 1; set at 0.5, T-2528): the terms that draw the network
  off its real scale each go to the dial's power: a hop's target over its real length (`(length / 1000 m)^-0.7` at 1,
  so its target is `1000 m x (length / 1000 m)^(1 - 0.7 x dial)`), the lonely and local hops' share (0.4 at 1) and
  the big sparse regions' floor (0.25 at 1). So 1 evens hop lengths fully and 0 draws every hop at its real length;
  the room for bundles and pills, the gaps the water and the blocks keep, the octilinear snapping and the routing are
  the same at any setting, so dense cores still open up. At 0.5 the power is 0.65, the lonely and local share 0.63 and
  the floor 0.5; the figures below are the dial's 1. `?distort=0.5` in the page's address lays out at that setting.
- **A**: real directions, each hop's length `1000 m x (length / 1000 m)^0.3`, 10 rounds up to 2,000
  complexes, rising evenly to 40 at 8,000 and over (a country's long stretches only reach their targets in
  that many; T-2289). With two blocks or more (below), A lays out each block as if alone: the hops between
  blocks or touching a station between them, and sides across blocks, are left out, and the complexes counted
  for its rounds are the biggest block's.
- **Blocks** (T-2407, `diagram/blocks.ts`): cities as `cityParts` finds them, joined where their stations lie
  within 15 km of each other, are blocks when they have 10 stations or more. Other stations, joined by their
  hops into groups: a group whose hops reach one block is taken into it (a branch out of a city), one reaching
  none is taken into the block with a station nearest it within 15 km (a piece of the city with no track to the
  rest, such as Staten Island's railway), else it is a block of its own (a small network alone); a group
  reaching two blocks or more stays between blocks. A block's box is its stations' bounding box. With one
  block nothing below happens, so a single city lays out as without blocks.
- **The blocks' stage**, between A and B, with two blocks or more: each block moves as one, and each station
  between blocks on its own, by one least squares solve: each hop between them runs its real bearing for its
  target plus, at each end in a block, the way from that station past the block's farthest station within 16
  pitches either side of the bearing, plus half the gap (weight 1); each crossing (below) between them is held
  (weight 1); blocks no track joins and no crossing ties are linked, the
  shortest links that join them all (each between the nearest pair of up to 64 stations a network), at a
  lonely hop's target for the real length (so islands and separate networks pack close); each block held to
  where it is at 0.001 times the root of its stations, each station between blocks at 0.001, which only picks
  where the hops leave a choice, the big cities staying. Then up to 11 more solves keep boxes apart (10): any
  two overlapping, grown by a fifth and padded by a gap of 16 pitches, are pushed apart along the axis they
  overlap less on until the spacing is met; two blocks a crossing ties are not, since one is drawn over the
  other's line. The hops between blocks then keep the length placed as their
  target, so B and the drawing do not pull cities together; a station between blocks on a hop 4 median hops
  long or more ends a segment. After B, boxes B brought within 12 pitches of each other are moved apart to the
  gap again, the hops between blocks held as B left them.
- **Crossings** (T-2534): two hops cross on the ground when they share no station and no line and their tracks
  cross an odd number of times, at 20 degrees or more, at least 100 m from every end of either. Outside dense
  places, where fewer than 12 stations lie within 2.5 km of the crossing, each crossing is held in the drawing:
  the point at its share of the shorter hop's track lies on the longer hop's line (A: its real way, weight 3,
  crossings within one block; the blocks' stage: 1, crossings between blocks; B: its direction, 10; the snap: its
  way after B, 30), and in the blocks' stage (0.3), B (3) and the snap (10) that point and the one at its share of
  the longer hop are the same. So a city slides along a long hop drawn short and keeps its own scale. A hop
  crossing others at two places or more is at least as long as the span between them at the crossed hops' scale,
  plus 4 pitches each side. Each station within 1.5 km of a crossing's longer hop, outside dense places, stays on
  its side of it in B and the snap (weight 3, its half width and 2 pitches off). So a long-distance line passing
  through a small city's network, stopping at none of its stations, is drawn through it; a dense core is left
  free, where holding every crossing would bend its lines about.
- **Big sparse regions** (T-2550): a region's scale round each 10 km square is the targets over the real lengths of
  the hops with middles within 60 km of it, each weighted by (1 - d^2 / 60 km^2)^2; where that is under 0.25 (a hop of
  1 km has its own length as its target, so this is a quarter of the scale of a core of kilometre hops), each hop there
  has its target raised by the shortfall. So a big region with few stations is drawn more by its area than by its
  stations (Anita, 2026-11-05: "the sparse big regions of tohoku and hokkaido ar e a bit small"): on all Japan, at the
  dial's 0.5, Hokkaido is drawn at about 0.25 to 0.26 of the Kanto core's scale and northern Tohoku 0.25 to 0.27,
  while a single city's network is hardly touched (its sparsest edges only). After the lonely, local, parallel and crossing hops.
- **Water** (T-2541): the layout reads the sea on the ground (Water below: the coarse and far grids and the land
  masses), which the worker finds before a first layout; a rebuild takes the last found. The scale round a station is
  the targets of the land hops with middles within 10 km over their real lengths. A **hop across water** (its ends on
  different land masses, found out to 2 km, or a run of sea on its track of 2 km and 30% of it) has at least twice its
  real length at the geometric mean of its ends' scales, and 12 pitches past its own target between land masses.
  **Stations with the sea between them**: each station within 3 km of the sea with its 4 nearest within 40 km that are
  on another land mass or have such a run of sea on the straight way between them. Each pair's scale is the targets
  over the real lengths along the shortest way between them by track, up to 6 times their distance and 200 km (else
  the scale round its ends), so a bay is as wide as its shores are drawn, not as a city on one of them; it is kept
  1.25 times its real distance at that scale apart along its real bearing, and 12 pitches between land masses: in A
  and B when nearer (weight 3; in A only within one block), and in the blocks' stage each pair of units is held to
  its pairs' lengths and bearings (weight 1 in all). So straits and bays get room before land does: Japan's four main
  islands are drawn apart, Tokyo Bay is open, and Boso and Izu read as peninsulas by the water round them, while a
  small piece of land with few stations keeps the lonely and local hops' short targets (Anita, 2026-11-05: peninsulas
  need not be big).
- **Lonely hops** (T-2289): a station is in a city when 3 others lie within 2.5 km of it in reality, a city
  being such stations within 2.5 km of each other (`cityParts`). A hop whose lines are all of one colour,
  between two stations with two hops or fewer each, that leaves a city or joins two, with an end alone (no
  station within 4 km whose lines share no colour with its own) has 0.4 of its target, at least 4 pitches
  (never more than it had), so the long thin single-line stretches between cities draw short while parallel
  suburban branches and junctions keep theirs. The pitch is set from the targets before this.
- **Local hops** (T-2491): a hop between two local stops (The drawing, Pills), not lonely, has 0.4 of its target, at
  least 3 pitches (never more than it had), so a long branch of plain stops is not drawn longer than the core it
  leaves, while interchanges keep their room. Also after the pitch is set.
- **Sides** (in A and B, each round): two stations that share no edge and lie nearer in reality than 1.5 times
  either one's mean real hop keep at least their spacing (their half widths plus half the median hop) between
  them along their real bearing (weight 3), so a line of short hops beside one of long hops cannot carry its
  stations past the other's to the wrong side (T-2290). In A's last three rounds such a pair's offset across its
  real bearing is also held at nothing (weight 1), so lines side by side are not sheared against each other (T-2520).
- **Parallel hops** (T-2520): two hops are parallel on the ground when they share no station and no line, their
  bearings are within 15 degrees, each one's middle lies within 1 km of the other's line, and they overlap along it
  by at least half the shorter (Manhattan's avenue lines). Their targets are drawn at one scale: each hop's drawn
  length per real metre is averaged with its partners' (weighted by real length) three times, after the lonely and
  local hops and before the room for bundles. In B each edge's angle, for choosing its direction, is the mean of its
  own and its partners', so lines side by side take the same direction.
- **B**: 30 rounds of octilinear least squares from A: each edge toward the nearest free 45 degree
  direction at both ends, straight through two-edge stations, near the last round (0.1) and A (0.01),
  unconnected stations nearer than 1.12 times their spacing (their half widths plus half the median
  hop) pushed out to that (3.5), and a line through a junction keeping its real turn there (1). A stroke
  pitch is the median hop over 9; a node's half width is its widest bundle's; every edge is at least 1.4
  times its ends' half widths.
- **Folding back**: a line that turns less than a right angle in reality never leaves a station the way
  it came in, or within 45 degrees of it (a U-turn or a hairpin). Each step below avoids it where it can.
- **Segments** between key nodes (a branch, a line's end, the lines changing), split at the station
  farthest off whenever its stations fit neither a straight run within 11 degrees nor one bend, within
  0.5 of the median hop.
- **The snap**: four rounds of one solve on every node: a piece within 11 degrees of a 45 degree
  direction after B, or in the last round, is made exact with its stations on it (weight 30), the way
  the real line runs; every hop keeps its length along its direction (its snapped piece's, else B's, or
  the real one where B turned it back), at least half its target (30); of two snapped pieces that would
  fold a line between them, the one further off its direction is left to the routing; a line turned
  more than 45 degrees past its real turn at a station, by B or by any round so far, is drawn back
  toward it (20); unconnected
  stations too close are pushed apart (2); everything else stays near B (1).
- **Routing**: a piece within 5 degrees is drawn straight (the separation makes it exact), unless that
  folds a line against a straight neighbour; any other as an orthogonal leg and a diagonal leg or the
  reverse, whichever folds fewer lines (4 times its length a fold), leaves its ends by ways no other
  piece takes (its length each) and passes nearer its stations, chosen in order and then again against
  all the others; each station goes where it projects onto the drawing, in order, at least half an even
  share of the piece from the next.
- **The separation** (3 passes, each drawn again as above): the drawing in terms that keep it
  octilinear: each piece's ends free, each station a distance along its leg, a bent piece's legs
  meeting where their directions say. Of two straight pieces leaving a node the same way, all but one
  get a first leg 45 degrees off toward where their stations lay after B, or the other side where that
  folds a line, and none where both would; the one kept is one that cannot turn without folding a line,
  else the one with most stations. Linear terms: straight pieces stay straight (a station pushed across
  one moves both its ends alike); stations keep their order along a leg and gaps of half their hop's
  target or 1.25 times their half widths together, round a bend too; a leg with no stations may shrink to
  nothing, but keeps 2 pitches where the piece would then leave its end a way that folds a line (and a
  pass whose drawing folds a line its start did not is done again with both legs of those bent pieces
  kept so); unconnected stations within 1.25 times their spacing are pushed to 1.15
  times it, and stations within their half width, the leg's and a pitch of another piece's leg are
  pushed off it. These are linearised about the drawing, 10 rounds a pass. A first layout meets them by
  sweeps of projections (each term not met moves its variables the least way); a rebuild by Hildreth's
  method from the last drawing (a new piece end, such as a line extended, starting at a neighbour's last place plus
  its routed offset from it, T-2556), which finds the drawing nearest it, each term's correction at most 0.3
  of a hop, so an edit moves little. Drawn after it, a station keeps only its order (a tenth of an even
  share of its piece).
- **The spread**: stations still nearer than their spacing slide along their pieces, each the least
  that clears the pair, between its neighbours, only where it leaves fewer close pairs; failing that
  both of the pair part of the way. Key nodes stay, so no angle changes.
- **Order**: on each piece one stroke per line, lines sharing a stroke on each of its edges sharing one,
  left to right by `lineBands.ts`' pairwise rule on
  the diagram's own places (follow two lines ahead to where they part, the one turning left goes left;
  else look back), with a fixed order where three or more go round in a circle.
- **A rebuild** starts from the last layout's B places by node key, skipping A, with 10 B rounds held
  to the last layout at 0.05 in place of A's 0.01: a new complex starts
  at its placed neighbours plus its real offset at the hop's scale, or failing any at its real place
  at the old layout's overall scale. Its separation keeps near the last drawing only when at least 90%
  of its nodes were in it; otherwise it separates as a first layout does. With blocks, the stage runs only
  where the hops between blocks and the links (by node keys) differ from the last layout's, or two boxes
  overlap: each hop and link the last layout had is held as it is drawn, the new ones get their targets, and
  boxes within the gap but not overlapping at the start are left so; each block's last places, B's and the
  drawing's, move by the block's move, so an edit inside a city moves no block.
- **Exposed** (`layout.blocks`, for the city names and T-2169's water): per node its block (or none), per
  block its drawn box, its move, and an affine fit from real places to drawn ones over its stations.

`LAYOUT_R2` keeps round 2's options (alpha 0.35, pitch the median hop over 12, edges 1.6 half widths).
