# The network diagram

The design, with its references and stages: [notes/T-2146.md](../notes/T-2146.md). Built so far: the
graph (T-2159), the layout (T-2160), the view and its drawing (T-2161), names (T-2162), and clicks, the
camera and the picture (T-2163), and city blocks (T-2407; notes/T-2407.md); the faint city outlines and water
are T-2168 and T-2169.

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
- **Faint levels** (T-2272): below those, k from -14 to -4 (0.62 to 3.5 px, as far out as pills go), the big stations only, drawn
  at 0.6 opacity: of the stations with 3 lines or more stopping, most lines first, each one with no station
  kept before it within 120 px at the level. Their names keep off each other's names only (over lines and
  pills, the halo keeping them legible). The levels proper are placed as if these were not there.

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
its own (T-2405). Each line carries its colour and whether it runs high-speed trains.
Places are metres east and north of the network's middle.

## The layout

In a worker of its own (`workers/diagram.worker.ts`, `diagram/layout.ts`, `diagram/lsqr.ts`), one
graph at a time, the newest waiting one next; the last drawing stays up meanwhile. The layout, drawing
and names use no maths beyond `+ - * /` and square roots (`diagram/fmath.ts` has atan2 and powers from
those), so every browser, and the Node checks, lay a graph out to the same bits.

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
  pitches either side of the bearing, plus half the gap (weight 1); blocks no track joins are linked, the
  shortest links that join them all (each between the nearest pair of up to 64 stations a network), at a
  lonely hop's target for the real length (so islands and separate networks pack close); each block held to
  where it is at 0.001 times the root of its stations, each station between blocks at 0.001, which only picks
  where the hops leave a choice, the big cities staying. Then up to 11 more solves keep boxes apart (10): any
  two overlapping, grown by a fifth and padded by a gap of 16 pitches, are pushed apart along the axis they
  overlap less on until the spacing is met. The hops between blocks then keep the length placed as their
  target, so B and the drawing do not pull cities together; a station between blocks on a hop 4 median hops
  long or more ends a segment. After B, boxes B brought within 12 pitches of each other are moved apart to the
  gap again, the hops between blocks held as B left them.
- **Lonely hops** (T-2289): a station is in a city when 3 others lie within 2.5 km of it in reality, a city
  being such stations within 2.5 km of each other (`cityParts`). A hop whose lines are all of one colour,
  between two stations with two hops or fewer each, that leaves a city or joins two, with an end alone (no
  station within 4 km whose lines share no colour with its own) has 0.4 of its target, at least 4 pitches
  (never more than it had), so the long thin single-line stretches between cities draw short while parallel
  suburban branches and junctions keep theirs. The pitch is set from the targets before this.
- **Sides** (in A and B, each round): two stations that share no edge and lie nearer in reality than 1.5 times
  either one's mean real hop keep at least their spacing (their half widths plus half the median hop) between
  them along their real bearing (weight 3), so a line of short hops beside one of long hops cannot carry its
  stations past the other's to the wrong side (T-2290).
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
  method from the last drawing, which finds the drawing nearest it, each term's correction at most 0.3
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
- **Exposed** (`layout.blocks`, for T-2168's outlines and T-2169's water): per node its block (or none), per
  block its drawn box, its move, and an affine fit from real places to drawn ones over its stations.

`LAYOUT_R2` keeps round 2's options (alpha 0.35, pitch the median hop over 12, edges 1.6 half widths).
