# The network diagram

The design, with its references and stages: [notes/T-2146.md](../notes/T-2146.md). Built so far: the
graph (T-2159), the layout (T-2160), the view and its drawing (T-2161) and names (T-2162); clicks,
city boxes and water are T-2163, T-2168 and T-2169.

## The button and key M

A button at the top left of the Cities tab, on the row with the leaderboards' trophy and the
achievements' medal at its right (`ui/DiagramButton.tsx`, placed by `AchievementsButton`; Anita,
2026-11-03), and key **M** (`diagram/view.ts`) swap the map for the diagram, and back. Hover: "Show
network diagram (M)", and while the diagram is up "Show map (M)". The diagram is a 2D canvas over the
map; the dock and top bar stay, and of the map's upper left only the busy tag, which says
"Calculating diagram" until the first layout is up.
Drag pans, the wheel zooms about the pointer, drawn once a frame. It opens centred on the complex
nearest the map's centre, a stroke pitch 7 px. `?diagram=r2` lays it out with round 2's options.

## The drawing

`diagram/draw.ts`. Each line is one ribbon (casing in the background colour, strokes 5/6.4 of a pitch
wide), corners rounded with radius 6 stroke widths plus the bundle's half width, plus or minus the
stroke's offset, so a bundle's bends are concentric. A slot change on a straight run is a 45 degree jog.

**Pills.** Each station's pill lies along an axis: of the four octilinear axes through its place, the
one every line stopping there crosses, and of those the one giving the shortest pill. Where none
crosses them all (a line turning at the station, or ending along an axis), the axis may pass up to 2
pitches to one side of the place, at a pitch of pill length a pitch. A line's stop is where its drawn
ribbon crosses the axis; a line ending there has its last run carried on, or cut back, to the axis; a
line running beside the axis within 1.25 pitches lies under the pill's edge. So a pill is always a
straight capsule on its lines, white with the ink outline in both themes, two pitches across. Stops
along the axis more than 1.6 pitches apart make separate pills, unless they are within 4 pitches and
no line crosses the axis between them; a line running along the axis further off gets a round stop
on its own ribbon. Pills of one station are joined by a passage, 0.45 of a pill wide. Below a pitch
of 2.5 device px, no pills.

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
  starting 3 px off the pill's outline at the point farthest that way; east, west, north and south
  also slid to be flush with that point either way. Off the pill's ends cost least (the lines cross
  its sides), then the corners. A name of 12 characters or more may go over two lines, broken at the
  space or hyphen nearest its middle.
- **One side along a run**: a station inside a piece's straight leg prefers the side of the leg with
  fewer other stations' pills within 6 pitches (east, else north, when even).
- **Two distances**: at the pill, or 1.5 pitches further out (over a line beside it). An interchange
  may also go 3 or 4.5 pitches out where lines meet all round it, but never over 32 px out.
- **Kept off**: a line's middle by half a pitch plus a px; other pills and passages; names already
  placed (`BoxGrid`, in px); and no name nearer another station's pill than its own. The box tested
  runs from the pill (or the lifted point) to the far side of the name.
- **Zoom levels**: placed afresh for a pitch of 7 px times 2^(k/4), k from -3 to 12 (4.2 to 56 px); a
  level's names are drawn at every zoom from it to the next, none below the first. Since text keeps its
  px size while the drawing scales, a name past its level only shrinks toward its pill, inside the box
  tested, so nothing it was clear of comes nearer. Each level favours the last level's place for a name
  (by 1.2 of cost), so names move little across levels.

## The graph

Built in the clock worker after each network state while the diagram is shown, and posted as
`diagramGraph` (`diagram/graph.ts`). Its nodes are station complexes (`complexOfNodes`), each keyed
by its lowest station node id on the lines. Each rail line becomes the chain of complexes it runs
through, first stop to last: its own path's stations, and the stations whose platform is within
35 m of its track and parallel (within about 25 degrees), found from `corridors.ts`' beside table (a
station's own edges' end samples name the edges beside it). A complex met again within 300 m with
one other between is met once. Edges join consecutive complexes of a chain and carry its lines.
Places are metres east and north of the network's middle.

## The layout

In a worker of its own (`workers/diagram.worker.ts`, `diagram/layout.ts`, `diagram/lsqr.ts`), one
graph at a time, the newest waiting one next; the last drawing stays up meanwhile.

- **A**: real directions, each hop's length `1000 m x (length / 1000 m)^0.3`, 10 rounds.
- **B**: 30 rounds of octilinear least squares from A: each edge toward the nearest free 45 degree
  direction at both ends, straight through two-edge stations, near the last round (0.1) and A (0.01),
  unconnected stations nearer than 1.12 times their spacing (their half widths plus half the median
  hop) pushed out to that (3.5). A stroke pitch is the median hop over 9; a node's half width is its
  widest bundle's; every edge is at least 1.4 times its ends' half widths.
- **Segments** between key nodes (a branch, a line's end, the lines changing), split at the station
  farthest off whenever its stations fit neither a straight run within 11 degrees nor one bend, within
  0.5 of the median hop.
- **The snap**: one solve on every node: a piece within 11 degrees of a 45 degree direction is made
  exact with its stations on it (weight 30), unconnected stations too close are pushed apart (2),
  everything else stays near B (1).
- **Routing**: a piece within 0.5 degrees is drawn straight, any other as an orthogonal leg and a
  diagonal leg or the reverse, whichever passes nearer its stations, and not leaving a node the way
  another piece does where the other choice avoids it; each station goes where it projects onto the
  drawing, in order, at least half an even share of the piece from the next.
- **The separation** (3 passes, each drawn again as above): the drawing in terms that keep it
  octilinear: each piece's ends free, each station a distance along its leg, a bent piece's legs
  meeting where their directions say. Of two straight pieces leaving a node the same way, all but the
  one with most stations get a first leg 45 degrees off toward where their stations lay after B. Linear
  terms: straight pieces stay straight (a station pushed across one moves both its ends alike); stations
  keep their order along a leg and gaps of half their hop's target or 1.25 times their half widths
  together, round a bend too; unconnected stations within 1.25 times their spacing are pushed to 1.15
  times it, and stations within their half width, the leg's and a pitch of another piece's leg are
  pushed off it. These are linearised about the drawing, 10 rounds a pass. A first layout meets them by
  sweeps of projections (each term not met moves its variables the least way); a rebuild by Hildreth's
  method from the last drawing, which finds the drawing nearest it, each term's correction at most 0.3
  of a hop, so an edit moves little. Drawn after it, a station keeps only its order (a tenth of an even
  share of its piece).
- **The spread**: stations still nearer than their spacing slide along their pieces, each the least
  that clears the pair, between its neighbours, only where it leaves fewer close pairs; failing that
  both of the pair part of the way. Key nodes stay, so no angle changes.
- **Order**: on each piece one stroke per colour, left to right by `lineBands.ts`' pairwise rule on
  the diagram's own places (follow two lines ahead to where they part, the one turning left goes left;
  else look back), with a fixed order where three or more go round in a circle.
- **A rebuild** starts from the last layout's B places by node key, skipping A, with 10 B rounds held
  to the last layout at 0.05 in place of A's 0.01: a new complex starts
  at its placed neighbours plus its real offset at the hop's scale, or failing any at its real place
  at the old layout's overall scale. Its separation keeps near the last drawing only when at least 90%
  of its nodes were in it; otherwise it separates as a first layout does.

`LAYOUT_R2` keeps round 2's options (alpha 0.35, pitch the median hop over 12, edges 1.6 half widths).
