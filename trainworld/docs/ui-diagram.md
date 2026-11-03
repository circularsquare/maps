# The network diagram

The design, with its references and stages: [notes/T-2146.md](../notes/T-2146.md). Built so far: the
graph (T-2159), the layout (T-2160) and the view and its drawing (T-2161); names, clicks, city boxes
and water are T-2162, T-2163, T-2168 and T-2169.

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

**Pills.** Each station's pill lies along an axis through its place: of the four octilinear axes, the
one every line stopping there crosses, and of those the one giving the shortest pill. A line's stop is
where its drawn ribbon crosses the axis; a line ending there has its last run carried on, or cut back,
to the axis. So a pill is always a straight capsule on its lines, white with the ink outline in both
themes, two pitches across. Stops along the axis more than 1.6 pitches apart make separate pills; a
line running along the axis more than half a pitch off it gets a round stop on its own ribbon. Pills
of one station are joined by a passage, 0.45 of a pill wide. Below a pitch of 2.5 device px, no pills.

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
- **The spread**: stations still nearer than their spacing slide along their pieces, each the least
  that clears the pair, between its neighbours, only where it leaves fewer close pairs; failing that
  both of the pair part of the way. Key nodes stay, so no angle changes.
- **Order**: on each piece one stroke per colour, left to right by `lineBands.ts`' pairwise rule on
  the diagram's own places (follow two lines ahead to where they part, the one turning left goes left;
  else look back), with a fixed order where three or more go round in a circle.
- **A rebuild** starts from the last layout's B places by node key, skipping A, with 10 B rounds held
  to the last layout at 0.05 in place of A's 0.01: a new complex starts
  at its placed neighbours plus its real offset at the hop's scale, or failing any at its real place
  at the old layout's overall scale.

`LAYOUT_R2` keeps round 2's options (alpha 0.35, pitch the median hop over 12, edges 1.6 half widths).
