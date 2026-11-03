# The network diagram

The design, with its references and stages: [notes/T-2146.md](../notes/T-2146.md). Built so far: the
graph (T-2159), the layout (T-2160) and a rough view; the real view, names, clicks, city boxes and
water are T-2161 to T-2163, T-2168 and T-2169.

## The button and key M

A button at the map's upper left, above the view rows (`ui/DiagramButton.tsx`), and key **M**
(`diagram/view.ts`) swap the map for the diagram, and back. Hover: "Show network diagram (M)", and while
the diagram is up "Show map (M)". The diagram is a 2D canvas over the map; the dock, top bar and the
map's upper left stay. Drag pans, the wheel zooms about the pointer. It opens centred on the complex
nearest the map's centre, a stroke pitch 7 px. `?diagram=r2` lays it out with round 2's options.

The rough view draws each line as one ribbon (casing in the background colour, strokes 5/6.4 of a
pitch wide), corners rounded with radius 6 stroke widths plus the bundle's half width, plus or minus
the stroke's offset, so a bundle's bends are concentric, and a white pill with the ink outline over
the stop points of a station within 1.6 pitches of each other, pills of one station joined by a
passage. No names, clicks or hovers yet.

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
  unconnected stations pushed to their half widths plus half the median hop apart. A stroke pitch is
  the median hop over 9; a node's half width is its widest bundle's; every edge is at least 1.4 times
  its ends' half widths.
- **Segments** between key nodes (a branch, a line's end, the lines changing), split at the station
  farthest off whenever its stations fit neither a straight run within 14 degrees nor one bend, within
  0.9 of the median hop.
- **The snap**: one solve on every node: a piece within 14 degrees of a 45 degree direction is made
  exact with its stations on it (weight 30), unconnected stations too close are pushed apart (2),
  everything else stays near B (1).
- **Routing**: a piece within 0.5 degrees is drawn straight, any other as an orthogonal leg and a
  diagonal leg or the reverse, whichever passes nearer its stations; each station goes where it
  projects onto the drawing, in order, at least half an even share of the piece from the next.
- **Order**: on each piece one stroke per colour, left to right by `lineBands.ts`' pairwise rule on
  the diagram's own places (follow two lines ahead to where they part, the one turning left goes left;
  else look back), with a fixed order where three or more go round in a circle.
- **A rebuild** starts from the last layout's B places by node key, skipping A, with 10 B rounds held
  to the last layout at 0.05 in place of A's 0.01: a new complex starts
  at its placed neighbours plus its real offset at the hop's scale, or failing any at its real place
  at the old layout's overall scale.

`LAYOUT_R2` keeps round 2's options (alpha 0.35, pitch the median hop over 12, edges 1.6 half widths).
