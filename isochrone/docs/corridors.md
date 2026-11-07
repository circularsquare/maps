# Corridors

Where many trips from the start share a route (T-046). Code: `js/corridors.js`, called from
`js/session.js` once every level of the view is solved and painted, L10 first and then each window
coarse to fine (a window's count takes the coarser counts at its edge). A newer click or view stops
the counting between levels; the next view counts what is left (T-077).

## Counting

- **Trips.** Every reached land cell (time <= the limit) is one trip, weighted by its area in km²
  (`dxKm x dyKm` of its row), so high latitudes and small cells do not count extra. Sea and lake
  cells are not destinations; on a canoe start the trips still end on land, so rivers and coasts
  carry the counts of the land behind them.
- **The tree.** Each node's predecessor code (`docs/canoe.md` 5) gives its parent: a grid move,
  a transfer between the land and canoe layers, or a river step. A node's count is its own
  weight plus its children's: the area whose routes pass through it.
- **Near-equal land parents** (T-056). Over open land many routes are within a fraction of a
  percent of each other, and the search's single fastest parent sends a whole sector of trips
  down one run of a single move, peeling them off in parallel strands (spokes along the grid
  directions). So a land node reached by a land move (walked, or carried) may take another land
  parent: one move back in a direction next to the search's move (one place either side round
  the 16), reached and settled earlier. First every trip is spread over the near-equal routes:
  each such node passes its trips to these parents in proportion to (1 - slack / (3% of its
  time))², slack being the weight lost going through the parent, time plus K x hazard as the
  search costs the move (T-077): directed by each cell's E/W/S/N hours per km where the level has
  them (`docs/resolution.md` "Directions"), with the slope at L0.1 (`docs/fine.md` "Terrain"),
  carried at 2x on a canoe start (T-091), none past 3%. Then each takes the parent most spread trips pass
  through, among those that keep its route's weight through the tree within 1% of its time of
  the fastest; the search's parent always qualifies. A node with fewer spread trips than 2% of
  the drawing threshold passes them all to the search's parent and keeps it (most nodes; T-077).
  Other nodes keep the search's parent. The time map, the hover's route and the search are
  unchanged; only the counted tree is.
- **Passes.** Reached nodes are found by reading the predecessor codes four at a time (most are
  0), the start as the one parent with no code. In cell order each node's parent and its possible
  other parents are found; the trips are spread leaves first (each node queued once every node
  that may pass it trips is in, no sort by time), the shares worked out as a node is reached and
  only if it has trips enough; that order, backwards, puts every node after its parents for the
  choice; the counts then add up along it.
- **Threshold.** An edge is drawn when its child's count is at least 0.2% of the area reached at
  L10; the same threshold at every level. Strength is log(count / threshold) over
  log(L10 total / threshold) in 8 steps; width and opacity rise with it.
- **Speed** (T-062, T-064). An edge's speed is the length travelled over its time in the search
  (time only, no hazard; a re-chosen land parent's edge, the move's own time costed the same way,
  so uphill and downhill read their own speeds), summed over the edge and up to 6 edges of the same mode each way along
  the route (towards the start, and on along the child with the most trips). A river step
  travels the river's own length (meanders included, `rKm` of the canoe layer), so a reach reads
  its model speed, and the window evens out routes that hop between rivers sharing cells. A grid
  move travels its own length. Getting in or out to a neighbouring cell is carried over that step,
  its 15 minutes included; within one cell it travels nothing and takes the speed round it.
- **Lines.** Edges join into a polyline while the next one on is the only drawn edge and has the
  same step and mode, and its speed is within 0.4 km/h of the speed class the line is in (classes
  at 1.5, 2.5, 3.5, 4.5, 5.5, 7 and 9 km/h, `SPEED_EDGES`). A line's km/h is the time-weighted
  mean of its edges' speeds and its colour that km/h's class, bright red through orange to yellow
  (ramp, opacity range and a dark casing are `CORR_*` constants in `index.html`), so the
  number and the colour always agree. Width and opacity follow the strength. The page simplifies
  each line (Douglas-Peucker, 0.7 cell) and smooths it with three Chaikin passes, ends fixed. Its mode is walked, or on a canoe start carried (on land,
  and getting in or out) or paddled. The pointer within 4 px of a line shows the nearest line's
  mode and km/h. A line's byte is step | mode << 3 | class << 5. Points in the middle of a
  straight run are dropped (~400 lines at L10 walking, ~2000 on a canoe start).

## Windows (L3, L1, L0.1)

A window solve holds only the trips that end inside it, and its border ring is seeded from the
next coarser level. So:

- **Flux out.** Every coarser edge from inside the window to outside carries its coarse count
  into the window: added to the nearest window node at least 2 cells in from the edge (the ring
  itself is seeded or reached along the border), of the same layer first, not a seed.
- **Start inside the window.** Near the border, and far down a river, the seeds often win (a
  coarse time a little earlier than the fine one), so the window's tree there leads back to a
  border seed, not to the start. Only nodes whose route goes back to the start are drawn and take
  flux; where the border is seed-owned the flux follows the coarse route inward to the first
  point the start's own routes reach. Where the window's routes do not go back to the start, the
  coarser level's lines are drawn instead, cut in pieces a coarse cell long.
- **Start outside.** Every route comes in from a seed; flux goes to the nearest non-seed node.
- **Why one threshold.** A lower one in windows drew each coarse edge's flux across the window
  as its own straight line (parallel stripes in the Alps), since a window's paths come from seeds
  all along one side and do not converge.

The page keeps each level's lines and cuts a level's where the next finer level with lines shows.

**Where the lines meet at a window edge** (T-104). A window's own lines start at the node that took
the flux, up to a coarse cell from where the coarser line was cut, and its coarser-line pieces
(the worker sends `np`, the number of lines before the pieces, and `src`, each piece's line in the
coarser level) come from the raw route, which lies up to 0.7 coarse cell from the smoothed line.
So the page cuts each piece from the coarser level's smoothed line, between the points nearest the
piece's two ends (`carve`; the raw piece is kept if an end is over a coarse cell from that line),
and joins the rest after smoothing, in `joinCuts`:

- each piece end within one coarse cell of a cut end moves onto it;
- then each of the level's own line ends within the larger of 5 of its cells and 1.5 coarse cells
  moves onto the nearest cut end left or piece end, nearest pair first, one end per target;
- the move dies away along the line (cosine taper over 5 times the move, at least 3 cells, at most
  half the line), so the line stays smooth and the rest of it does not shift.
