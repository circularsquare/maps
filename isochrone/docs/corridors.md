# Corridors

Where many trips from the start share a route (T-046). Code: `js/corridors.js`, called from
`js/session.js` after each level is posted, so it never holds up the paint.

## Counting

- **Trips.** Every reached land cell (time <= the limit) is one trip, weighted by its area in km²
  (`dxKm x dyKm` of its row), so high latitudes and small cells do not count extra. Sea and lake
  cells are not destinations; on a canoe start the trips still end on land, so rivers and coasts
  carry the counts of the land behind them.
- **The tree.** Each node's predecessor code (`docs/canoe.md` 5) gives its parent: a grid move,
  a transfer between the land and canoe layers, or a river step. A node's count is its own
  weight plus its children's: the area whose routes pass through it.
- **One pass.** Reached nodes are found by reading the predecessor codes four at a time (most are
  0), the start as the one parent with no code. Children are counted per node, then leaves are
  added to their parents, each parent queued once all its children are in (no sort by time).
- **Lines.** An edge is drawn when its child's count is at least 0.2% of the area reached at
  L10; the same threshold at every level. Strength is log(count / threshold) over
  log(L10 total / threshold) in 8 steps; width and opacity rise with it (0.12 to 0.6 opacity).
  Edges join into a polyline while the next one on is the only drawn edge and has the same step,
  and points in the middle of a straight run are dropped (~400 lines at L10).

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
