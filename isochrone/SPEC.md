# isochrone — spec

This file says what is true **now**. It is rewritten in place when a decision changes; the
history and the reasons live in `DECISIONS.md`. Where it says **open**, Anita decides.

---

## 1. What it is

A world map in the browser. You click a start point; every reachable place is coloured by how
long it takes to get there. You choose what you have to travel with and what exists in the
world: walking only, a canoe, a bike, a car; roads or no roads; today's land or the land of
10,000 BC. Hovering a destination says how long the trip takes and how likely you are to survive
it.

The motivating question (Anita): "if you just have a canoe in 10,000 BC, how far can you get?"

## 2. Now built

Walking only, on our own 1 km surface: the Malaria Atlas Project's recipe (Weiss et al. 2020)
rebuilt from Copernicus LC100 2019 land cover (`docs/surface.md`), its slopes from a 100 m walker
on Copernicus GLO-90 heights (T-011, section 3), with OSM roads and paths as a layer of their own
(`docs/roads.md`); the 3 km and ~10 km levels (4320 x 1740) are made by crossing the 1 km one.
One search from the clicked cell to everywhere (Dijkstra on a bucket queue, 16 neighbours, wraps
at 180 degrees, a knight move may not hop a water cell) takes 0.3-0.7 s in a worker with land
hazard. Bands from 1 hour to 128 days, a day being 8 hours of walking. Water is
crossed only on bridges. The default start is Cluj-Napoca.

A canoe start (the mode buttons, or `?mode=canoe`) searches carrying on land and the
canoe on sea, lakes and rivers with currents (`docs/canoe.md`), and paints the sea; ~0.8 s a
click at 10 km. The search minimises time + K x hazard (water and land, section 3), and keeps each node's move for tracing
the route to the pointer. Cells are drawn as hexagons by default (`?look=crisp` gives squares,
`?look=diag` cut corners); the look is paint only, the search runs on square cells.

Every level stores the off-road speed and, beside it, a road byte and four direction planes per
cell that the worker folds in as it loads the level, so turning roads off later needs no other
data. The land hazard band and the canoe water are cut to the same land and water, cell for cell
(T-084, T-085).

## 3. Travel model

- **Time and danger both count.** The search minimises time + K x hazard, one K for every mode,
  so a path pays extra hours to avoid risk (K = 1000 h: 10 hours to cut death risk by 1%). Hazard
  is a rate per cell and mode; survival along the chosen path is exp(-sum of hazard x time), shown
  in the hover. Water hazard is built (rivers, rapids, waves offshore, cold water, supplies far
  from land; `docs/canoe.md` 3), so a canoe hugs coasts where a crossing is risky and the open
  ocean is not crossed. Land hazard (thirst by distance to drinkable water, cold, ice, altitude;
  `docs/hazard.md`) is paid by walking, carrying and getting in or out, so walks keep to rivers
  and wells in deserts and go round Greenland's ice. A path stops where survival falls under 1%,
  on land as at sea, so the ice sheet and the emptiest deserts stay unpainted. The hover shows
  survival (rounded down) on either start once the path's risk reaches 0.5%: nearly every walk
  has some hazard, and "99%" on a walk through farmland would mislead.
- **Modes are layers of one graph.** Each mode (walking, canoe, later bike, car) is a layer of
  cells with its own speeds; switching between layers is an edge with a time cost. Turning a mode
  or a kind of infrastructure off removes its layer or its edges, so toggles do not multiply the
  data to prepare.
- **Directed edges.** Water flows: paddling downstream is faster than upstream, so the graph is
  directed. Walking is directed too (T-011): each cell has four speeds, going east, west, south and
  north, from a 100 m walker on Copernicus GLO-90 heights and LC100 crossing it (directed Tobler,
  research/friction.md 2), so downhill is faster than uphill and a path keeps to the valleys
  inside a cell. A move weighs the east-west and north-south speeds of the cells it touches by its
  share of each (`docs/resolution.md`, "Directions"); carrying is twice that. Hazard per km and
  getting in or out use the cell's one speed, the geometric mean of the four. The 100 m level
  walks each move on its grade (section 4).
- **Roads.** The 100 m walker walks a walkable OSM way at 5 km/h or its land cover's speed if
  faster, on its grade capped at 10% since roads wind up hills (T-084), and a 1 km cell's speeds
  with roads are its crossings on that grid (T-011). Where the walker crosses a cell in no
  direction (2.5% of 1 km land: past 80 N, small water, LC100's missing islands, cells walled by
  water at 100 m) the cell blends road and off-road time by three times its share of 100 m road
  cells, capped at all road (`docs/roads.md`). Walkable bridges carry walks over water; tunnels
  are left out.
- **Water.** A 1 km cell is water when LC100 has it half water or more; islands LC100 leaves out
  are put back from GMTED2010. Water patches under 5 km² are waded at 1 km/h (MAP's rate) instead
  of blocking (T-013). A coarse cell is land when a quarter of its 1 km cells are
  (`docs/resolution.md`).
- **Canoe**: a canoe start searches the canoe layer (sea, lakes, paddleable rivers from 1 m³/s,
  with currents) and carrying on land. The canoe is never left behind: overland it is carried, at
  twice the walking time (carry x2, Anita 2026-11-05). Paddling is 5 km/h on still
  water; downstream adds the river's current, upstream is 5 km/h minus 0.4 of the current with a
  floor of 2.5 km/h (lining or poling along the bank), and 2.5 km/h up rapids (1 m/km or
  steeper) on any river, where the canoe is poled, lined or waded. Design: `docs/canoe.md`; model and
  sources: research/canoe.md.
- **Eras.** Today, and later 10,000 BC (no roads, no cleared land, lower sea level, ice sheets,
  Beyer 2020 biomes at 12 ka for land cover). Sources: research/friction.md.
- **Season.** Annual averages for speeds on land, and the kinder of January and July at each place
  for land hazard; local summer at sea (waves and currents June-August north of
  the equator, December-February south; sea ice from the least-ice month, August or February),
  the season a paddler would choose. A season switch is a later extra.

## 4. Resolution

Coarse everywhere first, then finer wherever the viewer is looking: from zoom 5 a window round
the view is re-solved at 3 km cells, and from zoom 7 also at 1 km, each seeded at its edges by
the next coarser answer and drawn over it, fading in over 3 of its cells at the window's edge.
Hover reads the finest level shown. A coarse cell is
one node, so where water cuts its land in two (fjords, wide rivers) L3 and L10 join banks that L1
keeps apart and come out too fast (round Sogndal and Coari L1 takes 1.3-1.4 times as long;
T-007). Full design and the
thin-ridge question: `docs/resolution.md`; solve times: `notes/T-002.md`.

On walking starts, where 100 m tiles are built (towns, water country and mountains, T-057), a 100 m window is
solved once the view fits in 1.5 M cells (from about zoom 9.4): LC100 land cover, OSM roads and
bridges, and GLO-90 heights, each move walked by the rule of the 100 m walker that made L1
(directed Tobler on its grade, MAP's height factor, at most a 10% grade between two road or
urban cells, nothing steeper than 45 degrees; T-039), so a 100 m window and the 1 km level round
it agree (`docs/fine.md`). Which tiles get 100 m: T-047's rule, approved 2026-11-05
(`docs/fine.md`, "Where the fine level goes"); 2.15 GB built, nothing past 80 N.

## 5. Interface

The map is a globe (MapLibre 5.24.0) when zoomed out and turns flat (Web Mercator) between zoom
11 and 12, MapLibre's own switch-over; there is no projection switch. The time layers, corridors,
routes, river lines and hill shading draw on both. The worker is sent a view box clamped round the
centre, and a hover or click off the globe is ignored. House style (nycriders): bare title, legend
panel bottom right, `i` panel bottom left with the sources, and beside it the mode buttons
(walking, canoe; icons with hovers), which re-solve from the same start and keep `?mode=canoe` in
the URL. Click sets the start. Hover boxes appear at once and flip to stay on screen. Hover shows
the time to the point, on a canoe start how much was paddled and carried, survival (from 0.5%
risk), and the route from the start as a thin line, dashed where carried (`docs/canoe.md` 6). On
canoe starts water is painted 22% towards white, paler than land.

Top left, under the title, a collapsible speeds panel lists the model's speeds in km/h, each row
with a hover giving its basis: walking by ground type (LC100 classes at MAP's survey speeds), and on a canoe
start still water, small, big and huge rivers (1, 100, 10,000 m³/s) down and up, up rapids, a 1 m/s sea
current with and against, carrying and getting in or out. The worker sends them with `ready`
(`js/speeds.js`), worked out from the model constants. Two buttons in the row switch hill shading
(Terrain Tiles on AWS, under the basemap's water and the time colours) and river lines (the
canoe's own river steps, L10 world view below zoom 5, L3 tiles to 7, L1 above; wider for more
water, faint where unrunnable); both default on, kept for the browser session.

Corridors (default on, a button in the same row): lines over the colours, below the labels, where
many trips from the start share a route. Width and strength grow with the area the stretch leads
to, so trunks along rivers, valleys, roads and coasts branch and fade toward the tips. Colour is
the speed along the stretch, one ramp for every mode: bright red (under 1.5 km/h) through orange
to yellow (9 km/h and over). Lines are straightened and smoothed; hovering one shows whether it is
walked, carried or paddled, and its km/h. Counted in the worker from the path tree (near-equal
land parents re-chosen by the search's own move cost, directed, so there are no spokes), at L10
for the world and again in each window,
once every level of the view is painted (`docs/corridors.md`).
