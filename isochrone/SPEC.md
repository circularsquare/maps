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

Walking only, from the Malaria Atlas Project's 2020 walking-only friction surface (1 km, roads,
land cover and slope; Weiss et al. 2020, CC BY 4.0), averaged onto a global grid of ~10 km cells
(4320 x 1740). One search from the clicked cell to everywhere (Dijkstra, 16 neighbours, wraps at
180 degrees, a knight move may not hop a water cell) takes ~0.3 s in a worker. Bands from 1 hour
to 128 days, a day being 8 hours of walking. Water cannot be crossed. The default start is
Cluj-Napoca.

A canoe start (`?mode=canoe` until the mode panel, T-008) searches carrying on land and the
canoe on sea, lakes and rivers with currents (`docs/canoe.md`), and paints the sea; ~0.8 s a
click at 10 km. The search minimises time + K x hazard with every hazard 0 until T-006, and
keeps each node's move for tracing the route (T-024).

Our own off-road walking surface, MAP's recipe rebuilt at 1 km from Copernicus LC100 2019 land
cover and GMTED2010 elevation with each layer kept apart, is built but not yet read by the page
(`docs/surface.md`; T-028).

## 3. Travel model

- **Time and danger both count.** The search minimises time + K x hazard, one K for every mode,
  so a path pays extra hours to avoid risk (K = 1000 h: 10 hours to cut death risk by 1%). Hazard
  is a rate per cell and mode; survival along the chosen path is exp(-sum of hazard x time), shown
  in the hover. Until hazard data exists (T-006) the search is time only.
- **Modes are layers of one graph.** Each mode (walking, canoe, later bike, car) is a layer of
  cells with its own speeds; switching between layers is an edge with a time cost. Turning a mode
  or a kind of infrastructure off removes its layer or its edges, so toggles do not multiply the
  data to prepare.
- **Directed edges.** Water flows: paddling downstream is faster than upstream, so the graph is
  directed. Walking is symmetric while speeds come from the MAP surface; once slopes come from
  our own elevation (T-011) walking is directed too, downhill faster than uphill (directed Tobler,
  research/friction.md 2).
- **Small water.** Water patches under 5 km² at L1 are walked at MAP's 1 km/h instead of blocking
  (T-013).
- **Canoe** is the next mode (Anita, 2026-10-14): a canoe start searches the canoe layer (sea,
  lakes, paddleable rivers from 1 m³/s, with currents) and carrying on land. The canoe is never
  left behind: overland it is carried, at twice the walking time. Design: `docs/canoe.md`
  (for Anita's review, T-005); model and sources: research/canoe.md.
- **Eras.** Today, and later 10,000 BC (no roads, no cleared land, lower sea level, ice sheets,
  Beyer 2020 biomes at 12 ka for land cover). Sources: research/friction.md.
- **Season.** Annual averages on land; local summer at sea (waves and currents June-August north of
  the equator, December-February south; sea ice from the least-ice month, August or February),
  the season a paddler would choose. A season switch is a later extra.

## 4. Resolution

Coarse everywhere first, then finer wherever the viewer is looking: from zoom 5 a window round
the view is re-solved at 3 km cells, and from zoom 7 also at 1 km, each seeded at its edges by
the next coarser answer and drawn over it. Hover reads the finest level shown. Full design,
measured times and the thin-ridge question: `docs/resolution.md`.

## 5. Interface

House style (nycriders): bare title, legend panel bottom right, `i` panel bottom left with the
sources. Hover shows the time to the point. Click sets the start.
