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

## 3. Travel model

- **Fastest path, survival reported.** The search minimises time only. Survival is worked out
  along the fastest path afterwards, from a hazard rate per cell and mode: survival =
  exp(-sum of hazard x time). (Anita, 2026-10-14; a search that trades time against risk is a
  possible later extra, not planned.)
- **Modes are layers of one graph.** Each mode (walking, canoe, later bike, car) is a layer of
  cells with its own speeds; switching between layers is an edge with a time cost. Turning a mode
  or a kind of infrastructure off removes its layer or its edges, so toggles do not multiply the
  data to prepare.
- **Directed edges.** Water flows: paddling downstream is faster than upstream, so the graph is
  directed. Walking stays symmetric.
- **Canoe** is the next mode (Anita, 2026-10-14). Model: research T-003.
  **open**: whether the canoe can be left behind and picked up, or carried overland (portage).
  Proposed default: you carry it overland at a slower walking pace, never leave it.
- **Eras.** Today, and later 10,000 BC (no roads, no cleared land, lower sea level, ice sheets).
  Sources: research T-004.

## 4. Resolution

Coarse everywhere first, then finer wherever the viewer is looking: from zoom 5 a window round
the view is re-solved at 3 km cells, and from zoom 7 also at 1 km, each seeded at its edges by
the next coarser answer and drawn over it. Hover reads the finest level shown. Full design,
measured times and the thin-ridge question: `docs/resolution.md`.

## 5. Interface

House style (nycriders): bare title, legend panel bottom right, `i` panel bottom left with the
sources. Hover shows the time to the point. Click sets the start.
