# Tasks

Open agent work. Format and rules in `CLAUDE.md`. P1 = now or blocking, P2 = this milestone,
P3 = later. Finished or abandoned tasks move to `ARCHIVE.md`.

**Where things stand (2026-10-26).** Walking works worldwide at ~10 km (T-001), refined to 3 km
and 1 km round the view when zoomed in (T-002). Research for the canoe (T-003) and for our own
walking surface and 10,000 BC (T-004) is done. Now: the canoe design (T-005) and our own
walking surface (T-009), which 10,000 BC needs. The smoke check (T-017) and small water (T-013) are in.

Milestones answer Anita's question, "if you just have a canoe in 10,000 BC, how far can you
get?", in order: the canoe in today's world, survival, then 10,000 BC.

## In progress

- T-005 [P1] Canoe, stage 1: the design, from research/canoe.md, for Anita's review → docs/canoe.md. Design ready for Anita's review, docs/canoe.md; notes/T-005.md.
- T-009 [P1] Own walking surface, stage A: reproduce MAP at 1 km from LC100 land cover with MAP's speeds and formulas (research/friction.md 1, 3, 7), checked against MAP's surface. ~4 GB download. Taking, 2026-10-26 (agent).

## M1: canoe, today

- T-019 [P2] Split worker.js into ES modules (levels and grids, search, windows, paint) ahead of a graph with several layers; no behaviour change, smoke passes. After T-017.
- T-020 [P2] Canoe data, rivers: RiverATLAS to river step records at L1, L3 and L10 (Q >= 1 m³/s, v from Q, gradient classes), OSM falls, dams and weirs, HydroLAKES reservoirs (docs/canoe.md 4). After T-005's review.
- T-021 [P2] Canoe data, lakes and sea: water tiles (kind, shore, links) at all levels and the 0.25° sea grid from drifter currents and ERA5 summer waves, ice and sea temperature (docs/canoe.md 4). After T-005's review.
- T-022 [P2] Search with layers: canoe + carrying nodes, transfers, directed river and sea edges, weight = time + K x hazard (hazard 0 until T-006), seeding by layer, the sea painted; measure speed (docs/canoe.md 5, 7). After T-005's review, T-019; synthetic data until T-020/T-021.
- T-024 [P2] Canoe hover: time paddled and carried, the route line traced across levels (docs/canoe.md 6). After T-022.
- T-008 [P2] Mode panel: walking or canoe start, later K, the other toggles and the era switch. After T-022.
- T-023 [P2] M1 doc tidy: spec against code, prune TASKS, fold notes. Last in M1.

## M2: survival

- T-006 [P2] Danger and survival: per-cell hazard for walking and canoe (research/friction.md 6, canoe.md 5), the search's cost becomes time + K x hazard (SPEC 3), survival along the path in the hover. Climate from CHELSA (stage E). After T-022.

## M3: 10,000 BC

- T-012 [P2] 10,000 BC land: ICE-6G sea level and ice, glacial lakes (drawn outlines first), Beyer 2020 biomes (research/friction.md 4, stage D). After T-009.
- T-010 [P2] Roads as their own layer from GRIP4 (research/friction.md 5, stage B), so "no roads" is a toggle and 10,000 BC has none. After T-009.

## Later

- T-011 [P3] 90 m terrain: stream Copernicus GLO-90 tile by tile, directed crossing costs per cell (research/friction.md 7, stage C); makes walking directed. 100-130 GB streamed, run detached overnight. After T-009.
- T-007 [P3] Coarse solve with portals (docs/resolution.md, thin barriers). After T-011.
- T-014 [P3] Window edges: fade the finer window into the coarser one over a few cells instead of a hard hole edge, to remove the faint 1-3 px line (docs/resolution.md).
- T-015 [P3] Big screens: L1 windows of ~2M cells take ~1-1.5 s in headless tests. Try painting the inner view first, or a smaller L1 margin above 2560 px wide; measure in a normal browser first.
- T-016 [P3] Publishing: data/tiles is 135 MB (L1 117.5 MB); goes to R2 with the page. Try a tighter encoding (row deltas, brotli) before publishing.

## Someday

- Bike and car modes on the roads layer. In-between eras (1 AD, 1500) with HYDE. A season switch. OSM tracks and paths over passes. A heavy dugout and a big crewed canoe as canoe variants. Paleo rivers on exposed shelves. (No ids yet; filed when picked up.)
