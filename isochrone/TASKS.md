# Tasks

Open agent work. Format and rules in `CLAUDE.md`. P1 = now or blocking, P2 = this milestone,
P3 = later. Finished or abandoned tasks move to `ARCHIVE.md`.

**Where things stand (2026-10-26).** Walking works worldwide at ~10 km (T-001), refined to 3 km
and 1 km round the view when zoomed in (T-002). Research for the canoe (T-003) and for our own
walking surface and 10,000 BC (T-004) is done, and our own walking surface is built (T-009),
not yet on the page (T-028). Now: the canoe design (T-005). The smoke check (T-017) and small water (T-013) are in.

Milestones answer Anita's question, "if you just have a canoe in 10,000 BC, how far can you
get?", in order: the canoe in today's world, survival, then 10,000 BC.

## In progress

- T-022 [P1] Search with layers: canoe + carrying nodes, transfers, directed river and sea edges, weight = time + K x hazard (hazard 0 until T-006), seeding by layer, the sea painted; measure speed (docs/canoe.md 5, 7). Synthetic data until T-020/T-021. Taking, 2026-10-26 (agent).
- T-030 [P1] Finer than 1 km when zoomed in: design a 100 m level (LC100 land cover, roads, GLO-90 terrain) for windows from zoom ~10, with sizes, times and what it means for publishing; for Anita's review. Design ready for Anita's review, docs/fine.md (notes/T-030.md).
- T-037 [P3] Window edge strip: with nearest resampling, the coarser canvas shows as a brighter strip (up to one coarser cell, ~5 px at zoom 9) along a finer window's north and south edges, since the hole is rounded inwards to whole canvas rows. Off screen once the view settles; fix by aligning hole rows to the finer window or painting the coarse edge row clear.

## M1: canoe, today

- T-024 [P2] Canoe hover: time paddled and carried, the route line traced across levels (docs/canoe.md 6). After T-022.
- T-008 [P2] Mode panel: walking or canoe start, later K, the other toggles and the era switch. After T-022.
- T-031 [P2] River gradient south of 60N: `sgr_dk_rav` closes 24% of paddleable length (mostly Q < 10 streams over 5 m/km) and reads lowland Amazon streams as rapids (Manaus to the sea sums to 140 m, real ~20). Once canoe routes show on the page (T-022), compare a few basins with the GMTED drop-over-10 km used north of 60N (`rivers.py north_gradient`) and pick per region by eye; the Livingstone Falls stretch between OSM rapids points still runs (notes/T-020.md).
- T-023 [P2] M1 doc tidy: spec against code, prune TASKS, fold notes. Last in M1.

## M2: survival

- T-006 [P2] Danger and survival: per-cell hazard for walking and canoe (research/friction.md 6, canoe.md 5), the search's cost becomes time + K x hazard (SPEC 3), survival along the path in the hover. Climate from CHELSA (stage E). After T-022.

## M3: 10,000 BC

- T-012 [P2] 10,000 BC land: ICE-6G sea level and ice, glacial lakes (drawn outlines first), Beyer 2020 biomes (research/friction.md 4, stage D). After T-009.
- T-010 [P2] Roads as their own layer from GRIP4 (research/friction.md 5, stage B), so "no roads" is a toggle and 10,000 BC has none. After T-009.
- T-028 [P2] Feed the page from our own surface: build_grid.py and build_tiles.py read `data/surface/walk_today.tif` (uint16 km/h x 1000, 0 = water) plus the roads layer (T-010) instead of MAP's tif; decide the water rule from `lc_water.tif` shares with T-013's small-water rule; smoke passes. After T-010 (docs/surface.md).

## Later

- T-038 [P2] Fine level stage 1: 100 m tiles for New York (40-42 N, 75-73 W) and Cluj (46-48 N, 23-25 E) from LC100 classes, OSM roads and bridges (Geofabrik extracts) and the L1 terrain factor; the L0.1 window in the worker, seeded from L1, painted; smoke passes (docs/fine.md). After Anita's review of T-030.
- T-039 [P3] Fine level stage 2: GLO-90 elevation in the 100 m tiles, directed slope per move with the street grade cap, a bucket queue in the search for every level, times in a real browser (docs/fine.md). After T-038.
- T-011 [P3] 90 m terrain: stream Copernicus GLO-90 tile by tile, directed crossing costs per cell (research/friction.md 7, stage C); makes walking directed. 100-130 GB streamed, run detached overnight. After T-009.
- T-007 [P3] Coarse solve with portals (docs/resolution.md, thin barriers). After T-011.
- T-014 [P3] Window edges: fade the finer window into the coarser one over a few cells instead of a hard hole edge, to remove the faint 1-3 px line (docs/resolution.md).
- T-015 [P3] Big screens: L1 windows of ~2M cells take ~1-1.5 s in headless tests. Try painting the inner view first, or a smaller L1 margin above 2560 px wide; measure in a normal browser first.
- T-016 [P3] Publishing: data/tiles is 135 MB (L1 117.5 MB); goes to R2 with the page. Try a tighter encoding (row deltas, brotli) before publishing.

## Someday

- Bike and car modes on the roads layer. In-between eras (1 AD, 1500) with HYDE. A season switch. OSM tracks and paths over passes. A heavy dugout and a big crewed canoe as canoe variants. Paleo rivers on exposed shelves. (No ids yet; filed when picked up.)
