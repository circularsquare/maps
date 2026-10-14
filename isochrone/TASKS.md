# Tasks

Open agent work. Format and rules in `CLAUDE.md`. P1 = now or blocking, P2 = this milestone,
P3 = later. Finished or abandoned tasks move to `ARCHIVE.md`.

**Where things stand (2026-10-14).** Walking works worldwide at ~10 km (T-001), refined to 3 km
and 1 km round the view when zoomed in (T-002). Next: canoeing (T-005).

## In progress

## Open

- T-005 [P2] Canoe layer, stage 1: the design, from T-003, for Anita's review. After T-003.
- T-006 [P2] Survival: per-cell hazard for walking, survival along the fastest path, shown in the hover. After T-004.
- T-007 [P3] Coarse solve with portals (docs/resolution.md, thin barriers). After T-002 and T-005.
- T-008 [P3] Toggle panel: modes and infrastructure on and off, era switch. After T-005.
- T-009 [P2] Own walking surface, stage A: reproduce MAP at 1 km from LC100 land cover with MAP's speeds and formulas (research/friction.md), checked against MAP's surface. ~4 GB download.
- T-010 [P2] Roads as their own layer from GRIP4 (research/friction.md stage B), so "no roads" is a toggle. After T-009.
- T-011 [P2] 90 m terrain: stream Copernicus GLO-90 tile by tile, keep directed crossing costs per coarse cell (research/friction.md stage C; replaces averaging, overlaps T-007). 100-130 GB streamed; who runs it is open.
- T-012 [P3] 10,000 BC land: ICE-6G sea level and ice, glacial lakes, Beyer 2020 biomes (research/friction.md stage D). After T-009.
- T-013 [P2] Small water at L1: carry out Anita's call on one-pixel ponds (TODO.md), e.g. water patches under N km² walkable at MAP's 1 km/h, in build_tiles.py and build_grid.py alike. After her answer. notes/T-002.md.
- T-014 [P3] Window edges: fade the finer window into the coarser one over a few cells instead of a hard hole edge, to remove the faint 1-3 px line (docs/resolution.md).
- T-015 [P3] Big screens: L1 windows of ~2M cells take ~1-1.5 s in headless tests. Try painting the inner view first, or a smaller L1 margin above 2560 px wide; measure in a normal browser first.
- T-016 [P3] Publishing: data/tiles is 135 MB (L1 117.5 MB); goes to R2 with the page. Try a tighter encoding (row deltas, brotli) before publishing.
