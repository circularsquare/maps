# Archive

- T-009 [P1] Own walking surface, stage A. Done 2026-10-26: MAP's recipe rebuilt at 1 km from LC100 2019 + GMTED (`tools/surface/`, layers in `data/surface/`); off-road median ratio to MAP 1.00, terrain alone within 2%, land cover flips give p10-p90 0.72-1.18. docs/surface.md, notes/T-009.md.
- T-019 [P2] Split worker.js into ES modules. Done 2026-10-26: `worker.js` is a thin module-worker entry over `js/` (session, levels, search, windows, tiles, paint, config); screenshots at default and zoom 8 pixel-identical, smoke passes, timings unchanged.
- T-013 [P2] Small water walkable. Done 2026-10-26: water patches under 5 km² (543k of 602k, 8-connected, merged across strip edges) are land at 1 km/h in grid and tiles; Balaton and Geneva stay water; L10 times rise 3-8%. notes/T-013.md.
- T-017 [P1] Smoke check. Done 2026-10-26: `checks/smoke.mjs`, headless Chrome on 8800, six PASS/FAIL lines (load, click, L3, L1, hover, no errors), ~20 s.

- T-002 [P1] View refine, stage 1. Done 2026-10-14: `tools/build_tiles.py` writes L3 and L1 tiles (45 s); the worker solves an L3 window from zoom 5 and L3 + L1 windows from zoom 7, seeded at the border from the coarser level, painted as canvas sources with holes in the coarser layer; windows of ~0.1-1M cells solve in 40-240 ms on a 1400 x 900 screen. notes/T-002.md.

- T-004 [P1] Research: our own walking surface from primary layers → research/friction.md. Done 2026-10-14: MAP's full recipe found (land cover km/h table, elevation and Tobler factors, roads 5 km/h, no border penalty in walking); plan is LC100 + GLO-90 with directed per-cell crossing sweeps + GRIP4 roads layer + ICE-6G/Beyer 2020 for 10,000 BC + CHELSA hazard, in stages A-F.
- T-003 [P1] Research: canoeing worldwide → research/canoe.md. Done 2026-10-14: HydroRIVERS + HydroLAKES + drifter currents + ARCO-ERA5 waves and ice, all open; speeds, portage, guessed hazard rates with checks; flags that a time-only search will take deadly sea shortcuts.
- T-001 [P1] Walking prototype (Ashley, 2026-10-13). Done 2026-10-13: MAP 2020 walking-only friction surface averaged to a 10 km global grid (`tools/build_grid.py`), Dijkstra in `worker.js` (~0.3 s), banded overlay on a dark MapLibre map, hover time, click to set the start. Checked headless from Cluj-Napoca and the Sahara.
