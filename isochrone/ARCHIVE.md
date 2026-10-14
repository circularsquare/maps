# Archive

- T-002 [P1] View refine, stage 1. Done 2026-10-14: `tools/build_tiles.py` writes L3 and L1 tiles (45 s); the worker solves an L3 window from zoom 5 and L3 + L1 windows from zoom 7, seeded at the border from the coarser level, painted as canvas sources with holes in the coarser layer; windows of ~0.1-1M cells solve in 40-240 ms on a 1400 x 900 screen. notes/T-002.md.

- T-004 [P1] Research: our own walking surface from primary layers → research/friction.md. Done 2026-10-14: MAP's full recipe found (land cover km/h table, elevation and Tobler factors, roads 5 km/h, no border penalty in walking); plan is LC100 + GLO-90 with directed per-cell crossing sweeps + GRIP4 roads layer + ICE-6G/Beyer 2020 for 10,000 BC + CHELSA hazard, in stages A-F.
- T-003 [P1] Research: canoeing worldwide → research/canoe.md. Done 2026-10-14: HydroRIVERS + HydroLAKES + drifter currents + ARCO-ERA5 waves and ice, all open; speeds, portage, guessed hazard rates with checks; flags that a time-only search will take deadly sea shortcuts.
- T-001 [P1] Walking prototype (Ashley, 2026-10-13). Done 2026-10-13: MAP 2020 walking-only friction surface averaged to a 10 km global grid (`tools/build_grid.py`), Dijkstra in `worker.js` (~0.3 s), banded overlay on a dark MapLibre map, hover time, click to set the start. Checked headless from Cluj-Napoca and the Sahara.
