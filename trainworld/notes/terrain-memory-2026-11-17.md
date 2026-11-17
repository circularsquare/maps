# Terrain shading's memory (2026-11-17, agent terrainmem)

Terrain (`display.hillshade`) on vs off, master 85488bdd, dev server (`dev-nowatch.mjs`, so every figure carries the
dev build's extra weight; on/off differences are comparable, the absolute totals are not production's). Method as
t2802-memory.mjs: Chrome private bytes of the headless browser's processes (all / renderer / GPU), forced collection
before the stage figures; 3 pinned cores (TW_CORES=3), 2 demand workers (phone: 1, as T-3497). Setting is seeded into
localStorage `trainworld-ui` (`{v:15, display:{hillshade}}`) and the layer's visibility read back (visible / none).
Two runs per cell, interleaved on/off, machine busy. MB, total (renderer / GPU).

Pan: 36 jumps at 3.5 s (about 2 min) at fixed pseudo-random offsets and zooms, identical on and off.
Real save: 1600x900, T-007-real-nyc.save after demand final, Manhattan z12-14. Fresh: `?new=1`, flown to Innsbruck
(11.40, 47.27), z10-13; desktop 1600x900, phone Pixel 7 emulation 390x844 at DPR 3.

| Case | Stage | Off (run 1 / run 2) | On (run 1 / run 2) | On minus off (mean) |
|---|---|---|---|---|
| Real save, desktop | start | 2716 / 2650 | 2672 / 2782 | +44 |
| | after pan | 2933 (2284/562) / 2950 (2290/572) | 2984 (2310/586) / 2996 (2324/584) | +49 |
| | peak in pan | 3230 / 3212 | 3148 / 3098 | -98 (peak is noisy) |
| Fresh Innsbruck, desktop | start | 1427 / 1476 | 1567 / 1560 | +112 |
| | after pan | 1395 (931/376) / 1366 (919/359) | 1642 (1053/500) / 1629 (1059/482) | +255 (renderer +120, GPU +125) |
| | peak in pan | 1532 / 1545 | 1780 / 1791 | +247 |
| Fresh Innsbruck, phone | start | 1065 / 1070 | 1155 / 1159 | +90 |
| | after pan | 985 (713/185) / 978 (707/184) | 1151 (822/241) / 1152 (809/255) | +170 (renderer +105, GPU +65) |
| | peak in pan | 1096 / 1104 | 1272 / 1249 | +160 |

Run-to-run spread within a cell: 10-130 MB at start and peak, under 30 MB after the pan (real-save peaks 3098-3230).

## Terrain tile requests

Host: `s3.amazonaws.com` (path `elevation-tiles-prod/terrarium/{z}/{x}/{y}.png`, Terrain Tiles on AWS Open Data).
Counted during the pan only (page load not counted):

| Case | On | Off |
|---|---|---|
| Fresh Innsbruck, desktop, 2 min pan | 114 (both runs) | 0 |
| Fresh Innsbruck, phone, 2 min pan | 62 (both runs) | 0 |
| Real save, Manhattan z12-14 | 0 | 0 (the few tiles there are loaded before the pan; DEM maxzoom 12) |

## Reading

- Terrain on adds about 250 MB on desktop in a hilly place and about 170 MB at phone size; in dense, flat Manhattan
  at z12-14 the difference is within noise (about +50).
- The extra is roughly half renderer (DEM tile bitmaps and decoded textures) and half GPU, and it grows with the
  number of distinct DEM tiles visited, so panning over hills costs more than sitting still.
- Whether the "about 500 MB less" headline still holds depends on how it was measured; with terrain on, the
  saving would be smaller by the above (at most about half of it on desktop in hills, none to little over a flat city).
- Not done: production build (dev server weight), a phone run on the real save, a longer pan (more tiles cached; the
  cache is capped, so growth should level off).
