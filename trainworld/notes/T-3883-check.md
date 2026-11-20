# T-3883 check (rampbugchk, 2026-11-20): PASS, one claim in the builder's notes is wrong

Method: before = a copy of the tree's sim at HEAD (no fix) built on the agent profile; after = the tree. Native
`edit_bench` (TW_LINES, TW_DUMP) and a small `edge_dump` bin (edges_info vmin + edge_pis vertex labels) on seven
saves; the game on :6193 (tree) and :6194 (main tree, no fix) by pointer. Scratch: scratchpad/rampbugchk.

## 1 Real curves still limit: PASS
- Game, both trees, same pointer routes (clicked corners): hsr gentle 320 / 223.2 / 231.8 / 290.7 km/h, regular 160 /
  159.99, tram 80: every real-curve edge identical before and after. Preview speeds identical too.
- Saves, per edge (edges_info vmin), 7 saves: no edge's limit was lowered; 0 of 743 / 1,046 / 1,972 edges below 130 km/h
  changed (NA, asianita, bangkok); vertex labels of real curves (e.g. 3516/224, 2911/204) unchanged.
- Real NYC save (T-007-real-nyc), T-125-northeast, t1156-all-china, t1507-all-china-hsr: 0 edges and 0 lines changed
  (round trips, figures, lines_info all byte-identical).
- Not tested: a pathological hand-clicked polyline of vertices under 2 m apart (the notes' accepted lie).

## 2 Bug case: PASS
Slanted hsr at 320 on Long Island crossing a regular double track (ramp over track): preview 320 both trees. Built
piece Slowest: 160 on master, 320 with fix. Line over it (4/3/2 trains): average speed 200 -> 224 km/h, round trip
22m10s -> 20m25s, top speed 317 -> 320. Speed map mode: master shows the ramp stretch in a slower colour, fix one
purple throughout (artifacts/trainworld/t3883chk-before.png, t3883chk-after.png). Hudson bridge case (builder's
probe.js) also 320. Same build in probe 5: edges with ramps 0,1,5 went 160 -> 320. The Track inspector's corner
list was not opened in the game; its source (`edge_pis` via `vertex_kmh`) reads 1796/320 where the map was 1796/160.

## 3 Saves: PASS WITH A CORRECTION
Saved bytes are NOT always the same as master's. Four small saves: identical. Three big ones (NA day 491, asianita 557,
bangkok-tokyo-nyc 459): same length, 273 / ~190 / 70 bytes differ, all in the lines part and the stored-trip blocks
(LRT1..LSP1 and LPD1: stored round trips, train counts), none in nodes or edges. That is derived timing of lines
whose speed changed, so expected. Reload gives the new speeds (the fit is untouched; load recomputes).

## 4 Round trips on saves: PASS
| save | lines | changed | slower | over 5% |
|---|---|---|---|---|
| T-007-real-nyc | 109 | 0 | 0 | 0 |
| T-125-northeast | 110 | 0 | 0 | 0 |
| t1156 / t1507 all-china | 123 | 0 | 0 | 0 |
| north-america-day-491 | 175 | 29 | 7 | 8 |
| asianita-day-557 | 221 | 25 | 1 | 14 |
| bangkok-tokyo-nyc-els-459 | 131 | 5 | 4 (0.3 s or less) | 0 |
Over 5%: all high-speed lines, largest Floride -16.4%, Taiwan clone -13.4%, Hangzhou -13.3%, Shanghai Guangzhou
-12.6%. Edge level: 345 NA edges, 391 asianita, 7 bangkok changed vmin; every one rose (160 to 295..320 on 316 of
345 NA edges), none fell. 21 NA and 26 asianita changed edges have no level change (flat hsr edges, one vertex
labelled 1796/160 on a turn under 3e-4, e.g. asianita edges 1701, 1704 160 -> 320): phantom vertices off the
millimetre grid, not a bridge or ramp; flag for the builder, harmless. 34 regular-track edges moved 159.655 -> 160.0
(0.2%, same cause, cap 160).

## 5 Trams: PASS (limited)
Game: tram stays 80 in both trees. Saves: no edge under 130 km/h changed (the saves' trams and streetcars are among
them; the edge kind is not in edge_dump, so not counted separately).

## Skipped
Track inspector corner list by clicking; many-small-turn pathological polyline; Tokyo snapped route by hand (the
bangkok-tokyo save has 7 changed edges, 1 of them 160 -> 217, all vertices of ramps at levels -2/-1).
