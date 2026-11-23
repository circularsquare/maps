# Batch check 2, 2026-11-23 (agent batchcheck2)

Vite on 6150 (own), all against master at 1fa5d175. Scratch probes: C:/Users/anita/AppData/Local/Temp/claude/dce1fefcc1d43aef/28722621-86b3-4444-8131-f3aee02b18dd/scratchpad/batchcheck2/

## T-4104 (b3756860) PASS
Probes by edits on a NYC new game (sandbox) with defaults rail [600,480,900], fast [1200,900,1800], tram [300,240,420]:
- rail line on default headways, lineFast: rail -> fast headways; undo returns rail; redo fast; undo undo returns counts (null). Cash unchanged through the chain.
- line on counts (no headways): kind change leaves it on counts.
- one of three changed ([600,500,900]): kept through lineFast and its undo.
- chain: lineFast, rename, setRowHeadways, lineFast back, then 4 undos and 4 redos: each step the expected headways.
- no default for the new kind (only rail set): headways stay the old ones, kind changes, undo ok.
- 5-row line: default mapped to rows ([1200,1200,1200,900,1800]); undo/redo/back all fine.
- tram (never run before): a tram line (tram stops 10,11) setStops to rail stops [8,9] gave rail defaults [600,480,900]; undo gave tram [300,240,420]; redo rail; setStops back to tram gave tram defaults; undo/redo chains all matched, no page errors.
- old save (T-007-real-nyc.save): lines on counts untouched by kind change; with defaults set and the line's own set to rail default, lineFast -> fast defaults, undo x2 back to counts.
Not exercised in a game: bus and ferry lines (no bus line in the NYC real save or the busway fixture; code returns before any change when is_bus/is_ferry).

## T-4109 (a7e867de) PASS
- checks/t1007-panic.mjs: ALL PASS (B1 banner text "The train simulation stopped working. Reload the page to go on. Your last autosave is kept." + Reload page, B2 issues empty).
- own probe (down.mjs): a single panic gives no banner and the old restore toast; 3 more panics give exactly one banner; 5 edits and a preview refused quietly (ok false, no issues, answers arrive), banner count stays 1, toast unchanged; download works (12337 bytes) and loads in a fresh page with the same edges/nodes/km/cash as before the panics; a share export after the stop shows the stopped text as a toast (as designed).
- not checked: that autosave stays off (not instrumented).

## T-4130 (1fa5d175) PASS
- t1382-rows nyc 4x30 and 6x30: PASS (no junction crossing in NYC save, so the new case does not run there).
- t1382-rows asia 4x30: PASS, 4 of 4 seeds cleared a crossing from far away. --self-test and --self-test-touched both fail as they should.

- asia 10x40 run: seeds 1-8 clean, then FAIL seed 9 step 15 deleteEdge: "the regions differ from a fresh build (217 leaves against 217)". That is the T-4113 region shape (workers/regions.ts, not touched by any of the four commits; rows are not involved). Not rerun on the parent commit, so "pre-existing" is by reading, not by test. Worth a look (seed 9, 40 steps, asia).
- in game on the Asia save (t3600-asianita-1852.save): for 5 junctions with a crossing and two lines, removeEdge on the other line's farthest plain track (26-78 km away) turned the line not ok and cleared the junction's crossing in the page's rows; undo restored both, all 5.

## T-4111 (a841fb14) PASS
Asia save in game: quadTwin for all 8590 edges equals full quadPairs (285 twins), at load and after: doubleUp x3, parallel double routes 8 m beside an existing one (both sides), a quad route, removeEdge, 3 undos, 3 redos, undo to start (twin counts 285 -> 287 -> 288 -> 291 -> 288 and back to 285, zero differences at every step). constructAll was refused (RoadsMissing) so built pairs from doubleUp were not tested; the new parallel routes are unbuilt pairs.

## Smoke
node trainworld/checks/smoke.mjs against 6150, TW_CORES=4: SMOKE PASSED, 30 checks, 642 s.
