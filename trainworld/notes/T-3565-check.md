# T-3565 check (onewaycheck, 2026-11-17)

Verdict: FAIL on three or more leg stops per leg clicked in natural order; everything else passed.

Passed: cargo test --release (610 ok), --features demand --lib (801 ok), wasm rebuilt; tree's t3565-oneway-stops and t3537-tram-split PASSED;
unequal legs (0.7 km / 0.12 km), 3,640 / 3,220 m out/back, no turn back; two lines over the same pair identical (1054 s);
300 m edge: leg gap 249 m with only one listed = partner rule (served both ways, oneWay []), gap 343 m = leg stop (oneWay [it]); both listed fine at both;
leg stop as hand-ordered end: runs, same loop as before (1637 / 5122 m; old build same shape); leg station first by hand: fine (2078/2078);
save/reload of the unequal and multi fixtures loads; undo of dropStop restores the line exactly (twice); real NYC save: all 109 lines identical
(stops, status, reversals, round trip, lengths) between this tree and the 8800 snapshot build; Natural riders board one direction only;
buses have empty stops/path so legs() is empty (code reading, not run live).

FAIL repro (scratchpad onewaycheck\scen.mjs, STEP=1 node scen.mjs multi, fixture yN=1.2 yS=-1.2 km, N leg stops at x -0.2,0,0.2, S leg at -0.1,0.1,0.3; no partner within 300 m):
click AR, LO, then the six leg stations by x, then E12, UN. After click 3 the first leg station is the line's end (loop). Click 4 (a station on the other leg)
is put FIRST in the line; at the end: Guzmán Blanco > AR > UN > E12 > Washington > leg stops... with 3 reversals, out/back 9931 / 20446 m (old build 46 km back).
Saved and reloaded: unchanged (straighten_legs does not fix it). Clicking AR, LO, E12, UN first and then the leg stations gives a clean line (3549/3549 m).
One leg station per leg (mini fixture) is clean in either order.
Not tested cleanly: closed ring line over the pair (my orderStops harness produced a doubled stop list), riders on NYC (stats identical so none changed).
