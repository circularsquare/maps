# T-3578 / T-3581 check (farebasecheck, 2026-11-17)

Verdict: FAIL on one point (negative per-line revenue with every line at 0x). Everything else passed.

Passed
- cargo test --release: 606 passed; cargo test --release --features demand --lib: 799 passed (CARGO_BUILD_JOBS=3, WASM rebuilt first).
- checks/t3578-line-base.mjs and t3567-line-fare.mjs: all passed (line 14: 1.00x $1.491/ride; 3x $3.800, 239,189 riders; 0x $0.372, 1,292,295 riders; two-line 1.5x+2x pays 2.0000 bases).
- Master comparison: base build (worktree at d30ffc11, WASM built separately, Vite 6175) vs the tree (6174), real save, all 1.00: dump of rail trips and every line's riders and revenue has the same SHA-256 (504fd8aa...). Note it is the tree's parent commit, not current master.
- Line 0 (top rail line) at 1.2x / 3x / 0x: riders 1,084,617 / 254,203 / 1,519,129; revenue $95.02M / $39.22M / $11.05M; $/ride 0.876 / 1.543 / 0.073; all finite; sum of line revenues equals the income total at each.
- Delete line 0 at 1.5x: fares.lines null; undo: {"0":1.5}; redo: line gone, no multiple; undo again: {"0":1.5}. Export and reopen keep {"0":1.5} and identical figures.
- After a delete at 1.5x, a new line built at the freed id 0 has no multiple (starts at 1.00x).
- No page errors in any run. Finances tab text at all-0x: no NaN/Infinity/undefined.

Not exercised: the real save has only rail lines (109), so bus, ferry and high-speed multiples were not tried through the app (the Rust tests cover them).

FAIL: every line at 0x
Repro: open T-007-real-nyc.save, call world.setLineFare(num, 0) for all 109 lines, wait for demand.
- New build: 33 lines have negative revenue per day, for example line 1 -$17.75M, line 2 -$5.83M, line 4 -$14.95M, line 3 -$13.98M. Total income is +$38.6M a day (a little off 0), rail trips 15.06M.
- Base build (T-3567 rule), same steps: 0 negative lines, min $0.013M, max $81.5M, total $2,060M.
- The explain panel's leg fares at all-0x go as low as -6.6 (perceived units); the legs of a route share the negative base term by km, so a single leg can be below 0.
Cause (from the notes' design, not traced in code): the base pool is shared between lines by boardings, but the negative extra base (M-1 = -1 per trip) goes to lines by km, so a long-haul line takes more negative than its pool share.
A single line at 0x (line 14, line 0) stays positive.
