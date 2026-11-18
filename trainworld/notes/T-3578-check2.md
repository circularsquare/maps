# T-3578 check 2 (farecheck2, 2026-11-17)

Verdict: PASS, with one oddity to look at (below). Tree .tw-trees/farebase (Vite 6192), main tree on 6193.

Passed
- 1.00x everywhere: dump of rail trips and every line's riders and revenue (line 14 run, t3578-line-base.mjs --dump) has the same SHA-256 on the tree and on the main tree's Vite: 504fd8aa...
- The tree's own t3578-line-base.mjs: all passed (two-line 1.5x+2x pays 2.00 bases; line 14 3x $3.90/ride, 0x $0/ride, no period below 0; delete at 1.5x and undo gives 1.5x; all lines 0x: 0 negative, income $0.00M against $3880M; 0/3/0.5 mix: 0 negative).
- Own script (fc2.mjs), real save: all lines 0x: 0 negative lines (day or period), all finite, total $0.0000M, every line |rev| < $0.00001M; Money tab and a line's Money panel text have no NaN/Infinity/undefined. This was the earlier FAIL.
- Mixed 0/0.5/1/2/3 across the lines: 0 negative, sum of line revenues equals income total ($3371.93M), the 22 lines at 0x earn $0.00000M; $/ride rises with the multiple (0x 0, 0.5x 1.15, 1x 1.94, 2x 3.67, 3x 5.21).
- Bus and ferry built through the app (addBusLine x2, addFerryLine Whitehall-St George) on the real save: bus 2x, ferry 3x, bus+ferry 0x, all (rail+bus+ferry) 0x, bus 1.5/ferry 2/rail mixed: no negative, finite, sums to total, 0x bus/ferry earn nothing.
- Bus-to-rail transfer trip (bus 109 then rail 16 then 8): bus 1.25x with rail at 1x gives route baseMul 1.25.
- Delete line at 2x (other line 0.5x), undo: both multiples back; redo: line gone, other kept; undo again: back. Totals consistent afterwards.
- New bus line at the freed id 109 after deleting it at 2.5x: no multiple, panel reads 1.00x.
- hover lineMoney.fareTip reads as asked.

Not exercised: high-speed (no HSR line in the real save, none built), rail-ferry transfer baseMul, trip panel at mixed multiples (the panel I opened said the places were outside homes and jobs, so only that text was scanned), "income equals what demand charged" beyond line sum = total.
My D "undo" first try pressed Ctrl+Z after a plain setLineFare: that is not an undo step (same on master), so ignore it.

Oddity (not failed on)
- Trip 189 to 303, route 109+16+8 (bus then two rail lines), only bus 109 changed, others 1x: route share by bus multiple, tree: 1x .033, 0.9x .031, 0.75x .036, 0.5x gone from the top 8, 0.25x gone, 0x .039. Master is smooth (.033, .033, .033, .037, .041, .043). The route's baseMul is 1 throughout, so its cost should match master's. Line-level riders on 109 are monotonic in the tree (23126, 23677 at 0.9x, 24612 at 0.75x) so it looks like the trip panel's route list for one pair, not income. Worth a look by whoever owns trip.rs / the strategy's mean-of-branches base term.
- Cosmetic: app/src/game/money.ts has a literal backtick-r-backtick-n inside the baseTrips doc comment (a PowerShell line break gone wrong).
