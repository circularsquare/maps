# check7: T-4057 (7289f0d07) against 53ca4ef77, 2026-11-24

Builds: perf_bench dists c7base (53ca4ef77) and c7head (7289f0d07); same track/demand/sim WASM hashes in both (demand 582680013516b7c8). Games ran from those dists on my own static servers (6129 head, 6130 base), play.mjs, real clock running.

## Figures (state.trips byLine, every line compared)
- nyc base 2 workers vs head 3 workers vs head 1 worker: rail 12,672,465, trips/day 899,564,070, 109 lines, per-line riders identical on all 109 lines (both comparisons).
- tokyo (t3939 stand-in, 13 zones, 165 lines) base vs head, 1 worker each (the ledger gave 2 cores): rail 16,262,490, trips 922,216,329, per-line identical.
- perf_bench compare nyc,tokyo x2 (3+ workers): rail 12,672,465 and 16,262,490 in all 8 runs.
- Not done: a tokyo multi-worker per-line compare (cores not free; bench gave rail only), a player network over several zones.

## Edit
Head, nyc 1 worker: edit adding a 3-station line in Philadelphia (110 lines) reached final: version 6, rail 12,875,935. Not compared with base or a fresh load of the edited save (no save export from a production build in reach); zone removal not tried.

## perf_bench compare (2 runs each, machine 70-90% busy; compare-2026-11-241324.json)
tab peak MB: nyc base 2237/2225, head 2290/2370 (+4-6%); tokyo base 2904/2862, head 3000/2982 (+3-4%). Demand WASM per worker identical (334x3; 494/622/494). Under 10%, but head higher in every pair.
solver periodMs: nyc base 233k/176k, head 241k/171k; tokyo base 338k/225k, head 291k/299k (run-to-run swing is larger than any gap). trips wall s: nyc 130/100 vs 131/95; tokyo 181/134 vs 151/169.
