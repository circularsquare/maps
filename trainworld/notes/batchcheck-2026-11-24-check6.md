# check6: T-3160 stage 6 (53ca4ef77), 2026-11-24

Result: PASS on what was run; the phone setting and a 1-worker marketless run were not done.

- Builds: perf_bench dists c6base (65c20f3bd) and c6head (53ca4ef77); demand WASM hashes differ (1360...  vs 5826...), track WASM same.
- perf_bench compare nyc, 2 runs each (compare-2026-11-241210.json): rail 12,672,465 both. Means: load+edit CPU 238.3+8.3 -> 233.6+10.0 s, tab peak 2227 -> 2340 MB, demand WASM 1067 both, solver 173 -> 194 s (+12%, 5 cores shared with a busy machine, other agents' runs). Desktop workers hold a market, so memory is not expected to move; tab peak +5% is one noisy pair, not repeated.
- perf_bench compare tokyo, 3 runs each (compare-2026-11-241224.json): rail 16,262,490 both. Means: load+edit CPU 261.7+10.4 -> 254.3+9.9 s (edit CPU did not rise: the builder's 11.3 -> 13.0 was noise), tab 3067 -> 3008 MB, demand WASM 1751 -> 1739, solver 184 -> 182 s.
- Game, own Vite 6126, real save, ?demandWorkers=1: rail 12,672,465 (play.mjs trips), per-line riders recorded; a reload gives 12,672,465 again.
- Marketless then market on one worker: an edit's solve runs v3 with no market, then v4 with 9 market zones, on the same worker thread (tables cut marketless, then a market). Result 12,672,453 (899,562,200 trips/day). A fresh load plus the same edit gave the same 12,672,453, v3 m0 then v4 m9. The two runs share the pattern, so this shows no stale table in the market solve, not a base-build comparison.
- Toggling display.intercityTrips off then on in a running game started no new solve (setDisplay only); not a usable trigger.

Skipped: coarseDemand on, a base-build comparison of the edited figure, edit then undo (play.mjs has no undo), per-line riders against the earlier build (only against a reload of the head), emulated-iPhone memory (330 -> 261 MB claim not re-measured).
