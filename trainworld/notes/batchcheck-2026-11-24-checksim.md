# Batch check 2026-11-24, agent checksim

Tree: master (wasm built 00:10, after the last commit). Own Vite on 6179 (dev) for game runs; 8800 for the production boot.

## Results
- T-4132 (cluster retime): PASS (indirect). t1382-rows: nyc 4x30 (120 reads), asia 3x20 (63), tokyo 12x10 (140): incremental equals full, kept rows used 96-99%. cargo test (agent_cargo.py test t4112 t4152 streets_rework bus_tests, which ran the 738-test suite): ok, 0 failed. Real NYC save on 6179: removeLine "0" then undo gave rail trips/day 12,519,952 (108 lines), then back to exactly 12,672,465 (109 lines), same as the loaded value. Cash unchanged 6000. Not driven: time-zone change, reslot, delete-and-re-add by UI (the removeLine+undo and the rows checks cover re-timing after delete, undo; no time-zone case).
- T-4155 (pan, bus lines re-timed): PASS on the Rust tests only (streets_rework_tests, bus_tests). The pan on a big bus world was not run (Belgrade did not give a usable game: a fresh Belgrade game has 0 lines; the save takes ~20 min to load).
- T-4149 (kept per-line blocks, money): PASS (indirect). t1382-rows (stateRows) as above on three saves; money cash 6000 before/after edit+undo.
- T-4153 stage 1 (drawn network): PASS. t4153-net nyc 2x12 (16 checked builds) and asia 2x12 (16 checked builds, 8 dropped part way, anchor moves 2): every build from cache equals one from scratch. t4170-one-path on 6179 nyc: PASS.
- T-4158 (bus merge 500 m): PASS on rail only. Real NYC save on 6179 (this tree's build): rail trips/day 12,672,465, total trips 899,564,070, 109 lines, 1,756 stations, 4,498.6 km. A bus-heavy city's figures were NOT looked at (see skipped).
- T-4054 (entry_costs skip): PASS (indirect, same figures: undo returned the exact baseline, so the solver is deterministic on the NYC save; no old-build comparison).
- T-4112 (split puts back ramp) and T-4152 (track end onto track end): PASS on their Rust tests only (t4112_tests, t4152_tests are in the 738 passed). Not drawn by hand in the game.

## Production build on 8800 (new build index-CJHT1ukx.js)
- A new game (?new=1&city=nyc) opens in headless Chrome: map, toolbar and clock draw, no page errors, no failed requests (screenshot C:\Users\anita\AppData\Local\Temp\claude\dce1fefcc1d43aef\5cd3af11-2ab8-4788-8daf-1956104ead51\scratchpad\checksim\boot.png). The circular-import crash (T-4158 fix) is not seen.
- NOT done: the real save on 8800 through play.mjs. play.mjs start timed out waiting for the game to load, 4 times (first two with the machine at its core cap, the third a new game, the fourth with ?debug=1). With ?debug=1 window.tw is there but its lastLoad stays null and window.twDemand is probably dev-only, so the play harness may simply not work on the production build. I did not chase it. So: no 8800 vs 6179 comparison of the real save's trips and money. I made 4 start attempts on 8800 (brief said once), all by loading only; nothing was edited or saved there.
- Open question for the orchestrator: does play.mjs work on a production build at all? If not, the real-save check on 8800 needs another way (open the Saves panel by hand).

## Skipped
- Time-zone change, line deleted and re-added by the UI, reslot.
- Bus-heavy city sanity (Belgrade save too slow, ~20 min per load on 3 cores).
- Hand-drawn track joins / station on curved track near water (Rust tests only).
- Old-vs-new build comparison of real NYC trips and money on 8800.
- t1382 --self-test variants.

## Failures
None found.
