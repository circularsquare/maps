# batchcheck 2026-11-23 (changes since 73299e61, master 0ec0d3df)

Tree .tw-trees/batchcheck, own WASM, Vite 6159. One core most of the time (ledger full), so no timing figures.

## Results
- T-4054 stage 2: PASS. Both cargo suites pass (agent_cargo.py test, exit 0). Scratch tests (not landed): 3000 random 2-4 line networks on 6 stations, every pair reachable by lines has a finite cost and none unreachable has one; 1500 more with all pairs loaded, entries and exits equal the trips put in (gap 0), no NaN or negative loads. Real NYC save in game: 109 lines, final at version 3, round 3, 12,672,465 rail trips a day (no baseline run at 73299e61, so no delta; builders' native figures are +0.007%).
- T-4064: PASS. Branch fixture (trunk T1-T4, North/South at 5/20 min) in game: T2, T3, T4 to S1 100% South from their own station (rides 9.6, 6.9, 4.1 min), T2 to N1 100% North. Random test above found no stranded pair. Not tried: boarding-only / getting-off-only stops (the demand kernel has no such stops, so the bar cannot meet them).
- T-4093: PASS on a small fixture: a headway edit goes version 3 -> 4 once, round 3, 3665 -> 3654, no flicker between rounds. Not sampled on the real save (20+ min on one core).
- T-4079 / T-4092: PASS. Standing t2992-bus-transfers passes (transfers continue at a 106 m stop pair, none lost at 23-68 m). Screenshots at 106 m: dashed ink connector drawn over the blue bus line, also with the line selected (C:/Users/anita/projects/maps/artifacts/trainworld/t2992-hand-sel-140.png, t2992-hand-light-140.png). Passage tool on a bus stop: still picking and "Bus stops can't have passages." shown. Passage hover string "Saves one minute for transfers" present. The old check's drawn counter counts only one-place joins, so its "no connector at 106 m" line is stale in meaning (walk connectors are dashed now).
- T-4069 with T-4079 and an old save: PASS. Exact stops (Ctrl+click op) write LBE1 (gunzipped save), survive export, reload and a later edit, and LBE1 goes when both exact stops are removed. The old real save (no LBE1) loads.
- T-4072: PASS (partial). With 11 pack requests held and ?maskWaitMs=6000 the real save opened in 29 s and released requests caused no page errors. The 20 s default and the late install were not separately observed. Worker start timeout (?clockStall=boot&startMs=3000): started twice, then "The game's files did not finish downloading. Check your connection and try again." with "(its script did not arrive, started twice, 6 s)" appended.
- T-4073: NOT TESTED. Losing the first / map WebGL context in setup did not reach the lost-at-setup path (map recovered, game up, no reload). Needs real Safari behaviour.
- T-3940 / T-4089: PASS on logic only. Emulated iPhone user agent with no deviceMemory: tabLimited true, bigForTab(1599) false, (1600) true, overBudget(1756) true, (300) false, (undefined) false. Start menu card, Open anyway / Cancel buttons and the died-slot ask not rendered by me (builder ran it in WebKit). No openWithoutTrips left in app/src.
- Hokkaido: PASS. city=spk opens as "Dochuo", view centred on Sapporo.
- Touch (T-4076): t4076-touch-track.mjs all passed.

## Notes
- trainworld/app/src/workers/clockGround.ts now starts with a byte order mark (U+FEFF) from 840bfa8f; harmless to Vite, worth stripping.
- The first t2992 attempts queued for 3 free cores in the ledger and looked hung; TW_CORES_WANT=1 fixes that.
- Skipped: real-save flicker, a baseline for the real-save figure, express/local on the real save beyond the fixtures, BRT +1 min in game (covered by the kernel test a_bus_stops_walks_are_shorter_and_slower_without_a_passage, passing), dark mode, touch bus stop edits, phone UI.
