# T-3553 check (shorecheck, 2026-11-17): PASS

Tree: .tw-trees/shoreswap, WASM rebuilt, Vite on 6158 (stopped). Scripts: scratchpad shorecheck/ (sweep2, berl3, par, lib2).

- `cargo test --release --features demand --lib`: 795 passed, 0 failed, 39 ignored. `cargo test --release`: 604 passed, 0 failed, 11 ignored.
- Real save T-007-real-nyc.save, native: old code (HEAD cost.rs, run in a scratch copy) 154 edges with water, 67,998 wet m, built cost 733,794; shoreswap 161 edges, 72,279 m, 734,596 (+0.11%). Both reproduced independently. Save loads in the browser (1,647 edges, 109 lines, no page errors).
- TS vs Rust parity (lineProfile.ts `waterAt`, copied and run in Node, vs `RunMask::is_water` on the nyc pack): 63,500 points (60,000 random around Newtown Creek, 2,000 on cell centres/edges/corners, 1,500 along the raster's four edges and off it), 0 differences, 3,810 wet.
- World map: `between: false` path is the old `cell_water` call unchanged (the old `shore = 0` branch returned the same value); the world mask test and frame tests pass.
- Bank vs bridge (browser, ghost price per m, NYC Newtown Creek z16, 186 m parallel line stepped 10 px = 9 m across the creek): 0.045/m on land up to the bank, ramp 0.075, 0.12, 0.1425, plateau 0.1575 (wet) over 4 steps, then back to 0.045 on the far bank. A line grazing the edge (y=300) stays 1.00x with no Water tag; 9 m further in gets "Water +2.5x" (1.67x). No bridge charge on a bank.
- Berlin Spree (Mitte), 392 m horizontal crossings: a ~95 m crossing was charged 61 m wet (~65%). A crossing through ~45-50 m of Spree under a bridge deck (y=400 case) was charged nothing; that is the narrow-water limit (under 50 m opening), not new. Spreekanal (~30 m) is not charged at all, as the notes say.
- Station on a river crossing: a station on mid-river track places fine (priced as over water, as before). Further placements near it are refused with "There is already a station here", a spacing rule, not water; I did not get a clean station-near-bank comparison.
- Raster edge: covered by the parity test (edge repeats last cell, off raster is land, same as before) and the unit test.
- Not done: seeing the line profile chart's blue band next to a built line (the harness could not place the second station), so profile vs price is shown through parity of the rule, not by screenshot. Profile samples every 20 m, sim every 12.5 m (already so before).
