# Batch check 2026-11-24, check3 (T-4179 c842cd599, T-4184 20d3a0029)

Machine: cap 15, always full (other agents' builds); benches got 3-5 cores. Base = d991f7ab6 (perf_bench build c3-base2), head = 20d3a0029 (c3-head).

## Harness finding
perf_bench build of two commits started at once (c3-base) came out with the head's WASM in both (same md5): the builds share
perfbench/target-track and target-demand. Rebuilt the base alone (c3-base2): its demand WASM is the old one (same md5 as rel046),
but its track WASM is byte-identical to the head's (Dj3qmP-h, 5a329ec2...), although T-4184 changed boarding.rs. So no
true base track WASM exists in these builds: the T-4184 clock memory could not be A/B'd by bench. Not resolved why.

## T-4179 PASS
- bench, no ledger, base c3-base2 (4 cores) vs head c3-head (5 cores): rail trips a day nyc 12672465 = 12672465, tokyo 16262490 = 16262490.
  Demand WASM MB: nyc 1163 -> 1067, tokyo 1751 -> 1687; belgrade head 3711 (rel046 base 4189). Tab peak nyc 2450 -> 2469, tokyo 3096 -> 2971; page, clock heap the same.
- belgrade head 1396047 rail trips on 3 cores (2 demand workers) vs 1396045 in the ledger on 5 cores / 3 workers: 2 of 1.4M, worker count differs; no base run on 3 cores, so not shown to be nothing.
- Real NYC game (T-007-real-nyc.save), per-line riders of all 59 lines identical base vs head at load, after each of: setStops (drop a stop of line 0), setTrainCount line 0 [30,24,9],
  removeStation 120, and after each of three undos: rail 12672910, 12676081, 12676879, then 12676081, 12672910, 12672465; 0 lines differ at every step; final == original.
  (A first head run read 12672453 after undos because two waits timed out under CPU load and the read came mid-solve; a clean rerun matched base exactly.)
- cargo test (agent_cargo.py test): 746 + demand lib suite pass, including trip_order_reads_back_in_both_widths.
- Code read: Ix::of picks u16 only if every value < 65535; Shares.wait is empty in skims tables and only walk_column (before without_waits) and trip::routes (searches again) read it.
- Not tried: a zone-edge-values save (no fixture found); Belgrade base run.

## T-4184 PASS (partial)
- 16-bit order: unit test covers both widths and the 65,536 boundary (Short at <= 65,536 departures, so max slot 65,535). A run above that needs >2,700 departures an hour: no game network reaches it; Wide path only covered by the unit test.
- t4153-net asia and nyc: every build from the cache equals one from scratch. t1381-strokes: pass.
- Outlines t3491 (world, europe, germany, nyc, zone outlines; head on Vite vs base prod build): features and points identical (4792/906268, 1160/228667, 222/44583, 16/3019); screenshots equal below the top 71-118 px band (dev vs prod page chrome); one 404 in both.
- Corridors: no direct hover check run; covered only by t4153/t1381 and the NYC game running with no page errors.
- Clock worker memory (the headline, 225-244 -> 180-187 MB on an emulated iPhone) not reproduced: no true base track WASM, no phone emulation run. Bench clock WASM nyc 65, tokyo 97 same in base2 and head (expected, track WASM identical).
- Not tried: save/reload figures equal after a built-network save (NYC reload gave the same 12672465 final each time).
