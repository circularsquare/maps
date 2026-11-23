# batchcheck15 (2026-11-22, master 5e4ac09f, own Vite 6145; scratch: scratchpad\batchcheck15)

## T-4050 (evenly spaced lines) - PASS (a, b, c); d skipped
NYC sandbox, T-4019's Long Island Local (15 stops) + Express (4), plus Short (Local's first 10 stops). Local 15 min, others 30.
- (a) Pairs on the trunk (0-1, 0-3, 2-8): Local 67 / Short 33 (trains an hour). 0-5 with Express: 51 / 26 / 23 (Express apart, Local:Short 2:1).
  Past Short's end (3-13, 0-12): Local 94 / 80, Short>Local 6 / 5 (branch riders mostly wait for their own line). Same as T-4050's table.
  Riders: Local 28,063, Express 918, Short 8,776 (trips 37,259).
- (b) Added Skip (stops 0,1,3,4,5; 30 min). 0-1 (all three leave the same platform for the same one): 50 / 25 / 25. 0-3 (Skip rides 2
  stops, Local/Short 3): Local 53, Short 27 (still 2:1), Skip 20 (independent, not 25). 0-5: 42 / 21 / 20 / 17.
- (c) removeLine Short, then undo: after undo identical to before (trips 37,259, byLine and every pair split the same).
- (d) Skipped: no tram stop can be placed by act (T-4041's note); sim tests only.

## T-4054 stage 1 - PASS, with a figure to correct
Real save, 4 cores: load 18 s, final round 3 in about 2 min. Rail trips 12,660,073 a day (not 13,022,833; that figure is smoke's
readout and notes' T-4041 line. batchcheck14 read 12,660,073 on master before these commits: same answers). Boardings 23,539,563 vs
alightings 23,539,608, as batchcheck14. removeLine 0: final, 12,507,332, b/a 22,924,621/22,924,653. Undo: final, 12,659,907
(166 below the first, 1e-5), b/a 23,537,723/23,537,767. No page errors. The "across" edit near Manhattan was refused (CrossingOverlap), not a defect.

## Load card bar (0c336c7f) - FAIL (new-game loads)
bar2.mjs/bar.mjs (aria-valuenow every 40 ms), new Tokyo load, once paused and once running: bar rises monotonic 0 -> 2 -> 30 -> 41-43 and the card
goes with the last sample at 41-43%, no sample at 100. The fix adds endCardFull at demandClient.ts:2553 (the `solved` path), but a game
load with a hold ends at checkHold (demandClient.ts ~2937, the `s.published < ROUNDS` branch: endJob, loadProgress = null, no endCardFull);
only the cached-result branch (2931) fills the bar. Real-save load not sampled.
