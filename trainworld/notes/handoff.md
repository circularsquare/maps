# Orchestrator handoff, trainworld

Orchestrator: session 7735d270-22c0-47bd-b8fa-dca6f93e8773, name maps-82, since 2026-10-27
Updated: 2026-10-27 13:20

Read trainworld/CLAUDE.md and TASKS.md first. History before this session: notes/handoff-2026-10-26.md.

## State
- master a3d5fdd1: trainsworld logo on the start menu (`?logo=0` gives the old menu), bus tuning
  T-1576 (6 min boarding cost, 300 m, $9/bus-km), fare scale on one line, station circles off by
  default, tidy stages 1-2 (DECISIONS/ARCHIVE split at 2026-10-20 into *-to-2026-10-19.md).
  Before it 2340733d (UI batches, airports 135, rename, station circle scaling, reversal icons).
- 8800 runs 172383a3 (T-1577 bus speed by place, on T-1558 06c7692a, tidy ef30e01c, a3d5fdd1),
  cargo 433 pass, smoke 20/20 light and dark, real save 12,483,326. T-1466 and T-1575 archived
  (uncommitted TASKS/ARCHIVE edit). Ids handed out through T-1592.
- Nothing published since 2026-10-25b (deploy only on Anita's word). shonei-server 6d9196d (error
  text says trainsworld) committed, not deployed: Anita says leave it.
- Never commit Anita's TODO.md.
- Images for Anita go in artifacts/trainworld/review/ (one current file each; give her that path).
- Core cap 10. `python cores.py list` at maps root.
- Stray Vites seen at 6160 (pid 27204) and 6198 (pid 19308) from finished agents; harmless.

## In flight
- T-1558 LANDED 06c7692a (all-China peak/worker 2.04 -> 1.54 GB, Japan 2.90 -> 2.71, figures
  identical); cargo 425 pass; 8800 rebuilt from 06c7692a, smoke 20/20 light and dark.
- T-1559 memory stage 2b (market packs held once): same Sonnet aba2af67a79bdeb61, tree
  .tw-trees/t1559 (from 06c7692a), Vite 6120-6129, 4 cores, may file T-1591-1592.
- After T-1559: T-1545 stage 3 (Opus), and time a cold all-Japan load before/after (Anita asked
  how load scales with stations; answered: market s x s tables quadratic, routing, pack opening).
- T-1586 tidy DONE, committed ef30e01c (no app change, 8800 not rebuilt). Kept-if-unsure list in
  notes/T-1586.md (Canada/US join backups, t279, t1543, publish/).

## Next
1. Land T-1558 and T-1577 as they finish; commit T-1586 stages 3-4; rebuild 8800.
2. Memory: T-1559, then T-1545 (Opus, window over destination zones), T-1223. T-1560 waits (Anita).
3. Calibration before more world: T-1542 (US car), T-1508, T-1374, T-1520. Also T-1580 (t1148
   period-shares check stale), T-1561.
4. Rest of the world after calibration (Anita 2026-10-27).
Ids handed out through T-1590.

## Anita's standing asks
- Performance: approximations welcome when a player can't tell (CLAUDE.md, 2026-10-27). Brief
  T-1545 / T-782 agents with it.
- Buses settled: no more tuning unless she asks.
- Several agents going and one compute job; use idle cores; batch rebuilds.
- Usage week to ~11-01: Sonnet for settled work, Opus for design, data correctness, unknown bugs.
- Images: one real file in review/, full path plus link; never a wildcard.
- Thanks to agents in my own reply, not by message; ask after agents quiet 45+ min.
- Long jobs claim cores in cores.py; refused steps: change and retry, don't stall.
- Someday tasks are real open tasks.
