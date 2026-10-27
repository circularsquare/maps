# Orchestrator handoff, trainworld

Orchestrator: session 7735d270-22c0-47bd-b8fa-dca6f93e8773, name maps-82, since 2026-10-27
Updated: 2026-10-27 13:20

Read trainworld/CLAUDE.md and TASKS.md first. History before this session: notes/handoff-2026-10-26.md.

## State
- master a3d5fdd1: trainsworld logo on the start menu (`?logo=0` gives the old menu), bus tuning
  T-1576 (6 min boarding cost, 300 m, $9/bus-km), fare scale on one line, station circles off by
  default, tidy stages 1-2 (DECISIONS/ARCHIVE split at 2026-10-20 into *-to-2026-10-19.md).
  Before it 2340733d (UI batches, airports 135, rename, station circle scaling, reversal icons).
- 8800 runs d746b717 (T-1559 market packs compact: all-China 3 workers fit, 354 -> 276 s; on
  T-1593 062b915b), cargo 434, smoke 20/20 light and dark, real save 12,483,326. Uncommitted:
  ARCHIVE.md header line moved back to the top. Ids handed out through T-1597.
  Logo agent a59c94fa4d8cd48e7 at ~414k tokens: next logo round goes to a fresh agent
  (notes/T-1567.md has everything).
- Nothing published since 2026-10-25b (deploy only on Anita's word). shonei-server 6d9196d (error
  text says trainsworld) committed, not deployed: Anita says leave it.
- Never commit Anita's TODO.md.
- Images for Anita go in artifacts/trainworld/review/ (one current file each; give her that path).
- Core cap 10. `python cores.py list` at maps root.
- Stray Vites seen at 6160 (pid 27204) and 6198 (pid 19308) from finished agents; harmless.

## In flight
- T-1558 LANDED 06c7692a (all-China peak/worker 2.04 -> 1.54 GB, Japan 2.90 -> 2.71, figures
  identical); cargo 425 pass; 8800 rebuilt from 06c7692a, smoke 20/20 light and dark.
- T-1559 LANDED d746b717.
- T-1545 stage 3 (market by destination window, drops 2nd sweep; approximations OK per Anita;
  times cold Japan/China/NY loads before/after): Opus a979d3cad6318c51f, tree .tw-trees/t1545
  (from d746b717), Vite 6120-6129, 4 cores, may file T-1594-1597.
- T-1586 tidy DONE, committed ef30e01c (no app change, 8800 not rebuilt). Kept-if-unsure list in
  notes/T-1586.md (Canada/US join backups, t279, t1543, publish/).

- T-1580, T-1079, T-955 DONE, committed 6043e7f1; 8800 runs 6043e7f1, smoke 20/20 both shades.
  T-955 asks Anita: MSAA off at dpr>=3 OK?
- T-1508/T-1542 settled ("up to you" + reflect): T-1542 closed, no change; T-1603 (ground fade,
  plain double count) after T-1545; T-1599 refit deferred to a player-scale look. DECISIONS:
  stop single-pair tuning. Anita OK'd MSAA off at dpr>=3. Ids through T-1603.
- Perf T-1021 (Japan load main thread), T-1170 (anchor rebuild), T-1171 (unbuilt line quotes):
  Opus a5895f01c97e31d37, main tree, Vite 6160-6169, 4 cores, may file T-1600-1602.
- Core cap raised to 14 (Anita done with her game, "use all cores").

## Next
1. Land T-1558 and T-1577 as they finish; commit T-1586 stages 3-4; rebuild 8800.
2. Memory: T-1559, then T-1545 (Opus, window over destination zones), T-1223. T-1560 waits (Anita).
3. Calibration before more world: T-1542 (US car), T-1508, T-1374, T-1520. Also T-1580 (t1148
   period-shares check stale), T-1561.
4. Rest of the world after calibration (Anita 2026-10-27).
Ids handed out through T-1590.

## Anita's standing asks
- Every demand task (and loosely perf) has a "done when" in TASKS.md and its brief; stop once met
  (CLAUDE.md, 2026-10-27). Added to T-1169, T-1603, T-1599, T-1374, T-1520, T-1021, T-1170, T-1171.
- 2026-10-27 ~15:45: keep at least 4 cores free for about an hour (to ~16:45): T-1545 told to stay
  within its 4 incl. browser workers; no 8800 rebuilds or smokes meanwhile.
- Performance: approximations welcome when a player can't tell (CLAUDE.md, 2026-10-27). Brief
  T-1545 / T-782 agents with it.
- Buses settled: no more tuning unless she asks.
- Several agents going and one compute job; use idle cores; batch rebuilds.
- Usage week to ~11-01: Sonnet for settled work, Opus for design, data correctness, unknown bugs.
- Images: one real file in review/, full path plus link; never a wildcard.
- Thanks to agents in my own reply, not by message; ask after agents quiet 45+ min.
- Long jobs claim cores in cores.py; refused steps: change and retry, don't stall.
- Someday tasks are real open tasks.
