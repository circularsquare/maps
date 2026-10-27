# Orchestrator handoff, trainworld

Orchestrator: session 7735d270-22c0-47bd-b8fa-dca6f93e8773, name maps-82, since 2026-10-27
Updated: 2026-10-27 18:00

Read trainworld/CLAUDE.md and TASKS.md first. History before this session: notes/handoff-2026-10-26.md.

## State
- master a497ea41; 8800 runs it, smoke 20/20 light and dark, real save 12,483,326, cargo 434.
- Landed today (all in ARCHIVE.md): two UI batches, rename to trainsworld, the script logo on the
  start menu (T-1567, T-1593), airports 135 min (T-1564), buses tuned and settled (T-1576, T-1577),
  station circles off by default and scaled by zoom, fare row, tidy (T-1586: DECISIONS/ARCHIVE
  split at 2026-10-20, notes/archive by id range, checks/old), memory stages 2, 2b, 3 (T-1558,
  T-1559, T-1545: all-Japan peak/worker 2.90 -> 1.33 GB, cold load 1,627 -> 1,107 s; all-China
  peak 2.04 -> 1.30 GB), main-thread perf (T-1021, T-1170, T-1171), small fixes (T-955, T-1079,
  T-1580). T-1542 closed no change; T-1466, T-1575 closed.
- Nothing published since 2026-10-25b (deploy only on Anita's word). shonei-server 6d9196d not
  deployed (Anita: leave it).
- Never commit Anita's TODO.md. Images for Anita: one current file in artifacts/trainworld/review/.
- Core cap 14 (Anita said use all cores, 2026-10-27 evening).

## In flight
- T-1603 LANDED 893a5044 (US 300 km+ 0.0195 -> 0.0100 vs NHTS 0.0098; China 1.55x; NY save
  unchanged). T-1508 closed, T-1599 deferred to a player-scale look.
- T-1594 all-China load ~10% slower after T-1545: Opus ab3d59c8a6ee019c6, tree .tw-trees/t1594
  (from a497ea41), Vite 6180-6189, 6 cores, may file T-1605-1606.
- Land each: smoke in tree, `agent-tree.mjs land <name> "<sentence>"` (rebuilds main WASM),
  remove, cargo test on master, rebuild 8800 (scratchpad snapbuild.cmd = the CLAUDE.md recipe),
  smoke both shades.

## Next
1. After T-1603: T-1599 only if a built Beijing - Shanghai line falls short of its done-when.
2. Memory: T-1223 (coarse zones) toward T-1169's done-when (all-China every zone under 3.2 GB).
   T-1560 waits (Anita).
3. T-782 (market crowding rounds: most of Japan's remaining 18 min), T-1520 (income groups' cost),
   T-1374 (superholidays stage 3), T-1561.
4. Rest of the world after calibration (Anita 2026-10-27).
Ids handed out through T-1606.

## Anita's standing asks
- Every demand task (loosely perf) has a "done when"; stop once met (CLAUDE.md).
- Performance: approximations welcome when a player can't tell; name them in DECISIONS.md.
- Buses settled: no more tuning unless she asks.
- No single-pair demand tuning (DECISIONS 2026-10-27).
- Images: one real file in review/, full path plus link; never a wildcard.
- Thanks to agents in my own reply, not by message; ask after agents quiet 45+ min; final checks
  in the foreground; browser loads count their demand workers in core claims.
- Usage week to ~11-01: Sonnet for settled work, Opus for design, data correctness, unknown bugs.
- Someday tasks are real open tasks.
