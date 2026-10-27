# Orchestrator handoff, trainworld

Orchestrator: session 7735d270-22c0-47bd-b8fa-dca6f93e8773, name maps-82, since 2026-10-27
Updated: 2026-10-27 (commit round)

Read trainworld/CLAUDE.md and TASKS.md first. History before this session: notes/handoff-2026-10-26.md.

## State
- Commit round 2026-10-27: T-1547-1554 (two UI batches), T-1564 (airports 135), T-1565/1566
  (station circles), T-1568 (rename to trainsworld), T-1567 logo v5 behind `?logo=1`, T-1581
  (small reversal icons), T-1169 stage 1 notes. Smoke 20/20 on dev before commit (real save
  12,483,326), t1554 check passes. 8800: see the commit after this file's.
- Nothing published since 2026-10-25b (deploy only on Anita's word). shonei-server 6d9196d (error
  text says trainsworld) committed, not deployed: Anita says leave it.
- Never commit: Anita's TODO.md, notes/t1150/run/ (196 MB).
- Core cap 10. `python cores.py list` at maps root.
- Stray Vites seen listening at 6160 (pid 27204, T-1169's, finished) and 6198 (pid 19308, logo
  agent's?) — harmless; stop by PID only if sure.

## In flight
- T-1558 memory stage 2 (exact fixes; finals identical): Sonnet aba2af67a79bdeb61, tree
  .tw-trees/t1558 (branch tw/t1558 from b03b751f), Vite 6120-6129, 4 cores, may file T-1578-1579.
  Land: smoke in tree, `agent-tree.mjs land t1558 "<sentence>"`, remove, `npm run wasm` in main app.
- T-1567 logo: v5 done behind `?logo=1` (Opus a59c94fa4d8cd48e7, resume it). Waiting on Anita's
  look; my notes: t's crossbar busy (3 stations), the a reads a bit like a u. When it goes in,
  the welcome heading goes.

## Next
1. After this commit: rebuild 8800, smoke light and dark, give Anita the test list.
2. T-1576 bus tuning (Anita yes: 6 min boarding cost on every bus boarding incl. changes, 300 m,
   $9/km), then T-1577 (speed by place). Sonnet; reruns t1466 checks.
3. Memory: after T-1558, T-1559, then T-1545 (Opus, window over destination zones), T-1223.
   T-1560 waits (Anita).
4. Calibration before more world: T-1542 (US car), T-1508, T-1374, T-1520. Also T-1575 (42nd St
   bus pushes riders to cars), T-1580 (t1148 period-shares check stale), T-1561.
5. Rest of the world after calibration (Anita 2026-10-27).
Ids handed out through T-1581.

## Anita's standing asks
- Several agents going and one compute job; use idle cores; batch rebuilds.
- Usage week to ~11-01: Sonnet for settled work, Opus for design, data correctness, unknown bugs.
- Link every image as a clickable workspace-relative markdown link; give a full absolute path when
  she asks where something is.
- Long jobs claim cores in cores.py; refused steps: change and retry, don't stall.
- Someday tasks are real open tasks.
