# Orchestrator handoff, trainworld

Orchestrator: session f28006bc-4241-4116-9970-a65fc772bb95, name maps-a2, since 2026-10-27 23:08
Previous: session 7735d270-22c0-47bd-b8fa-dca6f93e8773, name maps-82, handed over 2026-10-27 23:08, finishing in-flight work
Updated: 2026-10-28 00:40

Read trainworld/CLAUDE.md and TASKS.md first. History before this session: notes/handoff-2026-10-26.md.

## State
- master 45c0dffb; 8800 runs it, smoke 20/20 light and dark, real save 12,483,326 rail trips a day.
  Uncommitted: Anita's TODO.md (never commit), this file, T-1616's notes/scripts (notes/T-1616.md,
  notes/t1616_*.mjs; ARCHIVE/TASKS edits), checks/t1223-build.mjs (untracked, from T-1223's agent).
- Landed this session: 7bc670bc (T-1612 start menu delete/load saves, T-1615 auto-ramp switch on by
  default, T-1614 capacity tags follow every frame, T-1561 no bug), f56389a1 (T-1223 coarse far
  zones; all 723 Chinese zones load, worker 3.4-3.67 GB vs 3.2 target), 45c0dffb (no "Rough figures"
  line: Anita 2026-10-28 "close enough"; T-1617 notes).
- Earlier landings (2026-10-27): ARCHIVE.md and notes/handoff-2026-10-26.md.
- shonei-server 6d9196d (error text) not deployed: leave it.
- Images for Anita: one current file in artifacts/trainworld/review/, full path.
- Core cap 12 (Anita 2026-10-28: keep 4 cores free).
- My own Vite: 6163 (dev-nowatch). A three-day-old dev-nowatch on 6160 (PID 27204, not mine,
  maybe stale) exists: don't use it for checks.

## Deploying 2026-10-28 (Anita approved, push included; T-782 goes in the next one)
- R2 upload of 2026-10-28 done and verified (log artifacts/trainworld/deploy-2026-10-28.log);
  website pushed 7663f36 (Anita allowed, 00:50); live, `--verify --page` passes. Live smoke fails
  "far new game final" twice (Tokyo new game, 90 s timeout; 8800 passes in 4-7 s) plus 2 404s:
  an Opus agent (scratchpad tlive) is diagnosing, report only. Don't purge 2026-10-25b until it's
  understood (rollback = revert website 7663f36). Original plan: `git -C ..\website add trainworld/`, commit
  "trainsworld", push; after Pages builds `deploy.py --verify --page --version 2026-10-28`; open
  https://anita.garden/trainworld/; then `rclone purge r2:anitamaps/trainworld/2026-10-25b
  --s3-no-check-bucket`. Recipe: notes/T-924.md "Commands for the real upload and push".

## In flight
- T-782 market crowding (maps-82's agent, tree .tw-trees/t782 from e202c0de): report due ~01:45;
  one crowding pass cuts Japan's load ~a third, 500 s not met, follow-ups named. maps-82 forwards
  it; maps-a2 rebases tw/t782 onto master, lands, removes tree, cargo test, 8800, smoke both.
- T-1611 all-China worker memory (maps-a2's Opus agent, tree .tw-trees/t1611, Vite 6190-6194, 4
  cores, ids T-1629-1630): measure first, fix only if substantial and not super hard (Anita
  2026-10-28: 3.2 GB is a target; method my call), else close with the breakdown.
- T-1613 load estimate (maps-a2's Opus agent, main tree, Vite 6156, 2 cores, ids T-1631-1632).
- Land a tree: smoke in tree, `node trainworld/checks/agent-tree.mjs land <name> "<sentence>"`
  (rebuilds main WASM; if master moved, `git -C .tw-trees/<name> rebase master` then
  `git merge --ff-only tw/<name>`), `remove <name>`, cargo test, 8800 via scratchpad snapbuild.cmd
  <commit> (C:\Users\anita\AppData\Local\Temp\claude\c--Users-anita-projects-maps\f28006bc-4241-4116-9970-a65fc772bb95\scratchpad\snapbuild.cmd), smoke light and dark.

- T-1617 globe slits, retry (same Opus agent resumed, Vite 6153, up to 4 cores) with her
  conditions: light, all of Japan in view, terrain on, still, the t737 Japan save. Screenshot
  from her 2026-10-28.
- T-1605 all-China signature varies (maps-a2's Opus agent, main tree, Vite 6157, 5 cores, ids
  T-1633-1634): investigation, small local fix only.

## Anita's calls 2026-10-28 (night)
- T-1616 done (notes/T-1616.md): she was on the stress-test save, so losses are the save (T-1625).
- T-1626: settable high-speed fare multiplier, default 1.5x, hidden until the player has a
  high-speed line. After T-782 lands.
- Overnight: all cores (cap 15). Set it back to 12 when she's up.

## Next
1. Finish the deploy. Land T-782 when it reports.
2. After T-782 lands: T-1626 (HSR fare multiplier), T-1618 GPU use, T-1520 (income groups' cost).
3. T-1374 (superholidays stage 3), T-1605, T-1621 (names follow per frame), T-1609/T-1610.
4. Rest of the world after calibration (Anita 2026-10-27).
Ids handed out through T-1634.

## Anita's standing asks
- Every demand task (loosely perf) has a "done when"; stop once met. No single-pair demand tuning.
- Performance: approximations welcome when a player can't tell; name them in DECISIONS.md.
- Buses settled: no more tuning unless she asks.
- Thanks to agents in my own reply, not by message; ask after agents quiet 45+ min; final checks
  in the foreground; browser loads count their demand workers in core claims.
- Usage week to ~11-01: Sonnet for settled work, Opus for design, data correctness, unknown bugs.
- Someday tasks are real open tasks.
- Keep 4 cores free (cap 12) while she's up; all cores overnight.
- Tokens tight this week (2026-10-28): prefer compute-heavy, token-light approaches (scripted
  sweeps, long detached runs over agent iteration); fewer agents at once, investigation tasks
  stop at a written answer.
