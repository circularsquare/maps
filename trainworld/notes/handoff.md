# Orchestrator handoff, trainworld

Orchestrator: session f28006bc-4241-4116-9970-a65fc772bb95, name maps-a2, since 2026-10-27 23:08
Updated: 2026-10-28 (overnight)

Read trainworld/CLAUDE.md and TASKS.md first. History before this session: notes/handoff-2026-10-26.md.

## State
- master 85732f70; 8800 runs it (smoke 20/20 light and dark, real save 12,483,326). Published is
  45c0dffb (2026-10-28). Uncommitted: Anita's TODO.md (never commit), this file,
  checks/t1223-build.mjs (untracked, from T-1223's agent).
- Landed this session: 7bc670bc (T-1612, T-1615, T-1614, T-1561), f56389a1 (T-1223 coarse far
  zones), 45c0dffb (no "Rough figures" line), e5fda94f (T-1613 load card text, T-1616 notes),
  035ea692 (T-1605 drives-file retry), f8bd7e67 (T-782 (b) only: Japan 1,579 -> 1,311 s, exact;
  round skip (a) left out, Anita asked), bb00bdbd (T-1626 HSR fare multiplier 1.5x, settable,
  hidden until a high-speed line), bc3dc742 (T-1611 all-China worker peak 3.67 -> 2.99 GB, exact).
  Not yet published: everything after 45c0dffb.
- Earlier landings (2026-10-27): ARCHIVE.md and notes/handoff-2026-10-26.md.
- shonei-server 6d9196d (error text) not deployed: leave it.
- Images for Anita: one current file in artifacts/trainworld/review/, full path.
- Core cap 15 overnight (Anita); back to 12 when she's up (keep 4 free).
- My own Vite: 6163 (dev-nowatch). A three-day-old dev-nowatch on 6160 (PID 27204, not mine,
  maybe stale) exists: don't use it for checks.

## Deploying 2026-10-28 (Anita approved, push included; T-782 goes in the next one)
- R2 upload of 2026-10-28 done and verified (log artifacts/trainworld/deploy-2026-10-28.log);
  website pushed 7663f36 (Anita allowed, 00:50); live, `--verify --page` passes. Live smoke fails
  "far new game final" twice (Tokyo new game, 90 s timeout; 8800 passes in 4-7 s) plus 2 404s:
  re-checked by an agent: passes live now (Tokyo 5-8 s), so transient (CPU contention or CDN cold
  just after publish); one 404 is T-941's, the other error filed as T-1635. The purge of
  2026-10-25b was refused by the auto-mode classifier ("Cloud Storage Mass Delete"): ask Anita to
  run `rclone purge r2:anitamaps/trainworld/2026-10-25b --s3-no-check-bucket` or allow it.
  Ids handed out through T-1635. Original plan: `git -C ..\website add trainworld/`, commit
  "trainsworld", push; after Pages builds `deploy.py --verify --page --version 2026-10-28`; open
  https://anita.garden/trainworld/; then `rclone purge r2:anitamaps/trainworld/2026-10-25b
  --s3-no-check-bucket`. Recipe: notes/T-924.md "Commands for the real upload and push".

## In flight
- T-1618 GPU use (maps-a2's Opus agent, main tree, Vite 6158, 4 cores, ids T-1638-1641):
  measure on the Japan save, rank, file fixes; fix only a small safe big gain.
- Asked Anita: take T-782's round skip (a) for Japan -23% with 8 of 358 lines off up to 11%? I
  recommend no (T-1607 instead). T-1613's card text "Taking a bit longer" is the agent's wording.
- Land a tree: smoke in tree, `node trainworld/checks/agent-tree.mjs land <name> "<sentence>"`
  (rebuilds main WASM; if master moved, `git -C .tw-trees/<name> rebase master` then
  `git merge --ff-only tw/<name>`), `remove <name>`, cargo test, 8800 via scratchpad snapbuild.cmd
  <commit> (C:\Users\anita\AppData\Local\Temp\claude\c--Users-anita-projects-maps\f28006bc-4241-4116-9970-a65fc772bb95\scratchpad\snapbuild.cmd), smoke light and dark.

- T-1617 globe slits: retry with her exact conditions and a real GPU (d3d11) also clean; stopped,
  notes/T-1617.md uncommitted. Waits on her screenshot (promised 2026-10-28).
- T-1605 done (drives-file retry); leftovers T-1633 (tiny wobble), T-1634 (40 zones closed and
  reopened in every all-China load).

## Anita's calls 2026-10-28 (night)
- T-1616 done (notes/T-1616.md): she was on the stress-test save, so losses are the save (T-1625).
- T-1626: settable high-speed fare multiplier, default 1.5x, hidden until the player has a
  high-speed line. After T-782 lands.
- Overnight: all cores (cap 15). Set it back to 12 when she's up.

## Next
1. Land T-1618. Next deploy when Anita says (everything after 45c0dffb).
2. T-1520 (income groups' cost), T-1607 (crowding rounds split across workers), T-1634.
3. T-1374 (superholidays stage 3), T-1605, T-1621 (names follow per frame), T-1609/T-1610.
4. Rest of the world after calibration (Anita 2026-10-27).
Ids handed out through T-1641.

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
