# Orchestrator handoff, trainworld

Orchestrator: session f28006bc-4241-4116-9970-a65fc772bb95, name maps-a2, since 2026-10-27 23:08
Previous: session 7735d270-22c0-47bd-b8fa-dca6f93e8773, name maps-82, handed over 2026-10-27 23:08, finishing in-flight work
Updated: 2026-10-27 23:08

Read trainworld/CLAUDE.md and TASKS.md first. History before this session: notes/handoff-2026-10-26.md.

## State
- master e202c0de; 8800 runs it, smoke 20/20 light and dark, real save 12,483,326 rail trips a
  day, cargo 434. Main tree clean apart from Anita's TODO.md (never commit) and this file.
- Landed 2026-10-27 (all in ARCHIVE.md): two UI batches, rename to trainsworld, the script logo
  on the start menu (T-1567, T-1593; `?logo=0` old menu), airports 135 min (T-1564), buses tuned
  and settled (T-1576, T-1577), station circles off by default and scaled by zoom, fare row, tidy
  (T-1586), memory stages 2, 2b, 3 (T-1558, T-1559, T-1545) and memory growth (T-1594): all-Japan
  peak/worker 2.90 -> 1.33 GB and cold load 1,627 -> 954 s, all-China peak 2.04 -> 1.30 GB and load
  ~325 -> 97 s; ground-curve fade (T-1603); main-thread perf (T-1021, T-1170, T-1171); small fixes
  (T-955, T-1079, T-1580). Closed: T-1466, T-1575, T-1542 (no change), T-1508 (refit = T-1599).
- Nothing published since 2026-10-25b. shonei-server 6d9196d (error text) not deployed: leave it.
- Images for Anita: one current file in artifacts/trainworld/review/, full path plus link.
- Core cap 12 (Anita 2026-10-28: keep 4 cores free). Agents told to fit.

## In flight (agents of maps-82; reports come here)
- T-782 market crowding rounds: Opus ab0c4e7e9b0968421, tree .tw-trees/t782 (from e202c0de),
  Vite 6170-6179, 6 cores, ids T-1607-1608. Done when all-Japan cold load < ~500 s, China/NY no
  slower, real save rail trips within 0.5%, Japan station/line figures within ~2%. Notes T-782.md.
- T-1223 coarse zones (T-1206 stages 2-4): Opus a1b9c1726d8ff1596, tree .tw-trees/t1223 (from
  e202c0de), Vite 6180-6189, 6 cores, ids T-1609-1611. Done when all-China every zone < 3.2 GB a
  worker, zones in view within ~2% of fine. Review shot: artifacts/trainworld/review/t1223.png.
- maps-82 lands both and messages maps-a2 after each; the deploy is maps-a2's (agreed 23:10).
- T-1223 done, handed to maps-a2 to land: squashed on branch tw/t1223 (1b8fb7dd on e202c0de), tree
  .tw-trees/t1223 kept; ff blocked by main-tree edits. Plan: when T-1614 reports, commit the
  main-tree batch, rebase tw/t1223, ff, remove tree, typecheck, 8800, smoke both shades. Done-when
  NOT met (all-China worker 3.4-3.67 GB vs 3.2; T-1609-1611 filed), so ask Anita before deploying.
- T-1561 done 2026-10-27 (no bug: test lines cost more than a new game's 6,000; checks/lib.mjs
  buildLine now reports the Money refusal). Uncommitted, maps-a2's: checks/lib.mjs, TASKS.md,
  ARCHIVE.md, notes/T-1553.md. Commit after maps-82 is done landing. Ids T-1612-1613 unused.
- Anita's TODO short term (night) filed as T-1612-1618. Agents of maps-a2, main tree, launched 23:30:
  T-1612 start menu delete/load saves (Sonnet, Vite 6151, ids T-1619-1620); T-1614 capacity
  indicator lag (Opus, 6152, T-1621-1622); T-1617 globe slits (Opus, 6153, T-1623-1624); T-1616
  all-Japan money analysis, notes only (Opus, 6154, T-1625-1626); T-1615 auto-ramp switch (Opus,
  6155, T-1627-1628). T-1612 done 00:05 (uncommitted: StartMenu.tsx, SavesPanel.tsx, style.css,
  workers/cloud.ts, clockClient.ts, clock.worker.ts delete branch, docs/ui-panels.md, DECISIONS,
  ARCHIVE, TASKS). T-1615 done 2026-10-28 (uncommitted: sim/src/track/bridge.rs, wasm/draw.rs,
  wasm.rs, wasm/ramp_tests.rs; app game/ui.ts, BuildTab.tsx, widgets.tsx, map/tools.ts,
  clock.worker.ts, protocol.ts, clockClient.ts; docs/track-and-service.md, ui-building.md,
  DECISIONS, ARCHIVE, TASKS; cargo test 286 pass, typecheck and smoke pass). Held: T-1613 (load ETA; after T-1223), T-1618 (GPU; after T-782, T-1223).
- If maps-82 is gone: each is done when its task is at the top of ARCHIVE.md; check its notes and
  `agent-tree.mjs list`.
- Land each: smoke in tree, `node trainworld/checks/agent-tree.mjs land <name> "<sentence>"`
  (rebuilds main WASM), `remove <name>`, cargo test on master, rebuild 8800 (CLAUDE.md recipe;
  scratchpad snapbuild.cmd wraps it), smoke light and dark.

## DEPLOY APPROVED (Anita 2026-10-27 evening)
After T-782 and T-1223 land, if they go okay: publish to the website, push included. Recipe in
notes/T-924.md "Commands for the real upload and push": deploy.py --dry-run, then
`--version 2026-10-27 --smoked` (~700 MB to R2), website repo add/commit/push (message
"trainsworld"), `--verify --page` after Pages builds, open https://anita.garden/trainworld/ and
play a minute, then purge the previous R2 prefix once live. If either task has problems, ask her.

## Next
1. Land T-782 and T-1223, then deploy.
2. T-1520 (income groups' cost, done-when in TASKS.md), T-1374 (superholidays stage 3), T-1605
   (all-China figures vary run to run), T-1561, T-1599 only if a built Beijing - Shanghai line
   falls short of ~50k riders a day. T-1560 waits (Anita).
3. Rest of the world after calibration (Anita 2026-10-27).
Ids handed out through T-1628. T-1520 waits for T-782 to land (both change the market solve).

## Anita's standing asks
- Every demand task (loosely perf) has a "done when"; stop once met. No single-pair demand tuning.
- Performance: approximations welcome when a player can't tell; name them in DECISIONS.md.
- Buses settled: no more tuning unless she asks.
- Thanks to agents in my own reply, not by message; ask after agents quiet 45+ min; final checks
  in the foreground; browser loads count their demand workers in core claims.
- Usage week to ~11-01: Sonnet for settled work, Opus for design, data correctness, unknown bugs.
- Someday tasks are real open tasks.
