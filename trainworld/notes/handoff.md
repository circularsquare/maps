# Orchestrator handoff, trainworld

Orchestrator: session f28006bc-4241-4116-9970-a65fc772bb95, name maps-a2, since 2026-10-27 23:08
Updated: 2026-10-28 (overnight)

Read trainworld/CLAUDE.md and TASKS.md first. History before this session: notes/handoff-2026-10-26.md.

## State
- master 629f285f; earlier 180dc5ec (T-1649 landed: "Keep" in the delete question used to lose the track; Anita
  confirmed the seam gone 2026-10-28); 8800 runs it (smoke 20/20 light and dark, real save 12,483,326). Published is
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
- T-1618 done, committed 131a06a3 (notes/T-1618.md): strokes redrawn every frame = 77% GPU on
  Japan while the clock runs (8% without); paused-and-still is already 2.6%. Fixes T-1638-1641.
- T-1638 done, committed 30ebdeea, on 8800 (smoke both pass): strokes cached in a texture; Japan
  with trains running 77% -> 9-17% GPU; `?perfOff=strokeCache` turns it off.
- T-1639 done, committed 2a397b76, on 8800 (smoke both pass): Japan panning 77% -> 34% GPU
  (MapLibre alone 22%); `?perfOff=strokeLod` / `strokeCull`. T-1644 filed (zoom 15 pan 37%).
- 2026-10-28 morning (Anita): deploy approved and running as 2026-10-28b (log
  artifacts/trainworld/deploy-2026-10-28b.log; then website add/commit "trainsworld"/push,
  verify --page, purge r2 2026-10-28). 2026-10-25b purged. T-1617 closed (transient). Cap 12.
  LIVE: 2026-10-28b (master 2a397b76), verify --page passes; live smoke 19/20 (only T-941/T-1635
  console errors). Website's Pages build needed Ruby 3.3 (ubuntu-latest is 26.04): fixed in
  website bf68979 with Anita's ok. R2 purged: only trainworld/2026-10-28b remains.
- T-1617 reopened: Anita's screenshot (artifacts/trainworld/t1617/anita-seam-tsushima.png) shows a
  dark meridian seam west of Tsushima during "Updating demand"; same Opus agent resumed (Vite 6153,
  2 cores) on the lead: translucent shade polygons split at a meridian, edge antialiased twice.
- Anita 2026-10-28: 10-20% off on lines is fine for ~50% faster loads (ideally the city in view
  re-solved exactly, T-1609); the -23% round skip alone isn't worth it. Order: T-1608, T-1607, T-1609.
- T-1617 update: seam stays after "Updating demand" ends (Anita), so not the loading shade; agent told.
- T-1649 guard bulk track deletes (Anita lost all Japan track by accident): Sonnet agent, main
  tree, Vite 6151, 2 cores, id T-1651. Anita's details: the existing confirmation appeared and
  she's fairly sure she pressed keep; she was viewing California yet Japan's track went; the
  dialog listed every broken line (cap it). Agent told: bug first (keep deleting? selection
  reaching off-view?), then cap, then size guard in the same dialog. T-1650 filed: cut big networks into parts for demand
  (Anita's "cutting points" idea), design after T-1608/T-1607/T-1609.
- Committed 7d9b00ce, on 8800 (smoke both): T-1640 (overlay draws 1/s when nothing visible
  moves; zoomed-out Japan 8% -> 4% GPU), T-1617 fixed (realm shade through the stencil; the seam
  was its own thin slivers at 128.43 E), T-1608 abandoned (exact early stop only 7-9%; T-1647
  filed), notes/perf-overview.md. Not yet published (live is 2026-10-28b = 2a397b76).
- Anita's TODO (2026-10-28 afternoon) filed as T-1654-1665. Running, Sonnet, main tree, 1 core
  each: sounds T-1658/1659/1655 (Vite 6152; knock from her Downloads); Money panel T-1660/1661
  loans/1662 round fares + why steeper past 20 km (6153, may file T-1666); map UI T-1657
  landmarks follow (+T-1621)/1663 crowding outline/1656 save tooltips (6154); clock T-1664
  speed slider/1665 4x cap while demand works + "constructing..." (6155). Held: T-1654 (real load
  estimate; after T-1607, needs Japan loads). Anita keeps deletes undoable (T-1649 call ok).
  Sounds done (uncommitted: map/buildSound.ts, buildVoices.ts, samples.ts, game/world.ts,
  loading.ts, openingDay.ts, ui/Credits.tsx, public/sounds/knock.wav, docs ui-map/code-map,
  TASKS, ARCHIVE). Commit the batch when the other three finish (world.ts may be shared).
  Clock done too (uncommitted: game/clock.ts, game/lineFigures.ts new, main.ts, TopBar.tsx,
  style.css, LinesTab.tsx, LineInspector.tsx, StationInspector.tsx, map/overlay.ts,
  clockClient.ts, docs ui-panels/ui-inspectors, DECISIONS). Anita: speed stays unsaved;
  "constructing..." per city wanted, folded into T-1650 (one cluster solves together today).
- Money done (uncommitted: game/loan.ts, fareDefault.ts new; MoneyTab, MoneyHistory, money.ts,
  regionMoney.ts, world.ts, clock.worker.ts, demand.worker.ts, saveFile.ts, protocol.ts; SPEC,
  docs ui-panels/track-and-service, DECISIONS). Loan 1%/game day (LOAN_RATE); asked Anita.
- Map UI done (uncommitted: attractionMarks.ts, stationLabels.ts, rendererShaders.ts,
  renderer.ts, ui/tip.ts, style.css, three checks t1657/t1663/t1656, docs). T-1621 closed too.
  ALL COMMITTED 629f285f with T-1667 (station keeps its place on extend), on 8800 (smoke both).
  Anita: straight-line fares confirmed; Corners section stays gone; loan 1%/day not objected.
- T-1673 feedback done: shonei-server 6051eed (not pushed/deployed; route POST
  /trainworld/feedback reusing feedback.go + Resend to SHONEI_FEEDBACK_TO; tools/feedback.py).
  trainworld side uncommitted (FeedbackPanel.tsx, SettingsTab.tsx, account.css, account.ts,
  docs ui-panels). shonei-server DEPLOYED 2026-10-28 ~14:50 with Anita's ok (config shows
  trainworldFeedback:true); the game side goes live with the next trainworld deploy. Not seen
  in the log whether feedback mail is configured: test once the game side is live.
- T-1674 [P1] line route runs to the far end and turns round through a branch (Anita's
  screenshot artifacts/trainworld/t1674-anita-route.png): Opus, main tree, Vite 6157, id T-1675.
  Done, committed 62540fe3, on 8800 (smoke both): ties in best_insert now go to the tightest
  stop spread (stop_spread), so branch stops land in route order. Saved lines unchanged.
- DEPLOYING 2026-10-28c from .tw-snap at 62540fe3 (log artifacts/trainworld/deploy-2026-10-28c.log;
  Anita ok). Don't rebuild .tw-snap until the upload ends. Then website push, verify, purge 28b.
- T-1607 landed e588aaf2 (-11% Japan load at 3 workers, exact; ARCHIVE/DECISIONS lines in main
  tree uncommitted). Not on 8800 yet (after the deploy upload).
- Running: T-1676/1677 crowding dashes half thick twice long + Lines list hover (Sonnet, 6152);
  T-1678 double-up "curve too tight" on straight track (Opus, 6157, id T-1681); T-1679/1680
  selected track highlight + junction-to-junction run select (Opus, 6158, id T-1682).
  Anita: keep Japan's lower long fares.
- 2026-10-28c LIVE (verify --page passes); website 8ee048d. Purging 28b and rebuilding 8800 at
  e588aaf2: done, R2 holds only 2026-10-28c, 8800 at e588aaf2 (smoke both pass, 3 workers).
- T-1676/1677 done (uncommitted: renderer.ts, rendererShaders.ts, game/today.ts, LinesTab.tsx,
  LineInspector.tsx, protocol.ts, clock.worker.ts, docs). Commit with the selection and
  double-track work (shared renderer.ts / clock.worker.ts). T-1679/1680 done too (uncommitted:
  blueprintPick.ts, blueprintBox.ts, renderer.ts, Blueprint.tsx, toolKeys.ts, BuildTab.tsx,
  docs, DECISIONS): double-click = run to junctions, shift-click = path, Double up in the bar
  (whole runs only). Waiting on T-1678 and T-1683 to commit the batch.
- T-1683 construct sound from her gun-mag mp3 (Downloads), clipped to start on the click, for
  construct + station upgrades; a refused construct plays only the knock (Sonnet, Vite 6153).
- Committed 4d1a7a48 (T-1670 straight fares: NY $1/$2.60/$9, Japan $0.55/$1.35/$4.75; T-1671/1672
  crowding weekday + hover; T-1673 feedback game side), on 8800, smoke both. Commit message's
  "$1 ... $0.08" got eaten by PowerShell quoting (cosmetic; never amend). Not published:
  everything after 2a397b76. Panels done (uncommitted:
  CommutersInspector.tsx, TrackInspector.tsx, docs). T-1669 removed the whole Corners section
  (add corner, per-corner remove/elevation); asked Anita whether to restore all but the radius.
- Filed T-1670 (default fares $1 + $0.08/km straight), T-1671 (crowding figures always a
  weekday), T-1672 (crowding hover wording): Sonnet, Vite 6153. T-1673 feedback form in
  Settings via shonei-server (store + tool, email if the server can; no deploy; her address
  never in the page): Opus, Vite 6158.
- Filed T-1667 (station blueprint jumps on extend; Opus, Vite 6157, 2 cores), T-1668 (local
  bubble: drop intercity lists) + T-1669 (blueprint track panel: no radius; Sonnet, Vite 6156).
- T-1607 spread the slow worker's crowding rounds, exact, no extra memory (Opus agent, tree
  .tw-trees/t1607, Vite 6170-6174, 5 cores, ids T-1652-1653).
- T-1640 redraw only when something visible moves (Sonnet agent, main tree, Vite 6159, 3
  cores, id T-1646); Anita sees ~5% GPU zoomed out with trains hidden.
- Perf/memory overview for Anita (Opus agent, read-only): notes/perf-overview.md, answering
  where time/memory go, how they scale with cities/stations/lines/trains, areas to improve.
- Anita on T-1607: no speed bought with memory; prefers algorithmic savings like T-782. T-1608
  (cheaper market searches, exact) fits that best.
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
1. Next deploy when Anita says (everything after 45c0dffb). Set core cap back to 12 when she's up.
2. T-1520 (income groups' cost), T-1607 (crowding rounds split across workers), T-1634.
3. T-1374 (superholidays stage 3), T-1605, T-1621 (names follow per frame), T-1609/T-1610.
4. Rest of the world after calibration (Anita 2026-10-27).
Ids handed out through T-1682.

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
