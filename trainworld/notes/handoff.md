# Orchestrator handoff, trainworld

Orchestrator: session cd5c271d-f4a2-4f0f-8c6e-5249850f0046, name maps-b9 [971c94], since 2026-11-07 00:05
Updated: 2026-11-07 10:35 (Get-Date). Past 500k tokens: ready to be replaced any time; a new orchestrator takes the
agents below over by SendMessage to their ids (they report to "maps-b9 [971c94]" today; tell them the new name).

Read trainworld/CLAUDE.md and TASKS.md first. Landings are in ARCHIVE.md and `git log`.

## State
- master: last trainworld commit **b2822e6a** (+ isochrone commits after). 8800 = **b2822e6a** built on freeze
  20261107-060902 (snapbuild-frozen.cmd), smoke light/dark 25/26 (the 1 FAIL, t1053-zone-ids, is the main tree's
  registry mid-worldfix, not the build). Real save 12,521,264 rail trips.
- **LIVE is still 2026-11-03b.** Publish of 2026-11-07 (b2822e6a): data on R2 `trainworld/2026-11-07` verified; website
  commit 6445d9b pushed, but GitHub Pages' deploy timed out 3 times (10-min sync limit; githubstatus "operational").
  **Do NOT purge R2 2026-11-03/03b** (live). Website repo has an UNPUSHED commit raising deploy-pages' timeout to 30 min
  (.github/workflows/jekyll.yml). Anita: "we should try to get our other stuff done before pushing again": next
  publish = everything below landed + worldfix's join, rebuild 8800 (plain snapbuild after the join), smoke light/dark,
  `deploy.py --version <new>` (dry-run first), website add/commit, ONE push (her go given for this publish), then
  `deploy.py --verify --page --version <new>`; then purge the old R2 prefixes except the live one.
- Cap **15** while Anita is off the computer (~10:20, "feel free to use more cores"); back to 10 when she's back.
- Uncommitted, mine: passenger grey #a4a19d (map/renderer.ts CROWD_RGB, game/shadeTable.ts light walkers, docs/ui-map.md);
  ROAD_CROSSING_COST 6 -> 10 (sim params.rs, docs/track-and-service.md, docs/ui-building.md; roads tests pass; main
  WASM rebuilt; T-2732's TASKS line says so; archive T-2732 when committed). Also notes/T-2732.md,
  research/upkeep-by-terrain.md (payback's analysis). Commit with the next smoked batch.
- 8800 has everything committed on master (T-2721's ferry fix included).

## In flight ([971c94]'s agents; SendMessage by id; all in the MAIN tree, no commits; each archives its own lines)
- **worldfix** ae45801a1a16dbcc8 (Opus): T-2702 coastal units + T-2712 Colombia blobs. All 53 rows gated; **JOIN running**
  (BR,MX,CO,CN,KR,VN,MY,BN; --jobs 5; log artifacts/trainworld/t2702-join.log; ~3-5 h from ~10:00). **Hold data/packs and
  data/terrain writers** until it reports. It runs zone_successors/t1053 before hand-back. Don't commit registry.json,
  market-rows.json, world_plan.json until it reports. Backup data/work/pre-T2702/pre (5.5 GB) to delete once committed.
  Also found+fixed brazil.unit_states (units across a state line got no census tracts).
- profile's revision DONE (scale lines removed, 0 m dashed rule, water band 4 m deep from masks/sea; smoke 26/26).
  New ja/zh keys unreviewed (a small review pass later). Picture review/cross-section-7-river.png. Delete strings
  approved by Anita.
- profile T-2734 (archived; smoke 26/26), UNCOMMITTED: ui/inspector/LineProfile.tsx, workers/lineProfile.ts,
  game/groundHeights.ts (new); clock.worker.ts, protocol.ts, LineInspector.tsx, map/basemap.ts, en.json, context.json;
  docs ui-inspectors.md, code-map.md; DECISIONS, ARCHIVE, TASKS, notes/T-2734.md. Strings shown to Anita. Pictures
  review/cross-section-nlr.png, cross-section-tohoku.png. Unfiled follow-ups: ground includes buildings in dense
  centres, click strip to move map, hover covers the panel above.
- loadbar DONE T-2729/2730/2737/2739/2743/2744 (archived; smoke 26/26, t2552 pass), UNCOMMITTED: demand.worker.ts
  (Work class + a DEV-only console.info "T2729"), demandProtocol.ts, demandClient.ts, game/demand.ts, StatusTag.tsx,
  ui/style.css, trackTagRows.ts, docs ui-panels.md, ui-building.md, DECISIONS, ARCHIVE, TASKS. Pictures
  review/chips-dark.png, status-bar-mid.png.
- l10nlast DONE (T-2584/2585/2643 archived): ja, zh-Hans, zh-TW in OFFERED, 2174/2174 each, 0 inline strings; clock
  width per language; smoke 26/26 and 26/26 ?lang=ja. UNCOMMITTED (files listed in its report: diagram/*, map/tools.ts,
  i18n/*, ui/style.css, checks/i18n.mjs, l10n-shots.mjs, docs). It kept the Settings Language row as the selector
  (Anita confirmed the Settings row). T-2753 DONE too: start menu language picker (ui/LanguageSelect.tsx new, shared
  with Settings; StartMenu.tsx keeps deleteall's edits; account button moved into the New game row), UNCOMMITTED.
- **bubbleclick** aad9e4c06a312476e (Opus): T-2746 slow bubble clicks. Cause: a quick second click waits behind the
  previous click's icSel (1-2.7 s) and its own. Approved: demandClient flows/icSel ordering hunks + drop superseded
  icSel; and T-2747 (sim selection.rs zone_row cache) if byte-identical.
- newgame a45e2b478492fea41: T-2748 items 1,2,3,5 built (smoke 26/26), UNCOMMITTED: game/palette.ts, map/intercityView.ts
  (shared with bubbleclick's edits), map/realNetwork.ts (real network from z6 not z4), perfFlags.ts, ui/NewGame.tsx,
  ui/style.css, pipeline/start_difficulty.py, docs zones.md, ui-demand-views.md, ui-map.md, DECISIONS, TASKS, notes/T-2748.md.
  Filed T-2749, T-2750. **Item 4 after the join: resume it** (`pipeline\start_difficulty.py --jobs 2`, ~10-12 min,
  writes app/src/game/start-difficulty.json; then it fixes zones.md cut numbers). Until then the game shows old thirds.
- deleteall DONE T-2738 (archived; t2738-delete.mjs 7/7, smoke pass), UNCOMMITTED: game/accountGames.ts, workers/cloud.ts,
  workers/saveFile.ts, ui/SavesPanel.tsx, ui/StartMenu.tsx, checks/t2738-delete.mjs, checks/t988-cloud.mjs, en.json,
  context.json, messages/*, sources/*, docs ui-panels.md, code-map.md, DECISIONS, ARCHIVE, TASKS. Strings shown to
  Anita; 5 keys passed to l10nlast for review. Cause: accountList not updated on local delete -> ghost cloud-only card.
- T-2745 DONE: Anita picked B; WASH.dark = rgb 190,178,164 (switch removed), archived. UNCOMMITTED: map/terrainTint.ts.
- Anita: keep going (this orchestrator) until this batch ends, then cycle.
- faces DONE T-2751 (archived; smoke 27/27), UNCOMMITTED: map/stationLife.ts, map/faces.ts, docs/ui-map.md, ARCHIVE,
  TASKS. Picture review/faces-grandcentral-after.png. Limit: lanes ~8 m apart, faces can overlap the next lane at z16-17.
- Landing plan: as each reports, check its strings and pictures (copy the best to artifacts/trainworld/review/ and show
  Anita), then one smoke of the main tree on my own Vite (dev-nowatch, TW_CORES=4) and commit by path in batches.
- Next unassigned id **T-2753**.

## Landed today (maps-b9 [971c94]), all on master
- Overnight: T-2630 checks pin Chrome; T-2703/2705/2704/2710 failed loads retry or stop with a reason; T-2682 edit note;
  T-2631 smoke catches a refused market pack (26 checks); T-2691 "Nn" = MapLibre AJAXError; T-2637; T-2588/T-2718 NA cache
  hits on reload; T-2699/2700/2716 (f9142979) sim heap + boarding 311 -> ~4 ms; T-2707/2722 boarding memory 1,056 ->
  257 MB; T-2681/2720 city names ja/zh (rerun tools/place_names.mjs after a join); T-2625 continent diagram
  condensation; T-2695/6/7 15 new countries (8ca54120); research/post-framing.md.
- Morning: T-2721 trip explorer ferry = sim's; diagram borders 2x wide, 18% (b9fd8ca7); T-2728 no status ring, T-2731
  date without the year (b2822e6a).

## Waiting on Anita
- Two strings (shown): clockClient.storageUnreadable, startMenu.savesUnreadable (no answer = fine).
- darkterrain's 4-up when ready; T-2724 (diagram far land smaller) after her look at T-2625 (she said "looks ok").
- (T-2620 abandoned by Anita 2026-11-07: upkeep stays as it is, "simple and interpretable".)
- T-2667 OSM attractions: after release. T-1856 hex terrain: after release (Anita).

## Next
1. Land agents as they report (above); then, after worldfix: rerun place_names.mjs, tell newgame to run difficulty for
   the new countries, smoke, commit, rebuild 8800, smoke light/dark, PUBLISH (State).
2. Queued: T-2713, T-2714 (census totals for the new countries), T-2698 (start cameras; registry, not during a join),
   T-2627, T-2638 (later), T-2593, T-2592, T-2701 (big: design to Anita first), T-2715 (leave unless seen).

## Noticed by capview, not filed yet (worth small tasks)
- A transfer-fare edit (setFareTransfer 0 -> 0.5) right after a stop-drop's long update never started a solve within 60 s.
- linesPending cleared on any demand update end (T-1733): T-2658 replaced the blanket clear; check it is gone.

## How to
- My scratch = C:\Users\anita\AppData\Local\Temp\claude\dce1fefcc1d43aef\cd5c271d-f4a2-4f0f-8c6e-5249850f0046\scratchpad:
  snapbuild.cmd <hash>, snapbuild-frozen.cmd <hash> <freeze> (packs from a freeze while a run owns data/packs),
  smoke8800.cmd (light+dark, TW_CORES=2 now), opencc\ (opencc-js). Run via PowerShell:
  `cmd /c "<scratch>\snapbuild.cmd <hash> > <scratch>\snapbuild.log 2>&1"`. The old orchestrator's scratch
  (...\b35f881f-...\scratchpad) has merge3json.py, resolve_*.py, translate/back/review briefs, the worldnext kit.
- Smoke dev: own Vite via `node trainworld/checks/dev-nowatch.mjs <port>` (restart after app or WASM changes), then
  `$env:TW_CORES='4'; node trainworld\checks\smoke.mjs http://localhost:<port>/` (~3.5 min).
- Publish: notes/T-924.md "Commands"; `venv\Scripts\python.exe trainworld\tools\deploy.py --dry-run --version X`, then
  `--version X --smoked` (~15 min upload), website `git add trainsworld/ trainworld/`, commit, push; Pages deploy can be
  slow (`gh run list --repo circularsquare/circularsquare.github.io`), then `--verify --page --version X`.
- Build number = `git rev-list --count`. If 8800 does not answer, restart serve.py detached.
- Landing a tree: `node trainworld/checks/agent-tree.mjs land <name> "<sentence>"`, then `remove <name>`.
- Commit by path: `git commit -m "..." -- <paths>`; `git add` a new file first.

## Lessons from this run
- Always Get-Date before writing a time: this session's own estimates ran ~2 h ahead by mid-morning.
- Two sessions can share a name (both maps-b9): message by ref.
- A world run's gates serve the main tree's app/ live: hold app/ writes (not just commits) during gates; commits,
  .tw-snap rebuilds and freeze-based publishes are safe.
- Agents sharing the main tree work well when each brief lists the others' files; commit app work in one smoked batch.
- T-2716: inline(never) on GlobalAlloc methods; a debug-build abort can be an LLVM fold, not corruption.
- Copying files back after a source swap keeps old mtimes: cargo/wasm-pack skip the rebuild (touch them).
- Agents that leave a background wait re-notify: TaskStop them after their report.

## Anita's standing asks
- Player text: agents propose wording per the style rules, list every string old -> new; land without waiting; show
  her. Zone, region and realm names need no review.
- TODO.md short-term = prompts: copy her words verbatim into briefs.
- Every demand task has a "done when"; demand changes go to her first. Buses settled. Keep the work moving.
- Disk: delete orphaned temp files; keep only current and latest published builds.
