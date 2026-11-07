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
- **profile** a0bea940649aa47c0 (Opus): T-2734 line cross-section in the line inspector (terrain from the hillshade's
  Terrarium tiles, 5 m a level, crossings, stations, 2 km hover lens; low priority, never holds up the panel).
- **loadbar** ac01a9ea182c43905 (Opus): T-2729 bar progress for whole solves, T-2730 Trains column 32 px, T-2737 settings
  rows spacing, T-2739 Cities Profit column, T-2743 crossings chip tint (white/tan/red-brown by cheapest level, $10M),
  T-2744 0.5x cost chip halfway toward 1.0x.
- **l10nlast** ac4f1b469c1b1e554 (Opus lead + its own subagents): last ja/zh round (11 inline strings, 2 missing keys),
  language selector top right under the first row (hover: AI-translated), layout pass, add to OFFERED. Ids T-2740..42.
- **bubbleclick** aad9e4c06a312476e (Opus): T-2746 slow bubble clicks. Cause: a quick second click waits behind the
  previous click's icSel (1-2.7 s) and its own. Approved: demandClient flows/icSel ordering hunks + drop superseded
  icSel; and T-2747 (sim selection.rs zone_row cache) if byte-identical.
- **newgame** a45e2b478492fea41 (Opus): T-2748 medium band +80%, lighter red/green, stable panel top, map-picking lag.
  Its item 4 (difficulty for the new countries) WAITS for worldfix's join: tell it when the join has landed.
- **deleteall** a10dd5e117aa37343 (Opus): T-2738 one delete removes every version, local and cloud; "Delete" red, "Keep".
- **darkterrain** a1b97c22a16cb85d7 (Sonnet): T-2745 three dark terrain variants behind ?debug=1&terrainDark=a|b|c;
  4-up picture artifacts/trainworld/darkterrain/darkterrain-nj-4up.png for Anita to pick; default unchanged.
- **faces** a60e075848e4ece1a (Opus): T-2751 rider faces at each line's own platform.
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
- T-2620 (keep old rates on track >5 km from stations).
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
