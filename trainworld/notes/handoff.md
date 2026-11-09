# Orchestrator handoff, trainworld

## Candidate ready for Anita, 2026-11-08 23:51 America/New_York

Anita: "lets also try to keep changes conservative in general. we are mostly just looking to fix bugs this round. make sure that the bugs are real, and we don't introduce any new issues. additional features are not a priority, though we should also add the changelog / update indicator thing. i will test before we deploy"

**HOLD DEPLOY until Anita has tested the candidate.** This supersedes all earlier publish-after-round go below.
She selected currencies, update/changelog, tighter curves and bug fixes; NO line money history or trip-purpose graph.
Those two implementations remain preserved in c18d2fa1 and b167196a and are held tasks T-2787/T-2790.

Current code/candidate: **917c738a**. Main batch c18d2fa1, mobile 309ff65b, scope reduction 967be7b6,
curve/deletion 917c738a. Tripkinds b167196a was landed then reverted by the scope reduction; no purpose export/UI remains.
Main WASM rebuilt. Snapshot built separately, local deploy preflight PASS (3,785 zones, 22,839 data files, no warnings).
**8800 now serves 917c738a**. Previous local build retained at `.tw-snap/trainworld/app/dist-before-codex-aac7d7e6/`.
No website change, R2 upload/purge, push, or production deploy. Production remains 2026-11-08 / 4f1b6fc8 per prior handoff.

Checks: integrated dev smoke **26/26 PASS** (12,521,264 NYC trips/day); currency **31/31 PASS**;
TypeScript and localization pass. Update UI visibility/open/dismiss/reload/2-day expiry pass; no held features in its list.
Graphics-loss/memory notice injection passes. Emulated phone tutorial controls pass without crash/console errors;
short test line reaches 0.5% of the final 1% goal, so real phone/full tutorial validation stays open.
Original history reload check passed 5/5, then that feature was removed per Anita's scope.
Rust tests **368 track + 544 demand passed**; production smoke **26/26 light + 26/26 dark PASS** on 8800.
North America fare replay (158 lines, read-only Downloads save day-476) PASSED dropStop and openZone, both
actually settled with no pending lines. Daily fares stayed >=99.68% in the stop edit and 100% on opening a zone;
no >10% transient dips or browser errors. T-2793/T-2773 remain open for the broader complaint/rough-path limits.
All test processes and owned temporary servers ended; the CPU ledger has no claims. No more checks are running.

Logs/artifacts: `artifacts/trainworld/codex-candidate-{dev,light,dark}-smoke.log`,
`codex-candidate-rust-{track,demand}.log`, `codex-candidate-build.log`, and
`artifacts/trainworld/codex-takeover/{integrated-fares.log,mobile-update-suite.log,text-review.md,changelog-review.png}`.
Original pre-takeover files are backed up in `artifacts/trainworld/codex-takeover/original-work.zip`.
Exact changed English strings: `artifacts/trainworld/codex-takeover/text-review.md`; preview image:
`artifacts/trainworld/review/changelog.png`. User TODO.md was never edited or staged by Codex.
The update timestamp in `ui/updates.ts` is still the preview's old 2026-11-08T22:00Z; **bump at the actual publish**.
Do not publish before Anita tests; do not claim real-phone crash resolution or all fare-drop paths solved.

## Earlier Claude handoff (historical context)

Orchestrator: session 70947ac4-74ea-441a-b6e5-c704eedaf536, name maps-68 [224d6e], since 2026-11-07 12:32
Updated: 2026-11-07 12:37 (Get-Date; maps-b9's clock ran ~15 min ahead). Anita 12:37: at the computer but idle,
keep cap 15; ja/zh review is ours (she reads neither). l10nreview DONE: 17 keys x 3 languages reviewed, one fix,
UNCOMMITTED: zh-TW.json diagramView.unnamed 軌道交通 -> 軌道運輸 (commit with the next batch); i18n.mjs and tsc pass.
tokyoload interim: no regression on a fresh page, even throttled; publish need not wait; it runs a 20-min session next.
2026-11-07b upload DONE (R2 verified 3/3, maps-b9). Website commit 57c68d2 PUSHED (with 8ec33bb, Pages 30-min timeout;
Anita's go 2026-11-07). Pages run 42398866884 succeeded; verify --page 6/6 PASS. **LIVE = 2026-11-07b.** R2 purged
2026-11-03, 03b and 07; only 2026-11-07b left on R2. Publish complete.
loadbar DONE T-2638 (archived; smoke 26/26, t2552 pass; no strings): bars shown as fraction^1.7. UNCOMMITTED:
game/loadBarCore.ts, game/demandClient.ts, checks/t2552-load-bar.mjs, docs/ui-panels.md, DECISIONS, ARCHIVE, TASKS,
notes/T-2638.md. Commit with the next batch.
tokyoload DONE T-2756 (smoke 26/26; maybe not archived yet, check TASKS at commit): cause = memCheck restarts all demand
workers in a long-open window, re-solve blocks a returning zone's cells. Fix: cache focus cells + last 6 zones' legs/bubbles.
UNCOMMITTED: workers/demandClient.ts (shared with loadbar), map/commuterView.ts, ui/StatusTag.tsx, game/loading.ts,
i18n messages/sources/context.json, docs/ui-panels.md, TASKS, DECISIONS, notes/T-2756.md. Filed T-2757 [P2] (other
questions still wait on the re-solve). Strings approved by Anita, with her rewording applied by maps-68
("Recomputing after freeing memory", "Internet connection seems slow"; ja/zh unchanged, already fit). Next free id T-2758.
**Anita posts Sunday 2026-11-08 morning.** Before then: worldfix lands -> commit batch, rebuild 8800, smoke, republish.
firstvisit DONE T-2758 (archived): live site cold loads playable in 6.5 s unthrottled, 17-18 s at 20 Mbit, ~45 s on
slow 4G from page open; no errors, no stalls. Minor: glyph 404s (expected), nyc.bin fetched twice on slow 4G (unknown),
outlines.json 4.55 MB already gzipped. notes/T-2758.md. Nothing blocks posting.
Anita will wait for worldfix before posting. flicker DONE T-2760 (archived; smoke 26/26; no strings): road tiles
landing re-ran RealSnap.again() and blanked the tag; now roadsLanded() re-prices in place. 32 flips/5 s -> 0. UNCOMMITTED:
map/realSnap.ts, map/tools.ts, checks/t2760-snap-flicker.mjs, TASKS, ARCHIVE. Land in the post batch. T-2761 (crossing-tracks highlight) filed P3, after the post.
sparse DONE T-2747 (archived; byte-identical NYC+NA, cargo 367+543, smoke 26/26; no strings): far zones packed by
gid; NYC rapid-click p95 1089 -> 194 ms, NA 1670 -> 279. In tree .tw-trees/sparse, NOT landed: **land only after
worldfix's join ends** (WASM rebuild mid-join can park units, T-2298): agent-tree land sparse, remove, rebuild main WASM,
then smoke the whole batch. Follow-up T-2762 filed. Next free id T-2766.
Anita's TODO 2026-11-07 afternoon, filed: editlag acafed71623273ce1 (Sonnet, T-2763 station/delete/bonk latency, Vite
6100-6149, no Rust) and linecolour a16b82c008c7452cc (Sonnet, T-2764 same-colour lines + T-2765 endpoint tangent, Vite
6150-6199). Both main tree, may share map/tools.ts. Land in the post batch if done in time.
linecolour DONE T-2764, T-2765 (archived; smoke 26/26; no strings). UNCOMMITTED: game/lineActions.ts, map/curveCircle.ts,
map/nodeDrag.ts, map/tools.ts, checks/t2764-line-colour-quick.mjs, docs ui-map.md, ui-building.md. editlag DONE T-2763 (archived; smoke 26/26; no strings): station ghost stays until the worker answers, blueprint
deletes hidden at click (renderer hideEdge/u_hide), worker skips trips/boarding for a bare blueprint removeEdge. Visible
in 0-3 ms. UNCOMMITTED: map/tools.ts, map/renderer.ts, map/rendererShaders.ts, game/remove.ts, main.ts,
workers/clock.worker.ts, checks/t2763-edit-lag.mjs, docs/ui-building.md, notes/T-2763.md. Proposals filed T-2766 (P2 stop
edits pipelined, likely the bonk), T-2767, T-2768.
uibits DONE T-2769..2772 (archived; smoke 26/26; no strings). UNCOMMITTED: ui/inspector/LineProfile.tsx,
map/intercityView.ts, ui/inspector/CityInspector.tsx, map/labelNames.ts, map/stationLabels.ts, ui/style.css, ui/tip.ts,
game/today.ts, ui/schedule.tsx, ui/LinesTab.tsx (crowdLevel ends), docs/ui-inspectors.md. T-2772 = the crowding
hover's bar (Anita confirmed). Picture review/load-ends.png.
Anita's string: linesTab.pendingTip + schedule.recalculating "Calculating riders after your change" -> "Calculating rides…"
riderdip ended T-2773 (open, see TASKS): settled-figure hold built, smoke 26/26, UNVERIFIED (no dip on NYC). UNCOMMITTED:
game/today.ts, workers/clockClient.ts, checks/t2773-pending-dip.mjs, DECISIONS. Ships in the batch; Anita retries on her save. Next free id T-2774.
(maps-68 edited en, sources, ja 乗車を計算中…, zh-Hans 正在计算乘车…, zh-TW 正在計算搭乘…; i18n check passes). UNCOMMITTED. Next free id T-2773.
linecolour follow-up DONE:
nextLineName counts names handed out in the last 5 s (check passes; smoke not rerun, the batch smoke covers it). Picture review/tangent-drag.png. T-2332 still open.
worldfix join at 13:51: Brazil sudeste's intercity solve took 5389 s, still in that group.
T-2546 (video) abandoned by Anita. T-2757 skipped for now (tokens). Next free id T-2759.
maps-b9 [971c94] wound down; its agents (tokyoload, loadbar, sparse, worldfix) report to
maps-68 now. The "Publish in progress" section below is the current state; the State bullets are from 10:35 and partly
stale (everything listed as uncommitted there is in de29a14c). Exception to "wound down": the deploy.py 2026-11-07b upload
runs in maps-b9's tab (log in its scratch, deploy2.log); maps-b9 messages maps-68 when it ends, then maps-68 does the
website commit/push (after tokyoload's answer), verify and R2 purge. Anita's new TODO item (drop the "Loading the bubbles
for Tokyo" hover, no "bubbles") was passed verbatim to tokyoload. Open for Anita: profile's new cross-section ja/zh keys
need a short review pass.

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

## Publish in progress (~12:40, Anita: publish now without waiting for worldfix)
- Committed de29a14c (everything but worldfix's files). 8800 = de29a14c on freeze 20261107-060902, smoke light/dark
  26/26, 12,521,264. deploy.py --version 2026-11-07b --smoked running (task, log scratch deploy2.log). Then website
  `git add trainsworld/ trainworld/`, commit, push (with the unpushed 30-min timeout commit), watch the Pages run,
  `deploy.py --verify --page --version 2026-11-07b`. Then purge R2 2026-11-03, 2026-11-03b, 2026-11-07.
- bubbleclick DONE T-2746 (archived; rapid-click p95 4.8 s -> 1.2 s on NYC); T-2747 sparse icSel filed (P1, own tree).
- **sparse** a5e09f5724cd0f3ea (Opus, tree .tw-trees/sparse from de29a14c, Vite 6193, 3 cores): T-2747 sparse icSel
  answers (WASM interface change). Land with agent-tree land sparse, then rebuild main WASM. May file T-2756.

- **HOLD the website push**: Anita saw Tokyo's "Loading the bubbles" stall >4 min on 8800 (de29a14c) after leaving and
  returning. **tokyoload** (Opus, main tree, Vite 6194, T-2756) checks whether it's a regression from de29a14c (vs
  b2822e6a) and fixes it; also "say why in the tooltip" when a stage runs long. Push only after its answer.
  Likely cause: our own R2 upload saturating her Wi-Fi (16.6 Mbps up), slowing tile/terrain downloads; tokyoload tests
  with throttled network and adds a "connection is slow" hover. Fresh reload loaded in ~40 s.
- **loadbar** ac01a9ea182c43905 resumed on T-2638 (reweight loading bars: early stages less, time-left estimate).
- **tokyoload** a92dc3a2c8eb3d255 (Opus, main tree, Vite 6194): T-2756.
- Next unassigned id **T-2757** (tokyoload may use it).

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
- deploy.py's rclone (32 transfers) saturated Anita's Wi-Fi upload (16.6 Mbps) and slowed her game's tile and terrain
  downloads on 8800 (2026-11-07). Anita: no need to cap it, just confirm the cause (tokyoload is checking).
- Always Get-Date before writing a time: this session's own estimates ran ~2 h ahead by mid-morning.
- Two sessions can share a name (both maps-b9): message by ref.
- A world run's gates serve the main tree's app/ live: hold app/ writes (not just commits) during gates; commits,
  .tw-snap rebuilds and freeze-based publishes are safe.
- Agents sharing the main tree work well when each brief lists the others' files; commit app work in one smoked batch.
- T-2716: inline(never) on GlobalAlloc methods; a debug-build abort can be an LLVM fold, not corruption.
- Copying files back after a source swap keeps old mtimes: cargo/wasm-pack skip the rebuild (touch them).
- Agents that leave a background wait re-notify: TaskStop them after their report.

## Wrap-up plan (Anita ~16:10: 4% weekly limit left; post Sunday morning)
worldfix: join ended 17:43, 53/53 shipped; market step failed (thl zone count), fixed in pipeline/market_zones.py, redoing
register --market --trips (pid 26856, log artifacts/trainworld/t2702-market.log), then checks; still hold pack writers.
Anita (back, ~18:00): "fine to publish and push" -> GO given for 2026-11-07c deploy + website push once landed.
video2 (Opus, T-2774, Anita asked for Opus; Sonnet "video" stopped): launch video from her day-473 save, Penn Station
8th Ave/33rd St; relay feedback by SendMessage "video2". DONE (not archived; T-2774 open until Anita records): ?video=1 recording mode,
V runs / Esc resets, &at=lat,lng,z &hold &secs &end. UNCOMMITTED: app/src/video.tsx (new), game/ui.ts, main.ts; deleted
videoLogo.tsx. URL after rebuild: localhost:8800/trainworld/?continue=1&video=1&at=40.7507,-73.9948,15.8. Frames p50 16.7 ms.
Pictures review/video-start.png, video-end.png.
~20:05 worldfix: the 17:49 panic was a partial retry; the full retry (t2702-market2.log) succeeded, stale 0, gaps 0,
3,722 shipped; terrain + market-drives done. Doing 8 metro load checks + smoke (~15-20 min), then RE-FREEZE and hand back.
~20:40 worldfix HANDED BACK (T-2702/T-2712 open only for load checks): publish from freeze data/frozen/20261107-202835.
Its files: pipeline/admin_totals.py, world_plan.py, brazil.py, market_zones.py, world_plan.json, tools/country/colombia_census.py,
checks/state_gaps.py, app/src/game/zone-names.json, registry.json, market-rows.json, zone-aliases.json, data/packs outputs,
notes/T-2702.md, TASKS, DECISIONS, docs adding-a-country.md, data-register.md, code-map.md. Delete after commit:
data/work/pre-T2702/ (5.2 GB), data/work/worldplan/state.pre-T2702.pkl. Retrospective in notes/T-2702.md; T-2775..2777 filed.
**Load checks of 8 new metros + control moe time out at "opening its pack" under the uncommitted app edits** (smoke passes).
~21:20: landing DONE (load checks 9/9 on a fresh Vite: the timeouts were worldfix's stale no-watch Vite; sparse landed
e941d196; difficulty + place names run; smoke 26/26). COMMITTED 094a132a (everything but TODO.md). 8800 = 094a132a,
smoke light/dark 26/26 (12,521,264). **LIVE = 2026-11-07c (094a132a)**, website 7fcc3b5, verify --page 6/6 PASS. R2 holds only 2026-11-07c (07b purged);
pre-T2702 backups deleted. Anita posts Sunday morning; video recording via ?video=1 on 8800 (T-2774 open until she
records). Open for her: T-2773 retry on her save. ~22:10 Anita: video ok; video2 resumed for a slow globe spin ~1 zoom before the end (&spin) and a look at dropped
frames at state zoom. Start URL now &at=40.75077,-73.99361,15.8 (100 m E, 8 m N). video2 round 2 DONE, committed 8ae3670e (video.tsx only: spin, fade 0.9 s from -0.5 s, &pace), 8800 rebuilt to it
(not published; only matters under ?video=1). Anita recorded; the state-scale hitch stays even at pace=2 (T-2778 P3).
Cropped her pick: C:\Users\anita\Videos\Captures\trainsworldchosen-cropped.mp4 (first 17 s cut, crop 1536x952 at 192,88). (Earlier: resumed at full speed on Anita's word ~21:40) (paused ~21:30 for her internet). If paused again, resume with
env RCLONE_BWLIMIT=1M (or lower), RCLONE_TRANSFERS=8, `deploy.py --version 2026-11-07c --smoked --resume`
(log my scratch deploy3.log); next deploy 2026-11-07c, push, verify, purge 07b, then
delete data/work/pre-T2702/ and state.pre-T2702.pkl.
(was) landing (Opus): diagnose+fix that, land sparse, place_names, start_difficulty, load checks, smoke; then I commit, snapbuild-
frozen with 20261107-202835, smoke 8800 light/dark, deploy 2026-11-07c, push (Anita's go given).
Asked worldfix for a retrospective in notes/T-2702.md (Anita: why ~5 h over); ids T-2775..2777 reserved for it.
Publish from that re-freeze (the 18:30 freeze 20261107-183057 lacks market-drives.bin and 53 terrain files).
(Earlier: Anita 16:25 out for a walk; deploy held until her word, now given.) Subagents
(Sonnet) are fine for landing ("in general its efficient to use subagents").
After worldfix reports, one Sonnet landing agent: land sparse (agent-tree land/remove, rebuild main WASM), run
start_difficulty.py --jobs 2 and tools/place_names.mjs, one smoke on own Vite; then orchestrator commits by path,
snapbuild 8800, smoke light/dark. On her word: deploy.py --dry-run, --version 2026-11-07c --smoked, website commit+push,
verify --page, purge 2026-11-07b.

## 2026-11-08 (Anita has posted; live 2026-11-07c; weekly tokens nearly out)
quickbits (Sonnet, T-2779..2782: cross section west-left, language select spacing/size, leaderboard points hover, crowding
bubble average), main tree, no commit yet. T-2783..2790 filed for later (leaderboard not updating T-2783 is the one
that matters most now that it's public). Next free id T-2791. Deploying needs her word.
Anita later 11-08: "lets do the rest of the changes too now". Sonnet agents, main tree: board (T-2783 + hover key), rename
(T-2784, 2785, 2786, 2791), extras (T-2789, 2788, 2787, 2790; may skip the expensive ones). After them: smoke, commit by
path (not TODO.md), rebuild 8800, ask her before publishing. T-2779..2782 are committed, on 8800, not live.
Next free id T-2793. Batch landed. Server deployed with 1e12 limit (shonei-server commit, unpushed). Anita: "lets deploy" (go for
this publish, version 2026-11-08): 8800 = 4f1b6fc8, smoke light/dark 26/26; **LIVE = 2026-11-08 (4f1b6fc8), website 7352eae, verify --page 6/6 PASS.** Purge R2 2026-11-07c after 2026-11-10 or so.
(done: deploy, website commit+push,
verify --page. **Do NOT purge R2 2026-11-07c the same day** (players are live now: open tabs keep reading their own version's data; wait a day or two).

## 2026-11-08 evening
Anita's X account locked (birth date set; under 13 at creation; ID submitted), so traffic from the thread stopped. 2,600 plays,
1,863 browsers on 11-08. Her complaints filed T-2793..2798, T-2793 and T-2794 wait for her next usage window; agent easy (Sonnet, ~40 calls max) on T-2796, 2795, 2798, 2797 (2% left).
Then smoke, commit, 8800; publish only on her word. Next free id T-2799.

8800 = aac7d7e6 (T-2796, T-2795, smoke 26/26), not live. UNCOMMITTED: app/index.html (data-nosnippet on #game, longer meta
description, so Google stops quoting the EuroGeographics notice); ships with the next batch.

## 2026-11-08 night round (Anita: "continue on some work without worrying about credit usage... if we get interrupted we can juts resume tomorrow"; hold publish until the whole round is done, tomorrow is fine)
Agents (each ends with its own archive; ids of follow-ups in parens): fareszero T-2793 Opus main tree (T-2800, 2801); mobile T-2794
Opus tree .tw-trees/mobile (T-2802-4); tripkinds T-2790 Opus tree .tw-trees/tripkinds, Rust (T-2805); curve T-2799 + T-2797 Opus tree
.tw-trees/curve (T-2806-7); currency T-2798 Sonnet main tree (T-2808); linemoney T-2787 Sonnet main tree (T-2809). Next free id T-2810.
Land: smoke each tree, `agent-tree.mjs land <name> "<sentence>"`, remove, main WASM rebuild if sim changed; commit main tree by path
(not TODO.md); snapbuild 8800, smoke light/dark. Publish only after the whole round, on her word (she's given it for after the round).
linemoney DONE T-2787 (archived; smoke 26/26): rail lines only, 8 days of fares and running cost, UNCOMMITTED in main tree: clock.worker.ts,
saveFile.ts, protocol.ts, ui/inspector/LineMoney.tsx (new), LineInspector.tsx, i18n, docs. Strings lineMoney.* not yet shown... shown 11-08 night.
tripkinds DONE T-2790 (archived; smoke 26/26, cargo 367+543; strings "Mode", "Purpose" shown): in tree tripkinds, NOT landed: land after
fareszero and currency finish (the land rebuilds main WASM under their running pages).
currency DONE T-2798 (archived; check t2798 passes; smoke 25/26, demand final timed out at 1 core: rerun in the batch smoke). UNCOMMITTED, main
tree: game/currency.ts (new), format.ts, ui.ts, FareGraph, MoneyTab, SettingsTab, ToolBar, trackPrice, TopBar, EditNote, Cities, tripText, MarketTrip,
TripInspector, AllCitiesView, CityDetails, style.css, i18n, checks/t2798-currency.mjs, docs. Strings shown to Anita 11-08 night.
curve DONE T-2799 (min radius 50 m, real-track snap keeps 100 m) + T-2797 (a station's end piece "removed" and re-laid: delete tool now takes the station,
refusals say why, Delete/Backspace keys work), cargo 368+544, smoke 26/26, in tree curve, NOT landed; land after fareszero with tripkinds, rebuild main WASM.
mobile first pass in tree mobile (phone load 950-1050 -> 610-840 MB settled, lost-context notice, tutorial tap fix; smoke 26/26), resumed on T-2802
(first line ~1,010 MB) + ja/zh. T-2802 pass DONE in tree mobile (first line 640-650 MB low-end; phones open no market = no trips between zones; demandSums NaN fix;
smoke 26/26); not landed, T-2794/T-2802 stay open. Land all three trees after fareszero.
Phone note wording set by Anita in tree mobile ("Intercity trips cannot be computed on mobile. Please try using a computer for bigger games."; ja/zh not re-translated).
Added to the round: changelog (Opus, T-2810 start-menu update row + changelog, main tree), boards (Sonnet, T-2811 points wording, T-2812 Marey crowding, T-2813 leaderboard city picker, main tree).
Next free id T-2814. Publish after: fareszero, changelog, boards, landing of trees mobile, tripkinds, curve; her go was "yes publish after the round" (she said "before we publish lets also...").
Not in the round: T-2796's ping (needs an outside health check), T-2757, T-2766.

## Anita's standing asks
- **Token-conservative (Anita 2026-11-07 ~13:00: "we're running close to weekly limit. lets be token conservative and
  lean to use sonnets going forward")**: new agents on Sonnet unless a task truly needs Opus; fewer, smaller agents;
  let the running Opus agents (worldfix, sparse, tokyoload) finish rather than restart them.
- Player text: agents propose wording per the style rules, list every string old -> new; land without waiting; show
  her. Zone, region and realm names need no review.
- TODO.md short-term = prompts: copy her words verbatim into briefs.
- Every demand task has a "done when"; demand changes go to her first. Buses settled. Keep the work moving.
- Disk: delete orphaned temp files; keep only current and latest published builds.
