# Orchestrator handoff, trainworld

Orchestrator: session cd5c271d-f4a2-4f0f-8c6e-5249850f0046, name maps-b9 [971c94], since 2026-11-07 00:05
Updated: 2026-11-07 00:05 (Get-Date). Both sessions are named maps-b9: message by ref ([971c94] new, [3e1b28] old).

Read trainworld/CLAUDE.md and TASKS.md first. Landings are in ARCHIVE.md and `git log`.

## State
- master **db6fc554** = 8800 **build 1315** (~23:40; dev smoke + 8800 light/dark 25/25). Has attractions T-2666 (WASM
  rebuilt), the load-failure work (reason on screen, WASM start retries), bunny back on the loading card, 13 px ring.
- Anita's NA save failed to load once on 1303 then loaded fine twice; cause not found (loadfail's report: likely a
  dropped .wasm fetch, now retried, or a load during the 8800 swap). Next failure names itself (`.load-why`).
- **Overnight runs started 00:03, cap 14** (Anita in bed). worldnext's three detached runs (python PIDs 8872 latam,
  9644 za, 34308 gulf; 11 cores), state in notes/T-2695.md "Runs" (`world_run.py status`). worldnext lives inside the
  old session's tab (keep it open) but reports to maps-b9 [971c94]. While its gates run: NO app/ or WASM landings;
  before its join it messages: hold pack writers.
- Uncommitted, worldnext's: app/src/game/realms.json, zone-names.json, pipeline/world_plan.json, world_plan.py,
  docs/world-run.md, notes/T-2695.md (it commits nothing; the orchestrator commits them when the runs ship).
- Cap back to 10 when Anita is up. LIVE = 821. Publish only on Anita's word. The upkeep tree still needs `agent-tree.mjs remove upkeep --force`.

## Landed this session (maps-b9)
- translations units 1-4; T-2656 + play count 5a71e6f6; T-2634 2c04f007; T-2641 75e04c44; T-2657 838c46e7; T-2635
  2c10069e; T-2640 093d7b82; borders2 e67d966d; 7e8d913e (huninn.css); T-2619 1c9b3c72; T-2661 18c10362; T-2664/T-2673
  75ea5b27; T-2660/T-2665 3d8cf723; T-2653 2776067f; T-2655 4da69460; T-2622 51d5e03f; T-2652 731614e5; T-2658 5a4aef70;
  about links cf923482 (website-58's relay); fares option N 08133e44 (T-2669, T-2670, T-2676); T-2678 2921b335 (ridersnow
  retired); pass 5 wording 1f5de214 (l10nshots: 15 fixes; shots in artifacts/trainworld/l10nshots/).

## In flight
- **worldnext** (old session's subagent, Opus, MAIN tree): T-2695 Latin America, T-2696 South Africa, T-2697 Gulf; 244
  rows (3 held); runs started 00:03, reports to [971c94].
- storeread finished T-2703 (archived; check t2703-storeread.mjs PASS, smoke 25/25), **UNCOMMITTED, commit after
  worldnext's gates end**: app/src/workers/saveFile.ts, clock.worker.ts, protocol.ts, clockClient.ts, en.json +
  context.json (string clockClient.storageUnreadable, shown to Anita 2026-11-07; ja/zh translation still to do),
  docs/ui-panels.md, docs/code-map.md, DECISIONS.md, ARCHIVE.md, TASKS.md, checks/t2703-storeread.mjs. Filed T-2705.
- pinchecks finished: T-2630 landed 983d5192 (checks/cores.mjs; TW_CORES=n, else a ledger claim; TW_NOPIN=1 for
  timing runs). Smoke on 2 cores takes ~6 min, so give it TW_CORES=4 when the machine allows. T-2706 unused.
- **boarding** ab7583ab6d3c9eea9 ([971c94]'s, Opus, tree .tw-trees/boarding, Vite 6173, 3 cores): T-2699 DONE in the tree
  (memory grows were the cost: new sim/src/heap.rs allocator grows ahead; WASM step p50 311 -> 6.6 ms, round 359 s ->
  6.6 s on all Japan; dumps byte-identical; filed T-2707). T-2700 DONE too (one hunk in track/world.rs
  `trial`: previews price the network 2x not 4x; median 2.4 -> 1.6 ms, byte-identical). boarding retired (context).
  **T-2716 is a landing risk**: heap.rs's grow-ahead (alloc+free a >=32 MB pad) aborts dlmalloc's validate_size in a
  cargo --release wasm build (fine in the app's --profile wasm); same trick in the demand module since T-1594.
  heapfix SETTLED it (T-2716 archived): an LLVM GVN fold of std dlmalloc's inlined validate_size header read after
  alloc_zeroed (false abort, no corruption found); heap.rs now a dlmalloc crate 0.2.14 over its own grow-in-big-steps
  system side, GlobalAlloc methods inline(never); demand/wasm.rs uses it too. Dumps byte-identical, step p50 4.4 ms.
  **LANDED** f9142979 (tree removed, main WASM rebuilt, smoke 26/26, same 12,521,264 rail trips). Was: the tree ready to land whole (T-2699 + T-2700 + T-2716; Cargo.toml/lock). t1148 "place pairs" fails
  in the tree before and after (unrelated; check on master). T-2707 Open.
  Order: netkey reports -> smoke main tree -> commit main-tree work by path (docs/code-map.md and architecture.md first,
  the tree touches them too) -> agent-tree land boarding -> rebuild main WASM -> smoke. 8800 after the join.
- editnote finished T-2682 (archived), **UNCOMMITTED, commit after worldnext's gates**: app/src/game/editNote.ts,
  checks/t2682-editnote.mjs (new; FAIL before, PASS after), docs/ui-map.md, docs/code-map.md, DECISIONS.md, ARCHIVE.md,
  TASKS.md. No strings. T-2708 unused.
- storeread finished T-2705 and T-2704 too (archived; checks t2703-storeread.mjs 4/4, t2704-worker-retry.mjs 2/2, smoke
  25/25), **UNCOMMITTED with T-2703**: also app/src/ui/StartMenu.tsx, app/src/workers/demandClient.ts,
  docs/architecture.md, checks/t2704-worker-retry.mjs; new string startMenu.savesUnreadable shown to Anita. Filed T-2710
  (clock worker download retry). ui-panels.md "Loading" is 160 lines (over the ~150 cap): a doc-tidy item.
- **framing** a1605459efab5fade ([971c94]'s, Opus, read-only, no cores): Anita's ask (relayed by old tab, 2026-11-07)
  "for when i post about it online, how do you think i should frame the optimization we've done, that sounds impressive
  and is honest? ... in general what we've done over other similar games". DONE: research/post-framing.md (6 framings,
  top pick the exact speedups; drop "fractal" for "level of detail"; Japan whole-solve 21 min predates this week's speedups).
- Cap 16 (Anita 2026-11-07 ~00:25: "feel free to use all coores"); back to 10 when she is up.
- marketpack finished T-2631 (archived), **UNCOMMITTED**: checks/smoke.mjs (new check "market and city packs open",
  26 checks now; `SMOKE_BREAK_MARKET=nyc` to test), checks/agent-tree.mjs (warns on game/*.json drift). No app code or
  strings. Commit with the rest after the gates (code-map/TASKS/ARCHIVE mix everyone's records).
- nnerr finished T-2691 (archived): "Nn" is MapLibre's AJAXError (a failed openfreemap basemap tile, stackless after
  the worker hop); status-0 now a console warning, HTTP errors readable. UNCOMMITTED: app/src/map/mapView.ts. T-2711 unused.
- worldnext 01:39: Gulf run 1 done (22 gated, 13 parked, no fault); Oman's plan_units rule fixed, 8 gate parks relaxed
  (dam/mei pop_vs_fua, sabkha water points, Qatar jobs 0.74); rerun run-gulf2.cmd python 25296, log t2697-run-gulf2.log.
- worldnext 01:44: South Africa run 1 done (45 gated, 2 parked, relaxed: sos pop_vs_fua, inh jobs); rerun python 46556,
  log t2696-run-za2.log. Latin America: AR 39, CO 48, PE 27 built, Chile running, EC/UY to come; 7 parks so far.
- worldnext ~02:10: Gulf (35) and South Africa (47) all gated; Latin America running (Chile, then EC, UY). Given T-2712
  (Colombia GHSL blobs tiq pis lve held; DANE admin_totals) and T-2713/T-2714 spare.
- sheetfix finished: T-2597 closed without change (already right since T-2567); T-2637 check edit only, UNCOMMITTED:
  checks/t2580-sheet-water.mjs. Filed T-2715 (two zones' stations in one grid cell, 1 case on Japan): left at P3.
- storeread finished T-2710 too (archived; clock worker holds messages until `booted`, retries 1 s / 4 s;
  t2704-worker-retry.mjs 4/4, smoke 26/26). UNCOMMITTED with T-2703. storeread done for the night (~245k).
- **netkey** a056de92a7ef940f3: T-2588 DONE (archived; saveNetKey hashed the eras tail, which autosave prunes; now
  hashed without it; demandCache.ts, checks/t2588-netkey.mjs, notes/T-2588.md; UNCOMMITTED). Now applying T-2718 (its
  tryCache patch in demandClient.ts: drop unknownReach strays from the read-ahead need).
- citynames finished T-2681 (archived; 431/432 cities matched, ja 428, zh 429). UNCOMMITTED: tools/place_names.mjs,
  app/src/i18n/place-names.json + places.ts, ui/NewGame.tsx, listParts.tsx, TopBar.tsx, Cities.tsx,
  inspector/CityInspector.tsx, docs/localization.md. **Rerun after the join**: `node trainworld/tools/place_names.mjs
  --opencc <scratch>/opencc/node_modules/opencc-js` (~1 min). Filed T-2720 (places still Latin).
- worldnext 02:29: Latin America run 1 done (138 gated, 20 parked; tiq pis lve held for T-2712; 16 relaxed); Guayaquil's
  broken OSM school ring fixed in pipeline tier0.osm_education (worldnext's, uncommitted). Rerun of 17: python 15992, log
  t2695-run-latam2.log, ~20-30 min, then the JOIN (hold pack writers then).
- worldnext ~03:00: **last gate done (app/WASM landings may resume); JOIN running** (15 countries, 237 rows gated, 6
  held), --jobs 11, log artifacts/trainworld/t2695-join.log, ~3-4 h. **HOLD all data/packs and data/terrain writers.**
  Undo: tools/country/restore.py T2695 prejoin --go. Plan: when netkey and citynames report (both mid-edit in app/),
  smoke the main tree once and commit tonight's app work by path; rebuild 8800 only after the join (it copies packs).
- Main-tree app work committed (one commit before f9142979; smoke 26/26). t1148 place pairs FAILS on master too: T-2721.
  Still uncommitted: worldnext's files (game/*.json, world_plan.*, tier0.py, world-run.md, notes/T-2695.md).
  After the join: rerun place_names.mjs, smoke, commit worldnext's files, rebuild 8800 (snapbuild), smoke light/dark.
- Next unassigned id **T-2722**.

## Noticed by capview, not filed yet (worth small tasks)
- A transfer-fare edit (setFareTransfer 0 -> 0.5) right after a stop-drop's long update never started a solve within 60 s.
- linesPending cleared on any demand update end (T-1733): T-2658 replaced the blanket clear; check it is gone.

## Waiting on Anita
- T-2620 (keep old rates on track >5 km from stations): maps-93's question.
- T-2667 OSM attractions: yes, after release work.
- Strings shown, no answer = fine.

## Next
1. Land trees as they report; commit records by path; show Anita strings and pictures (copy to artifacts/trainworld/review/).
   l10nfix's English changes and pictures (artifacts/trainworld/l10nfix/topbar-world-ja.png, trip-zh-Hans.png) and
   ridersnow's (artifacts/trainworld/ridersnow/pending-build-after.png) still to show her.
2. When the agents have finished, smoke master, rebuild 8800, smoke light/dark, tell Anita to reload.
3. Localization left: pass 5 screenshots (checks/l10n-shots.mjs) per language; T-2643 swept files (EditNote/editNote/
   profitEstimate and lineFigures/editCap after ridersnow; diagram/ after namespread) then their translation; T-2672
   (l10nfix's leftovers); OFFERED on Anita's release word.
4. On Anita's fares answer: build T-2669/T-2670 (+ the multiples display) per notes/T-2668.md.
5. Queued, not started: T-2624, T-2625, T-2627, T-2630, T-2631, T-2637, T-2638, T-2597, T-2593, T-2588, T-2592, T-2437
   (Brazil, pack lock free). T-2702 (coastal holes) after worldnext's join.

## How to
- Scratch = C:\Users\anita\AppData\Local\Temp\claude\dce1fefcc1d43aef\b35f881f-a7d6-4f9f-9c17-4b4b2ffbdf21\scratchpad:
  snapbuild.cmd, snapbuild-frozen.cmd, smoke8800.cmd, smokedev.cmd, mark_taking.py, keep_hunks.py,
  common-brief-tree.md (tree agents), translate-brief.md, back-brief.md, review-brief.md, merge3json.py (3-way json
  merge on a rebase conflict; prefer it over maps-93's mergejson.py, which resurrects removed keys), dropkeys.mjs,
  opencc\ (the opencc-js package for `--opencc`), resolve_*.py (conflict-hunk pickers). A new orchestrator copies these
  and fixes paths inside.
- Translation pipeline per unit: Opus translator (merge), Sonnet back-translator (only the unit's keys), fresh Opus
  reviewer (passes 3-4), zh-TW's opencc after zh-Hans is reviewed. Glossary false positives: add an `except` or a
  longer term (checks/i18n.mjs skips a term inside a longer term).
- 8800: `cmd //c "<scratch>\snapbuild.cmd <hash> > <scratch>\snapbuild.log 2>&1"` (Bash), claim 4, smoke8800.cmd.
  Build number = `git rev-list --count`. If 8800 does not answer, restart serve.py detached.
- Landing: `node trainworld/checks/agent-tree.mjs land <name> "<sentence>"`, then `remove <name>`. On a rebase conflict
  resolve in the tree, `git add`, `git -c core.editor=true rebase --continue`, typecheck, land again. A tree's own
  copied WASM bindings go stale after a sim landing (`set_far_chunks` type error in the tree only): harmless.
  Commit the main tree's uncommitted edits in files the landing touches first (ff fails otherwise).
- Commit by path: `git commit -m "..." -- <paths>`; `git add` a new file first.
- Agents pinning browsers: give them a core range (e.g. mask 0xF00); several pinned to core 15 and starved (ridersnow).
- Publish: see trainworld/docs (freeze.py, deploy.py); push only on Anita's word.

## Lessons from this run
- maps-93's handoff times ran ~4 h ahead of the machine clock: always Get-Date.
- A smoke that times out on page.goto at 30 s is the busy machine, not the change: rerun with 4 cores.
- Translators merging the same keys from two scopes: give each prefix one owner in the brief.
- Agents that leave a background wait re-notify: TaskStop them after their report.
- Earlier lessons (maps-93): pin every browser to its claim; agents' scratch can hold tens of GB; commit by pathspec;
  tree pages read the main data junction; serve.py dies with a VS Code restart; retire agents near 450-550k tokens.

## Anita's standing asks
- Player text: agents propose wording per the style rules, list every string old -> new; land without waiting; show
  her. Zone, region and realm names need no review.
- TODO.md short-term = prompts: copy her words verbatim into briefs.
- Every demand task has a "done when"; demand changes go to her first. Buses settled. Keep the work moving.
- Disk: delete orphaned temp files; keep only current and latest published builds.
