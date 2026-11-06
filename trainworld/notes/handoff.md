# Orchestrator handoff, trainworld

Orchestrator: session b35f881f-a7d6-4f9f-9c17-4b4b2ffbdf21, name maps-b9, since 2026-11-06 14:10
Updated: 2026-11-06 17:42 (Get-Date). maps-b9 is past 600k tokens: ready to be replaced any time.

Read trainworld/CLAUDE.md and TASKS.md first. Landings are in ARCHIVE.md and `git log`.

## State
- master **72d62101** = 8800 **build 1265** (18:40; dev smoke and 8800 light + dark all 25/25, no "Nn" errors: those
  were starved 1-2 core tree runs). Next rebuild: after fares (tree fares, waiting on Anita) and attract2 land.
  Rebuild: claim 4, `node trainworld/checks/dev-nowatch.mjs 6191`, scratch `smokedev.cmd 6191`, stop that Vite by PID,
  snapbuild.cmd <hash>, smoke8800.cmd, tell Anita to reload.
- LIVE = 821. Publish only on Anita's word (English first; ja/zh "in next few days"; languages switch on via OFFERED in
  i18n/intl.ts). maps-eb's play count ships with the next release build.
- Cores cap 12, 4 free while Anita is on; check real CPU with Get-Counter. iso-terrain (isochrone) holds 3.
- maps-93 is done (17:40, all its trees landed or forwarded). The upkeep tree still needs
  `node trainworld/checks/agent-tree.mjs remove upkeep --force` (a process held files there).
- Uncommitted records (commit after searchleaf and ridersnow land, by path): ARCHIVE/DECISIONS/TASKS lines for T-2652,
  T-2653, T-2622, T-2655, T-2658, T-2666?; notes/T-2652.md, T-2653.md, T-2658.md, T-2666.md. TODO.md is Anita's.

## Landed this session (maps-b9)
- translations units 1-4 (all messages in ja, zh-Hans, zh-TW, reviewed); T-2656 drives fix + play count 5a71e6f6;
  l10n-insp 2c04f007 (T-2634); farends 75e04c44 (T-2641); stopnote 838c46e7 (T-2657); l10n-core 2c10069e (T-2635);
  dianames 093d7b82 (T-2640); borders2 e67d966d; 7e8d913e (huninn.css to main.ts: dev workers had not started since
  2c10069e); upkeep 1c9b3c72 (T-2619); westleft 18c10362 (T-2661); riderspace 75ea5b27 (T-2664, T-2673); marey 3d8cf723
  (T-2660, T-2665); fartable 2776067f (T-2653, WASM rebuilt); l10nfix 4da69460 (T-2655); capview 51d5e03f (T-2622).

## In flight (maps-b9's agents; SendMessage by Agent id; trees under .tw-trees/)
- **ridersnow** (Opus, tree, ~350k): T-2658 done and smoked; its land stopped on conflicts with capview (lineFigures.ts)
  and l10nfix (LinesTab.tsx, schedule.tsx). Sent back to resolve mid-rebase, re-run t2622-cap-view + smoke, report;
  then `land ridersnow "<sentence>"` (land re-runs the rebase from its finished state). Retire after landing.
- **searchleaf** (Opus, tree): T-2652; running its smoke. Land (sim -> WASM rebuild), then T-2642 and T-2621 close (its
  ARCHIVE lines are already written for T-2652 and T-2642; check T-2621).
- **attract2** (Sonnet, tree): T-2666 attraction classes + 2 km reach; on cargo tests.
- **l10nsmall** (Opus, main tree, merge only): translates the 4 new mareyChart keys and reviews l10nfix's 9 changed keys.
- namespread STOPPED, tree .tw-trees/namespread kept (code + docs done, unchecked): T-2675 borders 30 km / 50% and
  T-2674 name spread. To run when cores free: `node trainworld/checks/t2438-city-names.mjs na,jpn,ne`, typecheck,
  smoke, shots (scratch namespread\shots.mjs), 0 stations on wrong side NA/NY/Japan, NY unchanged; then land.
  Its DECISIONS lines are in the main tree. Needs a fresh agent or a resumed one (its id is in maps-b9's transcript).

## Noticed by capview, not filed yet (worth small tasks)
- A transfer-fare edit (setFareTransfer 0 -> 0.5) right after a stop-drop's long update never started a solve within 60 s.
- linesPending was cleared when any demand update ends (T-1733), incl. a camera-started rough solve (ridersnow's T-2658
  rule replaces that blanket clear: check after it lands whether this is gone).

## Waiting on Anita
- Answered 17:50: fares option **N**. BUILT in tree .tw-trees/fares (agent farescale, 610k: retire after landing; smoke
  25/25 in the tree; T-2669, T-2670, T-2676 viaduct = 1.0x upkeep; notes/T-2668.md section 7). NOT landed: asked Anita
  (~18:30) whether the regional market's rail share halving on 100-250 km in India/Mexico (Mumbai-Pune 14->4%, Delhi-Agra
  42->22%; the car's fuel weighs half the minutes) and Mexico's in-city rail -2% (5-cent rounding) are fine. On landing:
  translate format.levelUpkeepLine, format.upkeepLines, format.upkeepRule + 2 new running-cost keys (removed from ja/zh);
  filed T-2677 (parking power 0.3 -> 0.5, a demand change). T-2667 yes but after release work (P2, open);
  her jekyll stays running (~3 cores outside the ledger: size claims for it).
- T-2620 (keep old rates on track >5 km from stations): maps-93's question.
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
   (Brazil, pack lock free). Next unassigned id **T-2678**.

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
