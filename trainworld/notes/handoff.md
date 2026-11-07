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
- **boarding** ab7583ab6d3c9eea9 ([971c94]'s, Opus, tree .tw-trees/boarding, Vite 6173, 3 cores): T-2699 boarding
  steps speed (sim/). May file T-2707. Land after worldnext's gates (WASM rebuild of main after).
- **editnote** a3e76ee8c5cadc238 ([971c94]'s, Opus, main tree, Vite 6174, 2 cores): T-2682 edit note never settles.
  May file T-2708. Owns editNote.ts and settle-rule code. App change: commit after worldnext's gates.
- **framing** a1605459efab5fade ([971c94]'s, Opus, read-only, no cores): Anita's ask (relayed by old tab, 2026-11-07)
  "for when i post about it online, how do you think i should frame the optimization we've done, that sounds impressive
  and is honest? ... in general what we've done over other similar games". DONE: research/post-framing.md (6 framings,
  top pick the exact speedups; drop "fractal" for "level of detail"; Japan whole-solve 21 min predates this week's speedups).
- Cap 16 (Anita 2026-11-07 ~00:25: "feel free to use all coores"); back to 10 when she is up.
- Next unassigned id **T-2709**.

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
