# Orchestrator handoff, trainworld

Orchestrator: session b35f881f-a7d6-4f9f-9c17-4b4b2ffbdf21, name maps-b9, since 2026-11-06 14:10
Updated: 2026-11-06 19:40 (Get-Date). maps-b9 is past 600k tokens: ready to be replaced any time.

Read trainworld/CLAUDE.md and TASKS.md first. Landings are in ARCHIVE.md and `git log`.

## State
- master **1f5de214**+. 8800 = **build 1265 (72d62101)**: NOT fares option N (08133e44), the about links (cf923482),
  T-2678 (2921b335), pass 5 fixes (1f5de214). Rebuild once namespread/attract2 land: claim 4,
  `node trainworld/checks/dev-nowatch.mjs 6191`, scratch `smokedev.cmd 6191`, stop that Vite by PID, snapbuild.cmd <hash>,
  smoke8800.cmd (light + dark), tell Anita to reload.
- Anita is back (~19:05); cap 12. Her jekyll server stays (~3 cores outside the ledger).
- LIVE = 821. Publish only on Anita's word (English first; ja/zh soon; languages switch on via OFFERED in i18n/intl.ts).
  maps-eb's play count ships with the next release build.
- maps-93 is done. The upkeep tree still needs `node trainworld/checks/agent-tree.mjs remove upkeep --force`.
- Core pins: parking/fareref 0xF0, namespread 0xF00, l10npolish 0x3000, l10nunits 0xC000.

## Landed this session (maps-b9)
- translations units 1-4; T-2656 + play count 5a71e6f6; T-2634 2c04f007; T-2641 75e04c44; T-2657 838c46e7; T-2635
  2c10069e; T-2640 093d7b82; borders2 e67d966d; 7e8d913e (huninn.css); T-2619 1c9b3c72; T-2661 18c10362; T-2664/T-2673
  75ea5b27; T-2660/T-2665 3d8cf723; T-2653 2776067f; T-2655 4da69460; T-2622 51d5e03f; T-2652 731614e5; T-2658 5a4aef70;
  about links cf923482 (website-58's relay); fares option N 08133e44 (T-2669, T-2670, T-2676); T-2678 2921b335 (ridersnow
  retired); pass 5 wording 1f5de214 (l10nshots: 15 fixes; shots in artifacts/trainworld/l10nshots/).

## In flight (maps-b9's agents; SendMessage by Agent id)
- **parking** a1e7652663fd3f414 (Opus, tree parking): T-2677 BUILT, stopped for Anita: in-city rail -3 to -7% in Mexico/
  India packs, parking dollars match real (Mumbai $0.55 vs $0.60). Asked her ~19:30; on yes: resume it to run cargo
  tests, wasm, smoke, records; then land.
- **namespread** ab73cb12c3a351833 (Sonnet, tree namespread): resumed to rebase, run checks, report (T-2674, T-2675).
- **attract2** a7442af28854bb4e7 (Sonnet, tree attract2): T-2666; quiet a long while: ask it.
- **fareref** a24127dfc645123a5 (Opus, tree fareref): T-2679 fare multiples relative to a reference country (Anita
  19:05, her words in the TASKS line); display only, fares charged unchanged.
- **l10npolish** ae9880d84bb49a85d (Opus, MAIN tree): T-2680 (zh-TW bold, ja Cities column, TrackInspector join, ja
  Riders word) + T-2643's EditNote/editNote/profitEstimate/editCap/StatusTag strings.
- **l10nunits** a36b3080f7932057e (Opus, MAIN tree): T-2672 ja/zh number, money and duration forms (orchestrator's
  call: the locale's own, 763万, $1.2億, 4分8秒), startup flash. Owns format.ts, walks.ts, tripView.ts, intl.ts, index.ts.
- Asked Anita ~19:30: one label "calculating..." for changed lines too (instead of "loading...")? maps-b9 recommends yes.

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
   (Brazil, pack lock free). Next unassigned id **T-2682**.

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
