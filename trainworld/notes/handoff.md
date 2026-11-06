# Orchestrator handoff, trainworld

Orchestrator: session b35f881f-a7d6-4f9f-9c17-4b4b2ffbdf21, name maps-b9, since 2026-11-06 14:10
Previous: session aa768402-8365-45e0-85b8-0c27fe256637, name maps-93, handed over 2026-11-06 14:10, finishing in-flight work
Updated: 2026-11-06 15:55 (machine clock, Get-Date). maps-b9 passed 500k tokens at ~15:50: ready to cycle.

Read trainworld/CLAUDE.md and TASKS.md first. Landings are in ARCHIVE.md and `git log`.

## State
- master **4a329a95**. 8800 = **build 1187 (ff7cb27f)**, old: everything since (translations, l10n stage 1 everywhere,
  T-2641 far ends, T-2656 drives fix, T-2657 notice, maps-eb's play count) is NOT on 8800. **Main-tree smoke not run**
  since these landings (machine at ~97% CPU, ledger over cap). Anita (~15:45): "we can wait for them to finish"
  (agents) before the rebuild. Then: claim 4, scratch smokedev.cmd (needs a Vite on 6190: `node
  trainworld/checks/dev-nowatch.mjs 6190`, the earlier one pid 11532 may be gone), snapbuild.cmd <hash>, smoke 8800
  light + dark (smoke8800.cmd), tell Anita to reload.
- LIVE = 821. Publish only on Anita's word (English first; ja/zh "in next few days"; release gate OFFERED in
  i18n/intl.ts). maps-eb's play count goes with the next release build (Anita: "yeah we can send it with next build").
- Cores cap 12, 4 free while Anita is on. Check real CPU with Get-Counter. Disk ~23 GB free at 14:20.
- **maps-b9 holds the landing role.** maps-93 forwards its remaining trees as they report: upkeep (T-2619/T-2620),
  capview (T-2622), borders2 (T-2639), dianames (T-2640). Its own Previous line goes when it has no agents left.
- Uncommitted in the main tree (all fine to commit by path at the next records commit): unit 4 translations in
  messages/sources ja/zh-Hans/zh-TW + glossary.json (flat junction term, excepts), TASKS.md, ARCHIVE.md, DECISIONS.md
  (agents' lines), docs/data-register.md (attract), notes T-2620 (upkeep's), T-2639 (borders'), T-2642/ (searchcost),
  T-2660 (marey), T-2663 (attract). TODO.md is Anita's: never commit or edit.

## Landed this session (maps-b9)
- translations units 1-2 292cf26c; T-2656 drives fix + maps-eb play count 5a71e6f6; l10n-insp 2c04f007 (T-2634);
  farends 75e04c44 (T-2641: NA market 1.94x, lines >10k within 1.9%, WASM set_far_chunks); stopnote 838c46e7 (T-2657);
  glossary/check ef51da93; l10n-core 2c10069e (T-2635, from maps-93: en.json merged with scratch merge3json.py, a true
  3-way; TripInspector switched to CostTerm.id); unit 3 translations 4a329a95.
- From maps-93 before the role passed: l10n-ui1 (T-2632), l10n-ui2 (T-2633), l10n-lead (T-2636), profitest (T-2626).
- "Running costs" -> "Ongoing costs" in the Money history hover (Anita agreed).

## In flight (maps-b9's agents; ids are Agent-tool ids, SendMessage by id)
Trees (land each: typecheck, smoke in the tree pinned, `agent-tree.mjs land`, remove; show Anita strings):
- **ridersnow** (Opus, .tw-trees/ridersnow): T-2658 riders shown as a settled 0 / "0 gain" while solving. ~275k tokens.
- **marey** (Opus, .tw-trees/marey): T-2660 timetable chart: other lines' trains on shared track (toggle, default on),
  flat junctions marked; plus T-2665 dense charts (default span by density, wheel zoom to a day). Use westleft's helper
  (game/northUp.ts `lineReversed`/`leftToRightReversed`) for a horizontal axis once landed.
- **westleft** (Sonnet, .tw-trees/westleft): T-2661 every left-to-right line drawing puts the west end on the left
  (north for N-S lines, ~22 deg). Helper game/northUp.ts.
- **riderspace** (Sonnet, .tw-trees/riderspace): T-2664 line panel riders block tighter (CSS).
- **attract2** (Sonnet, .tw-trees/attract2): T-2666 attraction classes (waterfall, cave, hot spring, opera, Ferris
  wheel, casino) + 2 km reach; Wikidata refetch.
Read-only / design:
- searchcost DONE (notes/T-2642.md): Japan pass ~1.45x exact from two stages. Building: **searchleaf** (Opus, tree
  .tw-trees/searchleaf, kernel/search.rs, T-2652) and **fartable** (Sonnet, tree .tw-trees/fartable,
  intercity/pass.rs + api.rs, T-2653). Land both (sim -> WASM rebuild), then close T-2642 and T-2621.
- Unit 4: rev-tw-4b DONE (no meaning errors, 10 rewrites; opencc owed); rev-hans-4b DONE (2 small meaning fixes, 26
  rewrites). rev-ja-4b DONE (Traffic toggle 本数 -> 乗客数 fixed, 16 fixes). rev-tw-4a DONE (2 small meaning fixes: kindHoliday was "public
  holiday", issues.crossing). rev-hans-4a DONE (3 meaning fixes), rev-ja-4a DONE (2). maps-b9
  fixed scheduleCopy.withTrains in ja/zh-TW (they read it as per hour; it is trains on the line) and its context
  note. **occ-tw-4** running; then occ-tw-4 for both halves, then commit.
- **farescale** (Opus): T-2668 fares by country: evidence vs ^0.7, Tijuana breakdown, value-of-time exponent too
  (Anita: "lower the money/time ratio scaling with income too"); proposal to Anita, no changes. Ids T-2669-70.
Translation unit 4 (game = 4a ~407 keys; map + workers + ui leftovers + clockWorker = 4b ~314; scratch unit4.md,
back-brief.md, review-brief.md + review4.md): all translated and merged. Reviewing: **rev-hans-4a, rev-hans-4b,
rev-tw-4a, rev-tw-4b**; back-translating **back-ja-4a, back-ja-4b** -> then rev-ja-4a, rev-ja-4b (review4.md; rev-ja-4b
also fixes unit 1's ja demandSelector.trackTraffic 本数 / trackTrafficTip, a meaning error: thickness = riders; fix
guide.traffic with it). achievements.* and guide.* are 4b reviewers' only. Then zh-TW opencc pass for unit 4 (one
agent, like occ-tw-3), commit, and pass 5 screenshots for all units (checks/l10n-shots.mjs). After that: the swept
files (EditNote, editNote/profitEstimate, lineFigures/editCap after capview, diagram/ after borders2+dianames; T-2643
leftovers) need stage 1 then translation; T-2655 code faults (list on its TASKS line) need a code agent.

## Translation status (16:30)
- **All 2124 English messages translated and reviewed in ja, zh-Hans, zh-TW** (units 1-4, last commit 70bd30cb).
- **l10nfix** (Opus, tree .tw-trees/l10nfix): T-2655 code that pieces English (list on its TASKS line). Its changed
  keys need a quick review pass after landing. Still to do: pass 5 screenshots (checks/l10n-shots.mjs) per language;
  T-2643 swept files (EditNote/editNote/profitEstimate after ridersnow; lineFigures/editCap after capview+ridersnow;
  diagram/ after borders2+dianames), then their translation; then OFFERED in i18n/intl.ts on Anita's release word.

## Waiting on Anita
- **T-2669** running costs and upkeep at each country's fare scale (construction, trains stay world prices): money
  only. **T-2670** value of time 0.7 -> 0.6 (demand change). farescale's note notes/T-2668.md (fares' ^0.7 is
  supported: fitted 0.66; Tijuana's loss is costs not scaling).
- T-2667: fill thin countries with OSM tourism places (changes demand). attract recommended yes. notes/T-2663.md.
- T-2668 proposal when farescale reports.
- Strings shown, no answer yet = fine: the demand-stopped notice ("Demand stopped, so the game is paused. Reload the
  page to start it again." / "Reload page"), held trains "1,234 trains held out of service", plurals (1 station,
  1 line, 1 car, 1 other place, buses).

## Next
1. Land trees as they report (mine and maps-93's four). Commit records and translations by path.
2. When the agents have finished (Anita's word), smoke master, rebuild 8800, smoke light/dark, tell her to reload.
3. Finish unit 4 (ja reviews, zh-TW opencc), then screenshots pass, then swept files + T-2655.
4. Relay farescale's proposal; start T-2642 build stages from searchcost's design.
5. Queued, not started: T-2624, T-2625, T-2627, T-2630, T-2631, T-2637, T-2638, T-2597, T-2593, T-2588, T-2592, T-2437
   (Brazil, pack lock free).

## How to
- Scratch = C:\Users\anita\AppData\Local\Temp\claude\dce1fefcc1d43aef\b35f881f-a7d6-4f9f-9c17-4b4b2ffbdf21\scratchpad:
  snapbuild.cmd, snapbuild-frozen.cmd (path fixed), smoke8800.cmd, smokedev.cmd, mark_taking.py, keep_hunks.py,
  common-brief-tree.md (tree agents), translate-brief.md, back-brief.md, review-brief.md, unit3.md, unit4.md,
  review3.md, review4.md, merge3json.py (3-way json merge on a rebase conflict; prefer it over maps-93's mergejson.py,
  which resurrects removed keys), dropkeys.mjs (drop keys from every translation; no merge lock: tell translators),
  opencc\ (the opencc-js package for `--opencc`). A new orchestrator copies these and fixes paths inside.
- Translation pipeline per unit: Opus translator (merge), Sonnet back-translator (only the unit's keys), fresh Opus
  reviewer (passes 3-4), zh-TW's opencc after zh-Hans is reviewed. Glossary false positives: add an `except` or a
  longer term (checks/i18n.mjs skips a term inside a longer term).
- 8800: `cmd //c "<scratch>\snapbuild.cmd <hash> > <scratch>\snapbuild.log 2>&1"` (Bash), claim 4, smoke8800.cmd.
  Build number = `git rev-list --count`. If 8800 does not answer, restart serve.py detached.
- Landing: `node trainworld/checks/agent-tree.mjs land <name> "<sentence>"`, then `remove <name>`. A tree's own
  copied WASM bindings go stale after a sim landing (typecheck error on a new method in the tree only): harmless.
  Commit the main tree's uncommitted edits in files the landing touches first (ff fails otherwise).
- Commit by path: `git commit -m "..." -- <paths>`; `git add` a new file first.
- Publish: see trainworld/docs (freeze.py, deploy.py); push only on Anita's word.

## Lessons from this run
- maps-93's handoff times ran ~4 h ahead of the machine clock: always Get-Date.
- A smoke that times out on page.goto at 30 s is the busy machine, not the change: rerun with 4 cores.
- Translators merging the same keys from two scopes (achievements/guide live in game/ but showed under ui): give
  each prefix one owner in the brief.
- Agents that leave a background wait re-notify: TaskStop them after their report.
- Earlier lessons (maps-93): pin every browser to its claim; agents' scratch can hold tens of GB; commit by pathspec;
  tree pages read the main data junction; serve.py dies with a VS Code restart; retire agents near 450-550k tokens.

## Anita's standing asks
- Player text: agents propose wording per the style rules, list every string old -> new; land without waiting; show
  her. Zone, region and realm names need no review.
- TODO.md short-term = prompts: copy her words verbatim into briefs.
- Every demand task has a "done when"; demand changes go to her first. Buses settled. Keep the work moving.
- Disk: delete orphaned temp files; keep only current and latest published builds.
