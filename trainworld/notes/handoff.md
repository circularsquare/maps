# Orchestrator handoff, trainworld

Orchestrator: session b35f881f-a7d6-4f9f-9c17-4b4b2ffbdf21, name maps-b9, since 2026-11-06 14:10
Previous: session aa768402-8365-45e0-85b8-0c27fe256637, name maps-93, handed over 2026-11-06 14:10, finishing in-flight work
Updated: 2026-11-06 16:20 (Get-Date). maps-b9 is past 600k tokens: ready to be replaced any time.

Read trainworld/CLAUDE.md and TASKS.md first. Landings are in ARCHIVE.md and `git log`.

## State
- master **db980d4f**. 8800 = **build 1187 (ff7cb27f)**, old: nothing since 11-06 morning is on it (translations, l10n
  stage 1 everywhere, T-2641 far ends, T-2656 drives fix, T-2657 notice, T-2640 diagram names, maps-eb's play count).
  **Main-tree smoke not run** since those landings (machine ~97% CPU). Anita (~15:45): wait for the agents to finish
  before the rebuild. Then: claim 4, a Vite on 6190 (`node trainworld/checks/dev-nowatch.mjs 6190`), scratch
  smokedev.cmd, snapbuild.cmd <hash>, smoke8800.cmd (light + dark), tell Anita to reload.
- LIVE = 821. Publish only on Anita's word (English first; ja/zh "in next few days"; languages switch on via OFFERED in
  i18n/intl.ts). maps-eb's play count ships with the next release build.
- Cores cap 12, 4 free while Anita is on; check real CPU with Get-Counter. Disk 99 GB free (16:00).
- **maps-b9 holds the landing role.** maps-93 forwards its last trees as they report: upkeep (T-2619/T-2620), capview
  (T-2622), borders2 (T-2639). dianames landed (093d7b82). maps-93's Previous line goes when it has no agents left.
- Uncommitted: little (TODO.md is Anita's: never commit or edit it; notes/T-2620.md upkeep's, T-2639.md borders',
  T-2660.md marey's, commit with their landings).

## Landed this session (maps-b9)
- translations units 1-4 (292cf26c, 4a329a95, 70bd30cb: all 2124 messages in ja, zh-Hans, zh-TW, reviewed); T-2656
  drives fix + play count 5a71e6f6; l10n-insp 2c04f007 (T-2634); farends 75e04c44 (T-2641); stopnote 838c46e7 (T-2657);
  l10n-core 2c10069e (T-2635; en.json merged by scratch merge3json.py; TripInspector -> CostTerm.id); dianames 093d7b82
  (T-2640). From maps-93 earlier: l10n-ui1/ui2/lead, profitest (T-2626).

## In flight (maps-b9's agents; SendMessage by Agent id; trees under .tw-trees/)
- **ridersnow** (Opus, tree): T-2658 riders as settled 0 / "0 gain" while solving. Cause found: lineFigures clears
  pending labels on demandUpdating=false for one tick before flush sends the held edit; fix = results carry the network
  world version, settle only from a later result. Touches game/lineFigures.ts (capview holds it: merge at landing),
  editNote.ts, demandClient.ts, demand.ts, today.ts, schedule.tsx, LinesTab.tsx. ~310k tokens: retire after landing.
- **marey** (Opus, tree): T-2660 timetable chart other lines + flat junctions; T-2665 dense charts span + wheel zoom.
- **westleft** (Sonnet, tree): T-2661 west end left (helper game/northUp.ts); marey uses it after landing.
- **riderspace** (Sonnet, tree): T-2664 riders block tighter + T-2673 build menu "Regular/High speed" 1px.
- **attract2** (Sonnet, tree): T-2666 attraction classes + 2 km reach (Wikidata refetch).
- **searchleaf** (Opus, tree): T-2652 leaf stations in kernel/search.rs (Japan searches 1.4x+, exact).
- **fartable** (Sonnet, tree): T-2653 far-zone table by zone (intercity/pass.rs, api.rs; spreading 3x+, bit-exact).
  Land both (sim -> WASM rebuild), then close T-2642 and T-2621.
- **l10nfix** (Opus, tree): T-2655 code that pieces English (list on its TASKS line); its changed keys need a short
  review pass in the three languages after landing.

## Waiting on Anita (asked ~16:20)
- Fares by country (notes/T-2668.md, "For Anita" round 2): option **N** = fares, value of time and train running costs
  all at income^0.5; upkeep, construction, trains at world prices (busy Tijuana -$10.3M -> +$7.1M a game day; every
  city breaks even at 4.6 riders a car; riders inside cities and NY unchanged; regional market's long trips shift some
  rail -> air/car in poor countries: measure before landing). Or O (running costs 0.3). Upkeep shown as multiples:
  base ground track 1x or the construction chip's base (maps-b9 leans to the chip's base). T-2669 (costs, money only),
  T-2670 (fares + value of time, demand change) are written for N.
- T-2667: OSM tourism places for thin countries (changes demand); recommended yes (notes/T-2663.md).
- Strings shown, no answer = fine: demand-stopped notice, held trains "1,234 trains held out of service", plurals.

## Next
1. Land trees as they report; commit records by path; show Anita strings and pictures (copy to artifacts/trainworld/review/).
2. When the agents have finished, smoke master, rebuild 8800, smoke light/dark, tell Anita to reload.
3. Localization left: l10nfix landing + review of its keys; pass 5 screenshots (checks/l10n-shots.mjs) per language;
   T-2643 swept files (EditNote/editNote/profitEstimate after ridersnow; lineFigures/editCap after capview+ridersnow;
   diagram/ after borders2) then their translation; OFFERED on Anita's release word.
4. On Anita's fares answer: build T-2669/T-2670 (+ the multiples display) per notes/T-2668.md.
5. Queued, not started: T-2624, T-2625, T-2627, T-2630, T-2631, T-2637, T-2638, T-2597, T-2593, T-2588, T-2592, T-2437
   (Brazil, pack lock free). Next unassigned id **T-2674**.

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
