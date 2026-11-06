# Orchestrator handoff, trainworld

Orchestrator: session b35f881f-a7d6-4f9f-9c17-4b4b2ffbdf21, name maps-b9, since 2026-11-06 14:10
Previous: session aa768402-8365-45e0-85b8-0c27fe256637, name maps-93, handed over 2026-11-06 14:10, finishing in-flight work
Updated: 2026-11-06 14:10 (machine clock, Get-Date; maps-93's times below run ~4 h ahead of it: its "18:15" = git's 14:03)

Read trainworld/CLAUDE.md and TASKS.md first. Landings are in ARCHIVE.md and `git log`.

## In flight (maps-b9's agents, 2026-11-06 15:40)
Agents report to maps-b9 (Agent tool, background). **maps-b9 holds the landing role** (maps-93 ~15:30: forwards its
five trees upkeep, capview, borders2, dianames, l10n-core as they report, then winds down).
- LANDED 15:35: translations 292cf26c, T-2656 + play count 5a71e6f6, l10n-insp 2c04f007 (T-2634), farends 75e04c44
  (T-2641, main WASM rebuilt). Main typecheck clean. **Smoke on master NOT yet run** (machine 96%, ledger full):
  Vite on 6190 (pid 11532) is up; run scratch smokedev.cmd after claiming 4, then rebuild 8800 (snapbuild.cmd
  75e04c44 or later) and smoke it light+dark, then tell Anita to reload.
- **marey** (Opus, tree .tw-trees/marey): T-2660 timetable chart interlining + flat junctions. Ids T-2662.
- **westleft** (Sonnet, tree .tw-trees/westleft): T-2661 line strips west on the left; tells main its helper for marey.
- Unit 3 translators (ui/inspector, 623 keys; scratch unit3.md): **tr-ja-3a/3b, tr-hans-3a/3b, tr-tw-3a/3b** (Opus).
- l10n-insp landed; still to look at: the "Part of [pill] group" row (.line-group CSS) on the line panel.
- **farends tree READY TO LAND** (T-2641 done ~15:20, records already in main tree: ARCHIVE, DECISIONS, notes/T-2641.md):
  NA market 1.94x (pairs 867k -> 422k a period), lines >100k worst 0.29-0.54%, >10k worst 1.88% (both "overstated" per
  the agent; its line-id join is weak, medians fine), LA-Chicago -0.48%; Japan 1.13-1.24x (searches are T-2642).
  LANDED 75e04c44. T-2621 stays open for T-2642.
- **searchcost** (Opus, scratch crate only, maps-b9 scratch\searchcost): T-2642 design and measure. Ids T-2652-53.
- **maps-eb's play count** (distinct browsers; server half live, shonei-server c4634b7), uncommitted in the main tree,
  checked: app/src/game/playtime.ts, app/public/privacy.html, docs/architecture.md, DECISIONS.md's last line
  ("counting distinct ppl"; commit only that hunk, keep_hunks.py). Commit with the next commit. Deploy: with the next
  release build (Anita ~14:30: "yeah we can send it with next build"); no one-off.
- **Translators, pass 1** (Opus, main tree, write only through `l10n.mjs merge`): tr-ja-1, tr-hans-1, tr-tw-1, each
  `--folder ui` (404 keys landed so far) plus the 61 no-file keys. Brief: scratch translate-brief.md. When each
  reports: a Sonnet back-translator then a fresh Opus reviewer per docs/localization.md "Translating". Later units:
  ui L-Z (after l10n-ui2), ui/inspector (after l10n-insp2), game/map/workers/diagram/root (after l10n-core); every
  stage-1 folder re-runs `i18n.mjs --list` (l10n-lead's scanner fix found more). Release gate: OFFERED in
  i18n/intl.ts. Follow-ups l10n-lead named: widgets.tsx Stepper "Less "+label (l10n-ui2's), English flash before a
  message file loads, MapLibre localIdeographFontFamily (T-2586). l10n-lead landed ca03a19a / d17706bb.
- Unit 1 (ui + `help.`, 388 keys) merged in all three (doubts in scratch tr-ja-1, tr-hans-1, tr-tw-1 \doubts.md:
  21, 12, 16). Back-translating: **back-ja-1, back-hans-1, back-tw-1** (Sonnet) -> scratch back-<x>-1\back.json; then
  a fresh Opus reviewer each (rev-hans-1 DONE 14:50: no meaning errors, 22 rewrites; rev-ja-1 DONE ~14:55: none, 17 rewrites;
  rev-tw-1 DONE ~15:10: 3 meaning errors fixed, 37 rewrites; unit 1 reviewed in all three).
  Code faults from reviews on T-2655 (+ Blueprint.tsx:232 count+已選 pieced; ui.ts:324 superholiday `on ${name}`).
  zh-TW click words added to docs/localization.md Style (maps-b9). Open small call: settingsTab.day label. ja "line" now accepts 航路 for ferry lines (glossary, maps-b9): rev-ja-2 should turn unit 1's
  船の路線 keys into 航路 too. Unit 2 (ui L-Z,
  374 keys, landed 711df0ad): all three translated (374 merged each, doubts in scratch tr-<x>-2\doubts.md); back-tw-2 done, **rev-tw-2**
  reviewing; back-hans-2 done, **rev-hans-2** reviewing; **rev-ja-2** reviewing (also turns unit 1's 船の路線 into 航路).
  rev-tw-2 DONE: 1 meaning error (moneyHistory.running), 36 rewrites. English fault found: moneyHistory.running
  "Running costs {amount}" is the total of running trains + upkeep (MoneyHistory.tsx:139), while "running costs"
  means trains alone elsewhere. Proposed to Anita: "Ongoing costs {amount}" (pairs with "One-off costs" below it).
  Apply in en.json once rev-hans-2 and rev-ja-2 have reported (then a stale batch re-translates the key).
  rev-hans-2 DONE: same meaning error, 12 keys changed. zh-TW unit 2's opencc pass DONE (7 fixes). **Units 1-2
  (853 keys: ui + help + game's 14) fully translated and reviewed in ja, zh-Hans, zh-TW** (15:05). rev-ja-2: same
  moneyHistory.running error, 13 rewrites, ja.json has no 船の路線 left. context.json's stationsTab.ridersTip/ridersWhat
  notes fixed by maps-b9 ("per weekday", not "on a weekday"); game/'s translator must render the day phrases
  (DEMAND_DAY_PHRASE) as adverbs (ja 平日1日あたり). Next units when stage 1 lands: ui/inspector (l10n-insp tree),
  game/map/workers/diagram/root (l10n-core). (only their tr.json's keys); then review as unit 1.
  Reviewers of unit 2: check moneyHistory.running (English "Running costs" totals running + upkeep, glossary's term
  is running alone: if the English label is wrong, tell the orchestrator); zh-Hans used 换线 for changing lines to
  pass the line check (換乘 is the glossary-free usual word: decide), brief scratch review-brief.md (passes 3-4; zh-TW's opencc after zh-Hans's review).
  Glossary widened by maps-b9 (uncommitted, app/src/i18n/glossary.json): track's single/double/quad words, fare
  票款/票箱, ferry 航路. Pass 5 screenshots for all units together later.
  Code faults translators find go on T-2655's line.
- **T-2656** demand-stopped bug on Anita's save (shared drives buffer detached by a replacement's reopen): fixed by
  maps-b9 in app/src/game/marketRule.ts (fetchMarketDrives returns a copy per caller), typechecked, uncommitted; commit
  with the next records and rebuild 8800.
- **stopnote** (Sonnet, tree .tw-trees/stopnote from b9690900): T-2657 unrecoverable demand error pauses the game,
  top-centre notice with Reload. Strings to show Anita.
- maps-93 landed profitest (T-2626) 42f9a877 / c942ee14; still has upkeep, capview, borders2, dianames, l10n-core.
  Swept-file l10n queue now free: ui/EditNote.tsx, game/editNote.ts, game/profitEstimate.ts (then lineFigures/editCap
  after capview, diagram/ after borders2/dianames).
- **ridersnow** (Opus, tree .tw-trees/ridersnow from c942ee14): T-2658 riders shown as a settled 0 (and "0 gain")
  while the solve is still running (Anita ~15:10). Ids T-2659.
- moneyHistory.running renamed "Running costs" -> "Ongoing costs" (Anita ~15:10: "we can rename to ongoing costs"),
  en.json + the three translations re-merged against it (meaning unchanged), uncommitted.
- **T-2660** timetable chart shows interlining lines + flat junctions (Anita ~15:20): start an agent (Opus, design
  calls) in a fresh tree once l10n-insp has landed. Asked maps-93 (~15:20) whether I may land l10n-insp and farends
  between its landings.
- Id ranges: maps-93's agents T-2643..T-2649, maps-b9 from T-2650 (next unassigned **T-2661**).
- Scratch = C:\Users\anita\AppData\Local\Temp\claude\dce1fefcc1d43aef\b35f881f-a7d6-4f9f-9c17-4b4b2ffbdf21\scratchpad
  (helpers copied from maps-93's; snapbuild-frozen.cmd path fixed; common-brief-tree.md rewritten for maps-b9).

## State
- master **b21c0c66** at 14:25 (l10n-ui1 landed by maps-93 at 7d0a32a8). Disk **23 GB free** (not 50).
  CPU at takeover: 92% real, ledger 13 of cap 12 (maps-93's agents and isochrone's 3).
- (older) master d5b02108 (Mexico committed). **Real save smokes 12,515,594** since farebase (T-2559).
- **8800 = build 1187 (ff7cb27f)**, everything landed through sheetscale (T-2609); smoke light and dark PASSED 25/25
  ~17:00 on 11-06, 12,515,594. LIVE = 821. Publish only on Anita's word (release "later this week"; Mexico
  and Brazil before release).
- **Cores cap 12, 4 free** (Anita up ~08:30 on 11-06; ~21:45 on 11-05: "keep 4 cores free": real use, not just
  claims). Check with
  `Get-Counter '\Processor(_Total)\% Processor Time'` (75% = 4 free), not summed per-process CPU deltas (maps-93's 5 s
  sample over-counted cachehit's Chrome, which pins itself to its granted cores).
- **Pack lock free.** Mexico T-2436 done (105 zones, 3,200 -> 3,305 shipped, freeze 20261105-200211; backups deleted).
  Next country: Brazil T-2437 (no prep; brazil.py hook done). Disk 50 GB free.
- maps-17 (now named maps-03) handed over fully at 21:37: no agents of its own left.

## In flight (maps-93's agents, 2026-11-06 ~18:15)
Agents report to maps-93. A new orchestrator taking over: their work is in the trees named; check each tree's
`git -C .tw-trees/<name> status`, the task's notes and TASKS line, then land per "How to".
- **upkeep** (Opus, tree .tw-trees/upkeep): T-2619 Anita's upkeep set (per game day: ground 35k a km, viaduct 50k, cut and
  cover 100k, bored 130k, underground station 200k new, other stations 50k) and T-2620: measure line profits on her
  saves and propose the smallest offset so long lines stay as hard as today (her words in the task line; not a fare
  ratio change; demand changes go to her first). Before release.
- **profitest** (Sonnet, tree): T-2626 profit estimate in the edit note ("Estimated −$5.3M profit/day"; +/- hovers if
  easy). Rerunning its check after copying master's game/*.json (T-2631 trap). Before release.
- borders LANDED f71755b6 (records b1fdceb9): T-2629 done, T-2610 closed (rivers open on master); T-2637 filed.
  Anita's NA load: the card said ~6 min, real ~3 min: not slow, the estimate is off (T-2638, later).
- **capview** (Opus, tree): T-2622 the 4x clock cap the same whatever the view. Told about T-2631 trap.
- coarsemkt DONE (38f3e633, notes/T-2621.md). Anita said yes to both: **T-2641** (coarser far ends, Sonnet, Rust tree)
  and **T-2642** (cheaper exact searches, Opus, design+measure first): not started, for the next orchestrator.
- l10n-insp STOPPED at ~359k (tree .tw-trees/l10n-insp, not landed): 30 of 32 inspector files moved (706 -> 236
  strings; TripInspector.tsx and MarketTrip.tsx left, need a restructure; T-2634 line says what's left). Its smoke
  and pseudo shots NOT run (cores full): run tree smoke pinned, then land. Visible English changes to show Anita:
  complex-view tooltip "A and B and C" -> "A, B and C" (3+ stations; maps-93 would accept), "1 stations" -> "1
  station", thousands separators on two counts. Check style.css .line-group row ("Part of [pill] group").
- **dianames** (Sonnet, tree): T-2640 diagram zone names by how many zones are on screen.
- Anita ~18:45: start a new orchestrator now; maps-93 keeps landing its agents' reports until done, per /handoff.
  She said not to worry about a B-drive access she noticed (not any maps-93 agent).
- **borders** resumed (tree .tw-trees/borders2): T-2639 water drop-out reversed, borders only within 50 km of track
  (whole outline when >30%).
- **l10n-ui1 / l10n-ui2 / l10n-insp / l10n-core** (Sonnet, trees): T-2632..T-2635 localization stage 1 by folder
  (rules: scratch l10n-brief.md). Land each as it reports; en.json/context.json merge on rebase.
- **l10n-lead** (Opus, tree): T-2636 Language setting (SettingsTab), fonts, glossary ja/zh-Hans/zh-TW, translate and
  review runbook in docs/localization.md, time estimate. Then start translator agents per its runbook.
- **Release plan (Anita ~18:00):** not today. In first: upkeep, profit estimate, diagram polish. Then publish on her word
  (English), then ja/zh "in next few days". T-428 done (intercity.json out of packs); 8800 = build 1187 (ff7cb27f)
  smoked; master has more since (hillwater, T-428): rebuild 8800 after the next landings.
- **Awaiting Anita:** (T-2606 closed "no cut" ~18:20.) Her North America load felt ~4 min vs ~2:
  probably CPU (agents ran over claims, fixed ~17:30); she will retry; if still slow, time against build 1135 quietly.
- Queued, not started: T-2624 (trains-only edit's traffic waits for the whole solve), T-2625 (2D condensation), T-2627
  (join market step 1800 s limit), T-2630 (checks pin themselves to their claim), T-2631 (refused market pack fails
  smoke), T-2597, T-2593, T-2588, T-2592.

## Landed this session (maps-93)
- uismall5 3d0939bc (T-2569..T-2572, Anita's evening TODO; also the map credits button, invisible since T-1435, now
  just left of the trouble button; Anita: "attr fix looks good"), farebase f3b3cb18 (T-2559, 598 -> 594), t409strong
  (T-2527, T-2573 filed), cleana4 97c6e153 (T-2512, T-2575 filed), kernelsplit 8696ae95 (T-2381), walkcost 83f433fd
  (T-2324: the walking rule costs 5.6% of New York's calib; 3.2% over T-1059 net of other tasks), followtrack 665dd025
  (T-1358, 0.55 accepted), toworld 3d0d0097 (T-2576), Mexico d5b02108 (T-2436; T-2578 border factor filed).
- From maps-17: watertoggle 7bba88c2 (T-2565), cachecard 9a68feb5 (T-2560; cache-hit reload playable ~16-25 s on her
  NA save), sheet's write-up db56555e (T-2566).
- Strings shown to Anita (no answer yet): "Show water" (new hover); "Nothing built" -> "Loading..." while an account
  game's picture loads; "No stations" (new hover, trouble row).
- Next free id **T-2637** (check TASKS/ARCHIVE; agents filed T-2624 themselves).

## Anita 2026-11-05, in her words where it matters
- Diagram: "i love how japan looks"; "perhaps we could turn down distortion if its easy" (dial at 0.8, asked whether
  to keep it); "peninsulas dont hae to be huge especially if theres nothing on them. middle groud"; islands with no
  stations "can be either ommitted (if small relative to their surroundings) or at least not expanded".
- Launch: she'll speed up her own gameplay recording (T-2546 on hold); no Discord/subreddit; no Cloudflare Cache Rule
  yet; phone tested (NA save refused with the memory message, Japan loaded).
- Disk: "anything that nonnegligibly slows down our edvelopment process to gain stoage we probably dont need ot do".
- Batches: notes/todo-2026-11-05a.md, -05b.md, -05c.md (evening, T-2569..T-2572, all done).

## Next
1. As agents report: land trees (typecheck, smoke in the tree, `agent-tree.mjs land`), commit records by path, show
   Anita strings old -> new and pictures (copy to artifacts/trainworld/review/).
2. When upkeep, profitest and borders have landed: rebuild 8800, smoke light and dark (pinned), ping Anita; publish on
   her word ("How to" Publish).
3. Localization: land T-2632..T-2635 as they come, then translator agents per l10n-lead's runbook (ja, zh-Hans,
   zh-TW), then the swept files (EditNote, editNote/profitEstimate, lineFigures/editCap, diagram/).
4. Cores: cap 12, 4 free while Anita is on the machine; check real CPU with Get-Counter, not just the ledger.

## How to
- Scratch = C:\Users\anita\AppData\Local\Temp\claude\dce1fefcc1d43aef\aa768402-8365-45e0-85b8-0c27fe256637\scratchpad
  (snapbuild.cmd, snapbuild-frozen.cmd, keep_hunks.py, mark_taking.py `<tasks.md> T-x=agent ...`,
  common-brief-tree.md for tree agents, lowdisk.ps1, splice_handoff.py). A new orchestrator copies these and fixes the
  path inside snapbuild-frozen.cmd.
- 8800: `cmd //c "<scratch>\snapbuild.cmd <hash> > <scratch>\snapbuild.log 2>&1"` (Bash), then claim 4 cores, smoke on
  http://localhost:8800/trainworld/ light and with `$env:SMOKE_SHADE='dark'`. Build number = `git rev-list --count`.
  If 8800 does not answer, restart serve.py detached (it dies with a VS Code restart).
- Landing a tree: `node trainworld/checks/agent-tree.mjs land <name> "<sentence>"` (rebases itself; rebuilds the
  main WASM when sim changed), then `remove <name>`. On a rebase conflict: resolve in the tree, `git -c
  core.editor=true rebase --continue`, typecheck, land again. ff fails if the main tree has uncommitted edits in a
  file the commit touches: commit those first.
- **Commit by path**: `git commit -m "..." -- <paths>` (other sessions stage too).
- Docs lint: `venv\Scripts\python.exe trainworld\checks\doc_lint.py`.
- Publish: `freeze.py build <stamp>` with TW_MARKET_DRIVES set, `deploy.py --dist <dist> --version <v> --smoked`,
  website `git add trainsworld/ trainworld/`, commit "anitabuilder", push (her word), `deploy.py --verify --page
  --version <v>`. Afterwards prune publish/ and data/frozen/.

## Open questions to Anita (asked, unanswered)
- None open. Sheet round 1 answered ~21:30-21:40 (notes/T-2566.md "Anita's review"; sheet2 builds round 2). The three
  new strings were shown ~21:05; no answer yet means fine unless she names one.
- Earlier, answered ~18:15: water is experimental behind a button, default off (T-2565); sheet first, rough
  prototype outside the game (T-2566); "this sort of distortion scheme would solve the issue of region borders not
  having their right shape". Distortion default 0.5 (her pick).

## Lessons from this run
- 2026-11-06 ~17:30: CPU at 100% with 11 cores claimed: an unpinned smoke Chrome took ~15 cores, agents' probe
  Chromes ran over or without claims, rustc unclaimed; Anita's NA load ~2 -> ~4 min. All agents told to pin
  (`start /affinity`), brief updated, smoke8800.cmd pinned to 0xF000, T-2630 filed. Check real CPU with Get-Counter
  after launching browser agents, not just the ledger.
- Agents' scratch folders can hold tens of GB (marketsearch: 37 x 1 GB dumps, found at 10 GB free during Brazil's run,
  2026-11-06 01:50). When disk drops, measure the session scratch folders first; the common brief now says to clean up.
- A UI landing can silently break every pipeline gate: T-2365 made the top bar hover a grid and
  checks/city-browser.mjs's text wait timed out on every zone from 13:42 (fixed cf102c0a). Before the next
  country gate after a landing that touches the top bar or its hovers (profit2's T-2443 does), run
  city-browser.mjs on one zone first.
- An agent that left a background wait running re-sends its final report on every wake; TaskStop it once done.
- Check the clock (`Get-Date`) before writing times into the handoff.
- Commit with a pathspec; another session's staged moves get swept in otherwise (314776ed).
- Headless pages run over their core claims (a page took 7-8 cores on a 3-4 claim); briefs must say "cap the page's
  demand workers to the claim, claim with --pid, release between runs".
- An 8800 build during a pack rewrite can serve a mixed set (console errors); rebuild after pack writes end.
- A gate's browser checks run on a Vite over the main tree; joins re-gate without a browser, so app/WASM landings
  are fine during a join but not during a browser gate.
- Tree agents' pages read the main data junction: a tree made before a replan's game json commit asks for retired
  packs ("New York could not load").
- `world_run.py step <x> --help` may run the step for real (T-2398).
- serve.py (8800) dies with a VS Code restart; check it answers after one.
- Agents near 450-550k tokens: retire after their report; give follow-ups to a fresh agent.

## Anita's standing asks
- Player text: agents propose wording per the style rules, list every string old -> new; land without waiting; show
  her. Zone, region and realm names need no review.
- TODO.md short-term = prompts: copy her words verbatim into briefs; save batches in notes/todo-<date>.md.
- Every demand task has a "done when". Buses settled. Thanks to agents in my own reply. Keep the work moving.
- Disk: delete orphaned temp files always; keep only the current build and the latest published ones; delete
  superseded backups once a later state checks clean.
