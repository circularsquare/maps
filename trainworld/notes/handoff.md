# Orchestrator handoff, trainworld

Orchestrator: session aa768402-8365-45e0-85b8-0c27fe256637, name maps-93, since 2026-11-05 17:55
Updated: 2026-11-06 18:15

Read trainworld/CLAUDE.md and TASKS.md first. Landings are in ARCHIVE.md and `git log`.

## State
- master **d5b02108** (Mexico committed). **Real save smokes 12,515,594** since farebase (T-2559).
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
- coarsemkt DONE (38f3e633, notes/T-2621.md): coarser far ends ~1.8x NA market pass, lines >10k within 1.5%;
  Japan needs exact cheaper searches (stage 2). Two questions to Anita (stage 1 yes/no; stage 2 next?). No ids filed.
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
