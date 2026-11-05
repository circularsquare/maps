# Orchestrator handoff, trainworld

Orchestrator: session 4a9d3a95-b5e4-4ac5-ab4b-4ef63d690cc6, name maps-17, since 2026-11-04 16:38 (past 500k tokens
at 16:13 on 2026-11-05: a fresh orchestrator can take over from this file any time)
Updated: 2026-11-05 16:15

Read trainworld/CLAUDE.md and TASKS.md first. Landings are in ARCHIVE.md and `git log`.

## State
- master **b2789380**. Main-tree smoke PASSED at 004354ca (12,515,600: the 2 over 598 are europe's uncommitted
  market-rows.json; trees give 598). Every tree landed today smoked 12,515,598.
- **8800 = build 956 (a8a72002)**, from before Europe's retire; waits for Europe's join to end. LIVE = 821. Publish
  only on Anita's word (release "later this week"; Mexico and Brazil before release).
- **Cores cap 12** (overnight 15 ended ~08:00). Disk 45.6 GB free.
- **Pack lock: europe's** (T-2302). Then Mexico T-2436 -> Brazil T-2437.
- Uncommitted: TODO.md (Anita's); europe's files (world_plan.json, registry.json, market-rows.json, zone-names.json,
  zone-successors.json, docs/world-run.md, notes/T-2302.md).

## In flight
- **europe** (agent a48ef8746df10ad7f, Opus, main tree, pack lock, ~345k tokens): join batch 20261105-095013, python
  PID 6408 (cmd 17556), log artifacts/trainworld/countries/t2302-join.log, since 09:50; at 16:13 on Kosovo (psh-xk),
  the late small groups. Then after-steps (terrain_fine, zone_terrain, market_drives, build_inputs, stale,
  zone_successors --set, smoke, docs), report. Kit: old scratch\europe; notes/T-2302.md has PIDs and restart
  commands. Backup data/work/pre-T2302/pre; undo `tools/country/restore.py T2302 pre --go`. Its background wait died
  in the 10:05 VS Code restart and was re-set by message.
- **loadbar** (agent a51952333953d7ac7, Opus, tree .tw-trees/loadbar from 88ea32a6, port 6194): T-2557 (does Anita's
  North America save's import+reload hit the demand cache? loadnotes saw bubbles only at ~193 s) and T-2552 (load bar
  model: 97% early then creeps, other zones' bars sit at 99%, "A few seconds left" for 22 s).

- **diagrivers** (agent a1fd51edf9f661771, Opus, tree .tw-trees/diagrivers from 2c951d73, port 6193): T-2528 dial
  default 0.5 (Anita: "i kidna like 0.5"), water width follows the drawn land (Hudson/East River ~1/4 of Manhattan,
  no collapse downstream, outskirt rivers thin), Long Island Sound as open water; Vignelli map as reference.
  notes/T-2528.md has her words. Anita ~16:30: "yes we can carry on" (maps-17 continues past 500k).

## After Europe's join (in this order)
1. Rebuild 8800 with scratch snapbuild.cmd (first use of the dist-new swap, T-2483), claim 4 cores, smoke light and
   dark; expect 12,515,598 once europe's market-rows.json is committed; check no water-mask RangeError. Ping Anita
   (`python C:/Users/anita/projects/maps/tools/ping.py`). Rerun checks/diagram-game.mjs captures after (diaghokkaido
   redid them, but recheck on 8800's build).
2. Disk: `venv\Scripts\python.exe -I trainworld\notes\t2554\after_join.py`, then `--go` (~15 GB; it refuses while a
   batch lock is held and keeps frozen sets until a freeze newer than 20261104-222232 exists). Delete
   data/work/pre-T2302 once Europe checks clean (10.6 GB).
3. T-2554's three small code changes (maps-17's call, Anita told, may veto): routes.py stops writing
   drive_near_f.npy; world_run drops cells.* and walk_zone.u8* when a row ships; raw t196/t142b caches as .json.gz.
   One small agent, between runs.
4. Mexico: wire pipeline/mexico.py per notes/T-2080.md "Wiring left" (ARCHETYPE MEX, ZONE_RULES module/plan_units,
   the ten metro rows to archetype C module mexico under the plan lock), then T-2436's run (expect pop_vs_plan parks:
   GHSL 10-15% under the census in big cities). Then Brazil T-2437 (no prep; brazil.py hook done).

## Landed 2026-11-05 (after build 956; none on 8800 yet)
- Loading and memory: loadwait (T-2476, T-2478), memhunt (T-2485), loadfix (T-2499 range race, T-2459), focusmem
  (T-2495, T-2500), bubblefix 810f0b9b (T-2508 bubbles seconds after figures; T-2531 no regression across builds),
  loadnotes 1cf096fc (T-2523 click note, T-2524 bar hover + stall line).
- Cleanup: simtidy (T-2379), codemap (T-2368, T-2504), drives (T-2384), flags (T-2370), demandtidy (T-2371),
  toolstidy (T-2372), t409 check (T-2513).
- Diagram: water on by default 2a0541c2 (T-2497, T-2516), city names (T-2521), parallel lines (T-2520), crossings
  (T-2534), islands (T-2540, T-2517), straits/bays (T-2541, T-2542), Hokkaido + sparse floor + station-less islands
  (T-2550), distortion dial default 0.8 (T-2556, closed T-2555).
- UI: uismall3 (T-2525, T-2526, T-2522, T-2519 headway 4 h), uismall4 (T-2535, T-2536), launchbits (T-2544 link card
  = logo over globe, T-2545 load-fail text, T-2548 purpose picker in settings on phones, T-2549 tab top line),
  newer-save notice reworded c1109058, load-fail text sentence case 6431a1fa, link card big centred logo 9dfc5db4.
- Prep: mexprep cd58b687 (pipeline/mexico.py, T-2080/T-2081, wiring left), diskprune (T-2554, 11 GB raw deleted),
  launch research (T-2537, research/launch.md).
- Strings shown to Anita today, all acknowledged or chosen by her: "Edit stops"; "calculating..."; the newer-save
  notice; "Not enough memory. ... This game works best on a computer."; "Ran out of memory. ..."; "Failed to load :(
  ..."; "Purpose" (settings row); link card "Build trains around the world!"; the click notes ("Calculating trips
  after your change…", "Calculating trips in N cities…", "Calculating trips in New York…", "Loading Boston…") and
  bar hovers ("About 2 min left", "No progress for 1 min").

## Anita 2026-11-05, in her words where it matters
- Diagram: "i love how japan looks"; "perhaps we could turn down distortion if its easy" (dial at 0.8, asked whether
  to keep it); "peninsulas dont hae to be huge especially if theres nothing on them. middle groud"; islands with no
  stations "can be either ommitted (if small relative to their surroundings) or at least not expanded".
- Launch: she'll speed up her own gameplay recording (T-2546 on hold); no Discord/subreddit; no Cloudflare Cache Rule
  yet; phone tested (NA save refused with the memory message, Japan loaded).
- Disk: "anything that nonnegligibly slows down our edvelopment process to gain stoage we probably dont need ot do".
- Batches: notes/todo-2026-11-05a.md, -05b.md.
- Next free id **T-2559**.

## Next
1. As agents report: land trees (typecheck, smoke in the tree, `agent-tree.mjs land`), commit records by path, show
   Anita strings old -> new and review pictures (copy to artifacts/trainworld/review/).
2. Open and ready to start (cores allowing; Europe's join holds 6): T-2532 (edit bubbles wait for the final; worker
   change), T-2533 (after a cache hit, solve the zone in view first), T-2553 (balance periods between workers), T-2512
   (cleanup A4 rest), T-2374 app splits (now unblocked: T-2371, T-2372 done), T-2381 kernel/intercity splits, T-2529
   and T-2538 (diagram routing loses parallels and crossings), T-2530, T-2539, T-2551, T-2528 (river width), T-2518 and
   T-2498 (water time), T-2510 (Japan masks miss big rivers: data, after joins), T-2527 (t409 check strength), T-2558.

## How to
- Scratch = C:\Users\anita\AppData\Local\Temp\claude\c--Users-anita-projects-maps\4a9d3a95-b5e4-4ac5-ab4b-4ef63d690cc6\scratchpad
  (snapbuild.cmd, snapbuild-frozen.cmd, keep_hunks.py, mark_taking.py `<tasks.md> T-x=agent ...`,
  common-brief-tree.md for tree agents, lowdisk.ps1). A new orchestrator copies these and fixes the path inside
  snapbuild-frozen.cmd.
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

## Lessons from this run
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
