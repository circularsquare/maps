# Orchestrator handoff, trainworld

Orchestrator: session 4a9d3a95-b5e4-4ac5-ab4b-4ef63d690cc6, name maps-17, since 2026-11-04 16:38
Updated: 2026-11-04 18:40. **Anita restarts VS Code now; the next orchestrator takes over with `/takeover
trainworld`.** No agent of maps-17's is running; nothing is mid-flight; maps-5c is done (Jakarta committed 670d3241).

## Start here (next orchestrator)
1. Land, in order: profit2, achieve, deadcode, diagborders, termtrain (details under "In flight"). The records are
   committed (03baa406); keep the main tree clean of edits in files a land touches, or its fast-forward fails.
2. Rebuild 8800 on the result (it is still build 917, without Jakarta), smoke light and dark. Then delete
   data/work/pre-T2341/ (2.3 GB), pre-T2294/, pre-T2418/.
3. Run checks/city-browser.mjs on one zone, then start a fresh Opus agent on Ukraine T-2344 (notes/T-2344.md
   "Next"; Crimea joins mos-ru, so that joint solve, ~1 h, reruns). After Ukraine, remove the kit's sim copy:
   `tools\country\setup_sim.py <kit> remove` (kit in session 8509086d's scratch, countries\).
4. Tell Anita (from countries3): Sukabumi+Cianjur is one 5.56M zone (could be two); Karawang+Purwakarta is out of
   Jakarta (in Jabodetabek it would push jak past the 40M split); Jakarta now on 8800 once rebuilt.

Read trainworld/CLAUDE.md and TASKS.md first. Landings are in ARCHIVE.md and `git log`.

## State
- master **e2a06912** (Anita's own "trainworld work" commit of the whole tree, 17:31) after cf102c0a (maps-5c: gate
  regex fix for the T-2365 hover grid). **8800 = build 917 (698d3954)**, all of
  today's landings to then. LIVE = 821. Publish only on Anita's word (release "later this week"; Mexico and Brazil
  before release, her ask).
- Cores cap 15 (Anita away for a few hours from ~17:30: "feel free to keep working"). Disk ~25 GB free.
- **Pack lock: free (the orchestrator's).** Jakarta T-2341 shipped and committed 670d3241 (t1053-zone-ids PASS, 750
  successors; countries3's smoke 23/23). Order now: Ukraine T-2344
  (fresh agent, state in notes/T-2344.md "State at countries3's hand-back") -> T-2302 Europe replan -> Mexico
  T-2436 -> Brazil T-2437 (Anita: "lets do euro preplan first").
- Worktrees: achieve, deadcode, profit2 (finished, to land), diagborders (T-2168 finished + diagnames working),
  termtrain (working).
- Uncommitted in the main tree: docsfix's docs (below); my TASKS/ARCHIVE/DECISIONS/notes records; countries3's
  pipeline/world_plan.py, docs/zones.md, docs/data-register.md, notes/T-2341.md, T-2344.md, notes/t2344/ and game
  json (registry.json, market-rows.json, zone-successors.json, world_plan.json): theirs to commit; TODO.md (Anita's).

## Anita today (2026-11-04), in her words where it matters
- Cleanup: "we might be overdue for a code and documentation cleanup"; "feel free to make decisions among yourselves
  regarding reorg" (T-2359 stages).
- Profit view: ratio colours, grey before figures, whole fare by km (T-2440 built); "profit display looks good".
- Money plots: one-off costs behind expenses, thicker bars, 1.5 px gaps (T-2444 built).
- Capacity marks keep the dashes ("i cannot tell what is a crowding indicator and what is just a red line"); draw
  order fix only (T-2347 built). Shift+1-7 yes (T-2285 built). Cross and Esc fine; bars' top edge 2 px (T-2446
  built). No fares hover. Mode colours good. T-2430 trips near fine, water fixes great, T-2409 strings fine.
- Diagram: blocks "nice"; faint real-shaped city borders in blocks (T-2168 built), city names only far out (T-2438),
  water across blocks (T-2169 later).
- Restarting the VS Code extension: she wants to, at a quiet point (no agent mid-task, Jakarta committed). Tell her.
- Batches: notes/todo-2026-11-04a.md .. -04h.md.

## In flight (maps-17's; common brief scratch/common-brief-tree.md)
- **To land when Jakarta is committed**, in this order (achieve and profit2 both edit main.ts, map/renderer.ts,
  map/rendererShaders.ts, ui/style.css):
  1. **profit2** (T-2440 profit view, T-2439 trip-graph hover lag, T-2443 split hover width, T-2444 money plots;
     T-2445 not a bug). Smoke ok. Checks: t2411-profit-map.mjs, t2439-hover-perf.mjs, t2444-money-plot.mjs.
  2. **achieve** (T-2431 achievements, T-2285 Shift+1-7, T-2347 dash draw order, T-2446 2 px edge). Smoke ok.
     Retake the achievements shot after (its shot.mjs in scratch/achieve/; the old one was overwritten).
  3. **deadcode** (T-2369 + T-1304, 38 files). Smoke ok. Then delete the main tree's stale trainworld/app/dist.
  4. (Anita committed the whole tree herself at 17:31 before going out, e2a06912 "trainworld work": docsfix's docs,
     my records and countries3's mid-Jakarta game json are in it. So only edits since then need committing.)
     docsfix's main-tree docs, were to be committed by path: trainworld/SPEC.md, docs/{world,ui-panels,ui-building,
     ui-inspectors,ui-map,ui-demand-views,track-and-service,demand-kernel,demand-mode,demand-plain,demand-runtime,
     demand-sources,demand-views,demand-worldwide,demand-market,regional,architecture,world-run,
     adding-a-city-by-hand,adding-a-country}.md, notes/archive/t0001-0499/T-304.md,
     notes/archive/t1000-1499/T-1204.md (profit2 and achieve also edited docs/ui-map.md, ui-panels.md,
     ui-building.md in the main tree: check the hunks).
  Then the records (TASKS, ARCHIVE, DECISIONS, notes/T-2375.md, T-2411.md, T-2267.md, todo-2026-11-04g/h.md,
  for-anita.md, handoff), rebuild 8800, smoke light and dark, run city-browser.mjs on one zone before the next gate.
- **diagnames** DONE (in tree .tw-trees/diagborders): T-2438 city names below 1.47 px pitch (smoke ok; shot
  artifacts/trainworld/diagnames/jpn-v2-0.3.png); T-2425 handed back with findings, no code (notes/T-2425.md: the
  long stretches are stations between blocks; B stops converging with long inter-block hops). Filed T-2447 (Nagano's
  smaller blocks unnamed). t2162-diagram-names fails 3 names at 14 px + 2 timing limits, untouched by it: compare
  against master at landing. The tree also holds diagborders' finished T-2168 (zone borders: warp.ts, borders.ts;
  98.5-99.6% of stations on their side locally; shots artifacts/trainworld/diagborders/jpn-v0-3.png). Lands as one.
  diagborders retired at 414k tokens.
- **termtrain** DONE (tree .tw-trees/termtrain): T-2429 fixed in sim/src/track/service.rs `hand_over` + test in
  transition_tests.rs; both cargo runs pass (361, 534), smoke ok, no strings. Land after the others (sim: the land
  rebuilds the main WASM). Doc sentence added to the main tree's docs/track-and-service.md (commit with docsfix's).
- Filed today by maps-17: T-2431, T-2435 (t325-bare-check fails on master), T-2436..T-2440, T-2443..T-2446.
- Next free id **T-2450**.

maps-5c: done, countries3 retired. Its earlier in-flight lines are in git history (1e34ad36).

## Next
1. On maps-5c's message: land profit2, achieve, deadcode, commit docsfix and records, rebuild 8800, smoke light/dark;
   delete data/work/pre-T2418/ once Jakarta checks clean; start a fresh Opus agent on Ukraine (T-2344).
2. On diagnames' and termtrain's reports: land (termtrain changes sim: both cargo runs in the tree first).
3. Then, one tree at a time: T-2368 code-map (main tree, docs, after the landings), T-2370 flags (tree), T-2379 sim
   small items (tree), T-2373 bus panel (Opus, tree), T-2384 drives removal (tree, sim), T-2435, T-2404 blueprint,
   T-2396 dock height, T-2169 diagram water (after diagnames).

## How to
- Scratch = C:\Users\anita\AppData\Local\Temp\claude\c--Users-anita-projects-maps\4a9d3a95-b5e4-4ac5-ab4b-4ef63d690cc6\scratchpad
  (snapbuild.cmd, snapbuild-frozen.cmd, keep_hunks.py, mark_taking.py `<tasks.md> T-x=agent ...`,
  common-brief-tree.md for tree agents, lowdisk.ps1). A new orchestrator copies these and fixes the path inside
  snapbuild-frozen.cmd.
- 8800: `cmd /c "<scratch>\snapbuild.cmd <hash> > <scratch>\snapbuild.log 2>&1"`, then claim 4 cores, smoke on
  http://localhost:8800/trainworld/ light and with `$env:SMOKE_SHADE='dark'`. Build number = `git rev-list --count`.
- Landing a tree: `node trainworld/checks/agent-tree.mjs land <name> "<sentence>"` (rebases itself; rebuilds the
  main WASM when sim changed), then `remove <name>`. ff fails if the main tree has uncommitted edits in a file the
  commit touches: commit those first.
- **Commit by path**: `git commit -m "..." -- <paths>` (other sessions stage too).
- Docs lint: `venv\Scripts\python.exe trainworld\checks\doc_lint.py`.
- Publish: `freeze.py build <stamp>` with TW_MARKET_DRIVES set, `deploy.py --dist <dist> --version <v> --smoked`,
  website `git add trainsworld/ trainworld/`, commit "anitabuilder", push (her word), `deploy.py --verify --page
  --version <v>`. Afterwards prune publish/ and data/frozen/.

## Open questions to Anita (asked, unanswered)
- Money plot: one-off costs hidden on days they are smaller than both other bars (drawing them behind): fine?
- T-2168's borders are faint (`BORDER_ALPHA` in diagram/draw.ts) and sparse (Japan's zones are prefecture-sized;
  New York is one zone): show her on 8800 once landed.

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
- Agents near 450-550k tokens: retire after their report; give follow-ups to a fresh agent.

## Anita's standing asks
- Player text: agents propose wording per the style rules, list every string old -> new; land without waiting; show
  her. Zone, region and realm names need no review.
- TODO.md short-term = prompts: copy her words verbatim into briefs; save batches in notes/todo-<date>.md.
- Every demand task has a "done when". Buses settled. Thanks to agents in my own reply. Keep the work moving.
- Disk: delete orphaned temp files always; keep only the current build and the latest published ones; delete
  superseded backups once a later state checks clean.
