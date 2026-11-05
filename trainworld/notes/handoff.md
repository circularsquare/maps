# Orchestrator handoff, trainworld

Orchestrator: session 4a9d3a95-b5e4-4ac5-ab4b-4ef63d690cc6, name maps-17, since 2026-11-04 16:38 (resumed after
Anita's VS Code restart, ~19:00)
Updated: 2026-11-04 20:42

Read trainworld/CLAUDE.md and TASKS.md first. Landings are in ARCHIVE.md and `git log`.

## State
- master **d7527845** (+ records). Landed this evening: fac6820c profit2 (T-2440, T-2439, T-2443, T-2444), 7479ba4d
  achieve (T-2431, T-2285, T-2347, T-2446), 1797c894 deadcode (T-2369, T-1304), e8af5ad8 diagborders (T-2168, T-2438),
  3dc94145 termtrain (T-2429), 92173326 ferrybus (T-2451 bus legs on roads only, T-2178; T-2450 ferries 32 km/h, saves
  write LFR2), bcf588bc cityfirst (T-2453 selected zone's figures and graph from its cluster's round 0), 5f634c25
  busviews (T-2452 Speed/Profit for bus and ferry; buses take no station upkeep), d7527845 barecheck (T-2435, new WASM
  fn `cell_unreach_mask_day`). No worktrees. Dev smoke on d7527845 PASSED (23).
- **8800 = build 932 (92173326)**, smoke PASSED light and dark. **Rebuild to d7527845 once ukraine's pack writes end**
  (join, zone_terrain, market_drives, build_inputs); ukraine will message. LIVE = 821. Publish only on Anita's word
  (release "later this week"; Mexico and Brazil before release).
- **Cores cap 12** (Anita: "lets keep 4 cores free"). Disk 45 GB free.
- **Pack lock: ukraine's** (T-2344). Order after: T-2302 Europe replan -> Mexico T-2436 -> Brazil T-2437.
- Uncommitted: TODO.md (Anita's); ukraine's work (zone-names, realms, registry, market-rows, borders, world_plan,
  notes/T-2344.md).

## Anita today (2026-11-04), in her words where it matters
- Cleanup: "we might be overdue for a code and documentation cleanup"; "feel free to make decisions among yourselves
  regarding reorg" (T-2359 stages).
- Profit view "looks good"; money plots, achievements, capacity dashes (kept dashed: "i cannot tell what is a crowding
  indicator and what is just a red line"), Shift+1-7, 2 px bar edge, cross and Esc: all built and on 8800.
- Diagram: blocks "nice"; faint zone borders and far-zoom city names built (T-2168, T-2438); water later (T-2169).
- Sukabumi+Cianjur stays one zone ("no preference ... lets keep it"). Europe replan before Mexico and Brazil.
- Batches: notes/todo-2026-11-04a.md .. -04i.md.

## In flight (maps-17's; common brief scratch/common-brief-tree.md)
- **ukraine** (Opus, main tree, pack lock, port 6197, ids T-2454..T-2455): T-2344. Run 20261104-194048: 33 of 33
  gated, 0 parked (30 UA rows, 3 Crimean RU rows sif sel kcx), 53 min. Join UA,RU (mos-ru solve, ~1 h, 8 cores) started
  ~20:40, then zone_terrain, market_drives, build_inputs, stale, smoke. After it: `tools\country\setup_sim.py <kit>
  remove` for countries3's kit (session 8509086d's scratch, countries\). Backup data/work/pre-T2344/ goes once clean.
- **buspanel** (Opus, tree .tw-trees/buspanel from 2bcab3a0, 6191, ids T-2460..T-2461): T-2373 bus/ferry panels
  share the rail panel's schedule parts; one Average-load basis.
- **smallui** (Sonnet, tree .tw-trees/smallui from 2bcab3a0, 6192): T-2462 bus/boat Speed ramp red to yellow, T-2463
  money plot one-offs stacked on the expenses bar (Anita's answers, 2026-11-04 ~20:50).
- **diagzones** (Opus, tree .tw-trees/diagzones from 2bcab3a0, 6193, ids T-2465..T-2466): T-2464 diagram zone borders
  redone the Tube way (from which zone each drawn station is in; octilinear, even width, closed, between blocks).
  Anita on T-2168's: "not really visible at all", "very wavy", "dont fully contain the region", "why can't we just
  draw them the same way we draw lines?"
- **ferrybus** (Opus, resumed, ~275k tokens; tree .tw-trees/ferrybus recreated from 2bcab3a0, 6187, ids T-2470..71):
  T-2467 bus tool (no straight blue preview, first point refused off roads, snap to reachable road), T-2468 ferry
  stops snap, T-2469 bus legs over bridges (Anita, 2026-11-04 ~21:00, verbatim in its brief).
- Next free id **T-2472** (T-2456..57 unused; T-2458, T-2459 filed).

## Next
1. As agents report: land trees (typecheck, smoke), commit records by path, rebuild 8800, smoke light/dark, show
   Anita strings old -> new.
2. Open, not started: maybe a slow speed ramp for bus/boat in Speed (asked Anita), T-2458, T-2459 (cityfirst's
   follow-ups), T-2447 Nagano's
   unnamed diagram blocks, T-2425 findings (notes/T-2425.md), T-2169 diagram water, T-2368 code-map, T-2370 flags,
   T-2379 sim small items, T-2373 bus panel, T-2384 drives removal, T-2404 blueprint, T-2396 dock height.

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
