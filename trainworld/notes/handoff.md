# Orchestrator handoff, trainworld

Orchestrator: session 5f661fa7-7343-4c77-9e53-f444fc8a7bd5, name maps-5c, since 2026-11-04 09:05
Updated: 2026-11-04 ~15:10 local clock (context past 500k: a fresh orchestrator can take over any time)

Read trainworld/CLAUDE.md and TASKS.md first. Landings are in ARCHIVE.md and `git log`.

## State
- master **dad3c5e6** (records) after 1fbe042b hsrfix T-2419. Today's landings, newest first: 1fbe042b T-2419 fast
  behind slow, 861369d9 T-2365 trip graph, a2eb391f T-2414 bridge ramps, f4977dd7 RU/BY shipped, 184937ae T-2420
  profit map, 67468b83 T-2407 diagram blocks, beec2537 uismall, adf36414 trip kernel, 83dee708 T-2405/T-2408 diagram
  strokes, d295f405 T-2403 lag, 441780dc panels3, 3cf52fc0 home, 3b795c19 T-2363, 6039c1b0 T-1920, 401fd11d cleanup.
- **8800 = build 907 (dad3c5e6)**, plain from data/packs. Light smoke failed once (53 console "pE") while
  waterroute's parts rerun rewrote 2,971 packs during the build's copy; rerun PASSED, dark PASSED. **Rebuild 8800
  once waterroute's re-route is written**, so its packs are one consistent state. LIVE = 821. Publish only on
  Anita's word (release "later this week").
- Cores cap 15 (Anita: "using lots of cores is fine"). Disk ~28 GB free.
- Worktrees: t1734 (solveend's), neartrips, waterroute.
- **Pack lock: waterroute's** (order: waterroute -> countries3 Jakarta T-2341 -> Ukraine T-2344 -> T-2302 Europe).
- Uncommitted in main tree, not mine to commit until their agent says: waterroute's pipeline/market_drives.py,
  parts_run.py, routes.py, notes/T-2424*; countries3's Ukraine code pipeline/world_plan.py, docs/zones.md,
  notes/T-2344.md, notes/t2344/; solveend's notes/T-1734*, T-2409.md, T-2410.md; TODO.md (Anita's).

## Anita today (2026-11-04), in her words where it matters
- Cleanup: "we might be overdue for a code and documentation cleanup"; "feel free to make decisions among yourselves
  regarding reorg" (T-2359 stages T-2366..T-2384; T-2366/67 done).
- Drives removed (T-2384), one Average-load basis (T-2373), T-1920 yes, home strings "yup text fine".
- Split bar: bar/ticks/star stay on trips; trip graph built as she picked ("i like the log graph", other places on
  the same chart, "Trip count"); T-2365 done.
- Diagram: "one unified view ... city/region blocks is nice"; faint real-shaped city borders inside blocks (T-2168),
  water across blocks (T-2169), thin lines fade (done).
- Demand behind the clock: "yes lets stop at 4x"; "time capped", "calculation 1 day behind" (T-2409).
- Fast behind slow: "we can go with proposed fix" as long as the 7 express+local on one double track stays bad (done).
- Water: "yeah lets do road detours ... keep the osm car ferries and just make ferries quite expensive" (T-2424).
- Batches: notes/todo-2026-11-04a.md .. -04f.md.

## In flight
- **waterroute** aaf4ca522c7b5427c (Opus, tree .tw-trees/waterroute, 6196, ids T-2426..T-2428): T-2421 Liberty islets
  DONE in packs (parts rerun, checks pass); T-2424 market drives re-route restarted 14:46 on 7 workers, done
  ~16:00-16:30, log artifacts/trainworld/t2424-market-drives.log; T-2423 NI fix in it. Then: its Rust (intercity.rs
  zone detours, ferry cost) lands from its tree, pipeline + notes committed, 8800 rebuilt, then tell countries3 go.
- waterroute ~16:05: market-drives.bin re-routed and checked (679,457 pairs, 2,507 with OSM car ferries, 2.68M zone
  detours, 11.5 MB; Toronto-Buffalo 0.96/1.00/1.04 of land path; Detroit-Windsor and Kutch no ferry; Belfast-Dublin
  by road). Its smoke and final report pending; then land its tree (intercity.rs), commit its pipeline files and
  notes, rebuild 8800, smoke light/dark.
- **countries3** a98664e86d75b14a3 (Opus, 6194, ids T-2398..T-2402): **GO given for Jakarta ~16:10** (pack lock its);
  asks "go" again before Ukraine. Ukraine dry run ready (27 zones + kyi/kha/ode; Crimea as 3 zones to RUS; 959 councils left out). Tell Anita
  after: Sukabumi+Cianjur one zone, Karawang out of Jakarta; ISW line not an open licence (only whole units dropped).
- solveend done: T-1734, T-2410, T-2409 landed 1179abad (+ notes dca06c0d); not on 8800 yet (post-waterroute rebuild,
  smoke light and dark). Strings to show Anita: "Time capped at 4x while updating demand", "16x (time capped at 4x
  while updating demand)", "Calculation 1 day behind" / "Calculation N days behind". Worktree t1734 removed.
  waterroute's code-map.md line committed early (24a: market drives zone detours) so the land could fast-forward.
- neartrips done: T-2430 landed 00791e1d (the tag's legs asked on tool pick, late answers kept, retried; was a
  12-20 s wait with bubbles off on Tokyo). Not on 8800 yet: goes with the post-waterroute rebuild.
- Next free id **T-2431**.

## Next
1. As agents report: land trees (rebase, both cargo runs if sim changed, typecheck, smoke), commit records by path,
   rebuild 8800, smoke light and dark, show Anita strings old -> new.
2. Pack lock: waterroute -> countries3 Jakarta -> Ukraine -> T-2302 Europe replan.
3. Waiting: T-2404 (blueprint follow-up), T-2425 (diagram Japan stretches), T-2429 (terminal train vanishes),
   T-2256/T-2216 islands (after waterroute), T-2422 Governors tunnel (next data batch), cleanup stages T-2368..T-2384
   (T-2368 code-map, T-2369 app dead code, T-2370 flags, T-2376 docs, T-2379 sim, T-2384 drives removal, T-2298),
   T-2168/T-2169 diagram borders and water, T-2396 dock height, T-2393 scheduler idea.

## How to
- Scratch = C:\Users\anita\AppData\Local\Temp\claude\c--Users-anita-projects-maps\5f661fa7-7343-4c77-9e53-f444fc8a7bd5\scratchpad
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
- T-2267/T-2270 design leftovers: optional cross "Stop (Esc)", 3 px coloured bar edge, T-2285 Shift+1-7, mode
  colours (notes/for-anita.md "Open now").
- Fares: optional hover "Riders pay the old fares until demand has settled"?
- Profit map calls she may change: half base fare by boardings, blue for profit, no key.

## Lessons from this run
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
