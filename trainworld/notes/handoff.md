# Orchestrator handoff, trainworld

Orchestrator: session 4a9d3a95-b5e4-4ac5-ab4b-4ef63d690cc6, name maps-17, since 2026-11-04 16:38 (resumed after
Anita's VS Code restart, ~19:00)
Updated: 2026-11-04 22:51

Read trainworld/CLAUDE.md and TASKS.md first. Landings are in ARCHIVE.md and `git log`.

## State
- master **8f9cb6af** (+ records). Landed 2026-11-04 evening: profit2, achieve, deadcode, diagborders, termtrain,
  ferrybus (T-2451, T-2450), cityfirst (T-2453), busviews (T-2452), barecheck (T-2435), buspanel (T-2373), smallui
  (T-2462, T-2463), ferrybus again (T-2467-69), diagzones (T-2464), diagnames (T-2447), "holiday load" hover, and
  8f9cb6af **Ukraine + Crimea (T-2344)**: 33 rows shipped, kyi-ua group, borders with RU/BY factor 0.
- **8800 = build at 8f9cb6af**: smoke PASSED light (dark ran in the background at ~22:55). LIVE = 821. Publish only on
  Anita's word (release "later this week"; Mexico and Brazil before release).
- **Cores cap 12** (Anita: "keep 4 cores free"); europe may raise to 14 after ~01:00 if she's quiet, back to 12 after.
- Disk 39 GB free (pre-T2344 deleted).
- **Pack lock: europe's** (T-2302). Then Mexico T-2436 -> Brazil T-2437.
- Uncommitted: TODO.md (Anita's); agents' work as they go.

## Anita today (2026-11-04), in her words where it matters
- Cleanup: "we might be overdue for a code and documentation cleanup"; "feel free to make decisions among yourselves
  regarding reorg" (T-2359 stages).
- Profit view "looks good"; money plots, achievements, capacity dashes (kept dashed: "i cannot tell what is a crowding
  indicator and what is just a red line"), Shift+1-7, 2 px bar edge, cross and Esc: all built and on 8800.
- Diagram: blocks "nice"; faint zone borders and far-zoom city names built (T-2168, T-2438); water later (T-2169).
- Sukabumi+Cianjur stays one zone ("no preference ... lets keep it"). Europe replan before Mexico and Brazil.
- Batches: notes/todo-2026-11-04a.md .. -04i.md.

## In flight (maps-17's; common brief scratch/common-brief-tree.md, from 8f9cb6af)
- **europe** (Opus, main tree, pack lock, port 6197, ids T-2479..80): T-2302 Europe replan per notes/T-1189.md "Next";
  go given if the dry counts match 628 -> 462 (399 removed, 233 new). It messages maps-17 when a browser gate starts
  and ends: land trees only between gates.
- **uismall2** (Opus, tree .tw-trees/uismall2, 6195, ids T-2481..82): T-2473 tutorial text stays greyed, T-2474 track
  setting carries to new track, T-2475 expected profit ticking (explain, steady if noise), T-2477 no camera snap.
- **loadwait** (Opus, tree .tw-trees/loadwait, 6196, ids T-2483..84): T-2476 worst-case demand wait, T-2478 Vancouver.
- **sizes** (Sonnet, main tree read-only, ids T-2485..86): T-2472 size summary into notes/T-2472.md.
- Anita's batch: notes/todo-2026-11-04j.md. Next free id **T-2487**.

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
