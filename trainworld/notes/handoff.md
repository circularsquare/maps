# Orchestrator handoff, trainworld

Orchestrator: session 4a9d3a95-b5e4-4ac5-ab4b-4ef63d690cc6, name maps-17, since 2026-11-04 16:38
Updated: 2026-11-05 09:45

Read trainworld/CLAUDE.md and TASKS.md first. Landings are in ARCHIVE.md and `git log`.

## State
- master **3a2be5bf** (+ records commit). Main-tree smoke at 004354ca PASSED, 12,515,600: the 2 trips over 598 come
  from europe's uncommitted market-rows.json (trees, with the committed one, give 598). Recheck after Europe's join.
- **8800 = build 956 (a8a72002)**, from before Europe's retire. Not rebuilt since: waits for Europe's join (pack
  writes). LIVE = 821. Publish only on Anita's word (release "later this week"; Mexico, Brazil before release).
- **Cores cap 12** (back from overnight 15 at ~08:00). Disk ~40 GB free.
- **Pack lock: europe's** (T-2302). Then Mexico T-2436 -> Brazil T-2437.
- Uncommitted: TODO.md (Anita's); europe's files (world_plan.json, registry.json, market-rows.json, zone-names.json,
  zone-successors.json, docs/world-run.md, notes/T-2302.md).

## Landed 2026-11-05 (after build 956; none on 8800 yet)
- loadwait 8e2bd280 (T-2476, T-2478), memhunt ecac1167 (T-2485), simtidy fb674fba (T-2379), codemap 27a59ef1 (T-2368,
  T-2504), loadfix c8d22020 (T-2499 range race: reloads missed the demand cache; T-2459 Okinawa 842 -> 13 s),
  drives 74c07452 (T-2384), focusmem a19c9e08 (T-2495 peak 3.9 -> 3.5 GB; T-2500 Tokyo graph 874 -> 193 s),
  flags 118c615a (T-2370), t409 check c50034b8 (T-2513: stale check, no double count; T-2527 filed),
  diagwater2 afe9bf07 (T-2497 octilinear coast + T-2516 continuous rivers to the sea, LI Sound joined, water colour
  nearer land; still behind ?water; T-2510, T-2511, T-2517, T-2518 filed), demandtidy fb66c4ad (T-2371; T-2512 rest),
  toolstidy 249ecb09 (T-2372: dragged curve labels culled), uismall3 004354ca (T-2525 "Done editing stops" -> "Edit
  stops" pressed; T-2526 auto-names count lines stopping in the zone; T-2522 track bar text 12 -> 14 px; T-2519
  headway cap 4 h, 15-min steps past 1 h), diagnames2 3a2be5bf (T-2521 big half-transparent grey city names centred
  on their block, under the lines; only below the 1.47 px pitch switch, as before).
- Strings shown to Anita: "Done editing stops" -> "Edit stops"; "worker 2: undefined" -> "worker 2: its code could
  not be downloaded" (loadwait). Nothing else.

## In flight
- **europe** (Opus, main tree, pack lock, port 6197): all 234 rows built; gate 230/234; bef, krz, gzd, nic relaxed.
  gate2 09:41-09:49, 234/234 gated; **join from ~09:50** (no browser: app landings fine; nothing writes data/packs or
  data/terrain until it reports), then
  prejoin backup, join (detached, several hours), after-steps (terrain, market drives, inputs, stale,
  zone_successors --set, smoke, docs). Kit: old scratch\europe; notes/T-2302.md has PIDs and restart commands.
  Backup data/work/pre-T2302/pre (4 GB); undo `tools/country/restore.py T2302 pre --go`. Agent at ~335k tokens.
- diagparallel LANDED 4797d948 (records 3f5f4a7c): T-2520 Manhattan parallel pairs 0.53 -> 0.71; NYC layout 2.8 ->
  3.6 s (+27%, over the brief's 20%: maps-17 landed anyway, the look is what Anita asked for); filed T-2529 (routing
  breaks parallels), T-2530 (B convergence, Japan wrong-side 387 -> 481). Review: review/diagram-parallel-before/after.png.
  Run checks/diagram-game.mjs captures again after the 8800 rebuild.
- VS Code closed ~10:05: join survived (detached); diagparallel, bubbleload, europe resumed by message.
- Anita ~10:20 asked for publishing / attention tips. ~10:40: "we can have an agent research more thoroughly and
  compile a list of things fo rme to do"; no share button ("schematic map export already covers"); "i like the parallel
  lines! :) i think itll look even better once we get water in" -> maps-17 switched water on by default (2a0541c2).
- New batch notes/todo-2026-11-05b.md (T-2534..T-2536) + T-2537. Running: **uismall4** (Sonnet, tree from 2a0541c2,
  port 6194: T-2535 no confirm for empty line delete, T-2536 no profit before first calc), **diagcross** (Opus, tree
  from 2a0541c2, port 6193, ids T-2538..39: T-2534 ground crossings kept outside dense cores, small city on its
  through line), **launchprep** (Opus, research only, writes research/launch.md: T-2537 launch checklist for Anita,
  ≤40 web searches).
- launchprep DONE (T-2537): research/launch.md, checklist on top, committed e05fbcc3. Findings: no og/twitter tags on
  the live page (T-2543, **linkpreview** Sonnet tree port 6191 adding tags + preview.png, wording for Anita); .wasm and
  .json from GitHub Pages are DYNAMIC in Cloudflare (needs a Cache Rule on anita.garden/trainsworld/assets/*, dashboard,
  Anita's); R2 cost ~$0 to 10k visitors/day; Reddit rules unread (403), Anita to read; phone test hers.
- linkpreview LANDED 368d3bd2 (T-2543: og/twitter tags in app/index.html, app/public/preview.png = map-nyc candidate;
  meta description "Build trains around the world!" -> "Build trains in any real city and see how many people ride.";
  Anita to pick wording (3 options) and image (artifacts/trainworld/linkpreview/). Smoke only partial (1 core).
- Anita ~11:30: no Cache Rule yet ("we wont have 100k visitors"); no Discord/subreddit ("a bit presumptuous"); card
  text stays "Build trains around the world!"; card image = load-screen logo before a small globe; phone: NA save
  failed early with the memory pre-check (4.3 GB vs 4 GB), Japan save loaded. **launchbits** (Sonnet, tree from
  c0721519, port 6194): T-2544 card text + logo image, T-2545 load-fail messages with "works best on a computer".
  tourvideo stopped and tree removed (Anita ~11:40: clip later, T-2546 on hold). Her phone notes: T-2548 hide the
  purpose picker on phones (settings panel instead), T-2549 faint line above the dock tabs: both sent to launchbits.
  "saved by a newer version" on her phone = live 821 opening a cloud save written by 8800 (newer); right for players
  after a deploy, so left as is.
- uismall4 LANDED 42ccfc96 (T-2535 empty rail/bus line deletes without confirm; T-2536 no profit until trains + first
  demand result, "calculating..." meanwhile; zero-rider case loosened via linesPending, untested). String added:
  "calculating..." (Profit row).
- Anita ~10:50: island shapes matter, Japan as a test case. Filed T-2540 (takes in T-2517); **diagislands** (Opus,
  tree from 4ba1808b, port 6192, ids T-2541..42): water files are its, layout.ts is diagcross's (coordinate via me).
- **bubbleload** (Opus, main tree read-only, scratch worktrees for old builds, ports 6188-89, ids T-2532..33):
  T-2531 [P1] Anita ~09:30 "i feel like loading of demand bubbles in the city we're looking at has honestly
  regressed"; measures builds 827 -> 956 -> master on her save artifacts/trainworld/anita-2026-11-05/
  north-america-day-361.save and the real save; also T-2508 (bubbles ~25 s after figures), which she wants looked into.

## After Europe's join
- Rebuild 8800 with the new snapbuild.cmd (first use of the dist-new swap, T-2483), smoke light and dark, confirm
  12,515,598 and no water-mask RangeError, ping Anita. Delete world_run `*-backup` folders (4.2 GB), data/frozen
  20261104-131517 and -182622 after Europe's freeze, pre-T2302 once clean.
- Waiting tasks: T-2523 + T-2524 (load messaging, stall profile; after bubbleload's findings), T-2528 river width
  (after T-2520), T-2512, T-2517, T-2518, T-2510.

## Anita 2026-11-05, in her words where it matters
- Overnight: "feel free to keep going with stuff after europe"; cores up and past the context cap overnight
  (recorded; maps/CLAUDE.md line committed 0221ad6f, "fine to commit").
- Diagram: borders "look great!! big improvement"; rivers (T-2516) "rivers look better! i think we could do a bit
  more with regards to reflecting how thick the water is, but its pretty good" (T-2528). "ok, we can do 4 hours and
  layout jobs first."
- Batch notes/todo-2026-11-05a.md (T-2519..T-2526).
- Next free id **T-2550** (T-2547 unused).


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
