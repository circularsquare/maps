# Orchestrator handoff, trainworld

Orchestrator: session 5f661fa7-7343-4c77-9e53-f444fc8a7bd5, name maps-5c, since 2026-11-04 09:05
Updated: 2026-11-04 13:30 (maps-c9 wound down ~13:15; its lines folded in below)

Read trainworld/CLAUDE.md and TASKS.md first. Landings are in ARCHIVE.md and `git log`.

## State
- master **314776ed** (maps-c9's last: US/PR replan T-2301 shipped; it also swept in tidy-docs/tidy-checks' staged
  `git mv`s of notes/ and checks/old/, ~466 paths: they were final, nothing lost). Before: eac6f326 panels2, 5b825d59
  crowd, e1076e5a trackbar, 695a441b countries2's plans, 7c7b87b5 morning notes.
- **8800 = build 891 (1debc265)**, plain from data/packs, smoke 23/23 light and dark; real save 12,517,619 rail
  trips. LIVE = 821. Publish only on Anita's word (release "later this week").
- **Cores cap 15** (Anita 08:05: "using lots of cores is fine"). Disk ~25 GB free.
- Worktrees: t1734 (kept), reach, bldgalign, home (all done, rebased on 314776ed, to land), panels3 (agent working).
- **Pack lock: maps-5c's**, held by agent countries3 once it gets "go".

## Anita today (2026-11-04), in her words where it matters
- 09:05 cleanup: "we might be overdue for a code and documentation cleanup ... have some agents look around the
  project and plan this out"; ~10:30 "feel free to make decisions among yourselves regarding reorg".
- T-1920 yes ("ah yes, lets do it"). Drives: "yes we can remove drives and close it" (T-2384). One Average-load basis
  for train/bus/boat (T-2373). Split bar: measure first, "report both trip count and by km", goal ticks stay a share
  of trips (T-2364). Home/All cities strings: "yup text fine".
- Her batches: notes/todo-2026-11-04a.md (morning, T-2333..47), -04b.md (T-2363..65), -04c.md (T-2386..95).

## In flight (all maps-5c's)
- **countries3** a98664e86d75b14a3 (Opus, port 6194, ids T-2398..T-2402): RU/BY join + T-2054 chain, then T-2341
  Jakarta, then T-2344 Ukraine (its uncommitted: pipeline/world_plan.py, docs/zones.md, docs/data-register.md,
  notes/T-2295.md, notes/t2344/). Ready to write RU/BY; **waits on my "go"** (after the three landings below). Claim
  8 cores at go. Its join re-gates without a browser, so app landings don't disturb it.
- **Landed ~13:45**: 401fd11d cleanup + mixed docs; 6039c1b0 reach (T-1920, main WASM rebuilt); 3b795c19 bldgalign
  (T-2363); 3cf52fc0 home (T-2389/T-2395). Trees removed; typecheck clean on master. **Not yet**: smoke on master
  (machine at 100% CPU: four measuring/UI pages over their claims plus the join; asked them to keep to claims) and
  the 8800 rebuild (wait for the RU/BY join to finish writing packs, or use the freeze).
- countries3's RU/BY join running since ~13:30: cmd 35764, python 37000, --jobs 8, log
  artifacts/trainworld/countries/t2295-join.log, prejoin backup done (1.7 GB).
- **panels3** a67f7bd7841ce0a23 (Opus, tree, port 6192): T-2392 Loading..., T-2387, T-2391, T-2397, T-2390, T-2394.
- Done, retired: splitkm (T-2364 archived; T-2365 design shown to Anita, review/split-by-length.png, waits on her),
  lagblue (T-2388 archived; fixes T-2403 running, T-2404 later).
- af04bc3536fc3910f **lagdemand** T-2386 (6190): one more Japan run at 4 cores when free, then reports.
- a7f3f79599d445573 **lagfix** (Opus, tree .tw-trees/lagfix, port 6195): T-2403 blueprint lag fixes 1-3, 5.
- Anita ~14:30 on the diagram (notes/todo-2026-11-04d.md): ac3669604eb2d8174 **diagpar** (Opus, tree .tw-trees/diagpar,
  port 6196) T-2405 no line hidden under another, parallel strokes. diagscale done (T-2406 archived; T-2407 stage 1
  waits on Anita's pick; review/diagram-blocks.png, diagram-today.png, diagram-country-level.png).
- **8800 = build 891 (1debc265)** ~15:30: has panels3 (441780dc), home, bldgalign, reach; smoke 23/23 light and dark on
  8800 and on master's Vite. Built mid-RU/BY join: the join writes only new RU/BY packs (not in the committed
  registry) and global files at its end, so the build's packs are the committed set.
- Anita ~16:00 (notes/todo-2026-11-04e.md): diagram = one unified view with city blocks (T-2407 go), faint real-shaped
  city borders inside blocks (T-2168), water across blocks (T-2169), thin lines fade not vanish (T-2408, queued to
  diagpar). Split bar: Other cities own row; hover with small mode bars; by-length as a graph (area = trips; x or y,
  linear if legible); rename "Within". Launched a614e73f8869f042b **diagblocks** (Opus, tree, port 6199, T-2407) and
  a0701e43d8c50c68d **tripgraph** (Opus, tree, port 6188, T-2365 stage 1: kernel bins by length + mockups, WASM change
  OK in its tree; UI after her look).
- Later: lagfix landed d295f405 (T-2403: ~450 -> ~190 ms freeze); **8800 = build 893 (8621b7dd)**, smoke 23/23 light
  and dark. lagdemand done (T-2386 archived): launched a6dd5545d2de795d1 **solveend** (Opus, tree .tw-trees/t1734
  rebased by it, port 6187): T-1734 land-ready, then T-2410 new line skips the local solve. T-2409 (estimate + clock
  caps + strings) waits on Anita's 1x-or-4x call. RU/BY join: in `step intercity --solve` (pid 41468) since ~11:14
  local clock, no log output while it solves.
- diagpar landed 83dee708 (T-2405, T-2408); **8800 = build 895 (61c7dd29)**, smoke 23/23 light and dark. diagblocks told
  to rebase. Anita: T-2409 caps at 4x only, strings "Calculation 1 day behind", "Time capped at 4x while updating
  demand"; queued to solveend after T-1734 and T-2410.
- tripgraph done (T-2365 stage 1 landed adf36414: kernel `trip_lengths_day`, main WASM rebuilt; retired at 471k).
  Stage 2 (UI) waits on Anita's pick: review/trips-by-length-log.png (its pick), -vertical, -bars, split-hover.png.
- Anita evening notes (notes/todo-2026-11-04f.md, T-2411..T-2418). Running: a99a2b6b0ed498dee **uismall** (Opus, tree,
  6190: T-2415 wrench, T-2416 diagram exit button + Esc, T-2412 track-under-station removal); a1cf37636d38484e7
  **ramps** (Opus, tree, 6191: T-2414 bridge ramps); a1a42c71ac616cd89 **carwater** (Opus, read-only, 6193: T-2418
  Liberty/Governors driving + car across water; islands fix waits on my go); abda6df47a7705c4e **hsrdelay** (Opus,
  read-only, 6192: T-2413); a27d271723090d34a **profit** (Opus, read-only, 6195: T-2411 design). T-2417 (diagram
  stretched to Philadelphia) queued for diagblocks after T-2407. Islands T-2256/T-2216 after carwater reports.
- hsrdelay done (T-2413 archived): fix T-2419 waits on Anita (overtake plan for every shared stretch).
- Next free id **T-2420**.
- Cleanup T-2359: stages T-2366..T-2384 in TASKS "Code health", order in notes/T-2359.md. Done: T-2366, T-2367
  (surveys T-2360..62 archived). Next after the landings: T-2368 code-map, T-2369 app dead code, T-2370 flags,
  T-2376 docs, T-2379 sim small items (own tree), T-2298, T-2384 drives removal (own tree).

## Next
1. Land reach, bldgalign, home (agent-tree land each; reach rebuilds main WASM), smoke, commit docs, "go" to
   countries3, 8800.
2. File replan2's follow-ups: TASKS header's "US replan waits for Anita" line; `pair_files.py unused --move` for old
   US units' joint files; its gates used the 11405b29 sim copy.
3. As reports come: panels3 land; splitkm/lagdemand/lagblue designs to Anita; countries3 commits per job.
4. Cleanup stages as files free up (above). Then T-2302 Europe replan (after countries3's queue).
5. Others when cores allow: T-2324 walk-rule cost, T-2300 Taiwan Central Range, T-2296 census plan, T-2321 diagram,
   T-2332 colours, T-2326 AWS seed, T-2216/T-2256 islands, T-2396 dock height.

## How to
- Scratch = C:\Users\anita\AppData\Local\Temp\claude\c--Users-anita-projects-maps\5f661fa7-7343-4c77-9e53-f444fc8a7bd5\scratchpad
  (snapbuild.cmd, snapbuild-frozen.cmd, keep_hunks.py, mark_taking.py `<tasks.md> T-x=agent ...`,
  common-brief-tree.md for tree agents, lowdisk.ps1). A new orchestrator copies these and fixes the path inside
  snapbuild-frozen.cmd.
- 8800: `cmd /c "<scratch>\snapbuild.cmd <hash> > <scratch>\snapbuild.log 2>&1"` (or `snapbuild-frozen.cmd <hash>
  20261103-162459` while packs are being written). Then smoke on http://localhost:8800/trainworld/ light and
  SMOKE_SHADE=dark (claim 4 cores). Build number = `git rev-list --count <hash>`.
- Pre-commit smoke: own Vite (`node trainworld/checks/dev-nowatch.mjs 6197`), smoke; stop it by its port's PID.
- Landing a tree: `git -C <tree> add --all -- trainworld`, commit "agent-tree: work in progress", `rebase master`;
  for sim changes both cargo runs and `npm run wasm` in the tree; typecheck and smoke; then `agent-tree.mjs land
  <name> "<sentence>"` and `remove`. ff fails if the main tree has uncommitted edits in a file the commit touches:
  commit those first.
- **Commit with a pathspec**: `git commit -m "..." -- <paths>` (other sessions and agents stage too).
- Mixed files: `python <scratch>\keep_hunks.py <maps root> <path> <marker>...` (-U0).
- Docs lint: `venv\Scripts\python.exe trainworld\checks\doc_lint.py` (exit 1 on findings).
- Publish: `freeze.py build <stamp>` with TW_MARKET_DRIVES set, `deploy.py --dist <dist> --version <v> --smoked`,
  website `git add trainsworld/ trainworld/`, commit "anitabuilder", push (her word), `deploy.py --verify --page
  --version <v>`. Afterwards prune publish/ and data/frozen/ to the current build and the latest published ones.

## Open questions to Anita (asked, unanswered)
- T-2267/T-2270 design leftovers: optional cross "Stop (Esc)", 3 px coloured bar edge, T-2285 Shift+1-7, mode
  colours (notes/for-anita.md "Open now").
- Fares: optional hover "Riders pay the old fares until demand has settled"?
- Strings maps-1c listed (Ferry tool T-2194, building tool T-2218, crowding T-2236, diagram T-2163, ferry demand rows
  T-2195, building hover T-2220): not acknowledged as far as known.
- Tell her after countries3: Sukabumi+Cianjur one zone (could be two); Karawang out of Jakarta; ISW line not an open
  licence (only whole units left out, nothing of it drawn) before publish.

## Lessons from this run
- `git add <paths>` then `git commit` also commits whatever another session already staged (314776ed swept in ~466
  staged moves). Commit with a pathspec.
- A gate's browser checks run on a Vite over the main tree: an app edit or `npm run wasm` mid-gate makes every page
  throw. Hold app/WASM landings while a browser gate runs (joins re-gate without a browser), or T-2298.
- Tree agents' pages read the main data junction: a tree made before a replan's game json is committed asks for
  retired packs ("New York could not load", packs served as HTML). Not a code fault.
- `world_run.py step <x> --help` may run the step for real (seams rewrote borders.json once).
- Before starting an "Open" task, `git log --grep "T-<id>"`. Her TODO.md can repeat an earlier batch: diff first.
- Any new save block goes after BLD1 at LEVEL 5. `cargo test` without `--features demand` skips the demand tests.
- C: can fill suddenly: check `Get-PSDrive C` before long runs.
- Agents near 450-550k tokens: retire after their report. An agent "waiting on its own background work" re-sends
  its report when that work ends: ignore repeats.
- When Anita asks for cores free, check the real load (Get-Counter), not only the ledger.

## Anita's standing asks
- Player text: agents propose wording per the style rules, list every string old -> new; land without waiting; show
  her. Zone, region and realm names need no review.
- TODO.md short-term = prompts: copy her words verbatim into briefs; save batches in notes/todo-<date>.md.
- Every demand task has a "done when". Buses settled. Thanks to agents in my own reply. Keep the work moving.
- Disk: delete orphaned temp files always; keep only the current build and the latest published ones in
  artifacts/trainworld/publish/ and data/frozen/; delete superseded backups once a later state checks clean.
