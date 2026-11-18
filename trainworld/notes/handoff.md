# Orchestrator handoff, trainworld

Orchestrator: session 59c85afe-f6fc-46fa-b1d6-fdaa946e7f95, name maps-4f, since 2026-11-18 12:08 UTC
Updated: 2026-11-18 12:30 UTC (maps-4f adopted maps-ca's three agents; their reports still reach maps-ca, which
relays them verbatim. maps-4f started vanish (T-3645) and setstub (T-3648); T-3622's ask restored to asks.json)

A short state page, rewritten (not appended) as things change. The diary is notes/handoff-log.md; older detail is in
git history. Read trainworld/CLAUDE.md and docs/orchestrator.md first. Her local time is UTC-5. She's up (12:05 UTC):
"can keep night runs going til finished". maps-ca's morning report (12:08) gave her the full list of overnight
landings, the strings to check and the asks below.

## State
- v0.4.1 LIVE since ~05:35 UTC (app only on data 2026-11-18). deploy.py --skip-data checks the registry against R2.
- master f039efbc+ pushed. Landed overnight, all archived: T-3636, T-3634, T-3638, T-3627, T-3632 stage 2 (stays
  open), T-3629, T-3589, T-3616, T-3640, T-3641, T-3597, T-3637, T-3643, T-3639, T-3644, T-3474 (co-author Cinnamon),
  T-3511 (night batch 1); T-3433 closed not a bug. agent-tree finish takes --coauthor now.
- 8800 = ff3a04cd on freeze 20261118-015639 (batch 1), smoke PASSED; missing everything after ff3a04cd (T-3640,
  T-3639, T-3644, T-3474) and yhu/suc don't open (the night run's registry edits are uncommitted in the main tree).
  Earlier 8800 builds tonight used plain snapbuild during the run (a slip; use snapbuild-frozen while a run owns packs).
- Unpublished changelog: c001-c014 (`changelog.py list`); c014 is T-3482's, whose code isn't landed yet.
- Up next: T-3604, T-2136, T-1080, T-930 (all after the night run).

## In flight (maps-ca's agents; maps-4f adopts)
- nightfix3 (Opus, MAIN tree, 9 cores, ledger name nightfix3): T-3455 batch 2. Fixed the shapely TopologyException,
  reran 8 parked rows, joined batch 2 (join log artifacts/trainworld/night1118fix-T3455-join.log, status-end 11:27 UTC,
  start_difficulty run); now "Malaysia again" (a rerun of Malaysian rows). Done when its ledger claim is gone and
  notes/T-3455.md "What ran" is updated. HOLD app/ writes and landings in the main tree while it gates. Then: commit
  the night run's game json (registry, market-rows, zone-*.json, world_plan.json, notes/T-3455.md + kit), tasks.py
  done T-3455, snap8800 at master (plain, run over), T-3620, T-3623, T-3604 can start.
- settle (.tw-trees/settle): T-3482 READY, checker 4/5 (the fail is master's undo behaviour, filed T-3646). Land after
  nightfix3: `node trainworld/checks/agent-tree.mjs finish settle "<sentence>" --task T-3482 --outcome "..."` (WASM
  rebuild); its DECISIONS lines and c014 are already in main. Follow-ups in its notes (unfiled): a running line's new
  headway on full track keeps its old rate; data/work/realnet/nyc_real.twt2 no longer loads. T-3645 filed.
- tramoneway (Opus, .tw-trees/tramoneway, port 6194): T-3647 (P1) dattera's Amsterdam tram save broken by 0.4's
  one-way stops; reproducing; fix (checker before landing) or options to her. His real save needs her word for ssh.

## In flight (maps-4f's agents)
- vanish (.tw-trees/vanish): T-3645 built (hand_over skips a trip redrawn at its first stop; Rust test; c014 in
  tree "fixed a train disappearing when a line's train count changed"). Checker PASS (9 Rust sweeps, 3 fail on old
  code; t1071 on the real save; skipped browser save/reload/undo). READY: land after settle; WASM rebuild.
- T-3655 filed (held line on full track keeps its old rate; settle's and vanishcheck's finding).
- setstub (.tw-trees/setstub): T-3648 READY. keep_under_stations used reach + 10 m, single Remove reach + 4 m
  (STATION_TRIM_MARGIN_M); one constant in sim wasm/links.rs + a Rust test; no changelog. The "176 m" was not
  reproduced (only the 6 m per end); no checker (one constant, tested). Land after nightfix3, WASM rebuild.
- helptrip (.tw-trees/helptrip): T-3649 READY (help.qTripMode/aTripMode, 13 languages, screenshot looked at;
  c014 in tree). Strings for her below. Land after nightfix3.
- weekendhw (Sonnet, .tw-trees/weekendhw): T-3650 READY (TS only, world.ts restHeadwayStep + typed weekend headway;
  t3557 passes; typed entry driven; no checker: UI-side, no sim/money). Land after nightfix3. Its changelog item is new.
- T-3652, T-3653, T-3654 filed P3 from her 12:24 pad (Community Newest, time controls, map drawing); not started.
- T-3651 (Zui: passing tracks over side platforms) filed; sent to tramoneway via maps-ca in case it is T-3647's cause.
- T-3622 closed on her 05:10 answer (T-3627 did it); its ask removed.

## Waiting on Anita (asks.json)
- T-3298 walking change limit 300 m -> 600 m? (recommended 600 m for all)
- T-3630 start menu settings A/B/C (artifacts/trainworld/review/t3630-landing.png; recommended B)
- Strings to check were listed in maps-ca's 12:08 report (T-3632, T-3597, T-3637, T-3634, T-3627 car hover).
- CinnamonVII's PR 21: close with a link after the next open-source sync; draft the reply for her first.

## Next
1. Adopt the three agents above; land settle after nightfix3; commit the run's json; 8800 rebuild + smoke.
2. Leftover .tw-trees: touchrule (empty, locked), shenzhen (dirty pipeline/osm_units.py, whose?), tt2 (stale).
- Uncommitted on purpose: TODO.md (hers), notes/T-3397.md, jordan-save-investigation-2026-11-15.md,
  tools/ops/archive_bad_saves.py, changelog/roadmap.json, the night run's files above.

## Her standing asks this week
- Small isolated fixes, open-task count down, merge duplicates; bug investigations give up if hard.
- She edits each update's changelog on the board before publish (unless ~2 items) (docs/orchestrator.md).
- Lean testing: concise time-boxed checkers, no checker for small UI, smoke per batch.
- One phrase per wait in player text: loads / shows up / calculating trips / memory.
- Long jobs in the background; answer her first. No windows on her screen.
- Every publish, deploy and server push needs her yes to that exact action; none overnight.
- Advisor trial: start successors with `next_orchestrator.py trainworld --advisor fable`; log each call in handoff-log.
