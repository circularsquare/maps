# Orchestrator handoff, trainworld

Orchestrator: session aa768402-8365-45e0-85b8-0c27fe256637, name maps-93, since 2026-11-05 17:55
Previous: session 4a9d3a95-b5e4-4ac5-ab4b-4ef63d690cc6, name maps-17 (ListAgents no longer shows that name; likely
maps-03 after the VS Code restart), handed over 2026-11-05 17:55, finishing in-flight work (mexico, cachecard,
diagsmooth, watertoggle, sheet)
Updated: 2026-11-05 18:20

Read trainworld/CLAUDE.md and TASKS.md first. Landings are in ARCHIVE.md and `git log`.

## State
- master **79d0c5ee**. **Real save smokes 12,515,594** since farebase (T-2559) landed.
- 8800 = build 1074 (163d9725), smoke light and dark PASSED. LIVE = 821. Publish only on Anita's word (release "later this week"; Mexico and Brazil before release).
- **Cores cap 12** (Anita home ~21:05; was 15 while she was out).
- **Pack lock: mexico's** (T-2436; batch 20261105-182304, python 27744, log artifacts/trainworld/t2436-run.log). Then
  Brazil T-2437. Mexico gates come every 10 built rows (~3 min each; gate 1 18:57-19:00, 10 gated, 0 parked);
  All 105 rows gated (98 + 7 relaxed: Los Cabos pop, three mountain towns jobs share, three desert zones water).
  Join done 21:22: all 105 shipped (3,200 -> 3,305), freeze 20261105-200211. After-steps running (terrain_fine,
  zone_terrain write data/terrain; market_drives), then mexico's load checks and smoke. **Then: commit Mexico's game
  files (registry.json, market-rows.json, zone-names.json, realms.json, world_plan.*), rebuild 8800, smoke, ping.**
- maps-17 (maps-03) relays mexico's and sheet's reports; landing is maps-93's since 18:15 (maps-17's word). Mexico's
  run starts after maps-93's landings; **no app/WASM landing into the main tree while mexico's browser gate runs**
  (it messages maps-17 before a gate; maps-17 asks maps-93).

## maps-93's agents
- Landed ~18:25-18:50: cleana4 97c6e153 (T-2512; T-2575 filed for its item 26 step two), kernelsplit 8696ae95 (T-2381,
  main WASM rebuilt in a gate-free window mexico confirmed), sheet's notes db56555e. **8800 = build 1074 (163d9725)**, smoke light and dark PASSED 18:44, 12,515,594.
- walkcost LANDED 83f433fd (records 55ab534f): T-2324, exact speedups, figures bit for bit; the rule costs 5.6% of
  the calib solve; 3.2% over T-1059 net of other tasks' 5.9% (its reckoning, in DECISIONS). Main WASM rebuilt during
  Mexico's join. **8800 not rebuilt since 163d9725: rebuild after Mexico's join ends** (mixed packs otherwise).
- followtrack LANDED 665dd025 (records 97530112): T-1358, own work 99 -> 54 ms at 6x (0.55, accepted), exact.
- toworld LANDED 3d0d0097 (records 5a4dd65a): T-2576, train-count edit's toWorld 5.9 -> 2.4 ms at 6x.
- **marketsearch** (Opus, tree .tw-trees/marketsearch from 55ab534f, port 6181): T-1647 native Japan pass 15% faster,
  exact.
- **followtrack** (Opus, tree .tw-trees/followtrack from c306d1bc, port 6184): T-1358 followTrack at least halved,
  zones identical.
- **cachehit** (Opus, tree .tw-trees/cachehit from 79d0c5ee, port 6183): T-2574 cache-hit bar, then T-2533 zone in
  view answers within ~20 s of a cache hit.
- Landed 18:15: uismall5 3d0939bc (T-2569..T-2572; also fixed the map credits button, invisible since T-1435, now
  just left of the trouble button), farebase f3b3cb18 (T-2559), records 79d0c5ee (with t409strong's T-2527).
  watertoggle (maps-17) 7bba88c2, cachecard (maps-17) 9a68feb5.
- Strings to show Anita: "Show water" (new hover); "Nothing built" -> "Loading..." while an account game's picture
  loads; "No stations" (new hover, trouble row). Review: artifacts/trainworld/uismall5/attrfix-both.png.
- Next free id **T-2575** (T-2567..68 are sheet's).
- maps-93 scratch: C:\Users\anita\AppData\Local\Temp\claude\dce1fefcc1d43aef\aa768402-8365-45e0-85b8-0c27fe256637\scratchpad
  (snapbuild.cmd, snapbuild-frozen.cmd fixed, mark_taking.py, keep_hunks.py, lowdisk.ps1, common-brief-tree.md).

## In flight (maps-17 winding down; maps-93 took over ~18:20)
- **mexico** (agent aa0915d6048b49d88, Opus, main tree, **pack lock**, port 6197, kit maps-17's old scratch\mexico):
  T-2436 + T-2080/T-2081/T-2070 + T-2554's three pipeline changes. Order: T-2554 changes, wire mexico.py (notes/T-2080.md
  "Wiring left"), dry counts to the orchestrator before building, then build/gate/join/after-steps; it messages at
  browser gates (no app/WASM landings into the main tree during one). Expect pop_vs_plan parks (GHSL 10-15% under the
  census in big cities). **Hand to maps-93.**
- **sheet** (agent a26fa2ea5d56190eb, Opus, read-only on game code, scratch\sheet, writes notes/T-2566.md and
  artifacts/trainworld/sheet/): T-2566 rubber-sheet warp, Anita chose "sheet first"; one-page summary of today's
  diagram, design, Node prototype of New York and all Japan beside today's, open questions for Anita. Nothing enters
  the game until she has seen the pictures. Ids T-2567..68. **Hand to maps-93.**
- **watertoggle** (agent ad851a86eb0be20d5, Sonnet, tree .tw-trees/watertoggle from 4021db95, port 6195): T-2565 water
  button in the diagram view, default off, persisted, off skips water work; hover "Show water". maps-17 lands it if it
  reports before winding down, else maps-93.
- **cachecard** (agent a89d9e3c777e837e0, Opus, tree .tw-trees/cachecard from 40bdc6a0, port 6194): T-2560, a cache-hit
  reload playable at ~25 s instead of ~213 s on Anita's NA save (perfOff=cacheCard gives the old card). Blocker before
  landing: cached vs fresh figure signature differ on the same network (6b5f90ad vs ca3d7bb9); told to explain (a
  T-328-era cache bug gets its own task id from the orchestrator), smoke, t328 prepare/reload/edit, records, then report
  once. Lands only if the difference is explained and not made worse.
- T-2561 paused (diagsmooth retired at 468k): findings in notes/T-2561.md, parked diff in maps-17's old
  scratch\diagsmooth\parked\ (t2561.diff); superseded by T-2566 unless the sheet is dropped.
- Today's landings, 8800 at build 1056 (e0534793, smoke light+dark PASSED 12,515,598), Europe committed e8eb9758, disk
  cleaned (67 GB free), Anita pushed master ~18:00.

## After Europe's join (steps 1-2 DONE; 3 is in mexico's brief; 4 is mexico's run)
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
- ANSWERED ~21:30 (verbatim in notes/T-2566.md "Anita's review"): land keeps its shape, water distorts; cores halfway
  back; more octilinear; line rivers with higher min width, Harlem River; stations on their side of zone borders; no
  demand sizing. **sheet2** (Opus, read-only, scratch\sheet2) prototypes round 2 as T-2577; T-2567/68 wait on it.
- (was) T-2566 sheet (to ask when she is back ~21:10; pictures review/sheet-nyc-core.png, review/sheet-japan.png; notes/T-2566.md):
  1 core size against countryside (agent: halfway back to today's); 2 scale at most double per 40 pitches, right?;
  3 busy places bigger by demand (left out: map would change size with ridership); 4 narrow rivers at real width too
  thin: keep today's thin band only where a line crosses?; 5 real state borders back, a station may sit a few pitches
  on the wrong side once straightened: fine or bend the border? T-2567/T-2568 wait on her go.
- Earlier, answered ~18:15: water is experimental behind a button, default off (T-2565); sheet first, rough
  prototype outside the game (T-2566); "this sort of distortion scheme would solve the issue of region borders not
  having their right shape". Distortion default 0.5 (her pick).

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
