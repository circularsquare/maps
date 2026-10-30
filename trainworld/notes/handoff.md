# Orchestrator handoff, trainworld

Orchestrator: session c9a4388f-3ba1-4290-bb0a-d9ad50756061, name maps-18, since 2026-10-30 11:22
Previous: session f537e44c-15c4-40d9-bcb8-a6d7cafc4ace, name maps-da, handed over 2026-10-30 11:22, finishing in-flight work
Updated: 2026-10-30 11:30
maps-da (11:25): maps-18 does all landings, 8800 rebuilds and the deploy from now. t1857, t1854b and t1859 still report
to maps-da, which relays each outcome to maps-18 to land, and messages when the last has reported.

Read trainworld/CLAUDE.md and TASKS.md first. Landings are in ARCHIVE.md and `git log`.

## State
- master a05c453d (evening batch). 8800 at a05c453d, smoke 20/20 both shades (rail trips 11,392,296, T-1874's
  figure); t1832-keep passes on a dev Vite (it imports source, so never against 8800).
  t1848 has DECISIONS and code-map lines written but has not reported; left out of commits until it does.
  network.ts and code-map.md were staged partially (t1848's and t1854b's hunks left out) by
  maps-18 scratchpad stage_t1859.py; do the same for any file two agents share.
- LIVE site still 2026-10-28g (= 1550be70). **Anita approved a deploy (2026-10-30)**: do it as soon as t1857's Australia
  run has ended (its join may rewrite market packs mid-run). Include Australia only if shipped and smoked; otherwise the
  committed state. After deploying, check the trainsworld page has no Google Analytics (deploy.py writes it without; the
  website repo's T-1833 change is not committed or pushed). The push step may be refused by the auto-mode classifier
  (it refused shonei-server's deploy.ps1): then hand Anita the exact commands.
- Deploy recipe: deploy.py --version <v> --smoked; website add trainsworld/ trainworld/, commit "trainsworld",
  push; gh run watch; deploy.py --verify --page --version <v>; rclone purge the old R2 prefix.
- shonei-server 5719c6f deployed by Anita (day/week backups; deleting is for good), not pushed.
- 8800 rebuild: `cmd /c C:\Users\anita\AppData\Local\Temp\claude\c--Users-anita-projects-maps\c9a4388f-3ba1-4290-bb0a-d9ad50756061\scratchpad\snapbuild.cmd <explicit master hash>`
  (log to a file; the script is 8 lines: checkout --detach in .tw-snap, CARGO_TARGET_DIR=.tw-snap-target,
  CARGO_BUILD_JOBS=2, cargo on PATH, npm run build in .tw-snap/trainworld/app), then smoke on
  http://localhost:8800/trainworld/ light and SMOKE_SHADE=dark. Never pass HEAD.
- World: 1,897 shipped (+ Australia when t1857 lands), 18 held, 5 parked (nic lez meo doo lel).
- Cores: cap 12 while Anita is up (keep 4 free).
- Uncommitted and not mine: t1857's (realms.json, Credits.tsx Australia line, pipeline/australia.py, arcgis.py,
  world_plan.*) and t1854b's (terrain.ts, terrainTint.ts, cost.rs, params.rs, zone_terrain.py, terrain_fine.py,
  tools/aws/launch.py + userdata/terrain.sh, checks/t1270-terrain.mjs, t1854-valley.mjs, notes/T-1854.md). Commit each
  only when its agent reports. Anita's TODO.md: never commit.
- Next free task id: T-1880.
- T-1857 Australia landed by maps-18 (28 of 28 shipped, world 1,925). T-1876 to T-1879 filed from its report.
  Deploy next. t1854b told (via maps-da) to hold data/packs and data/terrain writes until the deploy is done; it may
  already have rewritten data/terrain 12:39-13:03: check before deploying.
- Old tree .tw-trees/t1734 kept for later thought (Anita). Localhost needs `?acct=live` once per browser for accounts.

Anita away from ~evening 2026-10-30: cores cap raised to 15 (`cores.py cap 15`); lower to 10 when she's back.
T-1868 done too (uncommitted; TopBar.tsx MapSky, style.css, DemandSelector.tsx, Search.tsx, KeepDot.tsx, main.ts,
checks/t1832-keep.mjs not run: run it at landing). Land with T-1872 in the batch.
T-1871 done too (markData.ts, gradeMarks.ts, ui-map.md, code-map.md). String for Anita: "Tunnel portal down to -2" ->
"Ramp down to -2". Her call: drop triangles point downhill (toward their colour's level), ask if she wants uphill.
T-1869 done too (dragHandles.ts new, nodeDrag.ts, endMarkers.ts, blueprintBox.ts, one style.css rule, docs, check
t1869-drag.mjs). Ask Anita: a platform-end drag moves the whole station; pivot/stretch needs a WASM change.
T-1874 done too (sim/src/demand lod.rs, everyday.rs, drives.rs, tests; docs demand-sources/runtime/worldwide (only
its "Everyday samples" entry: the rest of demand-worldwide.md is t1857's); notes/T-1874.md). Columbia 16 -> 93 cells
of uneven size, network rail trips -0.005%, worst line -2.2%. WASM rebuilt; main tree already runs it.
T-1870 done too; the whole evening batch (T-1868 to T-1872, T-1874) landed in one commit by maps-18.
T-1875 filed (t671-ghost-hover near track ends, not from T-1870). New string to show Anita: "The track here is too
short for this platform." (t1870's own wording). Still waiting: maps-da's t1857 (Australia) and t1854b (terrain).
T-1872's cloneName changed by maps-18 per Anita ("X clone 1" -> "X clone 2"); typecheck at landing.
**Anita 2026-10-30 evening: at 94% of the week's usage. No new tasks; land what is in flight, then deploy.**
Held until next week: T-1873, T-1867, T-1849, T-1863, T-1864. T-1872 is done (uncommitted; tools.ts, DECISIONS,
code-map, ui-panels shared with running agents): land it with the rest in one batch once those agents report.
Her hills question (Tokyo, Nagoya, Shanghai centres look hilly) relayed to t1854b via maps-da: likely buildings in the
surface model; bare earth should fix it; check it in T-1854's report.

## In flight
- T-1857 Australia (agent t1857, Opus): world run detached, 9 cores, notes/T-1857.md (batch, log, restart). Credits line
  added: "Australian Bureau of Statistics, 2021 Census and statistical areas (CC BY 4.0)." (show Anita when landing).
- T-1854 finer terrain (agent t1854b, Opus): AWS run for a compact 115 m slope intermediate (stops for Anita if > ~$15),
  then 115 m bare-earth classes, recalibrated thresholds, sim/app change; waits for t1857's run before writing data/packs.
  Anita: 60 MB memory and a few hundred MB on R2 fine; asked about offline play (answer once known). T-1856 hexes after.
- T-1858 dropped to P3 (Anita: after the first city the rest load fast, "that's fine"); agent stopped.
- T-1859/T-1860 landed in 8df66c28. Question for Anita: keep the old straight-segment rule beside the drawn-line one
  (dropping it splits Grand Central, Penn, Atlantic Terminal, Times Sq, Franklin Av)? T-1861 filed (P3).
- (maps-18) Evening batch, all verbatim in notes/todo-2026-10-30.md: T-1868 top bar follow-up (agent t1865 resumed,
  Sonnet), T-1869 dragging curve middles / station ends / multi-select speeds (t1869, Opus), T-1870 station placement
  snapping + two refusal texts (t1870, Opus), T-1871 ramp triangles (t1871, Sonnet), T-1872 small batch: delete
  highlight, headway rounding, clone names, start on biggest city (t1872, Sonnet). T-1873 (muting worm) after T-1848.
  tools.ts is shared by t1869/t1870/t1872: stage by hunk when landing.
- T-1865 landed (top bar: cash after the share in a 4.7em box, weight 500, calendar icon onto the pink dot; zoom 7
  world/city; station display icon; World divider). Shown to Anita: one hover moved, no new strings.
- (maps-18) T-1866 answered (notes/T-1866.md); fix 1 drawing-only = T-1867 (Rust, own tree, after Australia);
  fix 2 (school sampling rule) asked of Anita.
- T-1848 corridors landed (sharp moves Tokyo 73 -> 20; Forest Hills' yellow band still outside its box, a recorded
  trade-off; rebuild +100-260 ms). T-1873 (muting worm) can start now.
- (maps-18) T-1874 uneven sample homes, Anita's pick for T-1866 (agent t1874, Opus, sim/src/demand). T-1867 on hold.
  Anita confirmed T-1859's straight rule stays.
- (maps-18) T-1850 design done (archived): the Tokyo re-solve is settle noise, not recomposition. Stage 1 T-1862
  landed (Ceuta edit in Tokyo: demand final in 2.2 s, 10.4 s as two edits, was 158 s). Not yet run: a browser load with
  `?demandWorkers=3` (cores were short); do it once Australia frees cores. T-1863 (sim/src/track, after t1854b), T-1864 (P3).
- Australia run checked 11:22: pid 35444 alive, region 1 (NSW/Qld/Vic extracts read), then join.cmd.

## Next
1. Deploy (above). Then T-1858's recommendation to Anita.
2. T-1849 (Rust, own tree) once Australia's run frees its cores. T-1863 once t1854b is out of sim/src/track.
3. T-1794 train-count settle (proposal in notes/T-1407.md "Left"). T-1845 only if a gate hangs again.
4. Southeast Asia next: Cambodia, Malaysia (T-1738), when cores are spare.
5. Waiting on Anita: bus design (T-1445). Line groups stage 4 (T-1791) kept as P3.

## Anita's calls today (in DECISIONS/notes)
- Text review: all 66 answers applied (notes/T-1840-answers.md); new text rules added to her global CLAUDE.md
  "User-facing text"; trainworld/CLAUDE.md: hovers only for what isn't written out, negative money red.
- "Recently deleted" removed everywhere (deleting is for good). Keep card became a yellow dot left of search.
- Corridor ribbons off until T-1848; her read: a line joining from the side pairs with one parallel line, then both.
- Expand platforms looks good; laying track feels good. Clock cap 4x now lifts once the viewed zone is final.

## Anita's standing asks
- Player-facing text: agents propose wording per the style rules and list every string old -> new; land without
  waiting; show her the list; she says "text acknowledged" or names lines. Never ask her to write text from scratch.
- TODO.md short-term = batch of prompts: copy item text verbatim into briefs (trainworld/CLAUDE.md); save a batch
  verbatim in notes/todo-<date>.md when filing it.
- Tokens tight this week (to ~11-01): token-efficient over compute-efficient; Sonnet for settled work, Opus for
  design, data correctness, unknown bugs.
- Every demand task (loosely perf) has a "done when"; stop once met. No single-pair demand tuning.
- Performance: approximations welcome when a player can't tell; name them in DECISIONS.md.
- Buses settled: no more tuning unless she asks.
- Thanks to agents in my own reply, not by message; final checks in the foreground; browser loads count their
  demand workers in core claims.
- Someday tasks are real open tasks.
