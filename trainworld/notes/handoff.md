# Orchestrator handoff, trainworld

Orchestrator: session 7c3c98d5-ca3b-4748-b736-bac9826c72c1, name maps-a7, since 2026-11-02
Previous (nothing in flight, no wait needed; it is not in ListAgents): session c9a4388f-3ba1-4290-bb0a-d9ad50756061, name maps-18, handed over 2026-11-02
Updated: 2026-11-02 (two rounds and two trees landed; six agents running)

Read trainworld/CLAUDE.md and TASKS.md first. Landings are in ARCHIVE.md and `git log`.

## State
- master: 0681dba2 (11-02 batch rounds one and two, T-1924/T-1925 and T-1895 landed). 8800 at 0681dba2, smoke 20/20
  both shades (2026-11-02). Uncommitted since: my ARCHIVE/TASKS moves of T-1895 and T-1900.
- 8800 rebuild: my script `C:\Users\anita\AppData\Local\Temp\claude\c--Users-anita-projects-maps\7c3c98d5-ca3b-4748-b736-bac9826c72c1\scratchpad\snapbuild.cmd <hash>`
  (checkout --detach <hash> in maps/.tw-snap; CARGO_TARGET_DIR=maps/.tw-snap-target; CARGO_BUILD_JOBS=2; cargo on PATH;
  npm run build), run through `cmd /c` with output to snapbuild.log beside it. Then smoke on http://localhost:8800/trainworld/
  light and SMOKE_SHADE=dark. Never pass HEAD. Checks that import source (t1832-keep) need a dev Vite, not 8800.
- My dev Vite: `node trainworld/checks/dev-nowatch.mjs 6102` (background); restart on a new port after app edits.
- **LIVE (2026-10-31): the build of c7c5785a on R2 data `trainworld/2026-10-30b`**; website cef6eb1. A code-only release:
  `deploy.py --skip-data --version 2026-10-30b --smoked`, then website add/commit/push (only on Anita's word), `gh run
  watch`, then `deploy.py --verify --page --version 2026-10-30b`. A data change needs a new `--version`. Not deployed
  since; publish only when she says.
- R2 holds only `trainworld/2026-10-30b`. Purges are Anita's (`rclone purge r2:anitamaps/trainworld/<v> --s3-no-check-bucket`).
- Shared files: when two agents' edits sit in one file, stage by hunk (HEAD text plus one agent's pieces, `git hash-object -w`
  + `git update-index --cacheinfo`; examples stage_*.py in the old scratchpad
  C:\Users\anita\AppData\Local\Temp\claude\c--Users-anita-projects-maps\c9a4388f-3ba1-4290-bb0a-d9ad50756061\scratchpad).
  So far I've avoided it by committing a round when the agents sharing files have all reported.
- World: 1,925 shipped (Australia 28 of 28), 18 held, 5 parked (nic lez meo doo lel).
- Cores: cap 10 (checked 11-02).
- data/work/pre-T1854/terrain (58 MB): delete once T-1854's terrain is seen clean in play. data/work/terrain-fine/
  (3.1 GB): T-1854's intermediate, kept for threshold tweaks.
- Old tree .tw-trees/t1734 kept (Anita). Localhost needs `?acct=live` once for accounts.
- 11-02 later: master afb75407 (T-1926 crowding strip 9ad82dee, T-1902 trip explorer cc3972d2, T-1906 fares/loans
  afb75407); 8800 at afb75407, smoke 20/20 both shades. Smoke's "far new game" Tokyo riders fell 76,870 -> 15,591 with
  T-1906 (a New York start charges US fares in Tokyo at 1.00x): asked Anita.
- t1899 stop logic DONE 11-02, uncommitted in the main tree (archived; no strings). Its edits share wasm.rs, params.rs,
  clock.worker.ts, protocol.ts with t1964/t1965: commit the main tree once those two report, then land the trees.
- Trees t1929 (road crossings) and t1898 (blueprint selection) DONE, waiting to land: the main tree's uncommitted
  sim/src/track/wasm.rs (t1964's) blocks the fast-forward. Land after t1964 commits; both add WASM functions.
- Disk (maps-d6's request 11-02): world_run's old join backups deleted on Anita's word (12.0 GB; kept 20261030-134355).
  sim/target/debug (4.3 GB): clear only between rounds with no main-tree agent building.
- Mine, uncommitted 11-02 (Anita's text review): "Trips possible by train" (MarketTrip.tsx, ui-inspectors.md), loan hover
  "Interest would be $X per day. Average profit over the past 7 days was $Y per day." (MoneyTab.tsx), bus average-speed
  hover dropped (BusInspector.tsx, it said "less the turning"). New style rule in her global CLAUDE.md: never "less" for
  minus. Asked her about "Car for these trips" (explained; proposal pending).
- t1906 follow-up DONE 11-02, uncommitted (scales["@"] = start country's income scale; Tokyo in smoke's far game 42,288,
  was 76,870 before T-1906, since all start fares are 2.5x the old). Anita 11-02: fares scale flat (whole curve), as built.
- Mine, uncommitted: trip panel order "Trips possible by train", "Effective time", "Car on the same trips" (Anita OK'd);
  no hovers on the first two; car hover "Effective time by car on trips possible by train."; a closed far end "No route."
- t1906 resumed 11-02: each country's default fare scale = its own income level relative to the start (^0.7 kept;
  Anita: "maybe even lower tbh but dont want linear").
- Next free task id: T-1974 (ranges up to T-1973 handed to agents below).

## In flight (maps-a7; her 11-02 batch is T-1892 to T-1907, verbatim in notes/todo-2026-11-02.md)
She said 11-02: skew to Opus, Sonnet for routine. Done and landed: T-1892, 1893, 1894, 1895, 1905 (+1932, 1935), 1924,
1925, 1928, 1930, 1931; designs T-1897, 1900, 1901. Her calls 11-02: prices round numbers (+1/-1 = 1.0x, base $75M),
upkeep 20/25/35/40k by structure, water limit 30 km, road crossings $6M (motorway 3), crowding strip in the dot's hover
(no pie, green = no delay), passing tracks someday (T-1927), no parking change (T-1934 abandoned), China x5 / Korea x2.5
attractions, rebuilds later (T-1933 waits).
Agents running (each with an id range for new tasks):
- a173638cfe272a6eb t1902 (Opus), tree .tw-trees/t1902 (from 153ff30f; land rebases), port 6140: trip explorer. T-1920-1923.
- a9d9bc7c54072f5d4 t1929 (Opus), tree .tw-trees/t1929, port 6180: road crossing charge. T-1944-1947.
- a29d2053b6c814ce7 t1898 (Opus), tree .tw-trees/t1898, port 6190: blueprint selection, 320 drag. T-1948-1951.
- t1926 crowding strip DONE 11-02, uncommitted (archived). clock.worker.ts mixes its edits with t1906's (fares, loan),
  so commit both together when t1906 reports (or stage by hunk). wasm.rs and station_sweep_tests.rs are t1964's.
  DECISIONS already holds t1898/t1902 lines for work not landed yet.
- a020af69a11d39802 t1899 (Opus), main tree, port 6120: stop logic (auto HSR, tiebreak, reversing). T-1956-1959.
- a2fb39543e201af98 t1906 (Opus), main tree, port 6150: fare dials 1.0x + starting fare by place, loan cap. T-1960-1963.
- a27084daa2f6582bd t1965 (Opus), main tree, port 6160: drawn trains never overlap on the same track and level (her test
  track: express drawn through the local). T-1966-1969.
- t1964 DONE 11-02, uncommitted (archived; no strings; real save unchanged, it has no plain joints). Main tree now
  waits only on t1965 (shares wasm.rs, params.rs, clock.worker.ts, protocol.ts with t1899/t1964): commit all main-tree
  work when t1965 reports, then land trees t1929 and t1898 (both touch wasm.rs; expect a rebase conflict there).
- Disk: only ~21 GB free on C: (11-02): no new worktrees; remove t1929/t1898/t1902 trees as they land.
- Mine, uncommitted: tangent line dotted and as faint as the circle (style.css, ui-building.md); Anita 11-02 OK'd junction
  and terminal prices following the level table, the capacity gauge, 12 px join reach.
- Anita 11-02: "this week i'd like to try adding more of the world"; first think about build speedups and a streamlined
  country process. a59f655aa8c81f4dc t1974 (Opus, research only): wall time per stage from world_run logs, ranked
  speedups, files T-1976-1981. ac0d3e79f8fced0e3 t1975 (Opus, research only): pitfalls, token costs, a draft playbook
  at notes/T-1975-adding-a-country.md (apply to docs/adding-a-city.md after Anita sees it), files T-1982-1987.
  Next free id: T-1988. t1975 DONE 11-02: notes/T-1975.md, draft notes/T-1975-adding-a-country.md (190 lines vs 710;
  to become docs/adding-a-country.md, rest moves to docs/world-run.md and adding-a-city-by-hand.md, T-1987); filed
  T-1982-1987. Tokens: clean country ~6M, faults decide cost; wakes 25-77% of tokens. Shown to Anita 11-02.
  t1974 DONE 11-02: notes/T-1974.md; filed T-1976-1981. Market register re-reads 900 MB of joint files per row (7-9 min,
  holds the registry lock). Proposed to Anita: before Cambodia/Malaysia build T-1976 (market reads only touched files),
  T-1977 (parallel outlines), T-1982 preflight, T-1983 kit, T-1985 drive >3.0, T-1986 heal by place + busy port, T-1987
  playbook swap; T-1979 and T-1981 alongside; T-1978 before Mexico; T-1980 + AWS before Indonesia/India/Brazil/replan. Next countries after: Cambodia, Malaysia (T-1738), then Mexico, Turkey.
Not started: T-1896 gap highlight (drawing; after t1929/t1898 land), T-1903 demand-bubble total, T-1904 speed (switching
homes/destinations, local before whole-world solve, global vs local label), T-1907 map drawing (hair/skin, overlapping
line clicks, fading, riders on track, rotation).

## Open questions to Anita (asked, unanswered)
- Blueprint row hover wording (T-1890): "Elevated +1: 1.0x", "Mountains +2.5x", "Cost factor 3.0x" to match the tag
  (now "Elevated +1 1", "Mountains + 2.5", "Cost factor × 3"); and "Water +2x" or "Water +2.0x".
- Mountains at grade: kept 3.0x by her additive rule (T-1890 replaced T-1887's 6x).
- Settings label "Lines side by side" vs "Draw lines side by side" (T-1886); Ctrl no-snap leaves the real-line ring on.

## Open problems
- T-1891: "Demand ran out of memory" once on a new NYC game on 8800; not reproducible by her or the agent; the label
  was wrong (any second worker death said "memory"); now "Demand stopped" with the panic in its hover. If it recurs,
  get that hover text or the console line `demand: worker N panicked`. notes/T-1891.md.
- T-1889: riverbanks priced as hills/mountains (East River at 42nd St) from T-1854's classes.

## Next (after her TODO batch)
1. T-1881 hillshade from a surface model shows dense centres as hills.
2. T-1880 buses as feeders (her question). T-1873 muting worm (now unblocked). T-1867 only if she wants it after T-1874.
3. T-1849, T-1863, T-1864 demand speedups; a `?demandWorkers=3` browser run of T-1862's keep.
4. T-1875, T-1876 to T-1879 (Australia follow-ups; T-1879 has a CLAUDE.md line for data/terrain during joins), T-1882.
5. T-1794 train-count settle; Southeast Asia (T-1738) when cores are spare. Waiting on Anita: bus design (T-1445).

## Lessons from this run
- A new row or hover that has an existing counterpart (the local trip panel's Effective time for the market's) copies
  its pattern; Anita 11-02: "theres a clear reference here, i think its bad that we didnt use that". Put the reference in
  briefs for any panel work.
- Look at every UI agent's pictures yourself for spacing before reporting done (Anita, 2026-10-30, the Cities World
  row: "i really feel like you guys should be able to check this"). Measure text-to-rule gaps against the row pitch.
- An agent's tooltip text can wrap mid-range in its own picture; read the pictures, not just the strings.

## Anita's standing asks
- Player-facing text: agents propose wording per the style rules and list every string old -> new; land without
  waiting; show her the list. Never ask her to write text from scratch.
- TODO.md short-term = batch of prompts: copy item text verbatim into briefs; save a batch in notes/todo-<date>.md.
- The tight-usage week (10-26 to ~11-01) is over unless she renews it: tier by her global CLAUDE.md rules.
- Every demand task (loosely perf) has a "done when"; stop once met. Approximations welcome when invisible.
- Buses settled unless she asks. Thanks to agents in my own reply. Someday tasks are real open tasks.
