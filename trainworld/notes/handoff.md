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
- a5dd38be8410cf911 t1964 (Opus), main tree, port 6130: stations across several pieces of track. T-1970-1973.
- Disk: only ~21 GB free on C: (11-02): no new worktrees; remove t1929/t1898/t1902 trees as they land.
- Mine, uncommitted: tangent line dotted and as faint as the circle (style.css, ui-building.md); Anita 11-02 OK'd junction
  and terminal prices following the level table, the capacity gauge, 12 px join reach.
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
