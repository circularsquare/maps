# Orchestrator handoff, trainworld

Orchestrator: session 7c3c98d5-ca3b-4748-b736-bac9826c72c1, name maps-a7, since 2026-11-02
Previous (nothing in flight, no wait needed; it is not in ListAgents): session c9a4388f-3ba1-4290-bb0a-d9ad50756061, name maps-18, handed over 2026-11-02
Updated: 2026-11-02 (batch landed; country speedups running)

Read trainworld/CLAUDE.md and TASKS.md first. Landings are in ARCHIVE.md and `git log`.

## State
- master: 0238b24d. 8800 at 0238b24d, smoke 20/20 both shades (2026-11-02). Everything from Anita's 11-02 batch that
  was built is landed (see ARCHIVE top). No trees open (t1895, t1898, t1902, t1924, t1929 landed and removed).
- 8800 rebuild: my script `C:\Users\anita\AppData\Local\Temp\claude\c--Users-anita-projects-maps\7c3c98d5-ca3b-4748-b736-bac9826c72c1\scratchpad\snapbuild.cmd <hash>`
  through `cmd /c` with output to snapbuild.log beside it; then smoke on http://localhost:8800/trainworld/ light and
  SMOKE_SHADE=dark. Never pass HEAD. My dev Vite: `node trainworld/checks/dev-nowatch.mjs 6104` (background).
- **LIVE (2026-10-31): c7c5785a on R2 `trainworld/2026-10-30b`**. Not deployed since; publish only on Anita's word
  (`deploy.py --skip-data --version 2026-10-30b --smoked`, website commit/push on her word, verify).
- Commits: when two agents' edits share a file, I either wait until both report or stage by hunk (scratchpad
  stage_clock.py is an example). Tree landings rebase; a conflict stops in the tree (resolve there, `rebase
  --continue`, run the tests, land again).
- World: 1,925 shipped, 18 held, 5 parked (nic lez meo doo lel). Cores cap 10.
- Disk: ~33 GB free after deleting world_run's old join backups (Anita OK'd 11-02). sim/target/debug (4.3 GB) can go
  between rounds with no main-tree agent building.
- Build numbers (Anita 11-02): the start menu shows "build N" bottom left, N = `git rev-list --count HEAD` of the
  built commit (vite.config.ts `__TW_VERSION__`; "dev" on a dev server). Name "8800 is at build N" in every report
  about what landed. Mine, uncommitted: vite.config.ts, StartMenu.tsx, style.css, ui-panels.md.
- Next free task id: T-1999 (ranges up to T-1998 handed out below).

## In flight (maps-a7)
Anita 11-02: "this week i'd like to try adding more of the world"; she approved the speedup plan (T-1974, T-1975
notes). Running, all in the main tree:
- t1976 DONE 11-02, uncommitted: pipeline/pair_files.py index (data/work/pair_files.index.json); register one row
  493 s -> 12 s, join market_zones (AU) 709 -> 119 s; byte-identical; 54 unused files (374 MB) moved to
  data/work/retired/; heal reads every pack; terrain cleared with row caches; port check at go(). checks/runner_fixes.py.
  First market step after landing redoes every row once (3-4 min; the record key changed). Filed T-1988.
- t1977 DONE 11-02, uncommitted: zone_outlines --jobs + cache in data/work/zone_outlines_cache (byte-identical; cold
  4.4 min at 4 jobs from 19.4, unchanged rerun 21 s, one zone changed ~45 s). Any edit to zone_water.py empties the
  cache. Memory per worker at --jobs 10 unmeasured. Filed T-1992 (cold run cheaper).
- a37e361794b90a63b t1984 (Opus): T-1982 country_preflight.py, T-1984 planner defaults, T-1985 drive factor fails
  only above 3.0 (my call). Dry runs only. ids T-1994-1996.
- t1983 DONE 11-02, uncommitted: tools/country/ kit (setup_sim, backup, restore, setbuilt, stale, start_hidden,
  waitany); docs/adding-a-country.md (playbook), docs/world-run.md, docs/adding-a-city-by-hand.md (adding-a-city.md
  deleted; links fixed incl. comments in world_run.py, world_plan.py, zone_outlines.py, pack_gate.py, which other
  agents are also editing). Filed T-1997 (first country run on the kit). Commit with t1976/t1977/t1984 when they report.
Anita 11-02: "yes fine to keep going with adding countries". After they land: Cambodia and Malaysia (T-1738) on the new playbook (a Sonnet agent per T-1975, Opus if a module),
with T-1979 (World Bank fetch once) and T-1981 (solve scaling measure) alongside; T-1978 before Mexico; T-1980 + AWS
before Indonesia/India/Brazil/replans. Anita asked why global sources first (answered 11-02: decide sources before
the run; a ready census hook goes in the first pass; jobs/flows modules later).
Not started from her 11-02 batch: T-1896 gap highlight, T-1903 demand-bubble total, T-1904 demand speed, T-1907 map
drawing; follow-ups filed: T-1916, T-1917, T-1920, T-1936, T-1937, T-1944 (Midtown priced as mountains), T-1948, T-1949,
T-1966 (flat crossings in the train guard), T-1967, T-1970.

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
