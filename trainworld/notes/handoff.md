# Orchestrator handoff, trainworld

Orchestrator: session 7c3c98d5-ca3b-4748-b736-bac9826c72c1, name maps-a7, since 2026-11-02
Previous (nothing in flight, no wait needed; it is not in ListAgents): session c9a4388f-3ba1-4290-bb0a-d9ad50756061, name maps-18, handed over 2026-11-02
Updated: 2026-11-02 late (past 500k context: ready for a fresh orchestrator; only t1738 in flight)

Read trainworld/CLAUDE.md and TASKS.md first. Landings are in ARCHIVE.md and `git log`.

## State
- master: 32fa60e4 (build 786). 8800 at build 786, smoke 20/20 both shades (2026-11-02). Everything built from Anita's 11-02 batch
  and the country speedups is landed (ARCHIVE top). No worktrees open.
- 8800 rebuild: my script `C:\Users\anita\AppData\Local\Temp\claude\c--Users-anita-projects-maps\7c3c98d5-ca3b-4748-b736-bac9826c72c1\scratchpad\snapbuild.cmd <hash>`
  through `cmd /c` with output to snapbuild.log beside it; then smoke on http://localhost:8800/trainworld/ light and
  SMOKE_SHADE=dark. Never pass HEAD. My dev Vite: `node trainworld/checks/dev-nowatch.mjs 6105` (background task of this session; dies with it).
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
  about what landed.

## In flight (maps-a7)
- master 32fa60e4 = build 786; 8800 at build 786, smoke 20/20 both shades (2026-11-02). The speedups and playbook are
  landed (T-1976, 1977, 1982-1987; docs/adding-a-country.md, tools/country/). Uncommitted: nothing of mine.
- a655261edf2119ea3 t1738 (Opus): Cambodia then Malaysia (+Brunei if simple) on the new playbook, first real use
  (doubles as T-1997: fixes the playbook/kit in place); one join if timing allows; CPU sampler during the joint solve
  for T-1981. Main tree; no other agent running, so it owns data/packs and data/terrain during its gate and join.
  Liveness, batch ids and restart commands go in notes/T-1738.md. ids T-2000-2004.
- Then: Mexico and Turkey (T-1978 chunked gates before Mexico; T-1979 World Bank fetch once). T-1980 + AWS before
  Indonesia/India/Brazil/replans. A global-sources country may go to Sonnet once the playbook has run clean once.
- Not started from her 11-02 batch: T-1896 gap highlight, T-1903 demand-bubble total, T-1904 demand speed, T-1907 map
  drawing. Follow-ups filed: T-1916, 1917, 1920, 1936, 1937, 1944, 1948, 1949, 1966, 1967, 1970, 1988, 1992, 1994,
  1995, 1999 (planner floor exemptions for islands/exclaves before any replan). These UI tasks can run while the
  country run goes, as long as they do not write data/; app work is safe.
- Next free task id: T-2005.

## Open questions to Anita (asked, unanswered)
- (From maps-18, possibly stale since T-1928 reworked the price table): blueprint row hover wording (T-1890);
  check the current strings before asking again.
- Settings label "Lines side by side" vs "Draw lines side by side" (T-1886); Ctrl no-snap leaves the real-line ring on.

## Open problems
- T-1891: "Demand ran out of memory" once on a new NYC game on 8800; not reproducible by her or the agent; the label
  was wrong (any second worker death said "memory"); now "Demand stopped" with the panic in its hover. If it recurs,
  get that hover text or the console line `demand: worker N panicked`. notes/T-1891.md.
- T-1889: riverbanks priced as hills/mountains (East River at 42nd St) from T-1854's classes.

## Next (older list from maps-18; her 11-02 batch and countries come first)
1. T-1881 hillshade from a surface model shows dense centres as hills.
2. T-1880 buses as feeders (her question). T-1873 muting worm (now unblocked). T-1867 only if she wants it after T-1874.
3. T-1849, T-1863, T-1864 demand speedups; a `?demandWorkers=3` browser run of T-1862's keep.
4. T-1875, T-1876 to T-1879 (Australia follow-ups; T-1879 has a CLAUDE.md line for data/terrain during joins), T-1882.
5. T-1794 train-count settle. Waiting on Anita: bus design (T-1445).

## Lessons from this run
- Text reviews (Anita 11-02): never "less" for minus (now in her global style guide); no two-part hovers joined by a
  comma; "in past 24h" for rolling counts; she answers "text acknowledged" or names lines.
- Commit rounds when agents sharing files have all reported; a tree's landing fails on any uncommitted main-tree
  edit to a file it touches, so commit the main tree first.
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
