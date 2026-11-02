# Orchestrator handoff, trainworld

Orchestrator: session 7ae5a6d2-b46c-4f33-9114-37a1eedcfcdb, name maps-b2, since 2026-11-02 13:52
Updated: 2026-11-02 ~17:30

Read trainworld/CLAUDE.md and TASKS.md first. Landings are in ARCHIVE.md and `git log`.

## State
- master 50b93f8d = **build 792**; 8800 at build 792, smoke 22/22 light and dark (2026-11-02 ~17:20). No worktrees open.
  Landed today (maps-b2): 428b047e (T-2005..2008), 8daed22e (Cambodia, Malaysia, Brunei: T-1738, T-1997), dd7357cf (T-1896,
  T-1903, T-1907, T-2016, T-2017, T-2019..2023, T-2031), 26528bb6 (T-2018 tree landing; main WASM rebuilt), 50b93f8d
  (rotation off by default). Uncommitted: only TODO.md (Anita's).
- 8800 rebuild: `cmd /c "<my scratchpad>\snapbuild.cmd <hash> > <my scratchpad>\snapbuild.log 2>&1"` (my scratchpad:
  C:\Users\anita\AppData\Local\Temp\claude\c--Users-anita-projects-maps\7ae5a6d2-b46c-4f33-9114-37a1eedcfcdb\scratchpad; the
  script is the same as maps-a7's), then smoke on http://localhost:8800/trainworld/ light and SMOKE_SHADE=dark. Never pass HEAD.
  Build number = `git rev-list --count <hash>`. My dev Vite when needed: `node trainworld/checks/dev-nowatch.mjs 6105`.
- **LIVE (2026-10-31): c7c5785a on R2 `trainworld/2026-10-30b`**. Not deployed since; publish only on Anita's word
  (`deploy.py --skip-data --version 2026-10-30b --smoked`, website commit/push on her word, verify).
- Commits: when agents' edits share a file, wait until all have reported, then one commit; a tree landing needs the main
  tree committed first for any file it touches.
- World: 1,972 shipped, 22 held (incl. Cambodia's tpe anv phm slk, T-2000), 5 parked (nic lez meo doo lel). Cores cap 14
  since ~17:40 2026-11-02 (Anita away: "feel free to use more cores"); back to 10 (`python cores.py cap 10`) when she returns.
- Disk: ~33 GB free (11-02 morning). sim/target/debug (4.3 GB) can go between rounds.
- Smoke's node list lacks checks/t2022-loan-schedule.mjs (needs `node --experimental-strip-types`); add when convenient.

## In flight (maps-b2)
- a245961e82247f810 (Opus) **Indonesia** (T-1738 section 12, kit tools/country/, scratchpad t1738c, Vite 6152, ids
  T-2038..2042): plan, runs, one join with MY (KL origin moved to the centre, T-2036), Brunei realm call (Anita: merge into
  Malaysia, or a Borneo realm once Indonesia is in), load checks. It owns data/packs and data/terrain: nothing else writes
  data/ until it reports. Liveness, batch ids, restart commands go in notes/T-1738.md section 12 before each wait. Over ~8 h
  estimated: it stops and reports before AWS.
- ad9f9e773bec0bad0 (Opus) **T-2037 + T-2030 saves: one card per game** (port 6153, scratchpad t2037, ids T-2043/2044).
  Her design (2026-11-02): group every save of one origin, copies too; the version with the latest player change wins (an
  idle tab's clock saves must not win); "11 other versions" line at the card bottom, expand lists each version with time and
  stations. Server unchanged if possible (stops and reports if shonei-server would need a change). App only.
  ad9f9e77 reported ~18:15: done, uncommitted (workers/versions.ts new, saveFile.ts, cloud.ts, clock.worker.ts, clockClient.ts,
  accountGames.ts, SavesPanel.tsx, StartMenu.tsx, style.css, checks/t2037-versions.mjs, t1826-saves.mjs, architecture.md,
  ui-panels.md, code-map.md, DECISIONS x5; filed T-2043 t988-cloud check needs updating). Resumed for a fix: a version
  identical to the card's own was listed (duplicate first row), and "Day 30 ·" trailing dot. Then commit (app only; do
  NOT rebuild 8800 while Indonesia writes data/packs). Tell Anita: her Tokyo card's top may be the idle 44-line copy until
  she edits one (old saves have no edit time); the Fukuoka version is listed under it; a save by hand does not count as a
  change (agent's call); idle money reaches other devices only with her next edit.
- Next countries after Indonesia (Anita 2026-11-02): Philippines, New Zealand, then India, Turkey, Eastern Europe, one
  easy Middle East country, Mexico, Brazil. T-1978 (chunked gates) before Mexico; T-1979 (World Bank fetch once).
  Country speed (T-1981 result): the joint solve is wide only ~70 s then a one-core tail; the first post-T-1976 register took
  12.8 min (not 3-4): worth a look before the big ones.
- Not started: T-1904 (demand speed: switching homes/destinations, local solves before the world solve; design first).
- Next free task id: T-2045.

## Open questions to Anita (asked, unanswered)
- (Answered 2026-11-02: loans 60 days yes; Total bubbles keep the destinations' density; gap to the far track's nearest
  point is right.)
- (Older) Settings label "Lines side by side" vs "Draw lines side by side" (T-1886).

## Open problems
- T-1891: "Demand ran out of memory" once on a new NYC game on 8800; not reproducible. If it recurs, get the hover text
  or the console line `demand: worker N panicked`. notes/T-1891.md.
- T-1889: riverbanks priced as hills/mountains (East River at 42nd St) from T-1854's classes.

## Next (older list from maps-18; her batches and countries come first)
1. T-1881 hillshade from a surface model shows dense centres as hills.
2. T-1880 buses as feeders (her question). T-1873 muting worm (now unblocked). T-1867 only if she wants it after T-1874.
3. T-1849, T-1863, T-1864 demand speedups; a `?demandWorkers=3` browser run of T-1862's keep.
4. T-1875, T-1876 to T-1879 (Australia follow-ups; T-1879 has a CLAUDE.md line for data/terrain during joins), T-1882.
5. T-1794 train-count settle. Waiting on Anita: bus design (T-1445). T-2009 Quad upkeep figures.

## Lessons from this run
- Text reviews (Anita 11-02): never "less" for minus (now in her global style guide); no two-part hovers joined by a
  comma; "in past 24h" for rolling counts; she answers "text acknowledged" or names lines.
- A new row or hover that has an existing counterpart copies its pattern; put the reference in briefs for panel work.
- Look at every UI agent's pictures yourself for spacing before reporting done.
- Her saves live per origin (localhost:8800 and anita.garden are separate IndexedDBs, Chrome Profile 1); to look at them,
  copy `IndexedDB\<origin>.indexeddb.leveldb` and `.blob` into a throwaway profile and read the "anitabuilder" db's "saves"
  store from a page on that origin (scripts listslots.mjs, dumpslot.mjs, dumpsync.mjs in my scratchpad). Never touch her
  profile itself.
- Anita: if an orchestrator starts on Sonnet, tell her at once to switch it to Opus.

## Anita's standing asks
- Player-facing text: agents propose wording per the style rules and list every string old -> new; land without
  waiting; show her the list. Never ask her to write text from scratch.
- TODO.md short-term = batch of prompts: copy item text verbatim into briefs; save a batch in notes/todo-<date>.md.
- Every demand task (loosely perf) has a "done when"; stop once met. Approximations welcome when invisible.
- Buses settled unless she asks. Thanks to agents in my own reply. Someday tasks are real open tasks.
