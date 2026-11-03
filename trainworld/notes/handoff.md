# Orchestrator handoff, trainworld

Orchestrator: session 7ae5a6d2-b46c-4f33-9114-37a1eedcfcdb, name maps-b2, since 2026-11-02 13:52
Updated: 2026-11-02 ~23:45

Read trainworld/CLAUDE.md and TASKS.md first. Landings are in ARCHIVE.md and `git log`.

## State
- master 7e730216 = **build 797**; 8800 at build 797, smoke 22/22 light and dark (2026-11-02 ~23:40). No worktrees open,
  no agents running. Uncommitted: only TODO.md (Anita's) and this file.
  Landed today (maps-b2): 428b047e (T-2005..2008), 8daed22e (Cambodia, Malaysia, Brunei), dd7357cf (big UI round),
  26528bb6 (T-2018), 50b93f8d (rotation off), d58da554 (T-2037/T-2030 saves one card per game), 60506f3c (T-1904 demand
  speed), 0b5d6c13 + 030f1c35 (T-2045 focus pool, capped at 2 workers), 7e730216 (Indonesia, T-1738/T-2036).
- 8800 rebuild: `cmd /c "<my scratchpad>\snapbuild.cmd <hash> > <my scratchpad>\snapbuild.log 2>&1"` (my scratchpad:
  C:\Users\anita\AppData\Local\Temp\claude\c--Users-anita-projects-maps\7ae5a6d2-b46c-4f33-9114-37a1eedcfcdb\scratchpad; the
  script is the same as maps-a7's), then smoke on http://localhost:8800/trainworld/ light and SMOKE_SHADE=dark. Never pass HEAD.
  Build number = `git rev-list --count <hash>`. Normal smoke times: new game ~104 s, real save final ~180 s.
- Staging only one agent's lines of a shared file (TASKS/DECISIONS): `python <my scratchpad>\stage_without.py <maps root>
  <path> "<marker>"` stages the working copy minus new lines containing the marker.
- **LIVE (2026-11-03 ~00:40): 7e730216 (build 797) on freeze 20261102-230529, R2 `trainworld/2026-11-03`, website
  f50e733**, verified (6 PASS). Publish only on Anita's word. From a freeze: `freeze.py build <stamp>` with
  `TW_MARKET_DRIVES=<main>\data\packs\market-drives.bin` set, then `deploy.py --dist <dist> --version <v> --smoked`,
  website `git add trainsworld/ trainworld/` commit push, `deploy.py --verify --page --version <v>`. Known gap: ~250
  zones (ID, MY) have no terrain file in this publish (flat prices there); the Philippines agent builds them.
- World: 2,174 shipped (Indonesia 203 of 205; frozen `data/frozen/20261102-230529`); held incl. Cambodia's tpe anv phm slk
  (T-2000) and Papua's puc (T-2038); parked nic lez meo doo lel and Seram Bagian Barat set (T-2039). Cores cap 10.
- Disk: check before the next country (Indonesia added packs; 71 GB free at Indonesia's start).
- Smoke's node list lacks checks/t2022-loan-schedule.mjs (needs `node --experimental-strip-types`); add when convenient.

## In flight (maps-b2)
- ada85057bf13b2345 (Opus) **Philippines** (T-1738 section 13, kit scratchpad t1738d, Vite 6152, run ports 6386/6387,
  ids T-2053..2057, up to 8 cores). Started ~23:55 2026-11-02 (Anita: "can move on to philippines"). Owns data/packs and
  data/terrain until it reports. Manila (>20M) gated alone per playbook 5.
  Told ~00:15 to also build missing zone terrain (250 zones lacked data/terrain/<id>.terrain: ID, MY...; playbook gap).
- Anita ~00:05 2026-11-03, going to bed: "feel free to use all cores. and keep going rather than waiting for handoff cuz
  i wont be able to start new agent". Cap 15. Indonesia text acknowledged. Publish today's work: yes (doing it).
- a49f9cff5f5f759a0 (Opus) **T-2058 interlining continuous** (main tree, map/lineBands.ts, corridors.ts; Vite 6156;
  scratchpad t2058; ids T-2059..2062): stage 1 = failure catalogue + 2-4 ideas + prototype behind a flag, and "muted
  lines thinner" (T-1873 alt) behind a flag. Stops for her pick.
- a56d6f32a2d5c7d06 (Opus) **T-1966 flat-crossing waits** (tree .tw-trees/t1966; Vite 6157; ids T-2063..2065). It was
  off because occupancy came from 16 s frames (100-150 trains held), not for speed.
- a16dcf08738642b86 (Opus) **T-1881 + T-1889 + T-1944 terrain hills/prices, then T-1856 hexagons** (tree t1881; Vite
  6158; ids T-2066..2069). Must not write data/terrain until the Philippines reports; gives me rebuild commands.
  zone_terrain.py already uses GEDTM30 bare earth for prices; the hillshade is the public Terrain Tiles.
- a7287a1c25e0066ee (Opus) **T-2040 + T-1979 + T-1978 + T-1980 build speedups** (tree t2040; Vite 6159, ports
  6388/6389; ids T-2070..2073). No real runs; land after the Philippines. T-1981 not started (needs a real join).
- Publish DONE and verified ~00:40 (see State). Detail: freeze 20261102-230529 built with `TW_MARKET_DRIVES=<main>\data\packs\market-drives.bin` (a freeze
  lacks market-drives.bin, so without it the dist had none!) into artifacts/trainworld/publish/20261102-230529/dist;
  `deploy.py --dist <that> --version 2026-11-03 --smoked` running, log my scratchpad deploy.log. Then website: `git -C
  ../website add trainsworld/ trainworld/`, commit "anitabuilder", push (her word given); leave her
  pages/thoughts/cities.md alone; then `deploy.py --verify --page --version 2026-11-03`.
- T-1445: she asked if it was done: it was (T-1464, T-1465, T-1514); archived 2026-11-03.
- T-1966: Anita "we only need to do it where we're looking closely": sent to its agent (waits only on screen, close zooms).
- a5625c4ecb8319d0b (Opus) **T-2074 player voices**: what players love/hate in similar games -> research/player-voices.md,
  15-25 ranked ideas for her to pick (no tasks filed).
- abe19b070fec1727d (Opus) **T-2075 next-country data**: census admin totals + joining boundaries for NZ, India, Turkey,
  E. Europe, a Middle East pick, Mexico, Brazil, and Indonesia/Cambodia/Philippines (for T-2038, T-2000) ->
  research/next-countries-data.md, data-register cells, may file T-2076..2085.
- T-2058 stage 1 COMMITTED 419b165d (flags bandsSmooth, muteThin, muteThinPale; default unchanged). Agent
  a49f9cff5f5f759a0 resumed ~01:00 for stage 2 on its own pick (A proper + C prototype as `bandsOrder`, defaults
  unchanged) and the t1799-bands click check. Anita picked ~01:10: "lets try A then C, and thin and faded": sent;
  muteThinPale becomes the default muting; side by side stays off in Settings until she sees A+C.
- Anita ~01:00 "lets do both exploration": ac175af5aa716ea9f (Opus) **T-2086 ridership benchmarks** ->
  research/ridership.csv + .md (may file T-2088); aa68e9bf0d0f1ed41 (Opus) **T-2087 first ten minutes** ->
  research/first-ten-minutes.md (3 cores for a walkthrough). Uncommitted outside trainworld/: maps/CLAUDE.md's "Keep the
  work moving" bullet (root file, hers to commit).
- a7287a1c25e0066ee **T-2040/T-1979/T-1978/T-1980 DONE in tree .tw-trees/t2040**, uncommitted (checks city-browser.mjs,
  pack_gate.py, runner_fixes.py; pipeline world_run.py, tier0.py, plan_config.py, nocar.py, arcgis.py, build_city.py,
  netget.py new; docs world-run.md, adding-a-country.md, code-map.md; notes/T-2040.md; main-tree TASKS/ARCHIVE/
  DECISIONS lines already written). Jakarta's hang = the headless renderer dying out of memory on removeLine's solve.
  runner_fixes 9/9, smoke 22/22 in the tree. **Land only after the Philippines reports** (the runner runs from the main
  tree). Watch: the main tree's checks/pack_gate.py has an uncommitted Philippines edit (18 lines): commit that first,
  then land, resolving if both touched the same lines. Filed T-2070 (confirm chunks/read-ahead on a real run: NZ),
  T-2071 [P2] (removing a line in a 20M+ zone crashes the page out of memory: maybe T-1891's cause too).
- T-2074 player voices DONE, research/player-voices.md committed (its ARCHIVE line is in the working copy, goes with the
  next commit). 18 ranked ideas, none filed: Anita picks. Top: a demand tutorial card, goals per city/world, "what this
  edit did" note, one plain reason for a quiet line.
- T-2075 next-country data DONE, research/next-countries-data.md committed; data-register cells and TASKS T-2076..T-2084
  in the working copy (commit with the next round). T-2084 [P3, do first among them]: tier0 `admin_totals` bug (whole unit
  total into the part inside the outline) + one generic reader; edits tier0.py, which the t2040 tree also changed: start
  it in a tree after t2040 lands. Then T-2038 (Indonesia, USCB HDX kabupaten), T-2000 (Cambodia, by hand), T-2076
  (Philippines barangays; BARMM GHSL 0.1-0.55x, told the Philippines agent). Question for Anita: Middle East = Israel (her
  territory call first) or Saudi Arabia.
- T-2087 first ten minutes DONE, research/first-ten-minutes.md committed 3d956ced (ARCHIVE line in working copy).
  13 ideas; worst stalls: default start lands on the world map; a new line's trains only at High so it stops at 20:00;
  game opens on Lines not Build. Her calls listed in the report (default schedule, step 1 wording, tutorial telemetry,
  hiding Build rows, the 10% tick on the top bar). Idea 1 (pre-pick a start city) left for her.
- T-2086 ridership DONE, research/ridership.csv + .md committed (ARCHIVE line + T-2088 in working copy). Finding: the
  game gives 0.4-0.7 rail trips/resident on every big real network; real cities 0.006 (Houston) to 1.13 (Tokyo). T-2088
  (a check script: real saves vs the table, 0.5-2x band, notes only) is next when 2 cores free. Any demand change from
  it is Anita's (she ruled out a per-place rail factor, T-1750).
- aa8b947bb8932ff73 (Opus) **T-2071 + T-1891 big-zone out of memory** (main tree, workers/demand*; Vite 6160; scratchpad
  t2071; ids T-2089..2091; 2 cores). Started ~01:45.
- a0281d35a107d9c64 (Sonnet) **T-2092 tutorial start fixes**: open on Build with the card; a brand-new line's first
  train setting fills the other 0 levels (my call, rail only). Vite 6161, scratchpad t2092, id T-2093, 1 core.
- **After the Philippines** (Anita 2026-11-03: anything; speedups, India, or smaller countries first): commit it, rebuild
  8800; land the t2040 speedup tree (smoke in tree), then New Zealand as the first real run with the speedups (and
  T-1981's CPU poll during its join), then India with T-2075's admin totals. Turkey alongside if cores allow.
- Standing rule now in maps/CLAUDE.md "Keep the work moving": sweep what's not done every report / couple of hours and
  start what can go; stale done lines to ARCHIVE. trainworld/CLAUDE.md "Look outward": research references freely.
- Tidied TASKS: T-1898, T-1902 (landed e8dc880a, cc3972d2), T-1974, T-1975 (designs done) moved to ARCHIVE. Tree
  .tw-trees/t1734 is kept on purpose (T-1734, Anita: come back to it).
- Next countries (Anita 2026-11-02): Philippines, New Zealand, then India, Turkey, Eastern Europe, one easy Middle East
  country, Mexico, Brazil. Before the giants: T-2040 (the gate's browser step hangs after a ~20M+ metro with other zones on
  the page; Indonesia lost 3 h to it; playbook now says gate such a metro alone), T-1978 (chunked gates) before Mexico,
  T-1979 (World Bank fetch once). Indonesia took ~8 h wall (Java 53 min to build, join 48 min).
- Anita 2026-11-02 answers: keep "cities" in the status tag; save by hand is not a game change (fine as is); the loading
  card stays until the whole world finishes (T-2049 abandoned).
- Next free task id: T-2094 (T-2041, T-2042, T-2047, T-2048, T-2050..2052 were handed out but unused).

## Open questions to Anita (asked, unanswered)
- (Older) Settings label "Lines side by side" vs "Draw lines side by side" (T-1886).

## Open problems
- T-1891: "Demand ran out of memory" once on a new NYC game on 8800; not reproducible. If it recurs, get the hover text
  or the console line `demand: worker N panicked`. notes/T-1891.md. (T-2045's focus pool adds ~0.9 GB peak on big saves.)
- T-1889: riverbanks priced as hills/mountains (East River at 42nd St) from T-1854's classes.
- T-2046: Seoul's first bubble switch after a zoom waits ~6 s on the Tokyo save; cause unknown.

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
