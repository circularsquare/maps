# Orchestrator handoff, trainworld

Orchestrator: session 7ae5a6d2-b46c-4f33-9114-37a1eedcfcdb, name maps-b2, since 2026-11-02 13:52
Updated: 2026-11-03 ~10:30 (context past 500k: a fresh orchestrator can take over any time)

Read trainworld/CLAUDE.md and TASKS.md first. Landings are in ARCHIVE.md and `git log`.

## State
- master 57a1a157. **8800 = build 815 (7b5a08ee)** on freeze 20261103-034613 (NZ), smoke 22/22 light and dark (~07:20).
  Commits since 815 (bdbfc79f terrain_fine fork fix, e9a82a7e T-1880 check, 57a1a157 T-2105 checks) change no app
  code; rebuild 8800 after the terrain rebuild lands. Smoke now has 23 checks (t2022 loan schedule added).
- Worktrees: .tw-trees/t1734 only (kept on purpose, T-1734).
- Uncommitted in the main tree (other agents' work in progress, commit with their reports): the census-rebuild agent's
  and India agent's files (pipeline/admin_totals.py, enrich_parks.py, india.py, realms.py, world_plan.py/.json,
  intercity.py, checks/admin_totals_check.py, pack_gate.py?, tools/country/cambodia_census.py, india_census.py, the
  app/src/game json files, docs adding-a-country/data-register/code-map?, notes/T-1738.md, notes/T-2111.md, DECISIONS,
  TASKS). TASKS.md also holds T-2104/T-2105/T-1880 line changes (T-1880 agent) not yet committed. TODO.md is Anita's.
- 8800 rebuild: `cmd /c "<my scratchpad>\snapbuild.cmd <hash> > <my scratchpad>\snapbuild.log 2>&1"` (my scratchpad:
  C:\Users\anita\AppData\Local\Temp\claude\c--Users-anita-projects-maps\7ae5a6d2-b46c-4f33-9114-37a1eedcfcdb\scratchpad),
  then smoke on http://localhost:8800/trainworld/ light and SMOKE_SHADE=dark. Never pass HEAD. Build number =
  `git rev-list --count <hash>`. **While a country run owns data/packs**: set `TW_PACKS=<main>\trainworld\data\frozen\
  <newest>\packs` and `TW_MARKET_DRIVES=<main>\trainworld\data\packs\market-drives.bin` before snapbuild (the commit's
  registry must match the freeze). Normal smoke: new game ~104 s, real save final ~150-200 s.
- Staging one agent's lines of a shared file: `python <my scratchpad>\stage_without.py <maps root> <path> "<marker>"`.
  Caution: a line another agent EDITED is dropped entirely; check `diff --cached` for unexpected removals.
- `checks/build_inputs.py`: every file a production build copies; 0 flags expected.
- **LIVE: build 797 (7e730216) on freeze 20261102-230529, R2 `trainworld/2026-11-03`, website f50e733**, verified.
  Publish only on Anita's word. Recipe: `freeze.py build <stamp>` with TW_MARKET_DRIVES set (a freeze lacks
  market-drives.bin), `deploy.py --dist <dist> --version <v> --smoked`, website `git add trainsworld/ trainworld/`,
  commit "anitabuilder", push, `deploy.py --verify --page --version <v>`. Leave her pages/thoughts/cities.md alone.
  Live 797 has the T-2071 out-of-memory bug: I proposed publishing once the terrain rebuild and census fixes are on 8800.
- World: Philippines (102 of 103, pti held) and New Zealand (6 of 6) shipped 2026-11-03; every zone has terrain and drives.
- Cores cap 15 (Anita asleep overnight; back to 10, `python cores.py cap 10`, when she's around).

## Landed 2026-11-02 evening to 11-03 (maps-b2)
T-2058 lines side by side A+C (still off in Settings), muting thin+faded default, T-2059/T-2060 (band build ~260 ms,
Tamachi, Jersey City), T-2092 tutorial start fixes, T-2071+T-1891 out-of-memory loop, T-1966 crossing waits from zoom 14,
T-1881 hillshade flat where priced flat + tower_ground code (prices wait for the rebuild), T-2040/T-1979/T-1978/T-1980
runner, T-2084 census totals (admin_totals.py), Philippines, New Zealand, T-1880 bus feeders (check), T-2105 checks,
research: player-voices, next-countries-data, first-ten-minutes, ridership (+ checks/t2088-ridership.mjs).

## In flight (maps-b2), ~10:30 2026-11-03
Data writers take turns, in this order: census rebuilds -> terrain half 2 -> India.
1. aea244d3f3bcf163b (Opus) **census rebuilds** (notes/T-1738.md 15, kit t2038, Vite 6152, ids T-2106..2110): T-2038
   Papua, T-2053/T-2076 BARMM + pti, T-2000 Cambodia NW (typed citypopulation.de districts), T-2077 NZ if time. Its join
   was running ~10:00 (claim t2038, 9 cores). Owns data/packs and data/terrain until it reports. On report: commit its
   files, then step 2.
2. ae84ece4d2ab0afec (Sonnet) **T-1881 terrain half 2**: parts staged and checked in data/work/t1881-staging/ (AWS run 2:
   19 min, US$0.38, torn down clean). Waiting for my "data/terrain is free". Resume it with that after step 1; it backs
   up, swaps parts, `zone_terrain.py build --force --jobs 8`, mix, build_inputs, t1881-prices.mjs, smoke, cleans up,
   archives T-1881/T-1889/T-1944. Then rebuild 8800 (normal build if no one writes data).
3. a57accbdda9f839db (Opus) **India** (T-2111, notes/T-2111.md, kit t2111, Vite 6165, ports 6388/6389): prep DONE,
   waiting for my go after step 2. Plan: 74 metros + 709 zones (782 rows), China's zone numbers (floor 1M, cap 2.5M),
   census OFF (2011 shares moved only 2.6%, mostly undoing growth; one line in india.plan_hooks to switch on), realms by
   zonal council ("India" renamed "West India", id kept), ~9-10 h (stops before the join if slow). 188 names listed in
   notes/T-2111.md for Anita. Its go message: "go" (plus "switch it on" only if Anita wants the census).
4. T-2093 part 1 DONE (committed): Osaka 0.35, Nagoya 0.42, Fukuoka 0.49, Sapporo 0.43, Hiroshima 0.52 of real (S12
   entries, floors); jpn7 itself runs ~0.73x the dedicated saves (not looked into; not filed). Japan reads low throughout,
   NY high: a demand question for Anita (she ruled out a per-place factor, T-1750). Part 2 (US car city) open.
- After India: Turkey (T-2079, COD-PS ilce), Mexico (T-1978 chunked gates done; T-2080/T-2081), T-1856 hexagons after
  the terrain rebuild (design in notes/T-1881.md, ~1 day), T-2064 (the real save's flat crossings that are really
  grade-separated), T-2054 (join builds terrain + drives), T-2095 (first register after a join rebuilds every place),
  T-2094 (world_plan's other Mollweide distances), T-2099 (shore clip loses coastal people).

## Open questions to Anita (asked, unanswered)
- Publish to live (build 815+, with the out-of-memory fix), ideally after the terrain rebuild and census rebuilds.
- Middle East country: Israel (her territory call first) or Saudi Arabia (T-2082).
- T-2104: split each trip's base fare between lines by km instead of per boarding (totals unchanged; feeders and
  crosstown buses read truthfully). I said I'd do it.
- Start menu: pre-pick a start city instead of "Choose on map" (T-2087 idea 1); her calls in research/first-ten-minutes.md
  (default schedule, tutorial telemetry, hiding Build rows).
- Which research/player-voices.md ideas to file (I suggested a demand tutorial card and a "what this edit did" note).
- NY 2.4x / Tokyo 0.36x real ridership (T-2088): I said wait for jpn7 (running) + a US car city (T-2093 part 2).
- Text to acknowledge: Philippines 3 realms + 64 names, NZ 2 realms + 5 names, India's 188 names (notes/T-2111.md).
- Test on 8800: Settings "Lines side by side" on the NY and Tokyo saves; muted lines thin+faded; crossing waits at
  Broadway-Lafayette (zoom 14+).
- maps/CLAUDE.md "Keep the work moving" bullet: uncommitted root file, hers to commit.
- (Older) Settings label "Lines side by side" vs "Draw lines side by side" (T-1886).

## Open problems
- T-2064 [P2]: most of the real save's 340 flat crossings are lines that really pass over/under (L under Broadway), so
  the sim charges crossing delays and trains now visibly wait there. Needs `EdgeData.sep` on the real save (T-1751).
- T-2046 Seoul switch 6 s. T-1889 / T-1944 close with the terrain rebuild.

## Next (older list; her batches and countries come first)
1. T-1849, T-1863, T-1864 demand speedups.
2. T-1875, T-1876 to T-1879 (Australia follow-ups), T-1882.
3. T-1794 train-count settle. T-2009 Quad upkeep figures.

## Lessons from this run
- "Keep the work moving" (maps/CLAUDE.md): sweep what's not done at every report and start what can go; stale done
  lines to ARCHIVE. "Look outward" (trainworld/CLAUDE.md): research references freely.
- Pipeline edits land only between runs (a run imports pipeline modules per subprocess); worktrees for pipeline agents.
- A freeze lacks market-drives.bin; the country playbook lacked terrain files and drives (fixed: playbook section 8,
  checks/build_inputs.py, T-2054).
- AWS terrain_fine: forked workers shared one file handle (fixed bdbfc79f); test parallel code under fork, not only spawn.
- Text reviews (Anita 11-02): never "less" for minus; no two-part hovers joined by a comma; she answers "text
  acknowledged" or names lines. Indonesia text acknowledged.
- Her saves live per origin (localhost:8800 and anita.garden are separate IndexedDBs, Chrome Profile 1); to look at them,
  copy `IndexedDB\<origin>.indexeddb.leveldb` and `.blob` into a throwaway profile (scripts listslots.mjs, dumpslot.mjs,
  dumpsync.mjs in my scratchpad). Never touch her profile itself.
- Anita: if an orchestrator starts on Sonnet, tell her at once to switch it to Opus.

## Anita's standing asks
- Player-facing text: agents propose wording per the style rules and list every string old -> new; land without
  waiting; show her the list. Never ask her to write text from scratch.
- TODO.md short-term = batch of prompts: copy item text verbatim into briefs; save a batch in notes/todo-<date>.md.
- Every demand task (loosely perf) has a "done when"; stop once met. Approximations welcome when invisible.
- Buses settled unless she asks. Thanks to agents in my own reply. Someday tasks are real open tasks.

Next free task id: T-2122 (T-2041, T-2042, T-2047, T-2048, T-2050..2052, T-2055..2057, T-2061, T-2062, T-2065,
T-2066..2069, T-2072, T-2073, T-2085, T-2121 handed out but unused).
