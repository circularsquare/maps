# Orchestrator handoff, trainworld

Orchestrator: session 7ae5a6d2-b46c-4f33-9114-37a1eedcfcdb, name maps-b2, since 2026-11-02 13:52
Updated: 2026-11-03 ~04:15

Read trainworld/CLAUDE.md and TASKS.md first. Landings are in ARCHIVE.md and `git log`.

## State
- master 04eca629 = **build 811**; 8800 at build 811, smoke 22/22 light and dark (~04:00). Worktrees: .tw-trees/t1734
  (kept on purpose, T-1734) and t2084 (in flight). Uncommitted: TODO.md (Anita's), this file.
- 8800 rebuild: `cmd /c "<my scratchpad>\snapbuild.cmd <hash> > <my scratchpad>\snapbuild.log 2>&1"` (my scratchpad:
  C:\Users\anita\AppData\Local\Temp\claude\c--Users-anita-projects-maps\7ae5a6d2-b46c-4f33-9114-37a1eedcfcdb\scratchpad),
  then smoke on http://localhost:8800/trainworld/ light and SMOKE_SHADE=dark. Never pass HEAD. Build number =
  `git rev-list --count <hash>`. **While a country run owns data/packs**: set `TW_PACKS=<main>\trainworld\data\frozen\
  <newest>\packs` and `TW_MARKET_DRIVES=<main>\trainworld\data\packs\market-drives.bin` before snapbuild (the commit's
  registry must match the freeze). Normal smoke: new game ~104 s, real save final ~150-200 s.
- Staging one agent's lines of a shared file: `python <my scratchpad>\stage_without.py <maps root> <path> "<marker>"`.
- `checks/build_inputs.py` (new): every file a production build copies, 0 flags expected (2,340 rows at 04:00).
- **LIVE: build 797 (7e730216) on freeze 20261102-230529, R2 `trainworld/2026-11-03`, website f50e733**, verified.
  Publish only on Anita's word. Recipe: `freeze.py build <stamp>` with TW_MARKET_DRIVES set (a freeze lacks
  market-drives.bin), `deploy.py --dist <dist> --version <v> --smoked`, website `git add trainsworld/ trainworld/`,
  commit "anitabuilder", push, `deploy.py --verify --page --version <v>`. Leave her pages/thoughts/cities.md alone.
  **Asked her: publish 804+ (the T-2071 out-of-memory fix; live 797 has the bug).** Next publish should be a freeze
  after the Philippines (20261103-011728) or later.
- World: Philippines shipped (102 of 103; pti held, T-2053); terrain files and routed drives for every zone (0 missing).
  Cores cap 15 (Anita asleep; back to 10 when she's around).
- Smoke's node list lacks checks/t2022-loan-schedule.mjs (needs `node --experimental-strip-types`); add when convenient.

## Landed tonight (2026-11-03, maps-b2)
419b165d + 082f7632 T-2058 lines side by side A+C (still off in Settings) and muting thin+faded default, T-2092 (new
tutorial game opens on Build; a new line's first trains fill every level), ef971553 T-2071+T-1891 (endless loop in
demand.worker.ts `toWhole`, the "out of memory"), afbae2d8 Philippines, d550ab27 T-2040/T-1979/T-1978/T-1980 runner,
17dbab9d T-1881 hillshade flat where priced flat + tower_ground code, b9411c52 T-1966 crossing waits from zoom 14
(main WASM rebuilt), research: player-voices, next-countries-data, first-ten-minutes, ridership (+ checks/t2088).

## In flight (maps-b2)
- a8236098de3c50a7c (Opus) **New Zealand** (T-1738 section 14, kit t1738e, Vite 6152, ports 6386/6387, ids T-2094..2098,
  up to 9 cores). First real run of the new runner (T-2070 findings) and T-1981's join CPU poll. Owns data/packs and
  data/terrain until it reports.
- ae84ece4d2ab0afec (Sonnet) **T-1881 world terrain rebuild**: half 1 on AWS now (launch, staging folder
  data/work/t1881-staging); STOPS and reports before writing data/terrain. **Resume it with "go" only after New Zealand
  reports** (half 2: backups, zone_terrain --force, mix, prices check, smoke). Then rebuild 8800.
- aeaa9c116f432e5ce (Opus) **T-2084 admin_totals fix + generic reader**: DONE ~04:40 in tree .tw-trees/t2084,
  uncommitted (tier0.py scale_to_units + ghsl_people, pipeline/admin_totals.py new with ID and PH configs,
  checks/admin_totals_check.py 10 PASS, docs; main-tree TASKS/ARCHIVE/DECISIONS lines written; T-2099 filed: shore clip
  loses coastal people, pti 1.63M of 1.88M). Dry: yai 480k -> 165k, mru 0.61x, mmi 0.67x; pti 617k -> 1.63M, mrw 1.61x.
  PH by municipality, not barangay. A pack takes the hook only once its tier0_signals cache is deleted. **Land after NZ
  reports** (pipeline). Then start T-2038 (Indonesia Papua), T-2053/T-2076 (Philippines BARMM + pti), T-2000
  (Cambodia, typed-in districts), each a few-zone rebuild.
- a34b87357c9bcc2be (Opus) **T-2059 + T-2060 band polish** (main tree lineBands.ts/corridors.ts, Vite 6156, scratchpad
  t2059, ids T-2102/2103, 2 cores).
- a1c6f7b238b0d5208 (Opus) **T-1880 buses as feeders** (measurement only, new check script; Vite 6164, scratchpad
  t1880, ids T-2104/2105, 3 cores).
- After NZ: India (T-2078 district shares via T-2084's old-census mode), Turkey (T-2079). T-1856 hexagons after the
  terrain rebuild lands (design in notes/T-1881.md, ~1 day).

## Open questions to Anita (asked, unanswered)
- Publish build 804+ (out-of-memory fix) to live.
- Middle East country: Israel (her territory call first) or Saudi Arabia (T-2082).
- Start menu: pre-pick a start city instead of "Choose on map" (T-2087 idea 1); her calls in research/first-ten-minutes.md
  (default schedule, tutorial telemetry, hiding Build rows).
- Which research/player-voices.md ideas to file (I suggested 1 demand tutorial card and 3 "what this edit did").
- NY 2.4x / Tokyo 0.36x real ridership (T-2088): look into it? I said wait for jpn7 + a US car city (T-2093).
- Test on 8800: Settings "Lines side by side" on the NY and Tokyo saves; muted lines thin+faded; crossing waits at
  Broadway-Lafayette (zoom 14+).
- (Older) Settings label "Lines side by side" vs "Draw lines side by side" (T-1886).

## Open problems
- T-2064 [P2]: most of the real save's 340 flat crossings are lines that really pass over/under (L under Broadway), so
  the sim charges crossing delays and trains now visibly wait there. Needs `EdgeData.sep` on the real save (T-1751).
- T-2059 band build 705 ms vs 240 after a path edit; T-2060 Tamachi drift; T-2046 Seoul switch 6 s.
- T-1889 / T-1944 close with the terrain rebuild.

## Next (older list; her batches and countries come first)
1. T-1880 buses as feeders (her question). T-1867 only if she wants it after T-1874.
2. T-1849, T-1863, T-1864 demand speedups.
3. T-1875, T-1876 to T-1879 (Australia follow-ups), T-1882.
4. T-1794 train-count settle. T-2009 Quad upkeep figures.

## Lessons from this run
- "Keep the work moving" is now in maps/CLAUDE.md (uncommitted root file, hers): sweep what's not done at every report
  and start what can go; stale done lines to ARCHIVE.
- "Look outward" is in trainworld/CLAUDE.md: research references freely.
- A freeze lacks market-drives.bin; a country playbook lacked terrain files and drives (fixed: playbook section 8,
  checks/build_inputs.py, T-2054 to fold into the join).
- Text reviews (Anita 11-02): never "less" for minus; no two-part hovers joined by a comma; she answers "text
  acknowledged" or names lines. Indonesia text acknowledged; Philippines' 3 realms + 64 zone names shown ~04:15.
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

Next free task id: T-2106 (T-2041, T-2042, T-2047, T-2048, T-2050..2052, T-2055..2057, T-2061, T-2062, T-2065,
T-2066..2069, T-2072, T-2073, T-2085 handed out but unused).
