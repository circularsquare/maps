# Orchestrator handoff, trainworld

Orchestrator: session 76ebbfee-75f0-49e0-be85-230cb2ef8d79, name maps-0f, since 2026-11-03 09:32
Updated: 2026-11-03 10:15 (local time, -0500; maps-b2's "~15:00" stamps below were not local time). maps-b2 handed
back 09:35: every running agent below was told to send its final report to maps-0f too (SendMessage); they are maps-0f's
to adopt. Their SubagentHandback still lands in maps-b2's session, so if a report seems missing, ask the agent by id.

Read trainworld/CLAUDE.md and TASKS.md first. Landings are in ARCHIVE.md and `git log`.

## State
- master a273050d (10:05): 5faa828f = T-2124, T-2125, T-2143-45, T-2123, T-2142, T-2132/33, T-2122, T-2146 design (smoke
  23/23 on my Vite 6175 first); a273050d = T-2104 landed from its tree (removed). T-2147's `crush2/crush15` measurement
  hunks went in with 5faa828f (demandClient packService, perfFlags): T-2147's report says whether they stay.
  a53f0cd8 (10:15) = T-2064, T-007-real-nyc.save all `sep` (smoke's real save now 11,590,441 rail trips).
  **8800 = build 825 (a273050d)** on freeze 20261103-051954 (my scratchpad snapbuild-frozen.cmd <hash> <freeze>), smoke
  light 22/23 (t2005 failed on T-2153's mid-edit main-tree demandLod.ts; passes from .tw-snap) and dark 23/23.
  LIVE still 821 (R2 `trainworld/2026-11-03b`, website dc49c5f). Publish only on her word.
- India: run 3 (25240) ended 10:02; its agent restarted as run 4 ~10:07 (cores t2111 pid 25628 "relax rerun", log
  artifacts/trainworld/t2111-run4.log, plus t2111fine terrain fetch pid 11216).
- Worktrees: .tw-trees/t1734 only (kept on purpose, T-1734).
- Uncommitted in the main tree (other agents' work in progress, commit with their reports): the census-rebuild agent's
  and India agent's files (pipeline/admin_totals.py, enrich_parks.py, india.py, realms.py, world_plan.py/.json,
  intercity.py, checks/admin_totals_check.py, pack_gate.py?, tools/country/cambodia_census.py, india_census.py, the
  app/src/game json files, docs adding-a-country/data-register/code-map?, notes/T-1738.md, notes/T-2111.md, DECISIONS,
  TASKS). TASKS.md also holds T-2104/T-2105/T-1880 line changes (T-1880 agent) not yet committed. TODO.md is Anita's.
- 8800 rebuild: `cmd /c "<my scratchpad>\snapbuild.cmd <hash> > <my scratchpad>\snapbuild.log 2>&1"` (my scratchpad:
  C:\Users\anita\AppData\Local\Temp\claude\c--Users-anita-projects-maps\76ebbfee-75f0-49e0-be85-230cb2ef8d79\scratchpad;
  copies of maps-b2's, whose scratchpad still holds listslots/dumpslot/dumpsync.mjs),
  then smoke on http://localhost:8800/trainworld/ light and SMOKE_SHADE=dark. Never pass HEAD. Build number =
  `git rev-list --count <hash>`. **While a country run owns data/packs**: set `TW_PACKS=<main>\trainworld\data\frozen\
  <newest>\packs` and `TW_MARKET_DRIVES=<main>\trainworld\data\packs\market-drives.bin` before snapbuild (the commit's
  registry must match the freeze). Normal smoke: new game ~104 s, real save final ~150-200 s.
- Staging one agent's lines of a shared file: `python <my scratchpad>\stage_without.py <maps root> <path> "<marker>"`.
  Caution: a line another agent EDITED is dropped entirely; check `diff --cached` for unexpected removals.
- `checks/build_inputs.py`: every file a production build copies; 0 flags expected.
- Publish only on Anita's word. Recipe: `freeze.py build <stamp>` with TW_MARKET_DRIVES set (a freeze lacks
  market-drives.bin), `deploy.py --dist <dist> --version <v> --smoked`, website `git add trainsworld/ trainworld/`,
  commit "anitabuilder", push, `deploy.py --verify --page --version <v>`. Leave her pages/thoughts/cities.md alone.
- World: Philippines (103 of 103) and New Zealand (6 of 6) shipped 2026-11-03; every zone has terrain and drives.
- Cores cap 10 while Anita is at the machine (12-15 when she's away: `python cores.py cap <n>`).

## Landed 2026-11-02 evening to 11-03 (maps-b2)
T-2058 lines side by side A+C (still off in Settings), muting thin+faded default, T-2059/T-2060 (band build ~260 ms,
Tamachi, Jersey City), T-2092 tutorial start fixes, T-2071+T-1891 out-of-memory loop, T-1966 crossing waits from zoom 14,
T-1881 hillshade flat where priced flat + tower_ground code (prices wait for the rebuild), T-2040/T-1979/T-1978/T-1980
runner, T-2084 census totals (admin_totals.py), Philippines, New Zealand, T-1880 bus feeders (check), T-2105 checks,
research: player-voices, next-countries-data, first-ten-minutes, ridership (+ checks/t2088-ridership.mjs).

## In flight (maps-0f, adopted from maps-b2 09:35 2026-11-03)
LIVE = build 821 (8212d530) on freeze 20261103-051954, R2 `trainworld/2026-11-03b`, website dc49c5f, verified 6/6.
8800 = build 821 too; master has more since (78e80f51 band gap, 73d722a2 achievement design note); rebuild 8800 after
the next landings. Cap 10 (Anita at the machine). Her batches today: notes/todo-2026-11-03.md and todo-2026-11-03b.md.

Running agents (all main tree unless noted; each claims 1-2 cores):
- a57accbdda9f839db (Opus) **India** (T-2111, notes/T-2111.md, kit t2111, Vite 6165, ports 6388/6389): told ~12:00 to
  STOP and restart at --jobs 5 (cap 10). Owns data/packs and data/terrain until it reports (~9-10 h from its start).
  Census off (its call). Its notes have batch ids, PIDs, restart. Check `cores.py list` that its claim fell to 5.
- **UNCOMMITTED, waiting**: T-2123 tutorial batch (a1e21311d532ee817, done; files guide.ts, world.ts (T-2092 fill
  reverted), Guide.tsx, ripple.ts, NewGame.tsx, BuildTab/Blueprint/LinesTab/Search.tsx, saveFile.ts, clock.worker.ts,
  DemandSelector.tsx SHARED with T-2125, LineInspector.tsx, checks/t2123-tutorial.mjs, t2092 check, docs, SPEC).
  **T-2142 DONE** too (~15:20; adds main.ts `pauseNewGame` and TopBar.tsx's play `data-guide`; t2123 check 46/46,
  smoke 23/23). New string: step "Unpause the game". Commit T-2123+T-2142+T-2125 together once T-2125 reports (shared
  DemandSelector.tsx; main.ts/TopBar.tsx may hold others' hunks: check diffs).
- aece7ca3711268182 (Opus) **T-2125** "what this edit did" note top left (Vite 6168, ids T-2130/31).
- a1ed1934fdce63416 (Opus) **T-2132 + T-2133** bands: weave/split from the zoom-fixed ribbon correction, crowding marks
  (Vite 6156, scratchpad t2132, ids T-2140/41). "Lines side by side" stays off by default.
- af2ef23e7fb203080 (Opus) **T-2104** fare split middle ground (tree .tw-trees/t2104, Vite 6169, ids T-2134/35).
- a4f37e8b845c63f4a (Opus) **T-2122** Jakarta playable on a phone's memory (Vite 6160, ids T-2136..2138).
- ac23a9b30c1ddf26a (Opus) **T-2143** profit history all time, **T-2144** explorer "No train trip found", **T-2145**
  share bar ticks 10/20 + real-network mark (Vite 6171, ids T-2148/49).
- ac6227143084a837b (Opus) **T-2124 achievements stages 1-2** with her tiers (cities 2/10/25/100; trips 1M/10M/100M/
  300M/1B; station 500k/1M/2M/5M; line 500k/1M/3M/10M; Bicontinental, Longest road 1,000 km, + 20% inter-city, 20% of a
  city ("Monopoly?"), a 12 h trip; sandbox none; average day). Vite 6170, ids T-2150/51 (sea crossing = T-2150 later).
- aec7e76708c801d8f (Opus) **T-2147** train capacity research + proposal (160 per car is crush in the game, about
  Japan's rated load; Tokyo runs 150-200%). Vite 6172. A proposal for Anita, no default change.
Done today and committed: census rebuilds (ea66c3b7), terrain rebuild (8212d530), T-2093 jpn7 (2b4a4ad9), T-2124 design
(73d722a2), T-2126 gap (78e80f51), publish 821.

Next after these: Turkey (T-2079) when India's data is free; T-1856 hexagons (design in notes/T-1881.md); T-2064 (the
real save's flat crossings that are really grade-separated); T-2054, T-2095, T-2094, T-2099 (pipeline); T-2150 sea
crossing (WASM, own tree); T-2146 network diagram (design); T-2152 Lines tab overlap; T-2106/T-2107 more census zones.

## Open questions to Anita (asked, unanswered)
- T-2147 capacity proposal: hers when it reports. T-2104 fare split numbers shown 11-03; landed (a273050d).
- Answered ~10:15 11-03: all text so far acknowledged (T-2123, T-2125, T-2142, T-2143-45, T-2124 achievements, T-2146
  diagram strings, T-2158); feedback note moves too (T-2166); zone/region names need no review (trainworld/CLAUDE.md);
  late-game expensive things filed as a brainstorm (T-2167); diagram design to round 2 (her words in TASKS T-2146).
- Test on 8800: Settings "Lines side by side"; muted lines; crossing waits at Broadway-Lafayette (zoom 14+).
- maps/CLAUDE.md "Keep the work moving" bullet: uncommitted root file, hers to commit.
- Answered today: publish 821 (done); Israel lower prio than Turkey, Mexico, Brazil; default start New York; flatter
  demand than real is fine (0.4-0.7); "Confirm?" kept; Philippines and tutorial text acknowledged; no daily challenges,
  no riders estimate before paying, no music yet, showing off later.

## Open problems
- T-2064 [P2]: most of the real save's 340 flat crossings are lines that really pass over/under (L under Broadway), so
  the sim charges crossing delays and trains now visibly wait there. Needs `EdgeData.sep` on the real save (T-1751).
- T-2046 Seoul switch 6 s. T-1889 / T-1944 close with the terrain rebuild.

## Next (older list; her batches and countries come first)
1. T-1849, T-1863, T-1864 demand speedups.
2. T-1875, T-1876 to T-1879 (Australia follow-ups), T-1882.
3. T-1794 train-count settle. T-2009 Quad upkeep figures.

## Lessons from this run
- maps-0f 11-03: her TODO.md short-term can repeat an earlier batch line for line (11-03c = 11-02b; T-2154-56 were
  already T-2006-08). Diff new TODO items against the earlier notes/todo-*.md files before filing.
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

Started by maps-0f:
- a50434fcc8ab5819c (Opus) **T-2064** stage 1, the real save's grade-separated "flat" crossings: cause + fix, code
  only if small (no pipeline/ or data/ writes while India runs); notes/T-2064.md, Vite 6174, ids T-2164/65.
- T-2146 design DONE ~10:30 (archived): option B octilinear least squares; stages T-2159..T-2163 wait on Anita's
  look (review/diagram-tokyo.png); strings proposed: "Show network diagram (M)" / "Show map (M)", "Calculating
  diagram", "Save picture", file "<game name> diagram.png".

Reported done to maps-0f, uncommitted: **T-2122** (09:50; ARCHIVE entry added by me), **T-2125** (09:55; strings
for Anita in its report: "+20k riders/day", "−$856k profit/day", hovers "Riders per weekday: X to Y" / "Profit per
weekday: X to Y" + line row; asks: feedback "sent" note to upper left too? 8 s right?). **T-2132 + T-2133** reported
~10:00 (files map/lineBands, network, pick, renderer, rendererShaders, strokeGather, strokeLod, strokePost, perfFlags
bandsHold, checks/t1799-bands.mjs, docs/ui-map.md; no strings; re-time the band build on a quiet machine vs 300 ms). **T-2143/44/45** reported ~10:05 (strings listed in its report: shown to Anita).
**T-2104** reported ~10:40, done in tree t2104 (money.ts, regionMoney.ts, main.ts debug hook, SPEC, code-map,
track-and-service, checks t2104-fare-share.mjs + t1880; no strings, no WASM); its DECISIONS/ARCHIVE lines are in the
main tree. Land it (`agent-tree.mjs land t2104 "..."`, then remove) right AFTER the main-tree commit: the ff stops
while main.ts/SPEC/code-map/track-and-service hold uncommitted main-tree edits. Results for Anita in its report.
Plan: one commit of the done set once T-2124 reports (its hunks sit in types.ts, Dock, ToolNotes, Inspector, widgets,
clock.worker, clockClient, protocol, saveFile, shared with T-2123/T-2143),
so their hunks need no untangling; T-2147's `crush2/crush15` hunks (demandClient.ts packService, perfFlags.ts) are
measurement code, ask its report whether they stay. Patch of shared files: my scratchpad t2122-shared.patch.
maps-b2's subagent ids can't be messaged from here ("no transcript"); their reports come through maps-b2.

- 48b7ae8b (~10:45) = UI batch (T-2157, T-2158, T-2009, T-2166; T-2154-56 were already done as T-2006-08) + T-2153.
  **8800 = build 827 (48b7ae8b)**, freeze 20261103-051954, smoke 23/23 light and dark (~11:00).
- **T-2147 DONE** ~11:00 (maps-b2's agent): proposal in notes/T-2147.md for Anita (keep 160/car as 100%, run to 150%,
  crowding from 100%; about +4%; not what holds Asia low). The `crush2/crush15` flag stays on master as measurement.
- (was) **T-2153 DONE** ~10:35 (demandLod, demandClient, openedZones, clockClient, protocol, clock.worker,
  saveFile, main.ts onStationsAhead 2 lines, checks/t2005, docs demand-runtime + code-map, DECISIONS, ARCHIVE): new
  save header field `demandBuilt`. Commit with the UI batch (shared main.ts, DECISIONS, ARCHIVE). Not India's files:
  pipeline/tier0.py, world_plan.json, registry.json, market-rows.json, notes/T-2111.md.
- **T-2146 round 2 DONE** ~10:50 (agent a7e46912ec3685dba, resumable): Vignelli-style ribbons, snapped 0/45/90 with
  orthogonal-then-diagonal legs; review/diagram-tokyo-r2.png; stages T-2159..T-2163 + T-2168 boxes, T-2169 water wait
  on her go. Known flaws: fat/tilted pills at big interchanges (T-2161), far names (T-2162).

## Queued (maps-0f), older
- Her third 11-03 batch (notes/todo-2026-11-03c.md): T-2153 to T-2158, filed 09:32, lines uncommitted in TASKS.md.
  Start one UI agent for T-2154-T-2158 once the T-2123/T-2142/T-2125 commit is in and the UI agents above have
  reported (they hold BuildTab, style.css and the like); T-2153 after T-2122 reports (same memory code).
- cores.py at 09:32: 13 claimed of cap 10 (a full cap grants 1 each); no new core-using work until it drops.

Next free task id: T-2173 (T-2168/69 t2146 round 2, T-2170/71 t2154, T-2172 t2153; T-2041, T-2042, T-2047, T-2048, T-2050..2052, T-2055..2057, T-2061, T-2062, T-2065,
T-2066..2069, T-2072, T-2073, T-2085, T-2121 handed out but unused).
