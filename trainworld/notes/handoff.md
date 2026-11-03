# Orchestrator handoff, trainworld

Orchestrator: session 6b129d28-dd4c-41cd-a38e-7497bd99e810, name maps-1c, since 2026-11-03 12:49
Updated: 2026-11-03 12:55 (maps-0f handed everything to maps-1c at 12:55; see "Handed over by maps-0f" below)

## Handed over by maps-0f (12:55): all of it is maps-1c's now
- master **565f0c59** = T-1849 landed (tree t1849 removed; main WASM rebuilt). 8800 still build 831 (00e3d300).
- **Unlanded, done:** T-2186 (capacity 150%) + T-2180 (edit note level-of-detail hold): main tree, uncommitted, mixed
  with t2226's hunks in LineInspector, style.css, trackHover, perfFlags, docs: commit all three together when t2226
  reports (or untangle with keep_hunks.py in maps-0f's scratchpad). **t2193** (ferries F2): rerun smoke in its tree on a
  warmed Vite, then land AFTER that commit (it also edits sim/src/track/boarding.rs). **t2217** (station buildings 1):
  after t2193, put BLD1 after LFR1, save LEVEL 4, save.rs pin (4, 13), tests, land. Then rebuild 8800 (freeze
  20261103-051954 while India owns data/packs; maps-0f's snapbuild-frozen.cmd).
- **Still running (my subagents; each told to SendMessage its final report to maps-1c too):**
  ad82780d44630e96a T-2226/27 track text + lag (Vite 6191); ad948a26ee8f95ddc T-2222/23 band snap + gap looks (6189);
  a80cbe682243da006 T-2192 ferries F1 routes (6187); a567603f3dc3fdad7 T-2237 edit note as rail trips (6193).
  If one goes quiet, ask maps-0f to message it (maps-1c can't reach them by id).
- **India join** (PID 42596) is maps-1c's: details under "In flight" below.
- **Next to start:** T-2231 passing tracks sim (own tree, Anita said go), T-2218 station building tool + T-2220,
  T-2194 ferry tool (after F1 + F2), T-2236 crowding by load (after T-2222), T-2162/T-2163 diagram.

Read trainworld/CLAUDE.md and TASKS.md first. Landings are in ARCHIVE.md and `git log`.

## State
- master afda82d7 (handoff, T-2207 notes, merge help; no app change since 8800's 00e3d300 but that one line). Today (maps-0f): 5faa828f (achievements, edit note, profit history, tutorial, bands, phone memory),
  a273050d (T-2104 fare split), a53f0cd8 (T-2064 real save all `sep`), 48b7ae8b (T-2153 keep last 2 cities, UI batch),
  57e1e7cc (notes), a5a5ce4b (achievement rings, band fixes T-2140/82/83/84, ferry design, late-game research),
  90981710 (T-2185 merges marked), 00e3d300 (diagram stages 1-2 for every player).
- **8800 = build 831 (00e3d300)**, freeze 20261103-051954, smoke 23/23 light and dark. LIVE = 821 (R2
  `trainworld/2026-11-03b`, website dc49c5f). Publish only on Anita's word (recipe below).
- Uncommitted: the running agents' files, India's (pipeline/tier0.py, world_plan.json, app/src/game/registry.json +
  market-rows.json, notes/T-2111.md: never commit those until India lands), TODO.md (Anita's).
- **Cores cap 15** since ~11:50 (Anita out); `python cores.py cap 10` when she's back at the machine.
- Worktrees: .tw-trees/t1734 (kept, T-1734), t1849, t2193 (agents running). Land with `checks/agent-tree.mjs land <name>
  "<sentence>"` then `remove`; a landed sim change: the main tree's WASM is rebuilt by the land script.

## How to
- 8800: `cmd /c "<scratch>\snapbuild-frozen.cmd <hash> 20261103-051954 > <scratch>\snapbuild.log 2>&1"` (scratch =
  C:\Users\anita\AppData\Local\Temp\claude\c--Users-anita-projects-maps\76ebbfee-75f0-49e0-be85-230cb2ef8d79\scratchpad;
  it sets TW_PACKS to the freeze and TW_MARKET_DRIVES, needed while India owns data/packs; plain snapbuild.cmd otherwise).
  The build runs tsc, so it checks the committed tree. Then smoke on http://localhost:8800/trainworld/ light and
  SMOKE_SHADE=dark. Build number = `git rev-list --count <hash>`.
- Mixed files: `python <scratch>\keep_hunks.py <maps root> <path> <marker>...` stages only hunks whose added lines carry a
  marker (-U0). `mark_taking.py <TASKS.md> T-x=agent ...` marks lines taken.
- Publish: `freeze.py build <stamp>` with TW_MARKET_DRIVES set, `deploy.py --dist <dist> --version <v> --smoked`, website
  `git add trainsworld/ trainworld/`, commit "anitabuilder", push (her word), `deploy.py --verify --page --version <v>`.

## In flight (all Opus; ids are this session's subagents unless noted)
- **India T-2111 JOIN RUNNING, now mine** (its agent a57accbdda9f839db handed back 12:16): batch 20261103-121621,
  python PID 42596 (cores t2111, 5), `--jobs 6`, log artifacts/trainworld/t2111-join.log, run logs
  artifacts/trainworld/world_run/20261103-121621. Liveness `Get-Process -Id 42596`; ~2.5-3 h. Kit (join.cmd, simroot)
  in maps-b2's scratchpad ...\7ae5a6d2-b46c-4f33-9114-37a1eedcfcdb\scratchpad\t2111. Undo: `restore.py T2111 prejoin
  --go`. After it ends: `world_run.py status`, then playbook 8 (docs/adding-a-country.md): stale.py, zone_origins.py,
  load checks (ned, kol, one rural zone), smoke, register row, ARCHIVE; commit India's files (pipeline/tier0.py,
  world_plan.json, registry.json, market-rows.json, notes/T-2111.md, its DECISIONS line) and a new freeze for 8800.
  Its sim copy differs from HEAD in intercity.rs (T-2122, to the bit) and capacity.rs (T-2185, track only): fine.
- **T-1849 DONE** in tree t1849 (sim/src/demand/intercity.rs + tests, docs/demand-market.md, notes/T-1849.md +
  T-1849/timing.mjs; corridor table 2.7-4.3 s -> 1.0-1.25 s per worker, same to the bit; no WASM interface change).
  Land after 12:50 (`agent-tree.mjs land t1849 "..."`, rebuilds main WASM), then remove the tree.
- Cap back to 15 ~12:35 (her game ended early; she said use more cores); t2226 and t2222 told the pause is over.
- **T-2186 DONE** ~12:35 (report in; smoke 23/23, real save now 11,721,426; gain +1% not +4% because crowding from 112
  holds lines at 120-130%; asked Anita keep or start crowding at 100%). PLAN: one commit of T-2186 + T-2180 + T-2226/27
  once t2226 reports (they share LineInspector, style.css, trackHover, perfFlags, docs); then land t1849 and t2193
  (t2193 after a smoke rerun in its tree), rebuild 8800. T-2186's files (for reference):
  its files: sim/src/demand api.rs kernel.rs (+tests) network.rs, sim/src/track/boarding.rs (+tests), game/boarding.ts
  trip.ts demand.ts?, map/stretchLoad.ts trackHover.ts networkTips.ts, LineInspector/TrainInspector, demand-*.md docs,
  notes/T-2186.md; it holds a t2186 claim (asked to release).
- **T-2180 DONE** ~12:35, uncommitted (editNote.ts, demand.ts `editsSettling`, demandClient.ts 3 hunks, perfFlags
  `lodHold`, docs ui-map + demand-runtime, notes/T-2180.md): levels of detail held while edits settle. Commit with
  T-2186 when it reports (shared demandClient/perfFlags/demand.ts?). Asked Anita: note riders as rail trips, not boardings?
- Committed ~12:25: T-2161/T-2160 diagram stage 3 + T-2221 achievement tweaks + T-2208 design. Follow-ups T-2205
  (spacing), T-2206 (hooks), then T-2162 names, T-2163 clicks. T-2186 done per ARCHIVE, report pending: commit + 8800.
- Started ~12:20: ac12bcf1fcabb49ba **T-2217** station buildings sim + BLD1 save (tree .tw-trees/t2217, Vite 6190,
  T-2229); ad82780d44630e96a **T-2226 + T-2227** track text ("Level", "crossings", the crossing refusal) + crossing
  lag (Vite 6191, T-2230); aefb42a10e6d5a67b **T-2228** passing tracks design, incl. her "all stations on a line"
  button idea (T-2231..34).
- ad948a26ee8f95ddc T-2222 band snaps to own track at one zoom (her shot5) + T-2223 gap looks to show her (Vite 6189).
- a80cbe682243da006 T-2192 ferries F1 routes (main, Vite 6187); told: one tool (ports only, no water points),
  refusal "No water path found"; writes "F1 -> F2 route format" in notes/T-2177.md.
- **T-2193 F2 DONE** ~12:30 in tree t2193 (ports as stations, ferry lines, LFR1, save level 3, WASM in
  sim/src/track/wasm/ferry.rs; 338 cargo tests, checks/t2193-ferry.mjs 16/16). Smoke failed twice on Vite cold
  starts: rerun smoke IN THE TREE on a warmed Vite before landing. It touches sim/src/track/boarding.rs, which T-2186
  also edits uncommitted in the main tree: commit T-2186 first, then `agent-tree.mjs land t2193` (rebase may need a
  hand merge in boarding.rs). F3 must save `fleet_ferry` in the header (notes/T-2177.md "F2 -> F3 sim interface").
- **T-2217 DONE** ~12:40 in tree t2217 (station buildings sim + BLD1, save LEVEL 3; WASM add/reshape/rename/remove
  _building, building_quote; app game/buildings.ts joins complexes; smoke 23/23 in tree; checks/t2217-buildings.mjs).
  **Collision with t2193**: both add a block after LPH1 and raise save LEVEL to 3. Land t2193 first; then in t2217
  put BLD1 after LFR1, LEVEL 4, the save.rs pin (4, 13) (notes/T-2208.md section 5), rerun tests, land. Next: T-2218
  tool (upgrade row only), T-2220 the 0.8 rule, then T-2219 drawing.
Everything above is maps-0f's to land (it was told at 12:49; T-1849 landed as 565f0c59). maps-1c's own:
- ac4cca2590899fad0 **t2205**: T-2205 diagram spacing, then T-2206 hooks (main tree, app/src/diagram + diagram worker +
  DiagramButton + checks/t2160*, t2161*; Vite 6194; notes/T-2205.md). Then T-2162 names to the same agent.
Each reports by SubagentHandback; files listed in their reports. Commit each set when it reports (smoke on its Vite
was run by the agent; the 8800 build typechecks the commit).

## Next
1. When F1+F2 land: T-2194 (F3 Ferry tool, one button, ports only), then T-2195, T-2196 (fans: visits and
   attractions, no errands), T-2178. notes/T-2177.md.
2. After India lands: Turkey (T-2079); T-2213 islands pipeline step (if Anita OKs T-2207); T-2137 Delhi on a phone.
3. Diagram T-2162 (names), T-2163 (clicks, Save picture), T-2168 boxes, T-2169 water.
4. Her answers ~12:45: passing tracks GO (T-2231 sim in its own tree next, then T-2232, T-2233; hover "Allow faster
   trains to pass slower trains that are stopped here"; 400 m price; line button every station); crowding stays from
   70%; capacity strings OK; T-2237 edit note riders = rail trips (STARTED a567603f3dc3fdad7, Vite 6193); T-2236
   crowding marks by load past 100% (after T-2222). T-2223 gap look (when shown);
   T-2228 passing tracks design (when it reports). T-2207 islands approved (car ferries ignored): T-2213 after India.
   T-2200 named crossings after T-2150, T-2201 maglev design. T-2235 diagram loading bar with T-2163.
5. Older: T-1863, T-1864 demand speedups; T-2198/T-2199 merge follow-ups; T-2150 sea crossing (own tree).

## Open questions to Anita (asked, unanswered)
- T-2207 islands: build it? car ferries ignored or slow roads? the bound? the word ("Stay home" / "No way there")?
- Answered today: all strings shown so far OK (achievements, merges, edit note, history, ferries with "No water path
  found"); ferries go; big stations dropped for T-2208; upgrade-in-place and signals no; named crossings (hidden until
  built) and maglev yes; capacity 150% with crowding from 70%; diagram: build it, more compact, exposed to everyone.

## Lessons from this run
- Before starting an "Open" task, `git log --grep "T-<id>"` (T-1929 had landed a day earlier, line never moved).
- Her TODO.md can repeat an earlier batch word for word (11-03c = 11-02b): diff against notes/todo-*.md before filing.
- Commit as agents report when their files are separable; untangle shared docs with keep_hunks.py (-U0, not -U3).
- A full cap grants 1 core each, so many agents can sit at 13-14 of 10; India needs 6 for joins: plan for it.
- maps-b2's subagents can't be messaged from here; go through maps-b2.
- Her saves per origin: copy `IndexedDB\<origin>.indexeddb.leveldb`/.blob into a throwaway profile (maps-b2's scratch
  scripts listslots.mjs, dumpslot.mjs). Her Tokyo save file: C:\Users\anita\Documents\trainworld\Tokyo Seoul Fukuoka
  Busan (from anita.garden).save (read only).

## Anita's standing asks
- Player text: agents propose wording per the style rules, list every string old -> new; land without waiting; show
  her. Zone, region and realm names need no review (2026-11-03).
- TODO.md short-term = prompts: copy her words verbatim into briefs; save batches in notes/todo-<date>.md.
- Every demand task has a "done when". Buses settled. Thanks to agents in my own reply. Keep the work moving.

Next free task id: T-2238 (T-2235 = diagram loading bar; unused but handed out: T-2173/74, T-2187..2191, T-2197, T-2224/25 maybe; check before use).
