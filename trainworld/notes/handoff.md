# Orchestrator handoff, trainworld

Orchestrator: session 76ebbfee-75f0-49e0-be85-230cb2ef8d79, name maps-0f, since 2026-11-03 09:32
Updated: 2026-11-03 11:55 (context past 500k: a fresh orchestrator can take over any time)

Read trainworld/CLAUDE.md and TASKS.md first. Landings are in ARCHIVE.md and `git log`.

## State
- master 00e3d300. Today (maps-0f): 5faa828f (achievements, edit note, profit history, tutorial, bands, phone memory),
  a273050d (T-2104 fare split), a53f0cd8 (T-2064 real save all `sep`), 48b7ae8b (T-2153 keep last 2 cities, UI batch),
  57e1e7cc (notes), a5a5ce4b (achievement rings, band fixes T-2140/82/83/84, ferry design, late-game research),
  90981710 (T-2185 merges marked), 00e3d300 (diagram stages 1-2 for every player).
- **8800 = build 831 (00e3d300)**, freeze 20261103-051954, smoke 23/23 light and dark. LIVE = 821 (R2
  `trainworld/2026-11-03b`, website dc49c5f). Publish only on Anita's word (recipe below).
- Uncommitted: capacityKinds.ts merge help (mine: "Place the merge farther away ... or run fewer trains."), notes/T-2207.md
  (design done), the running agents' files, India's (pipeline/tier0.py, world_plan.json, app/src/game/registry.json +
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
- **India T-2111**: maps-b2's agent a57accbdda9f839db (message it via maps-b2, who relays). All 782 rows gated; join
  approved for 6 cores (~2.5-3 h + 45 min drives/checks), started ~11:55 (t2111drv claim). Owns data/packs and
  data/terrain until it reports: no pack/terrain writes, 8800 builds use the freeze. notes/T-2111.md has PIDs, logs.
- a9b86111490529148 T-2186 trains fill to 150%, crowding from 70% (Vite 6185).
- a0aec0944e89e200c T-2180 edit note down both ways + "18 cities" (Vite 6182); long-running, ask if quiet.
- a164858ef4917b379 T-2161 + T-2160 spacing: diagram interchanges, view; also move the diagram button to the Cities
  tab's top row, left (Vite 6186).
- a335231643fa865c6 T-2221 achievement tweaks: smaller rings set lower, "/100M", smaller medals, toast fades (Vite 6181).
- ad948a26ee8f95ddc T-2222 band snaps to own track at one zoom (her shot5) + T-2223 gap looks to show her (Vite 6189).
- a80cbe682243da006 T-2192 ferries F1 routes (main, Vite 6187); told: one tool (ports only, no water points),
  refusal "No water path found"; writes "F1 -> F2 route format" in notes/T-2177.md.
- a7f9db866e5028da9 T-2193 ferries F2 sim line kind, tree t2193 (WASM + LFR1 save block).
- a549624d4ab855dea T-1849 corridor modes speedup, tree t1849 (re-measuring after T-2122 first).
- ad297ba0d90ddda72 T-2208 station buildings design (notes/T-2208.md, mockups artifacts/trainworld/t2208/).
Each reports by SubagentHandback; files listed in their reports. Commit each set when it reports (smoke on its Vite
was run by the agent; the 8800 build typechecks the commit).

## Next
1. When F1+F2 land: T-2194 (F3 Ferry tool, one button, ports only), then T-2195, T-2196 (fans: visits and
   attractions, no errands), T-2178. notes/T-2177.md.
2. After India lands: Turkey (T-2079); T-2213 islands pipeline step (if Anita OKs T-2207); T-2137 Delhi on a phone.
3. Diagram T-2162 (names), T-2163 (clicks, Save picture), T-2168 boxes, T-2169 water.
4. Waiting on Anita: T-2207 calls (car ferries ignored?, the 3-4x bound, "Stay home"), T-2223 gap look, T-2208 design,
   T-2200 named crossings after T-2150, T-2201 maglev design.
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

Next free task id: T-2226 (unused but handed out: T-2173/74, T-2187..2191, T-2197, T-2224/25 maybe; check before use).
