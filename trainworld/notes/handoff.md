# Orchestrator handoff, trainworld

Orchestrator: session 6b129d28-dd4c-41cd-a38e-7497bd99e810, name maps-1c, since 2026-11-03 12:49
Updated: 2026-11-03 14:30

Read trainworld/CLAUDE.md and TASKS.md first. Landings are in ARCHIVE.md and `git log`.

## Latest (14:30)
- 8800 = **build 837 (5cd94db2)**, smoke 23/23 light and dark.
- master **455bef52** = T-2231 passing tracks sim landed (tree removed; express +1.2 min at 10/10, 8 s of it overtakes,
  the rest existing queues: accepted, follow-ups T-2239/40/41 filed). Its TASKS/ARCHIVE/DECISIONS/notes T-2228 edits
  are uncommitted in the main tree: commit with the next batch.
- **T-2217 rebased and tested (352 cargo tests) as branch tw/t2217 commit 9477d889, NOT on master**: the fast-forward
  was refused because t2194 has uncommitted edits to game/types.ts, workers/clock.worker.ts, clockClient.ts,
  protocol.ts. When t2194 reports: commit its work, then `git -C .tw-trees/t2217 rebase master` (resolve those four),
  rerun tests, `land t2217` again, remove the tree. ARCHIVE line for T-2217 has a blank "landed" hash to fill.
- a15d9b19d67e1affb **t2232** passing tracks panels, tree .tw-trees/t2232 (from 455bef52), Vite 6188.
- T-2233 (map drawing) after t2194 leaves the renderer; T-2218 + T-2220 after T-2217 lands.

## State
- master **5cd94db2** (T-2193 ferry sim landed), before it 6587f89d (T-2186 150% load, T-2180, T-2237 edit note as
  "trips/day", T-2226/27 "Level" text + road tile decode, T-2192 ferry router, T-2222/23 band looks behind `perfTry`
  flags), ee978151 (maps-0f's handoff), 565f0c59 (T-1849). Tree t2193 removed.
- **8800 = build 831 (00e3d300)**; rebuild to 5cd94db2 STARTED ~13:58 (my scratch snapbuild-frozen.cmd, freeze
  20261103-051954, log scratch\snapbuild.log); then smoke light + SMOKE_SHADE=dark. LIVE = 821 (R2 `trainworld/2026-11-03b`,
  website dc49c5f). Publish only on Anita's word.
- Uncommitted: running agents' files; India's (pipeline/tier0.py, world_plan.json, app/src/game/registry.json +
  market-rows.json, notes/T-2111.md: never commit until India lands; its DECISIONS line went in with 6587f89d);
  TODO.md (Anita's).
- **Cores cap 15** (Anita away); `python cores.py cap 10` when she's back at the machine.
- Disk: 34 GB free after pruning 13:35. A background watch (scratch lowdisk.ps1, 6 h) reports under 3 GB.
- Worktrees: .tw-trees/t1734 (kept, T-1734), t2217 (rebase onto master IN PROGRESS, agent t2217b resolving),
  t2231 (passing tracks agent). Land with `checks/agent-tree.mjs land <name> "<sentence>"` then `remove`.

## How to
- Scratch = C:\Users\anita\AppData\Local\Temp\claude\c--Users-anita-projects-maps\6b129d28-dd4c-41cd-a38e-7497bd99e810\scratchpad
  (snapbuild.cmd, snapbuild-frozen.cmd, keep_hunks.py, mark_taking.py copied from maps-0f's).
- 8800: `cmd /c "<scratch>\snapbuild-frozen.cmd <hash> 20261103-051954 > <scratch>\snapbuild.log 2>&1"` while India
  owns data/packs (plain snapbuild.cmd otherwise). The build runs tsc on the committed tree. Then smoke on
  http://localhost:8800/trainworld/ light and SMOKE_SHADE=dark. Build number = `git rev-list --count <hash>`.
- Pre-commit smoke: own Vite (`node trainworld/checks/dev-nowatch.mjs 6197`), warm one page load, then smoke; stop it
  by its port's PID.
- A tree whose rebase will conflict: `git -C <tree> add --all -- trainworld`, commit "agent-tree: work in progress",
  `rebase master`, resolve (or brief an agent), `rebase --continue`, then `land` (it squashes).
- Mixed files: `python <scratch>\keep_hunks.py <maps root> <path> <marker>...` (-U0).
- Publish: `freeze.py build <stamp>` with TW_MARKET_DRIVES set, `deploy.py --dist <dist> --version <v> --smoked`, website
  `git add trainsworld/ trainworld/`, commit "anitabuilder", push (her word), `deploy.py --verify --page --version <v>`.
  Afterwards prune publish/ and data/frozen/ to the current build and the latest published ones.

## In flight (all Opus, this session's subagents)
- **India T-2111 JOIN** (handed back by its agent 12:16): batch 20261103-121621, python PID 42596 (cores t2111, 5),
  log artifacts/trainworld/t2111-join.log, run logs artifacts/trainworld/world_run/20261103-121621. Background
  Wait-Process notifies me. Kit in maps-b2's scratchpad ...\7ae5a6d2-b46c-4f33-9114-37a1eedcfcdb\scratchpad\t2111.
  Undo: `restore.py T2111 prejoin --go`. After: `world_run.py status`, playbook 8 (docs/adding-a-country.md):
  stale.py, zone_origins.py, load checks (ned, kol, one rural zone), smoke, register row, ARCHIVE; commit India's
  files and a new freeze for 8800 (then prune the old freeze). Then T-2213 islands step can use data/packs.
- ac4cca2590899fad0 **t2205**: T-2205 diagram spacing, then T-2206 hooks (main tree, app/src/diagram, diagram
  worker, DiagramButton, checks/t2160*, t2161*; Vite 6194; notes/T-2205.md). Then give it T-2162 names.
- a2b46e89c23cd6c45 **t2231**: passing tracks sim, tree .tw-trees/t2231 (from ee978151), Vite 6196. Bit in the
  existing upgrades field, no LEVEL bump if room. Lands after t2217 (rebase). Then T-2232 panels, T-2233 map.
- a3f9ae4a7c387b842 **t2217b**: resolving t2217's rebase (save.rs, wasm.rs conflicts): BLD1 after LFR1, LEVEL 4,
  pin (4, 13); tests, wasm, t2217 + t2193 checks, smoke in tree on Vite 6198. Doesn't land; I do. Then T-2218 tool +
  T-2220 0.8 rule (main tree, after T-2194 is out of the inspectors or coordinated).
- a3f8911247aa75454 **t2194**: ferry tool F3 (main tree, Vite 6199; checks/t2194-ferry-tool.mjs). Strings to show Anita.

## Next
1. When t2217b reports: land t2217, then t2231 when it reports (rebase). Rebuild 8800 after each batch.
2. T-2236 crowding marks by load past 100% (renderer; after T-2194 leaves the renderer). T-2222 waits on an export of
   her current Tokyo game + view and zoom. T-2223 waits on her pick of a gap look (pictures below).
3. T-2195, T-2196 ferries in demand (after T-2194); T-2178.
4. After India lands: T-2213 islands (pipeline, then T-2214, T-2215 "No route", T-2216); Turkey (T-2079); T-2137.
5. Diagram T-2162 (names), T-2163 (clicks, Save picture, T-2235 loading bar), T-2168 boxes, T-2169 water.
6. T-2238 the 2 s placement (Ozone Park); T-2230 road tile install; T-2209 narrow canals. Older: T-1863, T-1864,
   T-2198/T-2199, T-2150 sea crossing (own tree), T-2200 named crossings, T-2201 maglev design.

## Open questions to Anita
- "train trips/day" in the top bar's hover and the achievement ("1M train trips/day"): become "trips" too? (asked 13:40)
- T-2223 gap look: now / bandEdge / bandWide / bandFine (agent's pick) / bandSpaced, pictures
  artifacts/trainworld/t2222/look-*.png. (to ask)
- T-2222 band snap: not reproduced on her file; needs an export of her current Tokyo game plus the view and zoom. (to ask)
- Answered 13:40: islands go, word "No route"; edit note "trips"; temp files always delete; keep only the current and
  latest published builds. Earlier today: passing tracks go (400 m price, every station; hover "Allow faster trains to
  pass slower trains that are stopped here"), crowding from 70%, capacity 150%, ferries go (ports only, "No water path
  found"), named crossings and maglev yes, diagram for everyone.

## Lessons from this run
- Before starting an "Open" task, `git log --grep "T-<id>"` (T-1929 had landed a day earlier, line never moved).
- Her TODO.md can repeat an earlier batch word for word (11-03c = 11-02b): diff against notes/todo-*.md before filing.
- Commit as agents report when their files are separable; untangle shared docs with keep_hunks.py (-U0, not -U3).
- India needs 6 cores for joins: plan for it.
- maps-b2's and maps-0f's subagents can't be messaged from here; go through their session.
- Her saves per origin: copy `IndexedDB\<origin>.indexeddb.leveldb`/.blob into a throwaway profile (maps-b2's scratch
  scripts listslots.mjs, dumpslot.mjs). Her Tokyo save file: C:\Users\anita\Documents\trainworld\Tokyo Seoul Fukuoka
  Busan (from anita.garden).save (read only, older than her live game).
- C: can fill suddenly (0.5 GB free at 13:00 with a join running): check `Get-PSDrive C` before long runs.

## Anita's standing asks
- Player text: agents propose wording per the style rules, list every string old -> new; land without waiting; show
  her. Zone, region and realm names need no review (2026-11-03).
- TODO.md short-term = prompts: copy her words verbatim into briefs; save batches in notes/todo-<date>.md.
- Every demand task has a "done when". Buses settled. Thanks to agents in my own reply. Keep the work moving.
- Disk (2026-11-03): delete orphaned temp files always; keep only the current build and the latest published ones
  in artifacts/trainworld/publish/ and data/frozen/ (prune after each publish or freeze).

Next free task id: T-2239 (unused but handed out: T-2173/74, T-2187..2191, T-2197, T-2224/25; check before use).
