# Orchestrator handoff, trainworld

Orchestrator: session 6b129d28-dd4c-41cd-a38e-7497bd99e810, name maps-1c, since 2026-11-03 12:49
Updated: 2026-11-03 16:45

Read trainworld/CLAUDE.md and TASKS.md first. Landings are in ARCHIVE.md and `git log`.

## State
- master **e715556b** (T-2232 passing tracks panels). Today under maps-1c: 6587f89d (T-2186 150% load, T-2180, T-2237
  edit note "trips/day", T-2226/27 "Level" + road tile decode, T-2192 ferry router, T-2222/23 band looks behind
  `perfTry`), 5cd94db2 (T-2193 ferry sim), 455bef52 (T-2231 passing tracks sim), a1d96f2f (T-2205/06 diagram
  spacing + hooks), 736d8e88 (T-2194 Ferry tool), a2ffbef1 (T-2217 station buildings sim, save level 4), 26d78ae5
  (T-2162 diagram names), e715556b (T-2232).
- **8800 = build 843 (e715556b)**, freeze 20261103-051954, smoke 23/23 light and dark. LIVE = 821 (R2
  `trainworld/2026-11-03b`, website dc49c5f). Publish only on Anita's word.
- Uncommitted: running agents' files; India's (pipeline/tier0.py, world_plan.json, app/src/game/registry.json +
  market-rows.json, notes/T-2111.md: never commit until India lands); TODO.md (Anita's).
- **Cores cap 15** (Anita away); `python cores.py cap 10` when she's back at the machine.
- Disk: 34 GB free after pruning 13:35. Watch (scratch lowdisk.ps1, until ~19:05) reports under 3 GB.
- Worktrees: .tw-trees/t1734 (kept, T-1734), t2195 (agent). Land with `checks/agent-tree.mjs land <name> "<sentence>"`,
  then `remove`. If master's tree has uncommitted edits to a file the branch touches, the fast-forward is refused:
  commit those first.

## How to
- Scratch = C:\Users\anita\AppData\Local\Temp\claude\c--Users-anita-projects-maps\6b129d28-dd4c-41cd-a38e-7497bd99e810\scratchpad
  (snapbuild.cmd, snapbuild-frozen.cmd, keep_hunks.py, mark_taking.py).
- 8800: `cmd /c "<scratch>\snapbuild-frozen.cmd <hash> 20261103-051954 > <scratch>\snapbuild.log 2>&1"` while India
  owns data/packs (plain snapbuild.cmd otherwise). The build runs tsc on the committed tree. Then smoke on
  http://localhost:8800/trainworld/ light and SMOKE_SHADE=dark. Build number = `git rev-list --count <hash>`.
- Pre-commit smoke: own Vite (`node trainworld/checks/dev-nowatch.mjs 6197`), warm one page load, then smoke; stop it
  by its port's PID.
- A tree whose rebase will conflict: `git -C <tree> add --all -- trainworld`, commit "agent-tree: work in progress",
  `rebase master`, resolve, `rebase --continue`, rebuild its wasm, typecheck, smoke in the tree, then `land`.
- Mixed files: `python <scratch>\keep_hunks.py <maps root> <path> <marker>...` (-U0).
- Publish: `freeze.py build <stamp>` with TW_MARKET_DRIVES set, `deploy.py --dist <dist> --version <v> --smoked`, website
  `git add trainsworld/ trainworld/`, commit "anitabuilder", push (her word), `deploy.py --verify --page --version <v>`.
  Afterwards prune publish/ and data/frozen/ to the current build and the latest published ones.

## In flight (all Opus, this session's subagents)
- **India T-2111 JOIN**: batch 20261103-121621, python PID 42596 (cores t2111, 5), log
  artifacts/trainworld/t2111-join.log, run logs artifacts/trainworld/world_run/20261103-121621; at 16:40 on
  hyd-in-telangana. Background Wait-Process notifies me. Kit in maps-b2's scratchpad
  ...\7ae5a6d2-b46c-4f33-9114-37a1eedcfcdb\scratchpad\t2111. Undo: `restore.py T2111 prejoin --go`. After:
  `world_run.py status`, playbook 8 (docs/adding-a-country.md): stale.py, zone_origins.py, load checks (ned, kol, one
  rural zone), smoke, register row, ARCHIVE; commit India's files and a new freeze for 8800 (prune the old freeze).
  Then T-2213 islands can use data/packs.
- **T-2218 DONE ~17:10, uncommitted** (building tool: buildingView.ts, buildingDraw.ts, buildingFloors.ts, check
  t2218-building-tool.mjs 27/27, smoke 23/23; strings to show Anita). Its renderer.ts, pick.ts, tools.ts hunks sit
  beside t2233's in-progress ones: commit T-2218 + T-2233 together when t2233 reports. Follow-ups for T-2219: the
  Room row with the hall crowd, the dark floor rim (#8f8b85) faint; touch dragging of handles not built.
- af43a0646696e53af **t2163** diagram clicks, Save picture, loading bar (T-2163 + T-2235; main tree, Vite 6194).
- af2aca4951f7d73f6 **t2233** passing tracks on the map (main tree, Vite 6187; trainGuard.ts, station shapes).
- a285a0d99fbb3a25c **t2218** station building tool (main tree, Vite 6189; worker messages, StationInspector row,
  own floor/outline layer). Strings to show Anita.
- aa9e665d3b5427c97 **t2195** ferries in demand (tree .tw-trees/t2195 from e715556b, Vite 6191; sim/src/demand,
  clock.worker buildSnapshot, BusInspector rows). Its app edits will meet t2218's in clock.worker.ts at landing.

## Next
1. When t2233 reports: T-2236 crowding marks by load past 100% (renderer). T-2222 waits on her Tokyo export + view and
   zoom. T-2223 waits on her gap pick.
2. When t2218 reports: T-2220 the 0.8 rule (demand kernel, after t2195 lands), T-2219 building drawing.
3. When t2195 lands: T-2196 fans (10% of visits and attractions; no errands), T-2178.
4. After India lands: T-2213 islands (pipeline, then T-2214, T-2215 "No route", T-2216); Turkey (T-2079); T-2137.
5. Diagram: T-2163 (clicks, Save picture, T-2235 loading bar), T-2168 boxes, T-2169 water; T-2242/43/44/46/47 polish.
6. T-2238 the 2 s placement (Ozone Park); T-2230 road tile install; T-2209 narrow canals; T-2245 local waits check;
   T-2239/40/41 passing follow-ups. Older: T-1863, T-1864, T-2198/T-2199, T-2150 sea crossing (own tree), T-2200
   named crossings, T-2201 maglev design.

## Open questions to Anita (asked, unanswered)
- "train trips/day" in the top bar's hover and the achievement ("1M train trips/day"): become "trips" too? (13:40)
- T-2223 gap look: now / fine (pick) / wide / edge / spaced, artifacts/trainworld/review/gap-*.png.
- T-2222 band snap: needs an export of her current Tokyo game plus the view and zoom.
- Strings to acknowledge: passing tracks panels (T-2232) and the Ferry tool (T-2194), listed in the 16:45 message.
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
- Two worktrees that both bump the save level collide; any new save block now goes after BLD1 at LEVEL 5.
- A tree touching worker files (protocol.ts, clock.worker.ts) can't fast-forward while a main-tree agent edits them:
  land trees right after a main-tree commit.

## Anita's standing asks
- Player text: agents propose wording per the style rules, list every string old -> new; land without waiting; show
  her. Zone, region and realm names need no review (2026-11-03).
- TODO.md short-term = prompts: copy her words verbatim into briefs; save batches in notes/todo-<date>.md.
- Every demand task has a "done when". Buses settled. Thanks to agents in my own reply. Keep the work moving.
- Disk (2026-11-03): delete orphaned temp files always; keep only the current build and the latest published ones
  in artifacts/trainworld/publish/ and data/frozen/ (prune after each publish or freeze).

Next free task id: T-2248 (unused but handed out: T-2173/74, T-2187..2191, T-2197, T-2224/25; check before use).
