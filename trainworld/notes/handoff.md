# Orchestrator handoff, trainworld

Orchestrator: session 8509086d-46cd-4c5f-8d59-da9348f4a5e0, name maps-c9, since 2026-11-03 23:29
Updated: 2026-11-04 02:15

Read trainworld/CLAUDE.md and TASKS.md first. Landings are in ARCHIVE.md and `git log`.

## State
- master **8c965a29** (maps-c9 ~02:10: the UI batch T-2271/T-2269/T-2273..75/T-2286/T-2279-off and demand
  T-2265/T-2278/T-2276). Before: 3ecdc51c audit, bb48b358 tutorial/fares/Fleetwood/design note, 8dc8961f islands
  (maps-1c). Left out of 8c965a29 on purpose: diagram's WIP (app/src/diagram/*, its checks, ui-diagram.md, T-2272
  DECISIONS lines), and countries'/replan's pipeline and game data (registry.json, market-rows.json,
  zone-names.json, zone-successors.json, pipeline/*, tools/country/*, notes/T-1189.md, notes/T-2294.md).
- **8800 = build 870 (8c965a29)** on the freeze, smoke 23/23 light and dark; real save 12,509,800 rail trips (frozen
  packs lack the T-2257 Rikers fix: rebuild from data/packs once Taiwan's gate is done). LIVE = 821 (R2 `trainworld/2026-11-03b`,
  website dc49c5f). Publish only on Anita's word (release "later this week").
- Freeze: data/frozen/20261103-162459 (post-India). Build 8800 on it (`snapbuild-frozen.cmd`) while packs are written.
- **Cores cap 15** overnight (Anita asleep, "feel free to use all cores overnight"); `python cores.py cap 10` when
  she is back at the machine.
- Disk ~45 GB free. Worktrees: .tw-trees/t1734 (kept, T-1734) only.
- Uncommitted: agents' files; TODO.md (Anita's).

## Anita before bed (2026-11-04 ~01:20), in her words where it matters
- "we can continue to run more countries overnight? perhaps turkey or russia / eastern europe?"; "we can addd more
  countries though and replan old regions yes, as we prepare for a release later this week."; "feel free to use all
  cores overnight. and keep going een if you run into 50% context as i won't be up to start the next orchestrator."
- Closing done tasks: "i think theres quite a few candidates that we already did. im not sure why we're not closign
  these tasks but please have someone check again" (agent audit).
- Tutorial strings: her rewording applied in bb48b358 (green-bubble note dropped; "Red borders indicate track at full
  capacity." is my correction of her "crowded segments": red borders mark track at capacity, ui-map.md). Notes show
  only in the tutorial card (she asked; yes).
- Design T-2267/T-2270 (T-2282..T-2285): not answered yet; don't build.

## In flight (maps-c9's subagents, launched ~01:45; brief conditions in scratch common-brief.md, ownership there)
- ~03:15 countries: **Turkey 74/74 built** (batch 20261104-021141, build-only). **Russia + Belarus building**
  (batch 20261104-024516, 181 rows: 173 RU + 8 BY, build-only, claims 11). Russia's 13 rows and Minsk unheld; 4 lone
  Siberian towns held (suk xaf gub xag); realms "turkey", "belarus" added; geoBoundaries RUS stops at the recognised
  border (no Crimea/occupied areas built). **Waiting: Turkey's gate run (`run --countries TR --no-join`, 30-60 min)
  gets my go once Taiwan's join ends.**
- abb1ccd32095e266b **countries** (Opus): Turkey T-2079 + build, then Russia (unheld) + Belarus, Ukraine if data;
  territory default = recognised borders, Crimea/occupied unbuilt (ask Anita). Ids T-2294..99. Joins/pack writes
  only after my "go" (one writer at a time with replan).
- a24a3ce58427a5a79 **replan** (Opus): T-1189 (Taiwan, US+PR, Europe): prerequisites (T-1999, T-774 in
  pipeline/world_plan.py: go given 01:55, countries told to keep off that file), dry runs and a look, then builds.
  Ids T-2300..04. Pack writes only after my "go". **~02:40: Taiwan gated 7/7, successors final (ead->eai, xid/zht->zhb); join go given (backup prejoin-tw,
  join_tw.cmd).** Turkey's run batch 20261104-021141 (PID 22384) started without a go at ~02:11, at its extract;
  asked countries whether it gates soon. Earlier: **Pack lock: replan, since ~02:20** (Taiwan: retire ead/zht/xid,
  world_run TW --no-join, 7 new zones; backup data/work/pre-T1189/pre-tw; batch 20261104-015157, PID 17932).
  Build 8800 frozen until it releases. Successors interim (xid, zht -> tai; ead -> none) until the gate registers the
  new zones, then replan sets ead=eai, xid=zhb, zht=zhb. **Next lock: countries (Turkey, 76 rows: 8 metros, 68
  zones; TUR and RUS rules in world_plan.py)**, waiting for my go after Taiwan's gate.
- **audit DONE** (3ecdc51c): 13 closed or abandoned, 6 trimmed; CLAUDE.md "task is done" rule updated. Unsure, left
  open: T-430, T-340, T-2267/T-2270 (archive when T-2282..85 start), T-1193, T-1867, T-1444. notes/audit-2026-11-04.md.
- **diagram DONE, committed 47fe69ec** (a0c6d307f45a77263, retired, ~520k tokens): T-2272, T-2289; T-2321 filed (all
  Japan +3 overlaps/+4 close pairs; tyo-game close pairs 0 -> 2 new FAIL line; NYC wrong-side 10 -> 18). Not on 8800
  yet: next build after Taiwan's join.
- Next to start once the join ends and cores free (Turkey's build holds ~10): T-2137 Delhi phone memory (Sonnet),
  passing tracks T-2245/T-2248/T-2249 (Opus), T-2291 colour near-duplicates + T-2280 tutorial hovers (Sonnet).
- a92f3da5ec3aa7955 **solvespeed** (Opus, ~02:15): T-2307 (T-2265's walk rule made the solve ~20% slower; done when
  within 5% of before, rail identical), then sim FERRY_HEADWAYS_S, the "$6B" comment, T-2308 waiting room toggle.
  Owns sim/src/demand/*. Port 6180. Ids T-2324, T-2325.
- **DONE in 8c965a29** (agents retired): display (T-2271 map items; T-2279 behind `CROWD_MEAN_ON = false` in
  game/today.ts, `?crowdMean=1` previews, set true on Anita's OK), panels (T-2271 panel items, T-2286), buildtab
  (T-2269, level hover; ui/trackPrice.ts ready for T-2282), ferries (T-2273..75; T-2320 filed), demand (T-2265:
  real save 12,509,688 rail trips; T-2278; T-2276; T-2307, T-2308 filed). Strings for Anita in notes/for-anita.md.
- Next free id **T-2326**.

## Next
1. As code agents report: smoke on my Vite (6197), commit their files (explicit paths), rebuild 8800 (frozen while
   packs are written), smoke light and dark. Strings to Anita in the morning.
2. Grant pack-write "go"s one at a time (countries, replan); rebuild 8800 from packs after each ships.
3. When agents finish: T-2216 market across parts, T-2256; diagram T-2289, T-2242, T-2246, T-2247, T-2168, T-2169,
   T-2272; passing tracks T-2245, T-2248, T-2249, T-2239/40/41; T-2280, T-2291.
4. Morning: the design OK (T-2282..85), T-2279 curve, territory question, all strings.

## How to
- Scratch = C:\Users\anita\AppData\Local\Temp\claude\c--Users-anita-projects-maps\8509086d-46cd-4c5f-8d59-da9348f4a5e0\scratchpad
  (snapbuild.cmd, snapbuild-frozen.cmd, keep_hunks.py, mark_taking.py, lowdisk.ps1, common-brief.md). A new
  orchestrator should copy these to its own scratchpad and fix the path inside snapbuild-frozen.cmd.
- 8800: `cmd /c "<scratch>\snapbuild.cmd <hash> > <scratch>\snapbuild.log 2>&1"` (or `snapbuild-frozen.cmd <hash>
  20261103-162459` while packs are being written). Then smoke on http://localhost:8800/trainworld/ light and
  SMOKE_SHADE=dark (claim 4 cores). Build number = `git rev-list --count <hash>`.
- Pre-commit smoke: own Vite (`node trainworld/checks/dev-nowatch.mjs 6197`, restart after app edits), smoke; stop it
  by its port's PID (Get-NetTCPConnection -LocalPort 6197).
- Landing a tree: `git -C <tree> add --all -- trainworld`, commit "agent-tree: work in progress", `rebase master`,
  resolve, `rebase --continue`; both cargo runs, `npm run wasm`, typecheck and smoke in the tree; then
  `agent-tree.mjs land <name> "<sentence>"` and `remove`.
- Mixed files: `python <scratch>\keep_hunks.py <maps root> <path> <marker>...` (-U0).
- Publish: `freeze.py build <stamp>` with TW_MARKET_DRIVES set, `deploy.py --dist <dist> --version <v> --smoked`,
  website `git add trainsworld/ trainworld/`, commit "anitabuilder", push (her word), `deploy.py --verify --page
  --version <v>`. Afterwards prune publish/ and data/frozen/ to the current build and the latest published ones.

## Open questions to Anita (asked, unanswered)
- T-2267/T-2270 design: build T-2282..T-2284? optional cross "Stop (Esc)", 3 px coloured bar edge, T-2285 Shift+1-7?
  keep mode colours (11-15 dE from game colours)? (notes/for-anita.md "Open now").
- Fares: optional hover "Riders pay the old fares until demand has settled"?
- Strings maps-1c listed (Ferry tool T-2194, building tool T-2218, crowding T-2236, diagram T-2163, ferry demand rows
  T-2195, building hover T-2220): not acknowledged as far as I know.

## Lessons from this run
- Before starting an "Open" task, `git log --grep "T-<id>"`.
- Her TODO.md can repeat an earlier batch word for word: diff against notes/todo-*.md before filing.
- Commit as agents report when their files are separable; untangle shared docs with keep_hunks.py (-U0).
- Any new save block goes after BLD1 at LEVEL 5.
- `cargo test` without `--features demand` skips the demand kernel's tests (T-2250).
- C: can fill suddenly: check `Get-PSDrive C` before long runs.
- Agents near 450-550k tokens: retire after their report; give follow-ups to a fresh agent.
- An agent finished but "waiting on its own background work" re-sends its report each time that work ends: ignore
  the repeats.
- When Anita asks for cores free, check the real load (Get-Counter), not only the ledger: her games and our
  unclaimed headless pages don't show in cores.json.

## Anita's standing asks
- Player text: agents propose wording per the style rules, list every string old -> new; land without waiting; show
  her. Zone, region and realm names need no review.
- TODO.md short-term = prompts: copy her words verbatim into briefs; save batches in notes/todo-<date>.md.
- Every demand task has a "done when". Buses settled. Thanks to agents in my own reply. Keep the work moving.
- Disk: delete orphaned temp files always; keep only the current build and the latest published ones in
  artifacts/trainworld/publish/ and data/frozen/; delete superseded backups once a later state checks clean.
