# Orchestrator handoff, trainworld

Orchestrator: session 8509086d-46cd-4c5f-8d59-da9348f4a5e0, name maps-c9, since 2026-11-03 23:29
Updated: 2026-11-04 08:50 (context past 500k: a fresh orchestrator can take over any time)

Read trainworld/CLAUDE.md and TASKS.md first. Landings are in ARCHIVE.md and `git log`.

## State
- master **7c7b87b5** (Anita's morning notes filed), 20a6e867 Delhi phone check T-2137. Before, newest first: 95a724e0/889628f4 Turkey shipped + RU/BY gated + US/PR planned and
  built, ecce08c4 passing tracks, 784f0a82 line colours, ce6a8f6d walk-rule speed + T-2308, 11405b29 Taiwan replan,
  47fe69ec diagram T-2272/T-2289, 8c965a29 UI + demand batch, 3ecdc51c audit, bb48b358 tutorial/fares/Fleetwood.
- **8800 = build 877 (95a724e0)**, plain from data/packs, smoke 23/23 light and dark; real save 12,509,849 rail trips.
  Has Turkey (shipped), Taiwan's replan, Rikers. LIVE = 821 (R2 `trainworld/2026-11-03b`, website dc49c5f). Publish
  only on Anita's word (release "later this week").
- Freeze: data/frozen/20261103-162459 (post-India, old: lacks Rikers, Taiwan, Turkey). **Do not rebuild 8800 from
  packs while the US is half-retired** (from replan2's retire until its join ships); use the freeze if a code
  change must reach 8800 meanwhile.
- **Cores cap 15**: Anita, back at 08:05, said "using lots of cores is fine"; lower to 10 if she asks.
- Disk ~32 GB free. Worktrees: .tw-trees/t1734 (kept), crowd, panels2, trackbar. data/work/terrain-fine (3.8 GB) is kept.
- Uncommitted: replan2's app/src/game/registry.json, market-rows.json and pipeline/world_plan.json (US join under
  way: commit after it ships); TODO.md (Anita's).

## Anita before bed (2026-11-04 ~01:20), in her words where it matters
- "we can continue to run more countries overnight? perhaps turkey or russia / eastern europe?"; "we can addd more
  countries though and replan old regions yes, as we prepare for a release later this week."; "feel free to use all
  cores overnight. and keep going een if you run into 50% context as i won't be up to start the next orchestrator."
- Closing done tasks: "i think theres quite a few candidates that we already did. im not sure why we're not closign
  these tasks but please have someone check again" (agent audit).
- Tutorial strings: her rewording applied in bb48b358 (green-bubble note dropped; "Red borders indicate track at full
  capacity." is my correction of her "crowded segments": red borders mark track at capacity, ui-map.md). Notes show
  only in the tutorial card (she asked; yes).
- Morning (08:05+): design OK with changes (T-2282..84 building), all other answers in notes/todo-2026-11-04a.md.

## In flight (Anita awake since ~08:05; her morning notes: notes/todo-2026-11-04a.md, filed T-2333..T-2347)
- **Worktrees** (made 08:35 from 7c7b87b5; UI agents work in them so the main tree's app/ and sim/ stay still under
  pipeline gates): .tw-trees/crowd, panels2, trackbar (+ t1734 kept). Land each with `agent-tree.mjs land <name>`
  after smoke in the tree (How to below), one at a time, then rebuild 8800.
  - ac5dd0c7e9c9f4fd3 **crowd** (Opus, port 6184): T-2335 undo weekends-run-Medium, dot = mean, ring = peak (takes in
    T-2279); T-2334 merged crowding hover with bars and station dots; T-2333 tutorial strings. Ids T-2348/49.
  - af74ea5e3acd95ea2 **panels2** (Opus, port 6185): T-2338 D cycles demand, E enable/disable; T-2340 money bars, no
    Net; T-2343 bubble floor 0.5 px; T-2345 Day under Demand bubbles; T-2346 side by side on; T-2347 capacity marks
    inside the line as a `?capInside=1` mockup. Ids T-2350/51.
  - ad213cd44e2fcd6fc **trackbar** (Opus, port 6186): T-2282 bars (level buttons today's size), T-2283, T-2284 aura
    (8 px fade, controls brought in, Stop (Esc) cross), T-2339 breakdown stretches collapsed. Ids T-2352/53.
- **Pack lock: replan2** (abaa9dba7f94a9a7b): US/PR join running since 07:33 at 12 jobs (log
  artifacts/trainworld/t1189-us-join.log); the South Atlantic unit (atl, 30 new rows) was parked on old member brv's
  header check and gets fixed and re-joined after (hint: T-2213's parts fields). The US has holes until then: no 8800
  rebuild from packs.
- **Next pack lock: countries2** (a072f929f244f1cc5, Opus): RU/BY join + T-2054 chain (kit scratch countries\), then
  T-2344 Ukraine by de facto borders (Crimea with Russia if drawable, occupied areas not drawn), then T-2341 Jakarta
  (Sukabumi/Cianjur/Cipanas cut out). Ids T-2354..58. Asks my go for every pack write.
- Then: T-2302 Europe replan (628 zones to 462).
- Done overnight (agents retired): see State's commit list; strings for Anita: notes/for-anita.md "Open now".
- Next free id **T-2359**.

## Next
1. As agents report: trees -> smoke in the tree, `agent-tree.mjs land`, `remove`, one at a time (rebase on master
   first; a landing that changed sim/ rebuilds the main tree's WASM: do that only while no gate runs). Pipeline ->
   commit app/src/game/*.json, pipeline/ and notes after each join ships. Then rebuild 8800 from packs once the US
   is whole, smoke light and dark; show Anita the strings (old -> new) from each report.
2. Pack-lock queue: replan2's US join and atl fix -> countries2 (RU/BY join, Ukraine, Jakarta) -> T-2302 Europe.
3. Then, as cores and files allow: T-2324 walk-rule cost, T-2300 Taiwan's Central Range, T-2296 census plan, T-2321
   diagram overlaps, T-2332 colours for copied/bus/ferry lines, T-2326 AWS seed, T-2298 gate on a frozen app,
   T-2216/T-2256 islands, T-2242/T-2246/T-2247/T-2168/T-2169 diagram, T-2239/40/41 passing.
4. Open with Anita: T-1920 (rail commuters only over cells a station reaches?).
5. **Takeover note**: this session's subagents report only to this session (maps-c9). A new orchestrator should
   ask maps-c9 (SendMessage) to relay or land each in-flight agent's report, or wait for maps-c9 to wind down per
   /handoff, before relaunching anything listed above.

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
- A gate's browser checks run on a Vite over the main tree: an agent's app edit or `npm run wasm` mid-gate makes
  every page throw (72 Russian rows parked on "no page errors", 2026-11-04). Hold code agents' app/WASM work while a
  gate runs, or gate from a frozen tree.
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
