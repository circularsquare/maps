# Orchestrator handoff, trainworld

Orchestrator: session 8509086d-46cd-4c5f-8d59-da9348f4a5e0, name maps-c9, since 2026-11-03 23:29
Updated: 2026-11-04 06:20

Read trainworld/CLAUDE.md and TASKS.md first. Landings are in ARCHIVE.md and `git log`.

## State
- master **95a724e0**. Tonight, newest first: 95a724e0/889628f4 Turkey shipped + RU/BY gated + US/PR planned and
  built, ecce08c4 passing tracks, 784f0a82 line colours, ce6a8f6d walk-rule speed + T-2308, 11405b29 Taiwan replan,
  47fe69ec diagram T-2272/T-2289, 8c965a29 UI + demand batch, 3ecdc51c audit, bb48b358 tutorial/fares/Fleetwood.
- **8800 = build 877 (95a724e0)**, plain from data/packs, smoke 23/23 light and dark; real save 12,509,849 rail trips.
  Has Turkey (shipped), Taiwan's replan, Rikers. LIVE = 821 (R2 `trainworld/2026-11-03b`, website dc49c5f). Publish
  only on Anita's word (release "later this week").
- Freeze: data/frozen/20261103-162459 (post-India, old: lacks Rikers, Taiwan, Turkey). **Do not rebuild 8800 from
  packs while the US is half-retired** (from replan2's retire until its join ships); use the freeze if a code
  change must reach 8800 meanwhile.
- **Cores cap 15** overnight; `python cores.py cap 10` when Anita is back at the machine.
- Disk ~32 GB free. Worktrees: .tw-trees/t1734 (kept, T-1734) only. data/work/terrain-fine (3.8 GB) is kept.
- Uncommitted: TODO.md (Anita's) only, as of 06:20.

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

## In flight
- ~08:10 US join: the South Atlantic unit (atl, 42 packs) re-gate failed other_packs_unchanged on old member brv
  (built 10-23): group restored, 30 new rows parked (FL, GA, Carolinas, VA, WV are holes until fixed). Other units
  going on (Northeast shipped). replan2 will settle brv (hint: T-2213's parts fields in old headers), re-gate the
  30, re-join that unit alone: +30-45 min after the join.
- **US join running since 07:33** at 12 jobs (159/159 gated; python 23116, log artifacts/trainworld/t1189-us-join.log;
  1.5-3 h). It can't change jobs mid-run: if Anita needs cores before it ends, tell her it holds 12 until ~09:00-10:30.
  Then: T-2054 chain, names, successors, smoke, real save; then commit + 8800; then the RU/BY join.
- ~07:40 US gate 152/159; 7 parked (6 mean_commute_km 25.1-27.9 countryside zones, fab jobs share), relaxed with
  reasons, re-gating (log artifacts/trainworld/t1189-us-gate2.log); join straight after.
- ~06:40 replan2: 155 landed, **163 old US ids retired (the US is half-removed until the join ships)**, market heal
  20 packs; retire.py's stale-file move briefly took the 159 rows' files, all put back and the kit fixed. **US gate
  running** (python 21884, port 6373, log artifacts/trainworld/t1189-us-gate.log), 8 cores.
- **Pack lock: replan2** (abaa9dba7f94a9a7b, Opus), go given ~06:20 for T-2301's steps 1-6: backup prejoin-us,
  land.py t2301-us-3 (155 rows from AWS), retire_us.cmd (163 old ids), gate_us.cmd (159 rows, 8 cores, port 6373),
  join_us.cmd (1.5-3 h), T-2054 chain, names.py --write, zone_successors --set, smoke, real save; then
  `bucket.py rm t2301-us-3`. Kit: scratch replan2\; steps in notes/T-1189.md "State at hand-back". It reports any
  park between retire and join at once.
- **Next pack lock: Russia + Belarus join** (181/181 gated; countries agent retired): `backup.py T2295 prejoin
  RU,BY,<neighbours>`, `<kit>\join_ru.cmd`, then `terrain_ru.cmd` (terrain_fine, zone_terrain, market_drives,
  build_inputs). Kit: scratch countries\; steps in notes/T-2295.md "Next". Needs a fresh agent (Opus) after the US
  join, or run the .cmd files directly.
- Done tonight, agents retired: tutorial T-2263, fares T-2266, Fleetwood T-2290, design T-2267/T-2270, audit,
  UI batch T-2271/T-2269/T-2273..75/T-2286, demand T-2265/T-2278/T-2276, diagram T-2272/T-2289, walk-rule speed
  T-2307 (rest T-2324), T-2308, line colours T-2291, passing T-2245/48/49, Taiwan T-1189 part, Turkey T-2294 (74
  shipped), Russia/Belarus T-2295 (181 gated). Strings and pictures: notes/for-anita.md "Open now".
- **phone DONE, committed 20a6e867**: T-2137 met (Delhi renderer 1.22 GB at worst, both emulated phones; T-2333 unused).
- Not started, could go next: T-2324 rest of the walk-rule cost, T-2302 Europe
  replan after the US (628 zones to 462), T-2296 plan on census totals, T-2300 Taiwan's Central Range split,
  T-2321 diagram overlaps, T-2332 copied/bus/ferry colours, T-2326 AWS seed, T-2298 gate on a frozen app.
- Next free id **T-2333** (T-2327..29 handed to replan2, T-2330/31 unused).

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
