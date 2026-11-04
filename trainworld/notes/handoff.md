# Orchestrator handoff, trainworld

Orchestrator: session 8509086d-46cd-4c5f-8d59-da9348f4a5e0, name maps-c9, since 2026-11-03 23:29
Updated: 2026-11-04 03:05

Read trainworld/CLAUDE.md and TASKS.md first. Landings are in ARCHIVE.md and `git log`.

## State
- master **11405b29** (~02:55: Taiwan replan shipped, T-1999, T-774, Turkey/Russia plan rules). Before: 47fe69ec
  diagram T-2272/T-2289, 8c965a29 UI + demand batch, 3ecdc51c audit, bb48b358 tutorial/fares/Fleetwood.
- **8800 = build 872 (11405b29)**, plain from data/packs, smoke 23/23 light and dark; real save 12,509,688 rail trips.
  LIVE = 821 (R2 `trainworld/2026-11-03b`, website dc49c5f). Publish only on Anita's word (release "later this week").
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

## In flight (maps-c9's subagents; conditions in scratch common-brief.md)
- **Pack lock: countries, since 02:58**: Turkey's gate run, batch 20261104-025828 (74 rows, jobs 4, Vite 6371, log
  artifacts/trainworld/countries/t2294-gate.log). Then Turkey's join needs my go. Russia + Belarus build-only batch
  20261104-024516 (181 rows) runs alongside (claims 11).
- abb1ccd32095e266b **countries** (Opus): Turkey T-2079/T-2294 (74 rows built), then Russia (unheld; 13 rows + Minsk;
  4 Siberian towns held: suk xaf gub xag) and Belarus; Ukraine if data. geoBoundaries RUS stops at the recognised
  border. Ids T-2294..99.
- **Turkey 74/74 gated; join running**: batch 20261104-033213 (`join --countries TR --jobs 6`, log
  artifacts/trainworld/countries/t2294-join.log, python 32828; backup data/work/pre-T2294/prejoin 932 MB, undo
  `restore.py T2294 prejoin --go`). Census check: packs match ADNKS (Adana 2.28M census vs 2.30M pack); the plan was
  low because GHSL undercounts Adana's urban ilce (0.84M vs 2.01M registered); only Adana's shape is affected (one
  2.3M zone, no metro row): T-2296 plans zones on census totals. Russia/Belarus are GHSL-only (Rosstat VPN-2021 by
  municipal district suggested as a P3 source).
- ~03:45 countries: Turkey gate 62/74 gated, 12 parked (pop_vs_plan 4: Adana 2.44, Arnavutkoy 1.91, Torbali 1.58,
  Ceyhan 0.42 from the census hook moving people between outlines; pop_vs_fua 2: Gaziantep, Antalya; top_zone_jobs 4
  one-town zones; browser 2: Konya, Erzurum origins in villages, moved). 10 relaxed with reasons; relax rerun of 12
  rows under way (log artifacts/trainworld/countries/t2294-relax.log), countries still holds the lock; join next.
- ~03:25 replan2: PR/US plan written (PR 4 to 5, US 187 to 182; 163 old ids out, 159 new rows; backup
  pre-T1189/pre-us 2.7 GB); build-only run of the 159 rows next. **Order (decided ~03:40): Turkey's join next (go given
  once countries confirms its gate counts); the US retire waits until the 159 rows are built (hours at 1-4 cores;
  AWS suggested), then retire, gate and join back to back, so data/packs never sits for hours with the US
  half-removed.** Don't rebuild 8800 from packs between a retire and its join.
- ~03:55 **AWS run t2301-us-1 approved** (replan2): one m7a.8xlarge one-time spot, us-east-2, 4 h hard timer, the 155
  new US rows on 6 lines (~1-1.5 h, under $1); tools/aws/bundle.py gained `--head-sim` (commit it). Run id and
  timer end go in notes/T-1189.md; teardown.py must end it. PR's 4 rows build at home. Then land + retire + gate +
  join back to back, on my go.
- abaa9dba7f94a9a7b **replan2** (Opus, ~03:00): T-2301 US + Puerto Rico replan (then T-2302 Europe if time). Kit from
  scratch replan\, its own scratch replan2\. Asks my go for plan write, retire, register/gate, join. Ids T-2326..29.
- a92f3da5ec3aa7955 **solvespeed** (Opus): T-2307 (walk rule solve -20%), sim FERRY_HEADWAYS_S, "$6B" comment, T-2308.
  Owns sim/src/demand/*, StationUpgrades.tsx, ui-inspectors.md's T-2308 lines (uncommitted). Port 6180.
- **Done tonight** (agents retired): audit (3ecdc51c); replan Taiwan (11405b29; tig "Taitung" 1.29M Central Range
  zone, follow-up T-2300); diagram (47fe69ec; T-2321); display, panels, buildtab, ferries, demand (8c965a29).
  Strings and pictures for Anita: notes/for-anita.md "Open now".
- Next to start once cores free (after Russia's build): T-2137 Delhi phone memory (Sonnet), passing tracks
  T-2245/T-2248/T-2249 (Opus), T-2291 colour near-duplicates + T-2280 tutorial hovers (Sonnet).
- Next free id **T-2330**.

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
