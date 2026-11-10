# Orchestrator handoff, trainworld

Orchestrator: session d455ee97-22bd-4dd5-a487-a2a4e1449db1, name maps-15 [bf4395], since 2026-11-10 13:48
Updated: 2026-11-10 14:15

## Now

**LIVE = 2026-11-10b** (built from 97c11df7; Pages 5e68055 verified 13:40). v0.2 (trams) went out as 2026-11-10 the same
day. Due: purge R2 prefixes 2026-11-07c and 11-08 from 2026-11-11; 11-09, 11-09b, 11-09c a few days after; keep 11-10, 11-10b.
**8800 = 97c11df7.** maps-c4 finished 14:12 (no agents of its own left). Website: Anita's `pages/projects/abouttrainsworld.md`
edit is uncommitted and hers; `noritetsu/build_tiles.py` is uncommitted and hers.

**In progress: silent update 2026-11-10c** (pre-approved: "good to publish once it passes checks"). Contents: T-3050
1976e339, T-3055 34da31c3, T-3054 9003c8db, T-3053 6168e1aa, plus maps-15's commit of T-3051, T-3061, T-3062 (staged by
hunks with `keep_hunks.py`: the main tree's i18n files also hold wording's unfinished T-3064, so never commit them whole
until T-3064 is done). Steps: commit the index, snapbuild, `npm run check` in .tw-snap, smoke8800 light+dark,
`deploy.py --version 2026-11-10c --smoked` (`--skip-data` if no data changed), website `git add trainsworld/ trainworld/`,
commit "anitabuilder", push, `pages_verify.sh <website sha> 2026-11-10c` (maps-c4's scratch has pages_verify.sh and
silent.cmd as patterns). Set the updates.ts entry if 10c has one.
For Anita to test after 10c: a line ending at a through station, with and without a terminal (T-3053).

**Running:**
- manualstop DONE (tree `.tw-trees/manualstop`, T-3023 + T-3024 both stages; cargo 440/620, check 20/20, smoke 27 on its
  Vite). WAITING TO LAND: `agent-tree.mjs land` fast-forwards the main tree, which fails while trainwords2/transit have the
  i18n files and maybe LineInspector.tsx dirty. Land when both have reported and been committed; then wasm rebuild in the main
  tree, commit its DECISIONS.md line (already in the main tree), archive T-3023/T-3024 (and T-2988 if nothing left), 8800.
  Calls kept: toggle stays visible but disabled when it can't fold back; "+" per row, not a bar between rows.
- rebuild DONE (id a2b5ad8262d71e795, tree `.tw-trees/rebuild` from 9003c8db): T-3059, cargo+check+smoke pass. Lands after
  manualstop; its i18n inserts will conflict in the rebase: resume the same agent to rebase onto master (take both sides,
  `l10n.mjs merge`), add the T-3051 height-hint counting in `rebuildPicked` (`heightRunLength`/`noteHeight`), then land.
  Open: "Curves still limit speed" hover not added; junction/upgrade costs not charged in a rebuild.
- lowmem DONE: T-3071 scoped (notes/T-3071.md; T-3072 pool by memory ~half a day, T-3073 "Load without intercity trips").
  Waiting on Anita: what to assume when Chrome reports 8 GB (proposed 5 GB); T-3073 OK?
- T-3064 landed ec7b878d (after 10c's commit, so in the next update), with "Trams yield here" (Anita's wording).
- trainwords2 DONE (T-3067, archived, uncommitted: capacityKinds, capacityMarkers, TroubleList, LineInspector, upgrades,
  StationUpgrades, issues, world, today, schedule, LinesTab, networkTips, stopMarkers, checks/t3067-tram-words.mjs, i18n).
  Commit together with transit's T-3068 once it reports (both in the i18n files), then land manualstop. T-3068 "transit" for the whole system (demand view, mode shares, trip views): agent transit (Sonnet, Vite 6195),
  Anita OK'd "transit" in general. T-3064 strings: Anita "text all looks great".
- T-3074 landed 0119903a (row left out before a result). Precompute filed as T-3077 (with the next data batch / T-3033).
- Uncommitted DECISIONS.md line is manualstop's (T-3023): commit it with that landing.

- lowmem (Opus, id a372974d1a94dbeca, read-only, Vite 6192): T-3071 low-memory exploration into notes/T-3071.md. Ids T-3072-3073.
- 10c: 8800 = 33a8ccec; smoke8800 light 12 PASS, dark 12 PASS; data uploaded to R2 (deploy.py --version 2026-11-10c, all
  PASS). The website files are written but NOT committed/pushed: the auto-mode classifier refused the website commit+push
  ("Create Public Surface"). Waiting on Anita's word, then: `git -C ../website add trainsworld/ trainworld/`, commit
  "anitabuilder", push, then `pages_verify.sh <sha> 2026-11-10c`. Her own `pages/projects/abouttrainsworld.md` stays out.

**Queued:** T-3070 design agent once this round is done (Anita 14:30: "can wait then start a design agent"). T-3063 bus loops partway along a route (after manualstop lands: bus.rs). T-3033 rebuild the 37 split zones after
10c is published (writes data/packs; notes/T-3012.md; ~4 core-hours, 18 joins; unit-list edits only, never
`zones --replan`). T-3060 rebuild preview after T-3059. T-3069 trams on real rail + new street-snap icon. T-3070
non-home-based trips (P3, design, a demand change: Anita decides whether to scope now; I suggested after this round).
P3s: T-3042 one flat-junction mark per spot, T-3036 road gaps overlay, T-3027 slopes slow trains (design first),
T-3002/T-3003 tram ramps and road bridges, T-3019 buses in the diagram, T-3011 Cairo border.
Next free id **T-3078**.

**Open with Anita:**
- National achievement thresholds for countries without a save were guessed (Spain, Britain, Italy, Germany, Indonesia,
  Mexico, Brazil, Australia, Turkey); she raised KR/TW to 10% and CN to 2%. `NATIONS` in app/src/game/achievements.ts.
- Translations of the day's new strings (hidden achievements, v0.2 notes, hint, bus fare) are first pass only.
- The post `notes/v0.2-post.md` is hers to reword and post; its v0.1.1 / quiet-update boundary is inferred.
- Wiki-first: trainworld/CLAUDE.md now sends agents to wiki/index.md first (her OK); watch whether agents use it well.
- Not filed, to discuss later (her word): satellite map, single track.

**Today's work (detail in ARCHIVE.md, newest first):** v0.2 notes, looped-line view and its follow-ups (T-3025,
T-3039-3041), hidden achievements and player shares (T-3037/3038), road gaps 3 m at ends (T-3034), faster tram ghost and
sharp turns (T-3016/3035), end-of-line loop (T-3021/3022), bus fare multiple (T-3028), trams yield at crossings (T-3032),
split zones planner fix (T-3012), twstats/twfeedback (T-3030/3031), delete hint (T-3026), long number words (T-3020),
speed colour ramp and tram upkeep (T-3013/2999), tram playtest fixes (T-3014-3018), M key, dock, diagram trams
(T-3008-3010), Israel/Palestine (maps-8e), tramfix (maps-8e).

## How to
- Scripts in trainworld/checks/: `snapbuild.cmd <hash>`, `snapbuild-frozen.cmd <hash> <freeze>` (packs from a freeze while a
  run owns data/packs), `smoke8800.cmd` (8800 light then dark, only before a publish; logs in artifacts/trainworld/
  smoke8800-*.log), `keep_hunks.py`, `mark_taking.py`. Run via PowerShell **from the maps root** (Set-Location first; a
  PowerShell call from another folder fails with "cannot find the path"):
  `cmd /c "trainworld\checks\snapbuild.cmd <hash> > <scratch>\snapbuild.log 2>&1"`. Never run two snapbuilds at once
  (they share .tw-snap; one collided 2026-11-10).
- maps-c4's scratch (`...\6f38e27c-5131-4ca5-81d2-160b4694f258\scratchpad\`) has `archive_lines.py` (move TASKS lines to
  ARCHIVE with a given entry), `cargo2.sh` (both cargo suites), `pages_verify.sh`.
- Smoke dev: own Vite via `node trainworld/checks/dev-nowatch.mjs <port>` (restart after app or WASM changes), then
  `$env:TW_CORES='4'; node trainworld\checks\smoke.mjs http://localhost:<port>/` (~3 min).
- Publish: notes/T-924.md "Commands"; `venv\Scripts\python.exe trainworld\tools\deploy.py --dry-run --version X`, then
  `--version X --smoked` (~15 min upload), website `git add trainsworld/ trainworld/`, commit, push; Pages deploy can be
  slow (`gh run list --repo circularsquare/circularsquare.github.io`), then `--verify --page --version X`. The in-game
  updates list is `app/src/ui/updates.ts` (set the entry's `at` at publish).
- shonei-server deploy: `.\deploy\deploy.ps1` from that repo, alone in its own PowerShell call (combined with a push it
  was refused); then `git push` separately. Only with Anita's word.
- Build number = `git rev-list --count`. If 8800 does not answer, restart serve.py detached.
- Landing a tree: `node trainworld/checks/agent-tree.mjs land <name> "<sentence>"`, then `remove <name>`.
- Commit by path: `git commit -m "..." -- <paths>`; `git add` a new file first. Check `git diff` of shared docs before
  committing: an agent's in-progress paragraph rode along once (docs/ui-building.md, bf762960).

## Lessons from this run
- Never edit docs with PowerShell 5.1 Get-Content/Set-Content: it reads UTF-8 as cp1252 and writes a BOM, garbling every
  non-ASCII character. Use the Edit tool.
- deploy.py's rclone (32 transfers) can saturate Anita's Wi-Fi upload and slow her game on 8800.
- Always `date` before writing a time: both orchestrators' estimates ran 1-2 h ahead on 2026-11-10.
- Two sessions can share a name: message by ref.
- A world run's gates serve the main tree's app/ live: hold app/ writes during gates.
- Agents sharing the main tree work well when each brief lists the others' files; resume the same agent for follow-ups.
- Copying files back after a source swap keeps old mtimes: cargo/wasm-pack skip the rebuild (touch them).
- A trial that changes Rust constants with nothing visible goes into a notes/ patch, not the tree (T-3029).
- An agent's estimate of a change (T-3034 +32 joins) can miss what the code really touches: recount before landing.

## Anita's standing asks
- **Usage this week (Anita 2026-11-10 11:50):** "at this rate we are gonna run out of usage fairly early in the week. thats
  okay, its good to front load work this week so that we can get this important update out some time today. after we get
  v0.2 out though we can return to leaning slightly to use sonnets, and prioritize the token-cheap compute-expensive approach
  if we have a choice. the remaining work is also just not super urgent i think." v0.2 is out: Sonnet by default, fewer
  agents, prefer approaches that spend compute rather than tokens.
- Player text: agents propose wording per the style rules, list every string old -> new; land without waiting; show
  her. Zone, region and realm names need no review.
- TODO.md short-term = prompts: copy her words verbatim into briefs.
- Every demand task has a "done when"; demand changes go to her first. Buses settled. Keep the work moving.
- Disk: delete orphaned temp files; keep only current and latest published builds.
