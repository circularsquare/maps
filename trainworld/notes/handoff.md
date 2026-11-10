# Orchestrator handoff, trainworld

Orchestrator: session 6f38e27c-5131-4ca5-81d2-160b4694f258, name maps-c4 [3f93f3], since 2026-11-10 09:06
Updated: 2026-11-10 13:45 (maps-c4 is past 600k tokens; Anita will start a fresh orchestrator)

## Now

**LIVE = 2026-11-10b** (silent update after v0.2; built from 97c11df7; Pages 5e68055 verified 13:40, 7 PASS). v0.2 (trams)
went out as 2026-11-10 earlier the same day. Previous: 2026-11-09c. Due: purge R2 prefixes 2026-11-07c and 11-08 from
2026-11-11; 11-09, 11-09b, 11-09c a few days after; keep 11-10 and 11-10b.
**Master = 1976e339** (last pushed 20a6f601; push again soon). **8800 = 97c11df7.** Master smoke PASSED 27 at 12241ef1;
cargo 436/616 at 8c02a9a3 (sim unchanged since except as noted in agents' reports).
Website: game thumbnail and featured strip pushed 70db5ff (Anita's OK); her `pages/projects/abouttrainsworld.md` edit is
uncommitted and hers. `noritetsu/build_tiles.py` (T-3043's `mark_track_of_lines`) is uncommitted: noritetsu is Anita's to commit.

**Running (Sonnet, main tree), all for a second silent update 2026-11-10c:**
- termstop (id ae10bb7caf0dbd50f, Vite 6167): T-3053 [P1] trains stop halfway into a terminal's platform; with a terminal
  the line drew past the station. Anita's rule: "it should be able to go either on the left track and stay parked in the
  station for a bit before heading back, or on the right track. i dont ever want them to visually go past the last station".
- vehname (id a817414493b9f21f0, Vite 6168): T-3054 [P1] buses/trams called trains; "Bus 53 bus" -> "53 bus"; check a bus
  hover's 1/240 capacity. Strings to show Anita.
- greenmatch (id a95798ce10c36b6cb, Vite 6169): T-3055 green tram hint shows roads the snap can't use (Arakawa water pools).
- Already committed for 10c: T-3050 two-loop line view (1976e339).
**10c is pre-approved** (Anita: silent updates "good to publish once it passes checks"): commit the three, snapbuild,
smoke8800 light+dark, `deploy.py --version 2026-11-10c --smoked` (add `--skip-data` only if no data changed), website
`git add trainsworld/ trainworld/` only, commit "anitabuilder", push, `pages_verify.sh <website sha> 2026-11-10c`. Use
`scratchpad\silent.cmd` as the pattern (cd back to the maps root after each called script).

**Next for the new orchestrator (lean Sonnet, prefer compute over tokens):**
- **T-3023/T-3024 manual stop sequence ("exact mode"), P2**, Anita: "i do want an explicit like "show me all the stops in
  order, both ways, so i can fine tune exactly as i want"". Design and cost in notes/T-2988.md (2a sim, 2b app, most of a
  day); own tree (WASM interface).
- T-3051 Select tool hint after 4 height changes (small). T-3052 rebuild built track with new height or type through a
  blueprint (design first; her description in TASKS.md).
- T-3033 rebuild the 37 split zones (notes/T-3012.md; ~4 core-hours, 18 joins; plan unit-list edits only, never `zones --replan`).
- P3s: T-3042 one flat-junction mark per spot, T-3036 road gaps overlay, T-3027 slopes slow trains (design first),
  T-3002/T-3003 tram ramps and road bridges, T-3019 buses in the diagram, T-3011 Cairo border.
Next free id **T-3056**.

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
