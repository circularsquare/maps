# Orchestrator handoff, trainworld

Orchestrator: session d455ee97-22bd-4dd5-a487-a2a4e1449db1, name maps-15 [bf4395], since 2026-11-10 13:48
Updated: 2026-11-10 15:15

## Now

**LIVE = v0.2.1 = 2026-11-10d** (f5a2b979 on tw/release-021, website eff496a, verified ~15:45, all PASS). Before: 10c
(33a8ccec). Due: purge R2 prefixes 2026-11-07c and 11-08 from
2026-11-11; 11-09, 11-09b, 11-09c a few days after; keep 11-10, 10b, 10c and the next.
**v0.2.1 PUBLISHING** as R2 version 2026-11-10d from **f5a2b979** on branch `tw/release-021` (worktree `.tw-trees/hotfix`):
= 2e2adc09 (master before T-3059 landed) + the hide commit (3172bd30 on master). Anita 15:10: "ok actually i think probably
best to just hide "exact edits" and ship the fixes." So exact mode's toggle is hidden (`EXACT_EDITS = false` in
LineInspector.tsx) and T-3059 rebuild is left out (landed on master after, untested by her). Run:
`scratchpad\publish021.cmd` (snapbuild f5a2b979, tsc+i18n in .tw-snap, smoke8800 light+dark, deploy --version 2026-11-10d),
log `scratchpad\publish021.log`. Then website `git add trainsworld/ trainworld/`, commit "anitabuilder", push (Anita said
yes to this hotfix), `sh <c4 scratch>/pages_verify.sh <sha> 2026-11-10d`. Then snapbuild master again for 8800.
`.tw-trees/hotfix` and branches tw/hotfix-10d (ed09d682, abandoned) and tw/release-021 can go after the publish.
**Master = 3172bd30 + T-3059 archive** (pushed 52fa18bf; push soon). 8800 will be the release build until rebuilt.
Anita's player report "All my high speed lines turned in tram lines": app-side only (state edge kind 1 read as tram: drawn
as tram track, track panel "Tram track", 4-car stations, tram upkeep figure in the app); the sim and saves were never
changed. Fixed by T-3078, in v0.2.1.

v0.2.1 smoke8800 passed light+dark (f5a2b979), uploaded as 2026-11-10d, website eff496a pushed; pages_verify running.
Master pushed at 16992b65. 8800 rebuilding at 16992b65.
**Master = 16992b65**: T-3079 landed (ee29c42c); Exact edits shows only with `?exact` in the address
(`EXACT_EDITS` in LineInspector.tsx): Anita tests on 8800 at localhost:8800/trainworld/?exact=1, then make it always on.
After the v0.2.1 publish: snapbuild 16992b65 for 8800, one smoke on master (batch rule), push master.
**8800 = 16992b65** (Exact edits with ?exact=1, track rebuild). **No silent updates any more** (Anita 15:50): every
publish gets a version and an updates.ts entry.
**Running (Sonnet, main tree):** curve25 (id a390bcca8c970904e, Vite 6172) T-3082 curves to 25 m; buildexit (id
a950b264ff9052c93, Vite 6173) T-3083 build tools close line/station views; busedit (id af518479c4c4684d3, Vite 6174)
T-3084 bus edit route: old route muted, node drags go through Edit route and undo. T-3086 reserved for busedit.
Started 16:10: flyover (Opus, read-only, id a8360dea8512f2827) T-3085 design, aim to ship 11/11 ("start design tonight");
rebuildfix (Opus, id a321fbe3706c5ae16, tree `.tw-trees/rebuild2`, Vite 6175) T-3087 [P1] rebuild-to-level bug + T-3088
rebuild as tram and back; exactmap (Sonnet, id a3896fc2d95771b05, tree `.tw-trees/exact3`, Vite 6176) T-3089 + T-3080
(hover highlight, ends removable, half dot, numbers and circles on the map; x and station click do the same). busedit also
does ferries. Anita on Exact edits: "exact edits looks better".
line1a (Opus, id ae2330e4f40883a14, main tree, diagnose first): T-3099 [P1] Line 1a (plain, interlined with Line 1) drawn
with end loops, turning round Line 1's loop, strip crossing perpendicular (T-3081 folded in). Suspect: her Exact edits
on/off left 1a a one-way closed list. Fix patches touching LineInspector/lines.rs come to me to route (exact3 owns them).
Ids: flyover T-3092-94, rebuildfix T-3095-96, exactmap T-3097-98, line1a T-3100. Next free **T-3101**.

**Queued:** T-3070 non-home-based trips design agent (Anita: "can wait then start a design agent"); T-3063 bus loops partway
(can start); T-3033 split zones rebuild + T-3077 precomputed inter-city figure (one data batch); T-3060 rebuild preview;
T-3069 trams on real rail + street-snap icon. P3s: T-3042, T-3036, T-3027, T-3002/T-3003, T-3019, T-3011.
Open with Anita: standalone version (answered: wrapper alone gains little). Master 0cc01142: a start-menu pick without the
lite button clears the no-inter-city flag (Anita OK'd), not in v0.2.1.
Next free id **T-3087**.
Anita's priority plan (TODO.md, 2026-11-10): bug fixes; exact station mode; single track / flyover junctions (11/11);
community tab, shared maps, country leaderboards (11/13); better schematic with generated timetables (next week); open source
(next week).

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
