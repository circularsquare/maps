# Orchestrator handoff, trainworld

Orchestrator: session 6f38e27c-5131-4ca5-81d2-160b4694f258, name maps-c4 [3f93f3], since 2026-11-10 09:06
Updated: 2026-11-10 12:30 (maps-c4 is past 500k tokens; a fresh orchestrator can take over any time)

## Now

**LIVE = 2026-11-10 (v0.2, trams)**: Pages built 30b52da at 12:31, `--verify --page` all PASS (7 checks). The rest of
this paragraph is how it got there. **v0.2 (trams) is published as version 2026-11-10** (Anita's go 12:15: "ok, good to publish when they land"). Built from
88c3ebf3; smoke8800 light and dark PASS (12 each); `deploy.py --version 2026-11-10 --smoked` done, data verified on R2;
website commit 30b52da pushed ~12:25 (only trainsworld/ and trainworld/; her modified `pages/projects/abouttrainsworld.md`
left uncommitted, hers). **Pending:** background script `scratchpad\pages_verify.sh` (task b09t2sto6, log
`...\6f38e27c-...\scratchpad\pages_verify.log`) waits for the Pages build of 30b52da, then runs
`deploy.py --verify --page --version 2026-11-10`. If the session died: `gh run list --repo circularsquare/circularsquare.github.io --limit 1`,
then run the verify by hand. Once it passes, **LIVE = 2026-11-10** and tell Anita https://anita.garden/trainsworld/ is up.
Previous LIVE was 2026-11-09c (a6f43c1b). Due: purge R2 prefixes 2026-11-07c and 11-08 from 2026-11-11; 11-09 and 11-09b a
few days after; 11-09c once 11-10 has run a few days.

**Master = 20a6f601, pushed** (12:32). Master smoke PASSED 27 at 12241ef1; both cargo suites 436/616 at 8c02a9a3. **8800 = 88c3ebf3.**
shonei-server e9c9e1a (achievement counts) and 46876ae (signedIn) deployed and pushed with her approval.

**Running (Sonnet, main tree):**
- urlview done, committed d1229e50, T-3044 archived (sessionStorage `tw-view`; reload returns to the city; incoming hash
  honoured then removed). **8800 = d1229e50.** tramrow (Sonnet, Vite 6164, id a9efce75358329243): T-3045 tram line row says "Tram", then T-3046 the two snap toggles exclude each other. Then T-3047 trams' zoomed-out size halfway between train and bus (edit made by maps-c4 in rendererShaders.ts, tramrow checks it). Then T-3048 a setting to hide ramp marks. Next free id T-3049. Anita approved both rail fixes ("our goal for trainworld is not to show real world lines, just real world track"; freight stays out): railgap resumed for Fix 4 then Fix 1 (noritetsu's cn_register.py; noritetsu map changes too). railgap done (notes/T-3043.md): Nankun line is in OSM; dropped because noritetsu's China register cuts lines at the last 12306 passenger station and the layer hides track no line runs over (n=1); ~8-12k km like it (estimate), plus 43k km hidden by the r/p clause in `SHOWN` (realNetwork.ts). Asked Anita: Fix 4 (show all track a line runs over, app + realgraph ~5 min) is my call to do; Fix 1 (China tail sections) edits noritetsu's cn_register.py, so it also changes the noritetsu map: asked; freight stays hidden unless she says. Ships in the next publish (v0.2.1) on Anita's word.
- railgap (agent id a2344b053f2846378): T-3043 investigation only: the conventional line near Long'an missing from the
  real-rail layer (second such report), why the realgraph build drops it, how much else the cause drops, fix and rebuild
  cost; notes/T-3043.md. No rebuild without reporting.

**Queued (after v0.2, lean Sonnet, prefer compute over tokens):** T-3033 rebuild the 37 split zones (notes/T-3012.md;
~4 core-hours, 18 joins; plan unit-list edits only, never `zones --replan`). T-3042 one flat-junction mark per spot on a
looped line (P3). T-3036 road gaps debug overlay (P3). T-3023/T-3024 manual stop sequence (P3). T-3027 slopes slow trains
(P3, design first). T-3002 shorter tram ramps, T-3003 trams at grade over road bridges (notes/T-2996.md). T-3019 buses in
the diagram (P3, "until someone asks"). T-3011 Cairo border off the Nile (P3). Next free id **T-3045**.

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
