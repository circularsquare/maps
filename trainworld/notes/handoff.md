# Orchestrator handoff, trainworld

Orchestrator: session 95c539d5-0cf1-43f1-8067-d0686e8d7331, name maps-dc [52e59f], since 2026-11-10 16:07
Updated: 2026-11-10 16:51 (maps-15 done winding down)

## Now

**LIVE = v0.2.1 = R2 2026-11-10d**, built from f5a2b979 on branch `tw/release-021` (= 2e2adc09 + the Exact edits hide),
website eff496a, verified all PASS. It has: HSR-as-tram fix (T-3078), tram/bus/transit wording (T-3064/3067/3068), picker
inter-city row (T-3074), Load without inter-city trips + OOM fallback (T-3072/3073), 10c's work. Not in it: track rebuild
(T-3059), Exact edits (hidden). `.tw-trees/hotfix` holds tw/release-021; branches tw/hotfix-10d (abandoned) and
tw/release-021 can be deleted with that worktree (`git worktree remove .tw-trees/hotfix`).
Due: purge R2 prefixes 2026-11-07c and 11-08 from 2026-11-11; 11-09, 11-09b, 11-09c a few days after; keep 11-10, 10b-10d.
**No silent updates any more** (Anita): every publish gets a version and an `app/src/ui/updates.ts` entry (strings to her).
**Website push needs her word each time** (the classifier refuses it otherwise).
**Master = 8846f377 + maps-dc's later commits** (maps-15 pushed through 8846f377). **8800 = 8846f377**: batch smoke on master
PASSED 27 (16:50). Exact edits shows with `?exact` (or whenever a line is already manual).
**maps-15 landed after the takeover (all its agents are done; nothing of maps-15's is in flight):**
- T-3089 + T-3080 Exact edits round 2 (41202320): row hover lights the station, every visit removable (`drop_visit`), half
  dot, visit numbers and add circles on the map. Strings "Click to remove this stop", "Click to add a stop here".
- T-3094 one-way single track (97564abe): switch + reverse in the track panel, arrows, routing/capacity by direction, return
  runs take their own way back. Strings shown to Anita ("wording seems fine"). Follow-ups T-3101, T-3102.
- T-3087 rebuild-to-level fix + T-3088 rebuild as tram and back (90e97d88). String "tram track".
- T-3099/T-3081 Line 1a (8846f377): turnedLine never reads a turn-back as a loop; set_manual keeps a closed line's path.
  Her save fix: select 1a, Stop editing, click the pressed Exact edits once. Follow-up T-3100.
- T-3093 Single track in Build (40af0202); real flyover design (42460efc, T-3092 after T-3103).
**Open with Anita from maps-15's last exchange:** default new single track to one way? (She asked "single track should
default be one-way right?"; maps-15 suggested keeping two-way since plain single-track lines run both ways; her call.)
Anita on flyover strings: "like both text"; keeping the junction inspector's old "Add a flyover" for now.

**Anita's answers today (for the record):** tram stop length on tram track = last tram length (as built); rebuild = full new
price, price + confirm, final, refused pieces visible; "transit" for the whole system; Exact edits ship only once good;
flyover cost fine; one-way on single track only; muted route = own colour at 40% + a little grey (done).
**Open with Anita:** standalone version (answered: a wrapper alone gains little); Resend over its 100/day: Anita
16:40 "lets jut wait for resend" (T-3105 tells the player when the quota is out). No releases until real flyovers land,
hopefully tonight (Anita 16:40). T-3070 non-home-based trips design waits ("can wait then start a design agent").
**Queued:** T-3063 bus loops partway; T-3033 split zones rebuild + T-3077 precomputed inter-city figure (one data batch);
T-3060 rebuild preview; T-3069 trams on real rail + street-snap icon; Anita's plan: community tab, shared maps, country
leaderboards (11/13), better schematic with timetables, open source (next week). P3s: T-3042, T-3036, T-3027, T-3002/3003,
T-3019, T-3011.
**maps-dc:** T-3104/T-3105 server done by mailname (shonei-server e1ac732, 7b68d01). Client `?site=trainsworld` in app/src/game/account.ts + ui/AccountPanel.tsx is maps-dc's,
uncommitted, tsc clean: commit it with the next batch. Strings approved (mail wording, quota message untranslated).
Server DEPLOYED 2026-11-10 16:20 and pushed by Anita.
- T-3107: Anita decided 16:50: dead ends and through stations identical, hints only "build a terminal" (terminal =
  30 tph) or, with one built, more track; never mention a dead end; drop "Add more platform tracks at the terminus".
  My rule: turning = 15/h without a terminal, 30/h with, per track pair, any station, ending or reversing partway; no
  90 s crossing for turns, no dead-end pooling, no REVERSAL_LIMIT_S. deadend (Sonnet, a79eb44b3fb6bfdbf) was briefed
  for that (round 2), then STOPPED before editing so maps-15 can land oneway + rebuild2 (both edit en.json, and oneway
  capacity.rs). Round-1 work parked at <maps-dc scratch>\deadend\t3107-round1.patch, files restored to HEAD. After
  both land: resume deadend (SendMessage) to reapply the patch and do round 2 on top. Anita 16:55: no "more track /
  platform tracks" hint either (one line arrives on one pair, so a quad station can't help it): with a terminal built
  and still full, the hint is to turn some trains at another station. Tell deadend this on resume.
- T-3106 street tram over water at grade: brief once rebuild2 (T-3087) lands, same water/level check.
Next free id **T-3108**.
Anita's TODO "do trams have shorter dwell times?": yes, 15 s against 30 s for trains (T-2926), answered 2026-11-10.

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
