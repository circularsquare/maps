# Orchestrator handoff, trainworld

Orchestrator: session 8be9971b-5363-40be-bba3-d8c1eaa4ca85, name maps-04 [6bfb1b], since 2026-11-10 21:40
Previous: session 95c539d5-0cf1-43f1-8067-d0686e8d7331, name maps-dc [52e59f], handed over 2026-11-10 21:40, finishing in-flight work
Updated: 2026-11-10 21:40

## Now

**LIVE = v0.2.1 = R2 2026-11-10d** (from f5a2b979, branch tw/release-021, worktree `.tw-trees/hotfix`, kept until the next
release in case live needs a hotfix). Due: purge R2 prefixes 2026-11-07c and 11-08 from 2026-11-11; 11-09, 11-09b, 11-09c a
few days after; keep 11-10, 10b-10d.
**Master = a670f0d9** (pushed through 4358fa86). **8800 = a670f0d9.**
**v0.2.2 is HELD** (Anita 20:20: "kinda shippable. but still feels quite unpolished"): release after this round lands. Then:
translation batch for every new/stale key (last one: agent l10nbatch, 3 groups x 3 passes, ~15 min), snapbuild,
smoke8800.cmd light+dark, `deploy.py --dry-run --version 2026-11-10e` (or a new date), set `V0_2_2_AT` in updates.ts,
publish and website push on her word (notes/T-924.md "Commands"). The changelog entry is in updates.ts (13 lines + 4 nerf
lines in red, T-3133); she has not reworded it yet. Anita: "No silent updates": every publish gets a version + entry.
**Server (shonei-server, all pushed, deployed 21:17):** country leaderboards live (T-3116); community in PREVIEW
(`SHONEI_COMMUNITY=preview`, `SHONEI_COMMUNITY_TESTERS=uad72a8ab66f84c44` = Anita, Google login; env backup
/opt/shonei/shonei.env.bak-20261111); testers skip the 60-minute gate (04171ac). She tests at
localhost:8800/trainworld/?acct=live&community. Server deploys need her word each time; ssh is allowed now ("be careful").

**Uncommitted in the MAIN tree, finished (commit together once margins2 reports; one smoke first):**
- T-3123 busstops: manual bus stops on loops (bus.rs `set_bus_stop` compares plans; busStops.ts; loop_tests.rs;
  checks/t3123-busloop-stops.mjs) + the frozen bubbles fix (stopMarkers.ts hides its box with no line).
- T-3124 loopwords: "close loop"/"open loop"/"Open loop" (en.json), map chevron white/near-black on the line (style.css
  `--map-chev`, stopMarkers.ts `alongTrack`), t3111-chevron.mjs; maps-dc updated t2899-loops.mjs and t3109-loopclick.mjs.
- T-3128 citycircles: achievements.ts `networkFigures` needs no reversals; docs/ui-panels.md. Hover left as is (Anita).
- T-3133 maps-dc: updates.ts `nerfs`, StartMenu.tsx, style.css `.changelog-list li.nerf`, 4 `updates.v022Nerf*` strings.
TASKS lines of T-3123/T-3124/T-3128 are already archived by their agents; T-3133 still in TASKS ("Built by maps-dc").

**21:55: maps-04 has taken over; maps-dc lands in-flight work only, ending at an 8800 rebuild (NO publish: Anita tests
first).** Done and uncommitted in main now: T-3125 margins2, T-3127 ferryport, T-3130 tramred (wasm.rs street_over_water,
draw.rs one line, rendererShaders.ts, tools.ts, bridge_tests.rs, checks/t3130-tram-ghost.mjs), plus the batch above.
community2's WIP shares en.json/context.json/all message files/ui-panels.md with them, so the whole main tree commits in
one go once community2 reports (one smoke first). Edit tool trims a trailing space in new_string: check `":"` in i18n.
flyover2 DONE (T-3120/3121, tree .tw-trees/flyover2, smoke passed; TASKS lines already archived): land it AFTER the
main-tree commit (its draw.rs/docs overlap main's dirty files; draw.rs needs tramred's one-line `street_pieces` change
kept). Ramps stay 4% (Anita). Landing stoporder after the main commit conflicts in stopMarkers.ts: stoporder's
headings (`line_stop_ways`/`node_heading`, no `oneWayStops`) win; keep T-3124's chevron colour and T-3123's hide-box fix;
final file must point chevrons along the track and give a one-way balloon loop half dots (Anita tested a670f0d9).
stoporder DONE (T-3122, tree .tw-trees/stoporder, smoke 27 + t3122-stoporder.mjs 18 PASS; WASM interface gains add_stop,
line_stop_ways; rebuild main WASM after landing). Land order: main commit (after community2), then flyover2, then
stoporder (resolve stopMarkers.ts as above), then one smoke on master, snapbuild, tell maps-04. No publish.
**Running (all reports come to maps-dc; if maps-dc is gone, check each one's tree/files and adopt):**
- margins2 (Sonnet, ab04e98ac351d8fde) DONE 21:40, uncommitted: T-3125 trams ~40/h (TRACK_SPECS tram section 60 s,
  platform margin 60 s, conflict 45 s), `POOLED_TURN_MIN_S` 240 s, `CONFLICT_S` 120 s; cargo 470/650 pass; files params.rs,
  upgrades.rs, capacity/tests.rs, tram_tests.rs, track/tests.rs, track-and-service.md, wiki/read/line-capacity.md, DECISIONS.
  Its TASKS line still to archive. Plan: commit the finished batch once tramred and ferryport report (they have WIP in the
  same tree: ferry/*.ts, ferryDraw.ts, ferryWater.worker.ts; draw.rs, wasm.rs, bridge_tests.rs, tools.ts, rendererShaders.ts),
  so one smoke covers a clean state.
- flyover2 (Opus, a339de0188163bf37, tree `.tw-trees/flyover2`, Vite 6181): T-3120 tighter flyovers (click point = far
  leg's join, no kiss, tight curves in tight spots) + T-3121 typed target speed below 160. Land with agent-tree.mjs.
- stoporder (Opus, a46cd71277664d787, tree `.tw-trees/stoporder`, Vite 6183): T-3122 barbell stop order (stop at every
  pass, no double stop in a row at a terminal), dots = diagram = hover. Land with agent-tree.mjs.
- tramred (Opus, ac36b927cb729e7c9, main, Vite 6189): T-3130 player report, street tram ghost red with no tag or price.
- ferryport (Sonnet, a003a7ec70b2ec84b, main, Vite 6186): T-3127 port snap radius + port tag not showing.
- community2 (Opus, a020e0b40b1daf1c3, main + shonei-server, Vite 6187): T-3132 community round 2 (copy named "<name> -
  <world>", self-like, menu font sizes, [Random][Biggest][Most liked] in [place] with country/region, two-people Share
  icon, Community button in Settings, share-current button in the window). Its server part needs a deploy (her word).
**Anita 21:45:** nerf lines fine; heading "Community trains worlds" exactly; shares stay snapshots, but the Share hover says
"Last shared 11/10 17:25", an own share opened in Community gets Share again, the Share dialog states the 3-a-day limit
(all sent to community2). Still to see: the map chevron's look (on 8800 after the next commit).
**maps-04's (22:05):**
- singletrack (Opus, maps-04's subagent, id addd2bb7c40268aea; resume by SendMessage): T-3134 + T-3135. Stage 1 DONE
  22:10, design in notes/T-3135.md (stages A ghost lanes ~2 h main tree; B joins + inferred way in sim ~1 day, WASM
  interface change so own tree; C app snap/ring/ghost ~half day; D switch + oneway value 3). Resume for A-D once
  maps-dc's landings are in (tools.ts, network.ts, draw.rs all dirty/in flyover2). Anita 22:20: plan OK, terminus
  branch two-way, **v0.2.2 holds until single track is fully fixed**, and question every "double = one object"
  shortcut, probably split double track into two objects (never let a player delete one half). Filed T-3137; singletrack
  stage 1b DONE 22:22 (notes/T-3137.md: 31-place audit; (a) join side ~2.5 days, (b) two objects ~20 days, (c) lanes
  inside the piece + per-node move table ~7-8 days, RECOMMENDED). Asked Anita 22:25 to pick. Stage A (T-3134 ghost, ~2 h,
  independent of the model) starts once tools.ts is committed. singletrack is at ~320k tokens: a fresh Opus for C1.
  Filed T-3138 [P3] left-hand running (side finding). Next free id T-3139.
- T-3136 thinner single / thicker double drawing (blueprints and built): Sonnet after tramred lands (rendererShaders.ts).
- **Changelog rewording to apply after maps-dc commits** (Anita 22:00; en.json `updates.v022*`, then translations):
  Line "update v0.2.2! flyovers, single track and right click loops..." (drop "one-way");
  Flyovers "new junctions are built as flyovers when auto-ramp is enabled"; Single "added single track";
  LoopButton "way to make loops on mobile"; StreetTram "street trams can cross water at grade on bridges";
  Catalan "added catalan"; REMOVE NerfSingle (single track was not reachable before); NerfTerminals "terminal capacity
  reduced to 30 trains per hour"; NerfTrams "tram track capacity reduced to 40 trams per hour"; NerfJunctions "trains
  crossing at a flat junction or crossing block the junction for 2 minutes, up from 90 seconds".
  Heading change confirmed by Anita. Flyover ramps stay 4% like normal ramps (Anita 22:10: "should not be steeper
  than normal ramps"; relayed to maps-dc). Ask about the shonei-server deploy only when community2 is ready.
  Anita 22:15 on 8800 (a670f0d9, artifacts/trainworld/t3122-anita-balloon-fulldot.png): stop chevrons point at the next
  station, should follow the track; a full dot on a one-way balloon loop. Covered by T-3124 `alongTrack` + T-3122 rule 3
  (`line_stop_ways`/`node_heading`, replaces `oneWayStops`): they conflict in stopMarkers.ts, stoporder's wins (told
  maps-dc). Verify both on 8800 after the landing before telling her.
  Re-check "track can curve tighter, down to 25 m" after flyover2 lands; stoporder (T-3122) may want a line.
**Filed, not started:** T-3126 HSR quad, T-3129 snap to road for normal track (design first), T-3131 bus out-and-back by
hand near-overlap stops. Next free id **T-3139**.
**Anita's decisions today (for the record):** dead ends and through stations turn alike, hints only "build a terminal"
(terminal = 30 tph), never "dead end"; single track two-way default, margin 240 s; margins: trams ~40/h, terminals 30/h,
junction moves 120 s, platforms unchanged; community first version = share + share link + open a copy (ON by default) +
country leaderboards; City circles only for whole loops, earned ones kept; Catalan added; usage: free reset, Opus more
widely but "not fully gung-ho".

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
- `call snapbuild.cmd` inside another .cmd leaves the shell in .tw-snap\trainworld\app: `cd /d` back after it.
- Server deploys and ssh are refused by the classifier unless Anita's latest message says yes to that exact action.
- Landing a tree fails while the main tree has other agents' edits in the same files (en.json often): park them as a
  patch (`git diff -- <files>`), `git checkout -- <files>`, land, `git apply --3way` the patch, then `git reset -- <files>`.
- A translation batch that reorders a key can drop the space after a colon (`"key":"..."`); grep for `":"` before committing.
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
- **v0.2.2: no publish until she has tested 8800** (Anita 2026-11-10 21:50: "the last few times around there were things
  that i caught that weren't good"). After the landing: rebuild 8800, tell her, wait for her playtest and fixes.
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
