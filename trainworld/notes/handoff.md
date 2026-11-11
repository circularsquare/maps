# Orchestrator handoff, trainworld

Orchestrator: session 8be9971b-5363-40be-bba3-d8c1eaa4ca85, name maps-04 [6bfb1b], since 2026-11-10 21:40
Updated: 2026-11-10 22:50 (maps-dc finished winding down 22:45)

## Now

**LIVE = v0.2.1 = R2 2026-11-10d** (from f5a2b979, branch tw/release-021, worktree `.tw-trees/hotfix`, kept until the next
release in case live needs a hotfix). Due: purge R2 prefixes 2026-11-07c and 11-08 from 2026-11-11; 11-09, 11-09b, 11-09c a
few days after; keep 11-10, 10b-10d.
**Master = a5d6a81a, pushed.** **8800 = 8ae4e2ae** (maps-dc's last landing: 7db3b454 main batch T-3123/3124/3125/3127/
3128/3130/3132/3133, 7b260b67 flyover2 T-3120/3121, 8ae4e2ae stoporder T-3122; smoke on 8800 13/13). a5d6a81a adds only
the changelog rewording, so no rebuild for it.
**v0.2.2 is HELD until single track is fully fixed** (Anita 22:20, T-3134..T-3137; and 21:50: no publish until she has
tested 8800). Then: translation batch for every new/stale key (13 stale per language at 22:45; last batch: agent l10nbatch,
3 groups x 3 passes, ~15 min), snapbuild, smoke8800.cmd light+dark, `deploy.py --dry-run --version <date>`, set
`V0_2_2_AT` in updates.ts, publish and website push on her word (notes/T-924.md "Commands"). Changelog reworded per Anita
22:00 (a5d6a81a); two new lines she has not seen yet: `v022LoopStops` "a line on a loop stops at a station each time it
passes", `v022TargetSpeed` "type in a target speed for new track, down to 20 km/h". Her "auto-ramp" wording vs the switch's
tag "Ramp over track you cross": asked her. Anita: "No silent updates": every publish gets a version + entry.
**Server (shonei-server 80e3692, pushed, deployed 22:36 on her word):** community round 2 (Biggest/Most liked in city,
country or region, self-like, copy names, shares left today); country leaderboards live (T-3116); community in PREVIEW
(`SHONEI_COMMUNITY=preview`, `SHONEI_COMMUNITY_TESTERS=uad72a8ab66f84c44` = Anita, Google login; env backup
/opt/shonei/shonei.env.bak-20261111). She tests at localhost:8800/trainworld/?acct=live&community. Server deploys need her
word each time.

**Running (maps-04's subagents):**
- lanes (Opus, a5b1193bb6b1afcf8, tree `.tw-trees/lanes` from a5d6a81a, Vite 6192): T-3137 stage C1, double track as
  lanes inside one piece + per-node move table; routing, flat badge, junction moves, train guard read it; no visible
  change; parity check on NYC. Anita chose model (c) 22:35. Then C2 single joins + inferred way, C3 app snap/draw, C4
  switch/docs (notes/T-3137.md, notes/T-3135.md). Land with agent-tree.mjs.
- singleghost (Sonnet, a2bdab2e9d1eef273, main tree, Vite 6191): T-3134 ghost per-segment track count (sim preview answer
  + tools.ts, realSnap.ts, previewStrokes, flyover preview) and T-3136 single thinner / double thicker when zoomed out.
  23:00: sent back for a thicker zoomed-out floor for double (z11.5 still identical). Then commit + snapbuild.
- loops3 (Opus, a8acc7928a61a451b, main, Vite 6193): T-3140 no automatic loops (undo T-3122 rule 2), T-3141 lone branch
  = line end, T-3139 chevrons on the track tangent, 25% smaller and closer. May file T-3145/3146.
- onewayui (Sonnet, ac84a57324f0462ac, main, Vite 6194): T-3142 half dots on bus/ferry loops, T-3143 hover says one way and
  which way (new strings). May file T-3147.
- community3 (Sonnet, a74ecb7dd2921c010, main + shonei-server commits, Vite 6195): T-3144 Biggest/World empty bug, card
  space, share title (default save name, profanity screen, title bold + name grey), sandbox button alignment. Server
  deploy needs her word. May file T-3148. Next free id after these: **T-3149**.
- Anita 23:15: "auto-ramp" fine; changelog RightClick -> "more explicit loop controls" (done in en.json, uncommitted).
- singletrack (Opus, addd2bb7c40268aea) is done (design stages 1 and 1b; ~320k tokens); resume only for questions about
  its notes.

**Anita's calls tonight:** terminus branch on single track stays two-way with its flat badge; double track becomes lanes
in one piece, never deletable by halves; flyover ramps 4% like normal ramps; question every "double = one object" shortcut
from now on. Stop chevrons follow the track and one-way loops get half dots (stoporder; checks pass, not eyeballed on 8800).
**Filed, not started:** T-3126 HSR quad, T-3129 snap to road for normal track (design first), T-3131 bus out-and-back by
hand near-overlap stops, T-3138 [P3] left-hand running (fold into lanes if cheap). Next free id **T-3149**.
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
- **Keep 4 cores free** (Anita 22:32, 2026-11-10: "for the next few hours"): cores.py cap set 14 -> 12 at 22:32; briefs
  say `TW_CORES` small (Playwright self-claims under the cap); raise back to 14 once she says or after ~03:00.
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
