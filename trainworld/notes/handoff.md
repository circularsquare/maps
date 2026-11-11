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
**Master = d4e4d318** (pushed). **8800 = d4e4d318**: both cargo suites in .tw-snap OK (465, 645), smoke against 8800
PASSED (13 browser checks). Script: maps-dc scratch master_check.cmd <hash> [nobuild].
**Landed today since the takeover (archived):** maps-15's Exact edits round 2 (41202320; being dropped by T-3109), one-way
single track T-3094 (97564abe), rebuild fix + rebuild as tram T-3087/3088 (90e97d88), Line 1a T-3099/3081 (8846f377),
single track in Build T-3093 (40af0202). maps-dc's: T-3104/3105 account mails name the game + quota message (shonei-server
e1ac732, 7b68d01, deployed, pushed by Anita; client in 1719028f); T-3106 street tram at ground level over water and T-3107
turning alike at every station, 15/h or 30/h with a terminal, hints name only a terminal, kind 3 "Turning" (1719028f);
T-3108 blueprint Regular/High speed/Tram switch (b42a4dbc, `set_edges_hsr` -> `set_edges_kind`); T-3092 + T-3103 real
flyovers and flat badge only where moves cross (f7ed49e9; layout differs from the note: leg crosses ahead of J1, ~1.1 km).
**Anita on single track:** keep two-way default; too strong, so T-3112 margin 60 -> 240 s (10-20 min each way).
**v0.2.2 = 05c857b2** (translations all 11 languages complete, l10nbatch done; update notes committed, Anita still to
reword). 8800 = 05c857b2; smoke8800 light 13/13 and dark 13/13 PASSED, deploy dry run OK (19:40). READY TO PUBLISH.
DO NOT snapbuild until v0.2.2 is published: deploy.py uploads .tw-snap's dist, so 8800 must stay at 05c857b2.
Master = 4358fa86 (pushed): + T-3115 share link and open a copy (dev smoke PASSED 28; server 996cd4e committed, not
deployed; 8 English-only strings need an l10n batch before community goes public). Server commits waiting to deploy,
in order: 4cd43f0 + fc98bb8 country boards, 829788b preview, 996cd4e copies. Deploy server before any game with T-3115. After her word: `deploy.py --version 2026-11-10e --smoked`, website add/commit/push, `--verify --page`.
Set V0_2_2_AT in updates.ts at publish (a reword means another small l10n batch first).
**Release v0.2.2 in prep (earlier) (Anita out ~19:00-21:00, "feel free to keepgoing"):** master ee3a28ec has T-3117/3102 flyover
drawing, T-3118 loop gestures round 2, T-3113 Catalan. 8800 = ee3a28ec, cargo 465/645 and smoke against 8800 PASSED (19:10). updates.ts
v0.2.2 entry + 13 `updates.v022*` strings drafted (uncommitted; draft shown to Anita, waiting on her reword). l10nbatch
(Sonnet coordinator, a8ae0af9fd684d0d3) translating every missing/stale key in 11 languages. Then: commit, release build,
smoke8800 light+dark, deploy.py --dry-run --version v0.2.2; publish + website push wait for her word. Country boards'
server (4cd43f0) deploys with the community preview, her word.
**Community:** T-3114 done (32a7fcf8; server 829788b preview mode, not pushed/deployed). To let Anita try it: with her
word, deploy shonei-server (country boards + preview), append to /opt/shonei/shonei.env `SHONEI_COMMUNITY=preview` and
`SHONEI_COMMUNITY_TESTERS=pw:<her username>` (ask her username, or her account id from the console line in
notes/T-2887.md), restart, check the journal for "PREVIEW: 1 tester accounts"; she opens
localhost:8800/trainworld/?acct=live&community (needs 60 min playtime on the account). Full steps in notes/T-2887.md.
- sharecopy (Opus, aee2897be32c47b21) DONE: T-3115 in 4358fa86 (see above).
- chevron (Sonnet, aaeff40a11336de02, main tree, Vite 6188): T-3111 one-way stop chevron in the hover and while editing.
**Release (earlier note):** Anita 18:20: "flyovers + right-click loops is a good place to release" (no rush about tonight). So the next
release = T-3117 flyover drawing fixes + T-3109/3110 landed (+ T-3112, T-3108 already in). Then: Catalan batch if it is
in, a translation batch for stale keys, updates.ts entry (strings to her), smoke8800 light and dark, deploy.py, her word
for the website push.
- flyover (Opus, af64c7f0ed83c00b5) RESUMED in the MAIN tree (its tree is removed), Vite 6181: T-3117 flyover drawing
  (smooth divergence at J3, no flat mark there, line colour on the return leg = T-3102 for flyovers).
**Landed since:** T-3112 single track 240 s (93d77674); T-3109/T-3110/T-3063 right-click loops, Exact edits gone
(d762b813); T-3116 country boards game part (d4e4d318; server 4cd43f0 + fc98bb8 not deployed: deploy with
community preview, her word).
**Running:**
- loopclick RESUMED in the MAIN tree (Vite 6183): T-3118 loop gestures round 2 (right click any point of a closed loop
  opens it, ferries to right click, a loop button on touch). Anita approved T-3109's strings ("text good") and the
  country picker default (2 countries). T-3119 margin review filed (P3, to her before any change). Next free id T-3120.
- (landed) loopclick (Opus, a2a9a1fb062be2b58, tree `.tw-trees/loopclick` from 3a0a8ee3, Vite 6183): T-3109 drop Exact edits,
  right click loops back, left click add/remove, bus partial loops (T-3063 folded in), + T-3110 strip while editing.
- catalan (adcc19e6d8b045139) DONE, uncommitted: T-3113 Catalan translated (7 units x 3 passes), released in OFFERED;
  files intl.ts, glossary.json, messages/ca.json, sources/ca.json, checks/l10n.mjs, notes/T-3113.md. ca.json holds
  `community.errPreview`, which only communitycheck's uncommitted en.json has: commit Catalan WITH T-3114 (or after),
  never before. Points for a Catalan speaker in notes/T-3113.md "done". Run a stale batch before the release.
- communitycheck (Sonnet, a91f2b0e73e677d8e, main tree + shonei-server, Vite 6185): T-3114 re-test community stage 1
  (T-2887, dormant behind SHONEI_COMMUNITY=1) on today's master, fix, launch steps + all strings into notes/T-2887.md.
  Anita 2026-11-10: community is the next major push after the track work; "first version can be pretty minimal, but
  would be nice to get map sharing up at least". Added: preview mode (SHONEI_COMMUNITY=preview + tester accounts, game
  shows Community only with ?community) so Anita can share her own maps from 8800 on the real server; deploy that with
  her word. Her answers: share link and open a copy (ON by default) in the first version = T-3115 (after T-3114, same
  files); country leaderboards same push = T-3116.
**Queued:** T-3111 one-way stop chevron (after loopclick lands; same hover and editing view). T-3100 likely moot after T-3109.
`.tw-trees/hotfix` (tw/release-021, live v0.2.1's source) kept until the next release in case live needs a hotfix.
Next free id **T-3117**.
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
