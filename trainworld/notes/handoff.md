# Orchestrator handoff, trainworld

Orchestrator: session d516cc87-bbe1-4627-9f97-62b5c973620b, name maps-58, since 2026-11-17 16:41 UTC
Previous: session eda3ac12-4519-4720-a0c5-be8bdcb2ed0b, name maps-60, handed over 2026-11-17 16:41 UTC, finishing in-flight work
Updated: 2026-11-17 16:50 UTC (maps-58 took over and owns landings; maps-60's agents report to maps-60, which relays them verbatim while its tab is open; resources checked 16:44)

A short state page, rewritten (not appended) as things change. The diary is notes/handoff-log.md; older detail is in
git history. Read trainworld/CLAUDE.md and docs/orchestrator.md first. Her local time is UTC-5.

## State
- Live = v0.3.5 (maps bfe58c54, published 11-15). Next publish v0.4.0: no deadline; she picks when and tracks the landing cut-off herself: land things as they pass.
- master ca8a5601 (b322e367 T-3554 finish command, ca8a5601 tramsplit), pushed only to c350115a; 8800 = a0e4157b. Smoke PASSED on 66379382; tramsplit's own smoke passed.
- R2 version 2026-11-10d purged on her word 11-17.
- In master since v0.3.5 (for the v0.4.0 notes): everything maps-b8 listed (Central America + Venezuela, Lisbon/Glasgow/Porto, Istanbul/Colombia redraws, ~350 MB less memory, faster loads, bubble folds, purpose rows, buses obey one-way streets (Renkon), tram snap, terrain on by default, trainsfan, ferries 45 km/h) plus today: track speed limit per piece (T-3405), station delete faster (T-2768), track panel rebuild incl. levels and tram (T-3408), bus boarding 4 min (T-3522), help list + F1 (T-3527), shore 12 m margin (T-3524), snap to streets icon (T-3536), double track width / tram wire (T-3533/T-3534), settle speed (T-3512), tram crossing ghost (T-3535), complex pick (T-3539), reclaimed-land country check (T-3529, pipeline).
- Server: shonei-server 7a23cbc (playtime cap), 96e4f12 (save level 20, LPS1) and 95356ed (level 21, LWE1) committed, not pushed or deployed; all needed before v0.4.0 publishes (master already writes level 20). TestTwSaveCheckFollowsTheGame goes green when weekend lands.
- Before v0.4.0 ships: translation batch (busOneWay, T-3510's strings, trainsfan, T-3527 help keys, schedule window + weekend keys, updates.v040*; drop settingsTab.controls and help.title); `changelog.py line` + `publish --version v0.4.0`; freight tiles to R2 + hover + item (T-3464); real graphs v2 via deploy.py; smoke8800 light and dark; server deploy; deploy.py; website push; open-source sync + push; Renkon PR 1 / issue 8 replies (text in handoff-log 2026-11-17 maps-b8). Each needs her yes.
- Uncommitted on purpose: TODO.md (hers), notes/T-3397.md, jordan-save-investigation-2026-11-15.md, tools/ops/archive_bad_saves.py, changelog/roadmap.json; notes/T-3455/kit (commit with the night run).
- Board inbox watch: Monitor on `venv/Scripts/python.exe -u trainworld/board/watch.py` (30 min, re-arm).
- Landing: `node trainworld/checks/agent-tree.mjs finish <name> "<sentence>" --task T-x --outcome "..." [--notes ...]` does changelog move, land, wasm, remove, archive (T-3554). Move a tree's DECISIONS.md lines to main by hand first (top-of-file conflicts). A squash of a committed branch: `git merge --squash tw/<name>`, commit, then `agent-tree.mjs remove <name> --force`. Contributor co-author landing by hand (trailer lines). Kill a leftover agent Vite by its port's PID only.

## Night run (start this evening, about 21:00 local = 02:00 UTC, once landings stop)
- T-3511 enclave shifts + T-3455 coastal rows (28 rows incl. Mokpo), plan notes/T-3455.md, ~8.2 h, optional Philippines batch 3. Rehearsed and dry-run clean.
- Start: `powershell -File C:\Users\anita\projects\maps\trainworld\tools\country\start_hidden.ps1 C:\Users\anita\projects\maps\trainworld\notes\T-3455\kit\night1117.cmd`. Log artifacts/trainworld/night1117.log; PID in kit\night1117.state; stop: create trainworld\data\work\world_run\STOP, then kill only that PID. Claims 10 cores (waits for 6 free). Hold app/ landings during its gates. Disk 45 GB free, needs ~28.
- Later nights: nic (Nice) and pyo-kp (Sinuiju) now safe after T-3529; T-3530 Tokyo Haneda; T-3532 drive-grid water fix (she said yes, low priority).

## Agents (maps-58 owns all landings. The first five are maps-60's subagents, handed over un-landed: their reports come to maps-60, which relays them to maps-58 while its tab is open. If its tab is gone, check each directly: its tree's git status, notes, the ledger)
- drawapart (Opus, maps-58's, .tw-trees/drawapart, port 6131): T-3264 stage 0 (standing overlap check) and stage 1 (opposite directions drawn apart from z14) built to a mock; screenshots to review/ for her yes before polishing. May file T-3546-3549 for stages 2-6.
- shoreline (Opus, maps-58's, .tw-trees/shoreline, port 6133): T-3553, measuring shoreline accuracy options against T-3524's landed 12 m margin (her second thoughts 11-17); read-only on data/packs; results go to her to pick. Revert path if she wants the margin out: `git revert 66e46fed`, hide changelog c092, reopen T-3524.
- londonedit (Opus, maps-58's, .tw-trees/londonedit, port 6132): T-3388 London edits under 4 s, settles unchanged; checker before landing (demand). May file T-3550-3552.
- weekend2 (Opus, a8b211c279a6dfa8a, .tw-trees/weekend, port 6127): fixing T-3521's checker FAIL (headway lines ignore the weekend percent; save.rs test-string mojibake). Then: a fresh Sonnet checker on headway cases + default parity (checker-weekend's report: default parity NYC 12,679,834 / 12,384,581 / 10,797,533 weekday/Sat/Sun; repro scratchpad\checker-weekend\s4.mjs), then commit by hand with trailers `Co-authored-by: gabs <moraes.madm@gmail.com>` and Claude's; its 5 DECISIONS lines and changelog c094, c095 are already in master. Strings were shown to her.
- singlecap (Opus, adda47c60a86f18ba, .tw-trees/singlecap, port 6113): T-3526 fix 1 (her pick): a line's own trains don't queue against each other on single track; cap at trains that fit plus one. Sim change: fresh checker before landing.
- tramsplit (Opus, a089adcfacfecebf2, .tw-trees/tramsplit, port 6122): done: T-3537/T-3538 landed ca8a5601 (not on 8800 yet); T-3541 waits on her save (asks.json).
- tramstation (Opus, aa7ec7853a737250f, .tw-trees/tramstation, port 6126): T-3540, widened by her: every single-track station (rail and tram) draws one smaller platform to one side (outside by default), any passing track drawn on the far side; tram platforms 30% narrower.
- tiers (Opus, a8d20c85f2b79e30a, read-only, port 6128): done: T-3542 designed and answered by her; waits for T-3521 and T-3346 stage 2 (in the stale tt2 tree: look at tt2 before starting).

## Next
1. Land each agent as it passes (checker for sim/save/demand/money); smoke on master per batch; rebuild 8800; push.
2. T-3543 (one bigger schedule button, default headways replacing copy/paste) after weekend lands; same files.
3. The night run this evening.
4. When she says release: the v0.4.0 steps under State.
5. Leftover .tw-trees folders (c1, ferry45, fixes, freight, hotfix031-target, hover, touchrule, trackrebuild (empty)) and stale trees (safari-save-035, shenzhen, tt2, ttstyle, trams): look before removing; never delete by hand.

## Waiting on Anita
- asks.json holds every open ask (her rule 11-17: final messages carry only new asks; "catch me up" reads it back).
- PR 16 and PR 22 deferred by her; PRs 18 and 19 wait for Cinnamon's timings.

## Her standing asks this week
- Memory: done for now; non-memory performance only; no core-balance shifting.
- Small isolated fixes, open-task count down, merge duplicates; bug investigations give up if hard.
- No asking about changelog lines: she reviews them on the board (she may ask for a specific rewording, as c086 "about 350 MB less").
- Overnight: keep rebuilding while queued builds remain. No windows on her screen.
- Every publish, deploy and server push needs her yes to that exact action; none overnight.
- Advisor trial: start successors with `next_orchestrator.py trainworld --advisor fable`; log each call in handoff-log.
