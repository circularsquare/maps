# Orchestrator handoff, trainworld

Orchestrator: session f0b2a436-f42b-45c6-999f-26d970f36c1a, name maps-45, since 2026-11-17 18:15 UTC
Updated: 2026-11-17 18:20 UTC

A short state page, rewritten (not appended) as things change. The diary is notes/handoff-log.md; older detail is in
git history. Read trainworld/CLAUDE.md and docs/orchestrator.md first. Her local time is UTC-5.

## State
- Live = v0.3.5 (maps bfe58c54, published 11-15). Next publish v0.4.0: no deadline; she picks when and tracks the landing cut-off herself: land things as they pass.
- master = 8800 = pushed = 2066814c (smoke PASSED 29/29 on 68322201; since then only ponytail b, T-3577 dots, t3212 check). Server level 21 (95356ed) needed before publish: master writes level 21 since weekends.
- R2 version 2026-11-10d purged on her word 11-17.
- In master since v0.3.5 (for the v0.4.0 notes): everything maps-b8 listed (Central America + Venezuela, Lisbon/Glasgow/Porto, Istanbul/Colombia redraws, ~350 MB less memory, faster loads, bubble folds, purpose rows, buses obey one-way streets (Renkon), tram snap, terrain on by default, trainsfan, ferries 45 km/h) plus today: track speed limit per piece (T-3405), station delete faster (T-2768), track panel rebuild incl. levels and tram (T-3408), bus boarding 4 min (T-3522), help list + F1 (T-3527), shore 12 m margin (T-3524), snap to streets icon (T-3536), double track width / tram wire (T-3533/T-3534), settle speed (T-3512), tram crossing ghost (T-3535), complex pick (T-3539), reclaimed-land country check (T-3529, pipeline).
- Server: shonei-server 7a23cbc (playtime cap), 96e4f12 (save level 20, LPS1) and 95356ed (level 21, LWE1) committed, not pushed or deployed; all needed before v0.4.0 publishes (master already writes level 20). TestTwSaveCheckFollowsTheGame goes green when weekend lands.
- Before v0.4.0 ships: translation batch (T-3570 "Show stations", busOneWay, T-3510's strings, trainsfan, T-3527 help keys, schedule window + weekend keys, updates.v040*; drop settingsTab.controls and help.title); `changelog.py line` + `publish --version v0.4.0`; freight tiles to R2 + hover + item (T-3464); real graphs v2 via deploy.py; smoke8800 light and dark; server deploy; deploy.py; website push; open-source sync + push; Renkon PR 1 / issue 8 replies (text in handoff-log 2026-11-17 maps-b8). Each needs her yes.
- Uncommitted on purpose: TODO.md (hers), notes/T-3397.md, jordan-save-investigation-2026-11-15.md, tools/ops/archive_bad_saves.py, changelog/roadmap.json; notes/T-3455/kit (commit with the night run).
- Board inbox watch: Monitor on `venv/Scripts/python.exe -u trainworld/board/watch.py` (30 min, re-arm).
- Landing: `node trainworld/checks/agent-tree.mjs finish <name> "<sentence>" --task T-x --outcome "..." [--notes ...]` does changelog move, land, wasm, remove, archive (T-3554). Move a tree's DECISIONS.md lines to main by hand first (top-of-file conflicts). A squash of a committed branch: `git merge --squash tw/<name>`, commit, then `agent-tree.mjs remove <name> --force`. Contributor co-author landing by hand (trailer lines). Kill a leftover agent Vite by its port's PID only.

## Night run (start this evening, about 21:00 local = 02:00 UTC, once landings stop)
- T-3511 enclave shifts + T-3455 coastal rows (28 rows incl. Mokpo), plan notes/T-3455.md, ~8.2 h, optional Philippines batch 3. Rehearsed and dry-run clean.
- Start: `powershell -File C:\Users\anita\projects\maps\trainworld\tools\country\start_hidden.ps1 C:\Users\anita\projects\maps\trainworld\notes\T-3455\kit\night1117.cmd`. Log artifacts/trainworld/night1117.log; PID in kit\night1117.state; stop: create trainworld\data\work\world_run\STOP, then kill only that PID. Claims 10 cores (waits for 6 free). Hold app/ landings during its gates. Disk 45 GB free, needs ~28.
- Later nights: nic (Nice) and pyo-kp (Sinuiju) now safe after T-3529; T-3530 Tokyo Haneda; T-3532 drive-grid water fix (she said yes, low priority).

## Agents (all maps-45's; maps-58 closed out 11-17. Move a tree's DECISIONS lines first: scratchpad move_decisions.py <tree>)
- onewaystops (Opus, maps-45's, .tw-trees/onewaystops, port 6153): T-3565 one-way pair stops; routing: checker before landing.
- weekendstep: T-3557 done in tree (changelog item in tree; DECISIONS line in main; follow-ups T-3583, T-3584 filed); checker weekendcheck (Sonnet, port 6166) running; land on pass, then archive T-3557.
- tramterm (Sonnet, maps-45's, .tw-trees/tramterm, port 6157): T-3574 tram stop terminal + passing tracks, T-3575 tram crossovers; code + Rust done, stopped waiting on its wasm build (log scratchpad\tramterm\wasm.log); resume it for full cargo, check t3574-tram-terminal.mjs, screenshot, notes; then checker.
- shoreswap: T-3553 done in tree (changelog c113 in main); checker shorecheck (Sonnet, maps-45's, port 6158) running; land on pass, then archive T-3553.
- farebase (Opus, .tw-trees/farebase, port 6163): T-3578 fare multiple scales the base (max over legs) + T-3581 undo keeps the multiple; demand/money: checker before landing.
- boldlog (Sonnet, .tw-trees/boldlog, port 6169): T-3585 asterisks bold in changelog items (board + game update notes), T-3587 board changelog view keeps scroll.
- wffix (Sonnet, main tree, no worktree): T-3586 workflow review fixes (rules, skills, tasks.py/snap8800 write board-live, changelog reword, finish treats archived as done, handover 500k). Commit its edits after.
- Open to her: freeze for 0.4 this evening (my rec, T-3565 in) or drop T-3565 and freeze sooner; T-3543 strings text-ack and whether the chart (Marey) button goes too.
- glcontext (Opus, .tw-trees/glcontext, port 6165): T-3582 WebGL2 error after a lost graphics context: clear restart screen, then the cause (give up if hard).
- workflowreview (Fable, on her ask 11-17): read-only review of today's workflow to notes/workflow-review-2026-11-17.md; relay its top items.
- londonedit: T-3388 done in tree (archived itself; changelog c115, T-3550 follow-up in main); checker londoncheck (Sonnet, port 6162) running; land on pass.
- settingsmove (Sonnet, .tw-trees/settingsmove, port 6161): T-3579 settings reorder; UI, no checker.
- Landed 11-17 under maps-45: T-3573 fbdot (e85e761c), T-3563 tramloops (25c3c642, checker passed), T-3570/T-3571 realstations (0697cb3d), T-3576 ponytail (b on her pick), T-3569 t3212fix (141fbf62, fixture lengthened), T-3577 station dots (5bf18338), T-3566/T-3567 lineprofit (d30ffc11, checker passed), T-3553 shoreswap (1eceb454, checker passed), T-3579 settingsmove (3a980535), T-3543 schedbutton (07448241). londonedit: checker passed; londonmerge (Opus) resolving its rebase onto T-3521's weekend code in .tw-trees/londonedit, then finish again.
- Landed 11-17 under maps-58: T-3554, T-3537/38, T-3555 (+ translations), T-3540, T-3564, T-3560, T-3521 (gabs co-author), T-3526, T-3561, T-3562, T-3556 stage 1 (checks/lib harness), T-3541, T-3264 stage 0 check. Note: t3537-tram-split.mjs kept tramsplit's version at the T-3541 landing, so it is no longer on the harness.

## Next
1. Land each agent as it passes (checker for sim/save/demand/money); smoke on master per batch; rebuild 8800; push.
2. 8800 rebuild after the next landings (lineprofit, T-3579 settings, checkers' passes). 3. The night run this evening.
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
