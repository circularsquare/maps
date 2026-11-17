# Orchestrator handoff, trainworld

Orchestrator: session f0b2a436-f42b-45c6-999f-26d970f36c1a, name maps-45, since 2026-11-17 18:15 UTC
Updated: 2026-11-17 20:05 UTC

A short state page, rewritten (not appended) as things change. The diary is notes/handoff-log.md; older detail is in
git history. Read trainworld/CLAUDE.md and docs/orchestrator.md first. Her local time is UTC-5.

## State
- Live = v0.3.5 (maps bfe58c54, published 11-15). Next publish v0.4.0: no deadline; she picks when and tracks the landing cut-off herself: land things as they pass.
- master = pushed = f8978fd1; 8800 = 378869df (smoke PASSED 29/29 on 2f5e0ca8+). Since 8800: T-3388 London pacing (5eabd6d0), tram wire revert, board CSS fix. Server level 21 (95356ed) needed before publish: master writes level 21 since weekends.
- R2 version 2026-11-10d purged on her word 11-17.
- In master since v0.3.5 (for the v0.4.0 notes): everything maps-b8 listed (Central America + Venezuela, Lisbon/Glasgow/Porto, Istanbul/Colombia redraws, ~350 MB less memory, faster loads, bubble folds, purpose rows, buses obey one-way streets (Renkon), tram snap, terrain on by default, trainsfan, ferries 45 km/h) plus today: track speed limit per piece (T-3405), station delete faster (T-2768), track panel rebuild incl. levels and tram (T-3408), bus boarding 4 min (T-3522), help list + F1 (T-3527), shore 12 m margin (T-3524), snap to streets icon (T-3536), double track width / tram wire (T-3533/T-3534), settle speed (T-3512), tram crossing ghost (T-3535), complex pick (T-3539), reclaimed-land country check (T-3529, pipeline).
- Server: shonei-server 7a23cbc (playtime cap), 96e4f12 (save level 20, LPS1) and 95356ed (level 21, LWE1) committed, not pushed or deployed; all needed before v0.4.0 publishes (master already writes level 20). TestTwSaveCheckFollowsTheGame goes green when weekend lands.
- Before v0.4.0 ships: translation batch (tram-aware station upgrade hovers that say "trains" at tram stops (T-3574 notes), T-3543/T-3588 timetables strings, T-3582 loadText.graphicsGone, T-3570 "Show stations", busOneWay, T-3510's strings, trainsfan, T-3527 help keys, schedule window + weekend keys, updates.v040*; drop settingsTab.controls and help.title); `changelog.py line` + `publish --version v0.4.0`; freight tiles to R2 + hover + item (T-3464); real graphs v2 via deploy.py; smoke8800 light and dark; server deploy; deploy.py; website push; open-source sync + push; Renkon PR 1 / issue 8 replies (text in handoff-log 2026-11-17 maps-b8). Each needs her yes.
- Uncommitted on purpose: TODO.md (hers), notes/T-3397.md, jordan-save-investigation-2026-11-15.md, tools/ops/archive_bad_saves.py, changelog/roadmap.json; notes/T-3455/kit (commit with the night run).
- Board inbox watch: Monitor on `venv/Scripts/python.exe -u trainworld/board/watch.py` (30 min, re-arm).
- Landing: `node trainworld/checks/agent-tree.mjs finish <name> "<sentence>" --task T-x --outcome "..." [--notes ...]` does changelog move, land, wasm, remove, archive (T-3554). Move a tree's DECISIONS.md lines to main by hand first (top-of-file conflicts). A squash of a committed branch: `git merge --squash tw/<name>`, commit, then `agent-tree.mjs remove <name> --force`. Contributor co-author landing by hand (trailer lines). Kill a leftover agent Vite by its port's PID only.

## Night run (start this evening, about 21:00 local = 02:00 UTC, once landings stop)
- T-3511 enclave shifts + T-3455 coastal rows (28 rows incl. Mokpo), plan notes/T-3455.md, ~8.2 h, optional Philippines batch 3. Rehearsed and dry-run clean.
- Start: `powershell -File C:\Users\anita\projects\maps\trainworld\tools\country\start_hidden.ps1 C:\Users\anita\projects\maps\trainworld\notes\T-3455\kit\night1117.cmd`. Log artifacts/trainworld/night1117.log; PID in kit\night1117.state; stop: create trainworld\data\work\world_run\STOP, then kill only that PID. Claims 10 cores (waits for 6 free). Hold app/ landings during its gates. Disk 45 GB free, needs ~28.
- Later nights: nic (Nice) and pyo-kp (Sinuiju) now safe after T-3529; T-3530 Tokyo Haneda; T-3532 drive-grid water fix (she said yes, low priority).

## Agents (all maps-45's; maps-58 closed out 11-17. Move a tree's DECISIONS lines first: scratchpad move_decisions.py <tree>)
- onewaystops (.tw-trees/onewaystops): at landing its changelog item fixes a feature new in this update (T-3348's one-way pair stops): reword that item instead (changelog.py reword), hide the new one.
- farebase: T-3578 done in tree (T-3581 archived); checker farebasecheck (Sonnet, port 6174) running. At landing: its changelog item repeats T-3567's unpublished fare-multiple item: reword that one to the end state, hide the new one.
- She wants v0.4.0 out TONIGHT (11-17) if possible ("would rather do tonight, but dont worry about it"). Waiting on: T-3565 (onewaystops, redoing ordering on her model: click order doesn't matter), T-3578 checker, T-3595 (extendfix, Opus, .tw-trees/extendfix, port 6179: line extension still breaks, P1). Translation pass 1 running for UI keys: l10n-a (ja, ko, zh-Hans), l10n-b (zh-TW, ru, tr), l10n-c (es, pt, ca), l10n-d (de, fr, id), scratch folders scratchpad\l10n-*; then pass 2 (back, Sonnet) and pass 3 (review, Opus) per docs/localization.md; changelog updates.v040* keys are a second batch after `publish`. Night run shifts after the publish if they clash.
- Before asking her for the server deploy: run shonei-server's TestTwSaveCheckFollowsTheGame against current master (T-3557 added an LWE1 flag, T-3588 header fields, after the server commits). Commit each translator's messages/ merges as it reports, before landing farebase (its fareTip changes).
- T-3595 repro (hers): Caracas save, extend past Avenida Arismendi, put a station on the extension, construct: line breaks. extendfix checks master vs the onewaystops tree first; may fold into T-3565.
- v0.4.0 highlights marked: c084 regions, c051 long-distance fix, c130 schedule panel. serve.py board commits now have an allow rule in .claude/settings.json.
- onewaystops: checker FAILED T-3565 on six leg stops clicked west to east (notes/T-3565-check.md); back with its agent (Opus, may call the advisor freely on her word); then a fresh checker.
- defaulthw (Sonnet, .tw-trees/defaulthw, port 6170): T-3588 timetables window follow-ups (her list in the task and agent messages: set-as-default switches a counts line to headway, defaults per kind train/tram/high speed, show the game's numbers not "Game default", "Default not set", weekend apply-to-all also sets new lines' weekend default with the popup saying so, hide the lone "Simple" selector, whole-minute weekend steps, smaller text in the defaults row, Marey button same size as schedule). Checker if sim/save changes.
- logorder (Sonnet, .tw-trees/logorder): T-3590 board changelog hidden items last, drag to reorder, publish follows her order. Restart serve.py after landing if it changed.
- Landed 11-17 under maps-45: fbdot, tramloops, realstations, ponytail (b), t3212fix, station dots, lineprofit, shoreswap, settingsmove, schedbutton, workflow fixes (T-3586), boldlog (T-3585/T-3587, plus a CSS fix for doubled gray text), weekendstep (T-3557), glcontext (T-3582), tramterm (T-3574/T-3575; crossover item hidden), londonedit (T-3388, merged with weekends by londonmerge). Changelog tidy: notes/changelog-tidy-2026-11-17.md.
- Landed 11-17 under maps-58: T-3554, T-3537/38, T-3555 (+ translations), T-3540, T-3564, T-3560, T-3521 (gabs co-author), T-3526, T-3561, T-3562, T-3556 stage 1 (checks/lib harness), T-3541, T-3264 stage 0 check. Note: t3537-tram-split.mjs kept tramsplit's version at the T-3541 landing, so it is no longer on the harness.

## Next
1. Land each agent as it passes (checker for sim/save/demand/money); smoke on master per batch; rebuild 8800; push.
2. 8800 rebuild after the next landings (T-3388 and the tram wire revert are waiting). 3. The night run this evening.
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
