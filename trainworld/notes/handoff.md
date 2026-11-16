# Orchestrator handoff, trainworld

Orchestrator: session 00d74e61-e552-47a6-ba40-ee320d38eba1, name not shown by ListAgents (message by session id), since 2026-11-16 20:41 UTC
Updated: 2026-11-16 22:21 UTC

A short state page, rewritten (not appended) as things change. The diary is notes/handoff-log.md; older detail is in
git history. Read trainworld/CLAUDE.md and docs/orchestrator.md first.

## State
- Live = v0.3.5 (maps bfe58c54, published 11-15). The next publish is v0.4.0 (Anita 20:23: big, with Central America and Detailed mode scheduling).
- 8800 = origin = master = 04d59c74 (22:20; smoke 29/29 on master): timetable restyle and hover, points wording, touch rule, ferries 45 km/h (save level 19), crowding dot (30% skew), new game via menu + terrain retry, T-3395 stage 2 saves, gate fixes, Up next board column, globe connector fix (T-3481).
- master ba9d6e83 ahead of 8800/origin: phonemem stage 1 (outlines), memtext readout (T-3494, strings await her ack). Rebuild 8800 with the next batch.
- Before v0.4.0 ships: the stage 1 server (T-3395) deployed and knowing save level 19; translation batch (points keys cities.pointsPast, cities.pointsPastAll, topBar.pointsPast, cities.sortPoints, leaderboard.valueTip, cities.pointsRule, leaderboard.pointsRule, plus T-3395 stage 2 and T-3479 keys, and T-3494's settingsTab.memoryTip/memoryTipCap/memoryTipUnknown/memoryValueCap/memoryValueUnknown and savesPanel.memoryWarn/haveCap/haveAbout; docs/localization.md).
- Uncommitted on purpose in the main tree: T-3399's country plan edits (pipeline/world_plan.py/.json, realms.json, zone-names.json, docs/data-register.md incl. hkborder's Hong Kong/Macau commuting row, notes/T-3399.md): commit when T-3399 starts. Not ours: TODO.md (hers), notes/T-3397.md, jordan-save-investigation-2026-11-15.md, tools/ops/archive_bad_saves.py, changelog/roadmap.json (board-written).
- Board inbox watch: Monitor on `venv/Scripts/python.exe -u trainworld/board/watch.py` (30 min, re-arm). Up next queue: `tasks.py upnext list|add|remove` (board-live.json "upnext").
- shonei-server: T-3395 stage 1 (37217ea, 4623fde, 094b830) and stage 3 (85e2945) committed, not pushed or deployed. savecheck PASSED stage 1 (risks, not fails: one user's burst of big saves queues everyone ~12 s; 128 MB unpack cap may be close, needs a read-only look at the box's index, ask her). Stage 3 not yet checked. The server must know save level 19 before deploy.
- Resource check ran 20:43 (she leaves all as is). Next one 11-17.
- Sessions started by next_orchestrator.py before 22:10 counted as child sessions (no transcript, not messageable); the launcher now clears inherited CLAUDE* vars and sets CLAUDE_CODE_FORCE_SESSION_PERSISTENCE=1. This session (00d74e61) is still unreachable by SendMessage.
- Landing gotchas: agent-tree land refuses root files (serve.py): patch them over by hand; a contributor's co-author needs a hand commit; changelog items in a tree can collide by id with main's: drop the tree's next.json and re-add with changelog.py in main; tree names are lowercase.

## Running agents (mine, 00d74e61)
- memA a80653544bfb18cb5 (Opus, .tw-trees/mema, 6175): T-3373 stage A, fewer solving workers capped by memory.
- phonemem aba6e34aa7a8ac631 (Opus, .tw-trees/phonemem, 6176): stage 1 landed 3d480e46 (desktop outlines compact: start menu 1,090 -> 769 MB, one-city game 1,491 -> 1,295; phone unchanged ~595 MB, notes/T-3160.md). Now stage 2: T-3493 solve-time buffer swell (232 MB), realm-shade worker, clock worker outline copy.
- scancheck (Opus, no tree): checker for T-3395 stage 3 (shonei-server 85e2945), incl. whether the server accepts save level 19.
- freight (Sonnet, .tw-trees/freight, 6177): T-3464 freight and abandoned rail in All; one test area, full tile rebuild goes in an overnight batch.
- Both memory agents briefed with her 22:12 words: approximations allowed when the accuracy loss is measured and small (DECISIONS 2026-11-16).
- prsurvey (a02dbdc8's) surveyed PR 16 "Branch lines" into notes/pr-survey-2026-11-16.md: reply not drafted yet.
- Old trees to clear when sure: shenzhen (8 h), tt2 (4 days), ttstyle (superseded by ttstyle2), safari-save-035 (release branch: keep unless she says).

## Waiting on Anita
- Server deploy of T-3395 stages 1+3 (ask once stage 3 is checked, daytime).
- T-3399 Central America + Venezuela, T-3455 overnight scan: she will say when; don't list them in reports.

## Her standing asks this week
- Small isolated fixes, open-task count down, merge duplicates; bug investigations give up if hard.
- No asking about changelog lines: she reviews them all at once on the board. No changelog items about the changelog itself.
- Corrupted saves (T-3395) are priority. Optimization today: T-3472 then T-3473 (PR 12 CPU speedups, measured) then T-3373 A then B; C only if it helps noticeably.
- Quad paused (waits on T-3444). Pack rebuilds batch into overnight runs.
- No windows on her screen: background processes hidden.
- Every publish, deploy and server push needs her yes to that exact action; none overnight.

## Next
- Advisor trial (Anita 21:40): start the next orchestrator with `next_orchestrator.py trainworld --advisor fable`; rules in docs/orchestrator.md Agents. Log each call in handoff-log.
- Open-source issues triaged 21:10: 4 -> T-3484, 6+11 -> T-3485, 8 folded into T-3480 (roundabout), 10 declined (balloon loops; reply not drafted), 11 -> T-3488 (merge Arnavutkoy into Istanbul + two districts to Kocaeli, next Turkey rebuild), 9 folded into T-3069, 15 = PR 16 (prsurvey), 14 thanks only. No issue replies drafted yet.
- When cores free up (her 20:34): world build tooling ahead of coming world builds. Start with T-2298 (gate from a frozen app copy) and T-2399 (gate browser check fails silently); then T-1080, T-2627/T-2777, T-2070, T-930, T-2095, T-1992, T-2326.
- PR 16 reply after prsurvey reports. Replies posted: #7, #12, #13 closed; #1 stays open until T-3480 reaches the export.
- T-3480 (PR 1 bus one-way as a setting plus PR 12's debug check, P3) when cores free, with T-2298/T-2399.
- T-3474 partly built lines (low priority).
