# Orchestrator handoff, trainworld

Orchestrator: session 00d74e61-e552-47a6-ba40-ee320d38eba1, name not shown by ListAgents (message by session id), since 2026-11-16 20:41 UTC
Updated: 2026-11-16 22:21 UTC

A short state page, rewritten (not appended) as things change. The diary is notes/handoff-log.md; older detail is in
git history. Read trainworld/CLAUDE.md and docs/orchestrator.md first.

## State
- Live = v0.3.5 (maps bfe58c54, published 11-15). The next publish is v0.4.0 (Anita 20:23: big, with Central America and Detailed mode scheduling).
- 8800 = origin = master = b272f8ec (23:03): adds phonemem stage 1 (desktop outlines), T-3494 memory readout (text acknowledged 23:05), T-3464 freight code (no tiles yet). Smoke 28/29 then the i18n fault fixed (stale memoryTip translations dropped).
- Before v0.4.0 ships: the stage 1 server (T-3395) deployed and knowing save level 19; translation batch (points keys cities.pointsPast, cities.pointsPastAll, topBar.pointsPast, cities.sortPoints, leaderboard.valueTip, cities.pointsRule, leaderboard.pointsRule, plus T-3395 stage 2 and T-3479 keys, and T-3494's settingsTab.memoryTip/memoryTipCap/memoryTipUnknown/memoryValueCap/memoryValueUnknown and savesPanel.memoryWarn/haveCap/haveAbout; docs/localization.md).
- Uncommitted on purpose in the main tree: T-3399's country plan edits (pipeline/world_plan.py/.json, realms.json, zone-names.json, docs/data-register.md incl. hkborder's Hong Kong/Macau commuting row, notes/T-3399.md): commit when T-3399 starts. Not ours: TODO.md (hers), notes/T-3397.md, jordan-save-investigation-2026-11-15.md, tools/ops/archive_bad_saves.py, changelog/roadmap.json (board-written).
- Board inbox watch: Monitor on `venv/Scripts/python.exe -u trainworld/board/watch.py` (30 min, re-arm). Up next queue: `tasks.py upnext list|add|remove` (board-live.json "upnext").
- shonei-server: T-3395 stage 1 (37217ea, 4623fde, 094b830) and stage 3 (85e2945) committed, not pushed or deployed. savecheck PASSED stage 1 (risks, not fails: one user's burst of big saves queues everyone ~12 s; 128 MB unpack cap may be close, needs a read-only look at the box's index, ask her). Stage 3 not yet checked. The server must know save level 19 before deploy.
- Resource check ran 20:43 (she leaves all as is). Next one 11-17.
- Sessions started by next_orchestrator.py before 22:10 counted as child sessions (no transcript, not messageable); the launcher now clears inherited CLAUDE* vars and sets CLAUDE_CODE_FORCE_SESSION_PERSISTENCE=1. This session (00d74e61) is still unreachable by SendMessage.
- Landing gotchas: agent-tree land refuses root files (serve.py): patch them over by hand; a contributor's co-author needs a hand commit; changelog items in a tree can collide by id with main's: drop the tree's next.json and re-add with changelog.py in main; tree names are lowercase; an agent that edits TASKS/DECISIONS/ARCHIVE in its tree: revert those there and redo them in main before landing; never `remove --force` a tree whose land stopped on a conflict (its work is the WIP commit; recover by cherry-pick into a fresh tree).

## Running agents (mine, 00d74e61)
- phone2: T-3497 landed (phones -130 MB, desktop -50-100 MB, GPU buffers; 3x phones lose MSAA again per T-955). Paused at no easy gains; leftovers in notes/T-3497.md.
- wasmmem T-2496 stage 1 landed bb46fb98 (wasmcheck PASS: figures identical to the bit, -30 to -62 MB WASM per solving worker). Master smoked 29/29 with phonemem stage 2 + wasmmem + pool.
- T-3496 wording committed, her version 23:45: label stays "Use more cores", hover "Works out riders faster after a load or an edit by using more memory, up to 5 demand workers at about 400 MB each. Reload to apply." (changelog c069).
- T-3373 stage A landed b3d2ea95; T-1366 (pool grows back, closing retry fix) landed by pool. Stage B (T-1560) is a real per-worker cut, brief it fresh if memory work continues.
- freightbuild (Sonnet, main tree, 6179): T-3464 freight tiles for the 118 regions (her yes 23:05), Germany first; no R2 upload, hover string and changelog item come with the publish.
- phonemem stage 2 landed (corridor table on the page 32/16-bit, desktop shade worker per cut; corrcheck PASS: figures identical within 6e-8, cache refused across format). Master not yet smoked with it; rebuild 8800 after wasmmem lands.
- scancheck: T-3395 stage 3 FAILED narrowly (no good save ever flagged; but an alert is lost after a restart mid-scan, and a missing pinned file is not re-pinned). Server does not know save level 19 yet (patch in scratchpad/scancheck/level19.diff, plus twSaveCheckV 2). Production-size pass: 4 nightly runs of ~25 min, 573 MB peak. scanfix committed 3e8e501 (alerted set in state.json, missing pin re-pinned, level 19 + twSaveCheckV 2, pins re-read in full each pass); scancheck2 (Opus) rechecking; on PASS ask her for the deploy of 37217ea..3e8e501 (daytime; v2 re-checks every rev over a few days, ~1 GB/day).
- Both memory agents briefed with her 22:12 words: approximations allowed when the accuracy loss is measured and small (DECISIONS 2026-11-16).
- prsurvey (a02dbdc8's) surveyed PR 16 "Branch lines" into notes/pr-survey-2026-11-16.md: reply not drafted yet.
- Old trees to clear when sure: shenzhen (8 h), tt2 (4 days), ttstyle (superseded by ttstyle2), safari-save-035 (release branch: keep unless she says).

## Waiting on Anita
- Server deploy of T-3395 stages 1+3 (ask once stage 3 is checked, daytime).
- T-3399 Central America + Venezuela, T-3455 overnight scan: she will say when; don't list them in reports.

## Her standing asks this week
- Memory (23:45): pause once there are no easy gains left and move on; no core-balance shifting.
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
