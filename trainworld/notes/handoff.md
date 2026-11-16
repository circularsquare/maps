# Orchestrator handoff, trainworld

Orchestrator: session 00d74e61-e552-47a6-ba40-ee320d38eba1, name not shown by ListAgents (message by session id), since 2026-11-16 20:41 UTC
Updated: 2026-11-16 22:11 UTC (a02dbdc8 finished: all its agents landed or reported; its Previous line removed)

A short state page, rewritten (not appended) as things change. The diary is notes/handoff-log.md; older detail is in
git history. Read trainworld/CLAUDE.md and docs/orchestrator.md first.

## State
- Live = v0.3.5 (maps bfe58c54, published 11-15). The next publish is v0.4.0 (Anita 20:23: big, with Central America and Detailed mode scheduling).
- 8800 = origin = 45b7808c. master = 441d4a25, ahead and unpushed: timetable restyle T-3469, points wording, touchrule T-1764, Community count dropped, board bars. Next: land the agents below as they pass, then one smoke on master, push, snap8800 (snap_wasm emoji crash fixed in 45b7808c).
- Translation batch needed before v0.4.0: points keys cities.pointsPast, cities.pointsPastAll, topBar.pointsPast, cities.sortPoints, leaderboard.valueTip, cities.pointsRule, leaderboard.pointsRule (docs/localization.md passes), plus any keys the agents below add.
- Uncommitted on purpose in the main tree: T-3399's country plan edits (pipeline/world_plan.py/.json, realms.json, zone-names.json, docs/data-register.md, notes/T-3399.md): commit when T-3399 starts. Not ours: TODO.md (hers), notes/T-3397.md, jordan-save-investigation-2026-11-15.md, tools/ops/archive_bad_saves.py. notes/T-3395.md is savesafe's (untracked, commit with T-3395).
- Board inbox watch: Monitor on `venv/Scripts/python.exe -u trainworld/board/watch.py` (30 min, re-arm).
- shonei-server: T-3395 stage 1 (37217ea, 4623fde, 094b830) and stage 3 (85e2945) committed, not pushed or deployed. Deploy stages 1+3 in daytime on her yes once savecheck passes stage 1 and stage 3 is checked.
- Resource check ran 20:43: Resend daily cap hit daily (ask a-2026-11-16-2), website repo 1.1 GB (over the 1 GB GitHub recommends), R2 storage 22.6 GB (only 2026-11-10d is purgeable, about 5 GB, not worth breaking old tabs), R2 Class B month projected $22 (last 24 h is low: the cache fix took), plays/day 51% of ptMaxPlaysDay, games store 80% in 20 days.
- Production box rebooted 19:45 onto kernel 7.0.0-43 (T-2912 done). jsonbin retired (memory note).
- Landing gotchas this session: agent-tree land refuses root files (serve.py): git diff the tree's serve.py to a patch (add a trailing newline), apply in main, checkout in the tree, then land; a contributor's co-author needs a hand commit (land adds only the Claude trailer). Changelog items need --area (track, demand, scheduling, fares, schematic, community, performance, other, bug; --fix for lighter).

## Running agents (mine, 00d74e61)
- lines2 (Opus, .tw-trees/lines, 6171): T-3481 with the player's save (artifacts/trainworld/t3481/player-bangkok-tokyo-newyork-day459.save); lead: joins across cities in a 3-city game.
- gatefix: T-2298 and T-2399 done and archived, waiting in .tw-trees/gatefix to land once Previous is gone (tooling only, no checker needed; its full smoke passed 29/29). Then T-2627/T-2777 (same world_run.py) after it lands; T-1080, T-2095, T-1992 when cores free.

## Running agents (a02dbdc8's, reporting to it)
- ttstyle2: landed by a02dbdc8 (T-3475), not yet smoked on master.
- ferry45: T-3435 landed a689218f; ferrycheck PASS (123 saves: non-ferry saves byte-identical, 5 ferry saves faster only, round trip identical, level-18 build opens a level-19 save read-only). Save level 19: the server must know it before this game ships.
- crowddot: landed by a02dbdc8 (T-3476), not yet smoked on master. Anita 21:03 too green: crowdskew landed 27377dd2 (CROWD_SKEW = 0.3 in game/today.ts; tunable if she asks).
- savesafe2: T-3395 stage 2 landed 7cd2f5ac by a02dbdc8 (its changelog item re-added in main as c064) after re-check PASS (317 saves intact on both checks and both servers). Known gap until the stage 1 server is deployed: a save whose network is cut inside a whole gzip can still go up from the start menu to the old server (rare: only if the game writes an unreadable network). Ship the game with stage 2 only after the server deploy. Not yet smoked on master.
- savecheck: T-3395 stage 1 PASS (220 good saves all intact incl. fresh exports and a full-feature sandbox; ~44,000 damaged copies all refused or same-bytes; pin and quarantine right through rotation and delete; ~700 MB peak). Risks, not fails: one user's burst of big saves queues everyone (~12 s, no timeout); 128 MB unpack cap may be close (production's biggest unpack to 76 MB: check their compressed sizes against the 4 MB upload cap, needs a read-only look at the box's index, ask her); damage before compression after LBD1 still passes. Stage 3 (85e2945) not yet checked. Server must also know save level 19 (ferries) before deploy.
- newgame: T-3478, T-3479 landed by a02dbdc8 (checker passed; retry feedback fixed), not yet smoked on master.
- memcheck: T-3472 done, archived (notes/T-3472.md). No leak: all-China in Chrome 7.1 GB settled, 7.4-8.0 GB by minute 5, then flat; 3 solving workers hold 1.3 GB each (4.2 GB of 7.1), never shrinking. Base: 1.0 GB at the start menu, 1.6 GB in a one-city game. Could not reproduce ank's 5 s per track piece (0.1-0.3 s on all-China). T-3373 unblocked: A (3 -> 1 worker saves ~2.9 GB) and B act on the 4.2 GB. To file: zone outlines stored compactly on the page (~150 MB in every game); give back solving workers' empty memory (~0.7 GB on all-China, e.g. restart an idle worker); find the page's discarded-data peaks (up to 1.2 GB).
- prsurvey a46ce6d26ffe265b8 (read-only): PR 16 "Branch lines" appended to notes/pr-survey-2026-11-16.md.
- Agents that finished can't be resumed by SendMessage after a while ("No transcript found"): brief a fresh one.

- Anita 21:20 (via a02dbdc8): crowding dot 30% skew OK; T-3395 stage 2 strings and the T-3479 terrain tag hover acknowledged; no board light/dark switch (she uses the Windows setting).

## Waiting on Anita
- The old session a02dbdc8 is not reachable by SendMessage: sessions started by next_orchestrator.py inherited CLAUDECODE and counted themselves child sessions, never registering as peers (no ~/.claude/sessions/<pid>.json). Fixed in next_orchestrator.py (uncommitted); this session is unreachable too until the next cycle. She relayed to its tab 21:00.
- Server deploy of T-3395 stages 1+3 (ask once the checks pass, daytime).
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
