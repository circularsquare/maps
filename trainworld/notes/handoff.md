# Orchestrator handoff, trainworld

Orchestrator: session a02dbdc8-1ab4-4517-95dd-2f768d109806, name "Takeover trainworld orchestration" (message by session id), since 2026-11-16 18:38 UTC
Previous: session f7c8626c-6b89-4b09-983c-732243e787e9, name maps-d7 [e8137b], handed over 2026-11-16 18:38 UTC, finishing in-flight work (only feedsample's note and TASKS fields left to commit; told it may commit)
Updated: 2026-11-16 20:40 UTC (a02dbdc8 at ~390k tokens: starting its successor; my running agents below are the successor's to adopt)

A short state page, rewritten (not appended) as things change. The diary is notes/handoff-log.md; older detail is in
git history. Read trainworld/CLAUDE.md and docs/orchestrator.md first.

## State
- Live = v0.3.5 (maps bfe58c54, published 11-15). The next publish is v0.4.0 (Anita 20:23: big, with Central America and Detailed mode scheduling).
- 8800 = origin = 45b7808c. master = 441d4a25, ahead and unpushed: timetable restyle T-3469, points wording, touchrule T-1764, Community count dropped, board bars. Next: land the agents below as they pass, then one smoke on master, push, snap8800 (snap_wasm emoji crash fixed in 45b7808c).
- Translation batch needed before v0.4.0: points keys cities.pointsPast, cities.pointsPastAll, topBar.pointsPast, cities.sortPoints, leaderboard.valueTip, cities.pointsRule, leaderboard.pointsRule (docs/localization.md passes), plus any keys the agents below add.
- Uncommitted on purpose in the main tree: T-3399's country plan edits (pipeline/world_plan.py/.json, realms.json, zone-names.json, docs/data-register.md, notes/T-3399.md): commit when T-3399 starts. Not ours: TODO.md (hers), notes/T-3397.md, jordan-save-investigation-2026-11-15.md, tools/ops/archive_bad_saves.py. notes/T-3395.md is savesafe's (untracked, commit with T-3395).
- Board inbox watch: Monitor on `venv/Scripts/python.exe -u trainworld/board/watch.py` (30 min, re-arm).
- shonei-server: T-3395 stage 1 (37217ea, 4623fde, 094b830) and stage 3 (85e2945) committed, not pushed or deployed. Deploy stages 1+3 in daytime on her yes once savecheck passes stage 1 and stage 3 is checked.
- Production box rebooted 19:45 onto kernel 7.0.0-43 (T-2912 done). jsonbin retired (memory note).
- Landing gotchas this session: agent-tree land refuses root files (serve.py): git diff the tree's serve.py to a patch (add a trailing newline), apply in main, checkout in the tree, then land; a contributor's co-author needs a hand commit (land adds only the Claude trailer). Changelog items need --area (track, demand, scheduling, fares, schematic, community, performance, other, bug; --fix for lighter).

## Running agents (mine, the successor's to adopt)
- ttstyle2 a9f5ba644ea623cd6 (Opus, .tw-trees/ttstyle2, 6149): T-3475 timetable hover highlight, chart leads with the table's direction (reverse dimmed), vertical bullets centred. Timetables hidden behind ?timetables: land without a checker.
- ferry45 a49ebbc59481d527b (Sonnet, .tw-trees/ferry45, 6155): T-3435 ferries 32 -> 45 km/h (her call). Needs a checker before landing.
- crowddot a4726042dad524e70 (Sonnet, .tw-trees/crowddot, 6157): T-3476 crowding dot fill as a plain average, ring unchanged. Show her the before/after.
- savesafe2 ae548dbe1fab00e93 (Opus, .tw-trees/savesafe2, 6156): T-3395 stage 2, game side (strings proposed in notes/T-3395.md). Needs a checker.
- savecheck a2132c3f7caec1b0c (Opus, 6153): checker for T-3395 stage 1 (above all: a good save never refused). Then check stage 3 too.
- newgame a5fcacce9edc5fc19 (Opus, .tw-trees/newgame, 6160): T-3478 new game from a game goes via the menu, T-3479 click the terrain notice to retry.
- memcheck aab5014326634c569 (Opus, 6154): T-3472 long-session memory check, plus base memory before any city (her ask); feeds T-3373 A then B.
- prsurvey a46ce6d26ffe265b8 (read-only): PR 16 "Branch lines" appended to notes/pr-survey-2026-11-16.md.
- Agents that finished can't be resumed by SendMessage after a while ("No transcript found"): brief a fresh one.

## Waiting on Anita
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
- When cores free up (her 20:34): world build tooling ahead of coming world builds. Start with T-2298 (gate from a frozen app copy) and T-2399 (gate browser check fails silently); then T-1080, T-2627/T-2777, T-2070, T-930, T-2095, T-1992, T-2326.
- PR replies for 1, 7, 12, 13 (and 16 after the survey) to draft for her: 7 taken with credit, 13 reused at T-3346 stage 4, 12's partly built lines as T-3474, the rest left out (DECISIONS 2026-11-16).
- T-3474 partly built lines (low priority).
