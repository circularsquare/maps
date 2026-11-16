# Orchestrator handoff, trainworld

Orchestrator: session a02dbdc8-1ab4-4517-95dd-2f768d109806, name "Takeover trainworld orchestration" (per maps-d7; message by session id), since 2026-11-16 18:38 UTC
Previous: session f7c8626c-6b89-4b09-983c-732243e787e9, name maps-d7 [e8137b], handed over 2026-11-16 18:38 UTC, finishing in-flight work
Updated: 2026-11-16 18:38 UTC

A short state page, rewritten (not appended) as things change. The diary is notes/handoff-log.md; older detail is in
git history. Read trainworld/CLAUDE.md and docs/orchestrator.md first.

## State
- Live = v0.3.5 (maps bfe58c54, published 11-15). Data 2026-11-13 freeze 20261113-013559 on R2.
- master and origin = e240791c+ (pushed; T-3459 edit after). 8800 = cefc5ac1 on live data/packs (has feeders; lacks T-1791
  hover and later). Rebuild 8800 at HEAD once the next batch lands (she has finished playtesting).
- Uncommitted on purpose in the main tree: T-3399's country plan edits (pipeline/world_plan.py/.json, realms.json,
  zone-names.json, docs/data-register.md, notes/T-3399.md): commit when T-3399 starts. Also not ours: TODO.md (hers),
  notes/T-3395.md, T-3397.md, jordan-save-investigation-2026-11-15.md, tools/ops/archive_bad_saves.py (saves session).
- Board inbox watch: Monitor on `venv/Scripts/python.exe -u trainworld/board/watch.py` (30 min, re-arm); armed by the new orchestrator 18:38.
- Server shonei-server: T-3402 + T-3440 deployed and pushed 11-16.

## Running agents (mine)
- helpflag aba4e2e9f3bb3a8d4 (Sonnet, .tw-trees/helpflag): T-2398 world_run step --help: done, check passes; land after maps-d7 finishes.
- cleanup ac16d7ecaad5a6eb7 (Opus, .tw-trees/cleanup): T-3459 done (212 notes scripts deleted, docs fixed); land after maps-d7.
- touchrule ad892057e8cf3781d (Sonnet, .tw-trees/touchrule): T-1764 done, smoke 28/28; checker touchcheck ae668a2055bcb2407 (6145/6146) running.
- changelog a04e4119ccbf7788a (Opus, .tw-trees/changelog, 6143): T-3466 changelog icons, then T-3467 roadmap (her pad 18:58).
- weekendscore ace2b666427cd1078 (.tw-trees/weekendscore): T-3465 forecast weekday points; checker passed the mechanics, 2 stale strings fixed after. Ready to land. LANDING: checkout next.json in the tree, add its changelog item fresh in main ("points now come from a forecast weekday and update as soon as you change your network"); retranslate cities.pointsPast, cities.pointsPastAll, topBar.pointsPast, cities.sortPoints, leaderboard.valueTip.
- prsurvey a46ce6d26ffe265b8 (Opus, read-only): open PRs 1/7/12/13 and T-3373 options -> notes/pr-survey-2026-11-16.md (her 19:01 ask). Her calls 19:08: PR 1 only as an option; PR 13 vs T-3346 my call (pull pieces or reject).
- pr7 a6de192a893e47c70 (Sonnet, .tw-trees/pr7): T-3470 done, identical figures, 240-line load 22.4 s -> 11.0 s; changelog c055 already in main. Checker PASS (3.85M fuzzed builds, 76 WASM snapshots identical). Ready to land. Commit with "Co-Authored-By: Alice <alice@herkula.info>" (GitHub peanutmow). Note weekendscore's tree also uses id c055: add its item fresh in main at landing.
- ttstyle a5748c5a59486ff83 (Opus, .tw-trees/ttstyle, 6149): T-3469 timetable restyle; screenshots to Anita before landing.
- changelog: T-3466 done (icons, groups; group names and T-3465 points strings acknowledged by Anita 19:23, icons approved). LANDING: agent checks out next.json in its tree; after land run scratchpad\changelog\apply_areas.py trainworld/changelog/next.json from the maps root; 8800 serve.py restart for the board menu. T-3468 pages done too. No changelog items for T-3466/3468/3467 (Anita 19:26); agent removes them from apply_areas.py. Now on T-3467.
- Queued after landings: T-3472 long-session memory check, then T-3473 (PR 12 speedups, after T-3470), then T-3373 A then B (her 19:18 call). T-3474 partly built lines, low priority. PR replies (1, 12, 13) to draft for her.
- T-2912 done 19:46: production rebooted onto kernel 7.0.0-43 after a fresh backup, all healthy.
- savesafe a992a8c39662ee3cc (Opus, shonei-server, commits there, no deploy): T-3395 stage 1 server validation, quarantine, pinned intact copy; stage 2-3 design to Anita.
- changelog: T-3466, T-3468, T-3467 all done (roadmap at changelog/roadmap.json via serve.py /twboard/_roadmap). Her open question: keep the "other" area (plain dot)?
- maps-d7 landed T-3448 (ec4b53eb) and T-3450 (694a624b), unpushed, no smoke yet: mine to smoke, push, rebuild 8800 once it's done with feedsample. Filed T-3462, T-3463 from its findings.

## Running agents (maps-d7, it lands these; flow-board and flow-tools already landed in ba9430b9)
- names acde25cc42203d034 (Opus, main tree, i18n place-name files): T-3450 Shiga and a Wikidata match guard; told
  to hand-check only prominent places.
- regions ac47cdc31cb8469fc (Opus, .tw-trees/regions, 6181): T-3448 one schematic per region.
- feedsample aca180a7fbc3e6ae7 (Opus, .tw-trees/t3454-base reverts c9db1359 for "before", 6196/6197): T-3454.
- flow-board a6ef6e2aa7e353ca5 (Sonnet): asks on the board (notes/asks.json), Someday -> P4, P4 not counted.
- flow-tools afb0d935b17ce763c (Sonnet): tools/tasks.py start/done/close/check; agent-tree JSON handling; T-3395 format.
- Landing: restore app/src/game/*.json in a tree before agent-tree land (until flow-tools automates it).

## Waiting on Anita
- Auto-starting the next orchestrator (T-3451 rec 6 replaced by her question): see the latest chat; nothing built.
- T-3399 Central America + Venezuela, T-3455 overnight scan: she will say when (18:45 "you dont have to mention it again"); don't list them in reports.

## Her standing asks this week
- Small isolated fixes, open-task count down, merge duplicates; bug investigations give up if hard.
- Quad paused (waits on T-3444). Pack rebuilds (region redraws, data fixes) batch into overnight runs.
- No windows on her screen: background processes hidden (start_hidden.ps1, CREATE_NO_WINDOW).
- Manual agent checking only for prominent cases; prefer automatic checks (11-16, on T-3450).
- Every publish, deploy and server push needs her yes to that exact action; none overnight.

## Playtest
- 18:31 Anita on 8800 (cefc5ac1, feeders): "looks pretty good", score unchanged, long-distance lines bigger. Hold lifted.
- A permission-allowlist addition (14 read-only rules) was refused to me by the permission check: list in chat 18:33; hers to add.

## Next
- Land the five agents above as they report; smoke after each batch; 8800 rebuild; push.
- Then more small fixes from P2/P3 bugs (ready, no waits); her review calls on T-3451 recs as they come.
