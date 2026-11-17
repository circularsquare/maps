# Orchestrator handoff, trainworld

Orchestrator: session 83439634-6553-4808-8acd-c1aa1af76a63, name maps-b8, since 2026-11-17 11:13 UTC
Updated: 2026-11-17 13:50 UTC (nightwrap, T-3505, T-2627/T-2777 landed and pushed; 8800 rebuilding)

A short state page, rewritten (not appended) as things change. The diary is notes/handoff-log.md; older detail is in
git history. Read trainworld/CLAUDE.md and docs/orchestrator.md first.

## State
- Live = v0.3.5 (maps bfe58c54, published 11-15). The next publish is v0.4.0 (Anita: big, with Central America and Detailed mode scheduling).
- master: 948e61f3 nightwrap's work (T-3399 closed at 91/95, T-617, T-3488, T-3499, T-2725; changelog c079-c081, c084), c4982864 T-3505 bubble fold, f26a28b5 T-2627/T-2777 join market step, then bookkeeping. Smoke passed on c4982864 (29/29; the later commit is pipeline only). Pushed.
- 8800 = 3d1ac614 (rebuilt ~14:55, has T-3480); master 3d1ac614 pushed.
- data/packs: free (nightwrap done). Backups pre-T3399/T3488/T3499 deleted.
- Before v0.4.0 ships: shonei-server deployed with save level 19 (T-3395), translations for all twelve languages (plus settingsTab.busOneWay once T-3480 lands). Still: freight tiles to R2 with the publish (T-3464; hover "Also freight and abandoned rail, trams, narrow gauge, heritage lines and unused track" + changelog item then), real graphs (now v2) reach R2 via deploy.py.
- Uncommitted on purpose: TODO.md (hers), notes/T-3397.md, jordan-save-investigation-2026-11-15.md, tools/ops/archive_bad_saves.py, changelog/roadmap.json (board-written).
- Board inbox watch: Monitor on `venv/Scripts/python.exe -u trainworld/board/watch.py` (30 min, re-arm). Nothing from her this session so far. Up next queue: `tasks.py upnext list|add|remove`.
- Resource check: done 11:14 UTC. Flags told to her: Resend at its daily cap every day since 11-12 (recovery and reset mails fail; Pro $20/month is hers), R2 storage 22.6 GB ($0.19/month; not purged), Class B month 72M (last 24 h 95k: old spike), website repo 1.1 GB, plays/day 51% (shonei-server 7a23cbc raises the cap to 250,000, committed, not pushed or deployed; rides the next server deploy). The T-3395 alert mail is in her inbox, unread by me.
- Player requests: fetched 13:20 (12 new); reqsort sorting them.
- Landing gotchas: agent-tree land refuses root files (serve.py); changelog ids in trees collide with main (agents add items in main); agents that edit TASKS/DECISIONS/ARCHIVE in their tree: revert there, redo in main; never `remove --force` a tree whose land stopped on a conflict; after a sim/ landing the main tree's WASM rebuilds itself. agent-tree land writes one -m plus Claude's trailer, so a contributor co-author can't go in the trailer paragraph: commit a PR landing by hand.

## Runs
- Coastal scan (T-3455): pid 30892 (ledger "nightscan", 3 cores), read-only, 8 h at 13:10 and still writing w0-w2.jsonl in the 00d74e61 scratchpad's nightprep\scan; output flagged.json there, copy to artifacts/trainworld/t3455-flagged.json when it ends. Its rebuilds are next night's list (with T-3511).

## Agents
- settle2 (Opus, maps-44's subagent, .tw-trees/settle, port 6154): T-3512 round 2 (Next 1). Its report reaches me only through maps-44's tab. Check directly: the tree's notes/T-3512.md, `git -C .tw-trees/settle log`, the ledger's "settle2" claim. Then a fresh checker, land, re-add the changelog line in main.
- bubfold done with T-3510 (tree .tw-trees/bubfold, uncommitted; new map/purposeRows.ts, purposes.ts, en.json two strings "Local trip legs {phrase}" -> "Trip legs {phrase}", "{count} local trip legs {phrase}" -> "{count} trip legs {phrase}"; changelog c085; DECISIONS line in main; smoke 29/29). bubfold is at ~500k tokens: don't resume it, a fresh agent for anything next. Checker rowcheck (Sonnet, port 6166) running; land if it passes (could ride v0.4.0; two strings to translate). Follow-up unfiled: a re-solve's first round can give local-only legs when no earlier answer is kept.
- joinstep done and landed (f26a28b5, tree removed); `join --market-only` only tested with the step mocked: first real use is the next failed join. Unfiled follow-ups: local trips' 1500-zone chunks in parallel; the stale-market-file preflight from the T-2702 retrospective.
- stationedit (Sonnet, .tw-trees/stationedit, port 6163): T-2767 + T-2768 (station place/delete on built track rebuild only what they touch; notes/T-2763.md proposals); figures to the bit vs master; then a checker.
- reqsort done: 12 reports into board-requests.json (5 new groups g078-g082, 7 into g002/g003/g004/g015); g080 iPad can't load Kansai/Tokyo matched to T-3160.
- trainsfan (Sonnet, .tw-trees/trainsfan, port 6167): T-3520, no server changes (stops and says so if the server must count it).
- piecelimit (Opus, .tw-trees/piecelimit, port 6164): T-3405 per-piece speed limit in the track piece panel (her yes 14:45: "lets add that ... just set it in the track segment view"); new save block + level, and it updates shonei-server's save check (commits there, no push); that server change must deploy before a game version writing the new level ships. Then a checker.
- busoneway LANDED 89476a46 by hand (her "no more checks needed"), co-authored with Renkon; tree removed, T-3480 archived. PR 1 and issue 8 replies to draft for her (closed with a link after the next open-source sync). Was: T-3480 in .tw-trees/busoneway (setting "Keep buses to one-way streets", off by default; checker buscheck PASSED: 150 legs identical to master when off; on mode refuses 3 of 150 plain legs past the detour cap, gracefully; Edit-route tool with it on untested). Waits for her to see it (PR rule), then commit by hand with `Co-Authored-By: Renkon (Fernando Martín Besteiro) <1048587+Renkon@users.noreply.github.com>` beside Claude's; then a reply on PR 1 for her to OK. Changelog c082 already in main.

## v0.4.0 (she targets a release later today, 2026-11-17)
- Timetables are NOT ready: stage 1 (panel, printed-style table, chart) is on master behind ?timetables; stage 2 (per-train waits) sits in .tw-trees/tt2, 5 days stale, 30/10 map case failing, fix left 4-6 h, and it touches the settle code settle2 is changing; Detailed mode (stages 3-4, her "different headways for weekend/weekday, 5 periods") not started (18-25 h estimated). Proposed to her: ship 0.4.0 today without them; asked whether to unhide stage 1 as a preview.
- To do before publish: translation batch (only settingsTab.busOneWay since d41a157e, plus the updates.v040* keys), changelog publish (`changelog.py line`, `publish --version v0.4.0`), freight tiles to R2 + hover + changelog item (T-3464), real graphs v2 via deploy.py, smoke8800 light and dark, deploy.py, website push, open-source export sync + push, then post the PR 1 / issue 8 replies (her OK given 14:50) and close PR 1. Server: 7a23cbc (playtime cap) deploy on her yes; T-3395's level-19 check looks deployed (service restarted ~01:35 UTC after 3e8e501), unverified.
- Her 15:00 calls: no timetables in 0.4 (stay hidden; next release with Detailed mode); release late tonight US east (player peak), "theres still a lot of time"; server deploy not yet (she asked if we're ready: decide what else rides it first); terrain on by default (done, c087, UI store v15 drops stored off once); T-3510's two strings acknowledged; new T-3520 trainsfan achievement (100 h played) for 0.4 if ready.
- In or out: settle2's T-3512 if its checker passes in time; T-3510 if rowcheck passes; T-3520 if ready; stationedit and piecelimit wait for 0.4.1 unless done and checked (piecelimit's new save level would also need its server change deployed first).

## Next
1. T-3512 (settle2): checker then land when it reports.
2. Performance, non-memory only: T-3388 (London edits ~4 s) after T-3512. T-1080, T-930 (world tooling) free now. Memory tasks parked (standing asks). T-3511 after the scan. T-3495 freight snap after v0.4.0.
3. Leftover non-git folders in .tw-trees (c1, ferry45, fixes, freight, hotfix031-target, hover, touchrule) and stale trees (safari-save-035, shenzhen, tt2 with 50 dirty files, ttstyle, trams): look before removing; never delete by hand (node_modules junction).

## Waiting on Anita
- PR 16 and PR 22: she defers both (14:00, leaning to leave 16; 22 "mostly covered by our new track visibility things"). No reply on either yet. Resend: leave it (13:14).
- PRs 17-20: replies posted 14:10 with her OK; 17 and 20 closed; 18 and 19 left open asking Cinnamon for timings (their PRs carry none). Watch for an answer.
- T-3480: arrows too thick (her: "closer to proportional to the visual weight of the road itself ... maybe 1.5x"); busoneway redid them at 1.5x the road stroke (from z15, 90 px apart); she wanted 1.2x (14:30: "the opacity difference is what made me underestimate"); done (review/bus_oneway_arrows12_z17.png, _z15.png); next lever if still heavy: arrow opacity 0.7 (the road hint is ~55%). Label is now her wording "Buses obey one-way streets". Waiting for her OK on the arrows, then commit by hand with Renkon's trailer and remove the tree.
- Keep the maps-44 tab open until settle2 reports.
- Told in chat earlier (by maps-44), nothing blocking: T-3503 sum rule and Mettmann +57%; T-3503 memory; snap icons (artifacts/trainworld/review/snap_icons.png).

## Her standing asks this week
- Memory: done for now (Anita 14:30: "~500MB savings ... a noticeable improvement ... going forward, for our performance stuff we can focus on non memory stuff for a while"). The headline figure for the changelog and her posts is about 500 MB less on desktop (changelog group c086 over c066, c070-c072). No memory tasks (T-3492, T-2136, T-3160...) unless she asks; no core-balance shifting.
- Small isolated fixes, open-task count down, merge duplicates; bug investigations give up if hard.
- No asking about changelog lines: she reviews them on the board. No changelog items about the changelog itself.
- Overnight: keep rebuilding while queued builds remain.
- No windows on her screen: background processes hidden.
- Every publish, deploy and server push needs her yes to that exact action; none overnight.
- Advisor trial: start successors with `next_orchestrator.py trainworld --advisor fable`; log each call in handoff-log.
