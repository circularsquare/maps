# Orchestrator handoff, trainworld

Orchestrator: session eda3ac12-4519-4720-a0c5-be8bdcb2ed0b, name maps-60, since 2026-11-17 14:40 UTC
Updated: 2026-11-17 14:40 UTC (maps-b8's agents report to maps-b8, which relays them verbatim to maps-60 while its tab is open)

A short state page, rewritten (not appended) as things change. The diary is notes/handoff-log.md; older detail is in
git history. Read trainworld/CLAUDE.md and docs/orchestrator.md first. (Times of her messages below are approximate:
maps-b8's earlier notes stamped some 1-2 h late.)

## State
- Live = v0.3.5 (maps bfe58c54, published 11-15). Next publish v0.4.0: NO deadline (Anita: "lets please forget about the whole tonight deadline thing"); she picks when, likely evening US east, and keeps track of the landing cut-off herself ("you guys dont have to worry about landing new work"): land things as they pass.
- master acc208b8 + bookkeeping (T-3405 speed limit 25736cbd, T-2768 station delete acc208b8 landed by maps-60); smoke PASSED on acc208b8 (29/29). master d607c0b2 pushed; 8800 = d607c0b2 (swapped ~15:10). Changelog c089 (busboard) and c092 (shore) are committed ahead of their code: hide them if a publish comes first.
- Server: shonei-server 96e4f12 (save level 20, LPS1) and 95356ed (level 21, LWE1; twSaveCheckV stays 2) are committed, not pushed or deployed; both are required before v0.4.0 publishes. TestTwSaveCheckFollowsTheGame goes green once weekend lands.
- In master since v0.3.5 (for the v0.4.0 notes): Central America + Venezuela (91/95 rows), Lisbon/Glasgow/Porto demand gaps, Istanbul and Colombia redraws, ~500 MB less memory on desktop, 240-line networks load about twice as fast, bubbles under 1 trip fold (T-3505), bubble purpose rows sum to the total (T-3510), buses obey one-way streets setting (T-3480, co-authored with Renkon), tram snap (T-3069), terrain on by default (UI store v15 drops a stored off once), trainsfan achievement (T-3520, 100 h played, counted per browser from zero), ferries 45 km/h (T-3435), join market step tooling (T-2627/T-2777).
- Changelog tense rule (CLAUDE.md, her call): past for the change, present for how things are now; c066/c068/c070-c072/c086 reworded; c086 "the game uses about 500 MB less memory on desktop" is the headline group (terrainmem is checking that figure with terrain on).
- Before v0.4.0 ships: translation batch (settingsTab.busOneWay, T-3510's two strings, trainsfan's 4 keys, the timetable window's strings once weekend ungates it, updates.v040* keys); `changelog.py line` + `publish --version v0.4.0`; freight tiles to R2 + hover + item (T-3464); real graphs v2 via deploy.py; smoke8800 light and dark; deploy.py; website push; open-source export sync + push; then post Renkon's PR 1 and issue 8 replies (her OK given; text in handoff-log under 2026-11-17 maps-b8) and close PR 1 with a link. Server: one deploy just before the publish carrying 7a23cbc (playtime cap 250,000) plus piecelimit's and weekend's save-check changes if they land; T-3395's level-19 check looks deployed (service restarted ~01:35 UTC after 3e8e501), unverified. Each publish/deploy/push needs her yes.
- Uncommitted on purpose: TODO.md (hers), notes/T-3397.md, jordan-save-investigation-2026-11-15.md, tools/ops/archive_bad_saves.py, changelog/roadmap.json (board-written).
- Board inbox watch: Monitor on `venv/Scripts/python.exe -u trainworld/board/watch.py` (30 min, re-arm). Up next queue: `tasks.py upnext list|add|remove`.
- Resource check done 11:14 UTC (told her; she said leave Resend on the free plan). shonei-server 7a23cbc committed, not pushed or deployed.
- Player requests: fetched and sorted 13:20 (reqsort; g078-g082 new on the board).
- Landing gotchas: agent-tree land refuses root files; agents add changelog items in main; never `remove --force` a tree whose land stopped on a conflict; after a sim/ landing the main tree's WASM rebuilds itself. agent-tree land writes one -m plus Claude's trailer, so a contributor co-author landing is done by hand (T-3480's way: add the tree's files, commit -F with both trailers, rebase onto master, `git merge --ff-only tw/<name>`, then `agent-tree.mjs remove`).

## Runs
- Coastal scan (T-3455) FINISHED: artifacts/trainworld/t3455-flagged.json (4119 scanned, 501 rows with pieces, 27 groups, 46 timeouts). Planner agent nightplan (maps-60's, Opus) is writing notes/T-3455.md: tonight's rebuild plan folding in T-3511. No runs yet.
- Ledger at 14:40: 7 of 14: settle2 3, weekend 3, busboard 1.

## Agents (all maps-b8's subagents except settle2; reports come to maps-b8 and are relayed. If maps-b8's tab is gone, check each directly: its tree's git status, notes, the ledger)
- settle2 DONE (T-3512; maps-44 closed, can't be resumed). maps-60 rebased tw/settle onto master d607c0b2 (clean, 3 WIP commits); checker-settle (maps-60's, Sonnet, ports 6114/6115) running. On PASS: land (WIP commit messages: squash or land by hand with one sentence), add its two DECISIONS lines (end of the tree's notes/T-3512.md), LOCAL_SHARE_MAX 0.35 accepted, changelog "train count changes on a one-city game settle in seconds to about a minute, not one to two minutes" (performance), archive. Cap-lowering undo bug filed as T-3528.
- weekend DONE (maps-b8 wound down 16:30; all its work is mine now). Tree brought up to master 1b75a5cf (clean). checker-weekend (Sonnet, ports 6123/6124) running; savecheck21 (Sonnet) adding LWE1/level 21 to shonei-server, commit there, no push/deploy. On PASS: commit by hand with co-author trailer "Co-authored-by: gabs <moraes.madm@gmail.com>" (GitHub GabrielMoraez) plus Claude's; changelog c094, c095; strings shown to her 16:35. Old notes follow.
- weekend (Opus, .tw-trees/weekend, port 6168): T-3521 weekend train counts, in v0.4.0 when ready. Her design: in the timetable window (the whole window ships; remove the ?timetables gate; stage 2's local-before-express glitch is acceptable to her); Weekdays and Weekends sections with count/headway per demand level; beside the Weekends header a percent stepper (default 100%, 10-200% in 5%) and an "apply to all lines" button with a confirmation; a percent line's weekends follow weekday edits (counts rounded up, headway divided and rounded down to a minute); a hand-edited weekend makes the percent show greyed; holidays run weekends; one weekend. Sim from GabrielMoraez's PR 13 (closed). Save level 21 (piecelimit 20); I (the orchestrator) add its block to shonei-server's save check after piecelimit's edit. Then a checker on the merged code (after piecelimit/stationedit land), she sees it, commit by hand with Gabriel's co-author trailer (name/email in its report).
- LANDED by maps-60: piecelimit T-3405 (checker PASS, strings acknowledged; follow-up T-3525) and stationedit T-2768 (T-2767 stays open, not fixed). Both trees removed.
- helplist LANDED 6953730f (T-3527 archived by the agent; changelog c096; strings shown to her). Old keys settingsTab.controls and help.title to drop in the translation batch.
- singlecap BUILDING fix 1 (her pick 16:20), plus one (agent adda47c60a86f18ba, .tw-trees/singlecap at f30ba353, port 6113). Sim change: checker before landing.
- helplist resumed (agent ae2017dd4f8b762f0, .tw-trees/helplist2, port 6112): adding entries edit built track, four tracks (new stations + line), express (Copy line), and recalc only if it needs a non-default setting; reporting what the 4 "single to double" reports really asked (she asked).
- trackrebuild LANDED 61800d64 (T-3408, archived; changelog c093; string "Use the Select tool (4) to rebuild longer stretches" for her; Remove track moved beside Rebuild; rebuild stays final, no undo, like the bar). Tree removal failed (its Vite held files): agent asked to stop, then `agent-tree.mjs remove trackrebuild`.
- trackrebuild follow-up LANDED 9b1eaeaa (level and tram/rail options in the panel only where that piece can change; c093 reworded by the agent). Done; tree removed.
- tramlag (maps-60's, Opus, agent a408e1164d74c0a7f, .tw-trees/tramlag, port 6118): T-3531 closed (not reproduced; she found 8800 fine later, machine at the 14-core cap). Now T-3535: tram-rail crossing mark in the drawing preview.
- tracklook LANDED 81c6fb64 (T-3533, T-3534 archived by the agent; c098, c099). The "white stripe" was the tram wire on single tram track; rail single track had none. Not on 8800 yet.
- tramstation (maps-60's, Opus, .tw-trees/tramstation, port 6126): T-3540 tram stations draw their passing loop, one-track stations a smaller platform to the outside, tram platforms 30% narrower. She confirmed the white stripe was tram (fix kept).
- complexpick (maps-60's, Sonnet, .tw-trees/complexpick, port 6125): T-3539 clicking a station inside a selected complex selects that station.
- bamvd DONE (notes/T-3532.md): drive grid crosses the Parana delta/Plata head as land, 5 h 41 vs real ~7 h; NE sea burn-in tested 7 h 22. Waits on her (asks a-2026-11-17-7): mask fix + drives rebuild overnight; measure other pairs first.
- LANDED later: settle T-3512 fbadc218 (squash; checker PASS), tramlag T-3535 e9547f93, complexpick T-3539 66379382; smoke PASSED on 66379382. Changelog c100-c102 added in main for them.
- weekend checker FAILED: headway lines' weekend headway ignores the percent (and save.rs test-string mojibake). weekend2 (maps-60's, Opus, agent a8b211c279a6dfa8a, in .tw-trees/weekend, port 6127) fixing; then a fresh checker round on headway + parity, then hand commit with Gabriel's trailer.
- tramsplit also has T-3541 (blueprint extension off the split line's end stops the line; her pad 16:24).
- LANDED this round: shore 66e46fed (T-3524, checker PASS), reclaim cde0f3fb, snapicon fff4fbd9 (T-3536), helplist2 8261f6eb (4 more help entries; recalc included since the setting defaults on). Smoke PASSED on 8261f6eb.
- tramsplit (maps-60's, Opus, agent a089adcfacfecebf2, .tw-trees/tramsplit, port 6122): T-3537 street snap merges a double track's split (her pad), T-3538 split tram line switchback mark at its end.
- 8800 = 1b75a5cf (adds shore margin, snap icon, more help entries), master pushed. She OK'd the shore 12 m margin ("yup").
- busboard LANDED 9fd75399 (T-3522 archived; table shown to her). trackrebuild's tree folder is an empty leftover (.tw-trees/trackrebuild, branch tw/trackrebuild kept).
- DECISIONS.md holds weekend's 5 uncommitted lines: commit DECISIONS with weekend's landing.
- reclaim LANDED cde0f3fb (T-3529 archived): `reclaimed` drops pieces half or more abroad (NE 10 m), clips the rest; batch-2 rows unchanged. nic and pyo-kp are free for a later night (not tonight: the plan is already ~8.2 h).
- nightprep DONE: kit in notes/T-3455/kit (uncommitted; commit with the run), driver night1117.ps1 rehearsed, all dry runs clean, mop (Mokpo, 21,955) added to batch 2 (28 rows), abd nothing, enclave rescan 121 cases (one new: auo Puerto Ayacucho, likely keep), disk 45 GB free. NIGHT RUN (~8.2 h + optional cif-ph batch 3; batch 2 skips itself if it can't finish by 10:00 local): start once the day's landings are done, about 21:00 local (02:00 UTC), with `powershell -File C:\Users\anita\projects\maps\trainworld\tools\country\start_hidden.ps1 C:\Users\anita\projects\maps\trainworld\notes\T-3455\kit\night1117.cmd`. Log artifacts/trainworld/night1117.log; state/PID kit\night1117.state; stop: create trainworld\data\work\world_run\STOP then kill only that PID. It claims 10 cores (waits for 6 free). Hold app/ writes (landings) during its gates. T-3530 (Tokyo Haneda) filed.
- shore DONE (T-3524, 12 m WATER_MARGIN on pack masks; NYC built cost -0.4%; uncommitted in .tw-trees/shore, brought up to master d607c0b2 by maps-60, wiki/water.md line added by maps-60). checker-shore (maps-60's, Sonnet, ports 6116/6117) running, incl. in-game before/after shots. On PASS: run wiki_build in the tree, land, archive. For her: 12 m OK? (worst case ground track ~6 m into drawn water; NYC's TIGER water may still look off -> pack rebuild with OSM water).
- terrainmem DONE (notes/terrain-memory-2026-11-17.md): terrain on adds ~50 MB (flat NYC) to ~250 MB (hilly Innsbruck) on desktop, ~170 MB on phone; c086 now reads "about 350 MB less" (her pick, 15:10).

## Next
1. Land each agent as it passes (checker for sim/save/demand/money). Rebuild 8800 after each batch; push master.
2. When she says release: the v0.4.0 steps under State, each on her yes.
3. Performance, non-memory only: T-3388 (London edits ~4 s) after T-3512. T-1080, T-930 (world tooling). T-3511 after the scan. T-3495 freight snap after v0.4.0. Timetables: stage 2 fix (tt2, stale) and Detailed mode rows later.
4. Leftover non-git folders in .tw-trees (c1, ferry45, fixes, freight, hotfix031-target, hover, touchrule) and stale trees (safari-save-035, shenzhen, tt2 with 50 dirty files, ttstyle, trams): look before removing; never delete by hand (node_modules junction). A hand delete of old review copies was refused once: leave them.

## Waiting on Anita
- asks.json empty. T-3523 decided (14:50): B, the public repo becomes the real one, after v0.4.0; its design goes to her first. T-3405's strings acknowledged.
- PR 16 and PR 22: she defers both, thinking later (leaning to leave 16; 22 "mostly covered by our new track visibility things"). No replies yet.
- PRs 17-20: replies posted with her OK; 17 and 20 closed; 18 and 19 open asking Cinnamon for timings. Watch for an answer.
- Keep the maps-44 tab open until settle2 reports (she said yes).

## Her standing asks this week
- Memory: done for now ("going forward, for our performance stuff we can focus on non memory stuff for a while"); no memory tasks unless she asks; no core-balance shifting.
- Small isolated fixes, open-task count down, merge duplicates; bug investigations give up if hard.
- No asking about changelog lines: she reviews them on the board. No changelog items about the changelog itself.
- Overnight: keep rebuilding while queued builds remain.
- No windows on her screen: background processes hidden.
- Every publish, deploy and server push needs her yes to that exact action; none overnight.
- Advisor trial: start successors with `next_orchestrator.py trainworld --advisor fable`; log each call in handoff-log.
