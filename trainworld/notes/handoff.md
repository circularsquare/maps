# Orchestrator handoff, trainworld

Orchestrator: session 83439634-6553-4808-8acd-c1aa1af76a63, name maps-b8, since 2026-11-17 11:13 UTC
Updated: 2026-11-17 11:25 UTC (maps-b8 took over; nightwrap and settle2 adopted, bubfold, busoneway, prsurvey2 started)

A short state page, rewritten (not appended) as things change. The diary is notes/handoff-log.md; older detail is in
git history. Read trainworld/CLAUDE.md and docs/orchestrator.md first.

## State
- Live = v0.3.5 (maps bfe58c54, published 11-15). The next publish is v0.4.0 (Anita: big, with Central America and Detailed mode scheduling).
- master 079d9c8b + the bookkeeping commit after it (pushed), smoke passed on master at 079d9c8b (29/29, incl. t1053). Landed this morning: the night's world runs (c4d540a0: Central America + Venezuela, Lisbon/Glasgow/Porto, Istanbul T-3488, Bucaramanga/Colombia T-3499 + T-2725), T-3500 airport hover (d96e8f7d), T-3503 bubbles with trips between places (d5ea0ded), T-3069 tram snap + graph v2 (079d9c8b; the 120 v2 graphs copied into data/packs/realgraph, --check all pass). Main WASM rebuilt.
- 8800 = b42afa6f (rebuilt 11:15). data/packs: nightwrap's park reruns own it from 11:20.
- 32 zones lacked terrain after the night (Turkey 3, Colombia 29: TerrainMissing on a line build); built 11:00 with zone_terrain.py build (run it without -I). build_inputs now 0 flags.
- Before v0.4.0 ships: shonei-server deployed with save level 19 (T-3395), translations for all twelve languages. Still: freight tiles to R2 with the publish (T-3464; hover "Also freight and abandoned rail, trams, narrow gauge, heritage lines and unused track" + changelog item then), real graphs (now v2) reach R2 via deploy.py.
- Uncommitted on purpose: TODO.md (hers), notes/T-3397.md, jordan-save-investigation-2026-11-15.md, tools/ops/archive_bad_saves.py, changelog/roadmap.json (board-written).
- Board inbox watch: Monitor on `venv/Scripts/python.exe -u trainworld/board/watch.py` (30 min, re-arm). Up next queue: `tasks.py upnext list|add|remove`.
- Resource check: done 11:14 UTC. Flags told to her: Resend at its daily cap every day since 11-12 (recovery and reset mails fail; Pro $20/month is hers), R2 storage 22.6 GB (a $0.19/month bill; not purged), Class B month 72M (the last 24 h only 95k: old spike), website repo 1.1 GB, plays/day 51% (shonei-server 7a23cbc raises the cap to 250,000; rides the next server deploy, and check_resources reads the new limit from source meanwhile). The T-3395 alert mail is in her inbox, unread by me.
- shonei-server: 7a23cbc committed, not pushed or deployed.
- Landing gotchas: agent-tree land refuses root files (serve.py); changelog ids in trees collide with main (drop the tree's next.json, re-add in main); agents that edit TASKS/DECISIONS/ARCHIVE in their tree: revert there, redo in main; never `remove --force` a tree whose land stopped on a conflict; after a sim/ landing the main tree's WASM rebuilds itself.

## Runs
- Coastal scan (T-3455): started by the old nightqueue.ps1 (pid 30892, ledger "nightscan", 3 cores), read-only, still running at 6 h (11:20 UTC: workers still writing w0-w2.jsonl in the 00d74e61 scratchpad's nightprep\scan, on vea/nrw/klm); output scratchpad nightprep\scan\flagged.json, copied to artifacts/trainworld/t3455-flagged.json when it ends. Its rebuilds are next night's list.

## Agents
Both started by maps-44 at 11:20 and handed to maps-b8 (they will run for hours). Their reports go to maps-44, which relays them to maps-b8 while its tab is open; otherwise check their work directly:
- nightwrap (Opus, main tree): Next 1 below. Check: notes/T-3399.md (relaxations, batch report), `world_run.py status`, the ledger's "nightwrap" claim, ARCHIVE.md for closes of T-3399/T-617/T-3488/T-3499/T-2725, changelog/next.json for Istanbul/Colombia items, place-names-zones.json without arv. It does not commit. May file T-3517, T-3518.
- settle2 (Opus, .tw-trees/settle, port 6154): Next 2 below. Check: the tree's notes/T-3512.md and `git -C .tw-trees/settle log` (it rebases onto master), the ledger's "settle2" claim. It does not touch bookkeeping; any DECISIONS text comes in its report (or its notes). After it: a fresh checker, then land, re-add the changelog line in main.
Started by maps-b8 at 11:20 UTC (report to maps-b8):
- bubfold done (tree .tw-trees/bubfold, uncommitted): T-3505 stage 1, bubbles under 1 trip fold into the nearest bubble of 1+ at every level (map/bubbles.ts foldSmall + commuterView, commuters worker/protocol; none of settle2's files); changelog c083, DECISIONS line in main, notes/T-3505.md in the tree. Site folding is pipeline-only and small: filed T-3519 (4 km rural blocks for thin packs). Checker bubcheck PASSED in two rounds (Barrie, Sol Plaatjie: no bubble under 1, sums equal zone totals at every level, absorbing-bubble hover = inspector; NYC real save Homes fine; one hover/click pick mismatch between overlapping discs at z11, not chased). Land after nightwrap's checks (its world run ended ~12:30, it is still on morning checks); resume bubfold for T-3510.
- busoneway done (tree .tw-trees/busoneway, uncommitted; app only, no sim change): T-3480 setting "Keep buses to one-way streets" (Settings > Show, off by default; legs identical to master when off, 48 legs NYC fresh + real save); changelog c082 in main; smoke 29/29. Checker buscheck PASSED (150 legs identical to master when off, no console errors, same load time; on mode refuses 3 of 150 plain legs past the detour cap, gracefully; Edit-route tool with it on untested). Land after nightwrap's gates (main-tree app/ writes held), co-author `Renkon (Fernando Martín Besteiro) <1048587+Renkon@users.noreply.github.com>`; translations of settingsTab.busOneWay with the next batch; PR 1 reply drafted for her after landing.
- prsurvey2 done (notes/pr-survey-2026-11-17.md): #17-#20 already settled by T-3473 (draft reply shown to her; post and close once she OKs it); #21 is T-3474's reference (linked); #16 and #22 asked (asks.json). Maps is at save level 19 already, the export at 18. busoneway told #17 is already in maps.
- 8800 = b42afa6f (rebuilt 11:15).

## Next
1. Night follow-ups (one Sonnet/Opus agent): T-3399/T-617 morning steps per notes/T-3399.md (batch report; parks clo, mbm, led, sck, tsx, crj, auo, dnl, rta and t617's: relax and rerun, join); night-queue morning list in notes/night-queue-2026-11-17.md (changelog items for Istanbul and Colombia, drop arv names from app/src/i18n/place-names-zones.json, DECISIONS: Altinova to Yalova, Sile stays, Bucaramanga 1.51M over the cap; checks: stale, zone_origins, state_gaps TUR/COL, split_zones, the real save); close T-3488, T-3499, T-2725 (and T-3399/T-617 when parks are done); delete backups pre-T3488, pre-T3499, T3399 once clean. Before rerunning parks, check the gates don't read figures T-3503 changed.
2. T-3512 second round (fresh Opus; settle's tree .tw-trees/settle on 286ea890, uncommitted): drop or default-off stop-on-whole, rebase onto master (T-3503 touched the same files), re-run bubfix's Mettmann flicker repro on the combined code, the hang sequence 5+ times both modes, one unpinned run past a re-level; then a checker; re-add changelog "changing a line's trains at medium or low demand reached its final riders in 7 to 20 seconds on the new york save, where it took about a minute and a half" (performance). Results so far: NYC low 102 -> 6.6 s, medium 103 -> 20 s, every level 106 -> 57 s, high unchanged (T-3513).
3. T-3511 enclave shifts/merges as night stages (after the scan's flagged list is known; a night queue like nightprep's).
4. T-3510 bubble purpose rows (after bubfold, same code; resume bubfold for it). T-3492 after settle2 lands (same worker files). T-2136, T-3388 (after T-3512). T-1080, T-930, T-2627/T-2777 after nightwrap (pipeline, packs). T-3495 freight snap after v0.4.0.
5. Leftover non-git folders in .tw-trees (c1, ferry45, fixes, freight, hotfix031-target, hover, touchrule) and stale trees (safari-save-035, shenzhen, tt2 with 50 dirty files, ttstyle, trams): look before removing; never delete by hand (node_modules junction).

## Waiting on Anita
- Keep the maps-44 tab open until nightwrap and settle2 report (they are its subagents). Resend Pro ($20/month) or live with failed recovery mails.
- Told in chat, nothing blocking: T-3503 sum rule and Mettmann +57%; T-3503 memory (+31 MB page on NYC, cache cap 48 MB); train track now snaps to monorail/funicular/narrow gauge/heritage; enclave calls La Plata/Chascomus and Sanliurfa/Siverek; a checker killed an unrelated Codex node process (PID 5188) at ~09:05 UTC; snap icons to look at (artifacts/trainworld/review/snap_icons.png).

## Her standing asks this week
- Memory: pause once there are no easy gains; no core-balance shifting.
- Small isolated fixes, open-task count down, merge duplicates; bug investigations give up if hard.
- No asking about changelog lines: she reviews them on the board. No changelog items about the changelog itself.
- Overnight: keep rebuilding while queued builds remain.
- No windows on her screen: background processes hidden.
- Every publish, deploy and server push needs her yes to that exact action; none overnight.
- Advisor trial: start successors with `next_orchestrator.py trainworld --advisor fable`; log each call in handoff-log.
