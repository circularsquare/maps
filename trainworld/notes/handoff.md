# Orchestrator handoff, trainworld

Orchestrator: session f3eca8a5-6550-4b65-9245-def721281ba7, name maps-44, since 2026-11-17 03:51 UTC
Updated: 2026-11-17 11:07 UTC (Anita asleep from 03:45 UTC)

A short state page, rewritten (not appended) as things change. The diary is notes/handoff-log.md; older detail is in
git history. Read trainworld/CLAUDE.md and docs/orchestrator.md first.

## State
- Live = v0.3.5 (maps bfe58c54, published 11-15). The next publish is v0.4.0 (Anita: big, with Central America and Detailed mode scheduling).
- master 079d9c8b + the bookkeeping commit after it (pushed), smoke passed on master at 079d9c8b (29/29, incl. t1053). Landed this morning: the night's world runs (c4d540a0: Central America + Venezuela, Lisbon/Glasgow/Porto, Istanbul T-3488, Bucaramanga/Colombia T-3499 + T-2725), T-3500 airport hover (d96e8f7d), T-3503 bubbles with trips between places (d5ea0ded), T-3069 tram snap + graph v2 (079d9c8b; the 120 v2 graphs copied into data/packs/realgraph, --check all pass). Main WASM rebuilt.
- 8800 = 7653f19d until the rebuild started 11:10 (snap8800.py, log in maps-44 scratchpad snapbuild.log). data/packs is free now (no run owns it).
- 32 zones lacked terrain after the night (Turkey 3, Colombia 29: TerrainMissing on a line build); built 11:00 with zone_terrain.py build (run it without -I). build_inputs now 0 flags.
- Before v0.4.0 ships: shonei-server deployed with save level 19 (T-3395), translations for all twelve languages. Still: freight tiles to R2 with the publish (T-3464; hover "Also freight and abandoned rail, trams, narrow gauge, heritage lines and unused track" + changelog item then), real graphs (now v2) reach R2 via deploy.py.
- Uncommitted on purpose: TODO.md (hers), notes/T-3397.md, jordan-save-investigation-2026-11-15.md, tools/ops/archive_bad_saves.py, changelog/roadmap.json (board-written).
- Board inbox watch: Monitor on `venv/Scripts/python.exe -u trainworld/board/watch.py` (30 min, re-arm). Up next queue: `tasks.py upnext list|add|remove`.
- Resource check: due today (also look at the T-3395 alert mail).
- Landing gotchas: agent-tree land refuses root files (serve.py); changelog ids in trees collide with main (drop the tree's next.json, re-add in main); agents that edit TASKS/DECISIONS/ARCHIVE in their tree: revert there, redo in main; never `remove --force` a tree whose land stopped on a conflict; after a sim/ landing the main tree's WASM rebuilds itself.

## Runs
- Coastal scan (T-3455): started by the old nightqueue.ps1 (pid 30892, ledger "nightscan", 3 cores), read-only, still running at 6 h; output scratchpad nightprep\scan\flagged.json, copied to artifacts/trainworld/t3455-flagged.json when it ends. Its rebuilds are next night's list.

## Agents
- None running.

## Next
1. Night follow-ups (one Sonnet/Opus agent): T-3399/T-617 morning steps per notes/T-3399.md (batch report; parks clo, mbm, led, sck, tsx, crj, auo, dnl, rta and t617's: relax and rerun, join); night-queue morning list in notes/night-queue-2026-11-17.md (changelog items for Istanbul and Colombia, drop arv names from app/src/i18n/place-names-zones.json, DECISIONS: Altinova to Yalova, Sile stays, Bucaramanga 1.51M over the cap; checks: stale, zone_origins, state_gaps TUR/COL, split_zones, the real save); close T-3488, T-3499, T-2725 (and T-3399/T-617 when parks are done); delete backups pre-T3488, pre-T3499, T3399 once clean. Before rerunning parks, check the gates don't read figures T-3503 changed.
2. T-3512 second round (fresh Opus; settle's tree .tw-trees/settle on 286ea890, uncommitted): drop or default-off stop-on-whole, rebase onto master (T-3503 touched the same files), re-run bubfix's Mettmann flicker repro on the combined code, the hang sequence 5+ times both modes, one unpinned run past a re-level; then a checker; re-add changelog "changing a line's trains at medium or low demand reached its final riders in 7 to 20 seconds on the new york save, where it took about a minute and a half" (performance). Results so far: NYC low 102 -> 6.6 s, medium 103 -> 20 s, every level 106 -> 57 s, high unchanged (T-3513).
3. T-3505 0-trip bubbles / fold tiny sites (builds on T-3503's code, now on master).
4. T-3511 enclave shifts/merges as night stages (after the scan's flagged list is known; a night queue like nightprep's).
5. T-3492, T-2136, T-3388 (after T-3512), T-1080, T-930, T-2627/T-2777, T-3480; T-3495 freight snap after v0.4.0.

## Waiting on Anita
- Told in chat, nothing blocking: T-3503 sum rule and Mettmann +57%; T-3503 memory (+31 MB page on NYC, cache cap 48 MB); train track now snaps to monorail/funicular/narrow gauge/heritage; enclave calls La Plata/Chascomus and Sanliurfa/Siverek; a checker killed an unrelated Codex node process (PID 5188) at ~09:05 UTC; snap icons to look at (artifacts/trainworld/review/snap_icons.png).

## Her standing asks this week
- Memory: pause once there are no easy gains; no core-balance shifting.
- Small isolated fixes, open-task count down, merge duplicates; bug investigations give up if hard.
- No asking about changelog lines: she reviews them on the board. No changelog items about the changelog itself.
- Overnight: keep rebuilding while queued builds remain.
- No windows on her screen: background processes hidden.
- Every publish, deploy and server push needs her yes to that exact action; none overnight.
- Advisor trial: start successors with `next_orchestrator.py trainworld --advisor fable`; log each call in handoff-log.
