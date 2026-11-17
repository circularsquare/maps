# Orchestrator handoff, trainworld

Orchestrator: session 00d74e61-e552-47a6-ba40-ee320d38eba1, name not shown by ListAgents (message by session id), since 2026-11-16 20:41 UTC
Updated: 2026-11-17 03:52 UTC (00d74e61 at ~560k tokens: starting its successor; Anita asleep from 03:45)

A short state page, rewritten (not appended) as things change. The diary is notes/handoff-log.md; older detail is in
git history. Read trainworld/CLAUDE.md and docs/orchestrator.md first.

## State
- Live = v0.3.5 (maps bfe58c54, published 11-15). The next publish is v0.4.0 (Anita: big, with Central America and Detailed mode scheduling).
- 8800 = origin = 7653f19d (02:30). master is ahead and unpushed: eleven-language translations, Indonesian, task graph (+ serve.py timer), twstats, bookkeeping. Next: push, then rebuild 8800 with `snapbuild-frozen.cmd <hash> <freeze>` (the T-3399 run owns data/packs).
- Before v0.4.0 ships: done now: shonei-server deployed with save level 19 (T-3395), translations for all twelve languages. Still: freight tiles to R2 with the publish (T-3464; hover "Also freight and abandoned rail, trams, narrow gauge, heritage lines and unused track" + changelog item then), real graphs reach R2 via deploy.py.
- Uncommitted on purpose: TODO.md (hers), notes/T-3397.md, jordan-save-investigation-2026-11-15.md, tools/ops/archive_bad_saves.py, changelog/roadmap.json (board-written).
- Board inbox watch: Monitor on `venv/Scripts/python.exe -u trainworld/board/watch.py` (30 min, re-arm). Up next queue: `tasks.py upnext list|add|remove`. Task history graph: serve.py timer runs board/history.py hourly.
- Resource check: next due 11-17 (also look at the T-3395 alert mail; the server re-checks every rev over ~4 nights at ~1 GB/night).
- This session (00d74e61) is a child session: not messageable, transcripts not saved, its agents can't be resumed once done. The launcher now fixes that for successors.
- Landing gotchas: agent-tree land refuses root files (serve.py): patch over by hand with cmd /c git diff; changelog ids in trees collide with main: drop the tree's next.json and re-add with changelog.py in main; tree names lowercase; agents that edit TASKS/DECISIONS/ARCHIVE in their tree: revert there, redo in main; never `remove --force` a tree whose land stopped on a conflict (recover the WIP commit by cherry-pick); after a sim/ landing rebuild the main tree's WASM.

## Runs overnight
- T-3399 Central America + Venezuela then Lisbon/Glasgow/Porto (T-617): tonight.ps1 pid 35248, 8 cores, started 02:36 UTC, log artifacts/trainworld/t3399-tonight.log; plan in notes/T-3399.md (morning: batch report, relax parks, rerun and join them; checks listed there). Stop: trainworld/data/work/world_run/STOP then the PID.
- Night queue: nightqueue.ps1 pid 30892, log artifacts/trainworld/night-queue.log, plan notes/night-queue-2026-11-17.md: T-3455 coastal scan beside (waits for 3 cores), then after T-3399 T-3488 Istanbul (arv into ist, Karamursel to Kocaeli, Altinova to Yalova), T-3499 Bucaramanga + T-2725 Colombia census rows, then up to 2 flagged coastal groups; stops by 10:00 Anita's time (15:00 UTC). Scripts in 00d74e61's scratchpad nightprep\. Morning: changelog items for Istanbul and Colombia, drop arv names from place-names-zones.json, record its calls in DECISIONS (Altinova to Yalova, Sile stays, Bucaramanga 1.51M over the cap), delete backups once clean.
- freightbuild (Sonnet agent of 00d74e61, main tree): T-3464 freight/abandoned tiles, all regions done but the US (building after a 12 GB download). Output noritetsu/dist/data <cc>_extra.pmtiles + extra.json; no R2 upload.

## Agents
- prspeed (Opus, 00d74e61's, .tw-trees/prspeed, 6187): T-3473 PR 12 CPU speedups, measured; commit needs Co-authored-by CinnamonVII (name/email in its report); needs a checker (sim). If 00d74e61 is gone when it reports, read its tree and notes/T-3473.md.
- T-3500 airport hover fix built in .tw-trees/airhover (uncommitted; Istanbul hover 4,327 -> 111,287 trips; figures identical vs HEAD on 3 saves; smoke passed). Needs a Sonnet checker (sim change) once cores free, then land; drop the tree's changelog and re-add "fixed airport hovers and airport bubbles leaving out trips from across the zone line (istanbul airport showed 4,300 trips a day, now about 111,000)" (demand --fix); rebuild main WASM after.
- Old trees to clear when sure: shenzhen, tt2, ttstyle (superseded); safari-save-035 (release branch: keep unless she says). Stray empty folders in .tw-trees (c1, ferry45, fixes, hotfix031-target, hover, touchrule, freight): not worktrees; may hold node_modules junctions, so use agent-tree.mjs or check before deleting.

## Done 11-16 evening (for context)
- Memory round (her P1): desktop outlines compact, worker share 17.5% + grow back (T-1366), GPU buffers (T-3497), corridor table on page 32/16-bit (T-3493) and in workers (T-2496 stage 1); paused at no easy gains per her. Leftovers in notes/T-3160.md, T-3497.md, T-2496.md. T-3373 stage B (T-1560) is the next real cut if memory resumes.
- Memory readout 8 GB+ (T-3494), Use more cores hover (T-3496), globe straight lines fixed (T-3481), unused-rail snap with all graphs installed (T-3495 stage 1), board Up next column (T-3490) and task graph (T-3504), twstats (T-3508), gate fixes (T-2298, T-2399), Indonesian (T-3398) and the eleven-language batch, shonei-server T-3395 deployed.
- Open-source: PR 7, 12, 13 replied and closed, PR 1 replied (open until T-3480 ships), issues triaged (T-3484 done: HK border kept 0.05; T-3485 done: airports; T-3486/3487 closed). PR 16 reply not drafted (notes/pr-survey-2026-11-16.md).

## Waiting on Anita
- Nothing open. Asks file empty.

## Her standing asks this week
- Memory: pause once there are no easy gains; no core-balance shifting.
- Small isolated fixes, open-task count down, merge duplicates; bug investigations give up if hard.
- No asking about changelog lines: she reviews them on the board. No changelog items about the changelog itself.
- Overnight: keep rebuilding while queued builds remain ("if we finish with anything we should keep going with further rebuilds").
- No windows on her screen: background processes hidden.
- Every publish, deploy and server push needs her yes to that exact action; none overnight.

## Next (Up next queue on the board, in order)
- T-3503 demand bubbles include trips between cities, always on, adding up to local + intercity (3-5 h, Opus).
- T-3506 worldwide enclave-zone check; T-3505 0-trip bubbles / fold tiny sites (with T-3503's code).
- T-3492 idle worker memory, T-2136 corridor table off the page, T-3388 London edits, T-3069 trams on real rail (with T-3495's trams/heritage/narrow gauge), T-1080, T-930, T-2627/T-2777 world tooling, T-3480 bus one-way setting + PR 12 debug check.
- T-3495 later stage: freight/abandoned snap graph once T-3464 tiles exist (plan in notes/T-3495.md).
- T-3507 task graph counts only real closes (small).
- Advisor trial: start successors with `next_orchestrator.py trainworld --advisor fable`; log each call in handoff-log.
