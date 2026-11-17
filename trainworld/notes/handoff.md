# Orchestrator handoff, trainworld

Orchestrator: session 83439634-6553-4808-8acd-c1aa1af76a63, name maps-b8, since 2026-11-17 11:13 UTC
Updated: 2026-11-17 14:40 UTC (maps-b8 past 400k: starting its successor; its agents report to maps-b8, which relays them verbatim while its tab is open)

A short state page, rewritten (not appended) as things change. The diary is notes/handoff-log.md; older detail is in
git history. Read trainworld/CLAUDE.md and docs/orchestrator.md first. (Times of her messages below are approximate:
maps-b8's earlier notes stamped some 1-2 h late.)

## State
- Live = v0.3.5 (maps bfe58c54, published 11-15). Next publish v0.4.0: NO deadline (Anita: "lets please forget about the whole tonight deadline thing"); she picks when, likely evening US east, and keeps track of the landing cut-off herself ("you guys dont have to worry about landing new work"): land things as they pass.
- master 5effe9c7, pushed; 8800 = 5effe9c7. Master smoke passed at 85488bdd (29/29, terrain on); trainsfan and T-3510 landed after with their own smokes.
- In master since v0.3.5 (for the v0.4.0 notes): Central America + Venezuela (91/95 rows), Lisbon/Glasgow/Porto demand gaps, Istanbul and Colombia redraws, ~500 MB less memory on desktop, 240-line networks load about twice as fast, bubbles under 1 trip fold (T-3505), bubble purpose rows sum to the total (T-3510), buses obey one-way streets setting (T-3480, co-authored with Renkon), tram snap (T-3069), terrain on by default (UI store v15 drops a stored off once), trainsfan achievement (T-3520, 100 h played, counted per browser from zero), ferries 45 km/h (T-3435), join market step tooling (T-2627/T-2777).
- Changelog tense rule (CLAUDE.md, her call): past for the change, present for how things are now; c066/c068/c070-c072/c086 reworded; c086 "the game uses about 500 MB less memory on desktop" is the headline group (terrainmem is checking that figure with terrain on).
- Before v0.4.0 ships: translation batch (settingsTab.busOneWay, T-3510's two strings, trainsfan's 4 keys, the timetable window's strings once weekend ungates it, updates.v040* keys); `changelog.py line` + `publish --version v0.4.0`; freight tiles to R2 + hover + item (T-3464); real graphs v2 via deploy.py; smoke8800 light and dark; deploy.py; website push; open-source export sync + push; then post Renkon's PR 1 and issue 8 replies (her OK given; text in handoff-log under 2026-11-17 maps-b8) and close PR 1 with a link. Server: one deploy just before the publish carrying 7a23cbc (playtime cap 250,000) plus piecelimit's and weekend's save-check changes if they land; T-3395's level-19 check looks deployed (service restarted ~01:35 UTC after 3e8e501), unverified. Each publish/deploy/push needs her yes.
- Uncommitted on purpose: TODO.md (hers), notes/T-3397.md, jordan-save-investigation-2026-11-15.md, tools/ops/archive_bad_saves.py, changelog/roadmap.json (board-written).
- Board inbox watch: Monitor on `venv/Scripts/python.exe -u trainworld/board/watch.py` (30 min, re-arm). Up next queue: `tasks.py upnext list|add|remove`.
- Resource check done 11:14 UTC (told her; she said leave Resend on the free plan). shonei-server 7a23cbc committed, not pushed or deployed.
- Player requests: fetched and sorted 13:20 (reqsort; g078-g082 new on the board).
- Landing gotchas: agent-tree land refuses root files; agents add changelog items in main; never `remove --force` a tree whose land stopped on a conflict; after a sim/ landing the main tree's WASM rebuilds itself. agent-tree land writes one -m plus Claude's trailer, so a contributor co-author landing is done by hand (T-3480's way: add the tree's files, commit -F with both trailers, rebase onto master, `git merge --ff-only tw/<name>`, then `agent-tree.mjs remove`).

## Runs
- Coastal scan (T-3455): pid 30892 (ledger "nightscan", 3 cores), read-only, 9.4 h at 14:37 and still writing w0-w2.jsonl in the 00d74e61 scratchpad's nightprep\scan; output flagged.json there, copy to artifacts/trainworld/t3455-flagged.json when it ends. Its rebuilds are next night's list (with T-3511).
- Ledger at 14:37: full (14 of 14): nightscan 3, settle2 3, weekend 3, piecelimit 4, busboard 1.

## Agents (all maps-b8's subagents except settle2; reports come to maps-b8 and are relayed. If maps-b8's tab is gone, check each directly: its tree's git status, notes, the ledger)
- settle2 (Opus, maps-44's subagent, .tw-trees/settle, port 6154): T-3512 round 2 (train count edits settle 7-20 s instead of ~90 s at low/medium). Report reaches maps-44, which relays to maps-b8. Check: the tree's notes/T-3512.md, `git -C .tw-trees/settle log`. Then a fresh checker, land, re-add changelog "changing a line's trains at medium or low demand reached its final riders in 7 to 20 seconds on the new york save, where it took about a minute and a half" (performance). Its notes: an unfiled small bug (an edit right after a load, then undo, undoes the cap lowering).
- weekend (Opus, .tw-trees/weekend, port 6168): T-3521 weekend train counts, in v0.4.0 when ready. Her design: in the timetable window (the whole window ships; remove the ?timetables gate; stage 2's local-before-express glitch is acceptable to her); Weekdays and Weekends sections with count/headway per demand level; beside the Weekends header a percent stepper (default 100%, 10-200% in 5%) and an "apply to all lines" button with a confirmation; a percent line's weekends follow weekday edits (counts rounded up, headway divided and rounded down to a minute); a hand-edited weekend makes the percent show greyed; holidays run weekends; one weekend. Sim from GabrielMoraez's PR 13 (closed). Save level 21 (piecelimit 20); I (the orchestrator) add its block to shonei-server's save check after piecelimit's edit. Then a checker on the merged code (after piecelimit/stationedit land), she sees it, commit by hand with Gabriel's co-author trailer (name/email in its report).
- piecelimit (Opus, .tw-trees/piecelimit, port 6164): T-3405 per-piece speed limit in the track piece panel (her yes). Save level 20; it commits the shonei-server save-check change (no push). Then a checker. Server change must deploy before a game version writing level 20+ ships.
- stationedit (Sonnet, .tw-trees/stationedit, port 6163): T-2767 + T-2768, station place/delete on built track rebuild only what they touch, figures to the bit vs master. Then a checker.
- busboard (Sonnet, .tw-trees/busboard, port 6169): T-3522 bus boarding cost 6 -> 4 perceived minutes (her call; no ceiling tests, she says it doesn't touch rail), riders/profit before and after on a save or two. Demand number change: a light checker or her look at the table before landing.
- shore (Sonnet, .tw-trees/shore, port 6173): T-3524 track near the shore counted as over water (her pad); forgiving margin preferred if simple; no pack rebuilds.
- terrainmem (Sonnet, read-only, port 6171): memory with terrain on vs off, desktop and phone, to notes/terrain-memory-2026-11-17.md; decides whether the 500 MB headline holds.

## Next
1. Land each agent as it passes (checker for sim/save/demand/money). Rebuild 8800 after each batch; push master.
2. When she says release: the v0.4.0 steps under State, each on her yes.
3. Performance, non-memory only: T-3388 (London edits ~4 s) after T-3512. T-1080, T-930 (world tooling). T-3511 after the scan. T-3495 freight snap after v0.4.0. Timetables: stage 2 fix (tt2, stale) and Detailed mode rows later.
4. Leftover non-git folders in .tw-trees (c1, ferry45, fixes, freight, hotfix031-target, hover, touchrule) and stale trees (safari-save-035, shenzhen, tt2 with 50 dirty files, ttstyle, trams): look before removing; never delete by hand (node_modules junction). A hand delete of old review copies was refused once: leave them.

## Waiting on Anita
- asks.json: T-3523 open-source repo (A: sync on every push + a public in-progress list, ~2-3 h, my pick; B: public repo as the source, 1-2 days).
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
