# Task sweep, 2026-11-28

Asked 11-27 (Anita): "the number of tasks we have is getting quite high. what are some easy ones we could knock out?"
Before: 481 open (12 P1, 70 P2, 293 P3, 106 P4 that don't count as open). After: 476 (four closed here; one more went while the sweep ran, not by me). Bookkeeping only: no code run, nothing committed.

## What the sweep found

Few tasks are stale. Almost every open task is a follow-up filed with its cause still unknown, a stage of a larger piece (track rework 38 stages, other modes 20, edit speed 40), or a data fix that needs a world rebuild. About 100 ids were checked against ARCHIVE.md, git log and the code; four closed, and the rest are open for a stated reason (below). There are also few quick wins: 12 tasks an agent can finish in about an hour, and 5 more that run to about 2 hours.

## 1. Closed (4)

| Task | Reason |
|---|---|
| T-3697 | Done: t2289, t2407 and t2162 (plus t1201 and t2626) are retired to checks/old/ and checks/INDEX.md is updated. The move sits uncommitted in the main tree (git shows the deletes and the new files under checks/old/), so it lands with the orchestrator's next commit. |
| T-3943 | Done by T-4400 and T-4406: Settings has one Show block with a toggle per mode and one Show all; a hidden mode's lines, vehicles, track and unshared stations leave the map and schematic, and hidden levels do too. |
| T-3345 | The umbrella is overtaken: its three named parts are done or closed (T-3360 buses and ferries, T-2724 continent scale closed on her word, T-3358 rivers, option E built). What is left of the schematic stands as T-3453, T-3872 and T-3947. |
| T-3354 | Not reproduced since 11-16 (a station node's heading stays put on a drag), no report since, and the station tool rework (T-3963) replaces that placement code. T-3963's "Closes T-3354" was dropped from its line. Reopen with the exact drag if it shows again. |

## 2. Looked at and kept

Each of these looked stale and is not. Left open on purpose, so nobody re-checks them.

- **Waiting on a reading only she or a server can give:** T-3738 (the week of Worker request counts is only in her Cloudflare dashboard), T-3395 (the scan reports; may need the server, see the easy list), T-4501 (she says on 8800 whether a selected city loads fast enough).
- **Not reproduced, but parked on purpose:** T-4373, T-344 (both P4: reopen if another player reports it).
- **Umbrellas with open stages:** T-3444 (track rework), T-3346, T-3348, T-4097, T-3245, T-1344, T-1345.
- **Duplicates not folded.** The newer task says it closes the older one when it lands, and the stages are hers, so they stay as filed: T-3335 into T-3970, T-3079 into T-3980 and T-3981, T-3231 into T-3264. T-3836 overlaps T-2586 but is a design call for her.
- **Folded tasks the archive shows are fine:** T-2589, T-2498, T-2242, T-2755, T-2070 (the read-ahead is still unchecked on a real multi-region run; Brazil and Russia ran as one region).
- **Code still has what the task wants dropped:** T-4042 (the compose-afresh path is still in demand.worker.ts line 2757 and demandProtocol.ts line 240).
- **Fixed long ago, but the task text names a part not yet done:** T-1765 (its "61 county-town zones" count may be stale; China's 11 city zones and Canada, Greece and Poland were rebuilt), T-3011 (downtown is in the zone since T-3671; the Nile channels' part is her call).
- **P4 and "someday" tasks were not closed.** They don't count as open (her 11-16 rule). T-685 (stamps) and T-873 (train empties mid-line) may already be done; not checked.

## 3. Task tags that no longer hold (for the orchestrator, not edited)

- `{after:T-4444}` on T-4504 and T-4445 (T-4444 is archived). T-4504 is planbest's; T-4445 can start now. T-4451 to T-4454 still wait on T-4445 and T-4446.
- `{after:T-3959}` on T-4030, `{after:T-3962}` on T-3978 and T-3980, `{after:T-4114}` on T-4142, `{after:T-2486}` on T-2374, `{after:T-3873}` half of T-3875 (T-2374 still open): the named task is archived.
- `{after:the 0.4.6 publish}` on T-4116 and T-4037: 0.4.7.1 is out.

## 4. Easy ones, ranked by player value per hour

Held by running agents, so skipped on purpose: planbest (T-4504, anything in sim/src/capacity.rs or overtake.rs, T-3894, T-4021), realgaps (T-4514, docs/demand-*.md, T-3978, T-2879, T-1629), pagestage5 (T-4284), claude2 (T-4424). The edit-speed tasks (T-4429, T-4445, T-4456 and the rest of the T-4267 stages) are skipped because each needs a bench run and the edit-speed agents are still on that code.

Budget rows are from CLAUDE.md's table. "Sonnet" is for a brief that settles the open questions; the notes under each task give them. No task here needs a call from Anita. Where a string is new, a draft is given for her "text acknowledged".

### About an hour or less

1. **T-3675 [P3]** Feedback button on the graphics error screens. The no-WebGL2 and graphics-lost screens have no way to report; the failed-load screen right below them already has one.
   - Files: app/src/ui/LoadingScreen.tsx (the early-return block for graphicsGone, noWebgl2, noSimd); the button is the existing FeedbackOpen. Check startFeedbackUi is mounted when the no-WebGL2 case fires early in app/src/main.ts.
   - Budget: UI, text, CSS (typecheck and 1 screenshot). About 0.5 h. Sonnet.
   - Strings: none new; reuses feedbackPanel.open ("Submit feedback"), listed as reused.
2. **T-4327 [P3]** Key hints in ja, zh-Hans and zh-TW hovers. The hint styling only matches ASCII brackets, so the full-width ones ("（U）") are not lettered in the accent and don't go away on phones (25, 16 and 16 hovers).
   - Files: app/src/ui/tip.ts (the KEY_HINT regex, KEY_HINT_GONE, and appendWithKeys, which hard-codes "(" and ")").
   - Budget: UI, text, CSS. About 0.5-0.75 h. Sonnet. Strings: none.
3. **T-4049 [P3]** Cities served counts only busy cities. Her note: about 1,000 rides.
   - Files: app/src/game/achievements.ts (netFigures, metNow; demandFiguresOf already has the zone views and train rides). Move the cities count into demandFiguresOf, with a new CITY_MIN_RIDES constant.
   - Budget: App logic (typecheck and 1 game run). About 1-1.5 h. Sonnet. Strings: none (the achievement text stays).
   - Call to record in DECISIONS.md: read "about 1,000 rides" as per day, like the other thresholds. The count is then known only once demand is final, as the trips tier is.
4. **T-4356 [P3]** Ramp speed labels while dragging a node and in the blueprint box.
   - Files: app/src/map/nodeDrag.ts (line 247) and app/src/map/blueprintBox.ts (line 156): add rampLabels(pv.ramps) to the slowestFirst call, as app/src/map/edgeEdit.ts does for the drawing tool. rampLabels lives in trackTagRows.ts.
   - Budget: UI, text, CSS. About 0.5 h. Sonnet. Strings: none.
   - Left out: the idle selected stretch has no ramp data in its shape (1-2 h more); file it apart.
5. **T-4371 [P3]** Say what blueprints are. A Japanese player doesn't know what the Settings checkbox refers to.
   - Files: app/src/ui/SettingsTab.tsx (the Check around settingsTab.blueprints gets a tip, which Check in app/src/ui/widgets.tsx already supports); new key settingsTab.blueprintsTip in app/src/i18n/messages/en.json and context.json, then the other languages as docs/localization.md says. Optionally the same tip on toolBar.selectBlueprints in ToolBar.tsx.
   - Budget: UI, text, CSS. About 1 h (translations may take longer than the code). Sonnet.
   - String draft (new): "planned track, stations and lines you haven't built yet". Check it against how the game names a blueprint elsewhere before it goes in.
6. **T-3395 [P1]** Close once the save scan runs clean. An agent reads the scan reports since 11-20 and closes the task if no new damage shows; if the two damaged rev-1 games still read damaged, it records them as old damage.
   - Files: none to change; notes/T-3395.md ("Scan finding of 2026-11-19"), tools/ops/archive_bad_saves.py. Budget: tooling, no game run. About 0.5 h. Sonnet.
   - Catch: the notes say the reports were not readable from here (ssh or a new route). If so the agent stops and says what it needs. A deploy to add the damage reason to twScanItem.Note waits for her word.
7. **T-4509 [P3]** A standing check that zooms in and out between edits (T-4508's two bugs went unseen for two days).
   - Files: new checks/t4509-zoom-edits.mjs, modeled on checks/t1509-line-moves.mjs and checks/lib/play.mjs; one row in checks/INDEX.md. Runs the two sequences in notes/T-4508.md "Game run" with ?patch=check and fails on any "T-4262 check" or other console error.
   - Budget: this is a standing check for a bug that really happened (allowed by the budget table). About 1-1.5 h. Sonnet. Strings: none.
8. **T-4198 [P3]** Changelog "edited" mark only for her edits. changelog.py reword sets the same mark as her board edits, so agents' rewords (26 on one day) look like hers and get left alone.
   - Files: tools/changelog.py (update_item near line 201, the reword branch near 449, the edited field near 215); the board code that draws the mark (find it first; probably behind serve.py). Give reword its own field so list still shows it.
   - Budget: tooling. About 1 h. Sonnet. Strings: none.
9. **T-2165 [P3]** Mark the two Japan saves with flat crossings (44 and 60) as separated, then rerun the Japan ridership check.
   - Files: notes/T-2064-mark-sep.mjs (already built, run on each save), checks/t2088-ridership.mjs (rerun jpn7), notes/T-2064.md (figures).
   - Budget: tooling, no game run beyond the check. About 0.75 h. Sonnet. Strings: none.
   - Catch: data/work/realnet/jpn7_real.save may be gone; then it needs a rebuild and is no longer easy.
10. **T-4166 [P2]** First bench run on her own AWS account. About 5 cents. Read docs/aws.md first; run through tools/aws/bench.py only.
    - Files: none changed; notes/T-4166.md gets the result and the cost. Budget: tooling. About 1 h, mostly waiting. Sonnet.
    - Catch: every launch on 11-24 was refused by the account's limit; the account is now at 1 vCPU. A refusal is a result: record the error.
11. **T-4096 [P3]** A daily sweep of old session scratch (Temp\claude was 78 GB). The standing rule already lets agents delete finished sessions' scratch 2+ days old.
    - Files: new tools/scratch_sweep.py (delete folders under %TEMP%\claude whose newest file is over 2 days old; skip any that can't be removed because a session holds it). Ship it with a dry-run mode.
    - Budget: tooling. About 1 h. Sonnet. Strings: none.
    - Catch: registering it as a daily Windows scheduled task changes her machine; leave the script run by hand, and ask the orchestrator before the schedule.
12. **T-3457 [P3]** First slice of slimming the standing checks: retire what checks/old/ and checks/INDEX.md already mark as retired (51 files plus the five just moved), after a grep that nothing references them. The full audit of the other 320 checks is 4-8 h, Opus, and stays open.
    - Files: checks/old/*, checks/INDEX.md. Budget: tooling. About 1 h. Sonnet. Strings: none.

### Worth doing, about 1.5-2.5 hours (a worked-out cause is not there yet)

13. **T-4502 [P2]** A selected city shows "To other places 0" and "No intercity trips are modeled here yet." until its market pack opens. Say it is calculating instead. Most player value of the longer ones.
    - Files: app/src/ui/inspector/CityInspector.tsx (market is intercityDemand.value once rowsOf returns rows); app/src/game/demand.ts (IntercityDemand and CorridorBook need a "this place's pack is in the market" flag, from demandClient.ts ensure or the market recompose).
    - Budget: App logic, 1 game run on one worker. About 1.5-2.5 h. Opus. Strings: none new (reuses cityInspector.calculating).
14. **T-4368 [P3]** A high-speed line made normal is still drawn high-speed. Cause unknown: either the cached stroke after lineFast or the edge's own hsr flag.
    - Files: app/src/map/network.ts (the c.fast !== l.fast check near line 2932, skipped when sameRow; flags near 2952), app/src/map/lineToolAhead.ts, app/src/map/aheadShow.ts.
    - Budget: App logic. 1-2 h. Opus. Strings: none.
15. **T-4229 [P3]** Station names stay after zooming in. One report, no repro.
    - Files: app/src/map/stationLabels.ts, app/src/map/labelNames.ts, app/src/map/moving.ts (throttled placement). Budget: UI, text, CSS. 1-2 h. Opus.
16. **T-4091 [P3]** Undoing a bus reroute leaves a stop where the reroute put it. Cause may be older than T-4069.
    - Files: sim/src/track/wasm/bus.rs (set_bus_route), the Op::Line undo in the sim's edit history. Start with a Rust test: reroute, undo, compare stop coordinates.
    - Budget: App logic with Rust tests (agent_cargo.py test). 1.5-2.5 h. Opus. Strings: none.
17. **T-4376 [P3]** No sound for days for one player. One report, no diagnostics: the likely causes are the Sound switch, volume 0 or a muted site, with an AudioContext that never resumes after a reload as the one code cause.
    - Files: app/src/map/audioOut.ts (open, audioOut, the idle-gated listeners), app/src/game/ui.ts. Retry ctx.resume() on later clicks is the candidate fix. May end with no cause found. 1-2 h. Opus.

## 5. Checks run on the way

- tasks.py check: 1 problem, none of mine: notes/handoff.md is 61 lines (keep it under 60). Everything else passes, 476 open tasks.
- Task lines edited: the four closes through tasks.py, plus T-3963's "Closes T-3354" dropped.
