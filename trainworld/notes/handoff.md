# Orchestrator handoff, trainworld

Orchestrator: session 49d5e3c4-3205-457a-b910-06fb8268277a, name maps-76, since 2026-11-18 19:20 UTC
Updated: 2026-11-18 20:40 UTC

Live state: `venv\Scripts\python.exe trainworld/tools/handoff.py status` (T-3667). This file is plan and her words.
Diary: notes/handoff-log.md. Her local time is UTC-5.

## Next
-2. 21:20 UTC: landed since: T-3727, T-3728, T-3657 (c056 starred). Checkers running: exitchk (T-3722) and
   routechk (T-3724, tree railnone: intercity "no route" with riders, cause found on a player's save; sim, WASM
   rebuild at landing; in v0.4.2 if it passes). Filed T-3730 (same loop in trip.rs). She is editing the board.
-1. 21:05 UTC: everything for v0.4.2 so far is on 8800 (75f0b3bf, master pushed, smoke passed). Waiting on her:
   board changelog OK, text acks (consolidated list sent). Asks answered 20:55: profitpts in v0.4.2 on PASS; T-3726
   after release; hover row trimmed to the icon (hovericon, Sonnet, T-3727, for v0.4.2); shonei-server push: her
   yes came via the board and the permission check refused, asked her in chat. Checkers running: exitchk (T-3722,
   tree exitmenu) and ptschk (T-3657, tree profitpts). oneway T-3711 done in tree
   oneway: NOT for v0.4.2 (save LEVEL 22, new block LOS1; server savecheck twSaveLevel=21 marks such saves
   unverified, so update shonei-server savecheck before it ships); needs a checker; its DECISIONS line is in its
   notes (add at landing); her question: shift on a stop-list row already means "end the line here".
   Then: changelog publish, one translation pass (all stale/new keys + updates.v042*), smoke8800, new_zones_map,
   deploy on her yes.
0. PUBLISH v0.4.2 (her ask ~20:15 UTC: "more people complaining about bus/ferry schedules so i think its pretty high
   prio to get a fix out ... along with whatever safe stuff is ready to go in. as usual ill want to review changelog
   before it goes out"; 20:35: v0.4.2 OK, "dont worry about time. its not a hard deadline"). Rough aim: deploy
   ~23:15 UTC. Land what passes its checker; profitpts and oneway only if fully checked. Then: translation batch for
   new/stale keys (docs/localization.md passes), snap8800, smoke8800 light+dark, new_zones_map.py, her board edit
   of the changelog (draft line and 7 highlights set), changelog publish, deploy.py on her yes.
   Server DEPLOYED 20:12 UTC (12d58a0, 662ed86, c960adc) on her yes; shonei-server not pushed (asked).
   After publish: `latched.py --wrong`, check each against its save, unlatch (her yes 20:20 UTC: "ideally we unlatch").
   Landed for v0.4.2: T-3703 46fe6cec, T-3717 07fac734, T-3690 f06dbf47 (WASM rebuilt), T-3714 3aa71234.
   "by hand" -> "manually" in en.json (2 strings, uncommitted, mine) and c036; rule in CLAUDE.md.
1. Running (reports reach you through the inbox or directly; act on each, then land):
   - busched T-3703 landed 46fe6cec (checker PASS; c052 highlighted). Filed T-3719 (sandbox switch lowers counts):
     start after sandboxfix lands (same area). Server save check ignores the new header key (no newHeadway in it).
   - endgap T-3690 done in tree endgap (click slides instead of refusal); checker endchk running.
   - nohalo (Opus) T-3714, P1 for v0.4.2: no halos on the map, count icons in the placing tag, no cursor icon,
     "Quad"; rewords c046. If late, ship the removals only.
   - searchx (Sonnet) T-3715 (search clear button visible when typed in) + T-3716 (Newest tab second).
   - farecap T-3689 done in tree farecap (archived); checker capchk running; land for v0.4.2. Hover changed:
     "Paid each time a trip changes lines." -> "... At most the base fare." farecap killed profitpts' Vite on 6182
     by mistake and restarted one there (PID 32480, profitpts tree); if profitpts reports errors, that's why.
   - Her board edits 20:30 UTC: line "update v0.4.2: bus and ferry schedules, bug fixes"; c047 reworded to my fix of
     her wording (150 MB, "with intercity trips on", not "large saves": the table was world-wide on any save).
     T-3723 closed; then she reworded 3 intercity no-route messages (done by me in en.json, uncommitted:
     noRail other, noTrain, noAlt). railnone (Opus) T-3724: why "no route" shows on joined pairs (top complaint). Order now: land farecap, one translation batch after changelog publish (all new
     and stale keys plus updates.v042*), snap8800, smoke8800, deploy on her yes. Stage 2 T-3718 (trips start with a walk; her yes) after the release, Opus.
   - sandboxfix T-3687 done in tree sandboxfix (score send during a load latched the old game); checker sandchk
     running; filed T-3720. timekeys T-3717 landed 07fac734 (empty locked folder .tw-trees/timekeys left). searchx landed 35275e92 (T-3715; T-3716 needed nothing: Newest is already second).
   - oneway (Opus) T-3711: shift-click a stop to make it one way; reports options if blocked.
   - profitpts (Opus, maps-56's) T-3657.
   - Landed this session: T-3704 ec6c5602, T-3700 f087e6d2, T-2136 79a75030; 8800 = e16ba440, master pushed.
2. Up next (start as files free): T-2880 (market code), T-3495, T-2054, T-2901, T-2380, T-3354, T-3525, T-3584,
   T-3518, T-2598, T-3461; filed today T-3709, T-3710, T-3712, T-3713.
3. T-3395: close after a day of clean daily scans (deployed 11-18 13:30 UTC).

## Her words
- 11-18: close easy high-priority wins and old tasks, small over design-heavy; "prio should take into account
  easiness as well as impact" (notes/triage-2026-11-18-{a,b}.md, todo-sweep-2026-11-18.md).
- Track look (T-3714): "lets just get rid of the halos on the map entirely. i tihnk its a significant vibe change
  that i dont want to risk". Bus/ferry schedules "should work the same way as trains" (T-3703 calls OK).
- Points scale with each line's cost recovery capped at 1: "if fares cover 50% of expenses they should count as 50%
  points"; measure first (T-3657).
- Mod support (T-3377) serves hand-coders and assistant users alike; never say publicly it's aimed at vibe-coders.
- Venezuela intercity demand: "fine tuning we can do later" (T-3706 P4). Caribbean+: later, she'll say.
- caelum418's speed PRs (T-3659): "look at the next time we look at this area of performance".
- Standing this week: small isolated fixes, count down, merge duplicates; she edits each update's changelog on the
  board before publish (highlights over ~20 items); lean testing, smoke per batch; one phrase per wait; long jobs in
  the background, answer her first; every publish, deploy and push needs her yes to that action, none overnight;
  advisor trial (successors with `next_orchestrator.py trainworld --advisor fable`, log calls in handoff-log).

## Waiting on her (asks.py list has the open ones)
- Asks open: push shonei-server; c047 "200 MB" wording (I proposed ~150 MB, whole-process figure).
- Running for her: exitmenu (Opus) T-3722 exit-to-menu missing; noroute (Sonnet) T-3723 review page
  artifacts/trainworld/review/t3723-no-route.md (she reviews each no-route message herself; she never saw them).
  T-3721 (Copy label, server deploy) after the release.
- Text acknowledged 21:05 UTC ("other strings look good"), except "driven by a parent" (explained to her, waiting)
  and "Search settings" (being replaced by an icon, T-3728, setsearch2 running). Points rule line not yet shown.
- She sends the T-3635 player reply herself (draft in notes/T-3635.md). Ask natteru (Discord) about Firefox's
  resist-fingerprinting if she likes (T-3705).

## Only a person knows
- Uncommitted on purpose: TODO.md (hers), notes/T-3397.md, jordan-save-investigation-2026-11-15.md,
  tools/ops/archive_bad_saves.py, changelog/roadmap.json; maps .claude/settings.json (her edits + T-3670 keys).
- Old trees shenzhen (dirty pipeline/osm_units.py, owner unknown) and tt2 (stale, stray cores.json): ask before
  removing. Empty locked folder .tw-trees/touchrule: remove once whatever holds it exits.
- Never delete a tree folder by hand: its app/node_modules junction deletes the main tree's (npm install from
  trainworld/app restores it).
