# Orchestrator handoff, trainworld

Orchestrator: session 49d5e3c4-3205-457a-b910-06fb8268277a, name maps-76, since 2026-11-18 19:20 UTC
Updated: 2026-11-18 19:20 UTC

Live state: `venv\Scripts\python.exe trainworld/tools/handoff.py status` (T-3667). This file is plan and her words.
Diary: notes/handoff-log.md. Her local time is UTC-5.

## Next
1. Running (reports reach you through the inbox; act on each, then land):
   - Landed by maps-76: busfix T-3704 (ec6c5602), wayback T-3700 (f087e6d2, stale check only; filed T-3709 for
     t3537-tram-split), corridor T-2136 (79a75030). Smoke on master running 19:45 UTC; then snap8800 and push.
   - busched (Opus) T-3703 in tree busched: buses and ferries get trains' schedule controls (bus fix T-3704 landed
     ec6c5602, c050).
   - wayback (Opus) T-3700 in tree wayback; endgap (Opus) T-3690 in tree endgap; xferfare (Opus) T-3689,
     investigation in the main tree, notes only, proposal goes to her. Started 19:25 UTC.
   - profitpts (Opus) T-3657: measures cost recovery first; stops and reports if honest networks lose >20% points,
     else builds her rule (points per leg x min(1, line fare income / running cost)). Checker before landing.
   - sandboxfix (Opus) T-3687: normal games treated as sandbox (lead: v0.4.1 takes any 403 on the score route as the
     sandbox latch). Server fixes commit in shonei-server, not deployed. Checker if it touches saves.
2. After those land: one smoke on master, then snap8800 (8800 = 07cc78bd; since then landed modelook T-3693 look C,
   firefox T-3705, modmark T-3707, and whatever lands next).
3. Up next (start as files free): T-3703, T-2880 (after corridor lands: market code), T-3495, T-2054, T-2901,
   T-2380, T-3354, T-3525, T-3584, T-3518, T-2598, T-3461.
4. T-3395: close after a day of clean daily scans (deployed 11-18 13:30 UTC).

## Server deploy waiting (her word; she said "no rush")
- shonei-server unpushed and undeployed: 12d58a0 (T-3652 Newest tab), 662ed86 (T-3707 published builds + modded
  latch), plus anything sandboxfix commits. After that deploy the Newest tab shows. builds.json appears with the next
  game publish (deploy.py writes it); old v0.4.1 tabs' unstamped scores stay accepted until 48 h after the server
  first reads it. Either order works.

## Her words
- 11-18 16:25 UTC: close easy high-priority wins and old tasks, small over design-heavy, to get the open count down;
  "prio should take into account easiness as well as impact". Triage done (notes/triage-2026-11-18-{a,b}.md,
  todo-sweep-2026-11-18.md).
- Buses and ferries should get trains' schedule controls, "we can leave out the marey diagram / timetable part for
  now ... not necessarily super high prio but would be nice to get out in next few updates"; "both whats pictured
  and what i mentioned should be things we try to get consistent" (T-3703, T-3704).
- Modded games off the leaderboards "soonish ... p2" (done, T-3707). Points scale with each line's cost recovery
  capped at 1: "if fares cover 50% of expenses they should count as 50% points"; measure first (T-3657).
- Mod support (T-3377) serves hand-coders and assistant users alike; never say publicly it's aimed at vibe-coders.
- Venezuela intercity demand: "fine tuning we can do later" (T-3706 P4).
- No changelog item for a change that is only a tooltip's or alert's wording (now in CLAUDE.md).
- Nothing may pop windows on her screen (CLAUDE.md Shell rule; checks' spawns now hide windows).
- 11-18 ~15:20 UTC: headache, out walking, on remote control; short messages.
- Caribbean+: "we can worruy about caribbean+ later ... i will lyk". Free-fare cheese now T-3657 (above).
- caelum418's speed PRs (T-3659): "look at the next time we look at this area of performance".
- Standing this week: small isolated fixes, count down, merge duplicates; she edits each update's changelog on the
  board before publish (highlights over ~20 items); lean testing, smoke per batch; one phrase per wait; long jobs in
  the background, answer her first; every publish, deploy and push needs her yes to that action, none overnight;
  advisor trial (successors with `next_orchestrator.py trainworld --advisor fable`, log calls in handoff-log).

## Waiting on her (asks.py list has the open ones)
- Asks open: Sierra Leone flag (T-3684), top bar skipping page translation (T-3635, I'd keep), bypass joined to the
  other track (T-3672 follow-up; file and do it?), track look letter and hues (T-3693; C landed).
- Text acknowledgements not yet given: settings search ("Search settings", "No settings match"), "Stations",
  time controls ("Slower (,)", "Faster (.)", "Slow down or speed up"), "driven by a parent" hover, leaderboard
  "Modded games are not listed" / "Scores from this version are not listed. Reload the page for the newest.".
- She sends the T-3635 player reply herself (draft in notes/T-3635.md). Ask natteru (Discord) about Firefox's
  resist-fingerprinting if she likes (T-3705).

## Only a person knows
- Uncommitted on purpose: TODO.md (hers), notes/T-3397.md, jordan-save-investigation-2026-11-15.md,
  tools/ops/archive_bad_saves.py, changelog/roadmap.json; maps .claude/settings.json (her edits + T-3670 keys).
- Old trees shenzhen (dirty pipeline/osm_units.py, owner unknown) and tt2 (stale, stray cores.json): ask before
  removing. Empty locked folder .tw-trees/touchrule: remove once whatever holds it exits.
- 18:03 UTC: sideplat2's tree removal emptied main app/node_modules through a stray junction; npm install restored
  it (run it from trainworld/app, not with --prefix); agent-tree.mjs now unlinks every junction first.
- Changelog: about 30 unpublished items; a publish needs her edit on the board first.
