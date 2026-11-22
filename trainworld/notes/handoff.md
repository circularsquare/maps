# Orchestrator handoff, trainworld

Orchestrator: session 360c2881-9196-44fd-8596-05db148a51cb, name maps-17 [098830], since 2026-11-22 14:16 UTC
Updated: 2026-11-22 16:00 UTC

Live state: `venv\Scripts\python.exe trainworld/tools/handoff.py status`; diary notes/handoff-log.md. Her time UTC-5.

## Next
U. USAGE (her 11-22 ~16:20 UTC): "we're low on remaining tokens for the week so lets keep the track work wound down
   actually. lets keep the rest for bug fixes for now. i'd consider pax not taking express as the major bug fix."
   Keep: firsttrain (T-4041), clockmem (T-3939, phone crashes), hokkaido (finishing its join). Paused at a clean
   point with notes, trees kept, nothing landed: addtrack, stationids, timetables, coreslider, perfledger, split6.
   Start only bug fixes, few at a time. The 8800 track preview still goes up after hokkaido (no agent needed).
0. LIVE v0.4.5. Cap 14 (her daytime pick 11-22). Cores 0-1 hers (`reserve_cores.ps1 -Loop 30` at takeover). No
   publish yet. Before one: fresh freeze, translations (T-3986, T-3991, T-3999, T-4003, T-4010, T-2264, T-3933,
   T-3378 s4, T-4013, T-4005, T-4040, and the new zones' names: place_names.mjs --zones), smoke on a quiet machine. Batch 13
   + the night's data passed a checker on 547389c2 (11-22). Server deployed 3b14158. Save level 38 on master.
1. 8800 = 547389c2. master since: T-4040 (UI), T-1354 (demand worker). Once hokkaido's join ends (it owns
   data/packs): sync-track, then 8800 = tw/track tip as her preview (her yes 11-22; saves made there won't load on
   master or live: told her), and a batch checker on master. All strings to 11-22 16:10 UTC acknowledged ("Part of a bundle of # tracks", her wording).
2. TRACK (her top priority): tw/track has 1-3g, 4a1, 4a2, 4c, 4e picture, T-4030 a-f, T-4038, levels (T-3691), 4b's
   two fixes. T-3961's merge waits for g; T-3964 Join waits T-4043. Running: stationids (T-4043, own checker before
   landing), addtrack (T-4024). g (T-4037) waits her two calls; then 4d (T-3963), 4f, 5a/5b. Pre-merge: T-4026,
   T-3958, a checker (save format, sim core). Plan: notes/T-4030.md. A finish can stop on a docs conflict: resolve in
   the tree (CRLF), rebase --continue, rerun tests, finish.
3. Master agents: split6 (T-3373 B6), hokkaido (T-3688, writes data/packs), clockmem (T-3939), timetables (T-3346
   stage 3), coreslider (T-3932), firsttrain (T-4041, P1), perfledger (T-4025).
4. Waiting on her: g's two calls (re-explained 11-22 ~16:10 UTC). Wiki on the website on hold.

## Her words
- 11-22 ~16:05 UTC: cap 14 (daytime); T-4041 yes, high prio (and buses beating parallel trams; tram boarding
  penalty maybe undone for street trams); China shape fix next week (T-4044); Georgia fine; T-4025 start; tw/track
  preview on 8800 yes. Earlier: "lets wind down", then "can keep going". Hokkaido via the old Japan tool: fine.
- 11-22 ~04:30 UTC: last day of the usage week, a decent chunk left: aim to finish track stage 4 overnight ("if we run
  out, we run out"); compute-heavy, token-light tasks too (the Caucasus and Afghanistan, done). China populations look
  old: T-4028. Overnight cap 16; back to 8 when she's up (ask).
- 11-22 ~04:10 UTC: 3d sheet fine; Manila keeps Malolos; Hokkaido "i'd rather do one big dochuo zone"; trip ? good.
- 11-21 ~22:40 UTC: deploy and wiki deferred till home; T-3376 yes; T-3742 planned, start later; T-4008 no soft
  rule, drop income groups (T-4019); busways any bus.
- 11-21 ~18:45 UTC: main priorities: keep the track rework moving, finish and polish BRT. Standby orchestrator tabs
  (notes/standby.txt, docs/orchestrator.md "Your own cycling").
- 11-21 ~17:00 UTC: levels -5 to +5 with the track rework, Settings toggle (T-3691). Station totals by mode with
  station timetables (T-3780). Exact sums OK ("<0.5% is never that much").
- 11-21 ~12:50 UTC: BRT stations for BRT lines only; never ask again whether big saves get more workers.
- 11-19: "player complaints take precedent over other new features"; full playtest list before every publish. When
  she plays: no uploads until she's done; she may ask for a lower core cap.
- Standing: lean testing, smoke per batch; long jobs in the background, answer her first; every publish, deploy and
  push to the website needs her yes in this tab, none overnight; advisor trial (log calls in handoff-log).

## Waiting on her (asks.py list has the open ones)
- All re-asked in the 11-22 ~16:00 UTC catch-up (maps-17), plus the 4-track sheet track-4034.png to look at.
- She sends the T-3635 player reply herself (sent her the text 11-22). T-3945 never had a reply draft. Quote text in
  the message, never send her a notes file (her 11-22).

## Only a person knows
- Uncommitted on purpose: TODO.md (hers), notes/T-3397.md, jordan-save-investigation-2026-11-15.md,
  tools/ops/archive_bad_saves.py, changelog/roadmap.json, changelog/next.json, notes/asks-log.jsonl,
  notes/board-live.json; maps .claude/settings.json. Not mine: notes/T-2380.md, notes/T-3382.md,
  tools/feedback-inbox/builds.json, tools/release/new_zones_map.py, notes/T-3821.md, notes/T-3896-report.png.
- notes/T-3688/kit and notes/T-3444-design/tasks-to-file.md are untracked working files (kit: the redraw scripts).
- Undo of +5 trains doesn't refund (batch checker 11-22): likely by design, not checked against the parent.
- A yes she gives in another session's tab doesn't pass the permission check here: ask again in this tab.
- A task with stages left: `agent-tree.mjs land` + `remove`, not `finish` (it archives); swap the taking mark by hand.
