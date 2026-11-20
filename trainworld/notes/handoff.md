# Orchestrator handoff, trainworld

Orchestrator: session 1a8ad997-acf4-4a89-963d-166c67fedf88, name maps-65 [fd56e0], since 2026-11-20 18:52 UTC
Updated: 2026-11-20 21:55 UTC

Live state: `venv\Scripts\python.exe trainworld/tools/handoff.py status`. This file is plan and her words.
Diary: notes/handoff-log.md. Her local time is UTC-5.

## Next
0. Live v0.4.4. Cap 14. 8800 = 27995e5e. Master pushed through 78c7f12f. Landed since 8800: toolsplit, frontier
   st1, checkfix (T-3893, T-3877): smoke + 8800 with the next batch.
1. Evening update, maybe tomorrow (her call, needs her playtest). Publish skill (sort, highlights, playtest list,
   translations incl. T-3822/3823 hovers + addCity + today's strings, zones image for PK/BD, smoke8800). Server
   first, her yes: shonei-server 2754ba1, fd294e7, 43360be, 39e1bd2 (not pushed). If ferrytypes bumps the save
   level, the server save check needs it too. After publish: PR 24 reply (notes/T-3659.md, her OK), open-source
   sync, then land binzip (T-2486, tree ready, checker's fixes in).
2. Frozen set (her 22:10 UTC), all mine, trees, checker before landing: plainbus T-3856 + reach 600 (compute cost
   >15% = numbers to her); walkab T-3888 option 4; ferrytypes T-3892; tram75 T-3895 (Sonnet, no checker);
   islands2 T-2216 st2. Then one frontier chart remake (T-3887, her layout in the task). fullrate T-3646 done,
   checker fullratechk running; T-3655 open (cause traced, notes).
3. Later: T-3378 BRT design to her; T-1183 (after fullrate); demandClient/demand.worker splits (T-2374) and
   T-2843 demand health after binzip; T-3894. Cleanup plan: notes/review-cleanup-2026-11-20.md.
4. T-3395: run `tools/ops/save_health.py` after the 11-21 04:00 UTC scan; clean = close it.

## Her words
- 11-20 ~22:10 UTC: "lets freeze the set of proposed changes here for now, and test with these"; chart remake
  after: no separate street tram, half the trip lengths, 2 more stop spacings at the high end, add HSR.
- 11-20 ~22:00 UTC: T-3888 option 4 (bonus 0 + A + 2+2 at stations below/above ground, trams too); ferry design OK
  + running cost -25%; T-2216 stage 2 go; bus reach 300 -> 600 "i do kinda want it" (if buses overshoot, make
  them slower); trams to 75.
- 11-20 ~21:00 UTC: ferry types, a fare multiplier per type; fans slightly up, all purposes, half for
  work/medical/airport. T-3889 string "Trains can't run on tram track." (landed).
- 11-20 ~20:10 UTC: likes the bus stop look. Edit speed paused (T-3550, T-3804 P4) "until people complain again".
- 11-20 ~19:45 UTC: out ~5 h (back ~00:45 UTC), remote control here only, can't playtest; after a handoff the old
  tab relays her messages. "if you find idle resources / compute ... please pick up additional tasks ... mostly
  self contained and can be done without too much input."
- 11-20 ~19:15 UTC: "trams can go on train track but trains cannot path onto tram track" (T-3889, landed).
- 11-20 15:20 UTC: cleanup tasks, then "full track rework" (T-3444 design talk).
- 11-20 ~17:45 UTC: PR 24 reply (notes/T-3659.md) "looks good": post it after the open-source sync.
- 11-20 ~14:00 UTC: no minor track work (T-3662, T-3353 wait on the rework design). 400 km per ferry leg stays.
- 11-19: "player complaints take precedent over other new features"; full playtest list before every publish.
- When she plays games: no uploads until she says she's done; she may ask for a lower core cap.
- Standing: lean testing, smoke per batch; long jobs in the background, answer her first; every publish, deploy and
  push to the website needs her yes in this tab, none overnight; advisor trial (log calls in handoff-log).

## Waiting on her (asks.py list has the open ones)
- She sends the T-3635 player reply herself. Text acks: T-3890's four strings, T-3889's station-rule line.

## Only a person knows
- Uncommitted on purpose: TODO.md (hers), notes/T-3397.md, jordan-save-investigation-2026-11-15.md,
  tools/ops/archive_bad_saves.py, changelog/roadmap.json, changelog/next.json, notes/asks-log.jsonl,
  notes/board-live.json; maps .claude/settings.json. Not mine: notes/T-2380.md, notes/T-3382.md.
- DECISIONS.md holds binzip's T-2486 line uncommitted until binzip lands.
- A yes she gives in another session's tab doesn't pass the permission check here: ask again in this tab.
- Never delete a tree folder by hand (its app/node_modules junction deletes the main tree's).
