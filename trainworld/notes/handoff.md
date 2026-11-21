# Orchestrator handoff, trainworld

Orchestrator: session ca3a5e8b-220b-4b3d-a5e9-87f500e687dc, name maps-98 [60f1c2], since 2026-11-21 12:25 UTC
Updated: 2026-11-21 12:55 UTC (12 agents running; all 8 asks answered)

Live state: `venv\Scripts\python.exe trainworld/tools/handoff.py status`; diary notes/handoff-log.md. Her time UTC-5. She is home, at this tab.

## Next
0. LIVE v0.4.5 (published 11-21 ~04:15 UTC). Cap 14 by day (overnight = machine max, docs/orchestrator.md). 8800 =
   4141e6bc (T-2624 landed, periodschk PASS, master smoke PASS). Since v0.4.5 on master: fares panel, binzip (packs at
   the next publish: walk her through it first), T-3904/3908/3909/3913/3916/3918/3919/3921/3922/3923/3925/3927/3928/
   3929, T-1183 st.1-2, T-3160 st.3, T-2624. Next update: translate moneyTab.brtFare/brtFareTip + T-3922 hovers + any
   new, nearer publish; changelog c001-c014 (c007, c012, c013 hidden).
1. Running: phonemem2 T-3160, autostops T-3924, busway T-3378 st.1-2 (own trees), iphonesaves T-3781 (no tree; ssh
   may be refused: then ask her here), trackdesign T-3444 (design lead with research subagents, no builds; writes
   notes/T-3444-design/ + review/track-rework-design.md). Landed since 8800 (4141e6bc), UNCHECKED BATCH: cachekey
   44316713, movedhour 4f8b9e0d, tweaks e778ce68, egyptunits c2ec5711 (pipeline), islands3 a2062187 (CACHE_FORMAT 13).
   Batch checker + smoke + 8800 rebuild wait for her OK on compute. T-3655 parked again (option 1 decided).
2. HER 11-21 ~14:05 UTC: the 13:20 "no more compute" was a moment, not a standing hold ("that doesnt mean we should
   not be using compute now"); keep 4-12 agents. Count every Auto stop (T-3924, autostops finishing). PR 24 reply
   posted 14:10. Batch 8 (cachekey, movedhour, tweaks, islands3, busway) checking: batchchk8 + smoke + 8800 at
   7e04ea1f. Running too: ownwaits T-3931, fullrate4 T-3655, splitb1 T-3373 B st.1, brtstations T-3378 st.3,
   clockmem T-3939, keihan T-3935. Next when a slot frees: T-3934 (after autostops lands) (her ask: bus reach 600 m vs 1 km vs none, on bus-heavy community
   saves; measure only); T-3373 B stage 1 (measure and simulate, 2-3 h, notes/T-3373-B-plan.md; raise A's
   limit to ~2 GB in it); T-3933 (load out of memory retries with fewer workers; wait for iphonesaves' findings);
   T-3932 (cores slider with memory and speed; after B's stage 1 so estimates match); T-1183 after T-3931; T-3378
   st.3 (BRT stations for BRT lines only, her OK ~13:00) and st.4 after st.1-2.
3. T-3395: save_health.py after the next scan; clean = close it (no 11-21 report at 04:30 UTC; 11-20 found 0).
## Her words
- 11-21 ~12:50 UTC (board answers + chat): Auto stops "i would rather coutn every auto stop for demand ... seems
  pretty misleading"; BRT design "yes this all sounds fine", asks why BRT stations are BRT-only ("not necessarily
  against it"; answered in chat: else plain buses get the same 3-4x jump and beat the off-road tram on money);
  upkeep 0.2%; tram brake 0.9 "for consistency"; tint: ground OK, teal/khaki hard to tell apart, "try something else
  like green for city"; iPhone saves "yes can read"; two workers: "you guys have asked this question many times ...
  its on the player to decide" -> T-3932 slider. Never ask her again whether big saves should get more workers.
  T-3373 B: "sounds good. we can pursue it. and yes we can raise memory limit, though ... players should be able to
  set it"; "if people are unable to lod the game at all because o fmemory issues it should probably autoset the
  memory cap lower and try again" (T-3933). Rounding change (~a millionth) taken as part of the approved design.
- 11-21 ~02:20 UTC, playtesting plain buses on 8800 (Houston): "buses feel really easy ... mostly profitable ... 10%
  of people taking the bus ... 90% driving. which feels pretty realistic". Bus speeds set 11-21: plain floor 25,
  BRT +10, fast roads 1.4x street; BRT charged in town only; open-country main roads keep 75. "all text ack" ~02:45.
- 11-20 ~22:10 UTC: "lets freeze the set of proposed changes here for now, and test with these"; chart remake
  after: no separate street tram, half the trip lengths, 2 more stop spacings at the high end, add HSR; a made-up
  uniform city run as a model, not Chicago in the sim.
- 11-20 ~22:00 UTC: T-3888 option 4; ferry design OK + running cost -25%; bus reach 300 -> 600 "i do kinda want
  it" (if buses overshoot, make them slower); trams to 75. ~21:00: ferry types, a fare multiplier per type.
- 11-20: cleanup, then "full track rework" (T-3444 talk); no minor track work till then; 400 km ferry legs stay.
- 11-19: "player complaints take precedent over other new features"; full playtest list before every publish.
- When she plays games: no uploads until she says she's done; she may ask for a lower core cap.
- Standing: lean testing, smoke per batch; long jobs in the background, answer her first; every publish, deploy and
  push to the website needs her yes in this tab, none overnight; advisor trial (log calls in handoff-log).

## Waiting on her (asks.py list has the open ones; 0 open)
- She sends the T-3635 player reply herself. PR 24 reply (scratchpad/pr24-reply.md): she posts or allows.

## Only a person knows
- Uncommitted on purpose: TODO.md (hers), notes/T-3397.md, jordan-save-investigation-2026-11-15.md,
  tools/ops/archive_bad_saves.py, changelog/roadmap.json, changelog/next.json, notes/asks-log.jsonl,
  notes/board-live.json; maps .claude/settings.json. Not mine: notes/T-2380.md, notes/T-3382.md.
- A yes she gives in another session's tab doesn't pass the permission check here: ask again in this tab.
- Never delete a tree folder by hand (its app/node_modules junction deletes the main tree's).
