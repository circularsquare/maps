# Orchestrator handoff, trainworld

Orchestrator: session ca3a5e8b-220b-4b3d-a5e9-87f500e687dc, name maps-98 [60f1c2], since 2026-11-21 12:25 UTC
Updated: 2026-11-21 12:30 UTC (maps-a3 wound down; periodschk still runs in its tab, report via the inbox)

Live state: `venv\Scripts\python.exe trainworld/tools/handoff.py status`; diary notes/handoff-log.md. Her time UTC-5. She is home, at this tab.

## Next
0. LIVE v0.4.5 (published 11-21 ~04:15 UTC). Cap 14 (her 11-21 ~12:15 UTC morning; overnight = machine max, now a
   rule in docs/orchestrator.md). Batches 1-7 checked (notes/batch-2026-11-21.md); T-3925 own check passed. 8800 =
   5e2d59a7. On master since 8800: T-3929 + its follow-up 9851cafb (timetable versions 16), T-3160 st.3 (bunny train).
   Next 8800 rebuild after T-2624 lands. Since v0.4.5 on master: fares panel, binzip (packs at the next publish: walk
   her through it first), T-3904/3908/3909/3913/3916/3918/3919/3921/3922/3923/3925/3927/3928/3929, T-1183 st.1-2.
   Next update: translate moneyTab.brtFare/brtFareTip + T-3922 hovers (+ any new); changelog c001-c011 (c007, c012 hidden).
1. Waiting on her (asks.py list, 7 open): PR 24 reply (scratchpad/pr24-reply.md; posting refused: she posts or allows);
   BRT design (review/brt-design.md); upkeep 0.1%; tram brake 0.8; Auto bus stops (T-3924); two workers; iPhone saves;
   town tint look (review/bus-town-tint.png).
2. T-2624 landed 4141e6bc 12:33 UTC (periodschk PASS; changelog c014, c013 hidden); 8800 rebuild + master smoke
   running. Started 12:35 UTC by maps-98, all Opus: crowdfast T-3513 (tree crowdfast), cachekey T-3926 (tree cachekey),
   movedhour T-3931 (tree movedhour), phonemem2 T-3160 (tree phonemem2), fullrate3 T-3655 (tree fullrate),
   splitplan T-3373 B (design only, notes/T-3373-B-plan.md, goes to her); ~12:50 UTC islands3 T-2216 st.3 (Sonnet),
   noroute T-3632, egyptunits T-3660 (each its own tree). Next: T-1183 after T-3931. Translations of the next
   update's strings wait until nearer publish (strings still changing). watch.py prints an hourly `idle:` line.
4. T-3395: save_health.py after the next scan; clean = close it (no 11-21 report at 04:30 UTC; 11-20 found 0).
## Her words
- 11-21 ~02:20 UTC, playtesting plain buses on 8800 (Houston): "buses feel really easy ... mostly profitable ... 10%
  of people taking the bus ... 90% driving. which feels pretty realistic"; wants the 25 km/h floor; likes the tags.
  Bus speeds set 11-21: plain floor 25, BRT +10, fast roads 1.4x street; BRT charged in town only; open-country
  main roads keep 75 (OK'd). ~02:45 UTC: "all text ack"; boarding 2+2 at every station (T-3910, her call).
- 11-20 ~22:10 UTC: "lets freeze the set of proposed changes here for now, and test with these"; chart remake
  after: no separate street tram, half the trip lengths, 2 more stop spacings at the high end, add HSR; ~22:20:
  a made-up uniform city run as a model, not Chicago in the sim.
- 11-20 ~22:00 UTC: T-3888 option 4 (bonus 0 + A + 2+2 at stations below/above ground, trams too); ferry design OK
  + running cost -25%; T-2216 stage 2 go; bus reach 300 -> 600 "i do kinda want it" (if buses overshoot, make
  them slower); trams to 75.
- 11-20 ~21:00 UTC: ferry types, a fare multiplier per type; fans slightly up, all purposes, half for
  work/medical/airport. T-3889 string "Trains can't run on tram track." (landed).
- 11-20 ~19:45 UTC: out ~5 h (back ~00:45 UTC), remote control here only, can't playtest; after a handoff the old
  tab relays her messages. "if you find idle resources / compute ... please pick up additional tasks ... mostly
  self contained and can be done without too much input."
- 11-20: cleanup, then "full track rework" (T-3444 talk); no minor track work till then; 400 km ferry legs stay;
  PR 24 reply (notes/T-3659.md) "looks good": post it after the open-source sync.
- 11-19: "player complaints take precedent over other new features"; full playtest list before every publish.
- When she plays games: no uploads until she says she's done; she may ask for a lower core cap.
- Standing: lean testing, smoke per batch; long jobs in the background, answer her first; every publish, deploy and
  push to the website needs her yes in this tab, none overnight; advisor trial (log calls in handoff-log).

## Waiting on her (asks.py list has the open ones)
- She sends the T-3635 player reply herself.

## Only a person knows
- Uncommitted on purpose: TODO.md (hers), notes/T-3397.md, jordan-save-investigation-2026-11-15.md,
  tools/ops/archive_bad_saves.py, changelog/roadmap.json, changelog/next.json, notes/asks-log.jsonl,
  notes/board-live.json; maps .claude/settings.json. Not mine: notes/T-2380.md, notes/T-3382.md.
- DECISIONS.md holds binzip's T-2486 line uncommitted until binzip lands.
- A yes she gives in another session's tab doesn't pass the permission check here: ask again in this tab.
- Tree .tw-trees/fullrate is T-3655's PARKED work (uncommitted, blockers in its task): keep it, don't land as is.
- Never delete a tree folder by hand (its app/node_modules junction deletes the main tree's).
