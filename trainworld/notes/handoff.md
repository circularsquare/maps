# Orchestrator handoff, trainworld

Orchestrator: session 4feb4373-1fa9-4b66-9592-396d40896c0a, name maps-a3 [ede33a], since 2026-11-21 03:24 UTC
Updated: 2026-11-21 05:25 UTC (overnight queue running)

Live state: `venv\Scripts\python.exe trainworld/tools/handoff.py status`; diary notes/handoff-log.md. Her time UTC-5. She is home, at this tab.

## Next
0. LIVE v0.4.5 (published 11-21 ~04:15 UTC; server 22c4a16 deployed+pushed; open-source 50f7b15 pushed). Cap 10.
   Batches 1-5 PASSED (notes/batch-2026-11-21.md). On master since v0.4.5, not on 8800 (8800 = e496074b, fares):
   b34a6c4b binzip (packs at the next publish: walk her through it first), 6c508f23 checks, 87bad9b7 T-3909 trams in
   everyday trips (DEMAND: needs a batch check, with homeload's T-3916), 4fb32508 T-1183 stage 1 (tests only).
   Next update: translate moneyTab.brtFare/brtFareTip + new strings; changelog c001-c003 so far.
1. Waiting on her: PR 24 reply (scratchpad/pr24-reply.md; posting refused by the permission check: she posts or
   allows it, then close PR with a link); T-3378 BRT design answers (review/brt-design.md, five questions).
2. OVERNIGHT 11-21 (she slept ~05:20 UTC). Her asks ~04:50-05:20: finish BRT/busways incl. off-road (T-3378);
   big-save lag (T-3917); cleanup; "full use of our resources overnight"; trams lose >= a train's time per stop
   (T-3921); upgradable bus stations (in T-3378); upkeep for station upgrades (T-3922). Running: lagsurvey (T-3917),
   homeload (T-3916), tramstop (T-3921), upkeepup (T-3922). Queue: T-3918 (red Rust test, Sonnet), batch check of
   T-3909+T-3916, T-3373 B or lagsurvey's top pick, T-3919, T-3920, T-3655, T-1183 stage 2 (Opus). No
   publish/deploy overnight. Rebuild 8800 after the next batch lands.
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
- Never delete a tree folder by hand (its app/node_modules junction deletes the main tree's).
