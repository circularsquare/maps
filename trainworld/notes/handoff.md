# Orchestrator handoff, trainworld

Orchestrator: session 4feb4373-1fa9-4b66-9592-396d40896c0a, name maps-a3 [ede33a], since 2026-11-21 03:24 UTC
Updated: 2026-11-21 04:20 UTC (v0.4.5 published)

Live state: `venv\Scripts\python.exe trainworld/tools/handoff.py status`; diary notes/handoff-log.md. Her time UTC-5. She is home, at this tab.

## Next
0. LIVE v0.4.5 (published 11-21 ~04:15 UTC: R2 2026-11-20, website 655e0b3, verify --page PASS; server 22c4a16
   deployed + pushed; open-source synced 50f7b15 + pushed; maps master pushed 936a6a55). Upload pause over (she
   pre-approved the deploy and publish). Cap 10. Batches 1-4 PASSED (notes/batch-2026-11-21.md).
   After the release: farecols landed e496074b (T-3915; changelog c001; 2 strings need translating next release:
   moneyTab.brtFare, moneyTab.brtFareTip); batchchk5 checking it; 8800 rebuilding at e496074b.
1. PR 24 reply: drafted in scratchpad/pr24-reply.md (her OK'd text, last line now "It's in the update that went
   out today (v0.4.5), and in this repo as of 50f7b15"); posting was refused by the permission classifier: ask her
   to post it or allow it. Then close the PR with a link. Next: land binzip (T-2486; three unread reports, its
   check failed narrowly then fixed), frontier results (T-3887, three unread reports, still "taking").
2. OVERNIGHT 11-21 (she went to bed ~05:00 UTC). Her 11-21 ~04:50: "finish BRT / busways ... allow building busways
   off road for high expense, sorta like trams off road"; "lag complaints are picking up. lots of people have 2000
   station saves ... memory / demand solve speed tasks"; "code cleanup would also be good"; "making full use of our
   resources overnight". Running: tramerr (T-3909, tree tramerr), chkfix (T-3904, tree chkfix), binzipchk2,
   busdesign (T-3378 design only -> review/brt-design.md for her; design waits for her before build), lagsurvey
   (T-3917, ranks big-save costs). Queue as cores free: T-1183 cleanup (Sonnet), T-3373 B (memory, she chose B
   11-16; plan + stage 1), whatever lagsurvey ranks top, T-3916, T-3655; land binzip if PASS (no publish). No publish/deploy/website
   overnight. 8800 = e496074b (fares panel). Then small next: T-3909 (errands may treat trams as trains), T-3916, T-3908, T-3913, T-3904, T-3655. Later: T-3378
   BRT corridors; T-1183; T-2374 demand splits and T-2843 after binzip; T-3894.
4. T-3395: run `tools/ops/save_health.py` after the next scan; clean = close it. At 04:30 UTC 11-21 there was no
   11-21 report yet (the server restarted 03:59 for the deploy, which may have moved the scan); 11-20's pass found 0.

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
