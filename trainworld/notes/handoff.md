# Orchestrator handoff, trainworld

Orchestrator: session d15915ac-6dd5-4586-bd98-843cbc616824, name maps-ac [1ef926], since 2026-11-21 15:53 UTC
Updated: 2026-11-21 18:20 UTC

Live state: `venv\Scripts\python.exe trainworld/tools/handoff.py status`; diary notes/handoff-log.md. Her time UTC-5.

## Next
0. LIVE v0.4.5. Cap 14 by day. Cores 0-1 are hers: `reserve_cores.ps1 -Loop 30` at takeover (docs/orchestrator.md).
   No publish yet (her 11-21: "no rush ... nothing super urgent/gamebreaking"). Before the next one: a fresh freeze
   (placeindex's world index, 135 MB gz, 4,097 files on R2), translations (T-3986, T-3991, T-3999, T-4003 keys).
   Server deployed 17:56 UTC (community limits; save levels 35-36), so publishing level-36 saves is safe.
1. Batches 9 and 10 checked PASS (notes/batch-2026-11-21.md); 8800 = 6a642fb1 (save level 36). Batch 11 on master,
   unchecked, not on 8800: oneway T-4000 (own check PASS), pausebp T-4002/T-4003, placeindex T-4004. Next: batchchk11
   when a few more land (add stationkinds' "stop on a station joined to nothing does nothing and says nothing").
   8800 = 3b521b1d (batch 11 minus brtdash 5b225105, landed after).
2. Running: trackprep T-3948 (Opus, track stage 1 + T-1183), oomretry T-3933 (Opus), timegroups T-4008 (Opus,
   measure only), hovers T-4009 (Sonnet), kinds2 T-4010 (Sonnet: train/tram split, BRT cap 100, icon, no parking).
   busways T-3378 stage 4 (Opus, started despite kinds2's BRT overlap: her BRT priority). hovers landed.
   Held for collisions: T-3373 B stage 3 and T-3932 (oomretry). T-4011 folded into kinds2.
3. Then: T-3937 needs a quiet machine; ownwaits T-3931 held for T-3231 (patch in notes/); T-3655 left: faster
   full-track edits; T-3939 left: B's ~140 MB unnamed; T-3395: save_health.py after the next scan.
4. Open ask: a-2026-11-21-15 income groups (notes/T-3246.md; I said keep out).
## Her words
- 11-21 ~19:20 UTC: BRT stations come with a waiting room, shown as bought (T-4012, brtwait running); tram icon's
  top piece bigger; "text ack" (kinds2 strings); boarding-only / alighting-only stops filed (T-4013).
- 11-21 ~18:45 UTC: main priorities now: keep the track rework moving, finish and polish BRT. She's out with remote
  control on this session; standby orchestrator tabs (notes/standby.txt, docs/orchestrator.md "Your own cycling").
- 11-21 ~17:50 UTC: "text ack" for everything listed to then (viewonly, BRT stations, modeldown, brtblueprint,
  community, stationkinds). Search must reach far towns, "general" against crashes (T-4004 built). Server deploy yes.
- 11-21 ~17:00 UTC: closed T-3365; levels -5 to +5 with the track rework, Settings toggle (T-3691). Station totals by
  mode with station timetables (T-3780). Exact sums OK ("<0.5% is never that much"). Community: open paused,
  blueprints hidden, no leave confirm (T-4002); bookmarks maybe (T-4005); raise limits (T-4006, deployed).
- 11-21 ~16:40 UTC: bus reach 1 km (T-3995); no borrowed-search bus stops (T-3992 closed, "a lie"); BRT built border
  inside the colour, dashes fixed on the ground (T-3996); station tool train/bus/port, open-ground stations (T-3999).
- 11-21 ~16:10 UTC: Report as a flag with a confirm (T-3991); Discord and leaderboards in Community (T-3990);
  itch.io $0-or-donate, $5 suggested (T-3988).
- 11-21 ~15:55 UTC: track rework all calls made (notes/T-3444-design/answers-2026-11-21.md). 38 stages
  T-3948..T-3985 on tw/track.
- 11-21 ~14:05 UTC: keep 4-12 agents. Count every Auto stop ("if people are seeing bus stops the bus stops have to
  be real").
- 11-21 ~12:50 UTC: BRT stations for BRT lines only; T-3373 B go, the memory limit rises (players set it: T-3932; a
  load out of memory retries lower: T-3933). Never ask again whether big saves get more workers.
- 11-19: "player complaints take precedent over other new features"; full playtest list before every publish.
- When she plays games: no uploads until she says she's done; she may ask for a lower core cap.
- Standing: lean testing, smoke per batch; long jobs in the background, answer her first; every publish, deploy and
  push to the website needs her yes in this tab, none overnight; advisor trial (log calls in handoff-log).

## Waiting on her (asks.py list has the open ones)
- She sends the T-3635 player reply and the T-3945 reply herself (draft in notes/T-3945.md).

## Only a person knows
- Uncommitted on purpose: TODO.md (hers), notes/T-3397.md, jordan-save-investigation-2026-11-15.md,
  tools/ops/archive_bad_saves.py, changelog/roadmap.json, changelog/next.json, notes/asks-log.jsonl,
  notes/board-live.json; maps .claude/settings.json. Not mine: notes/T-2380.md, notes/T-3382.md.
- A yes she gives in another session's tab doesn't pass the permission check here: ask again in this tab.
- Never delete a tree folder by hand (its app/node_modules junction deletes the main tree's).
- Commit messages through PowerShell: single quotes, or "$2M" becomes empty (49fa812e lost "US$2M").
- Landing scripts: wait.mjs refuses waits over 170 s; loop it.
