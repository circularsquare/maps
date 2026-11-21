# Orchestrator handoff, trainworld

Orchestrator: session 427323a8-53ce-483c-9c28-8f1452e36f9f, name maps-6a [a48b3e], since 2026-11-21 20:45 UTC
Updated: 2026-11-21 21:15 UTC

Live state: `venv\Scripts\python.exe trainworld/tools/handoff.py status`; diary notes/handoff-log.md. Her time UTC-5.

## Next
0. LIVE v0.4.5. Cap 14. Cores 0-1 hers (`reserve_cores.ps1 -Loop 30` at takeover). No publish yet ("no rush").
   Before the next publish: fresh freeze (place index), translations (T-3986, T-3991, T-3999, T-4003, T-4010,
   T-2264, T-3933, T-3378 s4, T-4013, T-4005 keys), smoke on a quiet machine (batchchk11's real-save final timed
   out at cap), one shonei-server deploy (asked, a-..-20; go test passes): 04369d6, f16278a, 3b14158; push it
   after. Standby tabs left: maps-5a, maps-17 (notes/standby.txt).
1. master 7964bcf2 = 8800 3f917ba3 (save level 38). Batch 12 (T-4005, T-4013, T-3373 s3 flagged, T-4017, T-4014
   with its own checker) landed; smoke on 8800 PASS. No batch checker: each risky one had its own or is flagged.
2. TRACK (her top priority): stage 1 landed; tw/track synced to 7964bcf2. bundles T-3949 stage 2a (Opus, maps-ac's)
   in a tree from tw/track (it will meet level 38 when it rebases): land with `agent-tree.mjs finish bundles "..."
   --into tw/track --task T-3949`. Next stages per notes/T-3444-design/10-build-plan.md (2b T-3950...).
3. BRT (her other priority): all landed. Open her call: busways 60 km/h for any bus (built) or BRT only.
4. Running: maps-ac's bundles T-3949 (report via the board); mine: splitb4 T-3373 B stage 4 (Opus), phonemem
   T-3160 s4, airfar T-3502 (Opus), fullspeed T-3655 (Opus).
   Held: T-3932 (B stage 7's pool), T-3060 (tool files of track 2b), T-1354/T-1610 (near splitb4's worker code).
5. Open asks: T-3742 nudge (a-..-16), T-4008 soft rule (-17), wiki push (-18), T-3376 industrial jobs at a
   quarter (-19); text acks for T-4005 and T-4013 strings; T-2264 help wording; busway speed;
   server deploy (-20); T-4013 per-direction follow-up (I'd file P3). Later: T-3937 quiet machine; T-3931 held for
   T-3231; T-3939 B's 140 MB.
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
- 11-21 ~16:10-16:40 UTC: bus reach 1 km; no borrowed-search bus stops ("a lie"); itch.io $0-or-donate, $5
  suggested (T-3988).
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
- A task with stages left: `agent-tree.mjs land` + `remove`, not `finish` (it archives); swap the taking mark by hand.
