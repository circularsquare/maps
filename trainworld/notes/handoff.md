# Orchestrator handoff, trainworld

Orchestrator: session ca3a5e8b-220b-4b3d-a5e9-87f500e687dc, name maps-98 [60f1c2], since 2026-11-21 12:25 UTC
Updated: 2026-11-21 15:40 UTC (10 agents running)

Live state: `venv\Scripts\python.exe trainworld/tools/handoff.py status`; diary notes/handoff-log.md. Her time UTC-5. She is home, at this tab.

## Next
0. LIVE v0.4.5 (11-21 ~04:15 UTC). Cap 14 by day; reserve_cores.ps1 -Loop 30 runs in my session (cores 0-1 hers).
   8800 = 7e04ea1f (batch 8 checked; islands3 ferry case FAIL -> islandfix). Next update: translations nearer publish.
1. Batch 9 on master, not on 8800: phonemem2 49a4142e, autostops 94832b99, dropindent d80eaa39, splitb1 2382b9b8;
   landing: brtstations (save level 35), editfail T-3945 (capacity panic fix) next. Smoke passed 15:20 at d80eaa39.
   After islandfix: batch-9 checker + smoke + 8800 rebuild.
2. Running: ownwaits T-3931, fullrate4 T-3655 (+T-3941), clockmem T-3939, keihan T-3935, viewonly T-3851,
   brtblueprint T-3944, islandfix (T-2216 fix), busreach T-3934, splitb2 (T-3373 B stage 2, same agent as splitb1).
   Next: T-3948 track stage 1 after ownwaits+fullrate4 land; T-3378 st.4 after T-3944; T-3986 (model-down messages);
   T-3933/T-3932 later; T-3937 needs a quiet machine. HER 11-21 ~14:05: compute is fine, keep 4-12 agents.
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
- 11-21 ~15:00 UTC: track rework design agreed (notes/T-3444-design/answers-2026-11-21.md); 38 stages filed T-3948..
  T-3985 on tw/track (merged at releases); stage 1 (T-3948, T-1183 folded in) starts once ownwaits and fullrate4 land.
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
