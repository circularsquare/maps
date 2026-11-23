# Orchestrator handoff, trainworld

Orchestrator: session dc375617-affd-42c6-aac0-505e37685277, name maps-44 [b23630], since 2026-11-23 12:41 UTC
Updated: 2026-11-23 14:50 UTC. Live state: `venv\Scripts\python.exe trainworld/tools/handoff.py status`; diary
notes/handoff-log.md. Her time UTC-5.

## Running (8 agents): solvespeed T-4054 (then T-4064; shifting candidates asked a-2026-11-23-6), solveprof T-4057
(+T-4082 Detroit), hokkaido T-3688, trackg1 T-4037 g1 (rebase over addtrack; g2 next), crashes T-4072/T-4073,
busconn T-4079, addbp T-4085 (tw/track), checkbns (checker of busnosnap's level-39 save change; tree busnosnap
waits, land on PASS). Landed: T-4056, T-4068, T-4076, community 378e3fe9 (shonei-server 4f71a2f needs a deploy on
her yes); T-4024 onto tw/track 3c62751b.

## Next (11-23)
0. Her goals, 11-23: compute speedups for T-4041 (main); loading consistency and crashes (main, short/medium term:
   T-4071); polish first-arrival riding; track rework all week. 2-8 agents this week ("more focused work"). Speedups
   may shift riders (orchestrator.md, trial). Resend over its cap is fine. Publish: "we will see", don't ask.
1. PUBLISH when ready (never ask her when): freeze 20261122-044711 works (old Hokkaido) until hokkaido lands. Since
   v0.4.5 and checked (batch 13 on 547389c2; batch 14 and 15 on master 5e4ac09f): T-4041 riders take the line that
   arrives first (0.75 softening, late joins), T-4050 evenly spaced shared platforms, T-4054 stage 1, T-1354, T-4040,
   load bars (skew + fill at end). Before it: fresh freeze, translations (T-3986, T-3991, T-3999, T-4003, T-4010,
   T-2264, T-3933, T-3378 s4, T-4013, T-4005, T-4040, the new zones' names: place_names.mjs --zones), smoke8800 on a
   quiet machine, the changelog review (46 unpublished; c051 is T-4041's). Then her text OK and her yes, in this tab.
2. Speed (T-4054 search, T-4057 elsewhere): same-answer wins land as they report; shifting ones bring a test shape
   for her to build (orchestrator.md "Speedups that shift riders"); batch checker after.
3. Next: T-4080 + T-4083 (one design: bus stops keeping identity across edits/saves; after busnosnap lands),
   T-4075, T-3940, T-4051, T-4055, paused trees (onewaylag after T-4054, clockmem, stationids, timetables,
   coreslider, perfledger, split6). After hokkaido: sync-track, 8800 = tw/track tip, checker.
4. TRACK (all week): on her (b), g1-g4 (notes/T-4037-design.md; g3, g4 side by side), then 4d, 4f, 5a/5b.

LIVE v0.4.5. 8800 = 73299e61. Cap 14 (her daytime pick). Cores 0-1 hers (`reserve_cores.ps1 -Loop 30` at
takeover). Server 3b14158. Save level 38. Pad 11-22, not started: T-4046-49, T-4052, T-4053 (design to her first).

## Her words
- 11-22 evening, T-4041: lines independent ("every line running at its own random times is probably correct");
  softening "75% of the way" (a quarter take the first train that can still make it); keep the cliff fix ("i do
  want the fix actually"); the first-train quarter stays agnostic about changes; the slowdown is "okay". T-4050:
  same track + same stops + no passing = evenly spaced, per shared stretch ("players dont have any tools to
  manually space out lines"). Load bars: what read 90% now reads half full (her idea). Her hope: "buses get more
  reasonable values relative to trams ... more uniform ... cuz everything is parallel".
- 11-22: China shape fix next week (T-4044); Hokkaido via the old Japan tool; a yes in another tab doesn't count here.
- 11-21: track rework, finish and polish BRT; T-3742 planned, later. Standby tabs: notes/standby.txt.
- 11-19: "player complaints take precedent over other new features"; full playtest list before every publish. When
  she plays: no uploads until she's done; she may ask for a lower core cap.
- Standing: lean testing, smoke per batch; long jobs in the background, answer her first; every publish, deploy and
  push to the website needs her yes in this tab, none overnight; advisor only for big design decisions (11-23); BRT
  stations for BRT lines only; never ask again whether big saves get more workers (11-21).

## Waiting on her
- Asks: `asks.py list` (phone load, Spain, bus connection timing; strings shiftdel/busnosnap). T-3635 reply is hers.

## Only a person knows
- Uncommitted on purpose: TODO.md (hers), notes/T-3397.md, jordan-save-investigation-2026-11-15.md,
  tools/ops/archive_bad_saves.py, changelog/roadmap.json, changelog/next.json, notes/asks-log.jsonl,
  notes/board-live.json; maps .claude/settings.json. Not mine: notes/T-2380.md, notes/T-3382.md,
  tools/feedback-inbox/builds.json, tools/release/new_zones_map.py, notes/T-3821.md, notes/T-3896-report.png.
- .tw-trees/arrivefirst is a leftover folder, not a worktree; a node_modules junction may be inside: never delete it
  by hand (agent-tree.mjs remove refused it; check for the junction first).
- Untracked working files: notes/T-3688/kit (the redraw scripts), notes/T-3444-design/tasks-to-file.md.
- A task with stages left: `agent-tree.mjs land` + `remove`, not `finish` (it archives); swap the taking mark by hand.
