# Orchestrator handoff, trainworld

Orchestrator: session dc375617-affd-42c6-aac0-505e37685277, name maps-44 [b23630], since 2026-11-23 12:41 UTC
Updated: 2026-11-23 13:50 UTC. Live state: `venv\Scripts\python.exe trainworld/tools/handoff.py status`; diary
notes/handoff-log.md. Her time UTC-5.

## Running (13:50 UTC, 9 agents, one over her 8: don't refill ridecheck's slot): solvespeed T-4054, solveprof T-4057,
ridecheck T-4058, hokkaido T-3688, addtrack T-4024, trackg1 T-4037 g1 (g2 to it next), shiftdel T-4068, busnosnap
T-4069 (both Sonnet), crashes T-4071 (triage, stage 1 report first). T-4056 landed 4169b78a.

## Next (11-23)
0. Her goals, 11-23: compute speedups for T-4041 (main); loading consistency and crashes (main, short/medium term:
   T-4071); polish first-arrival riding; track rework all week. 2-8 agents this week ("more focused work"). Speedups
   may shift riders (orchestrator.md, trial). Resend over its cap is fine. Publish: "we will see", don't ask.
1. PUBLISH when ready (never ask her when): can ship on freeze 20261122-044711 (has Manila,
   Muzaffarpur and the old Hokkaido; data/packs lacks old Hokkaido until hokkaido lands). Landed since
   v0.4.5 and checked (batch 13 on 547389c2; batch 14 and 15 on master 5e4ac09f): T-4041 riders take the line that
   arrives first (0.75 softening, late joins), T-4050 evenly spaced shared platforms, T-4054 stage 1, T-1354, T-4040,
   load bars (skew + fill at end). Before it: fresh freeze, translations (T-3986, T-3991, T-3999, T-4003, T-4010,
   T-2264, T-3933, T-3378 s4, T-4013, T-4005, T-4040, the new zones' names: place_names.mjs --zones), smoke8800 on a
   quiet machine, the changelog review (46 unpublished; c051 is T-4041's). Then her text OK and her yes, in this tab.
2. Speed (T-4054 search, T-4057 elsewhere): same-answer wins land as they report; shifting ones bring a test shape
   for her to build (orchestrator.md "Speedups that shift riders"); batch checker after.
3. Next as slots free: T-4051 (big save sits at 90% ~3 min), T-4055 (stale two-zone final, 1-2 h), then the paused
   trees (each tree's notes): onewaylag T-4045 (after T-4054), clockmem T-3939, stationids T-4043, timetables
   T-3346, coreslider T-3932, perfledger T-4025, split6. After hokkaido: sync-track, 8800 = tw/track tip, checker.
4. TRACK (all week): on her (b), g1-g4 (notes/T-4037-design.md; g3, g4 side by side), then 4d, 4f, 5a/5b.

LIVE v0.4.5. 8800 = fa196b91. Cap 14 (her daytime pick). Cores 0-1 hers (`reserve_cores.ps1 -Loop 30` at
takeover). Server 3b14158. Save level 38. Pad 11-22, not started: T-4046-49, T-4052, T-4053 (design to her first).

## Her words
- 11-22 evening, T-4041: lines independent ("every line running at its own random times is probably correct");
  softening "75% of the way" (a quarter take the first train that can still make it); keep the cliff fix ("i do
  want the fix actually"); the first-train quarter stays agnostic about changes; the slowdown is "okay". T-4050:
  same track + same stops + no passing = evenly spaced, per shared stretch ("players dont have any tools to
  manually space out lines"). Load bars: what read 90% now reads half full (her idea). Her hope: "buses get more
  reasonable values relative to trams ... more uniform ... cuz everything is parallel".
- 11-22: China shape fix next week (T-4044); Hokkaido via the old Japan tool; a yes in another tab doesn't count here.
- 11-21: main priorities the track rework, finish and polish BRT; T-3742 planned, start later. Standby
  orchestrator tabs (notes/standby.txt).
- 11-19: "player complaints take precedent over other new features"; full playtest list before every publish. When
  she plays: no uploads until she's done; she may ask for a lower core cap.
- Standing: lean testing, smoke per batch; long jobs in the background, answer her first; every publish, deploy and
  push to the website needs her yes in this tab, none overnight; advisor trial (log calls in handoff-log); BRT
  stations for BRT lines only; never ask again whether big saves get more workers (11-21).

## Waiting on her
- No open asks. She sends the T-3635 player reply herself. Quote text in the message, never send her a notes file.

## Only a person knows
- Uncommitted on purpose: TODO.md (hers), notes/T-3397.md, jordan-save-investigation-2026-11-15.md,
  tools/ops/archive_bad_saves.py, changelog/roadmap.json, changelog/next.json, notes/asks-log.jsonl,
  notes/board-live.json; maps .claude/settings.json. Not mine: notes/T-2380.md, notes/T-3382.md,
  tools/feedback-inbox/builds.json, tools/release/new_zones_map.py, notes/T-3821.md, notes/T-3896-report.png.
- .tw-trees/arrivefirst is a leftover folder, not a worktree; a node_modules junction may be inside: never delete it
  by hand (agent-tree.mjs remove refused it; check for the junction first).
- Untracked working files: notes/T-3688/kit (the redraw scripts), notes/T-3444-design/tasks-to-file.md.
- A task with stages left: `agent-tree.mjs land` + `remove`, not `finish` (it archives); swap the taking mark by hand.
