# Orchestrator handoff, trainworld

Orchestrator: session 5cd3af11-2ab8-4788-8daf-1956104ead51, name maps-c0 [58120f], since 2026-11-24 01:57 UTC
Updated: 2026-11-24 03:32 UTC. Live state: `venv\Scripts\python.exe trainworld/tools/handoff.py status`;
diary notes/handoff-log.md. Her time UTC-5.

## In flight (agents in my tab: editunify ae95d5e2, editnetwork2 a935f09f, orchnotes a4868653, textfix3 ae363d84, busland aa2d2bf6, opsvol a78e91e6; commutechunks T-921 in maps-6b's tab)
- Edit speed: landed tonight T-4153 s1 e7fa38bd, T-4147 2171510d+245735d7, T-4132 3656e5f7, T-4155 6dd59476, T-4146 e94813b8,
  T-4148 dc9ea63a, T-4149 6d7c50c5 (plus base and T-4114/4136/4143). Now: editunify T-4170 (one hook path, her "same
  way"), editnetwork2 T-4153 stage 2. Then: batch check on master (refusals, undo after shown edits, charges, the commits
  above plus rampport 380d8fc7, regionfix 8e1eb7ba, lineclick ef99d979, junctionmsg 9719e410, base 0c384f9e), one smoke on
  master, perf_bench all three saves (memory budget ~10 MB total, DECISIONS), rebuild 8800, one message to her.
- Leftovers filed: T-4146/4147/4148/4149 rewritten to what's left (P2), T-4150, T-4142, T-1031, T-4133/4134.
- zoneopen T-4051: perf compare nyc showed map 12.5->24.8 s, setupMs +70% (busy machine: two WASM builds). Rerun
  `perf_bench.mjs compare artifacts/trainworld/perfbench/dist/zoneopen-base ...dist/zoneopen-head --saves nyc --runs 2`
  when quiet before landing; real -> to her.
- Others running: busland T-4158 (500 m, her pick), textfix3 T-4162, opsvol T-4164 (server cleanup after backup, R2 purge,
  volume row), orchnotes T-4165. Waits on her: T-4156 (memory restart), hard 8-agent hook. Queued: T-4166 AWS bench (quota 5).
- 8800 = ef99d979. T-4112's tw/track changelog item parked in next-track.json: hide the port's duplicate at release.

## Next (night of 11-24; she aims to publish 0.4.6 in the morning after her playtest)
1. Land editunify T-4170 once its Gemini fixes are verified; then the one Gemini review of the whole edit-speed batch
   (her OK for this round; after it Gemini only on her ask or my suggestion).
2. Land busland T-4158, editnetwork2 T-4153 s2, trainmove T-4133 as they report (all risky: in the batch).
3. Batch checker (fresh Sonnet) on master over everything since 8800's ef99d979, one smoke on master, perf_bench all three
   saves (edit-speed memory budget ~10 MB); fix or revert failures; rebuild 8800; run the publish skill (no publish
   without her yes, none overnight); morning message to her with the playtest list.
4. Not in 0.4.6 unless done and checked: T-921 (commutechunks2, numbers to her), T-4178 reversals, T-4051 (map delay),
   T-4156 (waits on her). Track rework after 0.4.6.

## Her words
- 11-24 02:1x: edit-speed streams "in parallel" meant done the same way (similar changes), not independently; the Opus review checks every stream uses the base (T-4145) the same way. Ask before leaving 2-8 agents; the 12 run on, no pausing. 02:0x: Resend Pro no; R2 purge yes;
  R2 reads and the website repo (till 1.3 GB) not reported (docs/ops-checks.md).
- 11-23 00:0x: the drawn network rebuild is "the entire issue. this is what we should fix", not left to the track rework;
- 11-23 23:1x: edit speed: "just do all of the changes and i can test, no need for many rounds of implement-test. lemme
  know when everything is done". Run every T-4144 stream, land all, then one message to her; no interim test asks.
- 11-23: your two top priorities are answering her and never letting agents overbook or collide; agents 2-8 run by
  load; size tasks to about an hour; you don't investigate, agents do; cycle at 500k day and night; no drafted player
  replies; report times as percent of the wait; she has no Apple devices.
- 11-23: 2 crowding rounds refused, final; master heavier than v0.4.5 is expected (express fix, bus changes), look
  for gains elsewhere; coarse setting yes (on by default on phones); install banner yes after 2 h, dismiss forever;
  deleting track never deletes stations; lines side by side on and a $10B start are right.
- 11-19: "player complaints take precedent over other new features"; full playtest list before every publish. 11-22:
  lines independent; softening "75% of the way"; China shape fix next week (T-4044).
- Standing: every publish, deploy and website push needs her yes in this tab, none overnight; advisor only for big
  design decisions; BRT stations for BRT lines only.

## Only a person knows
- Uncommitted on purpose: TODO.md (hers), notes/T-3397.md, jordan-save-investigation-2026-11-15.md,
  tools/ops/archive_bad_saves.py, changelog/roadmap.json, changelog/next.json, notes/asks-log.jsonl,
  notes/board-live.json; maps .claude/settings.json. Not mine: notes/T-2380.md, notes/T-3382.md,
  tools/feedback-inbox/builds.json, tools/release/new_zones_map.py, notes/T-3821.md, notes/T-3896-report.png.
- .tw-trees/arrivefirst is a leftover folder with a possible node_modules junction: never delete it by hand.
- A task with stages left: `agent-tree.mjs land` + `remove`, not `finish`; swap the taking mark by hand.
