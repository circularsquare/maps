# Orchestrator handoff, trainworld

Orchestrator: session 28722621-86b3-4444-8131-f3aee02b18dd, name maps-6b [d9ec47], since 2026-11-23 20:08 UTC
Updated: 2026-11-23 20:08 UTC. Live state: `venv\Scripts\python.exe trainworld/tools/handoff.py status`;
diary notes/handoff-log.md. Her time UTC-5.

## In flight (adopt from the board)
- Adopted from maps-65 (reports reach the inbox): crossfix T-4130 (P1, master, fix before 0.4.6), headways T-4104 (UI; strings).
- Mine: quadcheck T-4111; tmprofile T-4131 (stalled once 20:24, resumed); onegraph T-4124 (opus; measures only, landing
  needs her yes on the shift); stationwater T-4112 (opus, tree off tw/track, also checks master); reloadmsg T-4109.
  Queued: T-4106 install banner once reloadmsg lands (both add en.json strings).
- Landed today after takeover: board3, hoverends T-4107, tutcard T-4099, onewayret T-4127 (fuzzer only, master clean),
  autostation T-4105. 8800 rebuild to eb3e9d3c started 20:25. Ask a-2026-11-23-16 (Auto as default) open.

## Next
1. No publish decided ("i didn't say we want to publish"); don't prep one unless she says. 0.4.6 would wait for most
   of edit speed. Edit speed: bounded edits next stage (T-4097 notes section 4: page panels part C, E, F; T-4111 quad
   check half), then T-4114 show at the click. Felt waits haven't dropped yet: measure click-to-map on the Asia save.
2. T-4124 one graph for all zones (shifts walks 1-2%: numbers to her first). T-4051 zone-open: two untimed fixes in
   tree zoneopen, need one quiet timing run. T-4109 reload message after the model stops. T-4106 install banner
   (strings to her). Solve triage order: notes/solve-triage-2026-11-23.md.
3. Track rework PAUSED until after the 0.4.6 publish (T-4037 g3b at a clean point, 79 tests left; T-4116 her panic).
   tw/track's sync with master conflicts (stateRows.ts, t1382-rows.mjs, demand-kernel.md) and waits for the resume.
   On resume ask her: once stations stop cutting track, single/double, rebuild, level and one-way act on the whole
   piece; accept, or cut at stations? Idle trees (coreslider, onewaylag, split6, stationids, timetables): bring up
   after 0.4.6, start one only if truly nearly done.

LIVE v0.4.5. 8800 = master always (809a0d16); the track build only at /trainworld-track/ (?track), own saves, no
account (snaptrack.py). Her North America save is level 42: opens only there; backups in artifacts/trainworld/savedl/.
Cap 14 daytime; cores 0-1 hers (reserve_cores.ps1 -Loop 30, now never touches terminals). Bench: perf_bench.mjs,
ledger notes/perf-ledger.jsonl.

## Her words
- 11-23: your two top priorities are answering her and never letting agents overbook or collide; agents 2-8 run by
  load; size tasks to about an hour; you don't investigate, agents do; cycle at 500k day and night; no drafted player
  replies; report times as percent of the wait; she has no Apple devices.
- 11-23: 2 crowding rounds refused, final; master heavier than v0.4.5 is expected (express fix, bus changes), look
  for gains elsewhere; coarse setting yes (on by default on phones); install banner yes after 2 h, dismiss forever;
  deleting track never deletes stations; lines side by side on and a $10B start are right.
- 11-22, T-4041: lines independent; softening "75% of the way"; keep the cliff fix. China shape fix next week (T-4044).
- 11-19: "player complaints take precedent over other new features"; full playtest list before every publish.
- Standing: every publish, deploy and website push needs her yes in this tab, none overnight; advisor only for big
  design decisions; BRT stations for BRT lines only.

## Only a person knows
- Uncommitted on purpose: TODO.md (hers), notes/T-3397.md, jordan-save-investigation-2026-11-15.md,
  tools/ops/archive_bad_saves.py, changelog/roadmap.json, changelog/next.json, notes/asks-log.jsonl,
  notes/board-live.json; maps .claude/settings.json. Not mine: notes/T-2380.md, notes/T-3382.md,
  tools/feedback-inbox/builds.json, tools/release/new_zones_map.py, notes/T-3821.md, notes/T-3896-report.png.
- .tw-trees/arrivefirst is a leftover folder with a possible node_modules junction: never delete it by hand.
- A task with stages left: `agent-tree.mjs land` + `remove`, not `finish`; swap the taking mark by hand.
