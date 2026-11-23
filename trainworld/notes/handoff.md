# Orchestrator handoff, trainworld

Orchestrator: session 28722621-86b3-4444-8131-f3aee02b18dd, name maps-6b [d9ec47], since 2026-11-23 20:08 UTC
Updated: 2026-11-23 20:08 UTC. Live state: `venv\Scripts\python.exe trainworld/tools/handoff.py status`;
diary notes/handoff-log.md. Her time UTC-5.

## In flight (adopt from the board)
- lineclick T-4114 stage 1 (opus: her slow cases, line path and line delete at the click); ghostlag T-4136 (opus, P1,
  "the rail placement ghost lags behind the cursor"); retime T-4132 (opus); regionfix T-4137 (opus, Asia rows seed 9
  regions drift, ships in 0.4.6 otherwise); editbench T-4139 (sonnet, tools/edit_bench.mjs; reuse perf_bench's core
  claim, pin and over-cap refusal); madison T-4140 (opus, agent af85cda5: draw the own zone's far ends for a bubble;
  trips Madison-La Crosse exist, about 160/day each way); rampport T-4112 (sonnet: port tw/track 269568c2 ramp fix
  to master; master has the bug).
- stationwater landed on tw/track 269568c2 (22:26), tree removed; its changelog item is parked in next-track.json, so hide the duplicate the port adds at the release merge. Track build not rebuilt: snaptrack needs sync-track first, which still conflicts (waits for the rework resume); she sees the fix on 8800 once rampport lands. T-4124 closed,
  measured not landed (no in-game gain, memory +13-15%); code on branch tw/onegraph-measured. 8800 = d4eabb5c6;
  master has T-4138 since. Batch check due for rampport (track split) when it lands.

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
