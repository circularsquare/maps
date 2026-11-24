# Orchestrator handoff, trainworld

Orchestrator: session 28722621-86b3-4444-8131-f3aee02b18dd, name maps-6b [d9ec47], since 2026-11-23 20:08 UTC
Updated: 2026-11-23 20:08 UTC. Live state: `venv\Scripts\python.exe trainworld/tools/handoff.py status`;
diary notes/handoff-log.md. Her time UTC-5.

## In flight (adopt from the board)
- Edit speed, plan notes/T-4144.md; her 23:1x "do all, land all, one message when done". TOP: editnetwork T-4153 (opus,
  bounded drawn network on master: 50-70% of the wait; her 00:0x). editbase T-4145 (base landable; land after its
  bench+smoke, then tell editstations/editlines to merge master), editstations T-4146 (B; written, untested, waits for base: delete its stub game/pending.ts before merging master), editlines T-4147 (A; stage 1 rows-first done in tree, smoke unverified, hooks in clock.worker.ts and clockDemand.ts may
  conflict with editbase/editworker at merge),
  editworker T-4149 (D worker: bayTimes, posting, then A/B/E; rows-first waits for base+B), trackstream (E, agent a2b95733 resumed: T-4155 streets resend, T-4142, T-1031); firstclick T-4143 landed de066376.
  Base API at the top of notes/T-4145.md: put it in C's brief. Queued: C T-4148, D page T-4150, T-4133/4134 after
  retime, T-4151. Her yes: refusals take a shown edit back, money at the click; refusals must answer fast too.
  Agent ids: editbase adf738eb, editstations a832c397, editlines a3199ea2, editworker a0d0baab, junctionmsg a23cb70f.
- Non-edit perf (her 00:3x: "other performance fixes ... load time, memory usage"): zoneopen T-4051, memwasm T-4156 (agent a17f9bab; T-1367 closed, gone on master);
  next from notes/perf-sweep-2026-11-23.md: T-921 (after searchleft; shifts riders: numbers to her), T-2589+T-2247.
- commutechunks T-921 (measure only; shifts riders, numbers to her). searchleft T-4054 landed e71ffeae (search 4.6%
  faster, same figures; u16 shares diff not landed).
- retime T-4132, junctionmsg T-4152 (her pad: junction heading message misfires; strings to her).
- Before telling her edit speed is done: one Opus review of all the show-at-click code together (one mechanism, no
  leftover per-feature stand-ins, editlines' no-stops special case, lineclick folded into the base), a smoke check
  that no pending entry outlives the final state, and a batch check of refusals, undo after shown edits, charges.
- Batch check due: rampport 380d8fc7, regionfix 8e1eb7ba, lineclick ef99d979; tree "rampport" to remove. 8800 =
  ef99d979. T-4112's tw/track changelog item parked in next-track.json: hide the duplicate at the release merge.

## Next
1. Edit speed until done, then one message to her. No publish unless she says.
2. T-4051 zone-open (two untimed fixes in tree zoneopen, one quiet timing run). Solve triage: notes/solve-triage-2026-11-23.md.
3. Track rework paused until after 0.4.6 (T-4037 g3b; T-4116 her panic); tw/track sync with master conflicts. Idle
   trees (coreslider, onewaylag, split6, stationids, timetables): bring up after 0.4.6.

LIVE v0.4.5. 8800 = master always (809a0d16); the track build only at /trainworld-track/ (?track), own saves, no
account (snaptrack.py). Her North America save is level 42: opens only there; backups in artifacts/trainworld/savedl/.
Cap 10 from 23:48 (her ask). Never raise the cap without her explicit word (her 00:4x), not even back to 14 or the
overnight max. While at 10 the reserve loop runs -Reserved 6 (agents on cores 6-15; task bjeex4bpp); cores 0-1 hers (reserve_cores.ps1 -Loop 30, now never touches terminals). Bench: perf_bench.mjs,
ledger notes/perf-ledger.jsonl.

## Her words
- 11-23 00:0x: the drawn network rebuild is "the entire issue. this is what we should fix", not left to the track rework;
  angry it was rediscovered and postponed. editnetwork T-4153 (opus) started; it is the top edit-speed priority.
- 11-23 23:1x: edit speed: "just do all of the changes and i can test, no need for many rounds of implement-test. lemme
  know when everything is done". Run every T-4144 stream, land all, then one message to her; no interim test asks.
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
