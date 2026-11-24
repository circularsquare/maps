# Orchestrator handoff, trainworld

Orchestrator: session 5cd3af11-2ab8-4788-8daf-1956104ead51, name maps-c0 [58120f], since 2026-11-24 01:57 UTC
Updated: 2026-11-24 02:22 UTC. Live state: `venv\Scripts\python.exe trainworld/tools/handoff.py status`;
diary notes/handoff-log.md. Her time UTC-5.

## In flight (adopt from the board)
- 11 agents, all inside maps-6b's tab (left open): a follow-up goes to maps-6b as "relay to <agent>: <text>". Over 8: start
  nothing until under 8 (her 11-24). Queue (upnext): T-4153 stage 2 (fresh agent on tree editnetwork), T-4165, T-4162, T-4164.
- Edit speed, plan notes/T-4144.md; her "do all, land all, one message when done". Landed: base T-4145, lines path
  T-4114, ghost T-4136, first click T-4143, network stage 1 T-4153 (e7fa38bd). Running: editstations T-4146,
  editlines T-4147, editpieces T-4148, editworker T-4149 (rows-first waits for B), trackstream T-4155/T-4142/T-1031,
  retime T-4132. Queued: T-4150, T-4133/4134. Before telling her: one Opus review of all show-at-click code (each
  stream uses the base the same way), a smoke check that no pending entry outlives the final state, a batch check
  (refusals, undo after shown edits, charges) incl. rampport 380d8fc7, regionfix 8e1eb7ba, lineclick ef99d979,
  junctionmsg 9719e410, base 0c384f9e, network e7fa38bd (perf_bench --saves nyc); then rebuild 8800, one message.
- T-4159 done (games on the volume). After the 03:23 UTC backup logs ok (~7.1 GB): T-4164's cleanup on the box.
  Server save level 39: T-4161. T-4160 after T-4153.
- Perf beyond edits: perfcompare T-4157, busmerge2 T-4158 (her pick), commutechunks T-921 (shift: numbers to her),
  zoneopen T-4051, memwasm T-4156.
- Her open asks: AWS own account (she hit "email in use"; told her a plus address; on "aws account ready, profile
  anita-own", queue a bench-machine agent); hard 8-agent hook (my proposal, unanswered).
- 8800 = ef99d979. T-4112's tw/track changelog item parked in next-track.json: hide the port's duplicate at release.

## Next
1. Edit speed until done, then one message to her. No publish unless she says.
2. T-4051 zone-open (two untimed fixes in tree zoneopen, one quiet timing run). Solve triage: notes/solve-triage-2026-11-23.md.
3. Track rework paused until after 0.4.6 (T-4037 g3b; T-4116 her panic); tw/track sync with master conflicts. Idle
   trees (coreslider, onewaylag, split6, stationids, timetables): bring up after 0.4.6.

LIVE v0.4.5. 8800 = master always (809a0d16); the track build only at /trainworld-track/ (?track), own saves, no
account (snaptrack.py). Her North America save is level 42: opens only there; backups in artifacts/trainworld/savedl/.
Cap 14 again from 01:4x (her word). Never raise the cap without her explicit word (her 00:4x), not even back to 14 or the
overnight max. Reserve loop back to -Reserved 2; cores 0-1 hers (reserve_cores.ps1 -Loop 30, now never touches terminals). Bench: perf_bench.mjs,
ledger notes/perf-ledger.jsonl.

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
