# Orchestrator handoff, trainworld

Orchestrator: session 28722621-86b3-4444-8131-f3aee02b18dd, name maps-6b [d9ec47], since 2026-11-23 20:08 UTC
Updated: 2026-11-24 01:56 UTC (cycling at 500k). Live state: `venv\Scripts\python.exe trainworld/tools/handoff.py status`;
diary notes/handoff-log.md. Her time UTC-5.

## In flight (adopt from the board; agent ids for SendMessage while my tab lives)
- Edit speed, plan notes/T-4144.md; her "do all, land all, one message when done". Landed: base T-4145, lines path
  T-4114, ghost T-4136, first click T-4143. Running: editnetwork T-4153 (TOP: bounded drawn network on master, her
  call), editstations T-4146 (a832c397), editlines T-4147 (a3199ea2), editpieces T-4148, editworker T-4149 (a0d0baab;
  rows-first waits for B), trackstream T-4155/T-4142/T-1031 (a2b95733), retime T-4132. Queued: T-4150, T-4133/4134.
  Before telling her: one Opus review of all show-at-click code, a smoke check that no pending entry outlives the final
  state, a batch check (refusals, undo after shown edits, charges) incl. rampport 380d8fc7, regionfix 8e1eb7ba,
  lineclick ef99d979, junctionmsg 9719e410, base 0c384f9e; then rebuild 8800 and one message to her.
- volmove T-4159 (opus): saved games onto the new 50 GB Hetzner volume; her yes 01:5x covers the move and cap deploy.
  It sends a phase-1 plan; read it, then tell it go (SendMessage to its agent id from the board/report).
- Perf beyond edits: perfcompare T-4157 (master vs v0.4.5, all three saves, memory too), busmerge2 T-4158 (wider bus
  stop merges: her pick), commutechunks T-921 (shift: numbers to her), zoneopen T-4051, memwasm T-4156 (a17f9bab).
- Her open asks in chat: AWS own account (guide notes/aws-own-account-setup.md; when she says "aws account ready,
  profile anita-own", brief an agent for a bench machine). Resend Pro and R2 purge asked, unanswered.
- Strings she was shown (not yet acknowledged): T-4152 "Track has to line up with the track it joins.", install note
  (her own words), station hover 600 m rows.
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
