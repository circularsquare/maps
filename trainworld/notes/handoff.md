# Orchestrator handoff, trainworld

Orchestrator: session 3a819365-cd3a-4850-a112-6396d17c97b4, name maps-06 [4d6442], since 2026-11-24 13:06 UTC
Updated: 2026-11-24 13:58 UTC. Live state: `venv\Scripts\python.exe trainworld/tools/handoff.py status`;
diary notes/handoff-log.md. Her time UTC-5.

## In flight (all maps-06's)
- 0.4.6: no publish ask, she says when. Candidate = master; 8800 = 61cbe801 (13:51: T-4196 rewording, T-4200,
  T-4203 Hokkaido names, reverse kept, sparseload). Smoke PASSED 13:28 at 0cca357a; rerun before publish. perf_bench
  rc046-f63b105b in ledger (belgrade CPU +24%, tokyo demand +11%; ask a-2026-11-24-3). Lists: notes/release-0.4.6.md
  (uncommitted); rewording review artifacts/trainworld/review/text-rewording-0.4.6.md.
- Landed since: sparseload dc48e20ef, stopclick 095480256 (T-4097 stays open). check8 (Sonnet, tree check8, port 6157)
  batch-checks both. Next 8800 rebuild after check8 + the next landings.
- Running: settle T-4194 (Opus, options, tree settle, 6144), corridors T-4199 s1 (Opus, tree corridors, 6155), bigload
  T-4202 s1 (Opus, tree bigload, 6156, asianita stress save; also times the less-detail setting), editpieces T-4148
  (Sonnet, 6158), routecycle T-4046 (Sonnet, 6159, click again cycles all routes), check8.
- Held for her word: T-4153 s2 (tree editnetwork; add mm.ribbons.delete(e) at lineBands.ts:571 if landed); T-921 (tree
  commutechunks); T-4156 P3; T-4124 numbers first. T-4161 shonei-server 7bdd559 unpushed, deploy on her yes, then push.
- T-4112's tw/track changelog item is parked in next-track.json: hide the port's duplicate at release.
- Uncommitted SPEC.md, DECISIONS.md, docs, checks/old moves, tools/asks.py etc. are partly not ours (likely Gemini in
  Antigravity): patch out to land if a landing touches them. Ours in them: TASKS.md task lines, DECISIONS Hokkaido entry.
- Core cap 14. reserve_cores.ps1 -All -Loop 30 runs in maps-06's background. Gemini: her ask or our suggestion.

## Next
1. Her answers: strings ack (a-2026-11-24-1), speed trade-offs (a-2026-11-24-3), T-4161 deploy, T-4153 s2 timing.
2. Land check8's batch; then T-4146/T-4147 (after T-4148, same files), T-4193. Track rework after 0.4.6.

## Her words
- 11-24 13:3x-13:57 playtest: she says when to publish; asianita save is a stress test only; "disembarking"; bus town tint
  20%; Hokkaido zones by city; keep "reverse"; leave bus vs tram (T-4201 closed); click again cycles all routes (T-4046).
- 11-24 13:10 (via maps-20): no "less memory" changelog items when overall memory is up (c019, c063 hidden; c064 memory
  clause cut; rule in CLAUDE.md); changelog items one line, she trims them herself; computer froze: keep cores 0-1
  free (T-4195 done 1474aeb4a: reserve_cores.ps1 -All -Loop 30 runs in maps-06's background, covers every session).
- 11-24 13:12: Gemini Neon suggests rewordings for all changelog and player text; we reject incorrect or misleading ones,
  accept the rest, and list what changed for her. Only the 0.4.6 text (T-4196), not the rest of the game.
- 11-24 05:0x: core cap 15 "for the night" (back to 14 when she's up); start T-3939/T-3160 overnight as cores free; T-4156 later (P3).
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
- A task with stages left: `agent-tree.mjs land` + `remove`, not `finish`; swap the taking mark by hand.
