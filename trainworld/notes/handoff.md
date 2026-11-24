# Orchestrator handoff, trainworld

Orchestrator: session 3a819365-cd3a-4850-a112-6396d17c97b4, name maps-06 [4d6442], since 2026-11-24 13:06 UTC
Updated: 2026-11-24 13:58 UTC. Live state: `venv\Scripts\python.exe trainworld/tools/handoff.py status`;
diary notes/handoff-log.md. Her time UTC-5.

## In flight (all maps-06's)
- 0.4.6: she says when to publish. 8800 = 4c096347. Landed since: T-4148, T-4206 timing log, T-4208 ghost, T-4204 s2
  (restart only when memory short; check NYC map +2.5 s in the release perf_bench), T-4209 time-left display.
  0.4.6 waits for (her word): settle1 T-4194 opt 1+2 (Sonnet, tree settle, 6144; measure stop adds on headway lines),
  stalefigs T-4202 s2 (Opus 6163), editstops T-4147 (Sonnet 6165). Then: batch checker (T-4194 + T-4202 + T-4204 +
  keepSet a3a084b87), smoke, perf_bench all three, changelog review, 8800 rebuild, publish skill on her word.
  Strings acked 15:3x (sections 2+4 of notes/release-0.4.6.md); new strings since: list at publish.
- Not for 0.4.6: corridors2 T-4199 s2 (Opus, tree corridors, 6155; NYC gain asked), ghostspeed T-4210 (Opus 6164).
- After 0.4.6: T-2188 (city edits end at the quick result, her yes), T-4146 (after editstops), T-4211, T-4193.
- Held for her word: T-4153 s2 (tree editnetwork; add mm.ribbons.delete(e) at lineBands.ts:571 if landed); T-921 (tree
  commutechunks); T-4156 P3. T-4161 shonei-server 7bdd559 unpushed: deploy on her yes, then push.
- T-4112's tw/track changelog item is parked in next-track.json: hide the port's duplicate at release.
- Uncommitted SPEC.md, DECISIONS.md, docs, checks/old moves, tools/asks.py etc. partly not ours (likely Gemini in
  Antigravity): patch out to land if a landing touches them. Ours: TASKS lines, DECISIONS entries (Hokkaido, T-4202,
  T-2188, T-4199 rule, T-4204's). A tree on modecolors2 may linger (held by a shell): agent-tree.mjs remove modecolors2.
- Core cap 14. reserve_cores.ps1 -All -Loop 30 in maps-06's background. Gemini: her ask or our suggestion.

## Her words
- 11-24 13:3x-13:57 playtest: she says when to publish; asianita save is a stress test only; "disembarking"; bus town tint
  20%; Hokkaido zones by city; keep "reverse"; leave bus vs tram (T-4201 closed); click again cycles all routes (T-4046).
- 11-24 13:10 (via maps-20): no "less memory" changelog items when overall memory is up (c019, c063 hidden; c064 memory
  clause cut; rule in CLAUDE.md); changelog items one line, she trims them herself; computer froze: keep cores 0-1
  free (T-4195 done 1474aeb4a: reserve_cores.ps1 -All -Loop 30 runs in maps-06's background, covers every session).
- 11-24 13:12: Gemini Neon suggests rewordings for all changelog and player text; we reject incorrect or misleading ones,
  accept the rest, and list what changed for her. Only the 0.4.6 text (T-4196), not the rest of the game.
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
