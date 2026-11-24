# Orchestrator handoff, trainworld

Orchestrator: session efd2dd48-edce-4472-b6ce-2993de0c867f, name maps-1c [b74f21], since 2026-11-24 17:11 UTC
Updated: 2026-11-24 17:11 UTC. Live state: `venv\Scripts\python.exe trainworld/tools/handoff.py status`;
diary notes/handoff-log.md. Her time UTC-5.

## In flight (maps-06's agents; reports reach the inbox)
- 0.4.6: she says when to publish; never ask. 8800 = c0d31916. On master since: T-4204 s2, T-4209, T-4213 bench gates.
  0.4.6 waits for (her word, "yes want all"): settle T-4194 (tree settle, reported 17:10): headway-only 0.25 loses the
  gain (jwxidghd circling lines are count lines: stop add 8.5-9.7 s, was 11-12; 0.25 on all lines gave 2.3-2.7 s but
  moves T-3931's express timetable so its test fails). Decide: 0.25 on all lines + loosen T-3931 (agent's pick, ask
  her: it changes a timetable), or ship option 1 only;
  T-4212 0 riders after "loaded" (agent zeroriders reported 17:10: not reproduced on fresh profile, old-build figures,
  Montana, Asia; suspects: no zone needed solving yet / packs failed; needs her diagnostics from such a load; notes in
  tree zeroriders, land them; decide whether it still blocks 0.4.6); stalefigs T-4202 s2 (done in
  tree stalefigs, held until T-4212's cause is known: may be the same path). T-4214's wrong deletion if a regression.
  Then: batch checker (T-4194, T-4202, T-4204, keepSet, t3084 ferry step base vs head; 25 min, no benchmarks), smoke,
  release perf_bench nyc/tokyo/belgrade 1 run each with --baseline (writes master baselines; check NYC map +2.5 s),
  changelog review (publish skill), 8800 rebuild, playtest note to her, publish skill on her word.
- Also running: ghostspeed T-4210 (Opus 6164: at-click ghost must follow the real curve and not blink; her 8800 notes
  sent), editseq T-4214 (Opus 6169: her three quick-edit sequences, design + Gemini, no repro per new rule; design goes
  to her before building; a contained fix for station-then-delete may come), diskclean (Sonnet: C: to 60 GB free).
- Parked in trees: corridors T-4199 s2 (fix NYC +13 MB download, then land; after 0.4.6), editstations T-4146 (run its
  checks, fix one-frame outline, after 0.4.6). After 0.4.6: T-2188 (city edits end at the quick result, her yes).
- Held for her word: T-4153 s2 (tree editnetwork; add mm.ribbons.delete(e) at lineBands.ts:571 if landed); T-921 (tree
  commutechunks); T-4156 P3. T-4161 shonei-server 7bdd559 unpushed: deploy on her yes, then push.
- T-4112's tw/track changelog item is parked in next-track.json: hide the port's duplicate at release.
- Uncommitted SPEC.md, DECISIONS.md, docs, checks/INDEX.md (regenerated 17:09), checks/old moves, tools/asks.py are
  partly not ours (likely Gemini in Antigravity): patch out to land if a landing touches them (orchestrator.md Traps).
  Ours in them: TASKS lines, DECISIONS entries (Hokkaido, T-4202, T-2188, T-4199 rule, T-4204). Tree modecolors2 may
  linger (held by a shell of maps-06): agent-tree.mjs remove modecolors2.
- Core cap 14. reserve_cores.ps1 -All -Loop 30 runs in maps-06's background: start one in yours (it exits if one runs).

## Her words
- 11-24 13:3x-13:57 playtest: she says when to publish; asianita save is a stress test only; "disembarking"; bus town tint
  20%; Hokkaido zones by city; keep "reverse"; leave bus vs tram (T-4201 closed); click again cycles all routes (T-4046).
- 11-24 13:10 (via maps-20): no "less memory" changelog items when overall memory is up (c019, c063 hidden; c064 memory
  clause cut; rule in CLAUDE.md); changelog items one line, she trims them herself; computer froze: keep cores 0-1
  free (T-4195 done 1474aeb4a: reserve_cores.ps1 -All -Loop 30 runs in maps-06's background, covers every session).
- 11-24 afternoon: testing budget adopted (CLAUDE.md table, tool-enforced); don't reproduce her bugs unless the cause is
  unclear; disk low means clear to 60 GB free; she feels agents' edit-speed fixes are hacky: fixes, not hacks.
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
