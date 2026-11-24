# Orchestrator handoff, trainworld

Orchestrator: session efd2dd48-edce-4472-b6ce-2993de0c867f, name maps-1c [b74f21], since 2026-11-24 17:11 UTC
Updated: 2026-11-24 19:56 UTC. Live state: `venv\Scripts\python.exe trainworld/tools/handoff.py status`;
diary notes/handoff-log.md. Her time UTC-5.

## In flight
- 0.4.6 published 11-24 19:36 UTC (data 2026-11-24, website 50454ff, then changelog trimmed per her list, website
  7f9ec77, build mwv3o291; open-source 3e9c056; master pushed e442d1ba1). perf_bench NYC marked published v0.4.6.
  NYC bench vs rc046: map 6.8 -> 9.7 s (unexplained, also seen 11-24 earlier), trips +5%, cpu -14%, tab -6%; she
  said ship. Since: T-921 commutechunks landed (3ae084a0d, sim; changelog c001), T-4221 closed (dup of T-4029).
- Running (root cause, her 11-24 call; up next on the board): netland (Sonnet, tree editnetwork, 6182: T-4153 s2
  merge master + checks + 1 edit_bench, then land), mapC (Opus, tree mappatch, 6183: T-4214 C1+C2 clickable map
  patched from the edit's touched records; waiting clicks released on the patch), quickend (Opus, tree quickend,
  6184: T-2188 city edits end at the touched cities' re-solve, idle refresh rule, drift on NYC). Briefs in scratch.
- Next after those: T-4214 stages A (stations without nameFor wait) and D (track runs without placing lock), then
  T-4149/T-4150, T-4220, T-4193, T-4187/T-3160, T-4080/T-4083, T-4199 (corridors kept, fix NYC +13 MB first),
  T-4222 (P3). Batch checker once mapC + netland + quickend land (riskiest: edits, undo, waiting clicks, drift).
- Core cap 10 (her 17:52; tell her if work goes core-heavy). reserve_cores: my retry loop bvc27j9av.
- Uncommitted SPEC.md ("What is covered"), docs/code-tour.md, docs/interface.md, checks/old moves, checks INDEX,
  tools/checks_index.py are not ours (an 11-23 checks reorganisation): leave them; landings touching them use the
  patch dance (orchestrator.md Traps).

## Her words
- 11-24 19:2x: up next set as proposed (board queue: T-4221, T-4153 s2, T-4214 stages, T-2188, T-4149/4150, T-4220, T-4193, T-4187, T-3160, T-4080/4083, T-4199 kept). "the next actual non-small features ... mostly track rework and also schedule nudging and bulk station improvement interface + crowding along the line panel. these are the big 3. but they can come after all this up next stuff" (T-3444, T-3742, T-3694, T-3456).
- 11-24 18:5x: publish 0.4.6 now ("lets just push whatever we have now, its good enough. i loaded up 0.4.5 and its definitely worse"), one NYC bench; then "work at the root cause / try something like a silo approach ... once we start working at that we really should deprioritize the shorter term surface level fixes". Gemini asked to check the plan (artifacts/trainworld/gemini/rootcause-plan-*.md). No measuring rounds ("i dont want to measure anything").
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
