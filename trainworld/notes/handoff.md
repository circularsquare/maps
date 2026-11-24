# Orchestrator handoff, trainworld

Orchestrator: session efd2dd48-edce-4472-b6ce-2993de0c867f, name maps-1c [b74f21], since 2026-11-24 17:11 UTC
Updated: 2026-11-24 17:57 UTC. Live state: `venv\Scripts\python.exe trainworld/tools/handoff.py status`;
diary notes/handoff-log.md. Her time UTC-5.

## In flight
- 0.4.6 (she says when to publish; never ask). Master ef9fb81f4 has T-4194, T-4202 s2, T-4210 ghost (f6e93c039).
  8800 rebuild to f6e93c039 started 17:55 (scratch snapbuild2.log); tell her when it's up (she wants to test the ghost).
  Checker check9 (notes/batchcheck-2026-11-24-check9.md): T-4202 s2 PASS; T-4194 partial (Rust test only); T-4202 s1
  and T-4204 not tested; t3084 ferry "Edit route" step FAILS on head. Running for it (17:56): waitclick (Opus, tree
  waitclick, 6173: T-4214 option B, her "ok we can continue with waiting clicks"), ferryfix (Sonnet, tree ferryfix,
  6174). When both land: checker round 2 (waiting clicks, ghost, ferry, plus check9's untested T-4194 edits, T-4202 s1,
  T-4204 hovers; Montana over NYC for time), 8800 rebuild, smoke, release perf_bench nyc/tokyo/belgrade, changelog
  review, playtest list, her OK on strings.
- Core cap 10 (her 17:52). She: "if we're doing core intensive work let me know, thats probably an indicator somethings
  wrong since we only have like 2 things running right now."
- shonei-server deployed 17:49 (T-4161, her yes) and pushed; SAVE_LEVEL 39. T-4212 closed (her: one-off).
- After 0.4.6 (proposed, her answer pending): land corridors T-4199 s2 (fix NYC +13 MB first), editstations T-4146;
  then T-4214 later stages (map patched after each edit = T-4097 stream D) beside T-2188, then T-4220.
- Held for her word: T-4153 s2 (tree editnetwork; add mm.ribbons.delete(e) at lineBands.ts:571 if landed); T-921 (tree
  commutechunks); T-4156 P3.
- T-4112's tw/track changelog item is parked in next-track.json: hide the port's duplicate at release.
- Uncommitted SPEC.md ("What is covered"), docs/code-tour.md, docs/interface.md, checks/old moves, tools/*.py are not
  ours (likely Gemini in Antigravity): landings touching them use the patch dance (orchestrator.md Traps).
  .tw-trees/modecolors2 is a leftover non-worktree folder: leave it. reserve_cores: my retry loop bvc27j9av.

## Her words
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
