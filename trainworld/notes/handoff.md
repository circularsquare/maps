# Orchestrator handoff, trainworld

Orchestrator: session 3793ea75-3fa8-4643-b04d-bea05a94ef29, name maps-d9 [d2420d], since 2026-11-24 20:20 UTC
Updated: 2026-11-24 20:34 UTC. Live state: `venv\Scripts\python.exe trainworld/tools/handoff.py status`;
diary notes/handoff-log.md. Her time UTC-5. Core cap 14 (her 20:12, "take on more tasks at once if you have bandwidth").

## Next
1. Running. Adopted from maps-1c 20:20 (reports reach the inbox; briefs in maps-1c's scratch <agent>/brief.md):
   mapC T-4214 C1+C2 (tree mappatch; clickable map patched from each edit's touched records), quickend T-2188 (tree
   quickend; drift over 15 edits goes to her before landing), buswait T-4052 (tree buswait), loopclose T-4223 (tree
   loopclose), busopts (read-only, notes/T-4224.md table for her bus pick). Mine (briefs in my scratch <agent>/brief.md):
   linesym T-4226 (Sonnet, tree linesym; line symbol in other languages, many players), bandorders T-4153 stage 3 (Opus,
   tree bandorders; pass-through test for band orders). Done: mapslow (no regression, landed 4bd32c838, T-4230 filed
   from it); busstopid (T-4080/T-4083 now {waits:Anita}, notes/T-4080-design.md; T-4047 folded); sortreq (93 player
   reports, 171 groups; bugs filed T-4226 to T-4229).
2. When mapC, quickend and buswait land: one batch checker on master (edits, undo, waiting clicks, figure drift),
   8800 rebuild, smoke. 0.4.6 is live (published 19:36 UTC, changelog trimmed and re-pushed 19:47).
3. Then T-4227 (weekend services bug; touches scheduling, maybe buswait's service.rs: after buswait lands), T-4214
   stages A and D (after mapC), T-4149/T-4150, T-4220, T-4193, T-4230, T-4187 then T-3160, T-4080/T-4083 once she
   picks, T-4199 (fix NYC +13 MB first), T-4222 P3. Board up next holds this.
4. After up next: her big three, T-3444 track rework, T-3742 schedule nudging, T-3694 bulk station improvements panel +
   T-3456 crowding along the line.

## Her words
- 11-24 20:1x on buses: "bus ridership is just maybe a bit too high overall"; leaning away from changing speed ("in some
  places buses do run fast ... balancing it with motorways seems quite hard"); a perceived cost should scale with ride
  time ("the annoying part is being on the bus"); bunching ("schedules are super unreliable"); "people are running buses
  at like 0.7min headways ... not sure if thats so realistic". busopts measures: bus time 1.2-1.3x, bunching waits,
  a street cap like BRT's 100/h (60/h), 500 m reach, every-other-stop Auto, fare 1.0x. My lean: bus time + street cap.
  Her pick decides; a demand factor waits for her word.
- 11-24 18:5x: shipped 0.4.6 ("its good enough. i loaded up 0.4.5 and its definitely worse"); then "work at the root
  cause / try something like a silo approach ... deprioritize the shorter term surface level fixes" (DECISIONS T-4097).
  "i dont want to measure anything" was about holding the release; root-cause agents still measure their own change.
- 11-24 19:2x: up next as set; T-4199 corridors kept; T-921 landed (she: dense-city gain is worth more).
- 11-24: no small fix per case for quick-edit sequences: general fixes, Gemini consulted on designs (she likes it).
- 11-24 13:10-13:57: she says when to publish (never ask); asianita save is a stress test only; no "less memory"
  changelog items when memory is up overall; changelog one line, she trims it on the board (start-menu line now
  editable there); keep cores 0-1 free (reserve_cores -All loop).
- 11-24: testing budget table in CLAUDE.md; don't reproduce her bugs unless the cause is unclear; disk low = clear to 60 GB.
- 11-23: answer her first; never let agents overbook or collide; agents 2-8 by load (she now allows more); tasks about
  an hour; you don't investigate, agents do; cycle at 500k; no drafted player replies; times as percent of the wait.
- 11-19: "player complaints take precedent over other new features"; full playtest list before every publish.
- Standing: every publish, deploy and website push needs her yes, none overnight; advisor only for big design
  decisions; BRT stations for BRT lines only.

## Only a person knows
- Uncommitted, not ours (an 11-23 checks reorganisation): SPEC.md "What is covered", docs/code-tour.md,
  docs/interface.md, checks/old moves, checks/INDEX.md, tools/checks_index.py. Landings touching them: patch dance.
- Uncommitted on purpose: TODO.md (hers), notes/asks-log.jsonl, notes/board-live.json, changelog/roadmap.json,
  maps .claude/settings.json, notes/T-3397.md, tools/ops/archive_bad_saves.py.
- Trees kept: _track (tw/track), tracksnap (snaptrack.py), _seed (Rust cache seed), corridors (T-4199 stage 1).
- Unmerged branches with no tree, left on purpose: tw/hotfix-10d, tw/release-021, tw/release-031, tw/onegraph-measured.
- A task with stages left: `agent-tree.mjs land` + `remove`, not `finish`; swap the taking mark by hand.
- reserve_cores: maps-d9's retry loop (task bpobbmf27) dies with its tab; start one in yours.
