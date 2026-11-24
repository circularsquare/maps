# Orchestrator handoff, trainworld

Orchestrator: session 3793ea75-3fa8-4643-b04d-bea05a94ef29, name maps-d9 [d2420d], since 2026-11-24 20:20 UTC
Updated: 2026-11-24 22:48 UTC. Live state: `venv\Scripts\python.exe trainworld/tools/handoff.py status`;
diary notes/handoff-log.md. Her time UTC-5. Core cap 14 (her 20:12, "take on more tasks at once if you have bandwidth").

## Next
1. Running (briefs in my scratch <agent>/brief.md): cutstretch T-4153 s4, busdemand T-4224 (her bus call), passx T-4048
   (her remove-button design), busstops T-4080 stages 1-2 (saves: own checker before landing), mcheck (T-4214's checker,
   tree mapcfix), check12 (batch checker on master). T-2188 landed 9f4405ca5 after qcheck (T-4237 filed from it).
2. T-4214: mapcfix fixed the extra cost (the patch set world: panels and layers ran twice per edit); its last bench was
   noisy. After mcheck: hold starts for a quiet window, run the tree's last edit_bench, land. Master baseline:
   artifacts/trainworld/editbench/baseline.json (b7b9e4ee; NYC track built 437 ms). 8800 = f733d3bf, smoke PASS.
   Waiting on her: T-4220 design yes (notes/T-4220.md). Landed: T-4241 (review fixes), T-4234 (bench stop click; rewrite the
   edit_bench baseline in the next quiet window: edit_bench --save asia,nyc --baseline on a master Vite).
3. "Read before you run" rules written 22:50 at her OK (CLAUDE.md, orchestrator.md): briefs start from the code, batch
   checkers are code reviews plus Gemini on every risky batch ("if we run out i'll likely upgrade the sub").
   reswait: T-4220 design (read-only, Gemini review; to her before building).
   settlestop T-4239 (her bricked interlined saves: the settle that never holds still), stationwait T-4240 (her Asia
   station wait, step table first). 8800 = 4e8c50f3 (all landings incl. T-2188, T-4214 C, T-4224 bus costs), smoke PASS 23:27.
   review13: batch code review + Gemini of T-2188, T-4237, T-4214 C, T-4153 s4, T-4224 (first under the new practice).
   Started 23:30 at her "agents + cores/2 < 8" rule: keepreview (T-4080 s1-2 review + Gemini, before landing; save level
   40, passx takes 41 if second), busload T-4193 (Opus), benchstop T-4234 (Sonnet), fixes13 T-4241 (review13's findings).
   review13 passed all five landings by reading; findings in notes/T-4241.md. passx done (T-4048, save level 40 too):
   passreview reviewing it; the second of busstops/passx to land takes level 41.
   Next when the formula allows: corridors3 (T-4199 stage 2: merge master into tree corridors, its demand worker files
   moved under T-2188/T-4237/T-4241; then NYC +13 MB), T-4238 (bus weekend headway, read first).
4. Then T-4080 stages 3-6, T-4214 stages A, D, E, T-4237, T-4220, T-4231, T-4193 (clock worker files after T-4214
   lands), T-4149/T-4150, T-4230, T-4187, T-3160, T-4199, T-4233, T-4234, T-4222 P3. After that, her big three: T-3444
   track rework, T-3742 schedule nudging, T-3694 + T-3456.

## Her words
- 11-24 23:35: "text ack" for T-4048's "Connection between {a} and {b} removed." (T-4223's two loop refusals not yet acked).
- 11-24 22:4x: "we're maybe relying a bit too much on testing ... you guys get a lot more mileage by thinking about actual
  code than by trying to test. the ways in which you think to test don't really align super well with how people play.
  how could we formalize this"; Gemini adversarial review "generally speaking". loopcase not needed (T-4235 closed).
- 11-24 22:3x: loop fix not trusted for ordinary circles; don't claim it fixed (c004 hidden). Counts once each: yes.
  Lines past 99 keep numbers (T-4236 landed). Buses: "1.1x bus minute mult, can maybe adjust more later, and do waits
  at least 2 minutes. and also 500m reach". Bus stop naming design OK. All in DECISIONS.
- 11-24 20:1x on buses: "bus ridership is just maybe a bit too high overall"; a perceived cost should scale with ride
  time ("the annoying part is being on the bus"); 0.7-minute headways unrealistic. Answered by T-4224 above.
- 11-24 18:5x: shipped 0.4.6; then "work at the root cause / try something like a silo approach ... deprioritize the
  shorter term surface level fixes" (DECISIONS T-4097). Root-cause agents still measure their own change.
- 11-24: no small fix per case for quick-edit sequences: general fixes, Gemini consulted on designs (she likes it).
- 11-24 13:10-13:57: she says when to publish (never ask); asianita save is a stress test only; no "less memory"
  changelog items when memory is up overall; changelog one line, she trims it on the board; keep cores 0-1 free.
- 11-24: testing budget table in CLAUDE.md; don't reproduce her bugs unless the cause is unclear; disk low = clear to 60 GB.
- 11-23: answer her first; never let agents overbook or collide; agents 2-8 by load; tasks about an hour; you don't
  investigate, agents do; cycle at 500k; no drafted player replies; times as percent of the wait.
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
