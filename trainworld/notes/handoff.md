# Orchestrator handoff, trainworld

Orchestrator: session 8e9c7c49-010f-4c78-a0db-d5be0797af3f, name maps-3e [111c39], since 2026-11-24 23:50 UTC
Updated: 2026-11-24 23:50 UTC. Live state: `venv\Scripts\python.exe trainworld/tools/handoff.py status`;
diary notes/handoff-log.md. Her time UTC-5. Core cap 14. Start more when agents + claimed cores / 2 is under 8 (her 11-24).

## Next
1. Running in maps-d9's tab (adopt from the inbox; briefs in maps-d9's scratch <agent>/brief.md): settlestop T-4239
   (Opus: the train-count settle stops once lines hold nearly still; her "fewer rounds", DECISIONS), stationwait
   T-4240 (Opus: her Asia station wait, step table first), busload T-4193 (Opus: bus streets before the load).
   T-4080 stages 1-2 landed a958d1870 (save level 40, LBK1); keepreview's findings for stage 3 are in notes/T-4080.md.
   Tree passx (T-4048) passed review except F1 (sim/src/demand/kernel/search.rs:389: a separated pair rejoined in one
   complex gets no walk): a fresh agent on tree passx fixes F1 (+ the cheap F2, F4-F7 in its notes), then lands at save
   level 41 (save.rs LEVEL, saveFile.ts SAVE_LEVEL, the pin to (41, 44), its block after LBK1).
2. 8800 = 4e8c50f3, smoke PASS. On master since: T-4241 (review13 fixes), T-4234 (bench tool). Rebuild 8800 after the
   next round of landings (snap8800 + smoke against 8800; await the whole chain in one script).
3. T-4220 approved 23:46 ("ok, we can build 4220"): build its 9 stages (notes/T-4220.md), stage 1 first, Opus.
   Waiting on her: T-4223's two loop refusals not yet text-acked. T-4242 filed from her tw-feedback question (P3).
4. Next when the formula allows: corridors3 (T-4199 stage 2: merge master into tree corridors, its demand worker
   files moved under T-2188/T-4237/T-4241; then NYC +13 MB), T-4238 (bus weekend headway, read first), T-4080 stages
   3-6 after busstops lands, T-4214 stages A/D/E, T-4231, T-4149/T-4150, T-4230, T-4187, T-3160, T-4233, T-4222 P3.
   After that her big three: T-3444 track rework, T-3742 schedule nudging, T-3694 + T-3456.
5. Rewrite the edit_bench baseline in a quiet window (master Vite, `edit_bench.mjs --save asia,nyc --baseline`): it
   has no NYC stop click yet. Batches are reviewed by reading plus Gemini (docs/orchestrator.md "Landing").

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
