# Orchestrator handoff, trainworld

Orchestrator: session 8e9c7c49-010f-4c78-a0db-d5be0797af3f, name maps-3e [111c39], since 2026-11-24 23:50 UTC
Updated: 2026-11-25 00:10 UTC. Live state: `venv\Scripts\python.exe trainworld/tools/handoff.py status`;
diary notes/handoff-log.md. Her time UTC-5. Core cap 14. Start more when agents + claimed cores / 2 is under 8 (her 11-24).

## Next
1. Running, maps-d9's tab (adopt from the inbox): settlestop T-4239 (settle stops once lines hold still). maps-3e's tab
   (briefs in its scratch <agent>/brief.md): passfix (tree passx: merge master, review fixes F1, F2, F4-F8, save level
   41 after LBK1; T-4048's line was archived early, so land passx with `land` + `remove`, no --task), geminirun
   (T-4252, main tree tools; also fixing gemini_audit.py stopping after one turn), cutstretch2 (T-4153 stage 5),
   aside2 (T-4220 stage 2), stationclick (T-4214 stage A + T-4249). Landed this session: T-4240, T-4220 s1, T-4193.
2. Master 6fb7cfa5e (T-4240 landed 00:08). Batch review (Gemini) due over T-4241, T-4234, T-4080 s1-2, T-4240, then passx.
3. Gemini read-only runs: T-4250 scope (out notes/T-4250.md), T-4248 scope (out notes/T-4248.md), PR survey (her 11-25 ask; brief artifacts/trainworld/gemini/prs-2026-11-25/brief.md, out
   notes/pr-survey-2026-11-25.md). 8800 = 6fb7cfa5e, smoke PASS 00:18. Next when the formula allows: T-4252 (Opus,
   the Gemini code-agent runner), then T-4231. First Gemini code trials once T-4252 works: T-4231, T-4247, T-4253 (fare multiple to 10x) (Sonnet, Apply to all weekend bug); T-4153 stage 5 (drawn network for a cut
   line, P1); T-4214 A/D/E (P1); T-4249 (T-4240's leftovers); T-4250 (relief line empty beside a crowded one, read the
   code); T-4251 (bus/BRT stations on busways); T-4243 speedometer; T-4247 tram stop parking; T-3742 nudging after
   settlestop lands (her 11-25: maybe wait for the trip-speed work; my call). Then corridors3 (T-4199 stage 2), T-4238,
   T-4080 stages 3-6, T-4149/T-4150, T-4230, T-4187, T-3160, T-4233, T-4242 design, T-4248 scope.
4. Waiting on her: T-4223's two loop refusals not yet text-acked.
5. Rewrite the edit_bench baseline in a quiet window (master Vite, `edit_bench.mjs --save asia,nyc --baseline`).

## Her words
- 11-25 00:22: Gemini code agents may run commands unattended inside their own worktree; "for the first 3 rounds, lets ask the geminis after they're done whether they have any thoughts on permissions" (sent to geminirun, T-4252).
- 11-25 00:17: "we're actually like probably going to run out of claude usage before gemini at this rate so in general we should be using gemini to handle more reading tasks, and more review"; wants Gemini Neon agents making code changes like Claude subagents (T-4252, DECISIONS, orchestrator.md Tools).
- 11-25 00:03: 180th meridian "if people want to build it, they should ideally be able to" (DECISIONS; T-4248 becomes scope then build). Airports: later, P3, "make sure we dont reinvent the wheel" (T-4244). T-3742: unsure, may wait for nearby trip-speed work; "if you think its fine, we can start it" (my call: after settlestop lands).
- 11-24 23:56 to 11-25 00:07 scratch pads: filed T-4243 to T-4251; tram stop parking and bus stops on busways allowed (DECISIONS).
- 11-24 23:55 (in maps-d9's tab): T-4242 weekend riders "p2 at least, its quite visible for some people" (raised to P2; design goes to her before building).
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
- reserve_cores: maps-3e's retry loop (task b19u3mylb; exits each pass while maps-d9's runs, takes over when that tab closes).
