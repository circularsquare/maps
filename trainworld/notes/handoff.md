# Orchestrator handoff, trainworld

Orchestrator: session 8e9c7c49-010f-4c78-a0db-d5be0797af3f, name maps-3e [111c39], since 2026-11-24 23:50 UTC
Updated: 2026-11-25 01:07 UTC. Live state: `venv\Scripts\python.exe trainworld/tools/handoff.py status`;
diary notes/handoff-log.md. Her time UTC-5. Core cap 10 from 00:52 UTC (her: "for a few hours"; back to 14 around 04:00 UTC as she asked, say so when doing it). Start more when agents + claimed cores / 2 is under 8 (her 11-24).

## Next
0. Usage stance (her 11-25): Claude usage runs out before Gemini's. Claude agents take P1/P2 code work; small clear
   tasks and PR ports go to Gemini code agents (gemini_agent.py, 1 core each at cap 10); reading and review to Gemini.
1. Running (briefs in maps-3e's scratch <agent>/brief.md): cutstretch3 (T-4153 s6), settlefix (Opus, T-4239 on tree settlestop:
   Gemini's findings, then land with finish --task T-4239), stationclick (T-4214 A + T-4249 + her station removal), rowsedit (T-4149 worker rows), pageedit
   (T-4150 page take), fixes14 (Sonnet, T-4260: batch14 findings). Gemini: faremult (T-4253, third try, 1 core).
   Landed this session: T-4240, T-4220 s1-2, T-4193, T-4048 (save level 41), T-4252 (tools), T-4231 (Gemini, round 1
   of 3: its permissions note: "|" is refused inside quoted git grep patterns), T-4153 s5 (5ec788b72). 8800 is behind.
2. 8800 = bb321567e, smoke PASS 01:02. Batch 14 reviewed by Gemini (artifacts/trainworld/gemini/batch14-review.md).
   T-4220 s2 landed 6b5df61c6 (stages 3-9 left: next agent aside3, Opus). faremult (Gemini) failed twice on a full cap:
   retry when cores free.
3. Gemini code trials next: T-4247, T-4254, T-4255, T-4256, T-4258, T-4259 (3 counted rounds ask about permissions).
   Claude next: T-4257 (bus fastest route), T-4251 (busways), T-4243 speedometer, T-3742 after settlestop lands,
   T-4214 D/E, corridors3 (T-4199 s2), T-4080 s3-6, T-4230, T-4187, T-3160, T-4233, T-4242 design.
4. Waiting on her: T-4223's two loop refusals not yet text-acked; closing #21 and #24 on GitHub is hers; T-4250, T-4248.
5. Rewrite the edit_bench baseline in a quiet window (master Vite, `edit_bench.mjs --save asia,nyc --baseline`).

## Her words
- 11-25 00:54: "when you start pythons, please make sure terminals dont pop up on my screen" (gemini_agent.py used DETACHED_PROCESS; fixed, CLAUDE.md Shell).
- 11-25 00:51: "the game reworking the whole network after each edit is really the target here and everything else is kinda just flavor. we should mostly be focusing on isolating the changes and cutting out the impacts on the broader map." (DECISIONS T-4097; T-4149, T-4150 raised to P1.)
- 11-25 00:40: "we should not close prs without comment" (orchestrator.md standing calls); #16, #22 closed bare, comments drafted.
- 11-25: Claude usage runs out before Gemini's: Gemini takes reading and review, Gemini code agents to come (T-4252;
  may run commands unattended in their tree; ask them about permissions after their first 3 rounds). PR calls: #16, #22
  closed; port #30, #34, #35 ("keep the ui impact minimal"); #32 becomes "fastest route rather than the shortest"
  (T-4257). 180th meridian: "if people want to build it, they should ideally be able to". Airports later, P3, "dont
  reinvent the wheel". T-3742 after settlestop (my call). Pads filed as T-4243 to T-4253; tram stop parking, bus stops
  on busways, fare multiple to 10x in DECISIONS. T-4242 "p2 at least".
- 11-24 23:35: "text ack" for T-4048's "Connection between {a} and {b} removed." (T-4223's two loop refusals not yet acked).
- 11-24 22:4x: "we're maybe relying a bit too much on testing ... you guys get a lot more mileage by thinking about actual
  code than by trying to test. the ways in which you think to test don't really align super well with how people play.
  how could we formalize this"; Gemini adversarial review "generally speaking". loopcase not needed (T-4235 closed).
- 11-24 20:1x on buses: "bus ridership is just maybe a bit too high overall"; a perceived cost should scale with ride
  time ("the annoying part is being on the bus"); 0.7-minute headways unrealistic. Answered by T-4224 above.
- 11-24 18:5x: shipped 0.4.6; then "work at the root cause / try something like a silo approach ... deprioritize the
  shorter term surface level fixes" (DECISIONS T-4097). Root-cause agents still measure their own change.
- 11-24 13:10-13:57: she says when to publish (never ask); asianita save is a stress test only; no "less memory"
  changelog items when memory is up overall; changelog one line, she trims it on the board; keep cores 0-1 free.
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
- reserve_cores: maps-3e's retry loop (task b19u3mylb, powershell PID 29592 since 01:03). It now follows the ledger's cap
  (cap 10 = cores 6-15) and pins Gemini runners; a stale 11-24 08:11 copy (old code) was stopped.
