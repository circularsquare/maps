# Orchestrator handoff, trainworld

Orchestrator: session 8e9c7c49-010f-4c78-a0db-d5be0797af3f, name maps-3e [111c39], since 2026-11-24 23:50 UTC
Updated: 2026-11-25 00:52 UTC. Live state: `venv\Scripts\python.exe trainworld/tools/handoff.py status`;
diary notes/handoff-log.md. Her time UTC-5. Core cap 10 from 00:52 UTC (her: "for a few hours"; back to 14 around 04:00 UTC as she asked, say so when doing it). Start more when agents + claimed cores / 2 is under 8 (her 11-24).

## Next
0. Usage stance (her 11-25): Claude usage runs out before Gemini's. Claude agents take P1/P2 code work; small clear
   tasks and PR ports wait for Gemini code agents (T-4252); reading and review go to gemini_audit.py.
1. Running, maps-d9's tab (adopt from the inbox): settlestop T-4239. maps-3e's tab (briefs in its scratch
   <agent>/brief.md): geminirun (T-4252; also fixing gemini_audit.py stopping after one turn, then re-running the
   T-4250 and T-4248 scopes), cutstretch2 (T-4153 s5), aside2 (T-4220 s2), stationclick (T-4214 A + T-4249 + her
   station removal), rowsedit (T-4149 worker rows by what an edit touched), pageedit (T-4150 page take).
   Landed this session: T-4240, T-4220 s1, T-4193, T-4048 (bb321567e, save level 41).
2. 8800 rebuilding to bb321567e (snapbuild3.log; smoke after). Gemini batch review due over T-4241, T-4234, T-4080
   s1-2, T-4240, T-4193, T-4220 s1, T-4048, once gemini_audit.py works.
3. Gemini code trials once T-4252 works: T-4231, T-4247, T-4253, T-4254, T-4255, T-4256, T-4258, T-4259. Claude next:
   T-4257 (bus fastest route), T-4251 (busways), T-4243 speedometer, T-3742 after settlestop, T-4214 D/E, corridors3
   (T-4199 s2), T-4080 s3-6, T-4149/T-4150, T-4230, T-4187, T-3160, T-4233, T-4242 design, T-4248 after its scope.
4. Waiting on her: T-4223's two loop refusals not yet text-acked; closing #21 and #24 on GitHub is hers.
5. Rewrite the edit_bench baseline in a quiet window (master Vite, `edit_bench.mjs --save asia,nyc --baseline`).

## Her words
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
- 11-24 22:3x: loop fix not trusted for ordinary circles; don't claim it fixed (c004 hidden). Counts once each: yes.
  Lines past 99 keep numbers (T-4236 landed). Buses: "1.1x bus minute mult, can maybe adjust more later, and do waits
  at least 2 minutes. and also 500m reach". Bus stop naming design OK. All in DECISIONS.
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
- reserve_cores: maps-3e's retry loop (task b19u3mylb; exits each pass while maps-d9's runs, takes over when that tab closes).
