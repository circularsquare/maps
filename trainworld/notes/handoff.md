# Orchestrator handoff, trainworld

Orchestrator: session 8e9c7c49-010f-4c78-a0db-d5be0797af3f, name maps-3e [111c39], since 2026-11-24 23:50 UTC
Updated: 2026-11-25 02:20 UTC (cycling at 500k). Live state: `venv\Scripts\python.exe trainworld/tools/handoff.py status`;
diary notes/handoff-log.md. Her time UTC-5. Core cap 10 since 00:52 UTC (her: "for a few hours", "be fairly aggressive
about 10 core cap"); put it back to 14 around 04:00 UTC and tell her. Start more when agents + claimed cores / 2 < 8.

## Next
0. Usage (her 11-25): Claude runs out before Gemini. Claude agents take P1/P2 code work; small clear tasks and PR ports
   go to Gemini code agents (`tools/gemini_agent.py start <name> --brief <f> --task T-x --new-tree --cores 1`; briefs
   say git grep with separate -e patterns); reading and batch or design review go to `gemini_audit.py`.
1. Running in maps-3e's tab (briefs in its scratch <agent>/brief.md; reports arrive in the inbox): settlefix (Opus,
   T-4239 on tree settlestop: checks Gemini's t4239-review.md, then `finish settlestop --task T-4239`), cutstretch3
   (T-4153 s6), rowsedit (T-4149), uploadedit (T-4262 renderer upload), busrebuild (T-4249 bus lines per station
   edit), aside3 (T-4220 s3). Gemini: pr30port (T-4256), pr29port (T-4255); land with `finish --coauthor`
   (Leni-Vienne: look up the noreply id with `gh api users/Leni-Vienne`), then a closing comment for her OK.
   Multi-stage tasks (T-4153, T-4220, T-4150, T-4214, T-4149) land with `land` + `remove`.
2. 8800: rebuild to a17111304 + smoke running in maps-3e's tab at 02:16 (scratch snapbuild4.log, smoke-a17111304*.log).
   When done, tell her it's ready for the test list she was given 02:16 (Asia station place/remove timing, line delete
   and undo, passage removal, undo loads, fare multiple to 10x, Apply to all weekend, bus save loading). T-4150 s1
   (5facb68f6) landed after it: next rebuild.
3. Gemini batch review due over what landed after batch14: T-4231, T-4253, T-4254, T-4153 s5, T-4214 A, T-4260, T-4220
   s2, T-4150 s1 (template artifacts/trainworld/gemini/batch14/brief.md). Fixes go to one Claude fix agent (as T-4260).
4. Next starts: T-4150 s2 (page: achievements, edgesOf, labels), T-4214 D/E, T-4257 (bus routes by speed), T-4251
   (busways), T-4243 speedometer, T-3742 once settlestop lands, corridors3 (T-4199 s2), T-4080 s3-6, T-4230, T-4187,
   T-3160, T-4233, T-4242 design. Gemini: T-4247, T-4258 (NYC figure diff), T-4259 (UI: needs a screenshot, T-4261).
5. Waiting on her: asks.py list (fare cap text); T-4223's loop refusals not text-acked; T-4250 (smooth fade, waits);
   T-4248 (keep P3, fix the "too far" message now?). PRs #21 and #24: draft thankful closing comments for her OK.

## Her words
- 11-25 00:51: "the game reworking the whole network after each edit is really the target here and everything else is
  kinda just flavor. we should mostly be focusing on isolating the changes and cutting out the impacts on the broader
  map." (DECISIONS T-4097; T-4149, T-4150, T-4262 at P1.) Asia: station adds 1-3 s, removal about 4 s.
- 11-25: PR comments: never close without one; the reason, thankful not apologetic, normal capitalization; her OK
  first. Gemini code agents may run commands unattended in their tree; first 3 rounds asked about permissions (done).
  No terminal windows from anything we start (CLAUDE.md Shell). Airports later, P3, "dont reinvent the wheel". 180th
  meridian: "if people want to build it, they should ideally be able to". T-3742 after settlestop (my call).
- 11-24 22:4x: "you guys get a lot more mileage by thinking about actual code than by trying to test" (CLAUDE.md).
- 11-24 18:5x: shipped 0.4.6; root causes over surface fixes (DECISIONS T-4097). She says when to publish (never ask);
  asianita save is a stress test only; no "less memory" items when memory is up overall; keep cores 0-1 free.
- 11-23: answer her first; agents 2-8 by load; tasks about an hour; you don't investigate, agents do; cycle at 500k;
  no drafted player replies; times as percent of the wait.
- 11-19: "player complaints take precedent over other new features"; full playtest list before every publish.
- Standing: every publish, deploy and website push needs her yes, none overnight; advisor only for big design calls.

## Only a person knows
- Uncommitted, not ours (an 11-23 checks reorganisation): SPEC.md "What is covered", docs/code-tour.md,
  docs/interface.md, checks/old moves, checks/INDEX.md, tools/checks_index.py. Landings touching them: patch dance
  (`git diff --output=<f> -- <files>`, checkout, land, `git apply --3way <f>`, `git reset -- <files>`).
- Uncommitted on purpose: TODO.md (hers), notes/asks-log.jsonl, notes/board-live.json, changelog/roadmap.json,
  maps .claude/settings.json, notes/T-3397.md, tools/ops/archive_bad_saves.py.
- Trees kept: _track (tw/track), tracksnap (snaptrack.py), _seed (Rust cache seed), corridors (T-4199 stage 1),
  settlestop (settlefix working in it).
- reserve_cores: maps-3e's retry loop (task b19u3mylb; powershell PID 29592) follows the ledger's cap and pins Gemini
  runners; it dies with maps-3e's tab: the successor starts its own (a second copy waits on the mutex).
- `land` (not `finish`) conflicts on changelog/next.json when the tree has items: use `finish` (it moves them).
