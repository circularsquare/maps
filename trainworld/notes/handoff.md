# Orchestrator handoff, trainworld

Orchestrator: session 59c85afe-f6fc-46fa-b1d6-fdaa946e7f95, name maps-4f, since 2026-11-18 12:08 UTC
Updated: 2026-11-18 15:15 UTC

Live state: `venv\Scripts\python.exe trainworld/tools/handoff.py status` (T-3667). This file is plan and her words.
Diary: notes/handoff-log.md. Her local time is UTC-5.

## Next
1. Workflow round (her yes 2026-11-18, all seven of research/workflow-2026-11-18.md, any order). Running: testbuild
   T-3664, guards T-3665, playkeep T-3666. Then T-3668 (report inbox, needs T-3667, landed) and T-3669 (checks
   index). Each agent hands back text for CLAUDE.md, orchestrator.md, skills or .claude/settings.json; apply it in
   the main tree at landing (only they don't edit shared rule files).
2. T-3651 (side platforms): checker sideplatcheck running; land after PASS (WASM rebuild), then 8800 rebuild.
3. T-3670 connectors: maps .claude/settings.json has disableClaudeAiConnectors (worked live), deniedMcpServers
   aws-mcp + stripe (stripe still loaded in this session), skillOverrides off for 24 AWS skills + unity. In a fresh
   session check aws-mcp/stripe are gone and measure an agent's first call (was 56k); if not, she toggles them off
   with /mcp in a maps session. Then archive T-3670.
4. T-3395: close after a day of clean daily scans (deployed 11-18 13:30 UTC; she reads the scan's emails).
5. dayrun (Opus, main tree, <= 6 cores): T-3671 daytime rebuild batch (her 11-18 idea; Caribbean excluded). Hold
   8800 rebuilds that touch packs during its join; its notes' top line says when gates run.
6. After the workflow round: T-3660 (downtown Cairo, next night run), T-2136, T-1080, T-930; T-3620 (Caribbean,
   Bolivia, Paraguay) only when she says.

## Her words
- "lets do them all, can start in whatever order you think makes sense" (workflow, 11-18).
- 11-18 ~15:20 UTC: mostly hands off today (headache): ask her nothing unless it truly needs her; short messages.
- Daytime builds on part of the cores when compute is free, thinking-heavy tasks beside them (11-18).
- Caribbean+: "we can worruy about caribbean+ later ... i will lyk" (11-18).
- Free-fare leaderboard cheese (T-3657): "needs more thought, lets put it off til later".
- caelum418's speed PRs (T-3659): "look at the next time we look at this area of performance".
- She sends dattera's reply herself (T-3647). Server log: left to the daily scan's emails.
- Standing this week: small isolated fixes, count down, merge duplicates; she edits each update's changelog on the
  board before publish (highlights over ~20 items); lean testing, smoke per batch; one phrase per wait (loads /
  shows up / calculating trips / memory); long jobs in the background, answer her first, no windows on her screen;
  every publish, deploy and push needs her yes to that action, none overnight; advisor trial (successors with
  `next_orchestrator.py trainworld --advisor fable`, log calls in handoff-log).

## Only a person knows
- Uncommitted on purpose: TODO.md (hers), notes/T-3397.md, jordan-save-investigation-2026-11-15.md,
  tools/ops/archive_bad_saves.py, changelog/roadmap.json; maps .claude/settings.json (her edits + my T-3670 keys).
- Old trees shenzhen (dirty pipeline/osm_units.py, 11-16, owner unknown) and tt2 (stale): ask before removing.
- Overnight undo points data/work/pre-T3455/{pre,prejoin} and pre-T3455my/prejoin: delete once she has played the
  new zones on 8800.
- Unfiled follow-ups from T-3630: last-load record for reports; feedback on the no-WebGL2 and graphics-lost screens;
  t2832's stale phone-note lines; trackInspector's 3 untranslated keys. From T-3647: a string for "track runs the
  other way"; "Run trains one way on this track?" reads backwards when turning it off.
- Changelog has 24 unpublished items; a publish needs her edit on the board first.
