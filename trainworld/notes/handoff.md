# Orchestrator handoff, trainworld

Orchestrator: session a883eda3-3bdb-42c8-904a-1a72316f71e3, name maps-1c, since 2026-11-17 21:18 UTC
Previous: session f0b2a436-f42b-45c6-999f-26d970f36c1a, name maps-45, handed over 2026-11-17 21:18 UTC, finishing in-flight work
Updated: 2026-11-17 21:28 UTC (times written before maps-1c ran ~1 h ahead of the clock)

A short state page, rewritten (not appended) as things change. The diary is notes/handoff-log.md; older detail is in
git history. Read trainworld/CLAUDE.md and docs/orchestrator.md first. Her local time is UTC-5.

## State
- Live = v0.3.5. She wants v0.4.0 out TONIGHT if it works ("would rather do tonight, but dont worry about it").
  T-3565 (one-way pairs) is NOT in: she said 11-17 ~22:10 UTC "write down ideas and start fresh" (below). My
  proposed cut-off: ship without T-3565 if not solid by ~8 pm her time (asked, unanswered).
- 8800 = fa0eeaea. master is ahead: pass-1 translations, arrows T-3596 (372ea8f7), workflow files 71d8ffae; pushed
  fa0eeaea. Rebuild 8800 after the next landing batch.
- v0.4.0 highlights marked (c084 regions, c051 long-distance fix, c130 schedule panel). Changelog sorted, grouped,
  tidied (notes/changelog-sort-2026-11-17.md, changelog-tidy-2026-11-17.md); she edits on the board.
- Release steps (each needs her yes to that exact action, none overnight): finish translations (below); `changelog.py
  line` + `publish --version v0.4.0`; then translate the new updates.v040* keys (second l10n batch, 3 passes);
  server save check: maps-1c fixed LWE1 flag 2 (T-3557 headways) in shonei-server a46c2e9, tests pass; T-3588's
  header fields aren't read by the check; server deploy (7a23cbc..a46c2e9, her yes); smoke8800 light and dark; deploy.py;
  website push; open-source sync + push; Renkon PR 1 / issue 8 replies (handoff-log 11-17 maps-b8). T-3464 freight
  tiles is {after:v0.4.0 publish} in TASKS but an old handoff listed it pre-ship: ask her if it matters.
- Uncommitted on purpose: TODO.md (hers), notes/T-3397.md, jordan-save-investigation-2026-11-15.md,
  tools/ops/archive_bad_saves.py, changelog/roadmap.json, notes/T-3455/kit (commit with the night run). Outside
  trainworld, uncommitted, hers to decide (asks.json): .claude/skills/handoff, takeover, hooks/orchestrator.py, maps
  CLAUDE.md 500k line. serve.py board commits: allow rule `PowerShell(git commit * -- serve.py)` added on her word.
- Board inbox watch: Monitor on `venv/Scripts/python.exe -u trainworld/board/watch.py` (30 min, re-arm).
- Landing: `agent-tree.mjs finish` (handles already-archived tasks now; serve.py changes must be carried by hand:
  diff to a patch with cmd /c, checkout in tree, finish, git apply, commit, restart_serve.py).

## In flight
maps-45's (it relays their reports to maps-1c verbatim):
- extendfix (Opus, .tw-trees/extendfix, port 6179): T-3595 P1, her Caracas San Martin save
  (artifacts/trainworld/t3595-caracas-sanmartin-broken.save). Diagnosing on master; no source edits yet.
- farebase (Opus, .tw-trees/farebase): T-3578 checker FAILED: all lines at 0x gives 33 lines negative revenue (base
  pool shared by boardings, negative base term spread by km); notes/T-3578-check.md. Fixing and rebasing. Landing is
  maps-1c's: commit the reviewers' messages/ merges first, then a fresh checker, then land; reword T-3567's fare
  item to the end state, hide farebase's.
maps-1c's:
- onewaystops2 (Opus, .tw-trees/onewaystops2 off master, port 6181): T-3565 fresh start; old tree .tw-trees/onewaystops
  is a read-only quarry (remove after). Partner rule kept pending her answer (asks.json). Keeps off T-3595's cause.
  No landing tonight; checker after.
- After-0.4 work, her ask 11-17 ("easily verifiable tasks, for after 0.4"); land after the publish: idlemem (Opus,
  T-3492, port 6182), trafficwidth (Sonnet, T-3580, 6183), capload (Sonnet, T-3559+T-3528, 6184), oldchecks
  (Sonnet, T-3462, 6185), each in .tw-trees/<name>.
- She wants a high-level update on the one-way station work every so often.
- Translations: pass 3 done and committed for all 12 (9b393ad7, c42aa7a9, cd10e0d3, ea5c651c). Running: l10n-shots
  (Sonnet, port 6186, pass 5 screenshots, may merge shorter wording; I commit). Later notes in each
  scratchpad\l10n-rev-<x>\notes.md (es tu/infinitive mix, fr marketTrip tronçons, ko/zh one-way trip word order).
  After farebase lands: `i18n.mjs --stale`, fold any stale keys into the updates.v040* batch.
- Landed by maps-45: arrows (T-3596) 372ea8f7, not on 8800 yet.

## Night run (she'd prefer the publish first; start after it, or at ~02:00 UTC if no publish)
- T-3511 enclave shifts + T-3455 coastal rows, plan notes/T-3455.md, ~8.2 h. Start: `powershell -File
  C:\Users\anita\projects\maps\trainworld\tools\country\start_hidden.ps1
  C:\Users\anita\projects\maps\trainworld\notes\T-3455\kit\night1117.cmd`. Log artifacts/trainworld/night1117.log;
  PID in kit\night1117.state; stop: create trainworld\data\work\world_run\STOP, then kill only that PID. 10 cores.
  Hold app/ landings during its gates.

## Next
1. Relay/land reports above; smoke on master per batch; rebuild 8800 (`snap8800.py <hash>` hidden, background); push.
2. Commit notes/T-3565.md, TASKS/board-live/asks edits and reviewers' merges in the next batch (after maps-45 is done).
3. Finish translations; then the release steps with her yeses.
4. Leftover .tw-trees: onewaystops (after the fresh agent takes what it needs), plus old c1, ferry45, fixes, freight,
   hotfix031-target, hover, touchrule, trackrebuild, safari-save-035, shenzhen, tt2, ttstyle, trams: look first.

## Waiting on Anita
- asks.json: T-3565 partner rule (drop recommended). Workflow files: she said yes, committed 71d8ffae.
- Asked again in chat by maps-1c: ship 0.4 without T-3565 (my pick; it won't be checked tonight); T-3464 before?

## Her standing asks this week
- Small isolated fixes, open-task count down, merge duplicates; bug investigations give up if hard.
- No asking about changelog lines; agents reword an unpublished item instead of adding one; orchestrator merges or
  hides duplicates itself. Highlights (1-8) for updates over ~20 items.
- Redraws with no numbers: three versions on one sheet first.
- Long jobs in the background; answer her first. No windows on her screen.
- Every publish, deploy and server push needs her yes to that exact action; none overnight.
- Advisor trial: start successors with `next_orchestrator.py trainworld --advisor fable`; log each call in handoff-log.
