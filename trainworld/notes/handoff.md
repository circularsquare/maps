# Orchestrator handoff, trainworld

Orchestrator: session a883eda3-3bdb-42c8-904a-1a72316f71e3, name maps-1c, since 2026-11-17 21:18 UTC
Previous: session f0b2a436-f42b-45c6-999f-26d970f36c1a, name maps-45, handed over 2026-11-17 21:18 UTC, finishing in-flight work
Updated: 2026-11-17 22:00 UTC

A short state page, rewritten (not appended) as things change. The diary is notes/handoff-log.md; older detail is in
git history. Read trainworld/CLAUDE.md and docs/orchestrator.md first. Her local time is UTC-5.

## State
- Live = v0.3.5. v0.4.0: SHE says when to start the publish; no fixed time, no cut-off (11-17). Until then get the
  one-way station work (T-3565 + T-3595) ready to go in if it comes together. Freight tiles (T-3464) are in: code
  on master b8d0003b; the R2 upload is a release step (commands in notes/T-3464.md, ~2 min at 2 MB/s).
- 8800 = fa0eeaea; master ahead (translations, arrows T-3596, freight hover, oldchecks). Rebuild 8800 after the next
  landing batch. Last push fa0eeaea.
- v0.4.0 highlights marked (c084 regions, c051 long-distance fix, c130 schedule panel); she edits on the board.
- Release steps (each needs her yes to that exact action, none overnight): translate stale keys (freight hover
  settingsTab.realAllTip, farebase's fareTip if it lands) with the updates.v040* keys after `changelog.py line` +
  `publish --version v0.4.0`, 3 passes; server deploy (shonei-server 7a23cbc..a46c2e9; a46c2e9 = maps-1c's LWE1
  flag-2 fix, tests pass); freight R2 upload; smoke8800 light and dark; deploy.py; website push; open-source sync +
  push; Renkon PR 1 / issue 8 replies (handoff-log 11-17 maps-b8).
- Uncommitted on purpose: TODO.md (hers), notes/T-3397.md, jordan-save-investigation-2026-11-15.md,
  tools/ops/archive_bad_saves.py, changelog/roadmap.json, notes/T-3455/kit (commit with the night run).
- Board inbox watch: Monitor on `venv/Scripts/python.exe -u trainworld/board/watch.py` (30 min, re-arm).
- Landing: `agent-tree.mjs finish`; when a task must stay open, move the changelog item by hand, `land`, `remove`.

## In flight
maps-45's (it relays reports to maps-1c verbatim): farebase (Opus, .tw-trees/farebase), T-3578: checker FAILED (all
lines at 0x gives 33 lines negative revenue; notes/T-3578-check.md), fixing. Landing is maps-1c's: fresh checker,
land, reword T-3567's fare item to the end state, hide farebase's.
maps-1c's:
- onewaystops2 (Opus, .tw-trees/onewaystops2, port 6181): T-3565 fresh start + T-3595 (a train may turn back at any
  track end; DECISIONS). Pairing per her 11-17: nice to have, ~50 m between platforms, any number of one-way stops per
  leg. Checker straight after it reports. Old trees onewaystops and extendfix: remove once it's done.
- l10n-shots (Sonnet, port 6186): pass 5 screenshots, may merge shorter wording; I commit messages/ after.
- For after the publish (her 11-17 ask): idlemem (Opus, T-3492, 6182), capload (Sonnet, T-3559+T-3528, 6184; done,
  running smoke + demand lib, then a checker; land WITHOUT its changelog item, her word: revert the tree's next.json),
  snapbar (Sonnet, T-3516, 6188), icskip (Sonnet, T-3471, no Vite). trafficwidth (T-3580) DONE in its tree, smoke
  passed: land after the publish with `agent-tree.mjs finish trafficwidth ... --task T-3580`.
- She wants a high-level update on the one-way station work now and then. Away a few hours from 22:00 UTC; said
  start new lines if little is running.

## Night run
- T-3511 enclave shifts + T-3455 coastal rows, plan notes/T-3455.md, ~8.2 h, 10 cores. It rebuilds packs, so never
  during a publish: start after the publish, or once she says she's done for the night without one. Start:
  `powershell -File C:\Users\anita\projects\maps\trainworld\tools\country\start_hidden.ps1
  C:\Users\anita\projects\maps\trainworld\notes\T-3455\kit\night1117.cmd`. Log artifacts/trainworld/night1117.log;
  PID in kit\night1117.state; stop: create trainworld\data\work\world_run\STOP, then kill only that PID. Hold app/
  landings during its gates.

## Next
1. Relay/land reports; smoke on master per batch; rebuild 8800 (`snap8800.py <hash>` hidden, background); push.
2. Leftover .tw-trees: onewaystops, extendfix (after onewaystops2), plus old c1, ferry45, fixes, freight,
   hotfix031-target, hover, touchrule, trackrebuild, safari-save-035, shenzhen, tt2, ttstyle, trams: look first.
3. Later translation notes in scratchpad\l10n-rev-<x>\notes.md (es tu/infinitive mix, fr marketTrip tronçons, ko/zh
   one-way trip word order). Pass 5 (scratchpad\l10n-shots\notes.md): ru shows "km", "km/h" in Latin beside
   translated "мин" (unit formatter? check other languages; file a task if real).

## Waiting on Anita
- Nothing open.

## Her standing asks this week
- Small isolated fixes, open-task count down, merge duplicates; bug investigations give up if hard.
- No asking about changelog lines; agents reword an unpublished item instead of adding one; orchestrator merges or
  hides duplicates itself. Highlights (1-8) for updates over ~20 items.
- Redraws with no numbers: three versions on one sheet first.
- Long jobs in the background; answer her first. No windows on her screen.
- Every publish, deploy and server push needs her yes to that exact action; none overnight.
- No deadlines unless she gives one; don't call things impossible by a time (trainworld/CLAUDE.md).
- Advisor trial: start successors with `next_orchestrator.py trainworld --advisor fable`; log each call in handoff-log.
