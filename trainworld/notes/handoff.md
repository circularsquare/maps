# Orchestrator handoff, trainworld

Orchestrator: session a883eda3-3bdb-42c8-904a-1a72316f71e3, name maps-1c, since 2026-11-17 21:18 UTC
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
maps-45 wound down 11-17 ~22:20 UTC. All in flight is maps-1c's:
- farebase (T-3578, .tw-trees/farebase, based on 71d8ffae): fixed after the first checker's fail (base extra on its
  own channel, a 0x line takes no base). Checker farecheck2 (Sonnet, ports 6192/6193) running. On PASS land for
  v0.4.0 (it's in the release); no changelog change (her c120 still holds); fareTip hover changed (approved by her),
  so fareTip joins the stale-key translation batch.
- onewaystops2 (Opus, .tw-trees/onewaystops2, port 6181): T-3565 fresh start + T-3595 (a train may turn back at any
  track end; DECISIONS). Pairing per her 11-17: nice to have, ~50 m between platforms, any number of one-way stops per
  leg. Checker straight after it reports; her ask 11-17: the checker also looks for regressions against live v0.3.5
  (bfe58c54): anything a player could do there that they can't after (stop editing, advanced editor order, loops,
  two-way lines, pairs 50-300 m apart, old saves loading the same). Old trees onewaystops and extendfix: remove once it's done.
- l10n-shots (Sonnet, port 6186): pass 5 screenshots, may merge shorter wording; I commit messages/ after.
- For after the publish (her 11-17 ask): idlemem (Opus, T-3492, 6182), capload (Sonnet, T-3559+T-3528, 6184; done,
  smoke passed; checker capcheck PASSED (notes/T-3559-check.md); land after the publish WITHOUT its changelog item, her word: revert the tree's next.json),
  snapbar (T-3516 DONE in its tree, smoke passed; c136 line shown to her), trafficwidth (T-3580 DONE, smoke passed),
  tramsnap (T-3515 DONE: pipeline/realgraph.py tram rules; land after the publish: branch, then copy the rebuilt
  graphs from trainworld\data\work\realgraph-t3515 (durable copy of scratchpad\tramsnap\graph-all) over data/packs/realgraph/ (they ship to R2 with the next publish's
  packs); notes/T-3515.md "Landing"), termdraw (Opus, T-3231, 6191). Land the done ones after the publish with finish.
  icskip (T-3471) landed cf3a678d, oldchecks (T-3462) bde8c1d8.
- She wants a high-level update on the one-way station work now and then. Away a few hours from 22:00 UTC; said
  start new lines if little is running.

- solveeta (Opus, .tw-trees/solveeta, port 6198): T-3599 time-left hover on the calculating tag, after the publish.
- kaohsiung (Opus, .tw-trees/kaohsiung, port 6199): T-3600 (5 min after a Kaohsiung edit) + T-3601 (lines with no
  trains keep riders after the recalc) on her saves artifacts/trainworld/t3600-asianita-1851/-1852.save.

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
