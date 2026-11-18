# Orchestrator handoff, trainworld

Orchestrator: session c8a9be2a-ffb0-43df-9308-d516bb9169f7, name maps-86, since 2026-11-18 01:23 UTC
Updated: 2026-11-18 01:33 UTC (maps-1c done 01:28; it only relays its three agents' reports)

A short state page, rewritten (not appended) as things change. The diary is notes/handoff-log.md; older detail is in
git history. Read trainworld/CLAUDE.md and docs/orchestrator.md first. Her local time is UTC-5.

## Release v0.4.0 IN PROGRESS (her "can start", 11-18 00:50 UTC)
- Done: changelog published (updates.ts; line "update v0.4! central america and venezuela, timetables, faster
  loading..."); all 12 languages translated and reviewed (i18n.mjs 2822/2822 each); glossary excepts for
  updates.v040Change15/33/58 on her word. In 0.4: one-way stations T-3565/T-3595/T-3605 (pairing dropped on her
  word), fare base T-3578, Use default T-3602, weekend decimal T-3603, help guide T-3598, freight hover T-3464.
- 8800 = 1cd05bdb, smoke8800 PASSED light and dark (artifacts/trainworld/smoke8800-light.log, -dark.log).
- Her yes to all four steps in maps-86's tab, 01:27 UTC. Done: (1) shonei-server deployed (active 01:27:58) and
  pushed 3e8e501..a46c2e9; (2) freight tiles + extra.json on R2 (206/200), T-3464 archived.
- PUBLISHED 2026-11-18 ~01:50 UTC: data 2026-11-18 on R2, website d13096e, `--verify --page` all PASS.
- Landed after the publish: T-3580, T-3516, T-3559/T-3528, T-3515 (+ realgraph copied into data/packs), T-3492
  fix, T-3607, T-3608, T-3615; bookkeeping 2d1567f3. Changelog c001-c006 unpublished. Smoke on master PASSED (29),
  8800 = master = 2d1567f3, pushed.
  (4) export pushed (trainsworld 8ebd2d2 = maps 5bb38d73). PR 1 and issue 8 replied and closed on her OK.
  She: "0.4 looks good on live".
  Later: old R2 versions via r2_purge (2026-11-11, 2026-11-11b can go; keep newest 3).

## In flight (maps-86's)
- T-3610 + T-3612 LANDED a6c617a9 (checker PASS, wart fixed, c009); T-3617 filed. Running: smoke on master + snap8800
  (\smoke_snap.ps1) for kaohsiung + saveread; push after. Main tree's app/node_modules lost @preact/signals and .bin
  ~02:10 UTC (likely scheck deleting a scratch worktree by hand); saveread's npm install restored it.
- T-3600 (a) + T-3601 LANDED c5319e6d after checker PASS (changelog c007, c008); not yet on 8800 (next rebuild,
  with saveread). Leftover filed T-3616 P3. Checker saw "demand sums worker: undefined" once at page close: untraced.

## In flight (maps-1c's agents; maps-1c relays their reports verbatim)
- splitflat: T-3609 DONE in tree (two drawing fixes, sim draw.rs; c001 in tree "fixed single track drawn where a
  double track splits sometimes joining the wrong track and making a flat junction"). Checker fcheck (Sonnet, port
  6188) running -> notes/T-3609-check.md; land on PASS. Rest of her case filed T-3618.
- solveeta (Opus, .tw-trees/solveeta, port 6198): T-3599 time-left hover on the calculating tag. After.
- ghostmatch (Sonnet, .tw-trees/ghostmatch, port 6187): T-3606 tram ghost fixed (c010; archived early), resumed
  for the single-track / passing-track ghost cases. Land when it reports.
- solveeta lands onto c5319e6d (kaohsiung's StatusTag.tsx / demand worker changes): expect a rebase at its landing.
- Parked: T-3231 as notes/T-3231-fix.patch {after:T-3271}.

## Night run (she says when; NOT straight after the publish, her 11-18)
- T-3511 enclave shifts + T-3455 coastal rows, plan notes/T-3455.md, ~8.2 h, 10 cores. Rebuilds packs, so never
  during a publish. Start: `powershell -File C:\Users\anita\projects\maps\trainworld\tools\country\start_hidden.ps1
  C:\Users\anita\projects\maps\trainworld\notes\T-3455\kit\night1117.cmd`. Log artifacts/trainworld/night1117.log;
  PID in kit\night1117.state; stop: create trainworld\data\work\world_run\STOP, then kill only that PID.

## Next
1. Her release yeses, then the steps above; smoke8800 must have passed first.
2. After the publish: land the done trees (one smoke on master per batch), rebuild 8800, push.
3. Leftover .tw-trees: removed the empty and landed ones (11-18). Left: touchrule (empty folder, locked by some
   process), shenzhen (dirty pipeline/osm_units.py + a test, 11-16: whose?), tt2 (50 dirty files from 11-12, stale).
4. Translation leftovers (scratchpad of a883eda3, l10n2-rev-*/notes.md): es/pt "interurbano" needs an "interurban"
   stem in glossary.json for natural wording; T-3597 (Calculating trips everywhere) after 0.4.
- Uncommitted on purpose: TODO.md (hers), notes/T-3397.md, jordan-save-investigation-2026-11-15.md,
  tools/ops/archive_bad_saves.py, changelog/roadmap.json, notes/T-3455/kit. Agents' new notes/T-34xx.md files
  (T-3462, T-3471, T-3515, T-3516 ...): commit with their landings.

## Waiting on Anita
- Nothing open (asks.json empty).

## Her standing asks this week
- Small isolated fixes, open-task count down, merge duplicates; bug investigations give up if hard.
- No asking about changelog lines; agents reword an unpublished item instead of adding one; no item for a change too
  minor or fiddly to describe. Highlights (1-8) for updates over ~20 items.
- Lean testing: concise time-boxed checkers, no checker for small UI, smoke per batch (trainworld/CLAUDE.md,
  docs/orchestrator.md).
- One phrase per wait in player text: loads / shows up / calculating trips / memory (trainworld/CLAUDE.md).
- No deadlines unless she gives one; don't call things impossible by a time.
- Long jobs in the background; answer her first. No windows on her screen.
- Every publish, deploy and server push needs her yes to that exact action; none overnight.
- Advisor trial: start successors with `next_orchestrator.py trainworld --advisor fable`; log each call in handoff-log.
