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
- Landing (background, scratchpad of c8a9be2a \land1.ps1, log \land1.log): trafficwidth, snapbar, capload,
  tramsnap (+ realgraph-t3515 copied into data/packs/realgraph), idlemem, chevrons, ctrlsnap. Then one smoke on
  master, snap8800, push.
  (4) export pushed (trainsworld 8ebd2d2 = maps 5bb38d73). PR 1 / issue 8 replies: texts in handoff-log line 19-20
  were written pre-release ("goes out with the next update"); updated wording shown to her, post + close on her OK.
  Later: old R2 versions via r2_purge (2026-11-11, 2026-11-11b can go; keep newest 3).

## In flight (maps-86's)
- T-3397 DONE (saverisk): no other documented bug writes a bad save and reports success; filed T-3610 (P2), T-3611,
  T-3612 (P3). Findings in notes/T-3397.md (commit with the next round).
- saveread: T-3610 + T-3612 DONE in .tw-trees/saveread (archived). Changelog c001 in tree "fixed one unreadable
  saved game hiding all the others; it now shows as damaged and the rest still open". Wart: the damaged card says
  "No undamaged copy of this game is left" while a backup is offered. Checker scheck (Sonnet, port 6184) running ->
  notes/T-3610-check.md; land on PASS.
- sidebox: T-3615 DONE in .tw-trees/sidebox (sheet review/t3615.png, looked at, fine); changelog c003 already in the
  MAIN tree's next.json. Touches rendererShaders.ts like trafficwidth: land after the batch.
- kcheck (Sonnet, checker, runs in .tw-trees/kaohsiung, port 6183): T-3600 option (a) + T-3601; writes
  notes/T-3600-check.md. kaohsiung's report: intercity within 0.4-2.5% of final in 2-6 s; changelog c136 (T-3601),
  c137 "after you turn a line's trains on or off, the intercity lines through that city show their new riders within
  seconds"; open: edits during a quick solve wait ~170 s (the T-1734 hold, option c, not built).
- T-3614 DONE (zonesmap): tools/release/new_zones_map.py, publish step in docs/orchestrator.md, image
  review/v0.4.0-new-zones.png (91 new, 6 grown; holes are T-3517's parked zones). Commit the tool + notes with the
  next round.
- sidebox (Opus, .tw-trees/sidebox, port 6181): T-3615 single-side station boxes with world width (her numbers).
  May touch oneWayMarks.ts like chevrons: merge at landing.

## In flight (maps-1c's agents; maps-1c relays their reports verbatim)
- chevrons (Sonnet, .tw-trees/chevrons): T-3607 DONE in tree: track arrows every 300 m (her word 11-18), bus
  chevrons from zoom 12, BUS_ARROW_GAP_PX 50. Sheet review/t3607.png looked at, fine. Changelog c001 in tree.
- ctrlsnap (Sonnet, .tw-trees/ctrlsnap): T-3608 DONE in tree (changelog c001 "holding ctrl while snapping to
  streets pauses the snap"; no strings).
- splitflat (Opus, .tw-trees/splitflat, port 6205): T-3609 split into singles marked a flat junction. After.
- solveeta (Opus, .tw-trees/solveeta, port 6198): T-3599 time-left hover on the calculating tag. After.
- kaohsiung (Opus, .tw-trees/kaohsiung, port 6199): T-3601 fix done (tag stays until numbers final, StatusTag.tsx);
  now building T-3600 option (a), her pick: the quick solve also covers market pairs through the edited city.
  Merge with solveeta's hover at landing. After the publish.
- DONE in trees, land after the publish with `agent-tree.mjs finish <tree> "<sentence>" --task ...`:
  trafficwidth (T-3580), snapbar (T-3516), capload (T-3559+T-3528, checker PASS; revert the tree's next.json: no
  changelog item, her word), tramsnap (T-3515; then copy trainworld\data\work\realgraph-t3515 over
  data/packs/realgraph/, ships with the next publish's packs; notes/T-3515.md "Landing"), idlemem (T-3492 archived
  unmet; small demandClient.ts fix: `land idlemem` + `remove`).
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
