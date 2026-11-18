# Orchestrator handoff, trainworld

Orchestrator: session a883eda3-3bdb-42c8-904a-1a72316f71e3, name maps-1c, since 2026-11-17 21:18 UTC
Updated: 2026-11-18 01:25 UTC (maps-1c past 500k, starting its successor)

A short state page, rewritten (not appended) as things change. The diary is notes/handoff-log.md; older detail is in
git history. Read trainworld/CLAUDE.md and docs/orchestrator.md first. Her local time is UTC-5.

## Release v0.4.0 IN PROGRESS (her "can start", 11-18 00:50 UTC)
- Done: changelog published (updates.ts; line "update v0.4! central america and venezuela, timetables, faster
  loading..."); all 12 languages translated and reviewed (i18n.mjs 2822/2822 each); glossary excepts for
  updates.v040Change15/33/58 on her word. In 0.4: one-way stations T-3565/T-3595/T-3605 (pairing dropped on her
  word), fare base T-3578, Use default T-3602, weekend decimal T-3603, help guide T-3598, freight hover T-3464.
- Running (hidden cmd, scratchpad of a883eda3 \release_prep.cmd): snap8800 at 1cd05bdb, then checks\smoke8800.cmd;
  logs ...\scratchpad\snapbuild-rel.log and ...\scratchpad\smoke8800-rel.log (ends "DONE <code>"). If master moved
  past 1cd05bdb only by bookkeeping, that build is the one to ship; 0b7f5a4f is bookkeeping only.
- ASKED HER (awaiting her yes to each, in her latest message, none overnight): (1) server deploy (shonei-server
  `.\deploy\deploy.ps1` alone, then git push; carries 7a23cbc..a46c2e9 incl. maps-1c's LWE1 flag-2 save check fix),
  (2) freight R2 upload (notes/T-3464.md "Release step", rclone --bwlimit 2M; tiles first, extra.json last; then
  `tasks.py done T-3464`), (3) deploy.py per notes/T-924.md "Commands" (--dry-run --version v0.4.0, then --smoked,
  website push, --verify), (4) open-source export.py + push in ..\trainsworld-export, then post PR 1 / issue 8 replies
  (texts in handoff-log 11-17 maps-b8 section; she OK'd them; change PR 1's "start them at zoom 15" to "zoom 13").
- Also asked: one-way track arrows at zoom 14 (300 px gap leaves 1 arrow on the Caracas pair): lower to 150 px?

## In flight (maps-1c's agents; their reports come to maps-1c, which relays them verbatim)
- chevrons (Sonnet, .tw-trees/chevrons): T-3607 DONE in tree (track arrows per 100 m across pieces; bus arrows from
  zoom 12, chevron stroke = bus line, BUS_ARROW_GAP_PX 50). Sheet artifacts/trainworld/review/t3607.png. Changelog
  c001 in its tree. Land after the publish (or into 0.4 if she wants; then the changelog line needs translating).
- ctrlsnap (Sonnet, .tw-trees/ctrlsnap, port 6204): T-3608 Ctrl pauses snap to street. After the publish.
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
3. Leftover .tw-trees from before 11-17: c1, ferry45, fixes, freight, hotfix031-target, hover, touchrule,
   trackrebuild, safari-save-035, shenzhen, tt2, ttstyle, trams: look first.
4. Translation leftovers (scratchpad of a883eda3, l10n2-rev-*/notes.md): es/pt "interurbano" needs an "interurban"
   stem in glossary.json for natural wording; T-3597 (Calculating trips everywhere) after 0.4.
- Uncommitted on purpose: TODO.md (hers), notes/T-3397.md, jordan-save-investigation-2026-11-15.md,
  tools/ops/archive_bad_saves.py, changelog/roadmap.json, notes/T-3455/kit. Agents' new notes/T-34xx.md files
  (T-3462, T-3471, T-3515, T-3516 ...): commit with their landings.

## Waiting on Anita
- The release yeses and the zoom-14 arrow gap (above).

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
