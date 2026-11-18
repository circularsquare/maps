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

## Landed since 8800's 3fd86cd2 (next rebuild): T-3606 98ca8aca, T-3619 f2894ea1, T-3599 63bacf31 (strings shown to
  her). 8800 has T-3600/T-3601/T-3610/T-3612. node_modules broke once ~02:10 UTC (a checker deleting a scratch
  worktree by hand); npm install restored it.

## In flight (maps-86's; maps-1c fully wound down 03:00 UTC)
- splitflat: T-3609 DONE in tree (two drawing fixes, sim draw.rs; c001 in tree "fixed single track drawn where a
  double track splits sometimes joining the wrong track and making a flat junction"). Checker fcheck (Sonnet, port
  6188) running -> notes/T-3609-check.md; land on PASS. Rest of her case filed T-3618.
- vot (Opus, .tw-trees/vot, port 6189): T-3622 player report, transit 8.86 vs driving 2.31 min/S$ in Malaysia.
  A demand-number change waits for her.
- secondedit (Opus, .tw-trees/secondedit, port 6191): T-3616 (her go 11-18): later edits wait ~170 s behind the
  first whole solve (T-1734 hold). Told to rebase onto 63bacf31 (T-3599's editLeft doubling while an edit waits).
- T-3619 DONE f2894ea1 (351 starts had no tier; c011). cairo (Opus, main tree, scratchpad \cairo): T-3621 why
  Cairo scores 439 (hard); may change start_difficulty.py + start-difficulty.json.
- Filed T-3620 Caribbean + Bolivia + Paraguay {after: night run}.
- Parked: T-3231 as notes/T-3231-fix.patch {after:T-3271}.

## Night run RUNNING (her yes 11-18 02:30 UTC; "leave a few cores open", she's up a few more hours)
- T-3511 enclave shifts + T-3455 coastal rows (+ cif-ph if time, -Until 10:00 local), plan notes/T-3455.md, ~8 h.
  Started 02:34 UTC via kit\night1118.cmd (= night1117.ps1 -Want 9), cmd pid 23028, script pid 30872 (claim
  nightqueue, 9 cores; 5 free for day work). Log artifacts/trainworld/night1117.log; state kit\night1117.state;
  stop: create trainworld\data\work\world_run\STOP, then kill only the script PID. Gates serve the main tree's app/:
  keep app/ landings short and between gate steps. After it: rerun start_difficulty.py (T-3619), then 8800 rebuild;
  then T-3620 (Caribbean, Bolivia, Paraguay) can start.

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
