# Orchestrator handoff, trainworld

Orchestrator: session f171f68d-958c-4fdd-a988-a53db2329545, name maps-ca, since 2026-11-18 04:48 UTC
Updated: 2026-11-18 05:40 UTC

A short state page, rewritten (not appended) as things change. The diary is notes/handoff-log.md; older detail is in
git history. Read trainworld/CLAUDE.md and docs/orchestrator.md first. Her local time is UTC-5; ~00:40 local she
said she's heading to bed: overnight, knock off Up next tasks with spare cores.

## State
- v0.4.1 LIVE since ~05:35 UTC (app only on data 2026-11-18; website 8fbf33e; verify --page PASS; export 405ecf5
  pushed). 13 items, her trim. Old R2 versions 2026-11-11, 2026-11-11b can go via r2_purge (keep newest 3).
- deploy.py --skip-data now checks the registry against R2 at that version (local packs mid-run tripped it).
- master 3ce746f9 pushed. 8800 rebuilding at 3ce746f9 + smoke on the 8800 URL (scratchpad batch1.ps1, task
  b6jazk9pt; log artifacts/trainworld/smoke-batch1.log). Landed since v0.4.1: T-3636 (bus hover), T-3634
  (shift+click track set), T-3627 (hovers: one money-to-minutes figure per place, demand unchanged).
- Unpublished changelog next.json: c001 T-3515 (data), c002 T-3634, c003 T-3636, c004 T-3627. She edits the
  changelog on the board before every publish now (docs/orchestrator.md).
- Board Up next (her ask: misc tasks for the next minor update): T-3597, T-3637, T-3638, T-3589, T-3604, T-3482,
  T-3433, T-3474, T-3630, then T-2136, T-1080, T-930. T-3597 waits for noroute2 to land (both add i18n keys).

## In flight
- DECISIONS.md uncommitted: T-3616 (secondedit), T-3627 (mine + carhover's, check for a duplicate), T-3634
  (multitrack). Commit with secondedit's landing.
- secondedit (.tw-trees/secondedit, port 6191): T-3616 built, archived early. Checker check3616 (Sonnet) running;
  then land (its tree's c002 item moves to main). Its idea for edits during the whole solve (gap = 1/7 of the solve,
  latest start 60 s; ~2-3 h): file it after landing.
- T-3632 stage 2 LANDED 7d14a945 (checker PASS; c006; WASM rebuilt). New strings for her: marketTrip.noTrain "No
  train runs between the two ends this period.", marketTrip.noAlt "No road or flight joins these two places, so for
  now none of their trips go by transit." T-3632 stays open (wider goal).
- calctag (Sonnet, .tw-trees/calctag, port 6198): T-3597 every calculating tag says Calculating trips.
- gfxreport (Sonnet, .tw-trees/gfxreport, port 6197): T-3589 graphics line in feedback reports.
- T-3298 (tramchange) cause found, waits on her (asks.json: walking change limit 300 m -> 600 m?); notes and scene
  script copied to main notes/, tree removed. T-3641 filed from it (panel head vs period rows), queued in Up next.
- T-3629 (trackpick) LANDED c2005ce4, c005; not yet on 8800 (next batch rebuild).
- tramchange (Opus, .tw-trees/tramchange, port 6199): T-3298 tram then train shows as a walk.
- scoreswing (Opus, .tw-trees/scoreswing, port 6190): T-3639 leaderboard score swings 450m-550m (her pad, Discord).
  May commit in shonei-server; deploy is hers.
- bigshare (Sonnet, .tw-trees/bigshare, port 6189): T-3640 Swattik's network can't be shared (after T-3440).
- 8800 = 3ce746f9, batch smoke PASSED (12 checks). She went to bed ~05:42 UTC ("gnight").
- Her islands idea (a fare point past which transit isn't taken) is in T-2216's goal.
- Night run b: kit\night1117b.ps1 (-Want 9 -NoCif), script pid 39588 (cmd 31772), log
  artifacts/trainworld/night1117b.log (step logs night1117b-*.log). On the rerun of sve,roc,jau since 23:01 local
  (yhu, suc etc. missing from data/packs mid-run: expected). 05:47 UTC: in the join (world_run join --countries
  TR,AU,AR,IN,CN, pid 23572 under 33100, since 05:00 UTC); its run log stops at 04:18 UTC, which is normal for the join. Stop: create trainworld\data\work\world_run\STOP, then
  kill 39588 only; on failure it writes the restore command. nightfix's notes: notes/T-3455.md "What ran (nightfix)".
  Disk ~32 GB free, backups ~20 GB: watch it. Monitor re-armed by maps-ca. After: tasks.py done T-3511/T-3455,
  start_difficulty.py --fresh once more, 8800 rebuild, then T-3620 and T-3623 can start.

## Waiting on Anita
- Nothing in asks.json. New strings for her when noroute2 lands (above) and from overnight agents.
- She does a feedback pass with artifacts/trainworld/review/feedback-triage-2026-11-18.md; T-3635 has a reply draft.

## Next
1. Land secondedit and noroute2 once their checkers pass; batch smoke + 8800 rebuild after.
2. Overnight: start Up next tasks as agents finish (T-3597 after noroute2; T-3637 small-fixes; T-3638; T-3589;
   T-3604; T-3482; T-3433; T-3474 is 4-6 h, credit CinnamonVII; T-3630 needs a proposal to her first).
3. Leftover .tw-trees: touchrule (empty, locked), shenzhen (dirty pipeline/osm_units.py, 11-16: whose?), tt2 (stale).
4. Translation leftover: es/pt "interurbano" glossary stem.
- Uncommitted on purpose: TODO.md (hers), notes/T-3397.md, jordan-save-investigation-2026-11-15.md,
  tools/ops/archive_bad_saves.py, changelog/roadmap.json, notes/T-3455/kit (but night1118.cmd is committed), main
  tree's game json (the night run's registry edits), world_plan.json and notes/T-3455.md (night run).

## Her standing asks this week
- Small isolated fixes, open-task count down, merge duplicates; bug investigations give up if hard.
- She edits each update's changelog on the board before publish (unless ~2 items). Highlights (1-8) for updates
  over ~20 items.
- Lean testing: concise time-boxed checkers, no checker for small UI, smoke per batch.
- One phrase per wait in player text: loads / shows up / calculating trips / memory.
- Long jobs in the background; answer her first. No windows on her screen.
- Every publish, deploy and server push needs her yes to that exact action; none overnight.
- Advisor trial: start successors with `next_orchestrator.py trainworld --advisor fable`; log each call in handoff-log.
