# Orchestrator handoff, trainworld

Orchestrator: session f171f68d-958c-4fdd-a988-a53db2329545, name maps-ca, since 2026-11-18 04:48 UTC
Previous: session c8a9be2a-ffb0-43df-9308-d516bb9169f7, name maps-86, handed over 2026-11-18 04:48 UTC, finishing in-flight work
Updated: 2026-11-18 04:48 UTC

A short state page, rewritten (not appended) as things change. The diary is notes/handoff-log.md; older detail is in
git history. Read trainworld/CLAUDE.md and docs/orchestrator.md first. Her local time is UTC-5; she said ~23:20
local she'd be up "another two hours or so".

## State
- v0.4.0 LIVE since 01:50 UTC (data 2026-11-18, server a46c2e9, freight tiles, export + PR 1/issue 8 closed). She:
  "0.4 looks good on live". Old R2 versions 2026-11-11, 2026-11-11b can go via r2_purge (keep newest 3).
- 8800 = 04763f61 (maps-ca, 04:55 UTC; hotfix build; smoke on 8800 running, log artifacts/trainworld/smoke8800-hotfix.log).
  master 04763f61 (T-3631 decision line), not pushed yet.
- Unpublished changelog c001-c018 (`changelog.py list`); c007 reword pending in the secondedit tree; c018 is
  bushover's (T-3636, in its tree, not landed).

## HOTFIX v0.4.1 (her ask 04:05 UTC, scope OK'd ~04:20 "that list sounds good")
- App-only: `deploy.py --skip-data --version 2026-11-18` (data/packs is mid night run and must not ship).
- In: everything on master since v0.4.0 (T-3610 saves list, T-3600/3601, T-3599 time left (translated 5275e53e),
  T-3609, T-3606, T-3607, T-3608, T-3615, T-3580, T-3516, T-3559, T-3619/3621 tiers, T-3622 hover, T-3626 no email,
  T-3628 download retry 2c196e38, T-3635 clock translate off df8d54f1) + T-3631 when landed. Not in: T-3515 (its fix
  is realgraph data). T-3616 only if its checker passes in time.
- T-3631 LANDED e55932be with the undo crash fix (T-3606's `!` lookup in stationGhost); archived; c017. Everything
  for the hotfix is on master now.
- maps-ca 05:00 UTC: 8800 = 04763f61, smoke on 8800 PASSED (12 checks). `changelog.py publish --version v0.4.1` done
  (16 items, line "update v0.4.1: station names in japanese, chinese and korean, plus fixes", uncommitted:
  updates.ts, en.json, context.json, changelog/v0.4.1.json, next.json). c004 (T-3515, data) held out and re-added as
  next.json c001; c007 reverted to its pre-T-3616 wording (T-3616's landing adds its own item). Agent l10n041b
  translating the 17 keys in the main tree. Then: commit, snap8800, smoke8800, her yes per step.
- Older plan, for reference: smoke on the 8800 URL (not a main-tree Vite: its game json carries the night run's
  registry), she tests station names on 8800, `changelog.py publish --version v0.4.1` (highlights only if >~20
  items), translate the new updates.v041* keys (l10n workflow, like l10n041 did), commit, snap8800 again, smoke8800,
  then her yes per step: deploy.py --skip-data, website push, --verify --page, export.py + push. Nothing overnight.

## In flight
- DECISIONS.md: secondedit's T-3616 line uncommitted (commit with its landing).
- secondedit (maps-86's, Opus, .tw-trees/secondedit, port 6191): T-3616 built, archived early. Resumed for: t3599-time-left
  fails only with the change; scoping edits >13 s apart. Then a checker (demand scheduling), then I land it.
- bushover (Sonnet, .tw-trees/bushover, port 6194): T-3636 fixed app-side (hover subtracts the 4-min bus boarding
  cost, BUS_BOARD_FELT_MIN in tripText.ts mirrors sim), archived, c018. Resumed for its screenshot; then land.
- multitrack (Opus, .tw-trees/multitrack, port 6196): T-3634 shift+click track selection + one Rebuild.
- noroute2 (Opus, .tw-trees/noroute2, port 6195): T-3632 stage 2, the "By transit" line says which reason (no train
  that period, no road or flight, stand-in answer). Sim + worker + panel + 12 languages; checker before landing. Avoid
  MarketTrip.tsx conflicts with the T-3627 display patch.
- T-3627 done (carmoney tree removed; notes, trial patch, display patch, ceiling harness copied to notes/). Waits on her
  (asks.json). notes/T-3627-display.patch (MarketTrip.tsx fare weight 0.40 -> income-group mean, display only, no
  changelog) lands after the hotfix ships, so 8800 stays what she tested.
- Night run b: kit\night1117b.ps1 (-Want 9 -NoCif), script pid 39588 (cmd 31772), log
  artifacts/trainworld/night1117b.log (step logs night1117b-*.log). On the rerun of sve,roc,jau since 23:01 local.
  Stop: create trainworld\data\work\world_run\STOP, then kill 39588 only; on failure it writes the restore command.
  nightfix's notes: notes/T-3455.md "What ran (nightfix)". Disk ~32 GB free, backups ~20 GB: watch it. Nobody
  watches it but the orchestrator's Monitor: re-arm one on the new session. After: tasks.py done T-3511/T-3455,
  start_difficulty.py --fresh once more, 8800 rebuild, then T-3620 and T-3623 can start.

## Waiting on Anita
- notes/asks.json: T-3631 Japanese names in China; T-3627 car cost in poorer cities.
- She does a feedback pass tomorrow with artifacts/trainworld/review/feedback-triage-2026-11-18.md; the inbox has a
  search box now. T-3635 has a reply draft for the player in her last message.

## Next (when cores free; the ledger was full most of the night)
1. The hotfix above.
2. T-3634 panel for several selected pieces of track (P2, her pad).
3. (English players' CJK station names folded into T-2586.)
4. Leftover .tw-trees: touchrule (empty, locked), shenzhen (dirty pipeline/osm_units.py, 11-16: whose?), tt2 (stale).
5. Translation leftover: es/pt "interurbano" glossary stem; T-3597 after 0.4.
- Uncommitted on purpose: TODO.md (hers), notes/T-3397.md, jordan-save-investigation-2026-11-15.md,
  tools/ops/archive_bad_saves.py, changelog/roadmap.json, notes/T-3455/kit (but night1118.cmd is committed), main
  tree's game json (the night run's registry edits).

## Her standing asks this week
- Small isolated fixes, open-task count down, merge duplicates; bug investigations give up if hard.
- No asking about changelog lines; agents reword an unpublished item instead of adding one. Highlights (1-8) for
  updates over ~20 items.
- Lean testing: concise time-boxed checkers, no checker for small UI, smoke per batch.
- One phrase per wait in player text: loads / shows up / calculating trips / memory.
- Long jobs in the background; answer her first. No windows on her screen.
- Every publish, deploy and server push needs her yes to that exact action; none overnight.
- Advisor trial: start successors with `next_orchestrator.py trainworld --advisor fable`; log each call in handoff-log.
