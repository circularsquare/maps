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

## 8800 d4c7e639 has everything landed tonight but T-3621's json (T-3599 strings: her "text ack" 11-18).
  node_modules broke once ~02:10 UTC (a checker deleting a scratch worktree by hand); npm install restored it.

## In flight (maps-86's; maps-1c fully wound down 03:00 UTC)
- splitflat: T-3609 DONE in tree (two drawing fixes, sim draw.rs; c001 in tree "fixed single track drawn where a
  double track splits sometimes joining the wrong track and making a flat junction"). Checker fcheck (Sonnet, port
  6188) running -> notes/T-3609-check.md; land on PASS. Rest of her case filed T-3618.
- T-3622: both rates right (fares weigh by income, car money flat 3 min/US$), display fix landed 2353f5ee (c014).
  Her: car cost should scale like fares; "lets test out how big of a change that is. dont necessarily want to do it".
  carmoney (Opus, .tw-trees/carmoney, port 6193): T-3627 trial A (equal) and B (US anchored), patch only; plus C
  (her idea: same numbers restated as cheaper car costs in poorer countries, "cuz of mopeds"; check vs real costs).
- nativenames: T-3631 DONE in tree (not a regression; names in the player's script for ja/zh/ko; c016 in tree).
  Resumed to fix an undo-after-placing crash in tools.ts stationGhost (maybe T-3606's): must be fixed before the
  hotfix. On landing run `tasks.py done T-3631` in the main tree. Found: English players get CJK names where data
  has no Latin (file). Asked her: Japanese player in mainland China gets Simplified names (浦东) or Latin?
- T-3630 filed (settings + feedback on the landing page); twitter reply sent by her.
- T-3626 LANDED 4065b244 (c015; strings "Email (optional)", "Email" removed; privacy page line changed).
- secondedit (Opus, .tw-trees/secondedit, port 6191): T-3616 built (edits within 60 s of a local solve go when it
  shows; c007 reworded). Resumed for: t3599-time-left fails only with the change; options for edits >13 s apart
  (still wait 150-170 s). Then a checker, then land (after nightfix's gates).
- T-3619 DONE f2894ea1 (351 starts had no tier; c011); T-3621 DONE 5acbd214 (Cairo's centre was a desert box
  centre; now easy). Filed T-3623 (Cairo's place point) {after: night run}. Tiers rerun once more after the joins.
- Filed T-3620 Caribbean + Bolivia + Paraguay {after: night run}; T-3624 school trips to transit (P3), T-3625 hide
  stations toggle (P3). T-3618 closed on her word (track rework covers it).
- Feedback triage DONE: notes/feedback-triage-2026-11-18.md (78 reports: 4 close, 49 reply+close, 11 reply+keep,
  14 her call) given to her. Feedback inbox (twfeedback on 8800) got a search box (34c0dbaa). Filed T-3629 (P3).
- T-3628 (P1, iPad "Tokyo could not load"): ipadload found WebKit "Load failed" on mask downloads, not memory
  (notes/T-3628.md). loadretry (Sonnet, .tw-trees/loadretry, port 6194) builds: failure reasons in diagnosis(),
  one retry on visibilitychange/online. Two more bugs wait on player answers (30911903, c742b993). Inbox rate-limits.
- Her: intercity minutes-per-money should be consistent too -> sent to carmoney (3.55 vs 3.58, dual display).
- Parked: T-3231 as notes/T-3231-fix.patch {after:T-3271}.

## Night run b RUNNING since 04:01 UTC: kit\night1117b.ps1 (trimmed copy; -Want 9 -NoCif), script pid 39588 (cmd
  31772), log artifacts/trainworld/night1117b.log; stop: world_run\STOP then kill 39588 only. nightfix found the
  first stop was stale zone origins (sve, roc moved with zone_new_origins.py), jau relaxed (pop_vs_fua 1.61, two
  rural tehsils); gates serve a frozen app/ copy, so landings are safe. maps-86 watches the log (Monitor). Disk was
  30 GB free, backups ~20 GB: watch it. After it ends: tasks.py done T-3511/T-3455, notes/T-3455.md "What ran".
- HOTFIX v0.4.1 (her ask 11-18 04:05 UTC, she's up ~2 h): app-only (`deploy.py --skip-data`, data 2026-11-18:
  data/packs is mid night run, must not ship). Candidates: everything on master + T-3631 + T-3628 retry if ready.
  T-3599 strings translated (5275e53e); T-3628 landed 2c196e38. Her OK on the scope 11-18 ~04:20 UTC ("that list
  sounds good"). Waits on nativenames (T-3631). Then: changelog publish v0.4.1, translate updates keys, 8800
  rebuild, smoke8800, her test, her yes per step. T-3515's tram fix is data (realgraph), so not in an app-only hotfix.
- tramterm (Sonnet, .tw-trees/tramterm, port 6197): T-3633 Terminal upgrade for tram stops (her "lets just do this").
  T-3634 filed (panel for several selected pieces, P2), not started: cores full.
- datestrip (Sonnet, no tree, port 6198): what the row of past dates in a Japanese player's top bar is (feedback
  1794744778-ef2051d1) -> notes/T-3635.md (id reserved, task not filed yet). She stopped her feedback pass for
  tonight; another pass tomorrow (notes/feedback-triage-2026-11-18.md).
- T-3632 stage 1 DONE (notes/T-3632.md): only 3 direct reports in the inbox; recommended first fix: the "By
  transit" line names the cause (strings drafted, shown to her). Asked her where she sees it (Discord/X?).
## Night run (her yes 11-18 02:30 UTC; "leave a few cores open", she's up a few more hours)
- T-3511 enclave shifts + T-3455 coastal rows (+ cif-ph if time, -Until 10:00 local), plan notes/T-3455.md, ~8 h.
  Started 02:34 UTC via kit\night1118.cmd (= night1117.ps1 -Want 9), cmd pid 23028, script pid 30872 (claim
  nightqueue, 9 cores; 5 free for day work). Log artifacts/trainworld/night1117.log; state kit\night1117.state;
  stop: create trainworld\data\work\world_run\STOP, then kill only the script PID. Gates serve the main tree's app/:
  keep app/ landings short and between gate steps. 8800 rebuilds during the run: before its joins only (packs are
  hard links to data/packs), else snapbuild-frozen with a freeze; smoke the 8800 URL, not a main-tree Vite (the
  main tree's game json carries the run's in-progress registry). 8800 = d4c7e639 (smoke on 8800 PASSED 12),
  master 5acbd214 pushed. 5acbd214's Cairo tier change is not on 8800 yet (json only; next rebuild). After it: rerun start_difficulty.py (T-3619), then 8800 rebuild;
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
