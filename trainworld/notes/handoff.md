# Orchestrator handoff, trainworld

Orchestrator: session 96357f14-1979-4190-bef7-751a0544778a, name maps-6f, since 2026-11-19 14:21 UTC
Updated: 2026-11-19 17:00 UTC (maps-6f past 500k, starting its successor)

Live state: `venv\Scripts\python.exe trainworld/tools/handoff.py status` (T-3667). This file is plan and her words.
Diary: notes/handoff-log.md. Her local time is UTC-5.

## Next
0. Live v0.4.2. Master at level 29, 8800 = 2a5413b4. Unpublished changelog c001-c025 (+ tree items). Cap 14.
   shonei-server: e604291 (levels 27-29), 93718d1 (T-3775 savehealth route), 5bf10c7 (T-3382) committed, not
   deployed or pushed: ask a-2026-11-19-15. The server is at level 26 until then.
1. v0.4.3 scope (her 11-19 16:20): hold for T-3781, T-3773, T-3686 (landed), T-3346 stage 2, T-3782, T-3767 (landed),
   plus the fare zone fixes. Then: server deploy on her yes, data upload (new/reshipped data/packs/real saves:
   ngo, fuk, spk, hir, osa), changelog review agent + highlights (25+ items), she edits the changelog on the board,
   her playtest list (docs/orchestrator.md), smoke8800, publish on her yes.
2. In flight (maps-6f's agents report to the inbox; a follow-up goes to a fresh agent on the tree):
   - zonepolish T-3783 done in tree zonepolish (Enable as an icon left of the bin with "Disable/Enable {name}", her
     heading hover + "Transfers in fare zones are free. Multipliers for mode of transport are applied
     proportionally.", 3 px gap under the Fare zones rule; fares proved per game). Land TOGETHER with zonexfer.
   - zonexfer (Opus, T-3784, tree zonexfer, 6193): no transfer fare between zone rides of one run. Needs a checker
     (demand numbers), then land with zonepolish.
   - failopen T-3773 done in tree failopen (failed opens go to the menu; notice "{name} could not be opened." per
     her; record tw-open-failures). Checker failopenchk (6194) running. After it: update t3395-saves.mjs (expects
     the old two-sentence notice), ui-panels.md and notes/T-3773.md to the new text, then land. Then start T-3774.
   - demandload T-3781 stage 2 in tree demandload (workers restart once on the watchdog; report diagnostics):
     rebasing onto 2a5413b4 now; then a concise checker; land; T-3781 stays open (cause unknown, one player).
   - ttfix (Opus, T-3346 stage 2, tree ttport, 6175/6176): fixing t2233-passing-map at 10/10 (map 50 s vs bar 52),
     an extra pass at a non-passing place, untested reload. Then land; remove tt2 after.
   - realjapan T-3782 (Sonnet, tree realfix2, 6173): Nagoya station missing from the Tokaido Main Line (add stops a
     line's track passes through), Fukuoka's 11 reversals. Reship only changed cities.
   - russave T-3786 (notes/T-3786.md): Rus's Java game was deleted (4 tombstones 11-19); copy likely in R2 backups
     11-18 and 11-19 (drop off ~11-25). Ask a-2026-11-19-17 (her backup key, one account folder). On yes, a fresh
     Opus agent per the plan in notes/T-3786.md. T-3787 filed (card delete check counts only the top version).
   - commentab T-3726: pair 2 scored (no difference; notes/T-3726.md). Pairs 1 and 3, then pair 2 again on fresh
     trees, once cores allow (scoring.md in maps-6b's scratchpad commentab; ab-1-*, ab-3-* trees still there).
3. Queued, not for v0.4.3: T-3774 (feedback carries save state, after T-3773), T-3787, T-3788, T-3779, T-3766
   (fares by city: country or city multiples, never both), T-3780 (station/complex timetables, after T-3346 stage 2
   lands), T-3785 (schematic hides hidden lines, with the next schematic work), T-3762 / T-1738 / T-3763 countries,
   T-3409, T-3360.
4. Steam (T-3729): fee paid, tax form in; her screenshots go in artifacts/trainworld/steam/screenshots.
5. Later: T-3744 per-stop fare, T-3740 Z3, T-3754, T-3747, busverify's train 4.14 min gap read (a short probe).

## Her words
- 11-19 v0.4.3 scope above ("yea no rush"); "player complaints take precedent over other new features", "only
  applies to like actual bugs". Fares by city: country or city multiples, never both.
- 11-19 fares: transfers inside fare zones free; heading hover her wording; Enable as an icon left of the bin
  ("Disable Caracas fare zones"); fare settings must be per game (they are). Failed-open notice "{name} could not
  be opened."; Josh's failure line stays as is.
- 11-19: before every publish, a full playtest list (docs/orchestrator.md). Scratch of finished sessions 2+ days
  old is deleted without asking (memory). Money history: keep the global record forever; per-line a month.
- 11-19 zones: "if a trip from zone 2 to 2 counts as 1 that does feel pretty weird to me"; lines choose zone or
  distance fares (T-3758). Station names: a bind setting, default on (T-3759).
- 11-18: close easy high-priority wins, small over design-heavy; "prio should take into account easiness as well
  as impact". "we need to be less willing to put stuff on the changelog". "manually", never "by hand".
- Share previews: free Worker, "please be careful" with the Cloudflare account. Mod support (T-3377): never say
  publicly it's aimed at vibe-coders. caelum418's speed PRs (T-3659): next time we look at performance.
- When she plays games: keep 4 cores free (cap 12) and no uploads until she says she's done.
- Standing: lean testing, smoke per batch; long jobs in the background, answer her first; every publish, deploy and
  push needs her yes to that action, none overnight; advisor trial (log calls in handoff-log).

## Waiting on her (asks.py list has the open ones)
- a-1 (T-3746 text), a-13 (read today's 2 damaged saves over ssh; or wait for the deploy), a-14 (drop the load
  card once the city in view is ready), a-15 (server deploy), a-17 (Rus's backup). Rus's tweet time asked too.
- She sends the T-3635 player reply herself (draft in notes/T-3635.md).

## Only a person knows
- Uncommitted on purpose: TODO.md (hers), notes/T-3397.md, jordan-save-investigation-2026-11-15.md,
  tools/ops/archive_bad_saves.py, changelog/roadmap.json, changelog/next.json, notes/asks-log.jsonl,
  notes/board-live.json; maps .claude/settings.json.
- Old tree tt2: keep until ttport lands. Empty locked folders .tw-trees/touchrule and .tw-trees/timekeys.
- Never delete a tree folder by hand (its app/node_modules junction deletes the main tree's).
