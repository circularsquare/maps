# Orchestrator handoff, trainworld

Orchestrator: session a0ff2795-d3f6-4076-b0f1-e96b327ba6cf, name maps-e8, since 2026-11-19 17:00 UTC
Updated: 2026-11-19 17:38 UTC

Live state: `venv\Scripts\python.exe trainworld/tools/handoff.py status` (T-3667). This file is plan and her words.
Diary: notes/handoff-log.md. Her local time is UTC-5.

## Next
0. Live v0.4.2. Master at level 29, 8800 = 2a5413b4 (rebuild after this round's landings). Cap 14.
   shonei-server: e604291 (levels 27-29), 93718d1 (T-3775 savehealth route), 5bf10c7 (T-3382) committed, not
   deployed or pushed: ask a-2026-11-19-15. keepdel's T-3789 commits join that deploy.
1. v0.4.3 (her 11-19 16:20 scope; I told her ~2-3 pm her time for playtest list + changelog review): left are
   T-3346 stage 2, T-3788, T-3782. If realjapan T-3782 is the last thing, ship without it (her ~17:35: "people
   can just fix them themselves ... its still mostly perfect"). Then: 8800 rebuild + smoke on master, changelog
   review agent + highlights, she edits the changelog on the board, her playtest list (docs/orchestrator.md),
   server deploy on her yes, data upload (ngo, fuk, spk, hir, osa + any realjapan reships), smoke8800, publish.
2. In flight (maps-6f's ttfix/realjapan live in its tab; follow-ups go to a fresh agent on the tree):
   - T-3346 stage 2, tree ttport: ttfix done (guard lag fix, following-loss cap in timing.rs). Checker ttchk2
     (6158) running, incl. whether master's reload trips/day differ (767.8M vs 836.3M). Then land; remove tt2.
   - T-3788 landed 9f22d830 (openchk: a badRev device never overwrites the newer rev; its edits split off).
   - T-3782 landed 9163ebd4 (pipeline only); 22 cities reshipped into data/packs/real, but rebuilds drop lines
     (Moscow 106 -> 90). linedrift (Opus, T-3790, 6161): count lines vs v0.4.2 for all 22, put v0.4.2 saves back
     where lines dropped, then the cause. The data upload waits for its step 2.
   - keepdel (Opus, T-3789): deleted games kept 7 days (her yes ~17:30), create-only admin restore route,
     tools/ops/restore_game.py, privacy line draft; then Rus's day 423 game (artifacts/trainworld/rus-recovery,
     from R2: her "do the R2") restored under a new id after the deploy; delete rus-recovery once restored.
   - T-3784 landed 7db542f2 (+ T-3783 2c541578). Fare hover now "... and {xferFare} in transfer fares"
     (48296702, her yes 11-19); its 12 translations are stale: the v0.4.3 translation batch covers them.
   - Her ~17:45: core cap lowered 14 -> 10 "for a few hrs"; raise back to 14 when she says (or ask ~21:00 UTC).
   - commentab T-3726: pairs 1 and 3, then pair 2 again on fresh trees, once cores allow (notes/T-3726.md).
3. Landed this session (on master, not 8800): T-3787, T-3779, T-3773 (c027), T-3774, T-3781 stage 2 (c028),
   T-3783. T-3781 stays open for the next report's diagnosis; T-3786 closes once Rus has the game.
4. Queued: T-3766 (fares by city), T-3780 (after T-3346 stage 2), T-3785, T-3762 / T-1738 / T-3763 countries
   (after the v0.4.3 data upload, so data/packs isn't shared), T-3409, T-3360, T-3740 Z3.
5. Steam (T-3729): fee paid, tax form in; her screenshots go in artifacts/trainworld/steam/screenshots.
6. Later: T-3744 per-stop fare, T-3754, T-3747, busverify's train 4.14 min gap read (a short probe).

## Her words
- 11-19 v0.4.3 scope ("yea no rush"); "player complaints take precedent over other new features", "only applies
  to like actual bugs". Fares by city: country or city multiples, never both.
- 11-19 saves: deleted games kept 7 days ("i think it makes sense"); restore Rus "under new game id", "do the R2".
- 11-19 fares: transfers inside fare zones free; heading hover her wording; Enable as an icon left of the bin
  ("Disable Caracas fare zones"); fare settings per game. Failed-open notice "{name} could not be opened.".
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
  card once the city in view is ready), a-15 (server deploy).
- She sends the T-3635 player reply herself (draft in notes/T-3635.md).

## Only a person knows
- Uncommitted on purpose: TODO.md (hers), notes/T-3397.md, jordan-save-investigation-2026-11-15.md,
  tools/ops/archive_bad_saves.py, changelog/roadmap.json, changelog/next.json, notes/asks-log.jsonl,
  notes/board-live.json; maps .claude/settings.json.
- A yes she gives in another session's tab doesn't pass the permission check here: ask again in this tab.
- Old tree tt2: keep until ttport lands. Empty locked folders .tw-trees/touchrule and .tw-trees/timekeys.
- Never delete a tree folder by hand (its app/node_modules junction deletes the main tree's).
