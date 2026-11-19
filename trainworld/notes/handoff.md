# Orchestrator handoff, trainworld

Orchestrator: session a0ff2795-d3f6-4076-b0f1-e96b327ba6cf, name maps-e8, since 2026-11-19 17:00 UTC
Updated: 2026-11-19 20:30 UTC

Live state: `venv\Scripts\python.exe trainworld/tools/handoff.py status` (T-3667). This file is plan and her words.
Diary: notes/handoff-log.md. Her local time is UTC-5.

## Next
0. Live v0.4.2. Master 8c4c00b2 = the v0.4.3 candidate (changelog published into the game, 12 languages
   translated, back-translated and reviewed). Cap 14. shonei-server not deployed or pushed: e604291 (levels 27-29),
   93718d1 (T-3775), 5bf10c7 (T-3382), 6ab8b68 / ada0bce / 8ba8657 (T-3789 deleted games kept 7 days).
1. v0.4.3 release, her out with remote control (~19:30: "ill enable remote control and approve final deploy while
   outside"). 20:30: smoke on master + snap8800 8c4c00b2 running; then smoke8800 light and dark; then ONE ask to
   her: (a) server deploy (`.\deploy\deploy.ps1` in shonei-server, then push) with SHONEI_GAMES_ADMIN_KEY set on
   the box and in ~/shonei-games-admin-key.txt (her yes to the key given ~18:05), then Rus's restore
   (`restore_game.py put uef1fbdd21e42c812 "artifacts/trainworld/rus-recovery/Jakarta 3.save" --name "Jakarta 3"
   --stations 1855 --lines 85`, dry run then --yes; delete rus-recovery after); (b) data upload (deploy.py, data/packs
   incl. real saves: 19 new, mos/sai/ber kept at v0.4.2); (c) publish (deploy.py --smoked, website push, --verify
   --page, open-source export + push). New zones image: `tools/release/new_zones_map.py --version v0.4.3` first.
   ~20:47 server deployed + pushed, key set (shonei.env line 11), Rus restored as 35kkah6wvvxq, rus-recovery deleted;
   T-3775/3786/3789 closed. v0.4.3 PUBLISHED ~21:05: data 2026-11-19 on R2, website e0bf45e pushed, --verify --page
   all PASS, open-source f7446ec pushed. Old R2 versions (2026-11-13 etc.) can be purged with r2_purge.py later.
   Running for the next round (her ~20:50 "lets also pick up and start work for the next round"): linesort (Opus,
   T-3798 + T-3423 nested groups, 6170), schembus (Opus, T-3360 [+T-3785], 6171), pollaudit (Opus, T-3801, 6172),
   oldgraph (Opus, T-3790 3b; never writes data/packs). Countries T-3762 etc. wait for the upload to finish.
2. After the release: T-3790 step 4 (old-rule real-networks graph, ~1 h + rebuild; notes/T-3790.md, saves in
   artifacts/trainworld/t3790/corridors), T-3801 (edits stuck behind panel polling + docs rule), T-3798 / T-3799
   (Lines tab sort and columns; T-3799 design to her first), T-3800 (de/ca period word), T-3766 (fares by city),
   T-3780 (station timetables), countries T-3762 / T-1738 / T-3763, T-3726 commentab pairs 1 and 3, T-3740 Z3.
3. T-3781 stays open for the next report's diagnosis; T-3786 / T-3789 close once Rus has the game; T-3775 closes
   once save_health.py reads the live route after the deploy.
4. Steam (T-3729): fee paid, tax form in; her screenshots go in artifacts/trainworld/steam/screenshots.
5. Later: T-3744 per-stop fare, T-3754, T-3747, busverify's train 4.14 min gap read (a short probe).

## Her words
- 11-19 ~20:20: the rename stuck behind polling is a shape to remember for future features; check for current
  parallel failures, "no rush" (T-3801).
- 11-19 ~19:30: "changelog looks good (i think lets not change it at all)"; start line hers: "update v0.4.3: separate
  rush periods, advanced fares, caribbean!". Before every final check, sort the changelog (docs/orchestrator.md).
- 11-19 playtest: fare zones one area per station with a confirm, unique names, checkbox Enable; station rings only
  while placing; renames faster; name setting asks both ways, click-off is No; schedule rows and her hover text.
- 11-19 v0.4.3 scope ("yea no rush"); "player complaints take precedent over other new features", "only applies
  to like actual bugs". Imperfect real networks ship ("people can just fix them themselves"). Fares by city:
  country or city multiples, never both.
- 11-19 saves: deleted games kept 7 days; restore Rus "under new game id", "do the R2"; privacy page in her words.
- 11-19 fares: transfers inside fare zones free; fare settings per game. Failed-open notice "{name} could not be
  opened.". Zones: lines choose zone or distance fares (T-3758). Station names: a bind setting, default on.
- 11-19: before every publish, a full playtest list (docs/orchestrator.md). Scratch of finished sessions 2+ days
  old is deleted without asking (memory). Money history: keep the global record forever; per-line a month.
- 11-18: close easy high-priority wins, small over design-heavy; "prio should take into account easiness as well
  as impact". "we need to be less willing to put stuff on the changelog". "manually", never "by hand".
- Share previews: free Worker, "please be careful" with the Cloudflare account. Mod support (T-3377): never say
  publicly it's aimed at vibe-coders. caelum418's speed PRs (T-3659): next time we look at performance.
- When she plays games: keep 4 cores free (cap 12) and no uploads until she says she's done.
- Standing: lean testing, smoke per batch; long jobs in the background, answer her first; every publish, deploy and
  push needs her yes to that action, none overnight; advisor trial (log calls in handoff-log).

## Waiting on her (asks.py list has the open ones)
- a-1 (T-3746 text), a-13 (read today's 2 damaged saves over ssh; or wait for the deploy), a-14 (drop the load
  card once the city in view is ready), a-15 (server deploy: part of the one release ask).
- She sends the T-3635 player reply herself (draft in notes/T-3635.md).

## Only a person knows
- Uncommitted on purpose: TODO.md (hers), notes/T-3397.md, jordan-save-investigation-2026-11-15.md,
  tools/ops/archive_bad_saves.py, changelog/roadmap.json, changelog/next.json, notes/asks-log.jsonl,
  notes/board-live.json; maps .claude/settings.json. Not mine, left alone: notes/T-2380.md, notes/T-3382.md.
- A yes she gives in another session's tab doesn't pass the permission check here: ask again in this tab.
- Leftover trees: ab-1-*, ab-3-* (commentab), ttchk2base (a checker's scratch; remove), empty locked
  touchrule and timekeys.
- Never delete a tree folder by hand (its app/node_modules junction deletes the main tree's). It happened again
  11-19 (~18:20 UTC: .bin and 15 packages gone; `npm install` inside trainworld/app put them back).
