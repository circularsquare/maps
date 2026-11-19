# Orchestrator handoff, trainworld

Orchestrator: session 953236ac-bf0a-4ac1-9b7d-bcd87f6165e0, name maps-c2, since 2026-11-19 21:24 UTC
Updated: 2026-11-19 21:30 UTC (maps-e8 wound down, all committed at 7fb47f75; its agents linesort, schembus,
pollaudit, guianas still run in its tab and report to the inbox; maps-c2 started limitcheck, cityfares, periodword)

Live state: `venv\Scripts\python.exe trainworld/tools/handoff.py status` (T-3667). This file is plan and her words.
Diary: notes/handoff-log.md. Her local time is UTC-5.

## Next
0. Live v0.4.3 (published 21:05; server deployed, Rus restored). Cap 14. Old R2 versions can be purged later
   (r2_purge.py). Sai and ber reshipped into data/packs/real for the next update (7b5ea227).
1. Running: linesort (Opus, T-3798 + T-3423, 6170), schembus (Opus, T-3360 [+T-3785], 6171), pollaudit (Opus,
   T-3801, 6172), guianas (Opus, T-3762, 6173, writes data/packs) - all in maps-e8's tab, reports via the inbox;
   cityfares (Opus, T-3766, tree, 6175; checker before landing). Landed: T-3800 d39db027; T-3798 + T-3423
   2997f81b (save level 30, LGN1; linechk passed). shonei-server 4ba1627 (savecheck level 30) committed, not pushed
   or deployed: that deploy must go out before the next game publish. Rebuild 8800 once after
   this round's landings (board's master hash stale till then).
1b. HOTFIX v0.4.4 (her ~21:35 via maps-e8, which relays her by remote control ~2 h): ringfix (Opus, T-3802, tree
   from d39db027, i.e. v0.4.3 + de/ca, NOT master's level 30; 6178). Publish from that branch only, with only its
   changelog item; work out before publishing how deploy.py builds a non-master commit and keep c001/c002 and
   cityfares' items unpublished. Then land the fix on master too. T-3766 landed 87c811bf (cityfchk passed; next
   translation pass drops advancedFares.byCountry). zonez3 (Opus, T-3740 Z3, tree from 87c811bf, 6180) running.
2. Next to start: T-3740 Z3 after cityfares lands (same fare panels); T-1738 / T-3763 after guianas; T-3790 step 4
   once she answers a-2026-11-19-20 (limitcheck's answer in notes/T-3790.md); T-3780 timetables, T-3799 design to
   her, T-3726 commentab pairs 1 and 3.
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
- a-2026-11-19-20: Moscow limits, build-only fix and keep Moscow near 106 lines? (T-3790). She tells Rus herself.
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
