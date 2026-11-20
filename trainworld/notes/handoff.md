# Orchestrator handoff, trainworld

Orchestrator: session 79e26413-4648-481b-ba5a-8ddcd716a0f0, name maps-60 [6ba449], since 2026-11-20 05:20 UTC
Updated: 2026-11-20 15:00 UTC (maps-60 [6ba449] past 500k, starting its successor)

Live state: `venv\Scripts\python.exe trainworld/tools/handoff.py status`. This file is plan and her words.
Diary: notes/handoff-log.md. Her local time is UTC-5.

## Next
0. Live v0.4.4. Cap 14. 8800 = 7637e0e0 (smoke passed). Landed since, not on 8800: T-3846, T-3847, T-3845,
   T-3870, T-3867 (board). Master pushed through ~14:00; push again after the next landings.
1. Running (reports come via the inbox; adopt them): gapfill (Opus, T-3764, main tree: JOIN of 338 PK+BD rows
   from 13:20, pid 9092, log artifacts/trainworld/t3764-join.log, at its register step 14:52; no data/packs writes,
   no WASM landing, no plain 8800 rebuild until it reports; then commit its files), riverfix (Opus, T-3839, her
   river report), bigload (Opus, T-3868, P1, her NA save artifacts/trainworld/saves/na-day-491.save, + bar
   calibration), savefail (Opus, T-3869, P1, player's big game can't save; its notes top has advice she can send
   the player), rampspeed (Opus, T-3027: checker before landing, sim speed).
2. After the join: land islands1 (T-2216 stage 1, tree islands1, islandchk PASSED; WASM rebuild; then rewrite
   T-2216 to stages 2-3), one smoke with free cores (islands1's own smoke timed out on 1 core), 8800 rebuild, push.
3. T-3395: 11-20 scan (pass 1, not done) found 0 new damage. Run `tools/ops/save_health.py` after the 11-21
   04:00 UTC scan; clean = close it (the two rev-1 games of ue83342ed78f8fc4c are old 11-19 damage).
4. Up next: P2 raised 11-20 (T-3655, 3646, 3735, 3513, 2486, 3829); T-2216 stages 2-3; after gapfill: T-3838,
   T-1376 retry, T-3843, rest of T-3764, T-1738 Bhamo. No minor track work (her 11-20).
5. Next update: unpublished changelog c001-c015 plus tonight's; untranslated: T-3822's and T-3823's hover strings,
   diagramView.addCity; use the publish skill (review agent, sort, playtest list, translations, new zones image).
6. Leftover trees ttchk2base, touchrule, timekeys. Steam (T-3729): her shots in artifacts/trainworld/steam.

## Her words
- 11-20 ~12:45 UTC, morning: new countries "look pretty cool"; rivers still break (T-3839, her words there); ferry
  order port, waypoints, port (T-3845); quieter related lines (T-3846); islands "3x, A", stage 2 yes.
- 11-20 ~13:40 UTC: T-3835 text ack. TODO.md "for anita" and player suggestions all filed ("basically want all of
  this eventually"; T-3850-T-3866, notes/todo-review-2026-11-20.md). Legacy time buttons for new players (T-3847).
  Raised 8 to P2 ("lets raise those yeah"); ferries "be able to run longer" (400 km per leg stays).
- 11-20 ~14:00 UTC: "i'd still rather not work too muhc on track stuff. i think i'd like to start thinking about
  full track rework in next few days": no minor track work (T-3662, T-3353 wait on the rework design; T-2920
  stays P4). Ferry text ack (T-3845). Board: waited-on task in its own preview section (T-3867, done).
- 11-20 ~14:45 UTC: slopes "kinda separate from track redesign ... so we can do this soon": 200 m per level = 160,
  400 m = 320, floors trains 5% / trams 8% (T-3027); real terrain as cost later (T-3871, P4).
- 11-20 early: board changelog review is NOT the publish text OK; "next up" bullets now and then and always after
  shipping an update; Resend stays free; intercity effective time "keep it as is".
- 11-19: "player complaints take precedent over other new features" (actual bugs). Before every publish, a full
  playtest list. Fares by city: country or city multiples, never both.
- 11-18: close easy high-priority wins, small over design-heavy. "less willing to put stuff on the changelog".
- Share previews: free Worker, careful with the Cloudflare account. Mod support (T-3377): never say publicly it's
  aimed at vibe-coders. caelum418's speed PRs (T-3659): next time we look at performance.
- When she plays games: keep 4 cores free (cap 12) and no uploads until she says she's done.
- Standing: lean testing, smoke per batch; long jobs in the background, answer her first; every publish, deploy and
  push to the website needs her yes in this tab, none overnight; advisor trial (log calls in handoff-log).

## Waiting on her (asks.py list has the open ones)
- She sends the T-3635 player reply herself.

## Only a person knows
- Uncommitted on purpose: TODO.md (hers), notes/T-3397.md, jordan-save-investigation-2026-11-15.md,
  tools/ops/archive_bad_saves.py, changelog/roadmap.json, changelog/next.json, notes/asks-log.jsonl,
  notes/board-live.json; maps .claude/settings.json. Not mine: notes/T-2380.md, notes/T-3382.md.
- A yes she gives in another session's tab doesn't pass the permission check here: ask again in this tab.
- Never delete a tree folder by hand (its app/node_modules junction deletes the main tree's).
