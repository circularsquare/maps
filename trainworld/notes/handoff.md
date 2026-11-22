# Orchestrator handoff, trainworld

Orchestrator: session 427323a8-53ce-483c-9c28-8f1452e36f9f, name maps-6a [a48b3e], since 2026-11-21 20:45 UTC
Updated: 2026-11-22 03:40 UTC (cycling at 500k to standby maps-5a)

Live state: `venv\Scripts\python.exe trainworld/tools/handoff.py status`; diary notes/handoff-log.md. Her time UTC-5.

## Next
0. LIVE v0.4.5. Cap 8 (her ask 11-21 evening, "for a while"; asked whether 16 overnight, no answer). Cores 0-1 hers
   (`reserve_cores.ps1 -Loop 30` at takeover). No publish yet. Before one: fresh freeze, translations (T-3986, T-3991,
   T-3999, T-4003, T-4010, T-2264, T-3933, T-3378 s4, T-4013, T-4005), smoke on a quiet machine, and one batch
   checker for batch 13 (T-3502 airports, T-3655 edit speed, T-4022, T-2264, T-3373 s4-5 flagged). Server deployed
   3b14158 (11-22 01:26 UTC). Standby tabs: maps-5a, maps-17.
1. master 7788f682; 8800 = b0647110 (save level 38).
2. TRACK (her top priority): stages 1-3d landed; tw/track 554a83b8 (TWT7 body level 2, save level 40). Quad call
   done in the design (T-4023: any single tracks, Add a track T-4024 with her refusal wording). Next 3e T-3956, 3g
   T-3958: asked her whether to start them at ~19% usage (I'd wait). Pre-merge items: T-4026. A finish onto
   tw/track can stop on a docs conflict: resolve in the tree (CRLF files), rebase --continue, rerun tests, finish.
3. T-3373 B: stages 3-5 landed under ?split=1; stage 6 (edits) next, not started (usage).
4. Overnight: Muzaffarpur rebuild job PID 35592 (haj split into haj, hjp, sxp; uji retired), log
   artifacts/trainworld/redraws-run.log; morning check and hand steps in notes/T-3688.md. Manila designed, not
   applied (a-..-22); Hokkaido designed, needs a day session via japan_zones.py (a-..-23).
5. Waiting on her: T-2264 help panel text (graph labels), T-4025 perf ledger go, 3d sheet look. Wiki on the website
   on hold (fan wiki; pages in artifacts/trainworld/wikisite/site-hold). No agents running.
## Her words
- 11-21 ~20:00 local: deploy yes (done); wiki link off the site; drop the city help panel, trip panel takes her
  text (notes/T-2264.md) and a graph; stop-mark and bookmark tweaks (T-4022); quad = any single tracks (T-4023).
  Asked how fast walking is: 6 km/h on 1.3x the straight line (about 4.6 km/h as the crow flies).
- 11-21 ~22:40 UTC: deploy and wiki deferred till home; T-3376 yes; T-3742 planned, start later; T-4008 no soft
  rule, drop income groups (players say riders skip the express: T-4019); busways any bus; hide empty Bookmarked
  tab; text ack (bookmarks, boarding-only); help panels not seen yet.
- 11-21 ~22:20 UTC (outside): "running high on usage so i think we can wind down starting new tasks, except we
  should keep working on anything related to track rework, or any pressing bugs". Running agents finish; no
  replacements unless track rework or a pressing bug (overrides 4-12 agents for now).
- 11-21 ~18:45 UTC: main priorities now: keep the track rework moving, finish and polish BRT. She's out with remote
  control on this session; standby orchestrator tabs (notes/standby.txt, docs/orchestrator.md "Your own cycling").
- 11-21 ~17:50 UTC: "text ack" for everything listed to then (viewonly, BRT stations, modeldown, brtblueprint,
  community, stationkinds). Search must reach far towns, "general" against crashes (T-4004 built). Server deploy yes.
- 11-21 ~17:00 UTC: closed T-3365; levels -5 to +5 with the track rework, Settings toggle (T-3691). Station totals by
  mode with station timetables (T-3780). Exact sums OK ("<0.5% is never that much"). Community: open paused,
  blueprints hidden, no leave confirm (T-4002); bookmarks maybe (T-4005); raise limits (T-4006, deployed).
- 11-21 ~12:50 UTC: BRT stations for BRT lines only; T-3373 B go, the memory limit rises (players set it: T-3932; a
  load out of memory retries lower: T-3933). Never ask again whether big saves get more workers.
- 11-19: "player complaints take precedent over other new features"; full playtest list before every publish.
- When she plays games: no uploads until she says she's done; she may ask for a lower core cap.
- Standing: lean testing, smoke per batch; long jobs in the background, answer her first; every publish, deploy and
  push to the website needs her yes in this tab, none overnight; advisor trial (log calls in handoff-log).

## Waiting on her (asks.py list has the open ones)
- She sends the T-3635 player reply and the T-3945 reply herself (draft in notes/T-3945.md).

## Only a person knows
- Uncommitted on purpose: TODO.md (hers), notes/T-3397.md, jordan-save-investigation-2026-11-15.md,
  tools/ops/archive_bad_saves.py, changelog/roadmap.json, changelog/next.json, notes/asks-log.jsonl,
  notes/board-live.json; maps .claude/settings.json. Not mine: notes/T-2380.md, notes/T-3382.md.
- A yes she gives in another session's tab doesn't pass the permission check here: ask again in this tab.
- Never delete a tree folder by hand (its app/node_modules junction deletes the main tree's).
- Commit messages through PowerShell: single quotes, or "$2M" becomes empty (49fa812e lost "US$2M").
- A task with stages left: `agent-tree.mjs land` + `remove`, not `finish` (it archives); swap the taking mark by hand.
