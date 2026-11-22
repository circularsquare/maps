# Orchestrator handoff, trainworld

Orchestrator: session 4f09b9f6-62ab-4330-bda8-3c96eb7a5896, name maps-5a [ac8bba], since 2026-11-22 03:42 UTC
Updated: 2026-11-22 03:42 UTC

Live state: `venv\Scripts\python.exe trainworld/tools/handoff.py status`; diary notes/handoff-log.md. Her time UTC-5.

## Next
0. LIVE v0.4.5. Cap 8 (her ask 11-21 evening, "for a while"; asked whether 16 overnight, no answer). Cores 0-1 hers
   (`reserve_cores.ps1 -Loop 30` at takeover). No publish yet. Before one: fresh freeze, translations (T-3986, T-3991,
   T-3999, T-4003, T-4010, T-2264, T-3933, T-3378 s4, T-4013, T-4005), smoke on a quiet machine, and one batch
   checker for batch 13 (T-3502 airports, T-3655 edit speed, T-4022, T-2264, T-3373 s4-5 flagged). Server deployed
   3b14158 (11-22 01:26 UTC). Standby tab: maps-17.
1. master 7788f682; 8800 = b0647110 (save level 38).
2. TRACK (her top priority): stages 1-3d landed; tw/track 554a83b8 (TWT7 body level 2, save level 40). Quad call
   done in the design (T-4023: any single tracks, Add a track T-4024 with her refusal wording). Next 3e T-3956, 3g
   T-3958: asked her whether to start them at ~19% usage (I'd wait). Pre-merge items: T-4026. A finish onto
   tw/track can stop on a docs conflict: resolve in the tree (CRLF files), rebase --continue, rerun tests, finish.
3. T-3373 B: stages 3-5 landed under ?split=1; stage 6 (edits) next, not started (usage).
4. Overnight: Muzaffarpur rebuild (haj into haj, hjp, sxp; uji retired), restarted 22:46 local after a Geofabrik 404
   (T-4027), PID 23872, log artifacts/trainworld/redraws-run.log, checks in notes/T-3688.md; 8800 keeps the old zones.
   Manila designed, not applied (a-..-22); Hokkaido designed, needs a day session (a-..-23).
5. Waiting on her: T-4025 perf ledger go, Hokkaido image. Wiki on the website on hold (pages in
   artifacts/trainworld/wikisite/site-hold). Agents: trackedit (T-3956), trackstarts (T-3958), manilaspec (T-3688
   Manila), caucasus (T-3764). Track order: 3e -> 3f -> 4a1 -> 4+/4a2 -> 4b-4e -> 4f.
## Her words
- 11-22 ~04:30 UTC: last day of the usage week, a decent chunk left: aim to finish track stage 4 overnight ("if we run
  out, we run out"); compute-heavy, token-light tasks too: Caucasus and Afghanistan (caucasus agent); China
  populations look old, filed T-4028. Cap 16 overnight per the rule; back to 8 when she's up (ask).
- 11-22 ~04:10 UTC: start 3e (trackedit, Opus, syncs master into tw/track first); 3d sheet fine; Manila: keep Malolos
  with Manila (manilaspec revising, run after the IN join); Hokkaido: Hidaka and Shiribeshi with Sapporo? (image to come); trip ? good.
- 11-21 ~22:40 UTC: deploy and wiki deferred till home; T-3376 yes; T-3742 planned, start later; T-4008 no soft
  rule, drop income groups (players say riders skip the express: T-4019); busways any bus. (Her 22:20 usage
  wind-down is replaced by the 11-22 04:30 line.)
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
