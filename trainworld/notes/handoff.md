# Orchestrator handoff, trainworld

Orchestrator: session 360c2881-9196-44fd-8596-05db148a51cb, name maps-17 [098830], since 2026-11-22 14:16 UTC
Updated: 2026-11-22 15:10 UTC

Live state: `venv\Scripts\python.exe trainworld/tools/handoff.py status`; diary notes/handoff-log.md. Her time UTC-5.

## Next
W. Back to normal (her "can keep going", 11-22 ~15:50 UTC, after a short wind-down). Running from 15:45: clockmem
   (T-3939), timetables (T-3346 stage 3). master has T-4040 (UI) and T-1354 (demand worker): 8800 rebuild and the
   batch checker wait for hokkaido's join to finish (it owns data/packs). Strings to ack collected in
   artifacts/trainworld/review/strings-to-ack.md.
0. LIVE v0.4.5. Cap is 16 (overnight rule); her daytime number is 8 ("for a while", 11-21): asked her at ~09:00
   local whether to set it back, no answer yet. Cores 0-1 hers (`reserve_cores.ps1 -Loop 30` at takeover). No
   publish yet. Before one: fresh freeze, translations (T-3986, T-3991, T-3999, T-4003, T-4010, T-2264, T-3933,
   T-3378 s4, T-4013, T-4005, T-4040, and the new zones' names: place_names.mjs --zones), smoke on a quiet machine. Batch 13
   + the night's data passed a checker on 547389c2 (11-22). Server deployed 3b14158. Save level 38 on master.
1. master and 8800 = 547389c2 plus bookkeeping (Muzaffarpur, Manila, GE AM AZ AF, T-4029 bubbles, T-4027 Geofabrik
   resolver; changelog c047 c048 c049).
2. TRACK (her top priority): tw/track b19961f2 has stages 1-3g, 4a1, 4a2, 4c, T-4030 a-f, T-4038. 4e's picture landed
   (T-3964 left: Join/Separate after 4b). Running, all landing --into tw/track: convert (4b T-3961, save converter: own checker
   before landing), addtrack (T-4024), levels (T-3691). g (T-4037) waits her two calls; then 4d (T-3963), 4f, 5a/5b. Pre-merge: T-4026,
   T-3958, a checker (save format, sim core). Plan: notes/T-4030.md. A finish can stop on a docs conflict: resolve in
   the tree (CRLF), rebase --continue, rerun tests, finish.
3. Master agents: split6 (T-3373 B stage 6, edits on split workers), packorder (T-1354), hokkaido (T-3688: her 4 regions, Dochuo one zone with Sapporo; writes data/packs: hold other data runs).
4. Waiting on her: T-4025 perf ledger go; T-4041 route choice; China top-up (ask). Wiki on the website on hold.

## Her words
- 11-22 ~15:20 UTC: "lets wind down"; ~15:50 "got it ok, can keep going". Hokkaido via the old Japan tool: fine.
- 11-22 ~04:30 UTC: last day of the usage week, a decent chunk left: aim to finish track stage 4 overnight ("if we run
  out, we run out"); compute-heavy, token-light tasks too (the Caucasus and Afghanistan, done). China populations look
  old: T-4028. Overnight cap 16; back to 8 when she's up (ask).
- 11-22 ~04:10 UTC: 3d sheet fine; Manila keeps Malolos; Hokkaido "i'd rather do one big dochuo zone"; trip ? good.
- 11-21 ~22:40 UTC: deploy and wiki deferred till home; T-3376 yes; T-3742 planned, start later; T-4008 no soft
  rule, drop income groups (T-4019); busways any bus.
- 11-21 ~18:45 UTC: main priorities: keep the track rework moving, finish and polish BRT. Standby orchestrator tabs
  (notes/standby.txt, docs/orchestrator.md "Your own cycling").
- 11-21 ~17:00 UTC: levels -5 to +5 with the track rework, Settings toggle (T-3691). Station totals by mode with
  station timetables (T-3780). Exact sums OK ("<0.5% is never that much").
- 11-21 ~12:50 UTC: BRT stations for BRT lines only; never ask again whether big saves get more workers.
- 11-19: "player complaints take precedent over other new features"; full playtest list before every publish. When
  she plays: no uploads until she's done; she may ask for a lower core cap.
- Standing: lean testing, smoke per batch; long jobs in the background, answer her first; every publish, deploy and
  push to the website needs her yes in this tab, none overnight; advisor trial (log calls in handoff-log).

## Waiting on her (asks.py list has the open ones)
- In the 11-22 morning summary, not yet answered: core cap back to 8; Georgia without Abkhazia and South Ossetia,
  Northern Cyprus with Cyprus (T-3764); text ack for tw/track strings (notes T-3956, T-3957, T-3962, T-4038); the
  4-track sheet artifacts/trainworld/review/track-4034.png; a tw/track preview on 8800 once 3e/3g landed (asked 11-22).
- She sends the T-3635 player reply and the T-3945 reply herself (draft in notes/T-3945.md).

## Only a person knows
- Uncommitted on purpose: TODO.md (hers), notes/T-3397.md, jordan-save-investigation-2026-11-15.md,
  tools/ops/archive_bad_saves.py, changelog/roadmap.json, changelog/next.json, notes/asks-log.jsonl,
  notes/board-live.json; maps .claude/settings.json. Not mine: notes/T-2380.md, notes/T-3382.md,
  tools/feedback-inbox/builds.json, tools/release/new_zones_map.py, notes/T-3821.md, notes/T-3896-report.png.
- notes/T-3688/kit and notes/T-3444-design/tasks-to-file.md are untracked working files (kit: the redraw scripts).
- Undo of +5 trains doesn't refund (batch checker 11-22): likely by design, not checked against the parent.
- A yes she gives in another session's tab doesn't pass the permission check here: ask again in this tab.
- A task with stages left: `agent-tree.mjs land` + `remove`, not `finish` (it archives); swap the taking mark by hand.
