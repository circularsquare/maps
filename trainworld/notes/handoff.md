# Orchestrator handoff, trainworld

Orchestrator: session 9036b094-ff61-4bec-9dce-e9d9c5d30c00, name maps-cb [60eeab], since 2026-11-22 16:42 UTC
Updated: 2026-11-22 16:55 UTC (maps-17's process was gone at takeover; its four agents died with it)

Live state: `venv\Scripts\python.exe trainworld/tools/handoff.py status`; diary notes/handoff-log.md. Her time UTC-5.

## Next
T. T-4041 LANDED ac526280 (her top priority; "use the rest of our usage toward this end"; reset ~11-23 10:00 UTC):
   first-arrival rule, 0.75 softening, late joins, street trams no 2+2. +45% solve / +14% heap (she said "probably
   worth it" to +13%: asked again; T-4054 filed to cut it). Running: batchcheck14 (sonnet, master, port 6141) and
   evenspaced (opus, T-4050, tree evenspaced). Both in the next update, "probably tomorrow evening" (11-23).
   batchcheck14: T-4041 PASS; bar never ended full -> fixed 0c336c7f; T-4055 a stale final after a two-zone edit,
   older than T-1354, queued (P2), not holding the publish. Her 11-22 ~23:30 UTC: keep the cliff fix, try the cheaper one (T-4054,
   P1); first-train quarter stays agnostic about changes (the deadline keeps them no later than direct).
   T-4050 LANDED bc5d4f39 (real save unchanged: its lines have their own stations). T-4054 stage 1 landed 4a78edcc
   (search 10% faster, solve ~2%, same answers; solve still ~+40%: rest waits on her). batchcheck15 PASS on T-4050
   + T-4054 (bar-end FAIL fixed fa196b91, smoke PASS). 8800 -> fa196b91. Pushed. Next: the publish skill for her
   update (11-23 evening) after the usage reset: freeze, translations, smoke8800, changelog review, her text OK.
   Slowdown accepted (her 11-23 ~02:00 UTC: "thats okay. we can look for further speedups elsewhere, tomorrow"):
   T-4054 P1 for 11-23, speedups anywhere in the solve.
   Leftover folder .tw-trees/arrivefirst (not a worktree; node_modules junction inside: don't delete by hand).
   Her hope: "buses get more reasonable values relative to trams ... more uniform ... cuz everything is parallel".
U. Everything else paused for usage, trees kept, {after:next week's usage}: T-4024, T-4043, T-3346, T-3932, T-4025,
   T-3373, T-3688 (hokkaido: join done, after-steps left; data/packs lacks sor/ibr/okh/ksr until it lands), T-3939,
   T-4045 (memo fix uncommitted in tree onewaylag, wants a checker; page profile left). Track preview after hokkaido.
0. LIVE v0.4.5. Cap 14 (her daytime pick 11-22). Cores 0-1 hers (`reserve_cores.ps1 -Loop 30` at takeover). No
   publish yet. Before one: fresh freeze, translations (T-3986, T-3991, T-3999, T-4003, T-4010, T-2264, T-3933,
   T-3378 s4, T-4013, T-4005, T-4040, and the new zones' names: place_names.mjs --zones), smoke on a quiet machine. Batch 13
   + the night's data passed a checker on 547389c2 (11-22). Server deployed 3b14158. Save level 38 on master.
1. 8800 = 547389c2. master since: T-4040 (UI), T-1354 (demand worker). Once hokkaido's join ends (it owns
   data/packs): sync-track, then 8800 = tw/track tip as her preview (her yes 11-22; saves made there won't load on
   master or live: told her), and a batch checker on master. All strings to 11-22 16:10 UTC acknowledged ("Part of a bundle of # tracks", her wording).
2. TRACK (her top priority): tw/track has 1-3g, 4a1, 4a2, 4c, 4e picture, T-4030 a-f, T-4038, levels (T-3691), 4b's
   two fixes. T-3961's merge waits for g; T-3964 Join waits T-4043. g (T-4037): her calls in the task (compare her
   cut-at-stations idea first); then 4d, 4f, 5a/5b. Pre-merge: T-4026, T-3958, a checker. Plan: notes/T-4030.md.

## Her words
- 11-22 ~16:05 UTC: cap 14 (daytime); T-4041 yes, high prio (and buses beating parallel trams; tram boarding
  penalty maybe undone for street trams); China shape fix next week (T-4044); Georgia fine; T-4025 start; tw/track
  preview on 8800 yes. Earlier: "lets wind down", then "can keep going". Hokkaido via the old Japan tool: fine.
- 11-21 ~22:40 UTC: deploy and wiki deferred till home; T-3376 yes; T-3742 planned, start later; T-4008 no soft
  rule, drop income groups (T-4019); busways any bus.
- 11-21 ~18:45 UTC: main priorities: keep the track rework moving, finish and polish BRT. Standby orchestrator tabs
  (notes/standby.txt, docs/orchestrator.md "Your own cycling").
- 11-19: "player complaints take precedent over other new features"; full playtest list before every publish. When
  she plays: no uploads until she's done; she may ask for a lower core cap.
- Standing: lean testing, smoke per batch; long jobs in the background, answer her first; every publish, deploy and
  push to the website needs her yes in this tab, none overnight; advisor trial (log calls in handoff-log); BRT
  stations for BRT lines only; never ask again whether big saves get more workers (11-21).

## Waiting on her (asks.py list has the open ones)
- All re-asked in the 11-22 ~16:00 UTC catch-up (maps-17), plus the 4-track sheet track-4034.png to look at.
- She sends the T-3635 player reply herself (sent her the text 11-22). T-3945 never had a reply draft. Quote text in
  the message, never send her a notes file (her 11-22).

## Only a person knows
- Uncommitted on purpose: TODO.md (hers), notes/T-3397.md, jordan-save-investigation-2026-11-15.md,
  tools/ops/archive_bad_saves.py, changelog/roadmap.json, changelog/next.json, notes/asks-log.jsonl,
  notes/board-live.json; maps .claude/settings.json. Not mine: notes/T-2380.md, notes/T-3382.md,
  tools/feedback-inbox/builds.json, tools/release/new_zones_map.py, notes/T-3821.md, notes/T-3896-report.png.
- notes/T-3688/kit and notes/T-3444-design/tasks-to-file.md are untracked working files (kit: the redraw scripts).
- A yes she gives in another session's tab doesn't pass the permission check here; undo of +5 trains doesn't
  refund (batch checker 11-22, likely by design).
- A task with stages left: `agent-tree.mjs land` + `remove`, not `finish` (it archives); swap the taking mark by hand.
