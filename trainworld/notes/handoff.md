# Orchestrator handoff, trainworld

Orchestrator: session 14439093-662c-49fa-b3bc-84f716dcef40, name maps-20 [14f6c1], since 2026-11-24 12:58 UTC
Updated: 2026-11-24 12:58 UTC. Live state: `venv\Scripts\python.exe trainworld/tools/handoff.py status`;
diary notes/handoff-log.md. Her time UTC-5.

## In flight (all mine)
- Landed today, all batch-checked PASS (check3-5) unless noted: T-4179 demand memory, T-4184 clock memory (c090), T-4075
  trams (c091), T-4051 zone open (c089), T-4055 stale final (c092), T-4183/T-4169/T-4189/T-4190 tools, T-3160 stage 6
  53ca4ef77 (c019 reworded; check6 running), T-4057 7289f0d07 (solve order; checker with stopclick). Smoke on master
  PASSED 10:19 at 20d3a002. 8800 = 66bdd9f6 (11:51). Open follow-ups filed: T-4187, T-4188, T-4191, T-4192.
- Running: check6 (T-3160 s6), stopclick T-4097 stage (Asia stop click line path 1.4-1.9 s to ~100 ms), phoneload
  T-4188 (save jwxidghd never loads on an emulated phone), sparseload T-4182.
- T-4161 done in shonei-server 7bdd559 (level 39 + LBE1), unpushed, not deployed: deploy.ps1 on her yes before 0.4.6,
  then push shonei-server. Asked her in chat.
- 0.4.6 prep (from maps-c0): smoke8800 light+dark PASSED, changelog reviewed (58 shown, 7 highlights), perf_bench
  rel046-e936af30 vs v0.4.5 (valid, T-4189 checked): demand memory +13/+12/+17% (T-4179 brings it to +3/+8/+3%),
  Belgrade CPU +20%, rail -7.1% Belgrade (trade-off to her). Left: check6, smoke on the final hash, perf_bench all three
  on it, playtest list, strings ack, her changelog edit on the board, T-4153 s2 call, T-4161 deploy, her yes.
- Held for her word: T-4153 s2 (tree editnetwork, station network build on Asia 333 to 124 ms, edit-speed memory ~10.6 MB vs
  her ~10 MB; add mm.ribbons.delete(e) at lineBands.ts:571 if landed); T-921 (tree commutechunks, tunable, no sparse gain);
  T-4156 parked P3 (ask a-2026-11-23-22); T-4124 (one graph per cluster) shifts answers: numbers to her first.
- T-4112's tw/track changelog item is parked in next-track.json: hide the port's duplicate at release.
- Uncommitted SPEC.md, docs, checks/old moves, tools/asks.py etc. are not ours (likely Gemini in Antigravity): leave;
  patch out to land if a landing touches them (checks/INDEX.md was regenerated once, 10:34).
- Core cap back to 14 at 12:58 (she's up; her 05:0x word). Gemini: only on her ask or our suggestion; Neon medium.

## Next (she aims to publish 0.4.6 after her playtest)
1. check6 PASSED. 0.4.6 candidate = f63b105b3 (incl. T-4057; check7 running on it). Smoke on it PASSED 12:33. Changelog
   review 2 done (c090, c092 hidden; c019 reworded). New zones image made (artifacts/trainworld/review/v0.4.6-new-zones.png).
   8800 = f63b105b (12:35). Running in maps-2c's tab: smoke8800 (logs artifacts/trainworld/smoke8800-*.log), then perf_bench
   build+run rc046-f63b105b all three (perf ledger; dist artifacts/trainworld/perfbench/dist/rc046-f63b105b). 12:53 Anita: her
   terminal stopped responding (board pad); maps-2c opened a successor tab for her.
2. Morning message: playtest list, strings, T-4153 s2 call, T-4161 deploy yes, publish yes (publish skill).
3. Land stopclick/sparseload/phoneload as they report; a checker per batch. Track rework after 0.4.6.

## Her words
- 11-24 05:0x: core cap 15 "for the night" (back to 14 when she's up); start T-3939/T-3160 overnight as cores free; T-4156 later (P3).
- 11-24 02:1x: edit-speed streams "in parallel" meant done the same way (similar changes), not independently; the Opus review checks every stream uses the base (T-4145) the same way. Ask before leaving 2-8 agents; the 12 run on, no pausing. 02:0x: Resend Pro no; R2 purge yes;
  R2 reads and the website repo (till 1.3 GB) not reported (docs/ops-checks.md).
- 11-23 00:0x: the drawn network rebuild is "the entire issue. this is what we should fix", not left to the track rework;
- 11-23 23:1x: edit speed: "just do all of the changes and i can test, no need for many rounds of implement-test. lemme
  know when everything is done". Run every T-4144 stream, land all, then one message to her; no interim test asks.
- 11-23: your two top priorities are answering her and never letting agents overbook or collide; agents 2-8 run by
  load; size tasks to about an hour; you don't investigate, agents do; cycle at 500k day and night; no drafted player
  replies; report times as percent of the wait; she has no Apple devices.
- 11-23: 2 crowding rounds refused, final; master heavier than v0.4.5 is expected (express fix, bus changes), look
  for gains elsewhere; coarse setting yes (on by default on phones); install banner yes after 2 h, dismiss forever;
  deleting track never deletes stations; lines side by side on and a $10B start are right.
- 11-19: "player complaints take precedent over other new features"; full playtest list before every publish. 11-22:
  lines independent; softening "75% of the way"; China shape fix next week (T-4044).
- Standing: every publish, deploy and website push needs her yes in this tab, none overnight; advisor only for big
  design decisions; BRT stations for BRT lines only.

## Only a person knows
- Uncommitted on purpose: TODO.md (hers), notes/T-3397.md, jordan-save-investigation-2026-11-15.md,
  tools/ops/archive_bad_saves.py, changelog/roadmap.json, changelog/next.json, notes/asks-log.jsonl,
  notes/board-live.json; maps .claude/settings.json. Not mine: notes/T-2380.md, notes/T-3382.md,
  tools/feedback-inbox/builds.json, tools/release/new_zones_map.py, notes/T-3821.md, notes/T-3896-report.png.
- A task with stages left: `agent-tree.mjs land` + `remove`, not `finish`; swap the taking mark by hand.
