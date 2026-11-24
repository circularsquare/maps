# Orchestrator handoff, trainworld

Orchestrator: session f51a9e36-f3e8-41ee-b3d9-e908c5e30b16, name maps-2c [36056b], since 2026-11-24 09:03 UTC
Updated: 2026-11-24 09:03 UTC. Live state: `venv\Scripts\python.exe trainworld/tools/handoff.py status`;
diary notes/handoff-log.md. Her time UTC-5.

## In flight (mine, maps-2c; phonemem2 and zoneslow run in maps-c0's tab: follow-ups via "relay to <agent>: <text>" to maps-c0, or a fresh agent on their tree)
- Landed: memgrowth T-4179 c842cd599, phonemem2 T-4184 20d3a0029 (changelog c090). Batch checker check3 (Sonnet) on both;
  check3 PASS both (T-4184's phone memory not confirmed: perf_bench bug T-4189);
  smoke on master PASSED 10:19 (30 checks; real save 13,035,408 in smoke's own count, the usual 1.0286x play.mjs's
  12,672,465); 8800 = 20d3a002 (master minus bookkeeping).
- Landed 10:26-10:35: trams T-4075 9fc02c858 (c091), zoneslow T-4051 96ad90da4 (c089 reworded); check4 PASS both (caveat: NA save live vs reload
  total trips 0.08% apart, rail equal; old build not run: give to stalefinal when it reports);
  8800 = 96ad90da (10:40). stalefinal T-4055 landed f3521af85 (c092), checker check5 running (also the 0.08% gap). maps-c0 closed out.
- stopclick T-4097 stage (Opus): Asia stop click line path 1.4-1.9 s to ~100 ms.
- Landed 11:35-11:45: T-4190 check fix, T-3160 stage 6 53ca4ef77 (phone solving worker 330 -> 261 MB; check6 running), T-4189
  perf_bench WASM fix (rel046 vs v0.4.5 compare was NOT affected). 8800 = 66bdd9f6 (11:51).
- phoneload T-4188 (Opus): Community save jwxidghd never loads on an emulated phone.
- solvereuse T-4057 landed 7289f0d07 (solve order reuses the kept table; Tokyo community -4%, same figures), T-4057 closed;
  needs a checker with stopclick. Next bigger cut T-4124 shifts answers: her call with numbers. sparseload T-4182 (Opus).
- T-4161 done in shonei-server 7bdd559 (level 39 + LBE1 block), unpushed, not deployed: deploy.ps1 on her yes before 0.4.6, then push shonei-server.
- editbench landed (2d8f5e3f9): its stop-click rows on Asia show the line path 1.4-1.9 s after a stop click (T-4097's 100 ms goal): next edit-speed round.
- 0.4.6 prep (from maps-c0): smoke on master PASSED, smoke8800 light+dark PASSED, two batch checkers PASS, changelog
  reviewed (58 shown, 7 highlights), perf_bench rel046-e936af30 vs v0.4.5: demand memory +13/+12/+17% (memgrowth brings
  it to +3/+8/+3%), Belgrade CPU +20%, rail -7.1% Belgrade (trade-off to her). Left: check3's verdict, perf_bench final hash, playtest list, strings ack, her changelog
  edit on the board, T-4153 s2 call, T-4161 deploy, her yes. Nothing published.
- Held for her word: T-4153 s2 (tree editnetwork, station network build on Asia 333 to 124 ms, edit-speed memory ~10.6 MB vs
  her ~10 MB; add mm.ribbons.delete(e) at lineBands.ts:571 if landed); T-921 (tree commutechunks, tunable, no sparse gain);
  T-4156 parked P3 (ask a-2026-11-23-22).
- T-4112's tw/track changelog item is parked in next-track.json: hide the port's duplicate at release.
- Uncommitted SPEC.md "what is covered" rewrite is not ours (likely Gemini in Antigravity): patch it out to land, re-apply.
- Core cap 15 "for the night": back to 14 when she's up (her word). Gemini: only on her ask or our suggestion; Neon medium.

## Next (morning of 11-24; she aims to publish 0.4.6 after her playtest)
1. memland reports: finish memgrowth, Sonnet checker on its diff, smoke on master, rebuild 8800, perf_bench all three on
   the final hash; then start T-3160 (phone memory, demand code) and T-4057's next cut (Opus: reuse the last one-off
   search table, same answers, ~18% Tokyo native; notes/T-4057.md "For solvespeed").
2. Morning message: playtest list, strings, T-4153 s2 call, T-4161 deploy yes, publish yes. 3. Land trams/playprod/cleanup after the publish unless trivially safe. Held while zoneslow runs (same code): T-4055, T-4182.
4. Track rework after 0.4.6.

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
