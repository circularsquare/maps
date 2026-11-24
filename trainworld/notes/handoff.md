# Orchestrator handoff, trainworld

Orchestrator: session 5cd3af11-2ab8-4788-8daf-1956104ead51, name maps-c0 [58120f], since 2026-11-24 01:57 UTC
Updated: 2026-11-24 07:33 UTC. Live state: `venv\Scripts\python.exe trainworld/tools/handoff.py status`;
diary notes/handoff-log.md. Her time UTC-5.

## In flight (agents in my tab: memgrowth T-4179, check2 (second batch check), zoneopen2 T-4051, hookfix T-4174)
- Landed overnight after the first checkers: T-4133 527a59e1, T-3939 fc4e5a14, T-4181 e2d72bf0 (Gemini final fixes),
  T-4185 e20fa32b, T-4178 6b68241c (reversals; 2 new strings for her ack). Smoke on master e936af30 PASSED (30 checks).
  perf_bench rel046-e936af30 vs v0.4.5: demand memory +13% nyc, +12% tokyo, +17% belgrade; belgrade CPU +20%, rail
  trips -7.1% belgrade (to her as a trade-off; memgrowth working on memory). snap8800 of 6b68241c running.
- Publish prep done 07:54: changelog reviewed (58 shown, 7 highlights; her edited c080, c035, c017 reworded), smoke8800 PASSED light and dark on 6b68241c. Left for the morning: her playtest, strings ack (reversal hovers), changelog edit on the board, T-4153 s2 call, rerun perf_bench on the final hash, then her yes.
- Held for her word: T-4153 s2 (tree editnetwork; ~10.6 MB edit-speed memory; add lineBands.ts:571 ribbons.delete),
  T-921 (tree commutechunks, tunable, no sparse gain).
- Uncommitted SPEC.md "what is covered" rewrite is not mine (likely Gemini in Antigravity): leave it; patch it out to land.
- 8800 = f4bcf26b until the snap swaps. T-4112's tw/track changelog item parked in next-track.json: hide the port's duplicate at release.

## Next (night of 11-24; she aims to publish 0.4.6 in the morning after her playtest)
1. Land editunify T-4170 once its Gemini fixes are verified; then the one Gemini review of the whole edit-speed batch
   (her OK for this round; after it Gemini only on her ask or my suggestion).
2. Land busland T-4158, editnetwork2 T-4153 s2, trainmove T-4133 as they report (all risky: in the batch).
3. Batch checker (fresh Sonnet) on master over everything since 8800's ef99d979, one smoke on master, perf_bench all three
   saves (edit-speed memory budget ~10 MB); fix or revert failures; rebuild 8800; run the publish skill (no publish
   without her yes, none overnight); morning message to her with the playtest list.
4. Not in 0.4.6 unless done and checked: T-921 (commutechunks2, numbers to her), T-4178 reversals, T-4051 (map delay),
   T-4156 (waits on her). Track rework after 0.4.6.

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
- .tw-trees/arrivefirst is a leftover folder with a possible node_modules junction: never delete it by hand.
- A task with stages left: `agent-tree.mjs land` + `remove`, not `finish`; swap the taking mark by hand.
