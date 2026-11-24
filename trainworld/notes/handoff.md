# Orchestrator handoff, trainworld

Orchestrator: session 5cd3af11-2ab8-4788-8daf-1956104ead51, name maps-c0 [58120f], since 2026-11-24 01:57 UTC
Updated: 2026-11-24 05:16 UTC. Live state: `venv\Scripts\python.exe trainworld/tools/handoff.py status`;
diary notes/handoff-log.md. Her time UTC-5.

## In flight (agents in my tab; ids: editnetwork2 a935f09f, trainmove acca7982, commutechunks2 a7094f4a, memgrowth a84b996f, checkclick a9c53454, checksim a170c31b, treefix a4a15622)
- Landed tonight (since 8800's ef99d979): T-4145 base, T-4153 s1, T-4147 s1-2, T-4132, T-4155, T-4146, T-4148, T-4149,
  T-4170 (one path, Gemini-reviewed, fixes verified), T-4158 bus merge 500 m (+ fix for a prod-build boot crash: circular
  import clockWord.ts/clockWordLines.ts), T-4162 text, T-4054, T-4143, T-4112, T-4152.
- Running: snap8800 of f4bcf26b (log scratchpad/snapbuild.log); Gemini final batch review -> notes/gemini-editspeed-final-
  2026-11-24.md (the last of this round; after it Gemini only on her ask or my suggestion); batch checkers checkclick and
  checksim (notes/batchcheck-2026-11-24-*.md); editnetwork2 T-4153 s2; trainmove T-4133; commutechunks2 T-921 (to her);
  memgrowth T-4179; treefix T-4180 (finish lost busmerge2's unfinished merge; recovered).
- Checkers done 05:58: both PASS what they ran; 8800's prod build boots; NOT checked: real NYC save on 8800 (play.mjs can't drive prod, T-4183: her playtest covers it), Belgrade figures after the bus merge (perf_bench belgrade will show rail trips).
- editnetwork2 T-4153 s2 done in tree editnetwork (station network build on Asia 333 to 124 ms) but edit-speed memory reaches ~10.6 MB with it (her budget ~10 MB): hold landing for her word in the morning. If she says yes, also add editfix's finding 10: mm.ribbons.delete(e) at lineBands.ts:571.
- Then: smoke on master once editfix lands; perf_bench all three saves; land editnetwork2/trainmove into a second
  small batch; publish skill prep; morning message. Queue: T-3939, T-3160 (her yes overnight), T-4178, T-4051.
- T-921 built in tree commutechunks (default 1 a day, tunable; no gain on sparse saves; NYC 6 trips in 12.7M move): land only on her word (morning). Its tree also has its own fix for the clockWord import loop: keep master's at landing. The main tree's app/src/wasm was rebuilt with the agent profile by that agent: the next sim landing (or npm run wasm) restores it.
- 8800 = f4bcf26b (05:23). T-4112's tw/track changelog item parked in next-track.json: hide the port's duplicate at release.

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
