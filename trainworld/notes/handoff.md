# Orchestrator handoff, trainworld

Orchestrator: session e152feed-f0d3-4dcd-84f4-e7cb5c5a0a5b, name maps-3f, since 2026-11-18 22:56 UTC
Updated: 2026-11-19 03:35 UTC (maps-3f past 500k, starting its successor)

Live state: `venv\Scripts\python.exe trainworld/tools/handoff.py status` (T-3667). This file is plan and her words.
Diary: notes/handoff-log.md. Her local time is UTC-5.

## Next
0. Live: v0.4.2 (data 2026-11-18b). Unpublished changelog c001-c010 (next update, v0.4.3). Server deployed 01:03
   UTC (shonei-server 962fcae, pushed). Cap 12 cores ("keep 4 cores free... no need to be super strict").
1. Mid-landing (maps-3f finishes it): periods0 T-3542 stage 0, checker slotchk5 PASS; rebased onto master (one
   import clash in clock.worker.ts resolved, typecheck clean); both Rust suites running on the merged tree, log
   scratchpad periods0-merged-test.log. Then `agent-tree.mjs land periods0 ...` + `remove periods0` (T-3542 stays
   open), main-tree WASM rebuild (`npm run wasm` in trainworld/app if land doesn't), 8800 rebuild, smoke on master.
   Then delete data/terrain/*.terrain (4,631 files, 704 MB; keep world.water). If maps-3f didn't get to it, you do.
2. Running, reports come to you through the inbox:
   - advfares (Opus) T-3739 advanced fare panel, tree advfares port 6108 (its Vite may have been stopped by a
     checker; it was told). Needs a checker (money, saves), then land. Then T-3740 zones and T-3744 per-stop fare
     (design first) per notes/T-3739.md "Her answers".
   - oneselbuild (Opus) T-3746 one track selection, tree onesel port 6114: controls only in the selection bar for
     1 or many pieces; shift-click fills between, big detour adds only the clicked piece (her answers in
     notes/T-3746.md). Needs a checker (rebuild charges money), then land.
3. DO NOT LAND tree autolatch as is: T-3720's fix (account.ts origin on score entries) FAILED its checker
   (latchchk): an account's newer copy with a different origin retires the SAME game's entry: 2 DELETEs, its
   Community share gone. Suggested guard: never retire an entry whose id equals the slot's current cloud.id, and one
   dropDeleted at a time. T-3720 was archived by its agent: move it back to TASKS.md (P3, {L:accounts-server}) and
   give a fresh agent the tree, notes/T-3720.md and latchchk's report (scratchpad latchchk), then a checker again.
4. T-3542 next: stage 1 (3/4/5 rows per line) as a fresh Opus agent from notes/T-3542.md "Stage 0" and "Her
   answers"; then stage 2 (hours). T-3346 stage 2 (tree tt2) ports after T-3542 (DECISIONS 11-18).
5. T-3738 share previews LIVE (worker trainsworld-sharepreview 772a7d85 on anita.garden/trainsworld/* + 13
   no-worker routes; free plan). She couldn't find fail open; skipped (35-51k loads/day vs 100k). Watch daily Workers
   requests for a week (~2.5k in its first 2 h, normal); then close T-3738 (c009). Undo = delete the route.
6. Up next: T-3354, T-2598, T-3743; T-3726 comments A/B (design in handoff-log 20:47). T-3742 nudges {after:v0.4.3}.
7. T-3729 Steam: on hold ("dont worry about steam now"). Don't raise it. T-3395: close after a day of clean scans.

## Her words
- 11-18: close easy high-priority wins and old tasks, small over design-heavy; "prio should take into account
  easiness as well as impact".
- "we need to be less willing to put stuff on the changelog". Player text says "manually", never "by hand".
- 11-18: a task kept open after its main work is retitled to what's left; Goal dropped when the title says it
  (CLAUDE.md Tasks). Text reused word for word from the game needs no ack (CLAUDE.md).
- v0.4.3 wishes (pad 11-18): fuller fare setting (T-3739, T-3740, T-3744), five schedule periods (T-3542); nudges
  "later, not in the same round".
- 11-19 playtest: "the shift-click works as intended!", "icons cute!", freight snap works; unify the two track
  selections (T-3746); hide flat marks no line uses (T-3745, landed).
- Share previews: "thatll make people feel a lot better about sharing"; free Worker, upgrade if we hit it
  regularly; "please be careful" with the Cloudflare account.
- No route: "showing anything, even if its not eprfect, is better than showing nothing at all" (T-3724).
- Mod support (T-3377) serves hand-coders and assistant users alike; never say publicly it's aimed at vibe-coders.
- Venezuela intercity demand: "fine tuning we can do later" (T-3706 P4).
- caelum418's speed PRs (T-3659): "look at the next time we look at this area of performance".
- When she plays games: keep 4 cores free (cap 12) and no uploads until she says she's done.
- Standing this week: small isolated fixes, count down, merge duplicates; she edits each update's changelog on the
  board before publish; lean testing, smoke per batch; long jobs in the background, answer her first; every
  publish, deploy and push needs her yes to that action, none overnight; advisor trial (log calls in handoff-log).

## Waiting on her (asks.py list has the open ones)
- None open. Discord paste test of a share link (she said she'd test after Cloudflare).
- She sends the T-3635 player reply herself (draft in notes/T-3635.md).

## Only a person knows
- Uncommitted on purpose: TODO.md (hers), notes/T-3397.md, jordan-save-investigation-2026-11-15.md,
  tools/ops/archive_bad_saves.py, changelog/roadmap.json, changelog/next.json, notes/asks-log.jsonl,
  notes/board-live.json; maps .claude/settings.json.
- Old trees shenzhen (dirty pipeline/osm_units.py, owner unknown) and tt2 (T-3346 stage 2, stale since 11-12, keep
  for the port): ask before removing. Empty locked folders .tw-trees/touchrule and .tw-trees/timekeys: remove once
  whatever holds them exits.
- Never delete a tree folder by hand: its app/node_modules junction deletes the main tree's (npm install from
  trainworld/app restores it).
