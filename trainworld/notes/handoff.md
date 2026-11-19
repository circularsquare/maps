# Orchestrator handoff, trainworld

Orchestrator: session aa626a2e-dc8d-4ec4-888a-66763905c165, name maps-6b, since 2026-11-19 03:22 UTC
Updated: 2026-11-19 09:25 UTC

Live state: `venv\Scripts\python.exe trainworld/tools/handoff.py status` (T-3667). This file is plan and her words.
Diary: notes/handoff-log.md. Her local time is UTC-5.

## Next
0. Live: v0.4.2 (data 2026-11-18b). Unpublished changelog c001-c015 (v0.4.3). Cap 14 overnight ("i will head to
   bed soon. can raise cap to 14"); back to 12 when she plays. She slept from ~05:10 UTC 11-19.
1. Landed overnight 11-19 (all on master, pushed): T-3720 autosave latch, T-3746 one track selection, T-3739
   advanced fares + T-3749/T-3751 her polish, T-3748 market fare scale (c012), T-3620 Caribbean/Bolivia/Paraguay
   (17 countries, freeze 20261119-020305, c013), T-3755, T-3756, T-3542 stages 1a and 1b (save level 23, c015),
   T-3750 station names (level 24, c014; reverses part of T-3288, DECISIONS 11-19). 8800 = e8d1dc60, smoke
   passed 29, pushed (~09:50 UTC).
2. shonei-server f5309bf (save levels 23-24) committed, not pushed or deployed; `go test` passes against master.
   Deploy needs her yes (ask a-2026-11-19-5), with or before the v0.4.3 publish.
3. Open asks a-2026-11-19-1..5: text acks (T-3746, T-3542 1b, c013), Guyana/Suriname/French Guiana, the deploy.
4. T-3542 DONE: stage 2 (period hours, level 25) LANDED 875b0070 after periods2chk passed; c015 reworded.
   shonei-server 1f29b5a (levels 23-25) committed, not pushed or deployed (ask a-2026-11-19-6); go test passes
   against master. 8800 = 38075a1a (everything above), smoke passed, pushed (~11:50 UTC).
   T-3740 Z1 (sim charging + save, level 26 LFZ1, no UI) BUILT in tree zonefares; rule: start and end zones only
   (DECISIONS line uncommitted in main; tell her, she can reverse). Checker zonechk (6151/6152) running, srvlevel
   adding level 26. Then land Z1, then Z2 (paint tool, tint, panels, strings; 5-7 h) as a fresh Opus agent from
   notes/T-3740.md, then Z3 (rings, new stations, bus stops; 3-4 h). T-3753 LANDED 20147ede (not on 8800 yet;
   rebuild with periods2's landing). Then T-3346 stage 2 ports (tree
   tt2). Fares: T-3740 zones (14-20 h: stage it first), T-3744 per-stop fare (design for her first), T-3753 (P3).
   T-3754 La Gonave, T-3747 Brazil RAIS (ends in a rebuild run: overnight).
5. Requests sorted 11-19 (23 new groups; done ones dismissed, duplicates merged). Open for her: g096 (merge
   Buenos Aires and La Plata) and the rest on the board.
6. T-3738 share previews live; watch Workers requests daily for a week, then close (c009). T-3729 Steam on hold
   ("dont worry about steam now"). T-3395: close after a day of clean scans. T-3354 needs her exact drag.
7. Slip 11-19 05:36: landed app/ two minutes into a world run's gates; check the run's notes for GATES/JOIN before
   any app/ landing while a run is on.

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
- Discord paste test of a share link (she said she'd test after Cloudflare).
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
