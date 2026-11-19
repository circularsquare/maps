# Orchestrator handoff, trainworld

Orchestrator: session aa626a2e-dc8d-4ec4-888a-66763905c165, name maps-6b, since 2026-11-19 03:22 UTC
Updated: 2026-11-19 03:22 UTC

Live state: `venv\Scripts\python.exe trainworld/tools/handoff.py status` (T-3667). This file is plan and her words.
Diary: notes/handoff-log.md. Her local time is UTC-5.

## Next
0. Live: v0.4.2 (data 2026-11-18b). Unpublished changelog c001-c010 (next update, v0.4.3). Server deployed 01:03
   UTC (shonei-server 962fcae, pushed). Cap 14 from 05:05 UTC 11-19, overnight ("i will head to bed soon. can raise cap to 14"); back to 12 when she
   plays. newam (Opus, T-3620 Caribbean, Bolivia, Paraguay) runs the world run overnight: it keeps GATES/JOIN
   times at the top of notes/T-3620.md; hold app/ landings during its gates, and use snapbuild-frozen.cmd for 8800
   while it owns data/packs. stnames (Opus, T-3750, renamed station shows old name by zoom) running. Player requests fetched 03:38 (41 new), reqsort (Sonnet) sorting them.
1. T-3542 stage 0 landed 807efd0b by maps-3f (8800 = 807efd0b, smoke passed, pushed; old .terrain files deleted).
2. Running, reports come to you through the inbox:
   - advfares (Opus) T-3739 advanced fare panel, tree advfares port 6108 (its Vite may have been stopped by a
     checker; it was told). Needs a checker (money, saves), then land. Then T-3740 zones and T-3744 per-stop fare
     (design first) per notes/T-3739.md "Her answers".
   - oneselbuild (Opus) T-3746 one track selection, tree onesel port 6114: controls only in the selection bar for
     1 or many pieces; shift-click fills between, big detour adds only the clicked piece (her answers in
     notes/T-3746.md). Needs a checker (rebuild charges money), then land.
1b. T-3620 DONE and committed 1c2ff68f: 17 countries (Caribbean, Bolivia, Paraguay) joined, freeze 20261119-020305;
   c013 changelog new text for her; Guyana, Suriname, French Guiana left of the Americas (her call); filed T-3754-56.
   T-3542 stage 1a LANDED c54415b0 (WASM rebuilt). stnames24 (Sonnet) merging master into tree stnames and moving
   it to level 24; then land stnames, 8800 rebuild, smoke, push, shonei-server `go test` with no override.
   Also started ~08:20 UTC: periods1b (Opus, T-3542 stage 1b UI, tree periods1b, port 6139; needs a checker),
   T-3755 committed a14a79b7 (Haiti none; Trinidad UIS 20,889 kept over the ministry's ~44k). T-3756 LANDED ff80b743.
2b. advfares BUILT (T-3739 archived by it; changelog advfares-1; 14 new strings in its report r-5579da31, to show
   her at landing). Checker farechk (Sonnet, ports 6124/6125) running; then land. Its pre-existing income bug filed
   T-3748 (P2). T-3746 LANDED 3d9e0789 (strings shown to her 04:27, no ack yet). T-3739 LANDED 2cc1f856
   (c011; its 14 strings shown to her 04:40, no ack yet). 8800 = a67c6d94, smoke on master passed (29), pushed.
   farefix (Opus, T-3748, tree farefix, port 6128) running. Her look at 04:55: "advanced fares looks ok" plus
   six fixes (T-3749): LANDED e388953c (not on 8800 yet; her check: "Fares by country" now heads the column
   row, review/advanced-fares.png; she: "sits in a good place"). Her next two (T-3751, P1): centred amounts, step
   km typed on the step row: LANDED 2dc34001 (review/advanced-fares.png updated). T-3748 LANDED 3d9d1277 (c012;
   checker passed; cross-border 1% residual filed T-3753 P3). Landed 05:36, two minutes into newam's gates (my
   slip; newam told to rerun any row that failed then). HOLD all app/ landings (periods1 too) until the JOIN line
   in notes/T-3620.md says gates are done (~08:30 UTC); then 8800 via snapbuild-frozen.cmd if the run owns packs.
   T-3750 BUILT in tree stnames (a rename renames the whole complex, buildings keep only typed names; stnames-1
   changelog; save LEVEL 22 -> 23); checker stnameschk PASSED (caveats for her: a building with a typed name keeps
   it, so z14 and z12.5 differ there; a station given its own typed name after a complex rename takes over the
   complex, by oldest-typed rule). SAVE LEVEL CLASH: periods1
   and stnames both take 23; the second to land moves to 24 (a fresh agent on that tree), and T-3752's server
   change covers both blocks. Tell her the T-3288 reversal (DECISIONS 2026-11-19). She is asleep from ~05:10 UTC 11-19. Rest of T-3739's text taken as
   acknowledged; T-3746's text still unanswered.
   Fares next: T-3748 income bug (P2, small); T-3740 zones (14-20 h: stage it); T-3744 per-stop fare (design first).
3. T-3720 LANDED eeb7b975 after two checker rounds (DECISIONS line uncommitted, goes with the next commit).
3c. Requests: 23 new groups; dismissed done g092 g093 g097 and praise g102; merged g083 g086 g087 g088 g090 g095
   into their tasks. Open for her: g096 (merge Buenos Aires and La Plata), g094 (no ask), g085, g098 and the rest.
3b. T-3743 landed 59643a0e. T-2598 done: RAIS is open; build filed as T-3747 (P3, not started: it ends in a
   Brazil rebuild run).
4. T-3542 stage 1a BUILT in tree periods1 (save LEVEL 23, LPD1; matches master to the bit at 3 rows; costs under
   estimates); checker periodschk PASSED (master loading a level-23 save drops midday/afternoon rows, no crash).
   Land FIRST after the gates (with WASM rebuild); stnames then moves to level 24. T-3752 done:
   shonei-server f5309bf (levels 23 and 24, LPD1 walker, ERA3; not pushed, not deployed). After both trees land,
   `go test ./...` there with no TW_SAVE_RS; then ask her for the deploy before the v0.4.3 publish (not overnight). Server change filed T-3752 (start it
   after landing; deploy with or before the publish, her yes). Then 1b (window controls, line panel 4/5 rows,
   readouts incl. running cost and trains/h), then stage 2 (hours). periods1 is past 500k: 1b is a fresh agent. T-3346 stage 2 (tree tt2) ports after T-3542 (DECISIONS 11-18).
5. T-3738 share previews LIVE (worker trainsworld-sharepreview 772a7d85 on anita.garden/trainsworld/* + 13
   no-worker routes; free plan). She couldn't find fail open; skipped (35-51k loads/day vs 100k). Watch daily Workers
   requests for a week (~2.5k in its first 2 h, normal); then close T-3738 (c009). Undo = delete the route.
6. Up next: T-3354 (needs her exact drag; not reproduced); T-3726 comments A/B (design in handoff-log 20:47). T-3742 nudges {after:v0.4.3}.
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
