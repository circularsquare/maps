# Orchestrator handoff, trainworld

Orchestrator: session a3a433e6-4dfe-47ce-ad1a-12e81c692557, name maps-56, since 2026-11-18 16:08 UTC
Updated: 2026-11-18 16:20 UTC

Live state: `venv\Scripts\python.exe trainworld/tools/handoff.py status` (T-3667). This file is plan and her words.
Diary: notes/handoff-log.md. Her local time is UTC-5.

## Next
1. Adopted from maps-4f (reports arrive through the inbox): sideplat2 T-3672 (P1, her Caracas save
   Downloads/trainsworld-caracas-day-53.save; checker before landing, WASM rebuild), tripswithin T-3509 (demand
   change: checker before landing), corridor T-2136 (smoke + checker), regate T-930/T-3656 (land after dayrun's
   join), dayrun T-3671/T-3660/T-3517 (main tree, 6 cores).
2. Mine, started 16:20 UTC (Sonnet, trees): onewaytext T-3676/3677/3678 (one-way warning and question strings,
   3 untranslated keys; Vite 6150); checkfix T-3663/3679/3617 (three stale or failing checks; Vite 6151);
   Landed: onewaytext (c027 hidden per her), checkfix, commload (c028). Running (all Sonnet unless said, own trees):
   setsearch (Opus, T-3681, 6153), qwapp T-2809/T-3463 (6154), qwcheck T-1233/T-1635/T-3558 (6155), qwpre
   T-2002/T-2299/T-2001 (preflight, no data writes), lbfix T-3683/3684/3685 (6156), qwdiag T-3446/T-2662 (6157).
   regate (T-930/T-3656) reported done: land after dayrun's join with its DECISIONS line (notes/T-930.md), file nothing
   more (T-3682 filed for its follow-up).
8. Triage done (notes/triage-2026-11-18-{a,b}.md, todo-sweep-2026-11-18.md): 15 closed or folded, 21 reprioritised
   or unblocked, 7 tasks filed from the old TODO (T-3682-3688). Up next holds the rest of the small wins in order;
   start each as its blocker frees: T-3625 after setsearch (SettingsTab), T-3611 after qwapp (clock.worker),
   T-2880 after corridor, T-3495/T-2054/T-2901/T-2380 after dayrun's PACKS FREE and regate's landing, T-3354 after
   sideplat2. Her 10 open asks (asks.py list) carry the Needs-her items.
10. Her answers 16:50 UTC (all asks answered, in asks-log): started clockdup (Opus, T-3635 P2, she wants it fixed
   before the reply goes), schoolcar (Opus, T-3624, demand: checker before landing), newest (T-3652, commits in
   shonei-server, deploy hers), selway (T-3101). Landed qwcheck, lbfix (c029, c030; T-3684 waits on her). T-3653 time
   controls next, after clockdup lands (both in TopBar.tsx). Filed T-3689-3696 from her answers.
   Landed since: qwdiag, qwapp (c031, c032), newest (c033; server shonei-server 12d58a0 unpushed and undeployed: the
   tab stays hidden until that deploy, so hide c033 at a publish that goes out before it). Started autolease T-3611
   (saves: checker before landing).
   Checkers running: leasecheck (T-3611), selcheck (T-3101, sim), withincheck (T-3509 tripswithin, adopted). On
   landing T-3509 (her OK 17:25 UTC: in-zone long commutes count, share drop fine): rerun checks/t2145-real-share.mjs (~25 min) and paste its rows into REAL_SHARE (realShare.ts).
   Smoke's t1053-zone-ids fails in trees on `slw` removed from registry.json (dayrun's uncommitted game json): expect
   it until dayrun lands.
   Waiting for dayrun to end (pipeline files in use): regate (T-930/T-3656) and qwpre (T-2002/T-2299/T-2001, edits
   pipeline/world_plan.py GB_SPLIT_FAR, preflight flags advisory only; tasks already archived): land both then.
   Landed since: autolease, clockdup (c037), tripswithin (c038, realshare agent re-measuring REAL_SHARE), timectl
   (c039), selway (c040). T-3700 (one-way leg back stops, t3348 fails on master): look after sideplat2 lands.
   schoolcheck checking T-3624 (her questions: keep 2x, hover term "driven by a parent"). modelook (Opus, T-3693) makes the three-look sheet; I land my pick, she swaps by letter.
11. Windows popping on her screen (her 16:45): checks' children lacked windowsHide; play.mjs's background game has no
   console so each python/powershell child flashed. Fixed in cores.mjs/lib.mjs/smoke.mjs, copied into running trees;
   CLAUDE.md Shell rule added. If she still sees windows, look at dayrun's world_run children next.
   Empty locked folder .tw-trees/touchrule (not a worktree): remove once whatever holds it exits.
9. tasks.py lost T-1970 when ARCHIVE.md's replace was refused (Windows lock); recovered by hand, and tasks.py now
   writes ARCHIVE first and retries the replace.
3. dayrun finished 18:10 UTC (T-3671 archived, data/packs free, committed). Undo points data/work/pre-T3671/
   {pre,prejoin,retired} (~3 GB) go once smoke and 8800 are clean. regate and qwpre landed 18:12.
4. 18:20 UTC: smoke on master PASSED (29); 8800 = 07cc78bd (everything landed today); master pushed;
   pre-T3671 undo points deleted. node_modules was emptied
   at 18:03 by sideplat2's tree removal (a junction git deleted through); npm install restored it; agent-tree.mjs
   now unlinks every junction before removing a tree.
5. T-3395: close after a day of clean daily scans (deployed 11-18 13:30 UTC; she reads the scan's emails).
6. Then: T-1080 (waits for a world run hitting the commit limit), T-3160 phone memory once corridor (T-2136) lands
   (both memory; avoid measuring under each other), T-3620 (Caribbean, Bolivia, Paraguay) only when she says.
   T-2216 islands needs a design to her first (a point past which transit isn't taken is a new demand rule).
7. Play/wait tools (T-3666) landed: CLAUDE.md now says smoke is waited on through its log (an agent saw shell
   calls moved to the background at 180 s). Watch whether agents' smoke runs still behave.

## Server deploy waiting (her word)
- shonei-server unpushed and undeployed: 12d58a0 (T-3652 Newest tab), 662ed86 (T-3707 published builds + modded
  latch). After that deploy the Newest tab shows; v0.4.1's unstamped scores stay accepted 48 h after the server first
  reads builds.json, so the next game publish (which writes builds.json via deploy.py) should follow within 2 days.

## Her words
- 11-18 18:40 UTC: modded games off the leaderboards "soonish ... p2" (T-3707, modmark running: modded mark +
  scores only from published builds on anita.garden; server deploy hers). Points scale with each line's cost
  recovery capped at 1, "if fares cover 50% of expenses they should count as 50% points"; measure first (T-3657,
  profitpts running: stops before building if honest networks lose >20%). Mod support (T-3377) for hand-coders and assistant users alike; never
  say publicly it's aimed at vibe-coders (her 18:50).
- 11-18 18:25 UTC: buses and ferries should get trains' schedule controls ("we can leave out the marey diagram /
  timetable part for now ... not necessarily super high prio but would be nice to get out in next few updates"):
  T-3703, after busfix's T-3704 (same code). Discord reports filed: T-3704 (busfix running), T-3705 Firefox
  (firefox running), T-3706 Venezuela trips to Caracas (P3). Restore copies pre-T3455 deleted on her OK.
- 11-18 16:25 UTC: close easy high-priority wins and old tasks, small over design-heavy, to get the open count
  down; retriage clear miscategorisations; "prio should take into account easiness as well as impact". Triage
  agents triageA/triageB write notes/triage-2026-11-18-{a,b}.md (Close, Small wins, Needs her, Reprioritise); I apply
  closes and priorities in one batch, queue small wins in Up next and start them. She is out walking, on remote control.
- "lets do them all, can start in whatever order you think makes sense" (workflow, 11-18; all seven done).
- 11-18 ~15:20 UTC: mostly hands off today (headache): ask her nothing unless it truly needs her; short messages.
- Daytime builds on part of the cores when compute is free, thinking-heavy tasks beside them (11-18).
- Caribbean+: "we can worruy about caribbean+ later ... i will lyk" (11-18).
- Free-fare leaderboard cheese (T-3657): "needs more thought, lets put it off til later".
- caelum418's speed PRs (T-3659): "look at the next time we look at this area of performance".
- She sends dattera's reply herself (T-3647). Server log: left to the daily scan's emails.
- Standing this week: small isolated fixes, count down, merge duplicates; she edits each update's changelog on the
  board before publish (highlights over ~20 items); lean testing, smoke per batch; one phrase per wait (loads /
  shows up / calculating trips / memory); long jobs in the background, answer her first, no windows on her screen;
  every publish, deploy and push needs her yes to that action, none overnight; advisor trial (successors with
  `next_orchestrator.py trainworld --advisor fable`, log calls in handoff-log).

## Only a person knows
- Uncommitted on purpose: TODO.md (hers), notes/T-3397.md, jordan-save-investigation-2026-11-15.md,
  tools/ops/archive_bad_saves.py, changelog/roadmap.json; maps .claude/settings.json (her edits + T-3670 keys).
- Old trees shenzhen (dirty pipeline/osm_units.py, 11-16, owner unknown) and tt2 (stale, has a stray cores.json):
  ask before removing.
- Overnight undo points data/work/pre-T3455/{pre,prejoin} and pre-T3455my/prejoin: delete once she has played the
  new zones on 8800.
- Changelog has 26 unpublished items; a publish needs her edit on the board first.
