# Orchestrator handoff, trainworld

Orchestrator: session d15915ac-6dd5-4586-bd98-843cbc616824, name maps-ac [1ef926], since 2026-11-21 15:53 UTC
Updated: 2026-11-21 17:05 UTC

Live state: `venv\Scripts\python.exe trainworld/tools/handoff.py status`; diary notes/handoff-log.md. Her time UTC-5.

## Next
0. LIVE v0.4.5 (11-21 ~04:15 UTC). Cap 14 by day. Cores 0-1 are hers: start `reserve_cores.ps1 -Loop 30` at takeover
   (docs/orchestrator.md). Next update: translations nearer publish (T-3986's, T-3991's new keys too).
1. Batch 9 = master 7e6791a6 = 8800 (phonemem2, autostops, dropindent, splitb1, brtstations at save level 35,
   editfail, islandfix, viewonly, modeldown, keihan; osa+ngo shipped, old saves data/work/pre-T3935). batchchk9
   (Sonnet) checking it + smoke on the main tree; publish waits for its pass; no landings until it reports.
2. Landing queue (batch 10), all done: community T-3990/91, linecols T-3994, credits T-3998 (UI); brtblueprint
   T-3944 (level 36 / BLOCKS 40, brtchk PASS; dry-run finish, 5 behind); clockmem T-3939 (sim, task stays open,
   DECISIONS written); splitb2 T-3373 B2 (behind `?exactSum=1`, off); reach1k T-3995 (bus reach 1 km; a lone NYC
   bus +34%). Then Rust suites on master (fullrate and brtblueprint share lane code), batchchk10 (add: a
   blueprint-only BRT line on a weekend takes no lane room), smoke, snap8800.
3. fullrate T-3655/T-3941: fullchk PASS (slow edits only where rates really move; shipped real starts lose no cash,
   only the dev lon_real.save). Land after land10.ps1: `finish fullrate ... --task T-3941`; T-3655 stays open.
   Then searchcrash T-3997 (done, archived; UI only: place lists bounded to 48 zones within 200 km).
   land10 landed community, linecols, credits, brtblueprint, clockmem, splitb2; land11.ps1 (scratch) runs reach1k,
   fullrate, searchcrash, then Rust suites. Then batchchk10 + smoke + snap8800. oneway T-4000 (done, archived, edit
   path): onewaychk (Sonnet) on its tree first, then land. pausebp T-4002+T-4003 (Sonnet) running.
   stationkinds T-3999 done, archived (UI + addPort op): land after land11, into batch 10's check.
   searchcrash and reach1k landed. fullrate and stationkinds stopped on rebase conflicts: rebasefix (Sonnet)
   resolving; then finish both, Rust suites on master, batchchk10, smoke, snap8800.
   placeindex T-4004 (Opus, search finds far towns, general bound) running. oneway T-4000 landed (onewaychk PASS);
   land12.ps1 lands fullrate + stationkinds (rebasefix resolved both) then Rust suites. T-4006 done: shonei-server
   513353d (open saves 300/h, burst 30). savelevel T-4007 (Sonnet) teaches the server LBT1/LBQ1 (levels 35-36); one
   deploy (`.\deploy\deploy.ps1` from shonei-server) for both after her yes; must precede the next publish. T-4005 bookmarks filed P3.
4. Running: batchchk9, fullchk, searchcrash T-3997 (Opus, P1 player bug: search at 100x crashes / stops trains),
   stationkinds T-3999 (Opus, station tool train/bus/port + free-standing stations by two clicks; lands after
   T-3944), oneway T-4000 (Opus, player report), incomes T-3246 (Opus, design and cost only, goes to her).
5. No publish yet: she said 11-21 ~17:15 "no rush to push any updates ... nothing super urgent/gamebreaking".
   Start next: one Sonnet agent on T-4002 + T-4003 (community opens paused; hide-blueprints toggle) after
   land10.ps1; T-3373 B stage 3 (Opus) once splitb2 lands; T-3996 after T-3944; T-3948 once fullrate lands;
   T-3378 stage 4 after T-3944; T-3933/T-3932 after clockmem and splitb2; T-3937 needs a quiet machine.
   ownwaits T-3931 held for T-3231 (patch in notes/). T-3395: save_health.py after the next scan.
## Her words
- 11-21 ~17:00 UTC: closed T-3365 (track rework carries it); levels -5 to +5 come with the track rework, a Settings
  toggle for showing them (T-3691). Station totals by mode go with station timetables (T-3780). Free-standing
  stations back, two clicks for start and direction (T-3999). Exact sums OK ("<0.5% is never that much").
- 11-21 ~16:40 UTC: bus reach 1 km (T-3995); no borrowed-search bus stops (T-3992 closed, "a lie"); BRT built border
  inside the colour, dashes fixed on the ground (T-3996); credits folded (T-3998); station tool picks train/bus/port,
  bus stations snap harder, ports without ferry routes (T-3999); community view-only "looks good".
- 11-21 ~16:10 UTC: Report as a flag with a confirm (T-3991); Discord and leaderboards in Community (T-3990);
  itch.io $0-or-donate, $5 suggested (T-3988).
- 11-21 ~17:50 UTC: "text ack" for everything listed so far (viewonly, BRT stations, modeldown, brtblueprint,
  community, stationkinds).
- 11-21 ~15:55 UTC: track rework all calls made (notes/T-3444-design/answers-2026-11-21.md). 38 stages
  T-3948..T-3985 on tw/track.
- 11-21 ~14:05 UTC: keep 4-12 agents. Count every Auto stop ("if people are seeing bus stops the bus stops have to
  be real").
- 11-21 ~12:50 UTC: BRT stations for BRT lines only; T-3373 B go, the memory limit rises (players set it: T-3932; a
  load out of memory retries lower: T-3933). Never ask again whether big saves get more workers.
- 11-21 ~02:20 UTC: buses "feel pretty realistic"; bus speeds plain floor 25, BRT +10, fast roads 1.4x street.
- 11-19: "player complaints take precedent over other new features"; full playtest list before every publish.
- When she plays games: no uploads until she says she's done; she may ask for a lower core cap.
- Standing: lean testing, smoke per batch; long jobs in the background, answer her first; every publish, deploy and
  push to the website needs her yes in this tab, none overnight; advisor trial (log calls in handoff-log).

## Waiting on her (asks.py list has the open ones)
- She sends the T-3635 player reply and the T-3945 reply herself (draft in notes/T-3945.md).

## Only a person knows
- Uncommitted on purpose: TODO.md (hers), notes/T-3397.md, jordan-save-investigation-2026-11-15.md,
  tools/ops/archive_bad_saves.py, changelog/roadmap.json, changelog/next.json, notes/asks-log.jsonl,
  notes/board-live.json; maps .claude/settings.json. Not mine: notes/T-2380.md, notes/T-3382.md.
- A yes she gives in another session's tab doesn't pass the permission check here: ask again in this tab.
- Never delete a tree folder by hand (its app/node_modules junction deletes the main tree's).
- Commit messages through PowerShell: single quotes, or "$2M" becomes empty (49fa812e lost "US$2M").
