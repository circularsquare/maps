# Orchestrator diary, trainworld

Older entries moved out of handoff.md on 2026-11-16 (T-3451 recommendation 4). New entries go at the top.

## 2026-11-17, maps-58 (16:41-18:14 UTC)
- Took over from maps-60 (handed its agents over un-landed). Landed 14 pieces of work (list in handoff Agents), 8800 rebuilt once (dc1be574), master pushed. Built the finish command (T-3554), the checks/lib playtest harness (T-3556 stage 1), short ask titles on the board.
- Her calls: report length 15 lines and only new asks in final messages ("catch me up" reads asks.json); 3/4/5 periods game-wide by default (T-3542); trains-drawn-apart postponed past v0.4.0 (keep train shape); passing stations get both platforms back; one-way pair stops one-way each (T-3565); trams path like rail; shoreline swap + OSM water for TIGER packs on a later night; currency in save P4; harness stages 2-3 P4.
- Lesson: agents left with background jobs re-sent reports up to five times; briefs now say stop them, and TaskStop after a report. My own long foreground waits made her wait 25 minutes: background them.
- Advisor: not called this session.

## 2026-11-17, maps-60 (14:40-16:40 UTC)
- Took over from maps-b8, landed its piecelimit, stationedit, busboard, shore, settle (squash) and my trackrebuild (+ levels/tram follow-up), helplist (+4 entries), reclaim, snapicon, tracklook, tramlag (T-3535), complexpick; 8800 rebuilt five times, last a0e4157b; master pushed.
- Her calls: public repo becomes the real one after v0.4.0 (T-3523); single-track cap fix 1 (T-3526); shore 12 m OK; c086 "350 MB"; BA-Montevideo water fix later, low priority; tram slowness (T-3531) was machine load.
- weekend checker FAILED on headway lines; weekend2 fixing. Advisor: not called this session.

## 2026-11-17, maps-b8 (11:13-14:40 UTC)
- Took over from maps-44; landed nightwrap's runs (948e61f3), T-3505, T-2627/T-2777, T-3480 (by hand, Renkon co-author), T-3520, T-3510, terrain default; posted PR 17-20 replies (17, 20 closed).
- Replies to post at the next open-source sync (her OK given):
  - PR 1: "Thanks for this, Renkon! It's in the game now as a setting, Buses obey one-way streets, off by default so existing bus routes don't change. We kept your routing as it was, made the arrows lighter and start them at zoom 15. You're credited as co-author on the commit. It goes out with the next update, and we'll close this with a link once the public repo syncs."
  - Issue 8: "Thanks for reporting this. The next update adds a setting, Buses obey one-way streets, that keeps bus routes going the way one-way streets and roundabouts run. It's off by default; turn it on in Settings > Show. We'll close this once it's out."
- The trainsfan agent kept re-notifying after finishing; stopped with TaskStop.

## 2026-11-16, maps-7e and maps-d7 (copied from handoff.md as it stood at 18:05 UTC)


Orchestrator: session f7c8626c-6b89-4b09-983c-732243e787e9, name maps-d7 [e8137b], since 2026-11-16 15:36 UTC
Previous: session 76388cba-0327-41aa-a4e4-4e7d7376bf41, name maps-7e [2cdf3c], handed over 2026-11-16 15:36 UTC, finishing in-flight work
Updated: 2026-11-16 15:38 UTC (she is starting a new orchestrator now)

## maps-d7 (new orchestrator, 15:38 UTC)
- Board inbox watch armed by maps-d7 (Monitor, re-arm every 30 min). Pending was empty at 15:36.
- 15:48 maps-7e stopped running commands (its permission rules were stale, prompting Anita) and handed all landing
  to maps-d7. Landed by it: glyph, b5, pick2 9f354228, sc 3a8603c3. Landed by maps-d7: doc 78fb0d5b, c2 77e9d121,
  pushed c7ca1a82. c3, e1, feeders, shenzhen still run and report to maps-7e, which forwards them; maps-d7 lands.
  maps-d7 agents: lintfix aaef44c2544b34f27 (lint tree mid-rebase: resolve, then I run rebase --continue and land),
  tidy adfe01689f8db5dd4 (archive pick2/sc/doc/c2/c1 tasks; find where c1's market_check.py went).
  c2 offered picker plan2 batch D (demand workers): held, not small fixes.
- 16:08 all of maps-7e's trees landed except feeders and shenzhen (c3 9d3cd02d, lint b22b9a13, c1 a1d5b113, e1
  b4fd4d98); tidy archived 20 (353 open). Smoke PASS on master b4fd4d98; master ab041040 pushed; 8800 = ab041040
  (frozen 20261113-013559), swapped 16:10. .tw-trees/c1 remove failed (something holds files): retry later.
- Running (maps-d7): bug-a ac57e25e3f3cf585b (T-3429 T-3353 T-3354 T-1681), bug-b adc9785ca19e09c3a (T-3431 T-1072
  T-3410 T-1791), bug-c a02274cc2e5549664 (T-3150 T-702 T-691 T-2148), each in .tw-trees/<name>, ports 6111/6121/6131.
  Before landing: restore the four app/src/game json files in the tree (they copied the main tree's).
- serve.py: the board rebuild runs with CREATE_NO_WINDOW (it was the python window she saw); serve restarted 15:54.
  Committing serve.py was refused by the permission check twice: uncommitted, ask her. Watcher also caught
  a scheduled task outside trainworld: graffiti's scripts/backup_paint.bat ("npm run backup-paint", Task Scheduler)
  opens a cmd window at 11:15 local; tell her, it's not ours to change.
- schem a4c88ef4ea564313f (Opus, .tw-trees/schem, port 6141): T-3448 player's NYC schematic in islands (filed
  16:21 from a report she passed via maps-4a). bug-c: nothing fixed (T-691 resumed for station refusal wording).
- T-3449 landed 16:33 (board.js cec4dba4; tree removed). serve.py holds its /twboard/_paste endpoint and the window fix,
  live (serve restarted 16:29) but UNCOMMITTED: the permission check refused committing serve.py three times; ask her.
  T-2724 closed on her word; T-3358 rivers back to P2 ready (she called rivers and squished cities the pressing issues).
- 16:41: bug-b landed T-1072 (sim; wasm rebuilt); bug-c landed T-691; sweep closed 4; master 717476c4 pushed. 348 open.
  Running: bug-a (T-3429 refusal message), bug-d a9e69f3784d4ceef7 (T-3252, T-1863; tree bug-d, port 6151),
  bug-e a6ade6052ca9999d7 (T-3042, T-1875, T-3101; tree bug-e, port 6161), schem.
- 17:03: bug-a landed T-3429 (sim); shenzhen committed 404768f5 and T-3428 archived; smoke PASS; 8800 = 404768f5 on
  LIVE data (no freeze); pushed. Shenzhen left six python scan processes (its scratchpad shenzhen/scan_all.py, ~3
  cores); its stop was refused by the permission check, so not stopped by me either: ask Anita. Its proposal for her
  call: a coastal-only gap scan (~2 h, 3 cores) then rebuilds (Lingang, Binhai, Macau, HK, Dubai, Singapore, Incheon,
  Osaka Bay): overnight material.
- 17:30: bug-d landed T-3252 + T-1863 (sim); schem landed the mask download fix dc184b11, T-3448 waits on her
  (A/B/C; review/nyc-schematic-far.png). Running: bug-e, bug-f a7d62b4a99930fbfd (T-2698, T-2757; tree bug-f, port 6171).
- 17:56 (Anita back 17:31; her answers): feeders LANDED c9db1359 (T-3426 archived; follow-up T-3454 feedsample
  aca180a7fbc3e6ae7, Opus, tree t3454-base reverts the fix for "before"). T-3448 decided: one schematic per region,
  regions joined by rail draw as one, nearest first then all: agent regions ac47cdc31cb8469fc (Opus, tree regions,
  6181); option B queued as T-3453. T-3410 folded into T-3079 (waypoints). T-1791 yes, hover only: agent hover
  a8e322d9d6dcef870 (tree hover, 6191). Shenzhen scans stopped on her OK. serve.py committed 15a54121 on her OK.
  Coastal gap scan queued as T-3455 for an overnight run on her word. Also from her pad: T-3450 Shiga (agent names
  acde25cc42203d034, Opus), T-3451 workflow review DONE (Fable, notes/T-3451.md; her calls pending), T-3452 code
  tour DONE (notes/code-tour.md, review/code-tour.png). bug-e landed T-3042 (+T-3101 part), bug-f landed T-2698.
  8800 = 943dfbb3.
- sweep a30f41793d9dfc434 (Sonnet, no tree): closing already-done/untrue/duplicate tasks below T-1500 (no waits, no taking).
- shenzhen (16:36, interim): shz rebuilt and joined (45 packs gated), no longer writing data/packs (8800 may use live
  data again); join's market step timed out (1801 s) so no new freeze; market files checked fine. Its uncommitted files
  (osm_units.py + test, registry.json, market-rows.json, outlines/realms) are to commit after its full report; note the
  main tree's zone-names/realms/registry also hold T-3399's plan edits: split by key when committing.
- T-3426 feeders: table relayed 16:10 (fix brings most corridors closer to real, NY-Philadelphia overshoots on
  commute legs); maps-7e put "land as is?" to Anita; land on her yes, then file the two follow-ups in notes/T-3426.md.
- Anita 16:00: away working on other stuff, no frequent reports until she says she's back.
- Handed to maps-d7: feeders T-3426 (land only after Anita sees the recalibration table; maps-7e forwards it),
  shenzhen T-3428 (its report forwarded; 8800 rebuilds frozen at 20261113-013559 until then), T-3399 (Anita 15:38 UTC, her 10:30am:
  "i will let you know when its time for that, otherwise dont worry about it": wait for her word; commit the plan
  edits when it starts). shonei-server pushed and T-2396 closed by maps-7e on
  her word.
- Stale below, checked 15:36: perdir landed as e79a5cd6 (no tree now); trees hotfix, hotfix031, stop-freeze-034 are
  gone (branches hotfix-stop-freeze-034 and tw/hotfix-10d remain); "State" 8800/master lines are older than "In flight"
  (8800 = b6c40df5, origin = b6c40df5, master 14 ahead).

Read trainworld/CLAUDE.md, docs/orchestrator.md and TASKS.md first. Earlier detail: this file's git history.

## State
- **Live = v0.3.5**, published by the saves session on Anita's explicit go, 2026-11-15: Safari gzip fix, maps
  bfe58c54, website c9b74a8, public source 6b8ef91. Pages run 43772084654 succeeded; live entry, clock worker and
  gzip worker SHA-256 match the tested build. Data stays 2026-11-13, freeze 20261113-013559. Release tree
  `.tw-trees/safari-save-035`, built package `artifacts/trainworld/publish/v0.3.5/dist`. 8800 was not swapped.
- **Previous live v0.3.4**: Anita published it herself 11-15 (with GPT) from branch `hotfix-stop-freeze-034` (66ef6ca1, tree
  .tw-trees/stop-freeze-034, built on the v0.3.3 release b2897ae2): fixes freezes when editing stops on large saves
  (T-3385; hidden stop rows fed negative measurements). Website 3205539 + her later "intercity" 74f6a57. v0.3.3 went out
  11-13 ~14:00Z (data 2026-11-13; terrain v2 on R2 at terrain-world/v2).
- **Master bfe58c54** (Safari save fix and v0.3.5 changelog; not pushed to maps). Board commits through 70cfd8af precede it; before them 6feec4b4, pushed (11-15): her v0.3.4 fix 0bb9fcfa; her freeze diagnostics 43b3acbe (off unless
  `?diagnostics=1`; T-3385 archived); v0.3.4 changelog translated 6feec4b4 (live v0.3.4 shows those two lines in
  English until the next publish). Typecheck and i18n clean.
- Uncommitted on purpose: perdir's DECISIONS line (2026-11-13, T-3348 stage 1): commit it when per-direction stops land.
- **8800** = 6feec4b4 from freeze 20261113-013559, smoke PASS (12,581,681 rail trips a day).
- Her release tree .tw-trees/stop-freeze-034 (branch hotfix-stop-freeze-034) is the previous release; keep it like hotfix031.
- **Server** shonei-server 8391326 deployed 11-13 16:28Z, pushed (twstats 15-minute playtime ring).
- **Open source**: github.com/circularsquare/trainsworld, last sync 6b8ef91 (maps bfe58c54; sync commits credit
  contributors). R2 data answers CORS `*` (forks can read it). Each publish: export.py, then `git push` there.
- **PRs**: #2 peanutmow: part 1 landed 19a7f1e2 (co-author), reply posted, left open for a settle revision (T-3387).
  #1 Renkon (bus one-way streets: she leans no, maybe an advanced toggle) and #3 CinnamonVII (run the built stretch
  of a partly built line: low priority) unanswered; she'll look Monday. Reply practice: docs/orchestrator.md.
- No agents running, no core claims. Data: nothing running; newest freeze 20261113-013559.

## In flight (maps-7e, updated 2026-11-16 15:32 UTC)
- Her plan: today "get the number of open tasks down, by handling straight forward tasks" (413 -> 377 by 15:30);
  tonight T-3399 Central America + Venezuela, then gla/lis/pot rebuilds (Lisbon/Glasgow gaps); tomorrow small tasks or
  the fare-setting system. Complex track work waits for T-3444 (her track rework notes; "later"). Quad work paused.
- Running agents (each in .tw-trees/<name>, landed with agent-tree.mjs land; i18n JSON conflicts merge by key):
  b5 a2cfa4b348a666bad (ferry/bus fixes T-3400, T-3419, T-3370), lint ab8cc3eadbb592dd4 (T-3443), c1 a7d543ca2406b54e8
  (checks/calib), c2 a88de27a9451b413c (loading; then start batch D from C:\Users\anita\AppData\Local\Temp\claude\dce1fefcc1d43aef\76388cba-0327-41aa-a4e4-4e7d7376bf41\scratchpad\picker\plan2.md), c3 a3a3eb40021a0bccf (label/frame costs),
  e1 a90d0ff7fa52b0379 (edit cost), sc ac2e02d431120b572 (schematic; looks-changes stop at screenshots for her), doc ab9e2a1110a3b25f6,
  T-3442 glyph fix landed 15:36. pick2 a4b7e53eb725fe978 (T-3406, T-3407, her picks), feeders af151363ff713e9bf (T-3426: calibrate against real corridors
  before landing; fix in .tw-trees/feeders), shenzhen a05e93ee1c13bd90f (T-3428 shz rebuild + Pearl River join writing
  data/packs: 8800 rebuilds use snap8800.py <hash> 20261113-013559 until it reports).
- Held: picker plan2 batch R (realnet rebuilds), B8 world_run scripts (after T-3399), B10 track panels (after T-3444).
- T-3399 ready: powershell -File trainworld\tools\country\start_hidden.ps1 C:\Users\anita\AppData\Local\Temp\claude\dce1fefcc1d43aef\76388cba-0327-41aa-a4e4-4e7d7376bf41\scratchpad\countries\tonight.cmd (waits for locks,
  ~10 cores, ~4 h + 1.5 h stage 2; NO-STAGE2 in the kit folder skips gla/lis/pot); log artifacts/trainworld/t3399-tonight.log.
  She OK'd it 15:33 UTC: start in her evening, not now ("it is only 10:30am"). Countries' plan edits are uncommitted in the main tree (world_plan.py/.json, zone-names.json,
  realms.json, data-register.md, notes/T-3399.md): commit them when the run starts.
- shonei-server deployed 15:34 UTC on her yes (T-3402 paging + T-3440 size caps; service active). NOT pushed: a push
  needs her word (asked). noritetsu/dist account.js + privacy.html (T-993 text, her "Your cloud saved trainsworld games...")
  uncommitted, hers.
- 8800 = b6c40df5 (frozen data). master pushed b6c40df5; later commits unpushed (push regularly).
- Permission rules she approved are in maps/.claude/settings.json; routine commands: snap8800.py, restart_serve.py,
  git -C <maps> commit/add/push, agent-tree.mjs. Avoid git checkout/reset/branch -D, ad-hoc kills, PowerShell steps.
- Waiting on her: T-3426 recalibration table; T-2396 (still unclear to her: explain with a screenshot, or close); shonei-server push;
  T-3442 if the glyph agent needs more; T-3435 fast ferry (later); per-direction refusal wording (stage 4).

## Earlier (maps-b8)
- T-3390 done by agent lines (Opus, id a75a19621b6669315): 58 lines in LINES.md, every task tagged, 11 done tasks
  archived; 35 close candidates wait on her in notes/T-3390.md "For Anita". Unfiled TODO.md items listed there too.
- T-3391 and T-3392 (board stages 1-2) built; serve.py restarted 16:08Z with the inbox endpoint (root serve.py is
  modified, uncommitted, outside trainworld: her commit). **Board inbox watch**: Monitor on
  `venv/Scripts/python.exe -u trainworld/board/watch.py` (30 min max, re-arm on expiry); on "batch ready",
  `watch.py pending`, apply (notes/T-3391.md "Stage 2" table), `watch.py handled <ids> --by <name> --note "..."`.
  A new orchestrator arms it at takeover. T-3393 (player requests, paste box) built too; whole board committed
  7c47580f (not pushed). Board data files with player text are gitignored (trainworld/.gitignore). No agents running.
  Agent board (id a75091c86d0c851a1, 365k tokens) knows the board code: resume it for board follow-ups.
  Format they share: notes/T-3389.md "Format (decided)".
  notes/board-live.json is the orchestrator's: update it when agents start or finish.
- Anita 11-15: "its sunday so im not gonna use usage for anything else today. lets start it up, and when we run out,
  we run out, and ill resume tomorrow morning right away at the same point." Board only today, no player bugs.
  Crash: "crash is fixed, i think. people are saying its nice :)".

- Agent board resumed 17:55Z for task-id links everywhere and an inline notes view (registers `twnotes` on 8800).
- Tomorrow (her words, "thats something we should do tmr"): T-3394, every task as title / need from Anita /
  context / goal / other info. Proposed shape: indented sub-bullets under each task line in TASKS.md (grep -A finds
  them); likely one Sonnet agent per area reading each task's notes. Waits on her pick.

- Saves session 11-15: the Safari client fix and checks are committed as bfe58c54 and published as v0.3.5
  on Anita's explicit request. Source smoke and exact release smoke in light/dark all passed (28 checks each),
  plus the compiled file-picker/save/reopen check. T-3395 automatic detection/pinned recovery remains open;
  tools/ops/archive_bad_saves.py and private incident notes remain uncommitted. Recovery archives exist on
  the server and, with Anita's explicit destination approval, in ignored artifacts/trainworld/save-recovery/2026-11-15.
  Jordan's two damaged games have no cloud predecessors; only original-device local backups might survive.
  See notes/jordan-save-investigation-2026-11-15.md. Scratch pad is T-3396 (renumbered; T-3395 was taken).

- 11-16 12:40Z resumed after the usage limit. v0.3.5 (her Safari save fix, bfe58c54) is live. Running: pad
  (Sonnet, id a0f0e9e24e5e22a04) finishing T-3396 then board fixes (wider panel, scroll kept on Back, no scroll
  jumps, worktrees in the strip); lines (Opus, id a75a19621b6669315) applying her close answers (in my message to it
  and her chat 11-16) and filing TODO.md short-term items as T-3398..T-3440; shape (Opus, id af72987d247ffa2b4)
  T-3394 pilot of 14 tasks + tools/restructure_tasks.py, no TASKS.md edits. After the pilot: show her, then fan out
  by area (block files, one apply). Board must then render the fields.
- 11-16 ~13:40Z: lines done (31 archived, T-3398..T-3420 filed), pad done (T-3396, 06a9b440, serve.py restarted).
  Shape pilot done: notes/T-3394.md, tools/restructure_tasks.py (uncommitted). Rollout estimate ~25 agent-hours,
  8 Sonnet agents ~3 h wall, by area (demand split in 2-3, performance in 2): waits on her OK of shape and cost.
  pad agent rendering the fields on the board meanwhile. After the rollout: CLAUDE.md Tasks rule, TASKS.md and
  Someday headers, docs/orchestrator.md filing; `--retag T-3345` (stale waits).

- **11-16 ~14:30Z PAUSED by Anita** ("lets pause work, i'd liek to try to turn on code mode"). No agents running.
  Her plan for this week when work resumes (her words): "lets limit work to straight-forward bug investigation (can
  give up if seems hard) and kanban. also track visibility since it seems easy enoguh." So, to launch on resume:
  1. T-3394 rollout, all tasks ("lets do all the new task layouts ... keep the mini sections very short"). Brief:
     notes/T-3394.md "Rollout brief". Nine Sonnet agents were launched then stopped at once by the pause (nothing
     written); relaunch them, each with a fresh snapshot of TASKS.md and these line sets: track (all track lines);
     schedule + community lines (skip T-3395, T-3397); schematic + interface lines; ops lines (skip T-3394); demand A
     region-redraws, borders, islands, new-countries, airports; demand B jobs-places, census-totals, attractions,
     wealth-split, world-rebuild; demand C demand-ideas, route-choice, ridership-checks, car-free; perf A
     solve-speed, solve-consistency, frame-rate; perf B edit-speed, loading, memory. Block files in
     scratchpad\rollout\<name>.md; then one apply with --keep-old notes/tasks-before-T-3394.md; then the CLAUDE.md
     Tasks rule, TASKS.md and Someday headers, orchestrator.md filing rule; retag T-3345.
  2. T-3424 changelog on the board + table column resize (resume or brief a board agent: notes/T-3391.md).
  3. Bug investigation, give up if hard, in worktrees: T-3421 ramp speed, T-3422 blueprint flyover, T-3411, T-3412,
     T-3379, T-3380, T-3381, T-3383, T-3386, T-3403.
  4. T-3420 line and level visibility to her design (in the task line).
  T-3264 approved by her ("yes this soudns good") but not to start this week. T-3213 and T-688 closed 11-16.

## Parked work in trees
- .tw-trees/perdir (tw/perdir from f8119117): T-3348 stage 1, per-direction stops, sim half built and tested (her
  split-legs case runs both ways). Missing: demand per run (2-3 h, plan in that tree's notes/T-3348.md "Stage 1 as
  built"), the stop marker at the return station, a named refusal on removing it. Her: hold if it costs much more.
- .tw-trees/tt2 (T-3346 stage 2, timetables; timetables are hidden behind ?timetables since v0.3.3).
- .tw-trees/hotfix, hotfix031: old release trees, removable.

## Next (her call; she's planning a reorganisation first)
1. T-3389, her board (TODO.md "first thing"): design proposal in notes/T-3389.md sent to her 11-15 15:30Z, waiting on
   her picks (own board vs beads, area list). Then per-direction stops' demand half (she wants to continue it).
   TODO.md's "continue trying to find fix for crash" looks like the T-3385 freeze v0.3.4 fixed; asked her.
2. P2 bugs from players: T-3379 single-to-double kink, T-3380 delete ghost on single, T-3381 complex name in the
   corner notice, T-3383 "busy single track" wording (+ the 24m00s check), T-3386 inferred one-way lost at a station.
3. T-3376 industrial jobs (Jurong 128k vs ~27k real; "semi high prio"), T-3388 London edits take 4 s.
4. Later/design: T-3378 BRT and bus balance, T-3377 mods (levels 2-4), T-3382 where people play, T-3384 waypoints,
   T-3375 zero-resident units beside a metro, T-3358 rivers (paused by her). Next free id T-3397 (check TASKS.md first: another session files ids too).

## Anita's standing asks
- This week (Anita 11-16 15:45 UTC): small isolated fixes, getting the open-task count down, merging duplicates.
  Quad work stopped (its tasks now wait on T-3444). Region redraws and other pack rebuilds are compute-heavy: batch
  them into an overnight run, not daytime agents. T-3399 starts only on her word ("i will let you know when its time").
- No visible windows on her screen: start background processes hidden (start_hidden.ps1 or CREATE_NO_WINDOW); she
  saw python consoles popping 11-16 (maps-d7 tracing the source).
- Every publish and server deploy needs her "yes" to that exact action; none overnight.
- A game change a player asks for (numbers, what can be built) waits for her word (CLAUDE.md).
- Replies on contributor PRs: written for a person, offers not requests, she sees each first.
- Cores: the ledger's cap. Usage has been tight: be economical.

## Advisor calls (trial)
- 2026-11-17 10:00 UTC, maps-44: whether to land T-3512 after setchk's partial pass (stop-on-whole never fired, no unpinned level change, one hang with the old-path flags). Changed things: T-3512 held for a second round; T-3069 given a checker it would otherwise have skipped (hand-drawn tram turn rule, monorail/funicular); a like-for-like memory figure asked for T-3503; master smoke's t1053 no longer counted as noise.
- 2026-11-17 11:20 UTC, maps-b8: takeover plan (which tasks to start beside nightwrap and settle2). Changed things: briefs re-read from exact task blocks (my first grep had mixed in the next tasks' fields); bubfold told to keep off settle2's files and not rebuild packs; Rust agents held to CARGO_BUILD_JOBS=2 while settle2 times solves; PRs 17-21 surveyed against PR 12's earlier decisions instead of from scratch; T-3512's taking mark fixed to settle2.
- 2026-11-17 ~14:20 UTC, maps-b8: how to fit weekend train counts in (PR 13 narrow cut vs Detailed mode). Changed things: save levels split 20/21 with one owner of the server check file; PR 13 port checked first; her 11-12 shape over the PR's; changelog tense fixed and the CLAUDE.md rule changed; a master smoke and a terrain memory check before publish.
- 2026-11-17 ~21:50 UTC, maps-45: how to help the one-way pair work (T-3565) converge, with her T-3595 repro on the same Caracas line. Changed things: extendfix narrowed to her repro and told to test master against the onewaystops tree before fixing; onewaystops given one consolidated brief (position-based model, normal editing only, no save change, a Rust click-order permutation test) and a context check; save check against shonei-server's test before the deploy ask; translators' merges to be committed before landing farebase.
- 2026-11-17 ~21:35 UTC, maps-1c: T-3565 fresh start (continue the old tree or not) and takeover plan. Changed things: fresh tree off master with the old tree as a read-only quarry; partner rule kept pending her answer (asked); click-order test equal up to direction, stated in the brief; T-3565 agent kept off T-3595's cause; reviewers' merges vs the farebase landing settled with maps-45 (landing is mine); the 4 stale keys translators kept unchanged added to every review; server save check not called clear on TestTwSaveCheckFollowsTheGame alone.
