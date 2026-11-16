# Orchestrator handoff, trainworld

Orchestrator: session 76388cba-0327-41aa-a4e4-4e7d7376bf41, name maps-7e [2cdf3c], since 2026-11-16 13:13 UTC
(Took over from maps-b8, session 7186e3f8, already gone at 13:13 UTC, paused with nothing in flight, so no wait.)
Updated: 2026-11-16 13:13 UTC (resuming her plan for the week: T-3394 rollout, T-3424 board changelog, easy bugs, T-3420 visibility)

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

## In flight (maps-7e, from 2026-11-16 13:15 UTC)
- Resumed her plan for the week ("lets limit work to straight-forward bug investigation (can give up if seems hard)
  and kanban. also track visibility"). All marked taking in TASKS.md; agents told not to edit TASKS/ARCHIVE this
  round (the restructure is pending), so I archive their done tasks after the apply.
- T-3394 rollout: nine Sonnet agents (r-track ae2d709cd87d8bec6, r-sched af20713845444111c, r-schem add08d4ba3de24723,
  r-ops a72d91039ed58903c, r-demA a32b133e0373b684b, r-demB aa1d58c0dc4945475, r-demC aa3bfcd2e27d2a0ec, r-perfA a69d34ee2a470b366,
  r-perfB aedb1e81ed16780da), line sets as in the paused plan below. Snapshot and block files in my scratchpad
  rollout\ (C:\Users\anita\AppData\Local\Temp\claude\dce1fefcc1d43aef\76388cba-0327-41aa-a4e4-4e7d7376bf41\scratchpad\rollout).
  When all nine report: one apply with --keep-old notes/tasks-before-T-3394.md; refused blocks back to their agent;
  then the CLAUDE.md Tasks rule, TASKS.md and Someday headers, orchestrator.md filing; retags.
- bugs1 (Sonnet, a32d9d2d77d3e0ce0), tree .tw-trees/single, port 6111: T-3379, T-3380, T-3411, T-3386, T-3383.
- bugs2 (Sonnet, ace743905b76bc49b), tree .tw-trees/fixes, port 6112: T-3381, T-3422, T-3421, T-3403, T-3412.
- visibility (Opus, a34f8356acf49ce26), main tree app/, port 6113: T-3420 to her design.
- changelog (Sonnet, acd06534dca27974c), trainworld/board, port 6114: T-3424 changelog on the board + column resize.
- Board inbox Monitor armed 13:14 UTC (30 min, re-arm).

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
- Every publish and server deploy needs her "yes" to that exact action; none overnight.
- A game change a player asks for (numbers, what can be built) waits for her word (CLAUDE.md).
- Replies on contributor PRs: written for a person, offers not requests, she sees each first.
- Cores: the ledger's cap. Usage has been tight: be economical.
