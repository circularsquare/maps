# Orchestrator handoff, trainworld

Orchestrator: session 7186e3f8-2ac6-4929-90c0-cf653363ab62, name maps-b8 [a8e864], since 2026-11-15 15:22 UTC
(Took over from maps-19, session 5c59ad2f, already gone at 15:22 UTC with nothing in flight, so no wait.)
Updated: 2026-11-15 15:22 UTC (first jobs from TODO.md short term: the crash, then a kanban view of the work)

Read trainworld/CLAUDE.md, docs/orchestrator.md and TASKS.md first. Earlier detail: this file's git history.

## State
- **Live = v0.3.4**: Anita published it herself 11-15 (with GPT) from branch `hotfix-stop-freeze-034` (66ef6ca1, tree
  .tw-trees/stop-freeze-034, built on the v0.3.3 release b2897ae2): fixes freezes when editing stops on large saves
  (T-3385; hidden stop rows fed negative measurements). Website 3205539 + her later "intercity" 74f6a57. v0.3.3 went out
  11-13 ~14:00Z (data 2026-11-13; terrain v2 on R2 at terrain-world/v2).
- **Master 7c47580f** (the board, unpushed); before it 6feec4b4, pushed (11-15): her v0.3.4 fix 0bb9fcfa; her freeze diagnostics 43b3acbe (off unless
  `?diagnostics=1`; T-3385 archived); v0.3.4 changelog translated 6feec4b4 (live v0.3.4 shows those two lines in
  English until the next publish). Typecheck and i18n clean.
- Uncommitted on purpose: perdir's DECISIONS line (2026-11-13, T-3348 stage 1): commit it when per-direction stops land.
- **8800** = 6feec4b4 from freeze 20261113-013559, smoke PASS (12,581,681 rail trips a day).
- Her release tree .tw-trees/stop-freeze-034 (branch hotfix-stop-freeze-034) is what's live; keep it like hotfix031.
- **Server** shonei-server 8391326 deployed 11-13 16:28Z, pushed (twstats 15-minute playtime ring).
- **Open source**: github.com/circularsquare/trainsworld, last sync a4a3210 (maps 6feec4b4; sync commits credit
  contributors). R2 data answers CORS `*` (forks can read it). Each publish: export.py, then `git push` there.
- **PRs**: #2 peanutmow: part 1 landed 19a7f1e2 (co-author), reply posted, left open for a settle revision (T-3387).
  #1 Renkon (bus one-way streets: she leans no, maybe an advanced toggle) and #3 CinnamonVII (run the built stretch
  of a partly built line: low priority) unanswered; she'll look Monday. Reply practice: docs/orchestrator.md.
- No agents running, no core claims. Data: nothing running; newest freeze 20261113-013559.

## In flight (maps-b8)
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
   T-3375 zero-resident units beside a metro, T-3358 rivers (paused by her). Next free id T-3390.

## Anita's standing asks
- Every publish and server deploy needs her "yes" to that exact action; none overnight.
- A game change a player asks for (numbers, what can be built) waits for her word (CLAUDE.md).
- Replies on contributor PRs: written for a person, offers not requests, she sees each first.
- Cores: the ledger's cap. Usage has been tight: be economical.
