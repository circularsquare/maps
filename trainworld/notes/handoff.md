# Orchestrator handoff, trainworld

Orchestrator: session 5c59ad2f-75a3-4536-8e9d-b38d3213ab6f, name maps-19 [f2de49], since 2026-11-12 21:52 UTC
Updated: 2026-11-15 (maps-19 winding down; Anita starts a fresh orchestrator, first job a reorganisation of how work
is tracked: "this project has definitely scaled past my clumsy todo list")

Read trainworld/CLAUDE.md, docs/orchestrator.md and TASKS.md first. Earlier detail: this file's git history.

## State
- **Live = v0.3.4**: Anita published it herself 11-15 (with GPT) from branch `hotfix-stop-freeze-034` (66ef6ca1, tree
  .tw-trees/stop-freeze-034, built on the v0.3.3 release b2897ae2): fixes freezes when editing stops on large saves
  (T-3385; hidden stop rows fed negative measurements). Website 3205539 + her later "intercity" 74f6a57. v0.3.3 went out
  11-13 ~14:00Z (data 2026-11-13; terrain v2 on R2 at terrain-world/v2).
- **Master**: origin at b1b88a66. Local master has her commit 0bb9fcfa (v0.3.4 fix + release notes + check
  t3385-hidden-row.mjs), NOT pushed. **Uncommitted in the main tree, hers (GPT's)**: app/src/game/world.ts,
  map/overlay.ts, ui/inspector/RelatedStops.tsx, workers/clock.worker.ts, clockClient.ts, protocol.ts, and the T-3385
  ARCHIVE entry with its TASKS line removed. Ask her before touching or committing these.
- Also uncommitted: perdir's DECISIONS line (2026-11-13, T-3348 stage 1): commit it only when per-direction stops land.
- **8800** = f8119117 from freeze 20261113-013559 (behind master: lacks the outline speedup 19a7f1e2 and v0.3.4's fix).
- **Server** shonei-server 8391326 deployed 11-13 16:28Z, pushed (twstats 15-minute playtime ring).
- **Open source**: github.com/circularsquare/trainsworld, last sync 2454e88 (maps e9fd4d34; sync commits now credit
  contributors). R2 data answers CORS `*` (forks can read it). Each publish: export.py, then `git push` there.
- **PRs**: #2 peanutmow: part 1 landed 19a7f1e2 (co-author), reply posted, left open for a settle revision (T-3387).
  #1 Renkon (bus one-way streets: she leans no, maybe an advanced toggle) and #3 CinnamonVII (run the built stretch
  of a partly built line: low priority) unanswered; she'll look Monday. Reply practice: docs/orchestrator.md.
- No agents running, no core claims. Data: nothing running; newest freeze 20261113-013559.

## Parked work in trees
- .tw-trees/perdir (tw/perdir from f8119117): T-3348 stage 1, per-direction stops, sim half built and tested (her
  split-legs case runs both ways). Missing: demand per run (2-3 h, plan in that tree's notes/T-3348.md "Stage 1 as
  built"), the stop marker at the return station, a named refusal on removing it. Her: hold if it costs much more.
- .tw-trees/tt2 (T-3346 stage 2, timetables; timetables are hidden behind ?timetables since v0.3.3).
- .tw-trees/hotfix, hotfix031: old release trees, removable.

## Next (her call; she's planning a reorganisation first)
1. With her: commit or finish her uncommitted v0.3.4 port on master, push, rebuild 8800, sync the open-source copy.
2. P2 bugs from players: T-3379 single-to-double kink, T-3380 delete ghost on single, T-3381 complex name in the
   corner notice, T-3383 "busy single track" wording (+ the 24m00s check), T-3386 inferred one-way lost at a station.
3. T-3376 industrial jobs (Jurong 128k vs ~27k real; "semi high prio"), T-3388 London edits take 4 s.
4. Later/design: T-3378 BRT and bus balance, T-3377 mods (levels 2-4), T-3382 where people play, T-3384 waypoints,
   T-3375 zero-resident units beside a metro, T-3358 rivers (paused by her). Next free id T-3389.

## Anita's standing asks
- Every publish and server deploy needs her "yes" to that exact action; none overnight.
- A game change a player asks for (numbers, what can be built) waits for her word (CLAUDE.md).
- Replies on contributor PRs: written for a person, offers not requests, she sees each first.
- Cores: the ledger's cap. Usage has been tight: be economical.
