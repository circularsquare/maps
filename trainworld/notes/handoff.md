# Orchestrator handoff, trainworld

Orchestrator: session 3397f27a-d2e0-438d-a3c0-8538d4d84f99, name maps-cb [f61388], since 2026-11-12 15:56 UTC
Updated: 2026-11-12 17:48 UTC (12:48 local; she is awake and answering). Context past 500k: ready for a takeover.
(Times in the earlier part of this session's notes ran 1-2 h ahead; these are checked with `date`.)

Read trainworld/CLAUDE.md and docs/orchestrator.md first. Earlier log: handoff.md in git history.

## Live
- **Live = v0.3.1** (tw/release-031, worktree .tw-trees/hotfix031; data 2026-11-11b; website cc14a89).
- **Hotfix v0.3.2 READY, waiting only on her explicit "publish"** (the permission check refused the website commit +
  push; it needs her latest message to say yes to it). Committed 6425f2f5 on tw/release-031 (T-3303 fare fix, FARE_MAX
  cap, patientFetch; changelog "update v0.3.2: fares hotfix" + 3 lines she acked, all 11 languages; staged smoke on live
  data 27/28 light + dark, the one fail is CORS on localhost). dist rebuilt from the commit; `deploy.py --skip-data
  --version 2026-11-11b --dist .tw-trees\hotfix031\trainworld\app\dist --app ...\app --smoked` already wrote the website
  files into ..\website (uncommitted). On her yes: `git -C ..\website add trainsworld/ trainworld/`, commit
  "trainsworld v0.3.2: fares hotfix", push (leave pages/projects/abouttrainsworld.md out, not ours), wait for Pages,
  `deploy.py --verify --page --version 2026-11-11b`. V0_3_2_AT is 18:00Z; if publish slips past ~19:00Z, set it,
  recommit, rebuild (scratchpad\hotfix\build.cmd) and rerun deploy.py. Then copy the v0.3.2 changelog entry and its
  translations to master (updates.ts + i18n messages/sources/context), after the gates. Next big update is v0.3.3.
- **Server** (shonei-server e878c0f, deployed 11-12). Next server deploy (her word): raise the play-stats browser cap
  (she OK'd), and flags3's flag set once it reports.
- **Ops**: `tools/ops/check_resources.py` daily (last 11-12 16:04Z: fine; Class B ~$21 accrued from before the cache
  fix; Resend at daily cap, no Pro). Feedback inbox `tools/ops/feedback.py`. Reading players' server saves: allowed but
  only when nothing else settles a report. Triage player reports by reach first (orchestrator.md).

## 8800 and master
- **8800 = 5f851cad**: double track both ways, currency, stuck-game fixes, speed control, real network setting, Japan
  resort towns, same-line change, x100 hover, WebGL2 message, fare cap. Her checks: speed button, rebuild good.
- **Master 44f6a621**, not smoked since e7fd13ba, not on 8800: + shapes, HSR fare 1.25, stall notice + Safari msg,
  fare fix, left-hand traffic e27c2790 (+ 9 left real saves in data/packs/real), gap plan, place names, Ko-fi link.
  After gaprun's gates: smoke master on my Vite, snapbuild 8800, push master (only pushed to e9dde1d1; a couple a
  session allowed).
- Uncommitted in main tree: gaprun's zone-name pass (zone-names.json etc.), flags3's work as it lands. TODO.md is hers.
- Stray Vite on 6164 (PID 21764, from 10:34 local, owner unknown): left alone.

## Running agents (all mine; maps-5e finished)
- **GATES ~13:30-15:00 local 11-12**: gaprun's batch A + NK gates serve main app/: no main-tree app/ writes and no WASM
  landings until gaprun says they ended (flags3 told; tell any new main-tree agent).
- **gaprun** a6810457fe5fd77c0 (Opus, main tree + data/packs): T-3287 stage 3, zone-name pass, then batches A (ES PT FR
  RO BA GR HR AL + North Korea; runs started 12:45 local), B (NO SE FI EE LV LT), C (NW Ontario). ~10-11 h machine;
  messages per batch and before each gate. Sole data/packs + terrain writer meanwhile.
- **onestop** a9928c13315344193 (Opus, .tw-trees/quad2, 6177): T-3333 quad two pairs, sim + app done (15/15), merging
  master 44f6a621 again, then Rust tests + smoke; then T-3332 one-track stations (her calls in DECISIONS). Save level
  17. Ids T-3335/3336. Strings sent to her (not acked yet).
- **dbl2** a717e96e565fe7bc8 (Opus, .tw-trees/dbl3, 6170): T-3339, quad -> two singles allowed; a station with
  passing tracks stays double as the loop, track either side single; both refusals go.
- **flags3** aef2267e60e174c2d (Sonnet, main tree + shonei-server commit, 6183): T-3328 region flags; server set
  committed there, not deployed.
- Done today and committed: lightrail, jpattract, transfer, money, haru, bugs, dbltrack, speed, currency, notices,
  shapes, hsrfare, fare2, lht, gaps (plan), names, kofi, hotfix (on tw/release-031). hsrprofit measured only.

## Queued (filed)
- T-3337 HSR top speed 350 (my call) after quad2 lands. T-3317 track bug batch (quad item is T-3333's wish).
- T-3326 feedback screenshots (app + server deploy), T-3329 save size vs memory (measure, then propose a display).
- T-3320 Berlin RE5 (P3), T-3322 t2266 check (P3), T-3325 Sendai rebuild (P3), T-3324-style check for other scripts.
- Before v0.3.3: translation batch for all new strings; website about-page flag credit line (notes/T-3269.md).

## Waiting on Anita
- "publish" for the v0.3.2 hotfix (above).
- Text acks: T-3333's five strings ("Run trains on two pairs of tracks" is hers; four "Trains run on one pair of
  tracks. ..." lines); T-3339's changed double-track strings once dbl2 reports.
- Later: T-3264 trains drawn through each other (mock).

Next free id T-3341.

## Her calls 11-12 (detail in DECISIONS and ARCHIVE)
With maps-cb: real network All without freight; WebGL2, real network, fare cap, notices (plus autosave age), left-hand
("Trains run on", "By default set from the country you started in.") text acked; one-track stations and quad two-pair
designs as recommended; currency for plain ja/ko/zh; v0.3.3 waits for left-hand + track work, no rush; HSR fare
default 1.25, no in-game hint; "Edit name, color, or symbol" kept; hotfix yes, "update v0.3.2: fares hotfix", no "!";
exploited leaderboard scores and mizushin left alone; Ko-fi link (red heart; text "please consider donating to keep
trains world running!", hover "this game is free, but the cost to host it is starting to get expensive. please
consider donating!"); quad -> two singles allowed, passing tracks question answered by T-3339; low-reach reports filed
for later. Earlier 11-12: option A, wealth tiers parked, China pull 0.6, squares not hexes, best game per board, zoom
buttons + flags, server caps, no Resend Pro, Japan tourism A, left-hand traffic, light rail, feedback reader.

## Anita's standing asks
- **No deploys overnight** without her there; every server deploy and publish needs her word.
- Cores: the ledger's cap (`python cores.py list`).
- **Usage**: Opus more widely but "not fully gung-ho"; where there's a choice, spend compute, not tokens.
- Watch whether agents use wiki-first reading well.
