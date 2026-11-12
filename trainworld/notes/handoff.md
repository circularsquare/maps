# Orchestrator handoff, trainworld

Orchestrator: session 3397f27a-d2e0-438d-a3c0-8538d4d84f99, name maps-cb [f61388], since 2026-11-12 15:56 UTC
Updated: 2026-11-12 17:31 UTC (12:31 local; she is awake and answering). Times below before this line ran ahead; trust
this one.

Read trainworld/CLAUDE.md and docs/orchestrator.md first. Earlier log: handoff.md in git history (e9dde1d1 and before).

## Live
- **Live = v0.3.1** (tw/release-031 head 71009033, worktree .tw-trees/hotfix031; data 2026-11-11b; website cc14a89).
- **Hotfix v0.3.2 in preparation** (her yes 11-12): agent **hotfix** a8894d084b7b7d4b7 (Opus) in .tw-trees/hotfix031:
  T-3303 fare fix (from 7aad5a0a), FARE_MAX (from cfc1e5ff), patientFetch (from 633dd492); changelog line "update
  v0.3.2: fares hotfix" (her name, no "!"), changes "fixed a new fare paying out on long trips before riders had
  answered it", "fares now have a cap", "fixed figures freezing when a download hung" (she acked "rest looks good"),
  translated; staged smoke on live data. Then I commit on tw/release-031, run deploy.py (--skip-data, its commands),
  website push, --verify --page. Not overnight. The big update after it is v0.3.3.
- **Server** (shonei-server e878c0f, deployed 11-12 with her word, pushed). At the next server deploy: raise the
  play-stats browser cap (her OK 11-12; 80% in ~19 days).
- **Ops**: daily `tools/ops/check_resources.py` (last 11-12 16:04Z, reported: GetObject 10-20k/h since the cache rule;
  Class B ~$21 accrued this month, ~$24 at month end; Resend at its daily cap, no Pro; R2 16.9 GB; website repo 1.1 GB).
  Feedback inbox `tools/ops/feedback.py`. Reading players' server saves allowed but only when nothing else settles it.

## 8800 and master
- **8800 = 5f851cad**: double track both ways, currency, stuck-game fixes, speed control, real network setting, Japan
  resort towns, same-line change, x100 hover, WebGL2 message, fare cap. Her checks: speed button good, rebuild good.
- **Master ab719459**, not on 8800 and not smoked since e7fd13ba: + shapes 8b58db89, HSR fare 1.25 2edb55ac, notices
  1e2ecf23, fare fix 7aad5a0a, left-hand traffic e27c2790 (+ 9 left real saves copied into data/packs/real), gap plan
  ab719459. Next: smoke master on my Vite, snapbuild 8800 (both after gaprun's gates, see below). Pushed to e9dde1d1 only:
  push once at a quiet point (she allows a couple a session).
- Stray Vite on 6164 (PID 21764, from 10:34 local, owner unknown): left alone.
- Uncommitted in main tree: names' T-3318 edits (places.ts, place-names.json, place-names-zones.json placeholder,
  localization.md, place_names.mjs, realnet.py, notes/T-3318.md, TASKS line). TODO.md is hers.

## Running agents (all mine; maps-5e has finished and dropped its Previous: line)
- **gaprun** a6810457fe5fd77c0 (Opus, main tree + data/packs): T-3287 stage 3, zone-name pass then batches A (ES PT FR
  RO BA GR HR AL + North Korea, one join), B (NO SE FI EE LV LT), C (NW Ontario). ~10-11 h machine. It messages before
  each join's gate: **hold every agent's main-tree app/ writes during gates**. No other data/packs or terrain writer.
- **hotfix** (above).
- **onestop** a9928c13315344193 (Opus, .tw-trees/quad2, 6177): T-3333 line on both quad pairs (hidden twin line, save
  level 17); sim done, app under way; told to merge master first. Then T-3332 one-track stations (her calls in
  DECISIONS 11-12). Ids T-3335/3336.
- **dbl2** a717e96e565fe7bc8 (Opus, .tw-trees/dbl3, 6170): T-3339, quad -> two singles allowed; a station with passing
  tracks stays double as the loop, track either side single; both refusals go. Answers her "when would this come up".
- **names** aa1e8c4993b55e4d6 (Sonnet, main tree, 6173): T-3318 place names, waiting on its Wikidata run; any pack
  rebuild comes back as a command for me (schedule around gaprun). Ids T-3324/3325.
- **kofi** a7fa9ca9091380371 (Sonnet, main tree, 6181): T-3340 Ko-fi link in Settings, her sentence verbatim.

## Queued (filed)
- T-3337 HSR top speed 350 (my call) after quad2 lands. T-3317 track bug batch (quad item = T-3333's wish).
- T-3326 feedback screenshots (app + server deploy), T-3328 flags (Spain regions, Lisbon, Jakarta; server flag set),
  T-3329 save size vs memory (measure, then a display proposal to her). T-3320 Berlin RE5. T-3322 t2266 check.
- Translation batch for all new strings before v0.3.3; website about-page flag credit line (notes/T-3269.md).

## Waiting on Anita
- Text acks outstanding: T-3339's changed double-track strings once dbl2 reports.
- Later: T-3264 trains drawn through each other (mock), longer-term TODO items not filed.

Next free id T-3341.

## Her calls 11-12 (detail in DECISIONS and ARCHIVE)
Today with maps-cb: real network All without freight; WebGL2/real-network/fare-cap/notices/left-hand text acked
("Trains run on", "By default set from the country you started in."), autosave age in the stall notice; one-track
stations and quad two-pair designs as recommended; currency for plain ja/ko/zh; v0.3.3 waits for left-hand + track
work, no rush; HSR fare default 1.25, no in-game hint; "Edit name, color, or symbol" kept; hotfix yes, named "update
v0.3.2: fares hotfix"; exploited leaderboard scores left alone; mizushin left alone; Ko-fi link for the next update.
Earlier 11-12: option A, wealth tiers parked, China pull 0.6, squares not hexes, best game per board, zoom buttons +
flags, server caps, no Resend Pro, Japan tourism A, left-hand traffic, light rail, feedback reader.

## Anita's standing asks
- **No deploys overnight** without her there; every server deploy needs her word.
- Cores: the ledger's cap (`python cores.py list`).
- **Usage**: Opus more widely but "not fully gung-ho"; where there's a choice, spend compute, not tokens.
- Watch whether agents use wiki-first reading well.
