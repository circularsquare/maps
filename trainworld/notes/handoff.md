# Orchestrator handoff, trainworld

Orchestrator: session 441e3071-8b6e-4109-9ab2-1339e17b92be, name maps-5e [3a1e06], since 2026-11-12 04:41 UTC
Updated: 2026-11-12 15:50 UTC (10:50 local; she is awake and answering). Context past 500k: ready for a takeover.

Read trainworld/CLAUDE.md and docs/orchestrator.md first. This session's log is in git history (handoff.md at
e9dde1d1 and earlier).

## Live
- **Live = v0.3.1** (tw/release-031 head 71009033, worktree .tw-trees/hotfix031, read-only for agents; data
  2026-11-11b; website cc14a89). A deploy yes is not a text ack: changelog/player text waits for "text acknowledged".
  Not live yet (next publish): option A eb23b84b, jpn7 Shinkansen (T-3272), the world rebuild + university fixes,
  flags client (server side is live), satellite, zoom buttons, Community View, station fixes. Translation batch for
  all new strings before that publish; website about-page flag credit line (notes/T-3269.md).
- **Server** (shonei-server e878c0f, deployed 15:01Z 11-12 with her word, pushed): caps 45k accounts / 22.5k
  password / warn 30k, browsers 400k; flags (trainworldFlagSet 2: NYC + 47 prefectures); best game per board;
  likes tiebreak; her 60-min exemption (SHONEI_COMMUNITY_NO_WAIT).
- **Ops**: daily `tools/ops/check_resources.py` (last 11-12 14:00Z: noritetsu cache rule works, GetObject 1.6M/h ->
  10-20k/h; Resend at its daily cap, she said no Pro; R2 storage 16.9 GB; website repo 1.1 GB). Feedback inbox:
  `tools/ops/feedback.py` (read-only, key ~/shonei-feedback-admin-key.txt). Cloudflare ops token ~/cloudflare-ops-token.txt.

## 8800 and master
- **8800 = e9dde1d1, smoke PASSED** (her checks: satellite fade good, shared station names good, lengthening works).
- Master 2ed7a727 (feedback reader). Pushed to e9dde1d1 (two pushes this session; she allows a couple).
- World rebuild done 11-12: light pass 20261112-025728 (3,919 packs) + T-3273 rerun 065909 (828) + T-3274/3275
  rerun 084020 (1,041); frozen data/frozen/20261112-*; backup data/work/pre-T3244/ can go now (state checked clean).
- Uncommitted in main tree (agents'): realNetwork.ts + realgraph.py (lightrail); market_tourism.*, market_zones.py,
  sim/src/demand/intercity*, regional.md, demand-plain.md, demand-worldwide.md, wiki between-cities (jpattract);
  data-register.md (jpattract + gaps); notes/T-3287.md + TASKS line (gaps). TODO.md is hers.

## Running agents (all report to maps-5e; resume by id)
- **lht** a9bf2515b4ef71c76 (Opus, worktree .tw-trees/lht, 6164): T-3297 left-hand traffic BUILD, her calls: new
  game side from the start country's railway side, toggleable; trams/buses follow; real saves of left cities ship
  left (saves to scratch, copy commands to me); flip-anyway with warning, broken-sites list, auto-fix safe cases.
  Reports after each stage (1 sim+drawing, 2 choice+real saves, 3 flip). Ids T-3311/3312.
- **bugs** a911a89a73ce91011 (Opus, worktree .tw-trees/bugs, 6165/6166): T-3313 stuck-game reports (figures stop
  updating, 0 passengers, trains vanish at terminals, construct button, all trains stop, mizushin other-device money).
  Told about feedback.py. Ids T-3314-3316. Report after ~3 h.
- **haru** ae9bb543ae404b5d5 (Opus, worktree .tw-trees/haru, 6159/6160): T-3302 Haru's two Tokyo saves don't load
  (scratchpad harusaves\in). Ids T-3305/3306. Wants a reply she can send.
- **money** a3df8170043e5a9f4 (Sonnet, worktree .tw-trees/money, 6161): T-3303 fare exploit (on-board riders pay a
  fare set mid-ride; she knows the detail), T-3304 x100 money hover (her sentence verbatim), T-3307 own WebGL2
  message (strings to her). Ids T-3308/3309. Says if the exploit fix is hotfix-worthy.
- **hsrprofit** a73b4d35d12f852f8 (Sonnet, read-only, 6162/6163): T-3310 Tokaido-style HSR profit on live vs master;
  skill issue or balance. Then I decide the HSR top speed (320 -> 350? a player asked; rail fact, my call).
- **transfer** a03f13c66621b0720 (Opus, worktree .tw-trees/transfer, 6156): T-3295 change between two stops of one
  line at a near self-intersection; also Caio's tram->train walk (feedback 1794440480-9c44ca08), file separately if
  different. Ids T-3298/3299.
- **jpattract** a970ecbf4714f9168 (Opus, main tree): T-3280 option A BUILD (her yes): Japanese resort towns'
  published visitors in Japan's long-trip weights, beta under rail counts. Hands back a Japan market-only rerun.
- **lightrail** a0244723dad581e04 (Sonnet, main tree, 6154): T-3285 light_rail in the real network layer + snapping,
  118 realgraphs + Berlin rebuilt; closes T-3265. Next for it: T-3289 (real network setting none/passenger/all).
- **dbltrack** (done): T-3296 single -> double rebuild, archived, in worktree .tw-trees/dbltrack, READY TO LAND but
  HELD until jpattract (sim edits in main tree) and hsrprofit (measures main tree's WASM) finish: a landing rebuilds
  the main tree's WASM. Strings sent to her (awaiting ack). Then start T-3317.
- Worktrees: transfer, dbltrack, haru, money, lht, bugs (+ hotfix031). Disk 25 GB free, ~1.2 GB each: no more than
  one more until some land. Worktrees from 32ac89b1/later must copy main's registry.json + market-rows.json before
  pages and never land them (revert in the tree first, as done for dbltrack).

## Queued (filed, start as slots free)
- T-3317 [P1] track/train bug batch from feedback (HSR platforms > 4 cars, quad upgrade not using 4 tracks, one-way
  singles meeting double cross lanes, local leaving ahead of express, loop lines, loopo's other-direction lengthening)
  -> after dbltrack lands; Opus.
- T-3318 place names (Changshu in Chinese, Hyogo shown as Osaka, Sendai Senzan skips Toshogu); Sonnet.
- T-3319 line icon shapes by mode with a per-mode default in Settings (her words in the line); Opus (UI).
- T-3289 freight/all rail setting -> lightrail.
- T-3287 gap regions: her answers 11-12 10:50: plan fine; ordinary zones; ALSO all north Scandinavia (Lappi,
  Norrbotten, Nordland, Troms, Finnmark; "the train to north norway is kinda famous"); NW Ontario's south rim along
  the east-west route; North Korea its own realm; Spain + Portugal first ("lots of players"), then Nordic + Baltic,
  NK alongside, Canada last. gaps ac5cdaf84781d40c7 resumed: rule in world_plan.py + plan rows + dry runs + a run
  plan per batch (no data/packs writes); the orchestrator schedules the world runs. T-772 archives into it.

## Waiting on Anita
- Double track strings (T-3296): "double track" in hover/confirm/notices; "Pick a level or track type" -> "Pick a
  level, track type or double track".
- Later: T-3264 trains drawn through each other, stage 1 needs her look at a mock (notes/T-3264.md, 13-19 h).
- Longer-term TODO items not filed: leaderboard map view, brainstorm, intercity-only mode, more languages, offline.

Next free id T-3320.

## Her calls 11-12 (detail in DECISIONS and ARCHIVE)
Option A stays; wealth tiers parked; China pull 0.6; Poland slight; undershoot rule for all demand fits; squares not
hexes; best game on every board; zoom buttons + flags; strings: "View a copy", "This network isn't saved. You can save
it to keep it.", "High-speed track can't be single track.", every Ctrl-draws-straight hint dropped; flag round 2
(her words in notes/T-3269.md); flag hover OK; server caps +50% / browsers x2; no Resend Pro; Georgia Gwinnett-style
campuses left (T-3276); Japan tourism option A; left-hand traffic as above; T-3265 light rail yes; T-3261 as built;
feedback reader yes. Earlier (11-10/11): single track two-way, margin 240 s; double track one piece; flyover ramps 4%;
train cap running + 1; HSR and tram quad yes; buses/trams on any road; text copied from her message needs no review.

## Anita's standing asks
- **No deploys overnight** without her there; every server deploy needs her word.
- Cores: ledger cap 14 (`python cores.py list`).
- **Usage**: Opus more widely but "not fully gung-ho"; where there's a choice, spend compute, not tokens.
- Watch whether agents use wiki-first reading well.
