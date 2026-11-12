# Orchestrator handoff, trainworld

Orchestrator: session a67ba911-f213-4096-86ff-b9c1620cbea0, name maps-37 [07bc40], since 2026-11-12 17:54 UTC
Updated: 2026-11-12 ~19:20 UTC (14:20 local; she is awake and answering). maps-cb has finished and holds nothing.

Read trainworld/CLAUDE.md and docs/orchestrator.md first. Earlier log: handoff.md in git history.

## Live
- **Live = v0.3.2** (tw/release-031 6425f2f5, worktree .tw-trees/hotfix031; data 2026-11-11b; website 966f101;
  published 17:55Z 11-12, verified). ..\website's only leftover change, pages/projects/abouttrainsworld.md, is not ours.
  To do: copy the v0.3.2 changelog entry (V0_3_2_AT "2026-11-12T18:00:00Z") and its translations from 6425f2f5 to
  master (updates.ts + i18n messages/sources/context). Next big update is v0.3.3.
- **Server** (shonei-server e878c0f deployed 11-12). Committed, not pushed or deployed: 7a635f7 + e5a132a (T-3326
  feedback screenshots; config flag hides the app control on an old server), 306fbe3 + c3403aa + b2ab066 (flag set 3,
  336 codes). Next deploy, on her word: those, plus the play-stats browser cap (she OK'd). Deploy before any app
  that uses them ships.
- **Ops**: `tools/ops/check_resources.py` daily (last 11-12 16:04Z: fine). Feedback inbox `tools/ops/feedback.py`.
  Daily also: `gh pr list` / `gh issue list` on circularsquare/trainsworld once it exists (she OK'd 11-12).
- **Open source**: export ready (tools/opensource/export.py; C:\Users\anita\projects\trainsworld-export 4485c89).
  She OK'd name/email, README, PR method, and "lets rebuild and create and push once 3.3 is done yes"; sync at each
  publish (orchestrator.md). At v0.3.3: re-export, `gh repo create circularsquare/trainsworld --public --source .
  --remote origin --push` from ..\trainsworld-export (confirm with her that day).

## 8800 and master
- **Master bd8894d0** (110d7700 T-3339, 95082088 T-3333, bd8894d0 flags/gap names/gate fix/export), smoke PASS 28 on
  6139. Pushed only to e9dde1d1: push after snapbuild.
- **8800**: snapbuild bd8894d0 started 14:15 local (log in my scratchpad snapbuild.log). Before: 5f851cad.
- Uncommitted main tree: T-3344 draw.rs + lanes_tests.rs (t3344, browser check pending), checks/t3279-flags.mjs
  (t3351). TODO.md is hers.
- Worktrees: fbshots (T-3326 stage 2), onetrack (T-3332), hotfix031 (release). Stray Vites on 6164, 6167, 6130:
  owners unknown, left alone.

## Running agents (all maps-37's)
- **gaprun2** aa8d3bc594c20a4d8 (Opus, main tree + data/packs sole writer): T-3287 continues from gaprun's recipe
  (notes/T-3287.md): smoke, retire 10 pair files, check crn-me/poi-mk, rerun batch A + KP (6 cores), gates, join,
  then B and C. ~10 h. Messages me BEFORE each gate and waits for my "go gate": then no main-tree app/ writes.
- **onetrack** ae97500d3a62d1d0f (Opus, .tw-trees/onetrack, 6150): T-3332 one-track stations, 11-16 h, stages.
  Ids T-3354/3355.
- **t3326** a819255edd5716af0 (Sonnet, .tw-trees/fbshots, 6155): T-3326 stage 2 app form + inbox thumbnails.
  Strings drafted ("Attach screenshots" etc., in notes/T-3326.md) go to her when it reports.
- **t3351** a0db4967a90cee41a (Sonnet, main tree app/ briefly, 6126): copying six flags in; told to finish app/
  writes within ~40 min of 14:15.
- **t3344** a5e962a7f0196cdfa (Sonnet, main tree, 6123): browser check of the join fix (WASM already includes it),
  then closes T-3344. Filed T-3353 [P3] second-single geometry.
- **t3352** a7db66e57cfea54da (Sonnet, read-only): T-3352 scope, single track for high-speed and tram (her TODO
  11-12); builds after T-3332 lands.
- **t3329** a3bb12c205199e3b6 (Sonnet, read-only, 6121): T-3329 save size vs memory, propose a display. Quiet since
  ~13:00: ask after it if no report by ~15:00.

## Waiting on Anita
- Schematic (T-3345 stage 1 done, notes/T-3345.md): approach 1 for country maps only as stage 2? the all-Japan shape
  change (artifacts/trainworld/t3345-mock-japan-real.png)? try approach 2 (bigger city centres)? OK that the agent
  downloaded 23 public "let others open a copy" Community saves to its scratchpad? First image
  artifacts/trainworld/t3345-mock-beijing-player1.png.
- Timetables (T-3346, revised to her panel idea; notes/T-3346.md, mock artifacts/trainworld/t3346-panel-mock.png):
  her 6 calls (averaging into the 5 periods, no warning; whole hours, 6 rows/day; Sat+Sun together; meets stage 6;
  express behind local; the draft strings). Then stage 1 (panel + board + chart) and stage 2 (per-train waits).
- Text acks: T-3326's strings after stage 2.
- Later: T-3264 trains drawn through each other (mock).

## Queued (filed)
- T-3337 HSR top speed 350 and T-3352 single HSR/tram: after T-3332 lands (same track code).
- T-3335 [P3] two-pair panel parts, T-3343 [P3] per-train holds (timetable stage 2 absorbs), T-3348 [P3] quad
  platforms, T-3353 [P3].
- T-3320 Berlin RE5 (P3), T-3322 t2266 check (P3), T-3325 Sendai rebuild (P3).
- Before v0.3.3: translation batch for all new strings; website about-page flag credit line (notes/T-3328.md);
  server deploy first.

Next free id T-3354 (T-3354/3355 given to onetrack; use T-3356 next).

## Her calls 11-12 (detail in DECISIONS and ARCHIVE)
With maps-37: next areas "single-track join fix and one-track stations, and timetables design", schematic as a stab
she reviews first, open source soon; v0.3.2 publish (via maps-cb); text acks T-3333, T-3339; flags: redraw the 4
missing Spanish communities, Bali, Yogyakarta, all simply, names English, purple cat lion fine; timetables: new-player
experience unchanged, advanced scheduling panel centre-screen, demand keeps 5 fixed periods, free intervals with a
cap, weekend rows, a multiple, one-click weekends only/off, rows "M-F 6-9" with day colours, holidays as weekends;
open source as above. With maps-cb earlier: see git history of this file.

## Anita's standing asks
- **No deploys overnight** without her there; every server deploy and publish needs her word.
- Cores: the ledger's cap (`python cores.py list`).
- **Usage**: Opus more widely but "not fully gung-ho"; where there's a choice, spend compute, not tokens.
- Watch whether agents use wiki-first reading well.
