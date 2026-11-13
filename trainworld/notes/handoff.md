# Orchestrator handoff, trainworld

Orchestrator: session 5c59ad2f-75a3-4536-8e9d-b38d3213ab6f, name maps-19 [f2de49], since 2026-11-12 21:52 UTC
Updated: 2026-11-13 00:50 UTC (19:50 local). maps-37 has closed (its agents all reported or handed over).

Read trainworld/CLAUDE.md and docs/orchestrator.md first. Earlier log: handoff.md in git history.

## Her focus now
**v0.3.3, then terrain** (Anita 11-12 evening, away ~16 h from ~19:00 local, back briefly ~01:00 local): "probably dont
start new work unless its obvious, mosty just wanna finsh the 0.3.3 stuff and alsowork on terrain". Use all cores
(cap set 16). Usage low this week: no big change, but economical. AA/BB (T-3348) and the track model (T-3365) PARKED.
Feedback triage rule: docs/orchestrator.md (bugs touching many: look; design: to her; few: file).

## Live, server, ops
- **Live = v0.3.2** (tw/release-031 6425f2f5, .tw-trees/hotfix031; data 2026-11-11b; website 966f101).
- **Server** shonei-server e878c0f deployed. Committed, NOT pushed or deployed, HOLD until v0.3.3 (her word): 7a635f7 +
  e5a132a (T-3326 feedback screenshots), 306fbe3 + c3403aa + b2ab066 (flag set 3), the play-stats browser cap.
- **Ops**: check_resources.py daily (last 11-12 16:04Z; note it now probes terrain v2, which fails until v2 is
  uploaded); feedback.py inbox.
- **Open source**: at v0.3.3, re-export and `gh repo create circularsquare/trainsworld --public ...` (notes/T-2913.md;
  she said yes "once 3.3 is done"; confirm that day).

## Master, 8800
- **Master c1e52978, pushed; smoke PASS 28 (terrain v2 + batch A).** **8800 = c1e52978 from freeze 20261112-204835**
  (snapbuild-frozen; serves terrain v2, checked on Chongqing's block). gaprun3 step 1 done (build_inputs 0 flags,
  smoke 28, real save 12,581,681 vs 12,581,682, freeze 20261112-204835). **Update ~22:50 local**: B shipped (650a3caf), after-join all passed, freeze 20261112-223751; **8800 = 650a3caf from
  that freeze**. Batch C gate given ("go gate" ~22:50): hold main-tree app/ until gaprun3 says the window ended. Master
  c83e5bf7 NOT pushed: the permission check refused the push this time (pushed several times already this session).
  **Her answers ~23:00 local**: drop the changelog touch-zoom line (updates.v033Touch: remove from updates.ts, en.json,
  context.json and all 22 language files, AFTER C's gate window ends); terrain +8% accepted, v2 publish waits till she's
  back; rivers build GO -> **rivers2** a85c231bd34d5f479 (Opus, .tw-trees/rivers, Vite 6158), she reviews before landing.
  Fixes line unanswered: kept as the full list.
  **She's home (~23:40 local)**: pushed master (e6ff5783, her OK); "terrain looks great"; touch line dropped e6ff5783;
  RIVERS PAUSED (she wanted the river straightened, disliked prototype 2's look): rivers2 stopped, tree removed,
  5d1dcc3c. Batch C join (CA) running, claim gaprun-joinC 12, then after-join + freeze, then 8800 rebuild.
  Earlier: B gated 19/19 (tlk relaxed, ute
  origin moved to Utena by hand), window ended 21:33; backup prejoinB; **B join running from ~21:35 local, 12 jobs,
  1.5-2 h: nothing else writes data/packs or data/terrain**. zone_terrain --force done (4,024 files, new method).
- Landed this session: e0bb2de1 T-3367 running side off new-world screen; dc2eb030 one-way station round (T-3332);
  feb0616b T-3366 bus drag; 7a8894cc station-type icons +15%; 23c0335a T-3368 touch zoom cap 20; f68a53d8 T-3352 single
  high-speed/tram (T-3337 350 km/h DROPPED, she keeps 320); 64715623 v0.3.3 changelog + translations (11 languages);
  d535f1ca gap batch A + North Korea shipped; d4c75567 T-3372 finer terrain + hexes (reads data/terrain-world-v2).
- Last smoke on master: f68a53d8 PASS 28 (T-3372's own smoke on its tree passed 28). Smoke master again before 8800.
- Worktrees: tt2 (T-3346 stage 2, not landing for v0.3.3), hotfix031, hotfix (old release-021). Shared-file staging:
  scratchpad `stage_lines.py <repo> <path> <marker>...` (per line, markers like 'T-3366 ').
- **Disk**: copying gaprun2's kit filled C: (~58 GB) at 19:45 local; copy deleted, 58 GB free. Never copy a run kit.

## Running agents (maps-19's)
- **gaprun3** a23dc863ea45d0dec (Opus, main tree, sole data/packs writer): T-3287 after gaprun2 (notes/T-3287.md
  "Stage 5: handover", it adds "Stage 6"). Adopted: market_drives (python 6304, claim gaprun-drives 8, ends ~22:30
  local), batch B build-only (python 14788, claim gaprun-B 4). Step 1 after drives: build_inputs, smoke, real save vs
  12,581,682, freeze.py, `zone_terrain.py build --force --jobs 4`; reports. Then B gate: it asks "ready to gate B", I
  answer "go gate" (no main-tree app/ writes or landings in the window), join, after-join; then C (dry, ago, CA).
  Uses gaprun2's kit in place (a67ba911 scratchpad\gaprun2).

## Waiting on her (all in the last For you block)
- **Terrain v2 publish**: `python trainworld/tools/deploy.py --terrain-world --terrain-dir trainworld/data/terrain-world-v2`
  (0.75 GB to R2). Release builds from master refuse until v2 is complete on R2.
- **Changelog text**: kept lines 18-25 + headline shortened to 80 chars; touch-zoom line reword or drop; fixes line
  full or "plus some smaller fixes". `V0_3_3_AT` set at publish.
- **Terrain prices**: +0.7-7.5% sample totals, single valley routes up to ~2x (I called it a correction).
- **Rivers (T-3358)**: prototype 2 (her idea: no straightening near rivers) works (London wrong pairs 97 -> 27 of 1,469,
  12-15 with a second pass; straightness unchanged). Build 6-8 h (fresh Opus in a tree) waits for her go.
- T-3373 filed (demand workers sharing zone data; design first), not started.

## v0.3.3 before publish
Translation top-up if any string changes; server deploy (her word); terrain v2 upload (her word); website about-page
flag credit (notes/T-3328.md); smoke8800 light + dark; deploy.py with a new data version (gap packs, after gaprun3's
freeze); open-source repo. Out of v0.3.3: T-3346 stage 2, two pairs (hidden behind ?pairs), rivers.

## Queued (filed)
T-3360 schematic bus/ferry settings; T-3346 stages 2-6 after v0.3.3; T-3370 bus drag follow-ups; T-3371 t3260 part B
stale; T-3369 (vague report). Parked: T-3348, T-3365. Next free id T-3374.

## Anita's standing asks
- No deploys overnight without her; every server deploy and publish needs her word.
- A game change a player asks for (numbers, what can be built) waits for her word (CLAUDE.md, 11-12).
- Cores: the ledger's cap. Usage: spend compute, not tokens.
