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
  Found the panic was T-2776 (register race on evo's unregister), not stale pair files: I OK'd fixing T-2776 in
  market_zones.py (heal orphaned market packs under the lock, ~7 s per register), skipping the pair-file move,
  withdrawing that DECISIONS rule. Smoke 28/28 on 6140. Committed 1595924e. Runs started ~14:35 local (A python
  6460, KP python 43924). **GATE WINDOW ~15:20-17:30 local, "go gate" given: no main-tree app/ writes, no landings,
  until gaprun2 says it ended.**
- **onetrack** ae97500d3a62d1d0f (Opus, .tw-trees/onetrack, 6150): T-3332 one-track stations, 11-16 h, stages.
  Ids T-3354/3355.
- **t3326** DONE stage 2, landed d78d1d16 (fbshots tree removed). Strings sent to her for ack. Left: server deploy
  (her word; asked 11-12), translations in the v0.3.3 batch.
- **t3351** a0db4967a90cee41a (Sonnet, main tree app/ briefly, 6126): copying six flags in; told to finish app/
  writes within ~40 min of 14:15.
- **t3344** a5e962a7f0196cdfa (Sonnet, main tree, 6123): browser check of the join fix (WASM already includes it),
  then closes T-3344. Filed T-3353 [P3] second-single geometry.
- **t3352** a7db66e57cfea54da DONE scope (notes/T-3352.md): 7-10 h, no save change, after T-3332 (same files). My
  calls, told her: single HSR/tram at 0.6x the kind's price (also fixes auto flyover legs priced as regular), single
  street tram allowed, two-way single HSR allowed, Select bar retypes single, tram stops no loop by default, One way on
  single tram. Build in a tree after onetrack lands (resume t3352 or fresh Sonnet).
- t3351 DONE a15deda8 (six flags). Server set 3 = 336 codes (b2ab066).
- **t3329** a3bb12c205199e3b6 DONE measure (notes/T-3329.md, 09dc97ab): memory follows zones/stations reached times
  demand workers (+0.5 GB each), not file size; proposal B live Settings reading first, A saves-list estimate. Sent
  her the A/B question with strings. On her answer: resume it to build.
- Committed since: 47324b04 T-3344 (closed), 08c27589 v0.3.2 changelog on master.

## Her 8800 test + answers 11-12 ~15:10 local
- Two-pair box doesn't show for her -> T-3356 P1, agent **quadfix** aebad7c10cd823406 (Opus): diagnose read-only now,
  fix in .tw-trees/quadfix.
- Pill shape out, "turn around" not "turn round" (rule added to CLAUDE.md Player text) -> T-3357: Sonnet agent in
  the main tree AFTER the gate window (en.json `issues.buildingNoStation`, `sideSwitch.ask`, `sideSwitch.reversed`).
- Memory: B live reading AND A estimate + warning, yes -> resume t3329 a3bb12c205199e3b6 after the gate window.
  Limit (8/6 GB, red 85%) and strings still unanswered.
- Schematic: city-turn fix yes, Japan shape yes, public Community saves fine "but efficient". Water continuity
  (rivers as oval blobs) -> T-3358. t3345 af6237aa0c40ccfc1 resumed: water study now, then city-turn in .tw-trees/schem.
  She asked what "bigger city centres" means (answered in chat).
- Timetables: "Custom" for own rows; whole hours, averaging, Sat+Sun, text all yes; express behind local yes; three
  modes (current / rows+weekends / exact: nudge, remove, add trains, Marey chart, conflicts highlighted with delay);
  big train-by-station grid, hideable. t3346 revised (Simple / Detailed / Exact; first-come conflicts resolved by the
  sim; stages 1-5 44-63 h, Exact 24-34 h more). Her round 3 (in notes/T-3346.md top + DECISIONS): plan and stages
  good, nudge 1 min / Shift 15 s / Ctrl 10 min, first come first served, arrows on the selected train, chart scaled by
  distance, text acked. Stage 1 (panel, Opus) held until onetrack or quadfix lands (5 Opus agents and 5 trees now). Stage 2 started: **tt2** a8e910f825e247d9e (Opus,
  .tw-trees/tt2, 6160).
- T-3358 water study done (notes/T-3345.md "T-3358"): rivers traced as short arcs between junctions, each drawn
  alone and short ones dropped (London 115 arcs -> 29 drawn). Approaches: 1 join arcs into courses (3-4 h, my pick),
  2 route rivers as their own paths off parallel lines (8-12 h, central London), 3 thin bands for uncrossed rivers.
  Sent her review/schematic-water-london.png. t3345 now on city-turn in .tw-trees/schem. Her answer: option 1 yes,
  2 and 3 not needed; priority is stations never on water and on the correct side of each river ("topological
  correctness is important"): told t3345 to measure, fix, and add a standing check, in the schem tree after city-turn.
- **schem tree READY to land** (t3345 finished, context 580k, done): city-turn fix for maps >= 600 km, sheet.ts +
  docs/ui-diagram.md, one-city maps bit-identical, ARCHIVE + DECISIONS in main tree. Land after the gate window, then
  smoke. River work NOT landed: joining rivers alone raises wrong-bank stations (NYC 6 -> 24, Berlin 27 -> 39); next is
  side-keeping inside the fit (4-6 h), code + check t3358-river-sides.mjs in t3345's scratchpad river-wip/.
  **schem2** ad078e33a689dbd26 (Opus, Vite 6175): now T-3362 Miami strip diagnosis + bigger-centres mocks (read-only);
  after schem lands, message it + make it a tree for the river side-keeping. T-3360 bus/ferry settings later.
- **onetrack tree READY to land** (T-3332 done, onetrack agent finished): save level 18, sim 554 / demand 739,
  t3332-onetrack 13/13, smoke 28/28. Main-tree ARCHIVE/DECISIONS/TASKS done, T-3354 [P3] filed. Possible conflict
  with quadfix in sim/src/track/twin.rs (twin_plan vs station_beside). Shots copied to artifacts/trainworld/t3332-*.png,
  review/onetrack-station.png. Seven strings + test + drawing check go to her once on 8800. Then T-3352 (resume t3352
  a7db66e57cfea54da, Sonnet) in a new tree.
- **onelook** abebe6d27b2d052aa (Sonnet, in .tw-trees/onetrack, 6185): one-track station drawn thinner with a bulge
  on the platform side, per her mockup (my scratchpad images\3.png). Land onetrack only after it reports.
- 11-12 ~16:00: v0.3.3 ships with two pairs HIDDEN (needs a flag hiding the box; give to onelook or the T-3357
  batch); quad station T-3363 folded into T-3348; future station editor direction relayed to aabb (DECISIONS).
  One-track look: she wants it a bit wider (more depth on the platform side, her mock in my scratchpad images\4.png):
  resume onelook abebe6d27b2d052aa in the MAIN tree after onetrack lands.
- 16:20 landed 5f41930e quadfix, 91162cad onetrack (save 18), f095231b gap batch A + NK registered; smoke PASS 28;
  snapbuild f095231b + push running; then "go join" to gaprun2 aa8d3bc594c20a4d8 (handed back at clean pause; resume
  by SendMessage; its kit + steps in notes/T-3287.md Stage 4).
- Main-tree app agents now: **onelook** (wider one-track look, two-pairs box behind `?pairs`, blueprint track drawn
  over blueprint stations), **t3357** a475d39b77ea321ab (Sonnet: pill out, "turn around", side hover dropped, T-3361
  follow a train), **t3329** (memory B + A). Commit each as it reports, smoke, snapbuild.
- 8800 = f095231b (city-turn, two-pair complex fix, one-track stations, gap A + NK registered but not joined),
  pushed. "go join" sent to gaprun2 ~16:40.
- **schem2** ad078e33a689dbd26 in .tw-trees/schem3 (6175): landing T-3362 Florida strip + T-3364 sea only where the
  ground has it (my call), then T-3358 rivers. Bigger centres (p 0.8 vs 1.2) waits for her pick:
  review/schematic-centres-london.png.
- ~16:50 her calls: one-track strings acknowledged; bigger centres yes (p 0.8; told schem2, job 3 after rivers);
  AA/BB 2.2x, no platforms on doubling, tracks per line and stop, direction colours red/blue in the station picture,
  plan for many parallel tracks; she's tired and confused about what quad IS -> T-3365 design (**t3365**
  a68eaf5cfd85988df, Opus, read-only, short plain notes for another day). AA/BB waits for T-3365.
  T-3365 DONE (notes/T-3365.md, review/track-model.png): quad not stored today (two doubles found by distance, four
  different distances), icons mean different things; model: simple = doubles, advanced = every track has a way; quad
  becomes one remembered thing (8-12 h) before AA/BB; 5 calls sent her for another day.
- onelook DONE, committed 74657141 (wider one-track look, blueprint track over blueprint boxes, two-pairs behind
  ?pairs). DECISIONS line for hiding is in the uncommitted DECISIONS (commit with the next round). Not yet on 8800.
- **tt1 tree READY to land** (T-3346 stage 1, agent finished): schedule panel Simple mode, departures board, chart,
  group chart; sim query station_departures (WASM rebuild on land); check 16/16, smoke 28. Overlaps main-tree app/
  edits (widgets.tsx, style.css, en.json, context.json, LineInspector): land AFTER t3357 and t3329 report and are
  committed. One line to change when tt2 lands second (notes "Stage 1 as built"). 5 strings sent her.
- T-3348 AA/BB design DONE (notes/T-3348.md; 27-39 h, 5 stages; her model + per-direction stops; 7 calls sent her;
  review/aabb-station.png, aabb-layout.png).
- **aabb** a12d41a05c626764b (Opus, read-only): T-3348 AA/BB quad design for her (her model: double in the middle + two one-way
  singles outside, smart extension, station outer/inner/both). notes/T-3348.md.
- T-3364 Nottingham-as-island (sea drawn round an inland network) added to schem2's read-only jobs.
- **tt1** a2276e5f4caba2ed9 (Opus, .tw-trees/tt1, 6180): T-3346 stage 1 panel (Simple mode, board, chart).
- **quadfix tree READY to land** (T-3356 fixed; agent aebad7c10cd823406 done): partner = nearest station of the
  stop's complex on the track beside (sim computes complexes itself, `Network::complex_of`), sim 547 / demand 731,
  t3333-pairs 16/16, smoke 27/28 (t1053 from copied registry, reverted). Sim change: WASM rebuild on land. Shots
  t3356-before/after.png. Quad station = T-3363 (3-4 h): resume quadfix agent after onetrack lands, in a new tree.
- Landing order after the gate window: schem, onetrack, quadfix (twin.rs hunks separate), then tt2/tt1 as they
  report; smoke; snapbuild; push.
- t1156-all-china.save is no reference (DECISIONS).
- Her TODO short-term 11-12 ~15:00 filed: T-3359 China snap (dropped: she tested it fine on 8800; agent stopped),
  T-3360 schematic bus/ferry settings and T-3362 Miami land strip (both after T-3345 stage 2,
  fresh agent; t3345 is deep in context), T-3361 follow a train, and the running-side hover drop folded into T-3357.
  After the gate window: one Sonnet in the main tree for T-3357 + T-3361; t3329 resumed for the memory build.
- Later 11-12: two-pair partner = any station in the same complex (not 4-16 m); "place quad track station" and
  one-track stations (T-3332) must ship together with two pairs (v0.3.3 waits for T-3332): told quadfix to scope/build
  the quad station. Server deploy: HOLD until v0.3.3 ("if it doesnt do anything now"). Memory limit 8/6 GB red 85% OK.
  Bigger city centres: try, as a mock after city-turn, plus distortion a notch higher ("a tad low"). Screenshot text
  acknowledged.
- Worktrees now: onetrack, tt2, quadfix, schem, hotfix031 (5; over the usual 2-3, disk checked).

## Waiting on Anita
- Schematic (T-3345 stage 1 done, notes/T-3345.md): approach 1 for country maps only as stage 2? the all-Japan shape
  change (artifacts/trainworld/t3345-mock-japan-real.png)? try approach 2 (bigger city centres)? OK that the agent
  downloaded 23 public "let others open a copy" Community saves to its scratchpad? First image
  artifacts/trainworld/t3345-mock-beijing-player1.png.
- Timetables (T-3346, revised to her panel idea; notes/T-3346.md, mock artifacts/trainworld/t3346-panel-mock.png):
  her 6 calls (averaging into the 5 periods, no warning; whole hours, 6 rows/day; Sat+Sun together; meets stage 6;
  express behind local; the draft strings). Then stage 1 (panel + board + chart) and stage 2 (per-train waits).
- Text acks: none open (T-3326 acked 11-12).
- Later: T-3264 trains drawn through each other (mock).

## Queued (filed)
- T-3337 HSR top speed 350 and T-3352 single HSR/tram: after T-3332 lands (same track code).
- T-3335 [P3] two-pair panel parts, T-3343 [P3] per-train holds (timetable stage 2 absorbs), T-3348 [P3] quad
  platforms, T-3353 [P3].
- T-3320 Berlin RE5 (P3), T-3322 t2266 check (P3), T-3325 Sendai rebuild (P3).
- Before v0.3.3: translation batch for all new strings; website about-page flag credit line (notes/T-3328.md);
  server deploy first.

Next free id T-3364.

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
