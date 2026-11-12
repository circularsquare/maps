# Orchestrator handoff, trainworld

Orchestrator: session a67ba911-f213-4096-86ff-b9c1620cbea0, name maps-37 [07bc40], since 2026-11-12 17:54 UTC
Previous: session 3397f27a-d2e0-438d-a3c0-8538d4d84f99, name maps-cb [f61388], handed over 2026-11-12 17:54 UTC, finishing in-flight work
Updated: 2026-11-12 17:54 UTC (12:54 local; she is awake and answering).
(Times in the earlier part of this session's notes ran 1-2 h ahead; these are checked with `date`.)

Read trainworld/CLAUDE.md and docs/orchestrator.md first. Earlier log: handoff.md in git history.

## Live
- **Live = v0.3.2** (tw/release-031 6425f2f5, worktree .tw-trees/hotfix031; data 2026-11-11b; website 966f101;
  published 17:55Z 11-12 by maps-cb, verified). ..\website's only leftover change, pages/projects/abouttrainsworld.md,
  is not ours. TODO after the gates: copy the v0.3.2 changelog entry (V0_3_2_AT "2026-11-12T18:00:00Z") and its
  translations from 6425f2f5 to master (updates.ts + i18n messages/sources/context). Next big update is v0.3.3.
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

## Split with maps-cb (from 17:54Z)
maps-37 owns commits, landings, 8800, pushes, publishes and data/packs scheduling. maps-cb starts nothing new and
relays its four agents' reports (below) to maps-37; follow-ups become fresh maps-37 agents from the task notes.
Exception: maps-cb is doing the v0.3.2 publish itself (her yes was in its tab); hands off ..\website when done.
T-3333 strings acked by her ("yes two-pair strings good").

## maps-37's agents
- **t3329** (Sonnet, main tree read-only, Vite 6121, claim t3329): T-3329 measure save size vs memory, propose a
  display. Id T-3341.
- **t3326** a819255edd5716af0 (Sonnet): T-3326 stage 1 DONE: shonei-server 7a635f7 + e5a132a (not pushed, not
  deployed; config flag trainworldFeedbackImages hides the app control on an old server), feedback.py --shot saves all
  (main tree, uncommitted). Plan notes/T-3326.md. Stage 2 (app form, drag-drop yes, inbox thumbnails yes, "screenshots"
  wording given) waits for the gates' end: message it. Strings to her once stage 2 reports. Server deploy needs her
  word, before the app ships. flags3's flag-set 3 edit breaks TestFlagSetTwoPlaces until flags3 fixes it (relayed).
- **t3317** a85b089dcb8998843 DONE (T-3317 closed): live release-031 predates T-3202/3286/3297/3296, so HSR >4 cars
  was already fixed (T-3078, live) and loopo's platform fix ships with v0.3.3. Filed T-3344 [P2] singles-then-double
  ends left unjoined (draw.rs end_joins, 3-5 h, no collision with quad2/dbl3; scratch tests in my scratchpad
  t3317\sim\src\track\wasm\lanes_tests.rs zz_*) and T-3343 [P3] per-train overtake holds. Loop not recognised: not a
  bug (T-3140), my call to leave it.
- Her picks 11-12 13:20 local: "single-track join fix and one-track stations, and timetables design"; schematic "take a
  stab ... i definitely want to review before we put anything in"; open source "somewhat soon".
- **t3344** a5e962a7f0196cdfa (Sonnet, main tree sim/ only, draw.rs): T-3344 fix; no app/ writes or `npm run wasm`
  until I message it that gates ended. Reports at Rust-done and at end. Vite 6123. Id T-3347.
- **t3346** a745e16f3053f63d6 (Opus) DONE: design in notes/T-3346.md (shape 1: game-written per-train timetable,
  boards/chart, nudge, five rows, weekends, single-track meets; 6 stages 43-64 h). Filed T-3348 [P3] quad platforms.
  Her answer 11-12 ~14:00: likes it; new-player experience unchanged, everything in an advanced scheduling panel
  (centre, like community); demand stays 5 fixed periods; free intervals with a cap, eat the solve time with a
  warning (I asked t3346 whether time-weighted averaging avoids it); weekend rows, a multiple, one-click weekends
  only/off; rows "M-F 6-9", day colours. t3346 resumed to revise the design + panel mock. Then stages to agents.
- Open source: she OK'd name/email, README, the PR method, and "lets rebuild and create and push once 3.3 is done
  yes" (11-12); sync at each publish (orchestrator.md). At v0.3.3: re-export, `gh repo create circularsquare/
  trainsworld --public --source . --remote origin --push` from ..\trainsworld-export (confirm with her that day).
- **t3345** af6237aa0c40ccfc1 (Opus, no app/sim edits): T-3345 schematic stage 1, failure catalogue + approaches +
  before/after mocks for her review; nothing lands before she reviews. Vite 6125. Id T-3349.
- **t2913** aa6a492085b82dfc3 (Sonnet) DONE export stage: tools/opensource/export.py (from `git archive HEAD`, so
  rerun after landings), C:\Users\anita\projects\trainsworld-export commit 4485c89, 64 MB, builds from clean clone
  and loads data from R2. Notes/T-2913.md "Export stage". Sent her: name/email on commit + copyright, README read
  (drop "and timetables" until they exist), publish command `gh repo create circularsquare/trainsworld --public
  --source . --remote origin --push` (hers). My calls: ODbL line, Windows paths left, PR flow as written, CI later.

## Landing queue (after gaprun's gates end)
gaprun 11-12 ~13:45: batch A stopped, nothing shipped. Every world-run browser gate has failed since 2946d5e5 (11-10):
checks/city-browser.mjs waits for "Train...Walking...Driving", the hover now says "Transit" (gaprun fixes it); and
market_pack panicked on old joint pair files naming rebuilt zones (moves them to data/work/retired before a row
registers). It restores preA, idles and releases cores: that is my landing window; I send "go gaprun" via maps-cb.
~10 h of batches still to go.
- **dbl3** (T-3339 DONE, dbl2 finished): uncommitted in .tw-trees/dbl3 (tw/dbl3 at master 44f6a621). Sim + app +
  WASM. Main-tree bookkeeping (TASKS line removed, DECISIONS, ARCHIVE) uncommitted: commit with the landing. Smoke 28
  PASS on 6170. Strings: two removals only, told her. Old Vite on 6167 from 11-10 (owner unknown).
- **quad2** (T-3333 DONE, onestop finished): uncommitted in .tw-trees/quad2 (tw/quad2 at 44f6a621). Sim + app +
  WASM, save level 16 -> 17. Main-tree bookkeeping (DECISIONS x2, TASKS line, notes As built, T-3335 P3 filed)
  uncommitted. t3333-pairs 15/15; smoke 27/28, the fail is gaprun's working registry.json (passes with the tree's):
  smoke master after landing. Strings acked. Her test on 8800 after: quad, line on one pair, stations beside each stop
  on the other, tick "Run trains on two pairs of tracks"; picture artifacts/trainworld/t3333-3-trains.png.
- Then T-3332 one-track stations (notes/T-3332.md, 11-16 h): fresh maps-37 agent (Opus) in a tree from master after
  dbl3 + quad2 land; sequence against t3317's fixes when it reports.
- **flags3** (T-3328 DONE): main tree uncommitted, all its: app/src/game/flags.json, app/public/flags/* (21 new SVGs +
  the old ones' deletions), tools/flags.mjs, checks/t3279-flags.mjs, notes/T-3328.md + line in T-3269.md, DECISIONS
  line, ARCHIVE entry, TASKS line removed. shonei-server 306fbe3 + c3403aa (flag set 3; go test passes), with t3326's
  7a635f7 + e5a132a: one server deploy on her word covers both. Website about-page credit line: text in notes/T-3328.md
  (with v0.3.3). Her answers 11-12: redraw the 4 missing communities simply, add Bali + Yogyakarta simply, names stay
  English -> T-3351.
- **t3351** a0db4967a90cee41a (Sonnet): T-3351 six flags; drawings + shonei-server commit (into undeployed set 3) now,
  app/ copy-in only when I message after gates. Vite 6126. Drawn (review/flags-six.png sent to her); shonei-server
  b2ab066 (set 3 = 336 codes). Waiting on my gates message.
- changelog v0.3.2 copy from 6425f2f5; t3326 stage 2 (message it); t3344 wasm + browser (message it).
- Only gaprun is still maps-cb's.

## Running agents (maps-cb's, relayed; maps-5e finished)
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
- Text acks: T-3326's screenshot strings after stage 2 (T-3339's acked 11-12).
- Later: T-3264 trains drawn through each other (mock).

Next free id T-3352 (T-3345 schematic, T-3346 timetables, T-3351 flags filed 11-12).

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
