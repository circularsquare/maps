# Orchestrator handoff, trainworld

Orchestrator: session a67ba911-f213-4096-86ff-b9c1620cbea0, name maps-37 [07bc40], since 2026-11-12 17:54 UTC
Updated: 2026-11-12 ~23:45 UTC (~18:45 local; she is awake, focused on v0.3.3). Context past 500k: ready for a takeover.

Read trainworld/CLAUDE.md and docs/orchestrator.md first. Earlier log: handoff.md in git history.

## Her focus now
**v0.3.3 only** (Anita 11-12: "just wanna focus on 0.3.3 for now and reduce scope of things im thinking about").
AA/BB (T-3348) and the track model (T-3365) are PARKED with her answers recorded in their notes and DECISIONS: do not
raise them until she does. Open asks to her: the v0.3.3 cut below, and publish timing (tomorrow daytime, so the
overnight gap runs make it in, or sooner without them).

## Live, server, ops
- **Live = v0.3.2** (tw/release-031 6425f2f5, .tw-trees/hotfix031; data 2026-11-11b; website 966f101). The v0.3.2
  changelog is on master (08c27589).
- **Server** shonei-server e878c0f deployed. Committed, NOT pushed or deployed, HOLD until v0.3.3 (her word): 7a635f7
  + e5a132a (T-3326 feedback screenshots; config flag hides the app control until deployed), 306fbe3 + c3403aa +
  b2ab066 (flag set 3, 336 codes); plus the play-stats browser cap (she OK'd earlier). Deploy before the app ships.
- **Ops**: check_resources.py daily (last 11-12 16:04Z fine); feedback.py inbox; once the public repo exists, a daily
  `gh pr list` / `gh issue list` on circularsquare/trainsworld (she OK'd).
- **Open source**: export tool tools/opensource/export.py, local copy C:\Users\anita\projects\trainsworld-export.
  At v0.3.3: re-export, `gh repo create circularsquare/trainsworld --public --source . --remote origin --push` from
  that folder (she said yes "once 3.3 is done"; confirm that day), then sync at each publish (orchestrator.md).

## 8800 and master
- **8800 = ac72df8d**. **Master c905e498, pushed** (memory card wording on top of 8800). Last smoke on master
  ac72df8d PASS 28 (my Vite 6139, still running, mine).
- Today's landings, in order: 110d7700 T-3339 quad/passing -> single, 95082088 T-3333 two pairs, bd8894d0 flags +
  gap names + gate fix + export, 47324b04 T-3344 joins, 08c27589 v0.3.2 changelog, a15deda8 T-3351 six flags,
  d78d1d16 T-3326 screenshots form, 1595924e T-2776 race fix, 85fdcb6e T-3345 city-turn, 5f41930e T-3356 two-pair
  complex partner, 91162cad T-3332 one-track stations (save level 18), f095231b gap batch A + NK registered,
  74657141 one-track wider look + blueprint track over stations + two pairs behind `?pairs`, 92ff0df9 T-3329 memory +
  T-3361 follow a train + T-3357 pill/around/hover, ac72df8d T-3346 stage 1 schedule panel, c905e498 memory card text.
- Uncommitted main tree: pipeline/world_plan.json (gaprun2's join; commit at its ship), TODO.md (hers).
- Worktrees: tt2, schem3, hsr, hotfix031. Stray Vites 6164, 6167, 6130 (owners unknown) left alone.

## v0.3.3 plan (proposed to her 11-12, awaiting OK)
In: everything on master (two pairs stays hidden behind `?pairs`); T-3352 single high-speed/tram + T-3337 350 km/h
(hsr tree); schematic T-3362/T-3364 + T-3358 rivers + bigger centres (schem3) if ready; gap zones A + NK (+ B, C if
done). Held for v0.3.4: T-3346 stage 2 (changes figures) unless it lands clean (<2% rider moves).
Before publish: translation batch for every new string since v0.3.1 (English-only keys: memory, follow, board,
one-track, screenshots, flags...; stale: sideSwitch.ask/reversed, issues.buildingNoStation); v0.3.3 changelog entry
(her text OK); server deploy (her word); website about-page flag credit (notes/T-3328.md); smoke8800 light + dark;
deploy.py with a new data version (gap packs); open-source repo.

## Running agents (all maps-37's)
- **gaprun2** aa8d3bc594c20a4d8 (Opus, main tree + sole data/packs writer): T-3287 join ES,PT,FR,RO,BA,GR,HR,AL,KP
  running (python pid 21260, claim gaprun-join 6; ~3-3.5 h from ~16:40 local), then after-join steps (stale.py,
  zone_origins, terrain, drives, build_inputs, smoke, real save vs 12,581,682 rail trips), then batches B (NO SE FI EE
  LV LT) and C (NW Ontario). Gates: it messages before each and waits for my "go gate"; then no main-tree app/ writes
  or landings until it says the window ended. Notes/T-3287.md Stage 4. Snapbuild while the join writes: hard links to
  data/packs (consider snapbuild-frozen; the freezes in data/frozen predate batch A).
- **tt2** a8e910f825e247d9e (Opus, .tw-trees/tt2, 6160): T-3346 stage 2 per-train waits + express behind local;
  build the two interface points (Trip.var only; departure lists). When it lands after tt1 (landed), change the one
  line in station_departures (notes/T-3346.md "Stage 1 as built"). Reports rider moves.
- **schem2** ad078e33a689dbd26 (Opus, .tw-trees/schem3, 6175): T-3362 Florida strip + T-3364 sea only where known
  BUILT (check t3364-off-sheet-water.mjs; sheetMap +0.4 s NYC, +1.7 s North America); now T-3358 rivers (stations on
  the correct bank, then join rivers, check t3358-river-sides, 0 wrong for NYC/London); then bigger centres p 0.8
  (her pick). Land the tree when it reports. t2169-water Tokyo arc 68 fails on master too (pre-existing).
- **hsr** a7db66e57cfea54da (Sonnet, .tw-trees/hsr, 6197): T-3352 single high-speed/tram (my calls: 0.6x price,
  street tram, two-way single HSR, Select retypes, no loop at tram stops, One way on single tram), then T-3337 HSR
  350 km/h. Reports after each.
- Finished, resumable by SendMessage: t3329 a3bb12c205199e3b6 (memory), t3357 a475d39b77ea321ab, onelook
  abebe6d27b2d052aa, quadfix aebad7c10cd823406, aabb a12d41a05c626764b, t3365 a68eaf5cfd85988df, t3346 design
  a745e16f3053f63d6, t3326 a819255edd5716af0, t3351 a0db4967a90cee41a.

## Queued (filed)
- T-3360 schematic bus/ferry settings (after schem3). T-3346 stages 3-6 (Detailed, weekends, meets, Exact) after
  v0.3.3; her round-3 calls at the top of notes/T-3346.md and DECISIONS.
- Parked: T-3348 AA/BB (27-39 h, waits for T-3365 cleanup 13-17 h). P3: T-3335, T-3343 (stage 2 closes), T-3353,
  T-3354, T-3320, T-3322, T-3325, T-3264.

Next free id T-3366.

## Her calls 11-12 (detail in DECISIONS, ARCHIVE, notes)
Areas: single-track joins, one-track stations, timetables design, schematic (she reviews), open source. Flags: all 17
Spanish communities, Bali, Yogyakarta, English names. Timetables: three modes, Custom weekends, whole hours, averaging
into 5 periods, nudge 1 min / Shift 15 s / Ctrl 10 min, first come first served, distance-scaled chart, train grid in
Exact. Schematic: city-turn, Japan shape OK, Community saves OK sparingly, rivers: stations off water and on the right
bank, no small rivers, bigger centres p 0.8. Two pairs: partner = same complex; ships hidden in v0.3.3 with quad
station + AA/BB later together. One-track look: thin bar + bulge, wider. Blueprint track over blueprint stations.
Pill out, "turn around" rule (CLAUDE.md), side hover dropped, follow a train. Memory: Settings reading + card "3.1 GB"
with hover "Uses about 3.1 GB of memory" + warning. Server deploy held to v0.3.3. China snap dropped. t1156-all-china
is no reference. All strings acknowledged so far.

## Anita's standing asks
- **No deploys overnight** without her there; every server deploy and publish needs her word.
- Cores: the ledger's cap (`python cores.py list`).
- **Usage**: Opus more widely but "not fully gung-ho"; where there's a choice, spend compute, not tokens.
- Watch whether agents use wiki-first reading well.
