# Orchestrator handoff, trainworld

Orchestrator: session 8509086d-46cd-4c5f-8d59-da9348f4a5e0, name maps-c9, since 2026-11-03 23:29
Updated: 2026-11-03 23:29

Read trainworld/CLAUDE.md and TASKS.md first. Landings are in ARCHIVE.md and `git log`.

## State
- master **8dc8961f** (maps-1c, ~01:30: T-2214 + T-2215 islands demand and No route landed, tree t2214 removed;
  before it af7a4271 T-2257 Rikers fix + ferry " bus" label fix). maps-1c finished winding down ~01:35. Today under maps-1c (oldest first): 6587f89d (T-2186 150% load, T-2180, T-2237, T-2226/27,
  T-2192 ferry router, T-2222/23 band looks behind `perfTry`), 5cd94db2 T-2193 ferry sim, 455bef52 T-2231 passing
  sim, a1d96f2f T-2205/06, 736d8e88 T-2194 Ferry tool, a2ffbef1 T-2217 buildings sim (save LEVEL 4), 26d78ae5 T-2162
  diagram names, e715556b T-2232 passing panels, 5a7a76e5 T-2218 building tool + T-2233 passing stations drawn,
  eb57aad3 T-2163/T-2235 diagram clicks, 27cad2f7 T-2236 crowding by load + T-2219 building drawing, 6329c909 T-2195
  ferries in demand, 5249035a Anita's text answers, 04dd3ef5 **India T-2111** (782/782 rows), 65562660 T-2250 tests,
  6a1b4f2a CLAUDE.md demand-tests rule, d3ad2f27 T-2196 ferry fans, 15034a96 T-2220 0.8 rule (fixed minutes only),
  6cd0a31d T-2243/44 diagram folds and hooks, 9ebc8fed **T-2213 land parts in every pack**, 6499883c T-2238/T-2230
  edit speed, then paperwork.
- **8800 = build 867 (8dc8961f)**, built from data/packs directly, smoke 23/23 light and dark; real save 11,721,348
  rail trips (islands: -84). LIVE = 821 (R2 `trainworld/2026-11-03b`, website dc49c5f). Publish only on Anita's word.
- Freeze: data/frozen/20261103-162459 (post-India). Build 8800 on it while an agent writes data/packs.
- Uncommitted in the main tree: running agents' files; TODO.md (Anita's).
- **Cores cap 10** (Anita at the machine, 23:29); `python cores.py cap 15` when she is away.
- Disk: ~46 GB free. A watch (scratch lowdisk.ps1, re-armed ~20:00, 6 h) reports under 3 GB.
- Worktrees: .tw-trees/t1734 (kept, T-1734) only.

## In flight under maps-c9 (main tree; all told to leave t2214's files and t2257's pipeline files alone until t2214 lands)
- **T-2263 tutorial DONE, uncommitted** (agent a12a2687d7e5186cc, retired): game/guide.ts, game/ui.ts, map/trackTagRows.ts,
  game/today.ts, ui/ripple.ts, ui/Guide.tsx, new ui/guide.css, ui/DemandSelector.tsx, checks/t2123-tutorial.mjs,
  docs/ui-panels.md, DECISIONS, ARCHIVE, TASKS (T-2280 filed). Smoke 23/23 and t2123 pass on 6170. Commit after
  maps-1c finishes. After t2214 lands: docs/ui-demand-views.md 70-71 "every row shows folded" -> "the station
  ridership and track colour rows show folded"; code-map.md 483 t2123's new order. Strings shown to Anita 23:5x.
- **Cores cap 6 until ~00:50** (Anita: keep 4 cores free for an hour; machine was at 97% with her League client).
  Then `python cores.py cap 10`.
- **T-2267 + T-2270 design DONE, uncommitted** (agent a17a59b5769d7736e, retired): notes/T-2267.md,
  research/tool-options-and-modes.md, stages T-2282..T-2285 filed (after her OK). Shown to Anita ~00:10 with
  review/track-bar.png and mode-colours.png; waiting on her OK. Dark/station/phone mockups skipped for cores
  (scratch t2267\mock.mjs makes them, ~5 min on 3 cores).
- **T-2266 fares lag DONE, uncommitted, smoke NOT run** (agent aad069e13aaf224e6, retired): new game/fareInEffect.ts,
  game/money.ts, game/regionMoney.ts, main.ts (wraps demandClient's private daysOf: stand-in until T-2286 [P3, after
  t2214] stamps fares in daysOf), checks/t2266-fare-lag.mjs, SPEC fares line, DECISIONS, ARCHIVE. Real save: income
  held at 148.7M/h through a 10x edit, 270.5M at the 5x final; a cluster final can still overshoot before the full
  final (419M, as before). Run smoke before committing. code-map line to add after t2214 lands: "`t2266-fare-lag.mjs
  [url] [save] [out.json]` (the real save: fares to 10x then 5x while solving; income held at the old fares until
  riders solved with the new ones show, and ends at the new figure with every line priced at 5x; port 6172, 2 demand
  workers, about 5 minutes; T-2266)". Proposed hover (asked Anita): "Riders pay the old fares until demand has settled".
- a40a9cd5aa265e0ef **t2290** diagram: Fleetwood drawn east of Port Chester so line 8 loops round it (Opus; rescoped
  after Anita's correction), app/src/diagram/*, Vite 6173, id T-2293.
- Queued for after t2214 lands (they touch its files): T-2271 small UI batch, T-2278 parking by income, T-2265
  walking rule, T-2276 buildings 1 min + waiting room, T-2279 crowding dot (show her first), T-2273/74/75 ferries and
  stop list, T-2269 blueprint breakdown (her answer filed: switch stretches and whole connected parts off/on).
  Diagram: T-2168 now "faint city outlines", T-2169 water "wider than a certain width", T-2289 condense lonely
  intercity stretches, T-2290 Fleetwood on the wrong side (agent t2290, above), T-2291 line colours avoid near-duplicates (P3), T-2292
  parallel-bundle idea (P3). T-2288 is maps-1c's (airport badge hover); T-2287 offered to it. Next free id T-2294.

## In flight under maps-1c, its to land (all Opus, its subagents)
- **T-2214 islands demand: DONE in tree t2214, held unlanded.** Sado test ferry: 55 by ferry, 2,475 no way, none
  car/walk; Athens/Kagawa/HK within 9%; solve +1.7%; both cargo runs and smoke 23/23 in tree. Held until T-2215 and
  T-2257 are ready (else island trips vanish from mode bars, and Rikers' 6,379 NYC jobs get no car).
- a6a17775cc9b4d02c **t2215** "No route" display, working IN TREE t2214 (Vite 6196). Land T-2214 + T-2215 together:
  rebase t2214, both cargo runs, wasm, typecheck, smoke in tree, `land t2214`, remove the tree.
- ad70096d7069a28aa **t2257** Rikers Island false split: main tree pipeline, writes data/packs (backup
  data/work/pre-T2257/). Land its packs before t2214.
- a93f41b475e25b962 **t2255** diagram on the game's own layouts (hairpins at 125 St, Woodhaven, hook at Grand
  Central; checks on the game's graphs). Main tree, Vite 6194.

- a7a0c49c5970ac78a **t2222b**: T-2223 bandFine as the look (her pick), T-2258 the rounded prong in the curve
  picture, T-2222 the band snap on her export (artifacts/trainworld/anita-2026-11-03e/tokyo-day-414.save, default
  zoom, Yamanote north of Shinjuku). Main tree renderer, Vite 6189.

- 23:00: c7066737 T-2255 (engine-independent diagram maths, checks on the game's graphs; T-2259/60/61 filed); 8800
  build at c7066737 on the freeze, **8800 = build 862**, smoke 23/23 light and dark. **T-2215 DONE in tree
  t2214** (check t2215-no-route 22/22, smoke 23/23; strings to show Anita; its ARCHIVE/TASKS/DECISIONS lines are
  already committed). Land t2214 after t2257 reports. After landing: fix "Cheung Chau ferry bus to" (RouteParts.tsx
  ~62 adds " bus" to ferries) and file the airport badge hover leaving No route out.

- ~23:40: de944172 T-2222/T-2258/T-2223 (8 m minimum stroke segment ends the snap and the prong; bandFine is the look). **8800 = build 863**, smoke 23/23 light and dark. T-2262 filed (float32 positions). Only t2257 still running; then land t2214.

## Next
1. When t2257 and t2215 report: commit t2257, land t2214, rebuild 8800 (plain snapbuild once nobody writes packs).
2. T-2216 the market across parts (rebuild the 901 zone files then; Sunshine Coast lie, notes/T-2213.md), T-2256.
3. Turkey (T-2079) when compute is spare (Anita away = cap 15).
4. Diagram: T-2242 close pairs, T-2246 room for names, T-2247 names speed, T-2168 boxes, T-2169 water.
5. Passing tracks: T-2245 local waits check, T-2248, T-2249, T-2239/40/41. Station buildings: the offered
   rectangle misses same-named stations (notes/T-2208.md follow-up; no id yet). T-2251 uji. T-2209 canals.
6. Follow-ups not filed yet: diagram picture tiling for all-Japan, demand loading card over the diagram, NYC's second
   "Times Sq-42 St" complex (notes/T-2163.md); StrokeSender patch on moved runs (notes/T-2238.md); drop the unused
   platform-wait numbers in Rust `crowding()` (T-2236); a trouble-list row past ~140% load.

## How to
- Scratch = C:\Users\anita\AppData\Local\Temp\claude\c--Users-anita-projects-maps\8509086d-46cd-4c5f-8d59-da9348f4a5e0\scratchpad
  (snapbuild.cmd, snapbuild-frozen.cmd, keep_hunks.py, mark_taking.py, lowdisk.ps1). A new orchestrator should copy
  these to its own scratchpad and fix the path inside snapbuild-frozen.cmd.
- 8800: `cmd /c "<scratch>\snapbuild.cmd <hash> > <scratch>\snapbuild.log 2>&1"` (or `snapbuild-frozen.cmd <hash>
  20261103-162459` while packs are being written). The build runs tsc on the committed tree. Then smoke on
  http://localhost:8800/trainworld/ light and SMOKE_SHADE=dark. Build number = `git rev-list --count <hash>`.
- Pre-commit smoke: own Vite (`node trainworld/checks/dev-nowatch.mjs 6197`), warm one page load, then smoke; stop it
  by its port's PID.
- Landing a tree: `git -C <tree> add --all -- trainworld`, commit "agent-tree: work in progress", `rebase master`,
  resolve, `rebase --continue`; run both cargo runs (`--features demand --lib` too), `npm run wasm`, typecheck and
  smoke in the tree; then `agent-tree.mjs land <name> "<sentence>"` and `remove`. A main-tree uncommitted edit to a
  file the branch touches refuses the fast-forward: commit it first, or set it aside (`git diff --output`, checkout,
  land, `git apply --3way`, `git reset` the file).
- Mixed files: `python <scratch>\keep_hunks.py <maps root> <path> <marker>...` (-U0).
- Publish: `freeze.py build <stamp>` with TW_MARKET_DRIVES set, `deploy.py --dist <dist> --version <v> --smoked`, website
  `git add trainsworld/ trainworld/`, commit "anitabuilder", push (her word), `deploy.py --verify --page --version <v>`.
  Afterwards prune publish/ and data/frozen/ to the current build and the latest published ones.

## Open questions to Anita (asked, unanswered)
- Friend playtest (notes/todo-2026-11-03h.md, filed T-2263..T-2272): asked about default train length 5 cars, moped parking in poorer countries, line crowding red as average vs peak, where the rail cost multiplier goes, and blueprint segments enable/disable.
- Strings to acknowledge: Ferry tool (T-2194), station building tool (T-2218), crowding (T-2236), diagram (T-2163),
  ferry demand rows (T-2195), building hover "Changing trains inside it feels a little quicker." (T-2220).
- Answered 22:30: buildings fixed minutes only OK; gap look = fine; band snap at default zoom, Yamanote north of Shinjuku, export copied. Earlier: islands go, word "No route"; edit note and everywhere "trips/day"; rates "per"; period on the
  passing hover; On board hover cut; temp files always delete; keep only current and latest published builds;
  passing tracks go (400 m price, every station); crowding from 70%; capacity 150%; ferries go (ports only, "No water
  path found"); named crossings and maglev yes; diagram for everyone.

## Lessons from this run
- Before starting an "Open" task, `git log --grep "T-<id>"`.
- Her TODO.md can repeat an earlier batch word for word: diff against notes/todo-*.md before filing.
- Commit as agents report when their files are separable; untangle shared docs with keep_hunks.py (-U0).
- Two worktrees that both bump the save level collide; any new save block now goes after BLD1 at LEVEL 5.
- Trees touching worker files (protocol.ts, clock.worker.ts) can't fast-forward while a main-tree agent edits them:
  land trees right after a main-tree commit.
- `cargo test` without `--features demand` skips the demand kernel's tests (two broke unseen, T-2250).
- C: can fill suddenly (0.5 GB free at 13:00 with a join running): check `Get-PSDrive C` before long runs.
- Agents near 450-550k tokens: retire after their report; give follow-ups to a fresh agent.
- maps-b2's and maps-0f's subagents can't be messaged from here; go through their session.
- Her Tokyo save file: C:\Users\anita\Documents\trainworld\Tokyo Seoul Fukuoka Busan (from anita.garden).save (read
  only, older than her live game).

## Anita's standing asks
- Player text: agents propose wording per the style rules, list every string old -> new; land without waiting; show
  her. Zone, region and realm names need no review.
- TODO.md short-term = prompts: copy her words verbatim into briefs; save batches in notes/todo-<date>.md.
- Every demand task has a "done when". Buses settled. Thanks to agents in my own reply. Keep the work moving.
- Disk: delete orphaned temp files always; keep only the current build and the latest published ones in
  artifacts/trainworld/publish/ and data/frozen/; delete superseded backups once a later state checks clean.

Next free task id: T-2273 (unused but handed out: T-2173/74, T-2187..2191, T-2197, T-2224/25, T-2252..54; check before use).
