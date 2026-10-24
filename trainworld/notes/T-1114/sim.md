# T-1114 review: sim/

Read-only review of `sim/` (the Rust crate) on the main tree as of 2026-10-24 (master 54005ec plus
uncommitted work). Line numbers are that tree's; a few may be off by one. Nothing was built.

## Summary

24 items, ranked by value against risk, in 8 batches. About 1,100-1,300 lines could go outright
(dead code, retired tools and their diagnostics, copied helpers, the fare curve if T-1112 makes it
unneeded), and about 9,000 lines of tests could move out of the five biggest files.

The five best:

1. **Dead code and the T-001 `Sim` skeleton** (items 1-3): about 135 lines nothing calls. No risk.
2. **Test-only and tool-only methods in `#[wasm_bindgen]` blocks** (item 5): about 30 methods the
   app never calls are compiled into the two WASM modules. Move them to plain `impl` blocks.
3. **Retire the `realnet` and `zones_day` bins** (items 6-7): `calib` replaced `realnet`. With it
   go four `DemandApi` readers and the walk-sum diagnostic that `mode_choice` builds inside the
   commute pair loop (outputs only, so no result changes). About 650 lines.
4. **One loader for a real-network JSON, and one `City::subset`** (items 8-9): the same 20-40
   lines are copied into 3 bins and about 12 tests. About 250 lines.
5. **The fare curve, if T-1112 lands as a two-node straight line** (item 12): `FareCurve`,
   `trip_km`, `rn_km`, the skim key's curve fields and three `set_fare_curve` exports. About 150
   lines in the sim and T-887 becomes moot.

Close behind: move the inline test modules out to their own files (item 17), split `track/wasm.rs`
(item 18), and fix the comments that call the walk-lost blocks "legacy zero" when they are not
(item 15).

How "unused" was checked: scripts in my scratchpad (not kept) counted every identifier in sim
library code, bins, tests, `app/src` (generated `app/src/wasm/` left out), `checks/`, `pipeline/`
and `tools/`, with test code found by brace-matching `#[cfg(test)]` items. Every name below came
out with no caller outside the place stated, and each one was then read by hand. The app has no
dynamic dispatch on the WASM objects (no `api[name]`), so a name missing from the TS source really
is never called.

No bugs found. Every `Params` field is read somewhere, and library code has no `println!` and no
`#[allow(dead_code)]`.

## Batches

Every batch that changes a `#[wasm_bindgen]` export changes the WASM interface. Under CLAUDE.md
it needs its own worktree, then `npm run wasm --prefix app` and `npm run check --prefix app`.
That typecheck is what proves the app never called a removed export.

| Batch | Items | Owns | WASM interface | Effort |
|---|---|---|---|---|
| A, dead code | 1-5 | `lib.rs`, `track/wasm.rs` (non-test part), `track/net.rs`, `track/world.rs`, `track/params.rs`, `track/bench.rs`, `track/mod.rs`, `demand/routes.rs`, `demand/wasm.rs`, `demand/intercity.rs` (wasm blocks only), `demand/api.rs` (wasm blocks only), `Cargo.toml`, `docs/code-map.md` sim lines | yes | 2 h |
| B, retire bins | 6-7 | `bin/realnet.rs`, `bin/zones_day.rs`, `Cargo.toml`, `demand/api.rs` 540-630, `demand/kernel.rs` `ModeOut` and `mode_choice`'s walksum lines, `docs/code-map.md` | no (native only) | 1-2 h |
| C, shared helpers | 8-11 | `demand/network.rs` (new loader), `bin/calib.rs`, test modules in `demand/api.rs`, `everyday.rs`, `trip.rs`, `intercity/tests.rs`, `kernel.rs`, `t720.rs`, `demand/pack.rs` | no | 2-3 h |
| D, fare curve | 12 | after T-1112: `demand/kernel.rs`, `api.rs`, `everyday.rs`, `intercity.rs`, `trip.rs`, plus the app's `curveKnots` call | yes | 1-2 h |
| E, test switches and wrong comments | 13-16 | `demand/kernel.rs`, `api.rs`, `intercity.rs`, `alloc.rs`, module docs | no | 1 h |
| F, move tests out | 17 | the seven files named in item 17 | no | 30-60 min |
| G, split big files | 18-19 | `track/wasm.rs` (whole); later `demand/kernel.rs` | no (same exports) | 2-4 h each |
| H, careful refactors | 20-24 | named per item | no | half a day each |

Order: A, E and F are mechanical and safe in any order. Do B before C, since C's loader then has
one less caller to port. D waits for T-1112. Do G's kernel split and H only after T-1092 (the
destination-choice worktree) has landed, since both touch `kernel.rs` and `everyday.rs`.

## Items

### Batch A: dead code

**Done** (simclean, 2026-10-24): items 1-5. Also removed with them: `Dirty::tiles` and `TILE_ZOOM`. `construct_cost` is private now. `DemandApi`, `EverydayApi` and `IntercityApi` readers that only tests and bins call sit in plain `impl` blocks.

**1. Delete the T-001 `Sim` skeleton.** `lib.rs:16-58`, with its test at `lib.rs:60-74`.
- What: remove `Sim`, `new`, `train_count` and `offsets`. Rewrite the crate doc at `lib.rs:1-4`
  ("Skeleton for T-001 ... the layout the real core will use"). Fix `lib.rs:8`: it says the
  demand module stays out of the app's build, but `wasm:demand` enables it.
- Why: the app imports only `TrackApi` and the demand APIs (`clock.worker.ts:10`,
  `demand.worker.ts:22`, `realRoute.worker.ts:20`). Both WASM modules still export `Sim`
  (notes/T-051.md:89 already said so).
- Risk: none; the bindings drop the class. Effort: 5 min. Mechanical: yes.

**2. Delete unused track and route functions.**
- `track/net.rs:661-665` `release_line`, `net.rs:1403-1409` `edge_tiles`, `net.rs:1760-1764`
  `built_after_ports` (with its doc lines).
- `track/world.rs:278-284` `apply_charged` (callers use `apply_paid`).
- `track/params.rs:34-35` `TRACK_SPACING`.
- `demand/routes.rs:240-243` `clear_station_walks` (its doc says "tests", but no test calls it;
  the cache bounds itself at 50,000 entries, `routes.rs:616`).
- Why: no caller anywhere, tests included. Risk: none. Effort: 10 min. Mechanical: yes.

**3. Delete WASM exports nothing calls, and the dirty-tile tracking behind one of them.**
- `TrackApi::many_to_lonlat` (`track/wasm.rs:362-371`), `remove_node` (`822-824`) and
  `split_edge` (`1126-1128`): no caller in the app, the checks or the tests.
- `TrackApi::dirty_tiles` (`wasm.rs:3097-3100`): the app reads `dirty_edges`, `dirty_lines` and
  `dirty_demand_lines` only (`clock.worker.ts:593-594, 2117`). Every edit still works out zoom-12
  tiles for it: `net.rs:1147, 1176, 1192, 1242-1243`, `tiles_of` at `net.rs:1395-1401` and
  `tile_of` at `1379` (used only there). Drop `Dirty::tiles`, the assert at `track/tests.rs:283`
  and the count printed at `track/bench.rs:332`.
- Risk: low. Edits do slightly less work, and no result the app reads changes. Effort: 20 min.
  Mechanical: yes.

**4. Drop the WASM benchmark exports and their two features.**
- `track_bench` and `track_bench_save` (`track/wasm.rs:4323-4337`); `demand_bench`
  (`demand/wasm.rs:1-20`, keeping `wasm_memory_bytes`, which the app imports).
- The features `demand-bench` and `track-bench` (`Cargo.toml:16-20`), and `track/mod.rs:26`
  reduced to `#[cfg(not(target_arch = "wasm32"))]`.
- `track/bench.rs:17-35` has its own `performance.now` binding, copied from `demand/clock.rs`.
  It can go once the wasm bench goes.
- Why: nothing in the app, checks or pipeline builds these features. Only notes/T-005.md:26 and
  notes/T-050.md:56 mention them. The native bins `demand_bench` and `track_bench` stay.
- Risk: low; old commands in those two notes stop working. Effort: 15 min. Mechanical: yes.
  Update `docs/code-map.md:16-17`.

**5. Move test-only and tool-only methods out of the `#[wasm_bindgen]` blocks into a plain `impl`.**
They stay callable from Rust tests and bins and stop being WASM exports with JS glue.
- `TrackApi`: `to_lonlat`, `from_lonlat` (`wasm.rs:351-360`), `set_water` (`375`),
  `new_node_id`, `new_edge_id` (`711-716`), `set_edge` (`827`), `set_station` (`1129`),
  `construct_cost` (`~2480`, called only inside `wasm.rs`, so it can be private), `set_line`
  (`~2846`), `line_cars_cap` (`2888`), `set_schedule` (`~3012`), `line_summary` (`~3284`),
  `train_at` (`3351`), `line_dwell_split` (`3479`). The `realsave` bin uses `set_node`,
  `set_edge`, `set_station` and `set_line` natively, which still works.
- `IntercityApi`: `market_count`, `zone_count`, `open_places` (`intercity.rs:2417-2428`),
  `pair_counts` (`2462`), `set_network` (`2466`; the app calls only `set_network_ll`,
  `demand.worker.ts:1045`), `served_commutes` (`2621`), `commute_corridors` (`2648`).
- `DemandApi::n_subzones` (`api.rs:~1199`).
- Weekday wrappers only tests and bins call: `transfers` in all three APIs (`api.rs:1088`,
  `everyday.rs:2467`, `intercity.rs:2696`; the worker calls `transfers_day`,
  `demand.worker.ts:734`; `load_factor(WEEKDAY, ..)` is exactly 1.0, `days.rs:115`, so
  `transfers(sel)` is `transfers_day(sel, 0)` to the bit). Also `flow_rail`, `station_riders`,
  `IntercityApi::selection_flows`, `station_flows` and `EverydayApi::cell_modes_of`.
- Risk: low. `npm run check` fails at once if the app used one. Effort: 1 h. Mechanical: yes.
  Needs a worktree (WASM interface).

### Batch B: retire superseded tools

**Done** (simclean, 2026-10-24): both bins gone with the readers (also `IntercityApi::served_commutes` and `commuters`, which only `zones_day` called), the walk-sum diagnostic in `mode_choice` and the `road_kmh` key. `calib --cells --exact` on New York, Seoul (fares 2, 0.15) and Tokyo was bit-identical before and after (every figure but the timings). Left: `pipeline/realnet_groups.py` still reads the result JSON the retired `realnet` wrote (its docstring names `data/work/realnet/result.json`); nothing writes it any more.

**6. Retire `bin/realnet.rs` (T-007), now covered by `bin/calib.rs` (T-509).**
- Why: nothing runs it. `pipeline/realnet.py` only writes the networks, and every recent run
  script calls `calib.exe` (notes/T-1056/run_real.py:10, T-918, T-938). Yet sweeping commits keep
  it compiling (last touched in 2d4b95c).
- What goes with it: the `Cargo.toml` entry (`57-61`). In `api.rs`, the readers only it calls:
  `station_board` (`553-556`), `access_bands` (`592-605`), `home_access_modes` (`606-623`),
  `walksum_bands` (`624-628`). The walk-sum diagnostic: `SubOut::walksum` (`api.rs:302-303, 308,
  318, 516`); `ModeOut::rail_walksum`, `N_WALKSUM`, `WALKSUM_BAND_MIN` (`kernel.rs:3586-3587,
  3606-3609, 3750`); and the per-lane band bookkeeping inside the commute pair loop
  (`kernel.rs:3876, 4040, 4201-4202, 4239-4248, 4313`). Also `Params::set_extra`'s `road_kmh`
  key (`kernel.rs:424-430`, "the drive before T-090"), which only realnet's doc uses.
- `rail_band`, `DIST_BANDS_KM` and `dist_band` stay: `demand_bench` prints them
  (`demand/bench.rs:131`) and kernel tests read them. They could go too if the bench drops that
  table, which takes `dist_band` out of the split-back loop (`kernel.rs:4233, 4296`).
- Risk: low. The removed parts are outputs only, so no result changes. Because it touches the
  commute pair loop, run `cargo test --release --features demand` and compare
  `calib --cells --exact` on one real pack before and after. Confirm first that nobody still uses
  `realnet --dump-synth` (`realnet.rs:43-65`); `calib` has no such option.
- Effort: 1 h. Mechanical: mostly; the `mode_choice` lines need care.

**7. Retire `bin/zones_day.rs` (T-276).**
- Why: unchanged since 2026-10-13 (722d2e4). Nothing in pipeline, checks or notes runs it; only
  `research/intercity/corridor_net.py:2` names its input format.
- What goes with it: the `Cargo.toml` entry. `IntercityApi::served_commutes`
  (`intercity.rs:2620-2623`) becomes dead; `pair_counts` and `commute_corridors` become
  test-only (item 5).
- Risk: low. Effort: 15 min. Mechanical: yes.

`realsave` stays: the procedure in `checks/t737-japan.mjs:7-8` uses it.

### Batch C: shared helpers

**Done** for items 8 and 9 (simclean, 2026-10-24): `demand::network::RealNet` and `route_nodes` (native only), used by `calib` and seven real-network tests; `City::subset` and `City::inner` (test only), used by nine tests. All seven real-network tests were run (`--ignored`) and pass. Items 10 and 11 not done (10 is smaller now that the curve is gone).

**8. One loader for a real-network JSON, with its fare km.**
- Copies: `arr_f32` and `arr_u32` in `bin/calib.rs:32-37`, `bin/realnet.rs:27-32` and
  `bin/zones_day.rs:21-26`. `f32s`/`u32s` closures in tests at `api.rs:2405, 2662, 2813, 3012`,
  `everyday.rs:731`, `trip.rs:1125` and `intercity/tests.rs:824`. The per-route-node hop km loop
  in `calib.rs:101-119`, `zones_day.rs:49`, `api.rs:2427-2446` (and the `rn_st, rn_line, hop_km`
  versions at `2668, 2833, 3025`), `everyday.rs:734` and `intercity/tests.rs:842`.
- What: a `demand::network::RealNet { st_xy, line_n, stops, times, tph, cars }` with
  `load(path)`, `hop_km()` (kernel route-node order: each line forward, then reversed, 0 at
  ends) and `set_on(&mut DemandApi)` style helpers.
- Why: about 150 lines copied, and the hop-km order is a rule that must match the app's `hopKm`.
- Risk: low (tests and tools only). Effort: 1 h. Mechanical: yes.

**9. `City::subset(&keep)` for tests.**
- Copies: "take `synth_city(1)`, keep these cells field by field" appears at `kernel.rs:5566,
  5608, 5682, 5748, 5790, 5848`, `everyday.rs:306`, `t720.rs:12-18`, `trip.rs:826-832` and
  `api.rs:1707`.
- What: one `#[cfg(test)]` method in `demand/pack.rs`. About 100 lines. Risk: none. Effort:
  30 min. Mechanical: yes.

**10. One fare setup for the three populations.**
- The `rn_fare` loop plus `put_fare_curve` is at `api.rs:979-982`, `everyday.rs:~2338-2341` and
  `intercity.rs:2523-2526`. `set_fares`' checks and change detection are at `api.rs:903-918` and
  `intercity.rs:2487-2499`.
- What: a `Fares { fare_min, per_km, km, curve }` with `set(..) -> changed` and
  `apply(&mut Graph)`.
- Why: one fare rule in one place. Bit-identical (same arithmetic).
- Risk: low. Effort: 45 min. Do it after item 12, which shrinks it.

**11. `Loads::msa(&mut self, &Loads, round)`.**
- The seven `msa` lines are copied at `api.rs:1031-1037`, `intercity.rs:2586-2592` (and three of
  them for the commutes at `2579-2581`) and `demand/bench.rs:168-170`. `everyday.rs:2392-2396`
  averages five of the seven; its `board` and `alight` are never read, so leaving them out is
  fine, not a bug.
- Risk: none (same operations in the same order). Effort: 15 min. Mechanical: yes.

### Batch D: conditional on T-1112

**Done** (simclean, 2026-10-24): `FareCurve`, `trip_km`, `rn_km`, the skim key fields, `put_fare_curve`, `check_fare_curve`, the three `set_fare_curve` exports, `Leg::km` and the kernel test for the cap. The app sends `set_fares` only (`demand.worker.ts`, `demandClusters.ts`, `curveKnots` removed from `fareCurve.ts`). The default and every old base and rate give the same figures: smoke 12,796,392 rail trips a day, `calib --exact` identical. A line below $0 at 0 km can no longer be set (`fitOrigin` holds the dragged or typed node, `tidyNodes` raises an older save's first node), so demand's base plus rate and `faresFor` agree and `faresFor` has no clamp.

**12. The multi-node fare curve (T-883), if the fare becomes two nodes on a straight line.**
- If the app turns T-1112's two nodes into `set_fares(base, per_km)` and stops calling
  `set_fare_curve` (`demand.worker.ts:1058`, `curveKnots`), all of this goes:
  - `kernel.rs`: `FareCurve` (`2406-2435`), `Graph::fare_curve` and `rn_km` (`2400-2403`, init
    `2580-2581`), `DestSearch::trip_km` (`3321-3361`) and its use (`2914-2920`), the skim key's
    curve fields (`3452-3453, 3458, 3464-3465`), the test at `4918-4919`.
  - `api.rs`: `check_fare_curve` and `put_fare_curve` (`51-73`), `fare_curve` field (`378-379`,
    `725`, `790`), `set_fare_curve` (`920-930`).
  - `everyday.rs`: `2313-2316`, `2341`. `intercity.rs`: `1698`, `1929`, `2500-2507`, `2526`.
  - `trip.rs`: `341-346`.
  - About 150 lines. T-887 (exact fare income on a bent curve) becomes moot.
- Watch: today any two knots (4 values) switch the curve machinery on (`put_fare_curve` returns
  early only below 4), so leaving the app as it is keeps all of it alive. A line through two
  nodes equals `base + per_km x km` only when its value at 0 km is at least 0. If T-1112 lets the
  line cross below 0 at short trips, `FareCurve::at`'s clamp at 0 is a real difference: keep a
  two-knot curve, or clamp the nodes in the app.
- Risk: moderate only in that it changes the WASM interface and fare results for any old save
  with a bent curve (T-1112 already keeps only the first and last node). Effort: 1-2 h.
  Mechanical: yes, once T-1112 decides.

### Batch E: test switches, the allocator, wrong comments

**Done** (simclean, 2026-10-24): items 13-16. `demand::alloc` and `demand::bench` are native only.

**13. Remove the T-119 `LEGACY` switch.**
- Where: `kernel.rs:3418-3424` and its uses at `kernel.rs:3494, 3528`, `api.rs:1014`,
  `intercity.rs:1939, 2563`, and the `TW_LEGACY` block in the ignored `northeast_profile` test
  (`api.rs:2447-2451`).
- Why: `cfg!(test)` only, so release builds never see it, and only one ignored test sets it.
  T-119 is long done.
- Risk: none. Effort: 10 min. Mechanical: yes. (`XFER_OFF` and `CHECK_LANES` are live test
  switches; keep them.)

**14. Keep the counting allocator out of the game's WASM.**
- Where: `demand/alloc.rs:53-54` installs `#[global_allocator]` in every `demand` build,
  including the app's demand workers. Only `demand/bench.rs` and the ignored `northeast_profile`
  (`api.rs:2496-2542`) read it.
- What: gate the module on `not(target_arch = "wasm32")` (once item 4 drops the WASM bench).
- Why: an atomic add and max on every allocation in the workers, for figures nobody reads there.
  Cheap to remove; results unchanged.
- Risk: none. Effort: 5 min. Mechanical: yes.

**15. Fix the "legacy zero walking-loss" comments: the blocks are not zero.**
- Where: `api.rs:288` ("legacy zero walking-loss arrays"), `api.rs:1204-1205` ("then legacy zero
  walking-loss blocks"), `api.rs:1407-1408`, and `api.rs:1575` ("Legacy loss reconciliation
  (zero since T-153); layout retained for worker compatibility").
- Why: since T-558 these blocks carry the walkers rail takes (`walk_lost_by_sub`, set at
  `kernel.rs:4258`), and `flows` uses them (`api.rs:1544-1545, 1576-1590`). The comments send a
  reader the wrong way about live demand code.
- Risk: none (comments only). Effort: 10 min.

**16. Stale module docs.** Comments only:
- `demand/mod.rs:1-14`: "T-005 spike ... Readable, not final", and it lists 4 of its 16 modules.
- `demand/api.rs:13`: "Commutes only (SPEC 4.1's other trips are not modelled yet)". The
  everyday layer and the market exist.
- `demand/everyday.rs:1-3`: "NYC-first home-based outings".
- `demand/network.rs:1-3`: "for the spike".
- `track/cost.rs:1`: "(T-030 will supply the real mask)".
- `track/net.rs:5-7`: "`edit.rs` builds undoable operations on top"; there is no `edit.rs`, it
  is `world.rs`.
- `track/mod.rs:22`: "`wasm`: ... (a sketch)" for a 4,300-line API; the list also leaves out
  `trace`.
- History narration to cut when the function is next edited: `api.rs:1244` ("This used to
  answer empty..."), `routes.rs` doc "as before", `bin/calib.rs:158`.
- Risk: none. Effort: 20 min. Keep to these lines; CLAUDE.md forbids sweeping comment rewrites.

### Batch F: move inline tests out

**Done** (simclean, 2026-10-24): all seven files (`intercity.rs` was already one). `track/wasm.rs` had five test modules, each now a file under `track/wasm/`; the others are `<file>/tests.rs`. `track/boarding.rs` was done last, after the T-1121 work had landed.

**17. Put each big inline test module in its own file, as `intercity/tests.rs` already is.**
- `track/wasm.rs:4362-8078` (3,700 lines), `demand/api.rs:1699-3129` (1,430),
  `demand/kernel.rs:4868-6091` (1,220), `demand/everyday.rs:277-1170` (900, sitting in the middle
  of the file between `DecayTable` and `Pair`), `track/boarding.rs:1313-2147` (830),
  `demand/trip.rs:789-1227`, `track/capacity.rs:1026-1477`.
- What: `#[cfg(test)] mod tests;` with the body in `wasm/tests.rs` (or `#[path]`). `use super::*`
  still sees private items from a child module.
- Why: halves the five biggest files, so agents reading code stop paging through tests.
- Risk: none. Effort: 30-60 min. Mechanical: yes. Each file needs a moment when nobody else is
  editing it.

### Batch G: split the two biggest files

**Done** for item 18's `track/wasm.rs` only (simclean, 2026-10-24): the exports are in `track/wasm/{money,edits,draw,crossings,links,lines,snapshot,service}.rs`, the shared helpers stay in `wasm.rs` (1,300 lines, was 8,077). Not moved: the drawing geometry helpers to `track/draw.rs`. Item 19 (`kernel.rs`) not done.

**18. Split `track/wasm.rs` along its own section markers.**
- `impl TrackApi` runs as one block over `211-3632`. Its sections (`// ----` at 493 money, 710
  ids, 815 low-level edits, 1149 drag, 1378 drawing tool, 2017 doubling, 2211 crossovers, 2413
  links and construct, 2842 lines, 3052 undo, 3092 dirty, 3113 snapshot, 3252 service) become
  files under `track/api/`. wasm-bindgen accepts several `#[wasm_bindgen] impl TrackApi` blocks.
- The drawing tool's geometry helpers (`3633-4300`: `stroke_pieces`, `express_verts`,
  `tangent_arc`, `design_radii`, `lead_in*`, `render_*`) go to `track/draw.rs`.
- Risk: low (moves only; the generated `.d.ts` should differ in order only). Effort: 2 h. Needs
  sole ownership of `wasm.rs` for the sitting.

**19. Split `demand/kernel.rs` (4,868 lines of code) by its numbered stages** (listed at
`kernel.rs:3-19`):
- `Params` (`33-~460`), gravity, access and subzones (`1796-2357`), the graph and search
  (`2380-3530`, skim cache included), mode choice (`3583-4321`), assignment (`4326-4628`),
  crowding (`4630-`).
- Risk: low (moves). Effort: 2-3 h. Do it after T-1092 lands; every demand task touches this
  file.

### Batch H: careful refactors (not mechanical)

**20. One assignment walk instead of two.**
- `kernel.rs:4555-4628` (`DestWalk::run`, scalar) and `everyday.rs:1407-1489` (`LaneWalk::run`,
  per-purpose lanes) are the same rule. The T-1055 boarding rule had to be written into both
  (both comments cite it), which is how such copies drift.
- What: make it generic over a small `Flow` trait (`f32` and `[f32; LANES]`: zero, add, scale).
- Checks: `check_lanes` (`everyday.rs:1498-1525`) already asserts lanes against the kernel to
  the bit; run it and the everyday tests, plus a timing (generic inlining must not slow it).
- Risk: moderate. Effort: half a day.

**21. Break up `mode_choice` (`kernel.rs:3689-4321`, 633 lines).**
- Candidates: the per-origin-zone factor fill (`3886-3909`), the T-874 chunk dealing
  (`3915-3948`), the T-1059 refused-pair lane block (`4072-4134`), the split back onto access
  stations (`4261-4298`).
- Risk: moderate. It is the hot loop, and it must stay bit-identical with `trip::pair_choice`.
  Do it only with `calib --exact` before and after on a real pack and a bench timing. Effort:
  half a day. Low priority.

**22. Break up `intercity::build_for` (`intercity.rs:877-1431`, 555 lines)** at its `// ----`
stages. The T-929 bit-identity tests on `world_rows_for` cover it. Risk: low to moderate.
Effort: 2-3 h.

**23. Break up `bin/joint_gravity.rs` `main` (`89-914`, 826 lines)** at its sections (`124`,
`201`, `242`, `277`, `381`, `426`, `491`, `585`, `628`, `843`, `881`). The pipeline depends on
it: compare `--dry --dump` output before and after. Risk: low to moderate. Effort: 2 h. Low
priority.

**24. Drop a per-pass copy of the station matrix.**
- Where: `api.rs:494` clones `mo.od` (S x S f32, about 4 MB at 1,000 stations) every pass, only
  because `compact_od(&mo.od)` at `api.rs:515` reads the original afterwards.
- What: build the compact table first, then turn `mo.od` into the two-way table in place
  (`std::mem::take`).
- Risk: low; bit-identical. Effort: 15 min.

## Checked and kept

- **The intercity single pack** (`LegacyPack`, `single_pack`, the closed world,
  `market_site_zones`, `intercity.rs:173-200, 804-869, 1107-1130, 1489-1508`): not dead.
  `pipeline/intercity.py:392` still writes `data/packs/intercity.json`, and `bin/joint_gravity.rs`
  reads it (651, 910) for `market_zones.py`. Keep it until that file retires.
- **Frequency-scheduled lines and the TWT2/TWT3 save readers** (`save.rs:11-33, 285-292`,
  `capacity.rs:4, 89`, `Network::set_schedule`): only old saves and tests make such lines now.
  Removing them is Anita's call about old saves (below).
- **`mode_choice` against `trip::pair_choice`, and `walk_kept`**: parallel on purpose, so the trip
  explorer shows the model's own bits, and tested to match. Keep.
- **`trip.rs` `cell_subzones`** (`472-536`): it finds a cell's K stations by scanning every
  station, not by reading `Access` (`build_access`, `kernel.rs:1796`). That is fine for one
  query, but it is a second copy of the access rule. Worth a test that the two pick the same
  stations, not a rewrite.
- **The pre-T-285 `selection_reference`** (`everyday.rs:805-841`): a test oracle. Keep.

## For Anita (via the orchestrator)

- Can frequency-scheduled lines and TWT2/TWT3 saves stop loading, or be converted to train
  counts on load? Either way the frequency path through capacity and world could go. Only worth
  it if no save she keeps is that old.
