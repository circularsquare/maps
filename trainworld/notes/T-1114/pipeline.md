# T-1114 part 4: pipeline/, checks/, tools/ and notes/ scripts

Read-only review, 2026-10-24, while US wave 3 (T-1018) runs the pipeline. Nothing was run or edited.
Every "unused" claim below was grepped across pipeline/, checks/, tools/, docs/, notes/, SPEC.md,
TASKS.md, app/src and sim/src.

## Summary

26 items (numbered 1-25, plus C1) in 9 batches, A to I, and 9 bugs listed apart. About 1,300 lines of pipeline/ code could go or
fold into shared helpers, about 8,700 lines in 100 finished notes/ scripts, and about 1,000 lines of
repeated Playwright scaffolding in checks/ could go into one shared module.

The five best:

1. **Delete 100 finished one-off scripts in notes/** (item 1): 8,712 lines, no code imports them,
   no pack bytes, no wave conflict.
2. **One helper to run a sim binary** (item 17): seven copies of the same cargo env and launch block
   in build_city, coarse_zones, market_zones, routes, intercity, zones and world_run.
3. **One pack writer** (item 18): the same "write .bin.part, .json.part with indent=2, replace both"
   block 13 times. Byte-identical if the helper keeps `json.dump(header, f, indent=2)`.
4. **T-594: `intercity.py --help` rewrites every Northeast pack's gravity** (item 13): any unknown
   flag falls through to the full sites, joint solve and market run.
5. **Retire the NYC-only T-006/T-034 checkers and old in-place migrations** (items 9-11):
   county_flows, road_speed, realnet_groups, realnet's `compare`/`single`, group_quarters' in-place
   path and regions' in-place path. About 800 lines.

Also urgent, not cleanup (bug B1): **wave 3's own toolkit sets `TW_FAULT_S=900`**, the setting that
killed a gate in T-903 and again in T-1078.

Rules for all batches:

- Nothing that changes a built pack's bytes is in a mechanical batch. Items that do (T-878, T-926)
  are in their own batch, marked.
- **Wave wait** means: no edits until no `world_run.py` process is live, since every step re-imports
  the files it uses (`world_run.py step` starts a fresh `python -I`). That covers all of pipeline/,
  `checks/pack_gate.py`, `checks/city-browser.mjs`, `checks/dev-nowatch.mjs`, tools/aws/, and
  `notes/t1018/us/` (wave 3's runbook).
- **Path constants and world_run's `Redirect`.** `--root` runs rewrite each pipeline module's own
  path constants as the module loads (`world_run.py:112-186`). `RAW` (`data/raw`) is deliberately
  left pointing at the real tree. A shared `ROOT`/`RAW` module would change which tree `RAW`
  resolves into, so constants stay per module. The helpers in batch F take paths at call time and
  are fine.

---

## Batch A: notes/ sweep (no wave wait, except `notes/t1018/`)

**Batch A done 2026-10-24** (staged, not committed): item 1: 107 files, about 9,000 lines `git rm`-ed
(the 100 listed, plus 4 `t727/` scripts: build_graph, export_pis, route_fit, size; `t727/edit_test.rs`
stays, since T-755 and T-763 still need a native harness) and the 3 `__pycache__` folders; each
note that cites a removed script gained "(script removed 2026-10-24, in git history)" once. Item 2:
the 5 curve fitters and `t314_loo.py` moved to `pipeline/curves/` (ROOT fixed one level deeper,
docs/adding-a-city.md, docs/code-map.md and research/world-sampler.md cites updated, a code-map entry
added); `t1056_ceiling.rs` and `run_ceil.py` moved to `checks/ceiling/` (`T1056_WORK` names the scratch
folder; cites in T-1059.md, T-1091.md and a code-map entry updated). Not done, by this section's own
rules: comments citing `notes/t277_fit.py` and the other fitters in build_city.py, france.py,
germany.py, japan.py, japan_zones.py and calib.py's `t509_nhts` cite (wave wait, batch D or later);
`fitted_curves.json`'s header still says `notes/t314_loo.py` until it is next written; `t509_nhts.py`,
`t726_pbfheader.py` (T-766) and `t562_fit.py` (batch G) stay in notes/; `t1018/us/backup.py` and
`restore.py` wait for the wave; `t928_breakdown.py` and `t929_check.py` (lean-keeps) stay. B6 stands:
`t314_loo.py` still imports the deleted `t162_fit`.

Owned files: notes/ only, plus the doc lines named in item 2. No pack bytes.

**1. Delete finished one-off scripts.** 100 files, 8,712 lines. Risk: none (all git-tracked, so
recoverable). Effort: small. Mechanical: yes. Each task below is archived (ARCHIVE.md line given)
and its basenames are cited only by their own task notes. The note lines can stay as history or
gain "script removed". Delete each group together, since they import one another.

| Task | Files | ARCHIVE | Cited only by |
|---|---|---|---|
| T-041 | t041_compare.py | :666 | T-041.md:4 |
| T-162 | t162_tables.py (already broken, imports deleted t159_flows) | :835 | T-162.md:66, T-198.md:143 |
| T-171 (part) | t171_gravity.py (broken: imports deleted t162_fit), t171_path_day.py, t171_waterfront.py | :737 | T-171.md |
| T-196 | t196_atus.py, t196_plot.py | :787 | T-196.md:22,90 |
| T-289 (part) | t289_jobs.py | :656 | T-289.md:9 |
| T-379 | T-379/ (decomp, fit_compete, model_bands, nhts_cbsa, table) | :574 | nothing |
| T-420 (part) | t420_figures.py | :536 | T-420.md:176 |
| T-453, T-456 | t453_common, t453_fetch_bres, t453_lon, t453_tyo, t456/ (14 files) | :528, :539 | T-453.md:8 |
| T-463 | t463_jp.py | :499 | T-463.md:3 |
| T-465 | t465_prep.py | :500 | T-465.md, T-498.md, T-500.md |
| T-500 | t500/ (nyseg, parse, seg, shares) | :494 (dropped) | nothing |
| T-509 | t509_centres, t509_fitdecay, t509_sao, t509s7.py, t509s56/ (14 files) | :466 | T-509.md, T-621.md, T-636.md |
| T-551 | t551_ceiling.rs, t551_copy, t551_nhts, t551_tables | :81 | T-551.md |
| T-572 | t572_measure.py (broken: imports deleted t360_jobs) | :420 | T-572.md:5 |
| T-636 | t636_ceiling.rs, _copy, _md, _r2_md, _run_all, _run_extra, _run_r2, _s7, _tables | :387 | T-636.md, T-637.md (T-636.md:417 says they only reproduce on pre-change code) |
| T-854 | T-854/ (batch.mjs, nocarpack.mjs, run.mjs, t854_ceiling.rs) | :148 | T-854.md:357, T-916.md:106 |
| T-918 | T-918/ (curves, run_real, t918_ceiling.rs, t918b_ceiling.rs, t918_rel.diff) | :100 | T-918.md, T-940.md:39 |
| T-938 | T-938/ (copy, run_ceil, run_real, sum_real, t938_ceiling.rs) | :87 | T-938.md:35 |
| T-1003 | T-1003/ (12 scripts, cn_all.txt, empty_net.json) | :57 | T-1003.md:85,264 |
| T-1056 (part) | T-1056/run_real, sum, t1056.rs, measure.patch (keep t1056_ceiling.rs and run_ceil.py, item 2) | :31 | T-1056.md |
| T-928 (part) | t928_exp.py (paths into T-928's scratchpad) | :82 | T-928.md:106 |

The 4 non-script files in that list (cn_all.txt, empty_net.json, measure.patch, t918_rel.diff) are
check outputs, which CLAUDE.md puts in `../artifacts/trainworld/`. `notes/__pycache__`,
`notes/t1087/__pycache__` and `notes/t509s56/__pycache__` are gitignored and can just go.

**2. Move the live notes scripts out of notes/.** Risk: low (doc paths go stale if missed).
Effort: small to moderate. Mechanical: mostly. No pack bytes. The pipeline-file comment edits wait
for the wave, so do them in batch D or later.

- The curve fitters go to a new `pipeline/curves/` (or `checks/`). docs/adding-a-city.md tells
  readers to run them and pipeline code cites them:
  - `t277_fit.py` (adding-a-city.md:545, build_city.py:224);
  - `t289_curve.py` (adding-a-city.md:615, build_city.py:281, france.py:10);
  - `t290_curve.py` (build_city.py:412, germany.py:13);
  - `t360_curve.py` (adding-a-city.md:605);
  - `t420_curve.py` (adding-a-city.md:650, build_city.py in 7 places, japan.py:19, japan_zones.py:199).
- **`t314_loo.py` is the live producer of `pipeline/fitted_curves.json`**: docs/code-map.md:424
  says so, and so does the JSON's own header. It imports t277_fit, t290_curve and the deleted
  `t162_fit` (line 174, bug B6), so it moves with the fitters and its US `setup` needs t162_fit back
  from git (`464f70f^`) or that stage cut.
- `t509_nhts.py` is cited at calib.py:56 as the source of `NHTS_NY`. Move it or drop the cite.
- `t726_pbfheader.py` is named by open T-766 (TASKS:203). Move it into pipeline/ when T-766 lands.
- `t562_fit.py` (nocar.py:82, data-register.md:203) goes with nocar in batch G.
- The ceiling harness: `T-1056/t1056_ceiling.rs` and `run_ceil.py` go to `checks/ceiling/`, as
  T-940.md:39 suggests. They are the current harness (T-1091.md:24 updated them).
- `t1018/us/backup.py` and `restore.py` go to `tools/aws/`, since docs/aws.md:71 names backup.py
  as step 8 of the landing runbook. **Wave wait**: wave 3 is using them.

**3. Leave alone (open tasks):** T-361 (T-361-measure.mjs and .ps1), T-1000 (t1000lib and four
analysis scripts), T-1004, T-1005, T-1018 (the rest of t1018/us, in use by wave 3), T-1087, and
t171_operators.py with t171_net.rs.txt (T-269). t727/ (T-727 archived) looks superseded by
realgraph.py, but T-755 and T-763 are open and cite T-727.md: delete all of it except
edit_test.rs once you have checked those two tasks. Two lean-keeps:
- `t928_breakdown.py`, a generic batch-minutes reader, could move to tools/;
- `t929_check.py` could move to checks/ before T-943 changes market_zones, but its paths point
  at a scratchpad.

---

## Batch B: checks/ fixes and the code-map's checks section (no wave wait, except as marked)

Owned files: the named checks and docs/code-map.md lines 201-352. No pack bytes. One exception:
`t883-fare-curve.mjs` has uncommitted edits in progress (97 lines, another agent), so leave it to
that agent.

**4. `t066-worker-check.mjs` and `t149-worker-check.mjs`: a fixed small zone set.** Lines 33-38
and 76 of t066, line 58 of t149. Both open every registered zone (1,667, 2.3 GB of packs) in one
Node worker. T-756 measured 1.5-2.5 GB a worker for Japan's 7 cities, so this almost surely runs
out of the 4 GB WASM cap. Use nyc, phl and bos with their market packs. Then fix t066:38
(`cityInfo` length against the whole registry) and t149:94 ("other eight cities"). Risk: low.
Effort: small. Mechanical: partly. Code-map says "a few minutes each".

**5. `t800-outlines.mjs`: add `&opened=all`, or retire it.** Lines 55-58 and 86-87. Since T-975,
hidden realms draw no outlines (`cityBoundaries.ts:102,152` filter by `zoneShown`), so a New York
start's "Europe has outlines at once, at least 40" should fail. `opened=all` turns realms off
(`realms.ts:78`); t975-realms covers the realm side. Risk: low. Effort: small. Mechanical: yes.

**6. Small check fixes.** Risk: none. Effort: small. Mechanical: yes.
- `t801-no-grey.mjs:91`: drop "no grey layer". Nothing defines `tw.greyAt`, so it always passes.
- `t496-edit-latency.mjs:10` and `t648-stop-edit.mjs:13` call `import(process.env.PLAYWRIGHT_MODULE)`
  with no fallback; use the resolver the other checks use.
- `t066-worker-check.mjs:40` uses the frame kind `'equirectangular'`, which is not a `FrameKind`
  (`frame.ts:18`); it is `'equirect'`.
- `t125-fingerprint.mjs:29-31`: the `append_water` fallback is dead, since `add_pack_water` exists.
- `t277-browser.mjs:7` gives its path as `notes/t277-browser.mjs`.
- Hardcoded `C:/Users/anita/...` paths in 13 checks (t1007, startup-timing, t124, t301, t303,
  t326, t328, t377, t409-split, t496, t648, t761, t908) read the main tree from an agent worktree.
  Use `new URL('../T-007-real-nyc.save', import.meta.url)` as smoke.mjs does.
- Temp Chrome profiles are never removed in t277, t496, t648, t756, t800 and t975 (`mkdtemp`, no
  `rm` in `finally`).
- History comments: t066:72,84; smoke:128,140; t715:2.

**7. Code-map checks section.** Risk: none. Effort: small. Mechanical: yes.
- Add the unlisted checks: `t675-wheel-lag` (cited at mapView.ts:62), `t1007-panic`,
  `t761-plain-net` (feeds `t761_market_bench`, sim intercity tests.rs:1027), and `token_usage.py`.
  Also list `world_field_check.py` here; today it appears only in the world section.
- Add pipeline's `test_education_sources.py` and `test_university_sources.py`
  (`python -m unittest` from pipeline/, T-142.md:128). Nothing lists them now.
- Fix stale timings: every-pack-opens "18 minutes for all" is about 7 hours at 16 s a zone today;
  t066/t149 "a few minutes".

**8. Retire finished one-off measurements in checks/.** Risk: none at runtime (a judgement call).
Effort: small. Mechanical: yes, once decided.
- `t303-measure.mjs` (T-303 closed): covered by startup-timing, t348 and t1008, and its default
  BASE is 8800, which CLAUDE.md says checks must not use.
- `t277-browser.mjs`: city-browser.mjs grew out of it.
- `t125-fingerprint.mjs`: T-125's before and after for a frame change.
- Keep t161-complexes (it feeds `api.rs:2804,3010`) and t761-plain-net.

---

## Batch C: checks/lib.mjs (no wave wait for most; city-browser and dev-nowatch wait)

Owned files: a new checks/lib.mjs and the checks moved onto it. No pack bytes.

**C1 (item 8a). A shared Playwright helper.** Today no shared module exists; the only reuse is
t671-station-snap importing t671-ghost-hover. Repeated blocks, by file count:

| Block | Files |
|---|---|
| the `playwright()` resolver (`'_npx'`) | 39 |
| Chrome launch with `chrome.exe`, a temp profile and cleanup | 47 |
| the `pageerror` and console error filter | 43 |
| real-save import (`kind: 'import', bytes`) | 30 |
| the demand-final wait | 18 files, 27 times |
| the test-line builder (near, route, addStation, addLine) | 12 |
| the Node demand-worker harness (rolldown, adapter, next) | 6 |
| the Vite virtual-module build in Node checks | 9 |
| `startVite` | 5, plus t756's own |

About 1,000-1,200 lines for a 150-line lib. Risk: moderate, since most of these checks are rarely
run and a slip goes unnoticed. Do smoke and the regularly run checks first, the rest when next
touched. The resolver and launch parts are near mechanical; the rest is not. **Wave wait** for
city-browser.mjs (run by `pack_gate --browser`, pack_gate.py:841) and dev-nowatch.mjs (run by
world_run.py:1088).

Overlaps, for whoever does this: t714-pan and t737-japan are one harness on two saves (merge
them, with a `SAVE` option). t1008-load-lag covers t348-lag's startup mode, so t348 keeps only
pan and zone-join. t124-periods, t326-clusters and t409-split each need two runs and a hand diff
of `sigs`: one runner looping over the `perfOff` values would replace all three (split is already
bit-checked by t409-worker-check). The click-latency checks (t496, t648, t1020, t671, t724) do
not cover one another.

---

## Batch D: pipeline dead code (wave wait)

**Batch D done 2026-10-24** (tree `tw/pipeclean`): item 9: county_flows, road_speed and
realnet_groups `git rm`-ed (grep found no caller). Item 10: realnet's comparison section,
`cmd_compare`, `cmd_single` and `SINGLE_LINE` gone, with their docstring lines and the dead `realnet`
bin's mention; T-269 (compare's operator count) closes with it. Item 11: group_quarters' `apply`,
`REBUILT`, `drop_cells`, `read_dtype`, `main` and regions' `add_to_pack`, `main` gone; code-map's
re-solve line now points at `pack_gravity --keep-zones` (B5), adding-a-city's clean-up line no longer
names `regions.py`. `boundary_only`'s header regex kept (optional, low value). Item 12: all done.
The fitter cites now say `curves/...`; `_japan_zone`'s cite of the missing `t420s3_curve.py` names
`japan_zones.fit`. calib.py's `t509_nhts` cite stays (the script is still in notes/).

Owned files: the named pipeline files and the code-map lines. No pack bytes: none of these is on
any build path.

**9. Delete the NYC T-034/T-078 checkers.** These are `pipeline/county_flows.py` (125 lines),
`pipeline/road_speed.py` (223) and `pipeline/realnet_groups.py` (70). Nothing runs them: they
are not in world_run, not in docs, and imported only by notes/t162_tables.py (deleted in item 1).
T-034 was superseded on 2026-10-09 (ARCHIVE:1115). They also carry their own copy of the gravity
decay: T-1091 had to patch county_flows.py:69 and road_speed.py:83,131 for `decay_flat_km`, and
they ignore T-852's blocks and T-1083's decay tables, so they are already wrong on newer packs.
Risk: none. Effort: tiny. Mechanical: yes. Cites to update: T-1091.md:23, T-034.md:14-16.
**Keep** `fit_decay.py`: `notes/t1018/us/lodes_mean.py` imports its `Zones` and `local_xy`, and
TASKS T-1018 names lodes_mean. **Keep** `lumps.py`: open T-043 is its follow-up.

**10. Delete realnet.py's old New York compare.** That is `compare` and `plot_compare`
(`realnet.py:997-1194`), `cmd_compare` (1394-1400), `cmd_single` with `SINGLE_LINE` (1368-1391,
T-025's batch-3 test line), and the `"compare"` and `"single"` entries in the dispatch (line
1437) and docstring. About 240 lines. They were superseded by `calib.py score` (T-509), and
nothing outside realnet calls them. Keep `cmd_targets`, `cmd_counties`, `COUNTIES` and
`COUNTY_ORDER`, which `calib.targets_nyc` reads. Check T-269 first (TASKS:45, "realnet's
operator count is wrong"): if it is about compare's counts it closes with this; if it is about
the `realnet` bin, it stays. Risk: low. Effort: small. Mechanical: yes. Update code-map line 473
("New York's older targets, `compare`").

**11. Delete the in-place migration paths for packs built before the rule.**
- `group_quarters.py:156-316`: `apply`, `REBUILT`, `drop_cells`, `read_dtype`, `main` and the
  CLI, about 160 lines, plus docstring lines 6 and 16-22. T-234 (2026-10-13) patched packs built
  before it, and every US build since takes institutions off in `build_city.blocks_for_state`.
  Keep `state_table`, `by_block`, `remove`, `cap_job_holders`, `summary` and the constants
  (build_city and usa use them).
- `regions.py:53-106`: `add_to_pack` and `main`, about 55 lines. That adds `region` to packs
  built before T-162; every pack has it now. Keep `add_columns`, `of_cells` and `header_block`.
  docs/code-map.md:365-366 still calls `regions.py nyc` "the way to re-solve after a decay or
  border factor change". It calls `add_gravity` on the live pack, which re-solves on fresh 2 km
  zones, so on a coarse or merged pack it likely drops the 4 km blocks and leaves routes stale
  (bug B5). Point that line at `pack_gravity --keep-zones` (code-map:381) instead.
- `build_city.py:1298-1318` `boundary_only`'s header regex (T-352's one-off rename): the
  boundary rewrite stays, the source-line patch could go. Low value; optional.

Risk: low. Effort: small. Mechanical: yes.

**12. Small pipeline slop.** Risk: none. Effort: tiny. Mechanical: yes. No pack bytes.
- `build_city.py:1204-1211`: "before T-013 ... kept for the comparison". This works out every
  block's h3 cell in a Python loop only to print a count. Delete it.
- `build_city.py:642` cites `notes/t420s3_curve.py`, which does not exist.
- `merge_cells.py:7`: the docstring says `--floor 100` runs the floor rule at 100, but main
  (line 573) refuses any floor but FLOOR without `--stats`.
- `merge_cells.py:52` and `:574`: "nocar.py and everyday_rates.py remerge by FLOOR". Only nocar
  remerges; everyday_rates only rewrites headers and the stamp.
- `read_pack.py:99`: `dec =(` spacing.
- `intercity.py:47`: "until T-276 places it" (history).
- realnet.py comments cite calib_*.py correctly; leave them.

---

## Batch E: runner-path fixes (wave wait)

**Batch E done 2026-10-24** (tree `tw/pipeclean`): item 13: T-1014 had already made a bare call
print the usage; intercity now also exits with its usage on any flag outside `FLAGS` (checked:
`--help` and `--group auto --tyop` print it). Item 14: pack_gate reconfigures stdout and stderr
under `__main__`. Item 15: `BACKUP` and the copy gone; `data/work/pre-T509s4/` (main tree) is for the
orchestrator to delete after landing, since wave 4's old code still writes it. Item 16: no ids
prints the usage; the docstring names the merge's `--undo`.

Owned files: intercity.py, checks/pack_gate.py, everyday_rates.py, enrich_all.py. No pack bytes.

**13. T-594: `intercity.py` usage on unknown flags.** Lines 401-431. Any argument that is not
`--groups`, `--solve` or `--group` falls through to `write_sites()`, `joint()` and
`market_zones.main([])`. So `--help` or a typo rewrites every Northeast pack's gravity and every
market pack. Print `__doc__` and exit on any flag outside a known set; keep the bare call (no
arguments) as it is. Risk: low (world_run passes `--solve ... --name ... [--fixed] [--hold-report]
[--new]`, so list those). Effort: small. Mechanical: yes. Also worth noting, but not changing now:
intercity imports build_city (and so plan_config's 3 MB plan) and reads the 1 MB registry at
import, only for `CITIES["nyc"]["origin"]` and `R_EARTH`. world_run pays that cost for each
`@ne_gap`, `@groups` and `@preview` step.

**14. T-980: `pack_gate.py` stdout.** Lines 1107-1108. Nothing in pack_gate reconfigures stdout,
and `-I` ignores PYTHONIOENCODING, so a hand run on a cp1252 console crashes on a non-Latin
detail. world_run's `step_main` (world_run.py:205) already reconfigures. Add
`sys.stdout.reconfigure(encoding="utf-8", errors="replace")` (and stderr) under `__main__`.
Risk: none. Effort: one line. Mechanical: yes.

**15. `everyday_rates.py`: stop the T-509 stage 4 backup.** Lines 39 and 114-121. `patch`
copies every pack's header (and its res-9 copy's) into `data/work/pre-T509s4/` the first time it
sees it. The runner calls it for every row it builds, so that folder grows with every pack and
holds 67 MB today. CLAUDE.md: a backup goes once a later state is checked clean. Drop `BACKUP`
and the copy, then delete the folder (no need to ask). Risk: none. Effort: tiny. Mechanical: yes.

**16. `enrich_all.py:41`: the no-argument default.** It is `list(b.CITIES)`: about 1,700 configs,
planned and tier 0 rows included. The first tier 0 row hits `enrich_everyday`, which reads a
blocks CSV only US packs have. The docstring says "every registered city". Require ids, or take
the US and Canadian hand configs only. It also runs on whatever is in data/packs, while
merge_cells.py says enrichments run on the res-9 pack (`--undo` first). A one-line note in its
docstring would do. Risk: none. Effort: tiny. Mechanical: yes.

---

## Batch F: shared pipeline helpers (wave wait; byte-identical, review closely)

**Items 17 and 18 done 2026-10-24** (tree `tw/pipeclean`; 19 and 20 not started). `simbin.py`
(`env`, `run`, `exe`, root at call time) replaces the copies in build_city, coarse_zones,
intercity, market_zones, routes and zones. `world_run.build_everyday` keeps its own: it is a `cargo
test --no-run` with `--jobs` forced, cargo's default target (the gate's own test build shares it)
and a missing cargo that is not fatal. `packio.py` (`write`, `write_header`,
`replace_cell_arrays`, `SIZES`) replaces the writer in build_city, canada, merge_cells, nocar,
routes, everyday_rates and the six enrich_*; tier0 and canada call its `replace_cell_arrays`;
pack_gate's inline size table is `packio.SIZES`; deploy.py's (a superset) stays.
Proofs, all in scratch: every sim launch caught at `subprocess.run` (build_city, coarse_zones on a
copy of mea's res-9 pack, intercity, zones, routes, market_zones; no TW_* variables, a TW_SIM_DIR
copy, CARGO_BUILD_JOBS set): 23 of 24 identical in command, environment and keywords; the 24th,
intercity with a forward-slash TW_SIM_DIR, spells the same manifest `D:/x/sim\Cargo.toml` where it
wrote `D:\x\sim\Cargo.toml`. The writers: two scratch trees (HEAD's pipeline, this one) on copies of
mea, ceu, mcu and let, each `merge_cells --undo`, routes.write_arrays and an enrichment's
replace-and-write, `merge_cells`, `everyday_rates`: all 77 files under data/ byte-identical (packs,
res-9 copies, stamps), HEAD's pre-T509s4 backups the only extra files.

Owned files: a new `pipeline/simbin.py` and `pipeline/packio.py`, plus the call sites named
below. Do not put the pack writer in read_pack.py: its docstring keeps it apart from the writer
on purpose ("so it tests the format, not the writer").

**17. One way to run a sim binary.** There are seven copies of the same block: PATH plus
`~/.cargo/bin`, a `CARGO_BUILD_JOBS` default, `CARGO_TARGET_DIR` from `TW_CARGO_TARGET_DIR` or
`ROOT/data/work/target`, `shutil.which`, the "cargo not found" exit, and the manifest from
`TW_SIM_DIR`. They are at:
- `build_city.py:1094-1110` (`add_gravity`)
- `coarse_zones.py:42-53`
- `market_zones.py:348-363` (`cargo_exe`, build only)
- `routes.py:685-700` (`cargo_run`)
- `intercity.py:365-377` (`cargo_bin`)
- `zones.py:49-57` (`local_trips`, with `cwd` and `capture_output`)
- `world_run.py:774-805` (`build_everyday`)

`pack_gate.py:660` runs `cargo test` its own way. Make `simbin.env()`, `simbin.run(name, args,
**subprocess_kw)` and `simbin.exe(name)`, each taking the tree root at call time (so `--root`
behaves as now). The only difference among the copies is the `CARGO_BUILD_JOBS` default
(2 everywhere, 3 in market_zones); keep a parameter. Saves about 70 lines. Risk: low (a wrong
env breaks a step loudly, never quietly). Effort: small to moderate. Mechanical: yes, with one
`--root` scratch row run after.

**18. One pack writer.** Thirteen copies of the same pattern: write `<stem>.bin.part`, write
`<stem>.json.part` with `json.dump(header, f, indent=2)`, then replace .bin and then .json. The
copies are at:
- `build_city.py:1066`
- `canada.py:635` (`rewrite`)
- `enrich_universities.py:198`
- `enrich_everyday.py:49`
- `enrich_schools.py:84`
- `enrich_airports.py:282`
- `enrich_parks.py:309`
- `enrich_medical.py:310`
- `merge_cells.py:446` (`write_pack`)
- `nocar.py:481`
- `routes.py:676` (`write_arrays`)
- `everyday_rates.py:98` (header only)
- `group_quarters.py:237` and `regions.py:88` (both go in item 11)

Every copy uses the same `json.dump` arguments (ASCII, indent 2), so `packio.write(stem,
header, buf)` writes the same bytes. Move `education_sources.replace_cell_arrays` (line 95; used
by enrich_schools, enrich_everyday and enrich_universities) into packio too. There are four
dtype-size tables:
- `read_pack.DTYPES`
- `education_sources.py:98`
- `group_quarters.py:262,295`
- `deploy.py:67`

`pack_gate.py:909` is a fifth, inline. One table in packio would do, but leave read_pack's own,
as above. Saves about 80 lines. Risk: low, and it is byte-identical if the dump arguments and
the order of the two writes are kept. Effort: moderate. Mechanical: yes. Verify with a byte
compare of one enriched or merged pack against a copy made before.

**19. One streaming download.** `routes.download` (`routes.py:88-104`), `osm_water._download`
(`osm_water.py:59-72`) and `world_run.Run.downloads` (`world_run.py:900-927`, with four tries) are
the same `requests.get(stream=True)` into `.part` and then `os.replace`. Take world_run's version
with its retries, as `download(url, path, tries=1)`. The same `UA` dict is defined 8 times
(arcgis, build_city, osm_water, overpass, places, routes, world_plan and world_run). Leave the
constants per module (see Redirect above); only the function needs sharing. Risk: low. Effort:
small. Mechanical: yes.

**20. Frame conversion helpers.** The pack frame's
`R_EARTH * radians(lon - lon0) * cos(radians(lat0))` and its inverse are written inline about 25
times. `R_EARTH = 6371008.8` is defined in build_city, read_pack, realnet, realnet_n02, routes,
water, pack_gate and t700, plus `EARTH_R` in zone_boxes. `water.to_local` and `water.to_lonlat`
already exist (`water.py:29-43`). Two cases:
- **Readers** (look_pack, look_water, read_pack.check_water, intercity.sites, market_zones.equirect,
  pack_gate lonlat and to_xy, t700): switch them to the shared helpers. Mechanical; no pack bytes.
- **Build paths** (build_city.py:1239, canada.py:550, tier0.py:254,952, japan_zones.py:204,
  fit_decay, routes.local_xy, realnet.to_xy): byte-identical only if the helper keeps each site's
  exact operation order. `enrich_parks.py:219` multiplies in a different order, and
  `hospital_beds.py:129` and `calib_seo.py:90` use 111,320 m a degree. Leave those three as they
  are. **Could change pack bytes in the last bit**: not mechanical. Do these only with a rebuilt
  row compared bit for bit, or skip them.

Risk: low for readers, moderate for build paths. Effort: moderate. Value: low to moderate.

---

## Batch G: the unread `nocar` layer, T-878 then T-926 (between batches; changes pack bytes)

Owned files: world_run.py, everyday_rates.py, nocar.py, usa.py, merge_cells.py comments,
tools/aws/userdata/proof.sh, docs (code-map:395, adding-a-city, data-register "Car
ownership"), and Credits.tsx (app side, T-926). **Changes pack bytes**: new packs lose the
`nocar` array and header block. Demand reads neither (`api.rs:770`, T-854), and no check reads it
(grepped checks/ and app/src), so game figures do not move. Not mechanical.

**21. T-878: drop the build step.** `world_run.py:83` (`BUILD_STEPS`), `:941`, and docstring
line 35. Before nocar can go, `everyday_rates.country_of` (`everyday_rates.py:88-91`) needs a new
home for `nocar.country_of`, `plan_country`, `PACK_COUNTRY`, `NATIONS` and the `worldbank()`
lookup (`nocar.py:104-139`): move them into everyday_rates or a small `countries.py`. Don't merge
them with `enrich_medical.country_of` (`enrich_medical.py:167`) or `intercity.country_of`
(`intercity.py:194`). Those answer differently: the config's `iso2` first, and "US" when nothing
is set. Unifying would change the `everyday` block or bed counts of some packs. The runner also
stops fetching nocar's ACS B08201 and TS045 sources for every US row.

**22. T-926: strip built packs, then delete `nocar.py`** (532 lines) and `notes/t562_fit.py`.
Do it as a one-off rewrite of data/packs and data/work/res9, outside a batch, with the
`pre-T926` backup CLAUDE.md asks for. Array offsets shift, and `merge_cells`' stamp in
`res9/<id>.merged.json` must be rewritten too, or the next merge refuses the pack
(`merge_cells.py:470`). Also: `usa.py:225` (the comment saying nocar reads the block table: the
table stays, since nocar's `country_of` test and enrich_everyday use it) and the import in
`tools/aws/userdata/proof.sh:59`.

---

## Batch H: build_city's Japanese metro configs (wave wait; byte-neutral)

**Batch H done 2026-10-24** (tree `tw/pipeclean`): `_japan_metro` builds the seven configs; its
lambdas still read `CITIES[cid]["prefectures"]` and `["neighbours"]` when they run, as before.
The water default was not done. Proof: every one of the 2,096 `CITIES` configs dumped from HEAD and
from this tree (keys in order; plain values by repr; callables by bytecode, constants, names and
closures, and those reaching modules through `__import__` by the calls they make on stand-in
modules): 33,096 values identical, the metros' 49 lambdas identical by their calls, none differing.

**23. One helper for the seven Japanese metros.** `build_city.py:418-601` (tyo, osa, ngo, fuk,
spk, sdj, hij) are about 25 lines each. They repeat the same nine lambdas over
`CITIES[cid]["prefectures"]` and `["neighbours"]`, the same `jobs_url`, `iso2`, `latin_names`
and `kana_romaji`. `_japan_zone` (`:639-666`) already does this for the 38 zones. A
`_japan_metro(cid, origin, prefectures, neighbours, water_extracts, extract, telework, decay,
region_name, boundary_notes, **extra)` with each city's own extras (`od_share`) would cut about
120 lines. Keep every key as it is today. Note the metros have no `utc_offset` and the zones set
9; do not add it as a side effect. Verify by dumping each config's non-callable items, and each
lambda's target function and arguments, before and after. Risk: low (a key dropped would change
a rebuild, so diff the dicts). Effort: moderate. Mechanical: yes, with the dict diff. Value:
moderate, and only while these hand configs are kept ("built cities are not fixed": a later
replan may turn them into plan rows anyway). The repeated `"water": {"cell_m": 25.0,
"min_width_px": 2}` in all 27 hand configs could become a default in the same pass.

---

## Batch I: tools/ (wave wait for aws; deploy.py any time)

**24. `tools/aws/userdata/proof.sh` and `launch.py --kind proof`.** This is T-1064's one-off
proof (T-1064 is still open, TASKS:202). Remove it when T-1064 closes. It imports nocar, so it
breaks after item 22. Risk: none. Effort: tiny.

**25. `tools/deploy.py`.** It is clean. Its `DTYPE` (line 67) could take packio's table
(item 18). Its `json.load(open(...))` calls (lines 96, 119, 128, 139) leave files to the
garbage collector. That is harmless in a one-shot script; leave it.

---

## Considered and kept

- **The calibration family**: calib.py, calib_lon, par, sdj, seo, tor, tyo; about 1,700 lines.
  It has been dormant since T-636, but T-621's archive line says "Its tooling and census walk
  targets stay", and code-map documents it. Keep it unless Anita retires it.
- **`fit_decay.py`** (used by notes/t1018/us/lodes_mean.py) and **`lumps.py`** (T-043 open).
- **`japan_zones.py`**: the producer of `japan_zones.json`, which build_city reads. T-462 is open.
- **world_run.py (1,414 lines) and world_plan.py (2,106 lines)** are long, but each section does
  one job. `zone_groups` (280 lines) and `plan_zones` (250) are the longest functions. Splitting
  either while waves run and replans continue risks the plan's output for no gain. Revisit when the
  US is done.
- **pack_gate.py (1,108 lines)**: sections are clear. The checks review found small duplication
  with read_pack and three slips:
  - a pack is read twice: `fig_read_pack` (pack_gate.py:163-171) calls `read_pack.check`, which
    loads the pack, and `figures` (397) loads it again;
  - `Mask` (130-151) is a vector copy of `read_pack.water_mask`/`is_water`;
  - `registry()` (1 MB) is re-read per pack in `changed_since` and `joint_borders`;
  - `fleet()` (1064) writes `"packs": len(reg)`, the registry size, not the packs with results;
  - the browser check's expected count, 8, is hardcoded at 562 and 858. If city-browser gains a
    check, every pack parks. Have city-browser write its own expected count into its JSON.

  All of these wait for the wave; low risk, low to moderate value. The hardcoded 8 is the one
  worth doing first.
- **The registry read in 20 places**: most go through shared_lock's locks. A helper would not
  shorten much and touches the locking. Leave it.
- **h3 resolution masks** (`>> 52 & 0xF`): only 6 places. Leave them.

---

## Bugs (separate from cleanup)

- **B1. Wave 3's runbook sets `TW_FAULT_S=900`** (`notes/t1018/us/mkrun.py:14`,
  `notes/t1018/us/join.cmd:6`). This is urgent: tell the T-1018 agent. That setting killed ves's
  gate at 900 s in T-903 (notes/T-903.md:26-27: "made the gate process exit when its stack dump
  fired mid-GEOS call. Runs since go without TW_FAULT_S"). It killed G2's region-2 gate at 903 s
  in T-1078 too (notes/T-1004.md:631). The runner's own `quiet_watch` (world_run.py:225-243)
  already dumps stacks after QUIET_S, 1200 s of log silence. It uses the same
  `faulthandler.dump_traceback_later(..., repeat=True)`, so a gate silent for 20 minutes may die
  the same way. Unexplained, so it needs a debug task, not a cleanup edit. Meanwhile, drop
  TW_FAULT_S from the wave cmds.
- **B2. T-594**: `intercity.py --help` runs the full Northeast solve (item 13).
- **B3. T-980**: pack_gate stdout on a cp1252 console under `-I` (item 14).
- **B4. `data/work/pre-T509s4` grows** with every built pack (item 15).
- **B5. `regions.py <city>`**, which code-map recommends for a re-solve, likely drops coarse blocks
  and leaves routes stale on today's packs (item 11). Not run, so unconfirmed; the fix is the doc
  line either way.
- **B6. `notes/t314_loo.py`, the producer of `fitted_curves.json`, imports the deleted
  `t162_fit`** (line 174, deleted in T-940's commit 464f70f), so its US `setup` stage fails.
  t162_tables, t171_gravity and t572_measure are broken the same way (they get deleted in item 1).
  Fixed 2026-10-24: `pipeline/curves/t162_fit.py` holds T-162's `City` (zones, LODES OD), all
  `setup_us` reads; the grid fit and `main` stay in git history. Not run (needs US blocks and OD).
- **B7. `enrich_all.py` with no ids** runs every config, tier 0 and planned rows included (item 16).
- **B8. `t800-outlines.mjs` should fail since T-975**, and **`t801-no-grey.mjs:91` always passes**
  (items 5-6).
- **B9. `t066`/`t149` worker checks** load all 1,667 zones in one worker (item 4). This is likely
  but unconfirmed, since they were not run. `t883-fare-curve.mjs` also sets `window.tw.ui?.shade`,
  which does not exist (`main.ts:558-585`; use `window.tw.shade.value`). That file is mid-edit by
  another agent: tell them.
