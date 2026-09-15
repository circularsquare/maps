# Geography playbook

Traps between a religion table and the dots that every country meets, whatever its source: boundaries, population
bases, joins, placement and the build tail. Survey and census-table traps are in their own playbooks.

## Boundaries and population base

- **A boundary file of the wrong vintage drops units without an error.** Use the vintage the table was published on;
  `boundaryYearRepresented` is a claim (geoBoundaries Mexico ADM2: 2,457 units against 2,469). Assert the unit count
  before joining, and where a publisher has two editions a year try both and let the leftovers pick. Caught by:
  `EXPECTED_*` counts per `<cc>_geo.py`, e.g. `sources/kr_geo.py::build_units`. Example: `de`. Detail: spec §8.1.
- **The general boundary sources can carry the wrong tier.** Look on the office's own site first (census atlas, a `gis.`
  host at `/server/rest/services?f=json`, its ArcGIS Online org); "excluding" row labels or HUCs mean a tabulation tier,
  which USCB country geodatabases carry. Caught by: Not checked yet (a search step; list the hosts tried in
  `sources/<cc>_geo.md`). Example: `md`. Detail: spec §12 "Joining to boundaries".
- **The drawn tier is newer than COD-AB.** A promoted unit is usually one level down in the same file: subtract it from
  its old parent and assert area against the census. Otherwise try statute-traced layers on ArcGIS Online; never dissolve
  on a stale file's parent codes. Caught by: `sources/tl_geo.py::main`, `sources/ao_geo.py::main`. Example: `tl`.
  Detail: spec §12 "When the drawn tier is newer than COD-AB, look one level DOWN in the same file first".
- **"ADM1" means different things in different files.** The Dominican ADM1 is ten planning regions (provinces are ADM2);
  São Tomé's districts are ADM1 in COD-AB and ADM2 in COD-PS, so `adm1_pcode` joins districts to provinces. Assert the
  count on each file separately. Caught by: `sources/do_geo.py::main`, `sources/st_geo.py::main`. Example: `st`.
  Detail: spec §12 "COD-AB AND COD-PS CAN DISAGREE ABOUT WHAT LEVEL A TIER IS".
- **A wrong polygon hides inside a correct national total.** geoBoundaries Iraq draws Baghdad at 912 km² against 4,555;
  Chad's COD moved two départements after 2009, by errors that cancel. Compare per-unit area with the office's area or
  density table. Caught by: `sources/iq_geo.py::main` (witness 2), `sources/td_geo.py::main` (`AREA_2009`). Example: `iq`.
  Detail: spec §12 "A BOUNDARY FILE'S ERROR HIDES INSIDE ITS CORRECT TOTAL"; "A CENSUS OLDER THAN ITS BOUNDARY FILE".
- **COD's capital polygon can be the old core.** COD-AB Belarus draws Minsk City at 86.8 km² against
  353.64, so the capital's outer districts fall in Minsk oblast and five of nine LiTS city PSUs land
  outside the city. Compare a capital's polygon area with the city's own figure, and put survey PSU
  coordinates in it. Caught by: `sources/by_geo.py::minsk_surgery` (OSM relation plus COD's polygon,
  area asserted), `::psu_witness`. Example: `by`. Detail: `sources/by.md` §5.
- **COD-PS is a projection, and a national match hides provincial error.** Dominican COD-PS is 0.56% off nationally,
  -23.7% to +10.3% by province. If the country counted after COD-PS's vintage, the office's count is the base. Caught by:
  `sources/do_geo.py::main` prints it per unit; not asserted there. Shared assertion for each `<cc>_geo.py` reading
  COD-PS: `sources/geo_checks.py::ratio_band` (tested on the Dominican figures; no builder calls it yet). Example:
  `ec`. Detail: `sources.md` §9bn; spec §12 "A COD-PS PROJECTION THAT AGREES NATIONALLY CAN BE WILDLY WRONG PER UNIT".
- **A boundary file's population column can be another collection or date.** GISCO `POP_2021` is 1 January 2020 for
  Norway and 0 for seven Skopje municipalities; a check failing with the sign of growth is a vintage mismatch. Assert a
  ratio band, or equality only against the office's own figure. Caught by: `sources/mk_geo.py::main`,
  `sources/md_geo.py::main` (`p_distrib`), `sources/ph_geo.py::main` (`RLG_HPOP`). Example: `no`. Detail: spec §12 "GISCO's `POP_2021`".
- **A second boundary source for part of a country fails an area test when right.** OSM's Karachi districts run out to
  sea (IoU 0.588 on a correct pairing): assert the grid people the second source leaves uncovered. Caught by:
  `sources/pk_2023_geo.py::main`. Example: `pk`. Detail: spec §12 "WHEN ONE PART OF A COUNTRY NEEDS A SECOND BOUNDARY SOURCE".

- **A boundary file newer than the census draws a later district as its own feature.** COD's Seychelles
  file has Perseverance Island, a district only since 2022, and puts six inner islands in `Other Islands`
  that the 2010 census counts with La Digue. The census's printed area per district says which unit absorbed
  each (English River 1.38 km² alone, 2.32 merged, 2.3 printed); drop grid cells built up after the count.
  Caught by: `sources/sc_geo.py::check_areas`, `::split_other_islands`. Example: `sc`. Detail: `sources/sc.md` §5.

- **A survey's capital stratum can be a commune the newer census no longer counts.** Togo's Afrobarometer
  samples the old commune of Lomé; the 2022 census counts 13 new communes, and COD-AB draws the old commune
  with two canton pieces of the same names left outside it. Rebuild the count from the new communes
  (Golfe 1-5) and let Kontur pick the polygon against the national ratio (0.90x on the commune alone, 1.12x
  with the pieces, 1.13x nationally). Caught by: `sources/tg_geo.py::main` (`LOME_KONTUR_TOL`). Example:
  `tg`. Detail: `sources/tg.md` §3.

- **A district reform after the census can usually be undone one level down.** COD-AB Sierra Leone
  draws 2017's Karene and Falaba but keeps the pre-reform chiefdoms whole, so the census's 14 districts
  come back by pcode. Assert each district's chiefdom count against the census's own chiefdom table, and
  use an older release (geoBoundaries' 14) as an IoU witness. A small coastal unit can fail that on one
  island the older file leaves out (Tasso Island, 7.7 km2 of an 82 km2 district, IoU 0.878): exclude the
  named ward from the comparison and assert its name, rather than lowering the bar for every unit. Caught
  by: `sources/sl_geo.py::main` (`EXPECTED_CHIEFDOMS`, `WITNESS_EXCLUDE`). Example: `sl`. Detail:
  `sources/sl.md` §5.

- **An area test between two traced layers can pass where people are misplaced and fail where
  nobody lives.** Brunei's geoBoundaries districts and mukims are separate Wikimedia tracings.
  Belait passed an IoU test at 0.942 while one 42.5 km2 coastal strip holding 29,972 Kontur people
  lay outside every Belait mukim; Brunei Muara failed at 0.831 on river that the district tracing
  leaves out. Compare Kontur people per unit on each layer against the census (districts 0.96x to
  1.01x; mukims dissolved to districts 0.71x for Belait, 25,130 people in no district), and keep the
  worse layer for membership only. Caught by: `sources/bn_geo.py::main` (membership by majority
  area, the per-district Kontur band); the two-layer comparison is a scratch script, Not shared
  yet. Example: `bn`. Detail: `sources/bn.md` §5.

## Joins

- **A join can lose a whole region and look fine.** Report unmatched rows, unmatched polygons and matched codes with
  empty geometry (Australia's have NaN area, not 0). Poland's files both hold 2,477 units and match zero as delivered.
  Caught by: a bijection assertion per `<cc>_geo.py`, e.g. `sources/tl_geo.py::main`; at scatter time
  `sources/geo_checks.py::check_unplaced`, from `scatter.py::main` (a counted unit with no polygon stops). Example: `pl`.
  Detail: spec §8.1.
- **A shared code shape is not a shared code.** Sri Lanka's pcode matched 13,472 of 14,003, and 13 renumbered divisions
  paired with real polygons elsewhere (762,824 people). Align the coarser level by name, match codes inside it, and use
  the unjoined key as the check. Caught by: `sources/ec_geo.py::check_code_join`, `sources/do_geo.py::check_code_join`.
  Example: `ni`. Detail: spec §12 "The shapes of failure that cost the most", item 2.
- **The wrong same-named twin passes every totals check.** Serbia's two Palilulas, Ghana's `TMA`, China's three `WEIXIAN`.
  Derive names that repeat on either side and qualify them by parent; fold only within a parent, 1:1; where both lists run
  in code order, a match outside its file neighbours' parent is the wrong twin. Caught by: `sources/rs_geo.py::_keys`,
  `sources/bj_geo.py::main` (rank moves), `tools/check_cn_prefecture.py::main`; shared form for each name-joined
  `<cc>_geo.py`: `sources/geo_checks.py::file_neighbour_outliers` (flags the same 24 Chinese counties as the China tool;
  no builder calls it yet). Example: `rs`. Detail: spec §12 "Joining on NAMES, where there is no code"; §14.19.
- **A join needs a witness that neither key determines.** A parent column read independently (Ghana's region against row
  order), the office's areas, or share smoothness between neighbours against shuffles. It must not assume where a group
  lives (Peru's altiplano witness fired on a correct join). Caught by: `sources/gh_geo.py::_verify_region`,
  `sources/pe_geo.py::main`, `sources/ni_geo.py::main`. Example: `pe`. Detail: spec §12 "The shapes of failure", item 2.
- **A duplicated key is a grouping instruction or a multiplied weight.** geoBoundaries Vietnam ADM1 has 64 features for
  63 `shapeISO`: dissolve, never `drop_duplicates()`. Multipart setores on one population row got five times the pull.
  Caught by: `sources/br_setores.py::main` (duplicate setor codes). Example: `vn`. Detail: spec §12 "Joining to
  boundaries"; §8.2d.
- **A unit with no polygon fails somewhere else.** Fall back to the parent's unmatched remainder, not the whole parent
  (Kalmunai). Korea's missing Yeonggwang made a city 500 km away fail; overlapping ADM1 enclaves broke point-in-polygon
  parents. Caught by: `sources/kr_geo.py::patch_hole`, `::build_units`. Example: `kr`. Detail: spec §12 "When a unit has no polygon".
- **A borrowed sub-layer for a capital can agree on area and names and be wrong.** Benin's arrondissements matched Cotonou
  to 1.3% at IoU 0.729; the capital stayed one polygon. Caught by: Not checked yet (an IoU assertion belongs in the
  `<cc>_geo.py` that borrows it). Example: `bj`. Detail: spec §12 "Capitals and sub-city geography".

## Placement

- **A Kontur r8 hex is about 0.74 km², not 0.16.** Divide the median unit area by 0.74 before a `place_weight`: single
  digits (median unit under 5-7 km²) or many hexless units means place uniformly, and say in `sources/<cc>_geo.md` it was
  measured. Caught by: `sources/geo_checks.py::check_grid_floor`, from `scatter.py::main` for any weighter on a grid layer
  (warns under a median 10 cells per unit or over 10% zero-weight units; `geo_checks.csv` `accepted` quiets it). Example:
  `vc`. Detail: spec §8.2e.
- **A unit missing from the `place` layer is not drawn on its polygon.** Its dots are allocated and never placed, and the
  carry hands fractions to it. Append the unit's own polygon at census population and assert the place layer's unit
  count. Caught by: `sources/geo_checks.py::check_unplaced`, from `scatter.py::main` (stops unless `sources/geo_checks.csv`
  names the unit); `sources/cy_grid.py::main` (Akrotiri). Example: `cy`. Detail: spec §12 "Choosing a placement layer".
- **A Kontur extract can be empty, or too coarse for dense units.** The `BQ` extract opens with zero features; the global
  r6 file gave four dense Serbian municipalities no hex. Use the per-country r8 extract and count features on read. Caught
  by: the `ZERO features` stop copied into most `*_grid.py`, e.g. `sources/vu_grid.py::main`; new builders call
  `sources/geo_checks.py::read_layer` instead; an empty place layer also stops `check_unplaced`. Example: `bq`. Detail:
  spec §12 "A KONTUR EXTRACT CAN BE A VALID FILE WITH NOTHING IN IT".
- **A Kontur/census band may not tell a right join from a shuffled one.** Benin's alike communes pass it shuffled;
  Zimbabwe's uneven provinces defeat the correlation. Measure both nulls, assert the one that discriminates, and check an
  enclave city with its ring. Caught by: `sources/bj_grid.py::main`, `sources/zw_grid.py::main`, `sources/lt_geo.py::report`.
  Example: `zw`. Detail: spec §12 "Choosing a placement layer".
- **A band failure is often the grid's fault.** Kontur put 43% of Eswatini where the census has 19%; Singapore's grid
  counts non-residents; three grids agreeing on Haiti's communes was one lineage. Test a second raster or COD-PS a tier
  down, and scale a false commune rather than cap it. Caught by: `sources/sz_grid.py::main`, `countries.py::_HT_COMMUNE_LEVEL`.
  Example: `ht`. Detail: spec §12 "Choosing a placement layer"; "THREE GRIDS AGREEING ON A COMMUNE CAN BE ONE ERROR".
- **Kontur's 46,200/km² cap makes false cities.** A capped block can hold most of a unit's weight off-centre (Tashkent
  58%, 15.6 km south), and real cores hit the cap too. Run `python kontur_cap.py <cc>` before scattering and add a
  `kontur_cap.csv` row per block (`real`, `capped`, `unreviewed`); an unlisted block at the cap stops the scatter. Caught
  by: `kontur_cap.py::apply`, from `scatter.py::main`. Example: `uz`. Detail: spec §12 "KONTUR'S DENSITY CAP MAKES FALSE CITIES".
- **A town list can be too thin to review cap blocks against.** GeoNames gives Yemen 9,134 places with
  a population and none for district towns such as Lawdar and Ja'ar. Where the population table is by
  district, pair each block's peak with its district: a block of two to four hexes holding most of a
  rural district's people, with no settlement of known size within 5 km, is false. Caught by: Not
  checked yet (scratch review; the method and the 69 rows are in `sources/ye.md` §9). Example: `ye`.
- **Kontur can put a fifth of a country in the wrong county, and a province-level check cannot see it.**
  Iran's grid drew Sarvestan (census 38,114) at 1.78 million and Shiraz county at two-thirds of its census,
  18.3% of Iran in the wrong county, with 78 blocks at the cap; capping blocks leaves the false weight in
  their ramps. Where a census county table exists on the boundary file's pcodes (COD-PS ADM2 is often the
  census itself: assert its sums against the counted units), compare Kontur per county and, if it is off,
  calibrate each hex to its county total. Caught by: `sources/ir_geo.py::read_counties`, `::main` (step 4);
  no shared helper. Example: `ir`. Detail: `sources/ir.md` §5.
- **A Kontur extract can stop at a claimed border, and the grid holds people the census leaves out.**
  Kontur's `PS` extract has no hexes in the part of Jerusalem Israel annexed (the `IL` extract holds
  350,549 people there), and both grids model the Israeli settlements, which the Palestinian census
  does not count (Ramallah & Al-Bireh read 1.85x the census before correction). Read every extract
  that touches a disputed line and de-duplicate on `h3`; take a population the census excludes out of
  the weights with the other side's own counts, and size a disc for any unit that is only a
  placeholder polygon (CBS draws 119 small localities at 0.008 km2). Witness it with a third point
  layer: the weight near the census's own communities must rise. Caught by: `sources/ps_geo.py::main`
  (the per-extract table, witness 4), `::settlement_units`. Example: `ps`. Detail: `sources/ps.md` §6.
- **Administrative units own water.** `water.py::clip` removes OSM's sea; a unit losing over 95% (`KEEP_WHOLE_ABOVE`)
  stays whole. Lakes are not removed: count dots in HydroLAKES, and read a cover of one polygon per unit (Bangladesh 544
  for 544) as water inside units. Caught by: `water.py::_keep_whole`; lakes only `sources/gh_geo.py::_drop_lakes` (a
  second country moves it into `water.py`). Example: `gh`. Detail: spec §8.2c, §8.2c-i.
- **Hex centroids fall just offshore of island units.** Dropping them moves weight inland (Vanuatu, 13.25% of people).
  `sjoin_nearest` with `max_distance` in a projected CRS (in EPSG:4326 the distance is degrees and every cap passes).
  Caught by: `sources/vu_grid.py::main` (`SNAP_M = 500`); no shared helper. Example: `vu`, `nl`. Detail: `sources.md` §9bg, §9cu.
- **The antimeridian tears geometry silently.** Reprojection tears Fiji's provinces, nine Kontur hexes ship torn, and a
  Pacific CRS moves the tear. Shift negative longitudes +360; assert the bounding box per country, or per feature where the
  country straddles 180 (Kiribati). Caught by: `sources/geo_checks.py::check_torn`, from `scatter.py::main` after the
  reprojection (any polygon part over 180° wide stops); `sources/fj_grid.py::main`, `sources/ki_geo.py::build` where the
  layer is built. Example: `fj`. Detail: spec §12 "The shapes of failure", item 4.

## Build

- **A country missing from `country_shapes.py` hands its legend to a neighbour.** Auto falls back to the dot tally. The
  script stops on a registered country Natural Earth lacks: add `ISO` or `FROM_UNITS` (a territory inside its sovereign)
  before registering. Caught by: `country_shapes.py::main`. Example: `bq`. Detail: spec §12 "Finishing".
- **A territory inside its sovereign's data can draw with no outline.** Australia's Norfolk,
  Christmas and Cocos SA2s are in `au.csv` with dots, but `country_shapes.py` emits one Natural Earth
  feature per code (`if cc in seen: break`), so `Indian Ocean Ter.` (`ISO_A2_EH` AU) is skipped and
  Norfolk Island is a separate `NF` feature nobody registered. Åland is the same shape inside `fi`
  (FI200 in `fi.csv`, an unregistered `AX` feature; sources.md §scout-2026-09-15-europe). List the Natural Earth features a
  country's dots fall in before calling it built. Caught by: Not checked yet (a dots-outside-outline
  count belongs in `tools/built_countries.py --check`). Example: `au`. Detail: `sources.md`
  §scout-2026-09-14-asia-oceania.
- **The tail rewrites whole-map files.** Run `tools/build_tail.py --id <sid>` (lock, derived country list, `--coarse`,
  `coverage.py` last) after scattering both editions; a lattice on a page open mid-build is not corrupt data. Caught by:
  `tools/build_tail.py::main`, `tools/built_countries.py --check`. Example: `gh`. Detail: spec §12 "Finishing"; `COMMANDS.txt` 10-12.
- **After a node rename the checks read the last build.** `coverage.py` reads `counts.json`, so before the tail it fails
  on the old node. `buffers.py` only warns `node(s) not in religions.json`: run `taxonomy/build_tree.py` first. Caught
  by: `tools/build_tail.py::main` (coverage), `buffers.py::main` (warning). Example: `mz`. Detail: spec §12 "`coverage.py` READS `counts.json`".
- **A two-level tier or a second vintage breaks the checks.** `tools/check_mapping.py` defaults to the level with most
  units (Ghana 1.7M short; use `DEFAULT_LEVELS`). `sources/pk2023.py` shadowed `taxonomy/pk2023.py` inside `tiles.py`:
  name it `pk_2023.py`. Caught by: `taxonomy/registry.py::discover` (two vintages need `OVERRIDE`); shadowing by
  `sources/geo_checks.py::check_module_shadowing`, from `scatter.py::main` (a name repeated across the top level,
  `sources/`, `taxonomy/`, `tools/` or the standard library stops). Example: `pk`. Detail: spec §12 "WHEN ONE PART OF A
  COUNTRY NEEDS A SECOND BOUNDARY SOURCE".
- **A correct legend can sit over a country that draws nothing.** Look in the default (separate dots) mode; headless needs
  `--enable-unsafe-swiftshader` and no `--disable-gpu`. Confirm tiles with `tools/check_tiles.py`, not a mid-write
  screenshot. Caught by: Not checked yet. Example: `gh`. Detail: spec §12 "Finishing".

## Shared code

Import these, do not copy them.

- `scatter.py` (Hilbert carry, then `kontur_cap.apply` and `water.clip`; `--no-weights`, `--no-water`, `--no-kontur-cap`).
- `kontur_cap.py` with `kontur_cap.csv`; `water.py::clip` with `KEEP_WHOLE_ABOVE`.
- `countries.py::_kontur_place_weight` (any place layer with `pop`); `_micro_place_weight` for one-unit countries.
- `country_shapes.py` (`ISO`, `FROM_UNITS`, `SPLIT`, `CLIP`); `tools/build_tail.py`, `built_countries.py`, `check_tiles.py`.
- `sources/geo_checks.py` with `sources/geo_checks.csv`: `read_layer` (zero features), `ratio_band` (a second population
  source per unit), `file_neighbour_outliers` (wrong twin), and the scatter-time `check_unplaced`, `check_torn`,
  `check_grid_floor`, `check_module_shadowing`. `scatter.py::read_place` loads the place layer as the scatter does.

Still no shared form, so copy from the worked example: name folds, the Kontur band and shuffle null, `sjoin_nearest` snaps.

## Rulings

- Anita 2026-09-06 (spec §3.9b): no minimum unit count. Does not require a finer tier that loses categories or vintage.
- Anita 2026-09-08 (`countries.py::_micro_counts`): very small island countries may be one unit. Sets no size cut-off.
- Anita 2026-09-03 (spec §4.1a): leftover dots follow a Hilbert carry, never the top n. Already in `scatter.py`.
- Anita 2026-09-05 (spec §8.2c-i): `KEEP_WHOLE_ABOVE = 0.95`, a known compromise. Does not rule out a per-region value.
- Anita 2026-09-06 (`country_shapes.py` `CLIP`): the wash leaves out territory the source does not cover. The wash only.
- Anita 2026-09-08 (spec §14.18): disputed land goes to its de facto administrator. Ukraine's occupied oblasts stay drawn from pre-war rounds (ask 016).
- Anita 2026-09-08 (ask 003): mixed vintages in one country are fine if the method is sound. Tibet was left on size alone.
- Anita 2026-09-08 (ask 001): Egypt at governorate, the instrument's ceiling. Decides nothing for other countries.
- Anita 2026-09-14 (ask 014): Japan not left as one national unit; 1996 may allocate. Japan only.
- Anita 2026-09-14 (`queue.md` night): one Christian share for Zanzibar, for safety. Chad (017), Burkina Faso and Mali (018) drawn at the published grain.
- Kontur cap 2026-09-14 (`queue.md` evening): six blocks capped, method delegated; the listed follow-ups are undecided.
- Standing (`AGENT_BRIEF.md`, spec §14): whether a country may be drawn at all is Anita's; raise it, do not decide it.
