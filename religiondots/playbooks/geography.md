# Geography playbook

Traps between a religion table and the dots that every country meets, whatever its source: boundaries, population
bases, joins, placement and the build tail. Survey and census-table traps are in their own playbooks.

## Boundaries and population base

- **A census table can count displaced people where they come from, not where they are.**
  Azerbaijan's 2019 headline (permanent) column puts 268,468 people in eight districts nobody lived in
  (Kalbajar 71,039, every one `temporarily absent`); the same table's existing column counts people
  where the census found them. A dot is a place, so draw the de facto column, and read both columns of
  any table that has two. Caught by: `sources/az_geo.py::main` (prints units empty in the existing
  column and partly emptied ones). Example: `az`. Detail: `sources/az.md` §4.
- **An old church count can place people at their village of origin, not their home.** Kohler's
  1978 New Caledonia count puts urban Kanak at their home village (p.9), so Nouméa has none and
  Lifou is counted twice its residents. Before laying such a count on a census, scale each origin
  unit by the group's growth, treat any shortfall against the census as people who left (at the
  origin's mix) and give each unit's surplus the mix of all who left; the two pools must balance.
  Caught by: `sources/nc.py::kanak_seeds` (asserts the balance, prints the pool). Example: `nc`.
  Detail: `sources/nc.md` §2.
- **A current baseline on HDX can be off limits while it still downloads, and Kontur will not stand
  in for it.** Syria's 2025 Population Task Force baseline says in its notes it "cannot be shared for
  academic or research purposes"; the public needs files print no population; and Kontur sat within
  0.86-1.24 of the 2011 estimate per governorate (Idlib 1.02), because it spreads census-era admin
  totals. Read a dataset's notes before using it, and say in the note when the dots are pre-war
  positions. Caught by: Not checked yet (a reading step; `sources/sy_grid.py` prints Kontur per unit
  against the base). Example: `sy`. Detail: `sources/sy.md` §4.
- **A COD unit can hold land this map draws in another country, and the snap step puts it back.**
  COD-AB's Quneitra includes the Golan, which `il` draws. Subtract Natural Earth's disputed-areas
  polygon from the unit, and drop placement hexes inside it before snapping strays to the nearest
  unit, or a village a kilometre from the line (Majdal Shams) snaps back in. Caught by:
  `sources/sy_geo.py::main` (`GOLAN_KM2`), `sources/sy_grid.py::main`. Example: `sy`. Detail:
  `sources/sy.md` §5.
- **A geoBoundaries capital traced from old OSM can be nearly twice the city.** gbOpen TJK ADM1
  (OSM, 2017) draws Dushanbe at 366 km2; the city gives 203.18 km2 and OSM's current relation is 197.
  The extra 180 km2 is suburb the census counts in the Districts of Republican Subordination (Kontur
  260,000 people), and Kontur over the census read 1.72 against 1.46 on today's line. Compare a
  capital's polygon with the city's own area and the current OSM relation, and move the difference to
  its neighbour; `polygons.openstreetmap.fr/get_geojson.py?id=<rel>` serves one relation when
  Overpass times out. Caught by: `sources/tj_geo.py::main` (`DU_KM2`, `OLD_DU_KM2`, the overlap
  check). Example: `tj`. Detail: `sources/tj.md` §7.

- **A boundary file of the wrong vintage drops units without an error.** Use the vintage the table was published on;
  `boundaryYearRepresented` is a claim (geoBoundaries Mexico ADM2: 2,457 units against 2,469). Assert the unit count
  before joining, and where a publisher has two editions a year try both and let the leftovers pick. Caught by:
  `EXPECTED_*` counts per `<cc>_geo.py`, e.g. `sources/kr_geo.py::build_units`. Example: `de`. Detail: spec §8.1.

- **A NEWER boundary file joins perfectly and still leaves land blank.** Every census unit finds its polygon, but
  units carved out after the census have no row, so their ground is in no placement polygon and their people are
  drawn on the parent's shrunken polygon. Indonesia lost ~7.2M Kontur people's worth of land this way for a month.
  Give each post-census polygon to the census unit that held it (formation law for a new unit; Kontur fit for an
  internal split). Caught by: `not_drawn.py` (hatches it); `sources/id_geo.py::add_post2010_ground` stops on an
  unassigned polygon. Example: `id`. Detail: `sources/id.md`, fix batch 2026-10-03.
- **The general boundary sources can carry the wrong tier.** Look on the office's own site first (census atlas, a `gis.`
  host at `/server/rest/services?f=json`, its ArcGIS Online org); "excluding" row labels or HUCs mean a tabulation tier,
  which USCB country geodatabases carry. Caught by: Not checked yet (a search step; list the hosts tried in
  `sources/<cc>_geo.md`). Example: `md`. Detail: spec §12 "Joining to boundaries".
- **The drawn tier is newer than COD-AB.** A promoted unit is usually one level down in the same file: subtract it from
  its old parent and assert area against the census. Otherwise try statute-traced layers on ArcGIS Online; never dissolve
  on a stale file's parent codes. Caught by: `sources/tl_geo.py::main`, `sources/ao_geo.py::main`. Example: `tl`.
  Detail: spec §12 "When the drawn tier is newer than COD-AB, look one level DOWN in the same file first".
- **COD-AB's first-level lines can predate a reform while its unit count matches.** Oman's COD has
  the 11 governorates the register has, but draws their pre-2011 lines: 6 of 61 wilayat fall in another
  governorate and Ash Sharqiyah South is half its area (6,376 km2 against 12,033). The office's
  population over its own density map gives the areas when no area table exists; geoBoundaries' OMN
  ADM2 is NCSI's own 2020 wilayat. Before joining a unit count to COD, compare each unit's area with
  the office's figure and look for a layer traced from the office's own geography. Caught by:
  `sources/om_geo.py::main` (prints each wilaya's COD governorate and both layers' areas; the density
  comparison is in `sources/om.md` §5, not asserted). Example: `om`. Detail: `sources/om.md` §5.
- **"ADM1" means different things in different files.** The Dominican ADM1 is ten planning regions (provinces are ADM2);
  São Tomé's districts are ADM1 in COD-AB and ADM2 in COD-PS, so `adm1_pcode` joins districts to provinces. Assert the
  count on each file separately. Caught by: `sources/do_geo.py::main`, `sources/st_geo.py::main`. Example: `st`.
  Detail: spec §12 "COD-AB AND COD-PS CAN DISAGREE ABOUT WHAT LEVEL A TIER IS".
- **A census density table is an area table, and geoBoundaries ADM1 can be a merged tier.** Where no
  area table is printed, population over density gives the office's area per unit; divide each
  unit's polygon-to-census ratio by the national one first (Niger's official 1,267,000 km2 is 7%
  above any GIS polygon), after which Niger's rural régions sit at 0.96-1.02. The same test caught
  COD's Niamey at 557 km2 against INS's 255, which Kontur then showed moves 3.6% of the city's people.
  geoBoundaries gbOpen NER ADM1 has six features for eight régions (two merged pairs), so count its
  units before using it as a witness. Caught by: `sources/ne_geo.py::main` (`AREA_PINNED`,
  `NIAMEY_CORE_MIN`). Example: `ne`. Detail: `sources/ne.md` §5.
- **A wrong polygon hides inside a correct national total.** geoBoundaries Iraq draws Baghdad at 912 km² against 4,555;
  Chad's COD moved two départements after 2009, by errors that cancel. Compare per-unit area with the office's area or
  density table. Caught by: `sources/iq_geo.py::main` (witness 2), `sources/td_geo.py::main` (`AREA_2009`). Example: `iq`.
  Detail: spec §12 "A BOUNDARY FILE'S ERROR HIDES INSIDE ITS CORRECT TOTAL"; "A CENSUS OLDER THAN ITS BOUNDARY FILE".
- **COD's capital polygon can be the old core.** COD-AB Belarus draws Minsk City at 86.8 km² against
  353.64, so the capital's outer districts fall in Minsk oblast and five of nine LiTS city PSUs land
  outside the city. Compare a capital's polygon area with the city's own figure, and put survey PSU
  coordinates in it. Caught by: `sources/by_geo.py::minsk_surgery` (OSM relation plus COD's polygon,
  area asserted), `::psu_witness`. Example: `by`. Detail: `sources/by.md` §5.
- **COD's capital can follow a different line from the office's, at both edges.** COD-AB Uruguay's
  Montevideo meets INE's own 2011 department layer, which is the union of INE's 62 barrios, at IoU
  0.835: 70 km2 of INE's Montevideo (about 19,700 Kontur people) is in COD's Canelones, and 18 km2 of
  COD's Montevideo (about 12,100) is outside every barrio. A department-level build never shows it;
  the office's sub-city layer does. Compare COD's capital with the office's department layer before
  splitting it, and cut the edge hexes on the office's line: pieces outside it go to their COD
  department, COD's leftover to the nearest neighbour, each hex's people shared over its land
  pieces. Caught by: `sources/uy_grid.py::montevideo_edge` (people moved per department printed,
  a drop bar asserted), `::barrio_check`. Example: `uy`. Detail: `sources/uy.md` §12.5.
- **COD-PS is a projection, and a national match hides provincial error.** Dominican COD-PS is 0.56% off nationally,
  -23.7% to +10.3% by province. If the country counted after COD-PS's vintage, the office's count is the base. Caught by:
  `sources/do_geo.py::main` prints it per unit; not asserted there. Shared assertion for each `<cc>_geo.py` reading
  COD-PS: `sources/geo_checks.py::ratio_band` (tested on the Dominican figures; no builder calls it yet). Example:
  `ec`. Detail: `sources.md` §9bn; spec §12 "A COD-PS PROJECTION THAT AGREES NATIONALLY CAN BE WILDLY WRONG PER UNIT".
- **COD-PS that is the census re-cut can carry county errors into its admin 1.** North Korea's (WFP, 2008) drops
  Samchon county (86,042) and counts Sindo twice (on its own and inside Ryongchon), so two provinces are off by
  0.4% in all. Match every COD-PS admin 2 value to a row of the census table before using its admin 1; where they
  differ, rebuild the units from the census rows. Caught by: `sources/kp_geo.py::table2` (rows sum to each printed
  total) and its pinned `MOVES`; no shared helper. Example: `kp`. Detail: `sources/kp.md` §4.
- **A boundary file's population column can be another collection or date.** GISCO `POP_2021` is 1 January 2020 for
  Norway and 0 for seven Skopje municipalities; a check failing with the sign of growth is a vintage mismatch. Assert a
  ratio band, or equality only against the office's own figure. Caught by: `sources/mk_geo.py::main`,
  `sources/md_geo.py::main` (`p_distrib`), `sources/ph_geo.py::main` (`RLG_HPOP`). Example: `no`. Detail: spec §12 "GISCO's `POP_2021`".
- **A census microdata file's unweighted counts are not the published table.** Uruguay's 2023 persons file
  reproduces INE's barrio table within one person weighted, and 0.78 to 0.99 of it unweighted, lowest in the
  poorest barrios, where the omission correction was largest. Its weight is a bare `W`, which a search for
  *peso* or *pond* misses. Sum the weight, and assert the file against the office's own table per unit before
  taking an age split from it. Caught by: `sources/uy_geo.py::build_barrios` (per barrio against Cuadro 15),
  `::census_extract` (the national weight equals the census count). Example: `uy`. Detail: `sources/uy.md` §12.4.
- **Cutting a five-year band at one age removes the years below the cut, not above.** Uruguay's 7+ cut
  subtracts three fifths of the 5-9 band, which is ages 7 to 9 and inside the universe, where two fifths (5 and
  6) was meant: 46,154 people, 1.32% of the country, through two reviews. The census's own under-7 share
  exposed it (Montevideo 6.95%, 8.39% as built). Take the cut from a single-year table or microdata where one
  exists; otherwise write the arithmetic in ages. Caught by: Not checked yet (known wrong and left as drawn,
  `sources/uy_geo.py` `BAND_5_9_OVER_6`). Example: `uy`. Detail: `sources/uy.md` §12.7.
- **Two boundary files can disagree on a small urban unit's area and it can move nobody.** COD-AB
  Gambia draws Kanifing at 93.7 km2 and geoBoundaries at 52.9, against the census's 75.55, on a
  line through the Serekunda sprawl; Kontur holds 479,694 people in one and 479,679 in the other,
  because the difference is wetland. Before choosing a layer on area, count Kontur people per unit
  on both, and band each unit against the nearest count divided by the national ratio (Kontur ran
  1.16x Gambia's 2024 count). Caught by: `sources/gm_geo.py::main` (both layers printed, one
  banded). Example: `gm`. Detail: `sources/gm.md` §5.
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

- **A survey's unit totals can be its sample times one weight, not a population.** American Samoa's
  2015 HIES gives every sampled person the weight 5.99668, so its county totals are completion by
  county: 0.83x to 1.37x the 2010 census, and blind to the 2010-2020 changes the census counts. Lay
  the survey's shares on a census count per unit, and compare each unit's survey total with the
  census unit of the same name, which also catches a shifted column in a PDF read. Caught by:
  `sources/as_geo.py::main` (`SURVEY_OVER_CENSUS_2010`). Example: `as`. Detail: `sources/as.md` §5.
- **A census older than the boundary file can be rebuilt from units whose numbers outlived the
  reform, and weighted by its own unit populations rather than a modern grid.** Qatar's ten 2004
  municipalities are eight today, but the zones under them kept their numbers: COD-AB's 91 zones
  rebuild all ten from the census's zone table (87 zones; two merged since, six with nobody in
  2004), with the zone names as a witness the number does not decide (65 of 85 agree, 20 renames
  pinned) and Kontur's rank of the zones as a second (Spearman 0.876; 1,000 shuffles at most
  0.365). Weighting by the census's zone populations matters even with the unit counts fixed:
  plain Kontur 2023 would put 16.6% of the dots in a different zone of the same municipality,
  and 30.5% in Mesaieed, whose industrial-area zone held 9 people in 2004 and a fifth of the
  municipality's Kontur people now. Scale each zone's Kontur pieces to its census population and
  assert that every unit's weights sum to its count. Caught by: `sources/qa_geo.py::main`
  (`ZONES_MERGED`, `ZONES_EMPTY_2004`, `ZONES_RENAMED`, the rank witness, the weight sum).
  Example: `qa`. Detail: `sources/qa.md` §5, §6.
- **An office's census dashboard can serve the current boundaries COD-AB lacks, one parent at a time.**
  Uganda's COD-AB is the 2020 edition (1,520 subcounties); the 2024 census has 2,207, and
  `statistics.ubos.org/nphc`'s map API serves them per county with the census's own codes. Look for the
  dashboard's GeoJSON call before rebuilding units, sweep it paced, join on its codes and witness by name.
  Refugee settlements counted as subcounties may have no polygon (Bidi Bidi): merge each with the hosts in
  its county whose Kontur/census ratio stands out, and check the surplus against the camp's count. Caught by:
  `sources/ug_2024_geo.py::build` (`CAMP_HOSTS`, `POLYGON_INTO`). Example: `ug`. Detail: `sources/ug.md` §0.5.
- **A district reform can move posts between districts, so a newer district total is not a sum of
  whole older districts.** Mozambique's 2013 and 2016 reforms mostly raised one post to a
  district, but Anchilo moved into Nampula city, Maquival into Quelimane, two Lichinga posts into
  the city and two Mandlakazi posts into Chongoene. Rebuild the old districts from the posts
  (COD-AB adm3), and when fitting old shares to new populations, take as one row target each
  connected group of old and new districts that share posts. Witness it with growth: new
  population over old per group, against the province's own (Rapale alone 0.82 against 1.38,
  0.52 for old Lichinga district); where no new district table exists, Kontur over the old count
  does the same, more noisily (Nicoadala 0.55 without Maquival, 0.90 with it). Caught by:
  `sources/mz_2007.py::clusters`, `::growth_witness`, `sources/mz_geo.py::build_units` (posts tile
  each province), `sources/mz_grid.py::kontur_witness`. Example: `mz`. Detail: `sources/mz.md` §7.
- **A regional survey share can pass an outside witness at the region and fail it one level down.**
  India's Pew Northeast Catholic share read roll / survey 1.17 against the 2004 diocesan rolls for
  the region, well inside the bar, and 0.18 to 2.21 state by state (Nagaland to Assam), because the
  region's churches follow the tribe. Run the witness at the finest level it reaches, not the level
  the share is published at. When a region is dropped, rerun every between-region test too: the
  Northeast had been what made Catholics differ between the drawn regions. Caught by:
  `in_split_christian.py::check_pew`, check (d) (printed, not a guard). Example: `in`. Detail:
  `sources/in.md` §11, §12.

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

- **Where Kontur misses most of a country, place on the office's localities instead.** Kontur 2023
  holds ~20,000 of Greenland's 56,740; Statistics Greenland counts residents by town and settlement
  (`BEXSTD`), so `sources/gl_geo.py` puts a weighted disc on each locality, geocoded from GeoNames.
  **Match GeoNames on primary names only**: its alternate-name field matched dozens of places for
  some settlement names and snapped two settlements onto Ilulissat's coordinates; tell namesakes
  apart by distance to the district town. Caught by: `gl_geo.py`'s TOWN_KM stop (a settlement over
  300 km from its district's town) and the printed choice for every namesake. Example: `gl`.
  Detail: `sources/gl.md` §3.
- **A survey codebook can list regions its data never reached.** The Global Flourishing Study's
  codebook has Egypt's five frontier governorates (423-427); the file holds nobody there. Count the
  data's region codes, never the codebook's. Example: `eg`. Detail: `sources/eg.md` "Frontier
  governorates from the census".
- **Count a small region's respondents before chasing gated microdata; the documentation often
  prints the count.** GESIS variable reports give each region code's unweighted frequency (EVS 2017
  `v275b_N2`, EVS 2008 `v371c_N2/N3`, where the code list itself holds only codes present), and a
  Dataverse DDI XML carries `catStat` frequencies per category even when the data file answers 403
  (Sciences Po's ENEF 2024). A national sample puts about its population share in a small region:
  Corsica is 0.51% of France, so 25 of 10,631, and 0 in both EVS rounds. Example: `fr`. Detail:
  `sources/fr.md` §15.

- **A Kontur r8 hex is about 0.74 km², not 0.16.** Divide the median unit area by 0.74 before a `place_weight`: single
  digits (median unit under 5-7 km²) or many hexless units means place uniformly, and say in `sources/<cc>_geo.md` it was
  measured. Caught by: `sources/geo_checks.py::check_grid_floor`, from `scatter.py::main` for any weighter on a grid layer
  (warns under a median 10 cells per unit or over 10% zero-weight units; `geo_checks.csv` `accepted` quiets it). Example:
  `vc`. Detail: spec §8.2e.
- **Below the floor, cut the hexes to the units before giving up on the grid.** Malta's 68 localities
  have a median 2.94 km² (4 hexes) and 8 are smaller than one hex; a centroid join left 4 with none.
  Intersecting the hexes with the units and sharing each hex's people by area gave every locality 2 or
  more pieces (median 9) and a per-locality Kontur/census agreement of p10 0.68, p90 1.38, where Saint
  Vincent's centroid grid had 0.00 and 2.68. Set the keep-or-drop bar before reading the numbers and
  register `grid_floor` as `accepted` with them. Caught by: `sources/mt_geo.py::main` (prints both
  joins and the agreement; the bar itself is not asserted); `sources/xs_geo.py` cuts the same way.
  Example: `mt`. Detail: `sources/mt.md` §5. **Where a hex runs into the sea, share its people over
  the area of its pieces, not the whole hex.** Dividing by the whole hex gave the sea its share and
  dropped it: 5.5% of Malta's Kontur people, with coastal localities tilted inland (L-Isla read 0.28 of
  the census, 0.93 once shared by land). `mt_geo.py` now divides by the sum of each hex's piece areas.
  That is only right where every piece of land is in some unit; where the rest of a hex is another
  unit's land outside the file (`xs_geo.py`, `ps_geo.py`), dividing by the whole hex is correct.
- **An equal share per placement polygon gives an uninhabited island a whole share.** INEGI makes
  nearly every Mexican cay its own rural AGEB, so 48 of Progreso's 113 AGEBs were Arrecife
  Alacranes and 42% of Progreso's dots sat on the reef; 171 dots in all on islands the census puts
  under 250 people. Where placement is an unweighted equal share, split the country's land into
  connected pieces, put the office's locality points on them, and cut islands below a bar out of
  the layer (no unit may lose its last polygon). A cheap scan for the rest of the map: dots in no
  populated Kontur r6 hex and on no Natural Earth land part of 2,000 km2 or more (open the r6
  file with a `bbox` that carries its CRS; Kontur is EPSG:3857). Caught by:
  `sources/mx_geo.py::main` (`WITNESS`, `EXPECTED_DROPPED`); the scan is not shared yet. Example:
  `mx`. Detail: `sources/mx.md` §9.
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
- **Capping a Kontur block and then calibrating to a census table can build a worse false city.**
  Mauritania's one block at the cap held 86% of Toujounine's Kontur weight; lowering it to its 3 km
  ring's median (2,378/km2) left the block 17,989 people, and scaling the moughataa back to its
  census count pushed the edge hexes to 76,322/km2. Where a unit's own census count is the fix,
  scan and cap raw Kontur before calibrating (a calibrated layer above the cap is skipped by
  `kontur_cap.apply`), print the calibrated densest hex, and leave a block that holds most of a
  unit nothing finer describes. Caught by: `sources/mr_grid.py::main` (`BLOCKS`, the raw scan).
  Example: `mr`. Detail: `sources/mr.md` §5.
- **A false cap block in empty country can have no ring, and `capped` then stops the scatter.**
  Sudan's Red Sea hills have three blocks (up to 92,082 people in 3 hexes) with no populated hex
  within 3 km, and two more whose only ring hexes are in another block, so `capped` lowers them to
  the cap itself and changes nothing (the scatter printed a ceiling of 46,199/km2). Mark such a
  block `isolated` (status added 2026-09-15): `kontur_cap.apply` lowers it to the median density of
  its unit's populated hexes outside every dense block, which spreads the excess over the rest of
  the unit in proportion to Kontur, and refuses the status on a block that has a populated ring
  outside the blocks. Sudan's five went from 308,381 people to 69 (Red Sea state's median outside
  the blocks is 9.9/km2). Count each block's ring outside every block before writing rows. A
  `capped` ring can also be only partly another block's: Sudan's Tokar block had 2 of its 4 ring
  hexes in a neighbouring block (one that never reaches the cap, so it needed no row), its ceiling
  came out at 15,053/km2 and it kept 11,771 people. Mark such a block `capped_apart` (status added
  2026-10-03), whose ring leaves out every dense block, and give the neighbour a row too if it is
  part of the same artefact; Tokar's pair went from 79,526 people to 16,648. It is opt-in because
  folding it into `capped` would move ceilings in eg, gn, mg, ng, so, uz, ve and ye. Caught by:
  `kontur_cap.py::apply` (stops on a ring-less `capped` row and on an `isolated` row with a ring;
  prints how many of a `capped` ring's hexes are another block's). Example: `sd`. Detail:
  `sources/sd.md` §6 and §11.
- **Somalia's town test can call a block real when its ring already holds the town.** Sudan's
  Jubayt, Aroma and Rahad al Bardi blocks are 2.7-2.9x their GeoNames towns, under the test's bar
  of 3, but as `real` they drew 107-111 dots within 3 km against towns of 12,708-30,856; capped they
  draw 25-65. Count the dots within 3 km both ways before calling a block real on the town test.
  Not checked in code. Example: `sd`. Detail: `sources/sd.md` §11.
- **A Kontur extract overlaps its neighbours' at the border, and an old boundary file can leave a
  border town beyond the snap.** Kontur `SO` shares 457 h3 cells with `ET` and 273 with `KE`; COD-AB
  Somalia's 1984 line leaves Cabudwaaq (about 124,000 Kontur people) 2-7 km outside Galgaduud, so a
  2 km snap drew it nowhere, while 112,000 people within 2 km were already in Ethiopia's or Kenya's
  place layer and would have been placed twice. Class every hex outside the units by distance, by
  Natural Earth country, and by whether a drawn neighbour's place layer holds the same hex (centroid
  within 10 m): leave those to the neighbour, snap the rest inside the country's own Natural Earth
  outline from further, and put a named town witness on the far snap. Caught by:
  `sources/so_grid.py::neighbour_held`, `::town_witness` (`OUTSIDE_PINNED`); no shared helper.
  Example: `so`. Detail: `sources/so.md` §5.
- **Across a land border, a hex outside the units is the neighbour's town, so drop it, not snap it.**
  Kontur `BT` holds 116 hexes (72,617 people) with centroids just outside COD-AB Bhutan, 52,177 of them
  Jaigaon in India beside Phuentsholing; a 2 km snap put them in Chhukha (1.83x its census share, 1.31
  dropped). Snap only on coasts, and print outside hexes by nearest unit before choosing. Caught by:
  `sources/bt_grid.py::main` (`EXPECTED_DROPPED`). Example: `bt`. Detail: `sources/bt.md` §5.
  Where the border is a river, COD-AB's line can stop a few hundred metres short of Natural Earth's,
  so "outside the units" is not "across the border": Namibia had 628 such hexes, half of them across
  the Kavango and Zambezi (Calai, Sesheke; dropped) and half inside Natural Earth's Namibia or the sea
  (snapped). Decide each hex by whether its centroid is inside a neighbour's Natural Earth polygon.
  Caught by: `sources/na_geo.py::border_and_coast` (`SNAP_MAX_PEOPLE`). Example: `na`.
- **A GeoNames point named for a camp can be a village of the same name.** GeoNames' `Mbera`
  (population 58,985) lies in El Megve commune, which the census counts at 15,232, while the Mbera
  camps are 33.4% of Bassiknou moughataa (about 41,200); only Vassala commune (79,508) can hold them.
  Before testing Kontur against a camp or town point, check that the commune the point falls in can
  hold the census's figure. Caught by: `sources/mr_grid.py::camp_check` (the witness moved to
  Vassala). Example: `mr`. Detail: `sources/mr.md` §5.
- **Kontur's false blocks at the cap can be rural, and capping a real city to its ring erases it.**
  Afghanistan's raw grid had 18 blocks at 46,200/km2. Sixteen were one to five hexes in rings of 2 to
  2,106 people per km2 (Daykundi's three held 23.6% of the province, which NSIA counts as having no
  urban population); two were Kabul and Herat, where the ring's median would leave 158,469 and
  29,044 people against NSIA's urban 5,361,333 and 760,907. Cap the small ones; leave a city only
  where capping would take it below the office's own urban count, and assert that the calibrated
  block holds no more than that count. Caught by: `sources/af_grid.py::main` (`BLOCKS`, the urban
  check). Example: `af`. Detail: `sources/af.md` §5.
- **Kontur can swap two units' people outright.** Kontur KM puts 343,284 people on Mohéli (census
  2017: 51,567; Fomboni hexes of 37,211, at the cap) and 67,617 on Anjouan (327,382; nothing above
  1,529 a hex, Mutsamudu included): each island holds about the other's count, while the footprint
  inside each is right (after scaling each island to its census, préfectures read 0.79-1.16 of
  their census shares). Run the per-unit Kontur/census band even on a three-unit country, test a
  failing pair against each other's counts, and where it is a swap, calibrate per unit before
  `kontur_cap.apply` sees the layer. Caught by: `sources/km_geo.py::main` (`EXPECT_SWAPPED`,
  `SWAP_TOL`), `::prefecture_witness`. Example: `km`. Detail: `sources/km.md` §4.
- **Kontur can put a fifth of a country in the wrong county, and a province-level check cannot see it.**
  Iran's grid drew Sarvestan (census 38,114) at 1.78 million and Shiraz county at two-thirds of its census,
  18.3% of Iran in the wrong county, with 78 blocks at the cap; capping blocks leaves the false weight in
  their ramps. Where a census county table exists on the boundary file's pcodes (COD-PS ADM2 is often the
  census itself: assert its sums against the counted units), compare Kontur per county and, if it is off,
  calibrate each hex to its county total. Caught by: `sources/ir_geo.py::read_counties`, `::main` (step 4);
  no shared helper. Example: `ir`. Detail: `sources/ir.md` §5.
- **Calibrating Kontur to a unit total builds a false city where Kontur has lost the unit.** DR
  Congo's grid holds 9,776 people in Lubefu territoire against COD-PS's 665,858; scaled to the total,
  one faint spot became a 214,456/km2 hex. Five Sankuru territoires read 12x to 59x the national
  factor and the next is 4.9x. Above a bar set in that gap (10x), keep only Kontur's footprint and
  spread the unit's total evenly over its hexes; pin the set, and print the calibrated densest hex,
  which a scaled city edge can still push high (Katanda 112,563/km2, beside Mbuji-Mayi). Caught by:
  `sources/cd_geo.py::main` (`HOLE_FACTOR`, `EXPECT_HOLES`). Example: `cd`. Detail: `sources/cd.md` §7.
- **Kontur can be wrong across a whole Gulf city, and then only its footprint is worth keeping.**
  Kuwait's grid holds 2,785 people in Sabah Al-Salem (census 88,904) and 257,607 in Wafra Farms
  (11,961); per unit its share runs p10 0.12 to p90 15.6 of the census, and half its people are in
  the wrong area. Its counts inside a unit would pile a suburb's dots on a few edge hexes. Where the
  census units are small (Kuwait's median 4.4 km2), keep the hexes Kontur puts at 50 people/km2 or
  more and weight them by area: uniform in a residential area, the settled 6% of a desert. Set the
  per-unit bar before reading and print the failure. Caught by: `sources/kw_grid.py::main` (the bar,
  printed and expected to fail; `FOOTPRINT_MIN`). Example: `kw`. Detail: `sources/kw.md` §4.
- **Kontur can scale a whole province wrong, evenly, and fail the province rank witness on a right
  join.** Kontur CU holds 0.17 of Granma's share of ONEI's count and about 0.4 of Santiago de
  Cuba's and Guantánamo's, so the 16-province Spearman was +0.52 and failed its shuffle bar. Per
  municipality the deficit is flat inside each province (Granma 0.16-0.19), and the municipal rank
  witness passed (+0.80 over 168). Run the witness one level down before blaming the join, and
  where the error is flat inside a unit, a per-unit scale fixes it; calibrate one level down only
  where that level's polygons are right (COD-AB Cuba's Havana municipalities are about 2 km east).
  Caught by: `sources/cu_grid.py::main` (both witnesses printed, `MUNI_HOLES`). Example: `cu`.
  Detail: `sources/cu.md` §5.
- **Where no census exists, every grid's split between units is a guess, and Kontur's cap blocks
  pile up on secondary towns.** Eritrea's Kontur read Maekel at 0.45 and Semenawi Keih Bahri at 2.68
  of the survey's zoba shares, with cap blocks on Massawa (268,920), Ghinda and Karora; Meta's 2020
  high-resolution grid had the same zoba error (2.26) but sensible towns inside each zoba. Take the
  unit totals from the survey's own weighted household population (or its frame), and use a grid
  only inside units; try Meta's `<iso3>_general_2020.csv` on HDX before capping Kontur. Meta's CSV
  runs into the neighbours, so drop points inside a Natural Earth neighbour before snapping. Caught
  by: `sources/er_grid.py::main` (per-unit print, the neighbour drop, `TOWN_LOW`). Example: `er`.
  Detail: `sources/er.md` §5. GeoNames can also file a neighbour's town under the country (ER's
  `Himora` is Humera, Ethiopia): check a failing town's coordinates before calling it a hole.
- **Kontur can spread a small capital into its villages, and a district band cannot see it.** Faroese
  Kontur put Tórshavn at 7,289 against the register's 13,999 and Hvítanes at 753 against 106; against
  the register of its own month N-streymoy read 1.158 and Sandoy 1.301, while moving a whole
  municipality's villages across the district line only took N-streymoy to 0.853, inside the band.
  Where the office publishes population by village, give each hex its nearest village in its unit and
  share that village's count in the census month over its hexes. Caught by: `sources/fo_geo.py::main`
  (witness 3, and the Kontur-alone ratios printed). Example: `fo`. Detail: `sources/fo.md` §5.
- **Kontur can put a country's countryside on its summer houses.** Iceland's grid read
  Skorradalshreppur at 9.6x its registered population and Grímsnes- og Grafningshreppur at 6.8x,
  33,165 extra people outside the capital area, which held 0.89 of its census share. A unit-level
  band catches the symptom; the per-municipality table shows the cause. Where the office publishes
  population by municipality, scale each municipality's hexes to it, and merge a boundary file's
  older municipalities forward to the table's map by code. Caught by: `sources/is_geo.py::build_grid`
  (prints the per-municipality ratios, calibrates, asserts the register is reproduced). Example:
  `is`. Detail: `sources/is.md` §5.
- **A municipality polygon can join two islands across a narrow sound.** GADM 4.1 draws Faroese Sunda
  as one part over the Sundini, holding villages from both sides, so splitting it by part fails. Cut
  land out of OSM's sea polygons (`water.WATER`) around the unit and label the pieces by village
  points. Caught by: `sources/fo_geo.py::main` (one land piece per side, asserted; every village in its
  district). Example: `fo`. Detail: `sources/fo.md` §5.
- **A Kontur extract can stop at a claimed border, and the grid holds people the census leaves out.**
  Kontur's `PS` extract has no hexes in the part of Jerusalem Israel annexed (the `IL` extract holds
  350,549 people there), and both grids model the Israeli settlements, which the Palestinian census
  does not count (Ramallah & Al-Bireh read 1.85x the census before correction). Read every extract
  that touches a disputed line and de-duplicate on `h3`; take a population the census excludes out of
  the weights with the other side's own counts, and size a disc for any unit that is only a
  placeholder polygon (CBS draws 119 small localities at 0.008 km2). Witness it with a third point
  layer: the weight near the census's own communities must rise. Caught by: `sources/ps_geo.py::main`
  (the per-extract table, witness 4), `::settlement_units`. Example: `ps`. Detail: `sources/ps.md` §6.
- **A Kontur extract can miss a whole town, and hold people the census count does not.** Kontur's
  `DZ` extract has 448 people within 5 km of Béchar, a wilaya seat GeoNames puts at 165,241, so the
  wilaya read 0.31 of its census share and its dots went to the small oases. In Tindouf it holds
  109,299 against 49,149 counted, the excess 25-50 km from the town where the refugee camps are.
  Check every unit seat against a gazetteer; fill a hole's shortfall with a small disc on the town,
  and where the excess is outside the town keep the hexes near it and assert they match the census.
  Caught by: `sources/dz_grid.py::seat_check` (`KONTUR_HOLES`), `::fill_holes`, `::town_only`;
  no shared helper. Example: `dz`. Detail: `sources/dz.md` §6.
- **A GeoNames seat check can pass a town Kontur has lost; the office's commune count cannot.**
  Morocco's Tan-Tan read 0.11 of GeoNames within 5 km, just over the 0.10 bar, while HCP counts
  76,134 in the commune and Kontur held 9,094; Smara and Assa were not GeoNames seats at all. Where
  a unit reads under half its census share, compare its main municipality with the office's own
  commune count and put a disc on the town for the shortfall. Caught by: `sources/ma_grid.py::main`
  (`LOW_RATIO`, `TOWNS`, `FILLED`); no shared helper. Example: `ma`. Detail: `sources/ma.md` §6.
- **A boundary file that stops at a disputed line can drop a whole province that straddles it.**
  COD-AB Morocco ends at 27°40'N and leaves out all six provinces with land south of it, including
  Tarfaya, whose town and Akhfennir lie north of the line and in neither the Morocco nor the Western
  Sahara file. A unit-count and totals check passes, because the people are in the census table and
  their hexes simply join nothing. Break the unjoined hexes down by where they are, and give each
  group a rule. Caught by: `sources/ma_grid.py::main` (the Tarfaya strip rule, asserted to reach
  GeoNames' Tarfaya and Akhfennir). Example: `ma`. Detail: `sources/ma.md` §5.
- **Kontur models built-up area, so ruins and emptied towns hold people.** Kontur AZ (2023) puts 2.7
  times Aghdam's 2019 census figure inside Aghdam, most of it on the ruins east of the 1994 line, and
  79,000 people in Jabrayil, where the census found 420. Where a unit was partly depopulated by a war
  or an exodus, mask the depopulated part before weighting, with a boundary of the census's own year
  (Natural Earth's old releases keep superseded disputed-area polygons: 4.1.0 has the 1994-2020
  Nagorno-Karabakh). Caught by: `sources/az_grid.py::main` (the per-unit Kontur/census print, and the
  mask's area asserted). Example: `az`. Detail: `sources/az.md` §5. Same failure in peace: Kontur SJ
  puts about 1,900 of Svalbard's 3,230 at Sveagruva, a mine closed in 2017, and 405 at Longyearbyen
  (SSB 2,512); calibrate to the office's settlement counts and keep the ghost hexes at weight 0.
  Caught by: `sources/no_geo.py::svalbard` (prints Kontur against SSB per settlement).
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
- **A survey's church cells by unit can be checked against OSM's churches, where churches are named for their
  body.** Rank each church's share of a unit's places of worship against its share of the unit's people; order
  only, since parish size differs between churches. Churches with near-identical names (Kerala's four
  "Malankara" bodies) can pass as a group and fail one by one, which means the respondents or coders swapped
  them and none of them can be drawn alone. Caught by: `sources/in_osm_churches.py::witness`; no shared helper.
  Example: `in` (Kerala). Detail: `sources/in.md` §14.

## Build

- **A country missing from `country_shapes.py` hands its legend to a neighbour.** Auto falls back to the dot tally. The
  script stops on a registered country Natural Earth lacks: add `ISO` or `FROM_UNITS` (a territory inside its sovereign)
  before registering. Caught by: `country_shapes.py::main`. Example: `bq`. Detail: spec §12 "Finishing".
- **A territory inside its sovereign's data can draw with no outline.** Australia's Norfolk,
  Christmas and Cocos SA2s were in `au.csv` with dots, but `country_shapes.py` took one Natural Earth
  feature per code, so `Indian Ocean Ter.` (`IOA`, `ISO_A2_EH` AU) was skipped and Norfolk Island
  (`NFK`, `ISO_A2` NF) matched nothing; Åland (`ALD`) was the same inside `fi`. A territory its
  country's own source counts joins that outline through `country_shapes.py::ALSO`; one with a
  source of its own is its own entry (`hk`, `bq`). Add only features the source's geography reaches:
  no ABS SA1 is near the Coral Sea or Ashmore features, so they stay out. Caught by:
  `country_shapes.py::main` prints each second feature for a code that it leaves out (Baykonur
  Cosmodrome holds 91 of `kz`'s dots and is not in `ALSO` yet); a dots-outside-outline count across
  countries is Not checked yet (the scratch method is in the detail). Example: `au`. Detail:
  `sources.md` §outlines-2026-09-15.
- **A `FROM_UNITS` territory is also inside its sovereign's outline, and `countryAt` names the
  sovereign.** Natural Earth's countries-file Netherlands includes Bonaire, Sint Eustatius and Saba;
  `bq`'s map unit is appended after it, and the viewer returns the first outline holding the point,
  so it names `nl` over all three islands (measured on the geojson with the viewer's test, not in a
  browser). Not fixed: cut each `FROM_UNITS` unit out of its sovereign's outline when both are
  registered. Caught by: Not checked yet (an overlap count between registered outlines belongs in
  `country_shapes.py::main`). Example: `bq`. Detail: `sources.md` §outlines-2026-09-15.
  Both known and left as is (Anita, 2026-09-15): `kz`'s 91 Baikonur dots sit outside its outline, and the viewer picks `nl` over `bq`.
  `tk` (map unit `TKL`, added 2026-09-15) is the same shape inside `nz`'s countries-file feature, so the viewer will pick `nz` over Tokelau; recorded in `sources/tk.md` §5, not decided.
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
  It writes dots and rings through a temp file and `os.replace` (`write_json_atomic`), so a build tail never reads half of one; `rollup.py` and `taxonomy/build_tree.py` do the same (2026-09-15).
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
