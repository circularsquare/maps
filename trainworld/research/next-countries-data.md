# Open data for the next countries, and census totals against GHSL's phantom people (T-2075)

Research 2026-11-03 for Anita's order of 2026-11-02: New Zealand, India, Turkey, Eastern Europe, one Middle East
country, Mexico, Brazil; plus the open tasks' needs: Indonesia's kabupaten (T-2038), Cambodia's districts (T-2000),
the Philippines' barangays (Muslim Mindanao, where GHSL is under the census).

What a run uses today is GHSL alone (archetype D). The fix for phantom people is tier0's `admin_totals` hook
(`pipeline/tier0.py` 548: a callable boundary -> [(name, geometry, population)]; cells inside each geometry are
scaled to its total; England's `lad_totals` is the only user). For each country below: the smallest unit with
open census totals and open boundaries that join on a code, then the gotchas, then anything on jobs or commuting.

## Summary

| Country | Best totals | Boundary that joins | Main gotcha | Help against GHSL |
|---|---|---|---|---|
| Indonesia | 2020 census (Long Form) by kabupaten, 514 units, USCB on HDX, CC BY 4.0 | same USCB file (COD-AB ADM2, 522 polygons), BPS 4-digit code | Long Form totals are mid-2022 (275.8M), not the full count (270.2M); COD gdb reads 0 features with pyogrio | high in Papua (1.3-4.7x), Riau |
| Cambodia | 2019 census by district (about 200) or commune (1,600): citypopulation.de only | COD-AB ADM2/ADM3, pcode `KH` + NIS code | no open machine-readable table; ODC's WFS times out | high in the north-west (3-10x) |
| Philippines | 2020 census by barangay (42,042), USCB on HDX, CC BY; local copy | same USCB file, 10-digit PSGC | COD-AB and COD-PS are on 88 plain provinces (double counts the 33 HUCs) | high in Muslim Mindanao (GHSL 0.1-0.55x) |
| New Zealand | 2023 census by SA1 (32,817) or SA2 (2,395), Stats NZ ArcGIS, CC BY 4.0; local copy | same services, `SA12023_V1_00` | clipped SA2 layer drops 84 water SA2s (24 with people) | low; easy hook (Australia's shape) |
| India | 2011 census by district (640), NADA catalog 6191 | Datameet `2011_Dist` (641), `censuscode`, CC BY 4.0 | 15 years old: use shares within the state, not levels; sub-district polygons (SHRUG) are non-commercial | unknown, check per district |
| Turkey | ADNKS register 31 Dec 2022 by ilce (973), COD-PS on HDX, CC BY-IGO | COD-AB TUR ADM2, `ADM2_PCODE` | ADNKS leaves out Syrians under temporary protection (separate COD-PS table); TUIK's own portal is a script-proof SPA | low to medium (refugees, 2023 earthquake zone) |
| Mexico | 2020 census by municipio (2,469) or locality (ITER), INEGI open terms | INEGI Marco Geoestadistico 2020, `CVEGEO` | soft 404s (HTTP 200 html) on wrong INEGI URLs; zero-pad CVEGEO | low to medium; easy hook (Brazil's shape) |
| Brazil | 2022 census by tract, already used (`brazil.plan_hooks`) | IBGE tract GeoPackages | none new | done |
| Israel (Middle East pick) | 2022 census by statistical area (about 3,200), CBS | CBS ArcGIS `Statistical__Areas_2022`, `SEMEL_YISHUV`+`STAT_2022` | which territory to draw is Anita's call | low; data the best in the region |
| Eastern Europe | built except Ukraine and Belarus (held) | | | low (GHSL is good in Europe) |
| Ukraine | nothing open and current: last census 2001; UN COD-PS 2026 is restricted | COD-AB UKR (raion, hromada) | war displacement makes every figure stale, GHSL included | held anyway |

Nothing better than GHSL: Ukraine (nothing current), Saudi Arabia (region level only), Belarus (district totals only
in Belstat PDF bulletins); for Cambodia only a scrape of citypopulation.de.

## The general fix (the shape of the problem)

Every source above that joins on a code is one of three shapes: a COD-PS or USCB table with pcodes beside a COD-AB
or USCB polygon file (Indonesia, Philippines, Turkey, Iran, Ukraine), a statistics office's own geography service
(New Zealand, Israel, Mexico, Brazil), or a census table plus a third-party polygon file with the census code (India).
So one generic `admin_totals` reader, configured per country by (polygon file, layer, code field, totals file, code
field, population field), covers most of them; filed as T-2084. Three rules for it:

1. **Scale by a factor, not to a total.** Today the hook sets the cells inside a unit to the unit's total. A census
   unit cut by the pack's outline (census polygons are not the plan's geoBoundaries units) then gets its whole total
   inside the outline. The hook should give each unit the factor census / GHSL over the whole unit and apply that to
   the cells inside.
2. **Shares within a parent for an old count.** India 2011, Iran 2016, Cambodia 2019: hold GHSL's total for the
   province or state and redistribute it by the census shares, so 5-15 years of growth do not count as error.
3. **Cap the change, report it.** Print units scaled by more than 2x either way (the preflight's T-2002 FLAG in
   reverse); those are the phantoms the hook is for, and anything else that large is more likely a bad join.

**HDX downloads to scripts** (2026-11-03): the CKAN API answers, but a resource download sent with a browser-like
User-Agent got an AWS WAF challenge (HTTP 202, `x-amzn-waf-action: challenge`, empty body); curl's own default agent
got the usual 302 to the S3 file. A fetch that comes back empty from `data.humdata.org` is this, not a dead link.

## Indonesia (T-2038)

- **Totals and boundaries in one file**: U.S. Census Bureau, "Indonesia Subnational Population and Housing Data
  Tables" on HDX (CC BY 4.0): `Indonesia.gdb.zip` (40 MB; ADM0-2, the ADM2 layer is COD-AB's 522 polygons with seven
  water and conservation polygons merged into neighbours, Danau Toba kept) and `Indonesia_uscb_202405.xlsx` (556 KB,
  opened). Sheet `Age-Sex`: one row per kabupaten/kota, `NSO_CODE` the BPS 4-digit kode wilayah, `BTOTL` the total.
  Checked: 514 kabupaten/kota rows with unique codes summing exactly to the 34 provinces and the nation, plus a
  Danau Toba row with no total (a lake polygon kept in the file). Papua, for T-2038: Yalimo `9432` 105,139, Mamberamo
  Tengah `9431` 51,719, Puncak Jaya `9411` 231,499, Puncak `9433` 116,279, Paniai `9410` 227,254, Tolikara `9418`
  244,345, Mimika `9412` 321,657, Merauke `9401` 232,357.
  https://data.humdata.org/dataset/indonesia-subnational-population-and-housing-data-tables
- **Join**: `NSO_CODE` = COD-AB `adm2_pcode` without its `ID` prefix (religiondots matched all 2010 codes to COD's
  polygons this way, `religiondots/sources/id.md` 6). COD-AB IDN: https://data.humdata.org/dataset/cod-ab-idn (CC BY-IGO;
  ADM3 7,069 kecamatan and ADM4 81,912 desa also there).
- **Gotchas**:
  - The table is the **2020 Long Form** (Table 3.1 of "The Result of Long Form Population Census 2020", reference
    mid-2022), national 275,773,774, not the September 2020 full count 270.2M. For scaling GHSL that is if anything
    better (nearer GHSL's epoch); say so in the pack's source line.
  - COD-AB's geodatabase reads **zero features with pyogrio** and through `zip://` with either engine (religiondots);
    take the shapefile or GeoJSON zip, extracted.
  - Papua was split into four provinces in 2022 (Papua Selatan, Tengah, Pegunungan, Barat Daya) with new BPS codes.
    The 2020 tables and COD-AB 2020 both use the old codes, so they join each other; any BPS table from 2023 on does not.
  - `bps.go.id` is Cloudflare-403 to scripts; `sensus.bps.go.id` was open for SP2010 (religiondots) but its SP2020
    table paths (`/topik/tabular/sp2020/10/...`) answered 404 today.
  - Alternative totals: COD-PS IDN ADM2 2020 (projections, licence "Other") and citypopulation.de (the full count).
- **Kecamatan**: SP2020 by kecamatan is printed in each kabupaten's "Dalam Angka"; no bulk open table found. Kabupaten
  is enough for Papua (zones of one to three kabupaten).
- **Jobs**: nothing by small area found.
- **Help**: high. GHSL is 1.3-2.2x the census in Papua's zones, 4.7x in Yalimo, 1.2-1.7x in Riau (notes/T-1738.md 12);
  scaling fixes yai and lets puc build.

## Cambodia (T-2000)

- **Totals**: the 2019 census by district or commune is not in any open machine-readable file found. What exists:
  - provinces with polygons: Open Development Cambodia `final_census2019_en` (CC BY-SA; sqlite and WMS;
    https://data.opendevelopmentcambodia.net/dataset/population-census-2019-final-population-totals);
  - districts and communes: citypopulation.de, figures credited to NIS, e.g.
    https://www.citypopulation.de/en/cambodia/admin/2205__trapeang_prasat/ (pages keyed by NIS code).
- **Join**: citypopulation's codes are NIS's (Oddar Meanchey's districts 2201-2205), and COD-AB KHM's pcodes are `KH` +
  the same digits (ADM2 197 districts, ADM3 1,633 communes; https://data.humdata.org/dataset/cod-ab-khm, CC BY-IGO).
- **Gotchas**:
  - citypopulation's district and commune figures are **household population** (no institutional, boat or transient
    people), and the whole census leaves out about 1.2M migrants abroad.
  - ODC's GeoServer WFS answered 504 to a 2-feature request (the T-1738 agent saw timeouts too); its 2008 commune
    layer ("Population density", census 2008 by commune) is only reachable that way.
  - NIS microdata (`microdata.nis.gov.kh`, catalog 39) is a 10% sample (1,555,221 persons); the province variable shows
    weighted counts, the district one (`c2_srok`) shows none, and the files sit behind "Get Microdata". IPUMS has
    districts but the account is dead.
- **Practical route for T-2000**: the north-west needs about ten districts (Oddar Meanchey, Banteay Meanchey, Pailin,
  Battambang's border districts): typed into a small `cambodia` module from citypopulation, cited to NIS 2019, as
  `india.py` types Mumbai's sub-districts. The country-wide hook would need a scrape of about 200 district pages.
- **Jobs**: none.
- **Help**: high where it matters. GHSL has 3-10x the census in the Thai border districts (Trapeang Prasat, Anlong
  Veaeng, Sala Krau, Phnum Proek); Kontur has Pailin at 4.6x too (religiondots `kh_geo.md`).

## Philippines (Muslim Mindanao under-count)

- **Totals and boundaries**: USCB "Philippines Subnational Population and Housing Data Tables" on HDX (CC BY 4.0):
  `Philippines.gdb.zip` (304 MB) with `PH_GEOG_ADM4_2020_uscb_202402`, 42,042 barangays, 10-digit PSGC in `NSO_CODE`,
  and `PH_AGE_SEX_2020census_uscb_202402` with `BTOTL` at every level.
  https://data.humdata.org/dataset/philippines-subnational-population-and-housing-data-tables
- **Local copy**: `religiondots/data/geo/ph/ph_barangays.gpkg` (layer `barangays`: `bgy` PSGC, `unit`, `name`, `pop`;
  42,042 rows, 109,033,245 people, the 2020 census less about 2,000 abroad in embassies). Built and checked by
  `religiondots/sources/ph_geo.py` (ph_geo.md).
- **Gotchas**:
  - PSA tabulates on 116 units (provinces without their highly urbanised cities, the HUCs apart); COD-AB PHL's ADM2 is
    the 88 plain provinces, so a COD-PS join double counts the 33 HUCs. COD-AB's ADM4 (11,920) is a Mindanao subset.
    Use the USCB file.
  - The BARMM Special Geographic Area's 63 barangays are filed under Cotabato in the USCB file (its `USCBCMNT` says so).
  - geoBoundaries PHL has no ADM4.
- **Municipality level**: sum the barangays by their first 7 PSGC digits (municipality), or use the USCB ADM3 layer.
- **Jobs**: none by small area.
- **Help**: high for one region. GHSL is 0.10x the census in Basilan, 0.24x Tawi-Tawi, 0.29x Sulu, 0.55x Lanao del Sur
  (notes/T-1738.md 13); scaling up fixes pti's 0.47M against 2.0M. Elsewhere GHSL is within 1.0-1.5x.

## New Zealand

- **Totals**: Stats NZ 2023 census usually resident population (`VAR_1_3`) by SA1 (32,817 units, about 150 people) and
  SA2 (2,395), in Stats NZ's ArcGIS Online services, no key:
  `https://services2.arcgis.com/vKb0s8tBIA3bdocZ/arcgis/rest/services/2023_Census_totals_by_topic_for_individuals_by_SA1/FeatureServer/0/query`
  (and `..._by_SA2`), paged 500-1000 records, `outSR=4326&f=geojson`. Licence CC BY 4.0 (the portal item's
  `licenseInfo`). Recipe: `religiondots/sources/nz_geo.md` 2.
- **Boundaries**: the same layers carry polygons (clipped to the coast); the full-extent product is
  `Statistical_Area_2_2023`. Key `SA12023_V1_00` / `SA22023_V1_00`.
- **Local copies**: `religiondots/data/geo/nz/sa1_2023_clipped.geojson` (140 MB) and `sa2_2023_clipped.geojson`.
- **Gotchas**: the clipped SA2 layer drops the 84 all-water SA2s (24 hold people, 309 in all); 16 SA2s are not
  digitised (24 people). The census night was 7 March 2023: no lockdown.
- **Hook**: a population hook by SA1 passes playbook 1.2 (2023 census, small areas with polygons, open endpoint, the
  shape of `australia.py`'s SA1 hook). Filed T-2077.
- **Jobs**: `2024_Business_Demography_employee_count_by_SA2` in the same org (employee count by SA2 of workplace,
  fields `ec`/`gc` by year; the service named `2025_employees_by_statistical_area_2` actually holds TALB polygons).
  Commute flows: Stats NZ's 2023 census "main means of travel to work by SA2" origin-destination tables are on
  datafinder.stats.govt.nz (export needs a free account; not opened). Urban/rural: `Urban_Rural_Areas_2023` layer.
- **Help against GHSL**: low (no known phantom areas); the gain is census-quality people and jobs for a small country.

## India

- **Is 2011 still the latest open count?** Yes. The 2027 census's houselisting ran April-September 2026; population
  enumeration is February 2027 (reference date 1 March 2027; snow-bound areas 1 October 2026). No population results
  are out, and none is announced; provisional totals usually follow a few months after enumeration, so late 2027 at
  the earliest. ([Wikipedia](https://en.wikipedia.org/wiki/2027_census_of_India),
  [Business Today](https://www.businesstoday.in/india/story/initial-phase-of-census-2027-to-begin-on-april-1-houselisting-to-be-done-within-30-day-period-509920-2026-01-08))
- **Totals**: Census 2011 Primary Census Abstract, India and States, state and district level:
  https://censusindia.gov.in/nada/index.php/catalog/6191 (NADA; the host's certificate chain is broken, `curl -k`;
  each file's download id comes from its catalogue page). Sub-district and town rows are in the per-state PCA files
  (catalog 6708-6711 for Maharashtra, as `india.py` uses).
- **Boundaries that join**: Datameet `Districts/Census_2011/2011_Dist.shp`, 641 polygons, fields `DISTRICT`, `ST_NM`,
  `ST_CEN_CD`, `DT_CEN_CD`, `censuscode` (the 2011 district code, e.g. Adilabad 532), CC BY 4.0
  (https://github.com/datameet/maps, opened). Sub-district polygons with census codes: SHRUG PC11 (5,969, exact join,
  local copy `religiondots/data/geo/in/in_subdistricts.gpkg`), but **CC BY-NC-SA**: checks only, as the register says.
- **Gotchas**:
  - 15 years old and India grew about 17% since, unevenly: scale by each district's share of its state, holding
    GHSL's state total (the general rule 2 above), not by the 2011 level.
  - About 780 districts now against 640 in 2011; the hook uses the 2011 polygons, so later splits do not matter.
  - Village-level work: 3,892 six-digit codes name both a village and a town; join on (unit, code). Not needed at
    district level.
  - `99999` "Area not under any sub-district" units (17.4M people, Kolkata's fringe) have no SHRUG polygon.
- **Jobs**: 6th Economic Census 2013 by village and ward (already N, T-388); Census 2011 B-28 distance to work by
  district (used for Mumbai's curve); census towns as an urban flag.
- **Help**: unknown. Mumbai's sub-districts were within 2% in sum but 0.5-2.6x unit by unit against OSM's lines
  (`india.py` GROUPS). Compare district shares in the dry run before deciding; filed T-2078.

## Turkey

- **Totals**: COD-PS TUR, ADNKS (address-based register) at 31 December 2022 by ilce, 973 districts, from TUIK; tables
  `tur_admpop_adm2_2022.csv` and a separate ADM2 table of Syrians under temporary protection. CC BY-IGO.
  https://data.humdata.org/dataset/cod-ps-tur (religiondots checked the province totals against NVI's copy: 85,279,553.)
- **Boundaries**: COD-AB TUR ADM2 (973 ilce, from the General Command of Mapping, 2014 lines; districts unchanged since
  the 2012-2014 metropolitan reform), `ADM2_PCODE`. https://data.humdata.org/dataset/cod-ab-tur
- **Gotchas**:
  - ADNKS leaves out Syrians under temporary protection (about 3M in 2022, fewer since); add COD-PS's own table.
  - `data.tuik.gov.tr` and `veriportali.tuik.gov.tr` are single-page apps with no reachable table API;
    `nip.tuik.gov.tr` goes only to province. Newer ADNKS years (2023-2025) by ilce: citypopulation.de.
  - Mahalle and koy totals exist in ADNKS but no open polygons were found.
- **Jobs**: nothing by small area (SGK insured workers by province only).
- **Help**: low to medium. GHSL has no known phantom areas here, but the 2023 earthquake zone (Hatay, Kahramanmaras,
  Adiyaman) lost people after GHSL's inputs; a 2022 register predates that too, so check against 2025 ADNKS figures
  for those provinces. Filed T-2079.

## Mexico

- **Totals**: INEGI Censo de Poblacion y Vivienda 2020.
  - Municipio and locality: ITER, `https://www.inegi.org.mx/contenidos/programas/ccpv/2020/datosabiertos/iter/iter_00_cpv2020_csv.zip`
    (36.6 MB; 193,094 locality rows; municipio rows are `LOC == '0000'`). Opened by religiondots (`sources/mx.md`).
  - Urban AGEB and block: "Principales resultados por AGEB y manzana urbana", one file per state from the census's
    "Datos abiertos" page: `https://www.inegi.org.mx/contenidos/programas/ccpv/2020/datosabiertos/ageb_manzana/ageb_mza_urbana_{01..32}_cpv2020_csv.zip`
    (confirmed 2026-11-05, T-2080).
  - Licence: INEGI Terminos de Libre Uso (attribution), https://www.inegi.org.mx/inegi/terminos.html
- **Boundaries**: Marco Geoestadistico 2020, national subset `mg_2020_integrado.zip` (257,527,530 bytes),
  `https://www.inegi.org.mx/contenidos/productos/prod_serv/contenidos/espanol/bvinegi/productos/geografia/marcogeo/889463807469/mg_2020_integrado.zip`:
  `00mun` 2,469 municipios, `00a` 81,451 AGEB (urban 13-char, rural 9-char CVEGEO), `00l` locality polygons, `00lpr`
  295,779 rural locality points. Join on `CVEGEO` (municipios 2,469 = 2,469, clean). Recipe: `religiondots/sources/mx_geo.md`.
- **Gotchas**: the widely cited `889463807469_s.zip` URL is a 2,263-byte HTML page with HTTP 200; INEGI answers any
  wrong path that way, so reject downloads under 1 MB. CVEGEO must stay a zero-padded string (entidades 01-09).
  Rural AGEBs are large (median 93 km2): for countryside, ITER's localities as points are finer.
- **Hook**: a population hook like `brazil.plan_population` (urban AGEB polygons plus rural localities), or
  `admin_totals` by municipio, filed T-2080. The 2020 enumeration (2-27 March 2020) was before the lockdown.
- **Jobs**: DENUE, INEGI's directory of every economic unit with coordinates and an employee band (per_ocu), a general
  jobs-at-work source for the whole country; Censos Economicos 2019 by AGEB; the census's extended questionnaire
  microdata has the municipio of work (commute flows by municipio). DENUE's bulk files are per state,
  `https://www.inegi.org.mx/contenidos/masiva/denue/denue_{01..32}_csv.zip`, the State of Mexico as `denue_15_1` and
  `denue_15_2` (no national file); used by `pipeline/mexico.py` (T-2081, notes/T-2080.md).
- **Help**: low to medium (highland and dispersed rural Oaxaca and Chiapas are where GHSL could be off); mostly
  census-quality people for little work.

## Brazil

- **Already done**: `pipeline/brazil.py`'s `plan_hooks` gives every planned Brazilian row Census 2022 people by tract
  (IBGE `{UF}_setores_CD2022.gpkg`, `v0001`), spread by GHSL inside each tract. Nothing more is needed for people.
- **Gotchas**: tract files are per state and large; `v0001` nulls are suppressed tracts (counted 0).
- **Jobs and commuting** (not opened): RAIS establishment employment by municipality (Ministry of Labour's PDET
  microdata), Census 2010 municipal commute flows; whether Census 2022 commuting is out was not checked.
- **Help**: the census already fixes favelas and the Amazon interior.

## Eastern Europe

Built (data-register): Poland, Romania, Czechia, Hungary, Bulgaria, Serbia, Albania, Slovakia, Slovenia, Croatia,
Bosnia, Montenegro, North Macedonia, Kosovo, Moldova, Estonia, Latvia, Lithuania. Held by Anita (2026-10-19): Ukraine,
Belarus, most of Russia.

- **For the built EU members**: Eurostat's Census 2021 1 km grid (GEOSTAT, released June 2024, version of
  22 January 2025; CSV, GPKG, raster; EPSG:3035) covers every EU country with the same method:
  https://ec.europa.eu/eurostat/en/web/gisco/geodata/population-distribution/geostat . The population attributes
  carry Eurostat's own download terms (accepted on download). GHSL is good in Europe, so the gain is small; T-771
  already holds Spain's. Filed as one Someday task for all of them, T-2083.
- **Ukraine**: the last census is 2001. The USCB HDX file has 2001 and 2017 estimates by oblast and old raion; COD-PS
  UKR 2026 (UNFPA, raion level) says "should be treated as restricted" (not usable). COD-AB UKR (raion 139, hromada
  1,769, CC BY-IGO) is open. With millions displaced, nothing open is current, GHSL included. Nothing to do until Anita
  lifts the hold.
- **Belarus**: 2019 census by raion appears in Belstat's PDF bulletins (no xlsx found); COD-AB BLR ADM2 (119, CC BY-IGO,
  pcodes, names to match). GHSL is likely fine there. Held.

## Middle East: the pick

Recommendation: **Israel**, with one call for Anita first.

- **Data**: the best in the region. CBS 2022 census by statistical area (about 3,200 units of about 3,000 people):
  `selected-data-by-localities-and-statistical-areas-2022-census.csv` on data.gov.il (plain `data.gov.il` host; the
  `aws-e.` host wants a login), and CBS's own polygons on ArcGIS Online, `Statistical__Areas_2022` (3,857 polygons keyed
  `SEMEL_YISHUV` + `STAT_2022`, EPSG:2039, org `xMRYm7cNgdR5RN6F`). Exact code join; units listed as `1022+1023+1024`
  in `StatAreaCmb` are polygons to dissolve. Recipe: `religiondots/sources/il.md` 8 and `il.py`.
- **Rail interest**: high. A dense national network on a small country (Tel Aviv to Jerusalem's high-speed line, the
  coastal line, Haifa), Tel Aviv's light rail; about 10M people in a small area, so a quick build (a few metros and
  10-15 zones).
- **The call**: which territory to draw (the West Bank, East Jerusalem, Gaza, the Golan). religiondots' precedent
  (Anita, 2026-09-07): the Green Line, East Jerusalem out, the Golan kept. A game is a different product; who a map
  depicts is hers.
- **If she would rather not**: **Saudi Arabia**, GHSL alone (archetype D), most of the country dead space, the cities
  clear in GHSL (no phantom risk in desert). Open data is thin: COD-PS by region (13) from GASTAT's 2022 census,
  COD-AB to region only. Rail: the Haramain high-speed line, Riyadh Metro, the north and east lines.
- **Others looked at**: Iran (COD-PS 2016 census by bakhsh, 429 units, joins COD-AB ADM2 by pcode, CC BY-IGO: good
  data and a big network, but 90M people is no "easy" country); Egypt (COD-PS only by governorate, 2016); Jordan and the
  UAE (no COD-PS; small).

## Sources searched and not used

- HDX COD-PS for India, New Zealand, Israel, Jordan, Belarus, UAE: none. COD-PS KHM, MEX, BRA, SAU: province level only
  (projections).
- citypopulation.de is the fallback everywhere for checks; its figures are the offices', its pages not a bulk source.
