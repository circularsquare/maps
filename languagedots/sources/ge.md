# Georgia (`ge`): 2024 census native language, 64 municipalities and cities

Drawn 2026-10-04 by session `d9e44929-ge`. No asks filed.

Files: `sources/ge_census.py` (normaliser), `sources/ge_geo.py` (units and placement),
`taxonomy/ge2024.py`, `taxonomy/tree.d/ge.txt`, `countries/ge.py`. Outputs `data/raw/ge/` (two
2024 workbooks, the 2014 PxWeb cube, COD-AB Georgia), `data/normalized/ge.csv`,
`data/geo/ge/ge_units.gpkg`, `ge_hexes.gpkg`, `data/geo/kontur/kontur_population_GE_20231101.gpkg`
(unpacked from religiondots' .gz by `_grid.py`), `dots_ge.geojson` (3,877 dots), `rings_ge.geojson` (1).

## 1. The table, and why 2024 and not 2014

The queue's lead was the 2024 census with the level unverified, and religiondots drew Georgia from
the 2014 census at 11 regions. Both censuses ask native language with the same eight answers, but:

- **2014** publishes it at **region** only: PxWeb `pc-axis.geostat.ge/PXWeb/api/v1/en/`, table
  `Population Census 2014/Demographic And Social Characteristics/20_Population_by_region,_by_native_languages_and_fluently_speak_Georgian....px`
  (11 regions; the same PxWeb quirks religiondots recorded: `dbid` root, explicit selection,
  json-stat v1 only).
- **2024** (final results, 22 June 2026) publishes it at **self-governed unit**: workbook "5.
  Population by regions, self-governed units, native language and Georgian language knowledge
  level.xlsx", `geostat.ge/media/80624/`, linked from the 2014 census's own category page
  `geostat.ge/en/modules/categories/910/demographic-and-social-characteristics`, which now carries
  a "2024 Population Census of Georgia results" section. The census2024.geostat.ge portal is a React
  app whose results page only links back to these category pages; PxWeb has no 2024 database yet.

So 2024, at 64 units (61,000 people each) instead of 11 (334,000). The 2014 cube is kept as a witness.

The table: one row per region then its units (Tbilisi is region and unit). Columns: Total, then
Georgian, Abkhaz, Ossetian, Azerbaijanian, Russian, Armenian, Other, Not stated; then four blocks
that split the non-Georgian native speakers by knowledge of Georgian (fluent, partial, none, not
stated). Note at the foot: "Does not include occupied territories of Georgia". Universe: everyone
enumerated, which in 2024 includes foreign citizens present in the country (Main Results, p.2).

geostat.ge's TLS chain does not verify from this machine (curl exit 60); the fetch uses
`verify=False` and checks each file's own shape.

## 2. Checks (all asserted in `ge_census.py`)

1. National figures equal the Main Results PDF (`geostat.ge/media/80541/`): total 3,929,581,
   Georgian 3,343,987, Azerbaijani 265,534, Armenian 139,438, Russian 54,460, Ossetian 3,839, Abkhaz 370.
2. The eight answers sum to Total on all 76 rows; every region's units sum to it per answer; the 11
   regions and the 64 units each sum to Georgia per answer, exactly.
3. The four Georgian-knowledge blocks sum to each non-Georgian native language on every row (0
   mismatches): an internal second cut of the same people.
4. **Second table of the same census:** table 4 (nationality by unit, `media/80623/`) has the same
   rows in the same order and the same total on every row. Per unit with 2,000+ members, Armenian
   speakers / Armenians run 0.52 (Batumi) to 1.00, median 0.98; Azerbaijani speakers /
   Azerbaijanis 0.93 (Tbilisi) to 1.00, median 1.00. A shifted column would be far outside.
   Asserted inside 0.5-1.15. Batumi's city Armenians are the outlier: half name another language.
5. 2014 against 2024 by region (printed): every region's shares within a few points, e.g.
   Kvemo Kartli Georgian 52%→49%, Azerbaijani 42%→44%; Samtskhe-Javakheti Armenian 50%→51%;
   Tbilisi Georgian 91%→87% (likely the foreign residents; "other" is 3.9% of Tbilisi).

## 3. Mapping (`taxonomy/ge2024.py`)

Seven answers, seven existing nodes; no new nodes (`tree.d/ge.txt` repeats them and their parents).
- Georgian → `kartvelian.georgian`. **Mingrelian and Svan are inside it**: the form has no answer
  for them, so Samegrelo and Svaneti draw as Georgian. Not split by place: nothing published says
  how many, and a split by region would be a guess.
- Abkhaz → `abkhazadyghe.abkhaz` (370; from `pl`/`fi`); Ossetian → `indoeuropean.iranian.ossetian`
  (3,839); Azerbaijanian → `turkic.azerbaijani`; Russian → `...east.russian`; Armenian →
  `indoeuropean.armenian.armenian`.
- **Other → bare `other`** (73,373, 1.9%). It mixes Georgia's own minority languages (Kurmanji of
  the Yazidis, Chechen of the Pankisi Kists: Akhmeta's "other" is 4,508 of 28,908; Avar in Kvareli,
  1,029; probably also Greek, Ukrainian and Assyrian; these attributions are from where the groups
  are known to live, the table names none of them) with immigrants' languages (24,000 Indians and 12,500 Arabs by table
  4, mostly in Tbilisi, where "other" is 52,197). The indigenous-remainder rule cannot separate
  them, so it is the narrowest node.
- Not stated (48,580, 1.24%) → not drawn; ENTRY `gap`.

## 4. Geography (`ge_geo.py`)

- **Boundaries: COD-AB Georgia** (HDX `cod-ab-geo`, CC BY-IGO, valid_on 2019-10-18), not
  geoBoundaries: geoBoundaries ADM2 has no Batumi, Kutaisi, Poti or Rustavi and draws Tbilisi at
  329 km2 (the city is ~502; religiondots' Tbilisi/ring artefact). COD-AB has all four cities and
  Tbilisi (ADM1 GE11) at 502 km2, and its ADM2 stops at the line of control: Abkhazia (GE12) and
  the "Provisional Administration" (GE48, Tskhinvali) are ADM1 with no ADM2.
- COD-AB's 70 ADM2 are the 2014 map: seven towns were self-governing cities until 2017 (Ozurgeti,
  Telavi, Mtskheta, Ambrolauri, Zugdidi, Akhaltsikhe, Gori). Each `<X> City` is dissolved into `<X>`
  (pinned by pcode, name asserted). 63 + Tbilisi = 64.
- Join by name (`Keda Municipality` / `Keda`, `C. Batumi` / `Batumi`, one spelling pinned:
  Sighnaghi/Sighnagi), asserted a bijection, and every pair must agree on its region (the table's
  region row against COD's adm1_name): 64 of 64.
- Placement: `_grid.hex_layer` on Kontur GE 2023. Kontur/census 0.865 nationally (Kontur is 2023,
  and the census counts foreign residents); per unit normalised p10 0.87, median 1.05, p90 1.21;
  0 of 64 outside a factor of 3; log r 0.986 against a best of 0.376 over 500 shuffles. Low end
  Mtskheta 0.59 and Khelvachauri 0.66 (rings around Tbilisi and Batumi), high end Gardabani 1.52
  (Tbilisi's south-eastern edge): the city/ring pattern religiondots met, at a finer grain. Reported,
  not corrected.
- Hexes outside every unit, classed in `snap_coast`: 297,822 Kontur people in COD's Abkhazia and
  Tskhinvali (dropped); 1,748 across the border in Natural Earth (dropped); 24,949 on Abkhazia's
  coast beyond COD's polygon (over 5 km from any unit, lon 40.00-41.47; dropped, asserted west of
  41.7); **12,613 just offshore or in slivers within 1 km of a unit, snapped** (Batumi 3,690,
  Kobuleti 2,596, Poti 1,611, Ozurgeti 1,400, Khelvachauri 1,034, Marneuli 814). Batumi 0.90 after.
- Median unit 787 km2, about 1,064 hexes: well above the grid floor. No Kontur cap block
  (religiondots had none for GE either; the scatter did not stop).
- Not done: languagedots has no not-drawn hatching, so the hexes are not cut to Natural Earth's
  Abkhazia/South Ossetia as religiondots' are; COD-AB's own line is used. If the viewer gains
  that hatching, check dots along Natural Earth's simplified South Ossetia line.

## 5. Calls someone might reverse

- 2024 at 64 units over 2014 at 11 (same eight answers; finer and newer).
- "Other" as one `other`, minority and immigrant languages together.
- The 12,613 coastal Kontur people snapped back at 1 km.

## 6. Scatter

3,881,001 people drawn, 3,877 dots at 1:1000; 4,001 people (0.10%) under one dot per language
nationally (Abkhaz's 370 among them). Colours are all existing ones: Georgian lavender, Azerbaijani
magenta, Armenian light teal, Russian green, Ossetian olive; they clear each other in Kvemo Kartli,
Javakheti and Tbilisi, so nothing was hand-picked.

## Abkhazia and South Ossetia (added 2026-10-06, session `5d7dac7e-cau`)

Anita, 2026-10-06: draw the hatched breakaway areas. Files: `sources/ge_breakaway.py` (fetch,
tables, units, placement), `taxonomy/ge2015_breakaway.py`, lines appended to `taxonomy/tree.d/ge.txt`,
`countries/ge.py` (counts, `parts`, `drawn_named`, note). Outputs `data/normalized/ge_breakaway.csv`,
`data/geo/ge/ge_breakaway_units.gpkg`, `ge_breakaway_hexes.gpkg`, `ge_plus_hexes.gpkg` (Georgia's
layer plus these; `countries/ge.py` now reads it, so re-run `ge_breakaway.py` after `ge_geo.py`).

**Geography follows religiondots**, which keeps both inside `ge` (blank there; `country_shapes.CLIP`).
They are units `AB-*` (8) and `SO-*` (5) of Georgia's entry, not entries of their own.

### Abkhazia: 2011 census, nationality only (tier `derived`)

- Table: the 2011 census of the Abkhaz authorities (published Sukhum 2012), nationality by district
  and Sukhum city, through Tim Bespyatov's transcription
  `pop-stat.mashke.org/abkhazia-ethnic2011.htm` (it cites ethno-kavkaz.narod.ru). 8 units, 15
  nationalities and "other", 240,705. Checks: every unit's nationalities sum to its total; the 8
  units sum to the total row in all 16 columns.
- No native-language table was published (searched: ru.wikipedia, the statistics office
  cgsra.org, which lists yearbooks 2018-2025 only). A settlement-level version exists
  (`abkhazia-ethnic-comm2011.htm`, downloaded to `data/raw/ge/`) but there are no settlement
  polygons to place it on; districts it is.
- Retention, nationality to language, rest to Russian:

  | nationality | people | language share | source |
  |---|---|---|---|
  | Abkhaz | 122,175 | 97% Abkhaz | 1989 Soviet census, all Abkhaz (Hewitt and Watson, *Encyclopedia of World Cultures*, "Abkhazians") |
  | Armenians | 41,906 | 66.2% Armenian | Russia 2021, Vol. 5 Table 7, Krasnodar Krai (134,609 of 203,251) |
  | Ukrainians | 1,743 | 26.5% Ukrainian | same, Krasnodar Krai |
  | Greeks | 1,381 | 24.0% Greek | same, Krasnodar Krai |
  | Roma | 261 | 79.5% Romani | same, Krasnodar Krai |
  | Turks | 731 | 93.2% Turkish | same, Russian Federation |
  | Ossetians | 605 | 95.1% Ossetian | same, Russian Federation |
  | Tatars | 338 | 83.2% Tatar | same, Russian Federation |
  | Belarusians | 285 | 19.0% Belarusian | same, Russian Federation |
  | Georgians, Mingrelians, Svans | 43,248 / 3,207 / 43 | 100% own | none; see the call below |
  | Russians | 22,064 | 100% Russian | |
  | Estonians | 351 | 100% Estonian | none (the Salme and Sulevo villages) |
  | other | 2,367 | `other` | |

- **Call: Mingrelian vs Georgian.** The census prints Georgians (43,248) and Mingrelians (3,207)
  apart and that split is kept. Most of Gal's Georgians speak Mingrelian at home, but Georgia's
  2024 native-language census draws Samegrelo, across the Enguri, as Georgian (the same people
  answer "Georgian" to a native-language question); drawing Gal as Mingrelian would put a language
  border on the ceasefire line. Reversing it: map Georgians in AB-GAL/AB-TKVARCHELI/AB-OCHAMCHIRA
  to Mingrelian in `ge_breakaway.py` (about 39,000 people).
- **Call: Armenians on the Krasnodar rate.** Abkhazia's Armenians are largely Hamshen, the same
  community as Sochi's; Krasnodar Krai's 2021 figure is the nearest measured one. Likely too low
  for the Gagra and Gulripsh villages. Hamshen (Homshetsi) is drawn as Armenian; no source names it.
- **Call: Georgians at 100%.** A compact community (91% of Gal); Krasnodar's 59% is a scattered
  diaspora. South Ossetia's own 2015 table gives Georgians there 90.7% Georgian (8% Ossetian), the
  nearest measured compact case; not used for Abkhazia, where Ossetian is not the local language.
- Drawn: Abkhaz 118,510, Georgian 43,248, Russian 42,633, Armenian 27,753, Mingrelian 3,207,
  other 2,367, then Turkish, Ossetian, Ukrainian, Estonian, Greek, Tatar, Romani, Belarusian, Svan.
- The figures are from 2011; nothing newer by district exists.

### South Ossetia: 2015 census, native language (tier `measured`)

- `ugosstat.ru/wp-content/uploads/2017/06/Itogi-perepisi-RYUO.pdf` (State Statistics Directorate,
  456 pages, %%EOF checked): tables 4.2.1-4.2.5 (pp. 105-109) give native language by nationality for
  Dzau, Znaur, Leningor, Tskhinval districts and Tskhinval city. Per unit, the language totals are
  the "stated a nationality" row plus the "nationality not stated" row.
- Checks: every row's languages sum to its "stated a native language" figure; the 5 units sum to
  table 4.2 (p. 104, the republic) in every language (Greek and Romani, printed only by two units,
  folded into "other languages" for that check); 53,439 stated + 93 not stated = 53,532, the census
  total.
- Drawn: Ossetian 48,671 (91%), Georgian 3,681 (2,362 of them in Leningor/Akhalgori), Russian 801,
  Armenian 123, other 89, Azerbaijani 44, Ukrainian 13, Greek 11, Romani 6. Ossetians named Ossetian
  at 99.7%: retention is measured here, not assumed.

### Units and placement

- OSM district relations via `polygons.openstreetmap.fr` (ids in `ge_breakaway.py`; found with
  Nominatim because Overpass timed out on every area query that day). Tskhinval district's relation
  is invalid and `difference()` silently did nothing on it until `make_valid`. Sukhum city and
  Tskhinval city are holes in their districts' relations.
- Areas: Abkhazia 8,676 km² (Natural Earth 8,652, 8,300 shared); South Ossetia 3,851 km² (NE 4,463,
  3,750 shared; OSM follows the line of control, NE is larger).
- Placement: every Kontur GE hex whose centroid is in a district goes to it; 113 free hexes (17,842
  people) offshore on Abkhazia's coast snap to the nearest district within 5 km (only Georgian/sea
  ground in Natural Earth, never Russia). 26 hexes of Georgia's own layer (313 Kontur people, in
  Khashuri, Dusheti, Gori, Zugdidi, Kareli, Kaspi, Tsalenjikha) lie inside the OSM districts and
  leave Georgia's layer: COD-AB's municipalities run slightly past the line of control.
- Kontur/census, normalised: 0.46-1.44 for most units; **Sukhum city 0.48 and Sukhum district
  5.96** (OSM's city relation is the core; Kontur's suburbs sit in the district polygon), and
  **Tskhinval city 0.30** (Kontur barely covers it). Counts are the census's; only placement inside
  each unit follows Kontur.

### Room for improvement

Abkhazia's census asked nationality only; a native-language table (2003 or 2011) would replace
every retention share. A settlement-level placement would separate the Armenian villages of Gagra
and Gulripsh from the Abkhaz ones.
