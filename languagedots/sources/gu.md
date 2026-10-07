# Guam: the record

Drawn 2026-10-05 (session edd42a8c-gu). 135,770 people aged 5 and over, 45 tracts plus one unit
for six suppressed tracts, 11 nodes from 12 labels, 131 dots and 1 ring at 1:1000. Scripts:
`sources/gu_census.py` (table and checks), `sources/gu_geo.py` (placement layer),
`taxonomy/gu2020.py`, `taxonomy/tree.d/gu.txt`, `countries/gu.py`. No asks.

## 1. Source and question

2020 Island Areas Census of Guam, Demographic and Housing Characteristics Summary File (U.S.
Census Bureau, public domain), table PCT25 "Age by language spoken at home for the population 5
years and over in households (excluding people in military housing units)". The Island Areas
censuses use the long-form questionnaire for everyone, so this is a full count, not a sample.
The question: does this person speak a language other than English at home, and if so which.
One answer per person, so a Chamorro-and-English home counts under Chamorro and "English" is
English only.

Files (`data/raw/gu/`): `gu2020.dhc.zip` (5.6 MB, pipe-delimited segments, latin-1 because of
Hagåtña), `2020-iac-dhc-guam-table-matrix.xlsx` (which table is in which segment and order),
`2020-iac-dhc-readme.pdf`, plus two reference files read while scoping
(`2020-iac-language-code-list.xlsx`, `2020-iac-guam-dct-list-of-tables.xlsx`) and the geographic
header layout. api.census.gov carries the same table (`dec/dhcgu`) but refuses unkeyed requests;
the summary file needs none.

PCT25 has 12 rows: speak only English, Chamorro, Carolinian, Palauan, Chuukese, Philippine
languages, other Pacific Island languages, Chinese, Japanese, Korean, other Asian languages,
other languages. Guam totals: English only 57,906 (42.7%), Philippine languages 33,297 (24.5%),
Chamorro 21,390 (15.8%), Chuukese 7,687 (5.7%), Korean 3,299, other Pacific Island 3,243,
Japanese 2,851, Chinese 2,114, other 1,907, Palauan 1,479, other Asian 487, Carolinian 110.

No finer language detail exists for 2020: the detailed cross-tabulations (CT25, CT46...) use
coarser groups still, and there are no public Island Areas microdata. So "Philippine languages"
cannot be split into Tagalog, Ilocano, Cebuano and the rest.

## 2. Grain and suppression

PCT25 is published for Guam, the 19 villages and 55 tracts (no block groups). Tracts are used.

The Bureau prints "." for six tracts and three villages, in every sample-type table:

| tract | village | total population (P1) |
|---|---|---|
| 9516 | Barrigada | 142 |
| 9519.01 | Tamuning | 4,081 |
| 9519.02 | Tamuning | 3,484 |
| 9524 | Tamuning | 1,637 |
| 9534 | Hagåtña (the whole village) | 943 |
| 9554 | Umatac (the whole village) | 647 |

Guam's own row is published, so Guam minus the 49 published tracts is exactly the six tracts'
people, row by row: 9,674 in the universe (0.88 of their P1, the same as Guam's 0.88). They are
one unit, `66010SUPPR`, with the census's counts. Inside it, each tract's placement pieces are
scaled to that tract's own P1, so Umatac gets about 6% of the pooled dots and Tamuning about
85%, but the mix is the pool's: an Umatac dot can be Korean where the true Umatac mix is mostly
Chamorro and English. At 1:1000 the pool is about 10 dots, Umatac under one. A finer split from
the published tract-in-village parts and block groups (PBG5, six groups) is possible in part but
not worth it at this size.

Tracts 9501 (Andersen), 9502, 9503, 9518, 9544, 9545 and 9801 hold under 100 people each in the
table though several hold over 600 in P1: military housing and group quarters are outside the
universe.

## 3. Checks (`sources/gu_census.py`; numbers from the last run)

1. PCT25 adds up (bands to total, languages to band) in all 506 published records.
2. The suppressed cells are exactly the pinned ones; Guam minus the published tracts (9,674) and
   minus the published villages (18,308) are non-negative in every row.
3. Where a tract and all its tract-in-village parts are published, the parts sum to it exactly:
   45 of 49 tracts and 11 of 17 villages. Completeness is judged by P1, because the geographic
   header leaves some parts out altogether (tract 9532 lacks a part of 27 people, 9543 one of
   91).
4. A second table: PBG5 (same universe, six groups, by block group) summed to tracts equals
   PCT25 collapsed to those groups in all 39 published tracts whose block groups are all
   published.

The 55 DHC tracts sum to 153,776 in P1 against Guam's 153,836; cb_2020 has 56 tracts, two
(9528, 9535) not in the DHC at all. The 60 people are not placed by any P1 weight but are inside
Guam's language totals, so they sit in the suppressed pool's residual or nowhere; immaterial.

## 4. Mapping (`taxonomy/gu2020.py`)

| label | node |
|---|---|
| Speak only English | indoeuropean.germanic.english |
| Speak Chamorro | austronesian.chamorro |
| Speak Carolinian | austronesian.oceanic.carolinian (new) |
| Speak Palauan | austronesian.palauan |
| Speak Chuukese | austronesian.oceanic.chuukese |
| Speak Philippine languages | austronesian.philippine (group: draws as "language not named") |
| Speak other Pacific Island languages | pacific_other (new root) |
| Speak Chinese | sinotibetan.sinitic (variety not named) |
| Speak Japanese | japonic.japanese |
| Speak Korean | koreanic.korean |
| Speak other Asian languages | other |
| Speak other languages | other |

Glottolog (`data/raw/glottolog/languages.csv`) puts Chamorro (cham1312), Palauan (pala1344),
Carolinian (caro1242) and Chuukese (chuu1238) in Austronesian. Carolinian sits beside Chuukese
under Oceanic (both Chuukic).

`pacific_other` (colour 0.66 0.06 210) rather than Austronesian or Oceanic: the Bureau's Pacific
Island language group also files Papuan languages (the 2020 Island Areas code list puts Kuman,
Enga and Wahgi beside Motu), so no family node holds everything in the label. In Guam it will be
mostly Pohnpeian, Yapese, Kosraean and Marshallese. It is a regional remainder like
`seasia_other`, kept off `other` per the brief's rule for indigenous remainders. "Other Asian
languages" goes on `other` as us2024 maps "Other Languages of Asia", since it spans families.

Colours: nothing hand-picked. English pale blue, Chamorro green (#44b782), Chuukese dark teal
(#009192), Philippine washed teal-grey (#9cc7c7), Korean pink, Japanese pale pink, Chinese
orange-red. Chamorro, Chuukese and the Philippine group are the neighbours that matter and
read apart; Japanese and Korean are both pinks but differ in lightness.

## 5. Geography (`sources/gu_geo.py`)

`data/geo/gu/gu_tracts.gpkg`: cb_2020_us_tract_500k, STATEFP 66 (religiondots/data/geo, read in
place): 56 tracts, every published tract with people and all six suppressed ones have a polygon;
5 others have nobody. Median tract 5.93 km2 (8 hexes' worth), p10 1.69, so the hexes are cut to
the tracts as Puerto Rico's are: Kontur GU (November 2023, downloaded to
languagedots/data/geo/kontur; 451 hexes, 171,958 people) gives 784 pieces, each hex's people
shared over its land pieces by area; 1 hex (3 people) touches no tract. Densest raw hex 4,384/km2,
far under the cap, so no cap rows.

Kontur against each tract's P1, normalised (national ratio 1.105): p10 0.73, median 0.97, p90
2.80; 5 of 51 tracts outside a factor of 3, all small base or airport tracts (9518 at 15.6x,
9544, 9516, 9801, 9503) where Kontur counts the base and P1 counts few. Log correlation 0.859
against a best of 0.455 over 500 shuffles. Kontur includes military housing, which the table
excludes, so inside a mixed tract the dots lean a little towards base housing.

Scatter: 2.54% of placement area was sea; one unit lost over 95% to the sea and was left whole
(water.py's rule). 4,770 people (3.5%) are under one dot per language nationally.

## 6. Second source

Not searched. A Guam Bureau of Statistics and Plans report or the 2010 census could hold a
split of the Philippine group at island level; a split by national shares would change counts
per tract, so it would be Anita's to allow, and it was not pursued.
