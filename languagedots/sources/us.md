# United States: the record

Drawn 2026-10-04 (session d9e44929-us). 316,142,548 people aged 5 and over, 83,610 units
(83,608 tracts and two Suffolk County remainders), 122 nodes from 126 labels, 316,078 dots, no
rings. Scripts: `sources/us_acs.py` (tables and split), `sources/us_geo.py` (placement layer),
`taxonomy/us2024.py`, `taxonomy/tree.d/us.txt`, `countries/us.py`. Ask 003 (new roots).

## 1. Source and question

American Community Survey 2020-2024 5-year (U.S. Census Bureau, public domain), released 2026.
Question, asked of everyone aged 5+ including group quarters: "Does this person speak a language
other than English at home? What is this language?" One answer per person, so a bilingual home is
counted under its non-English language and "English" means English only at home. No decennial
census asks language. Puerto Rico (PRCS) is left to its own entry (`pr` in the queue).

Three products of the same sample, all from www2.census.gov, no key:

| product | grain | categories | file |
|---|---|---|---|
| C16001 | tract (85,382) | English only, Spanish, 11 groups | table-based summary file `acsdt5y2024-c16001.dat` |
| B16001 | PUMA (2,462) | English only + 42 groups | `acsdt5y2024-b16001.dat`; since 2016 not below PUMA/CD/metro |
| PUMS person file | PUMA | 125 LANP codes | `csv_pus.zip` (2.3 GB; aggregated once to `data/raw/us/pums_lanp_puma.csv`) |

The 2020-2024 PUMS carries 2020 PUMAs for every year (data dictionary), so no PUMA10/PUMA20 mix.
Code list `ACSPUMS2020_2024CodeLists.xlsx` (sheet Language) gives every detailed language inside
each LANP code; the mapping's remainder calls rest on it.

## 2. How the counts are built (all `tier` derived except as said)

Per tract t, C16001 group g, LANP code l in B16001 group b inside g:

    count(t, l) = C16001(t, g) * B16001(P, b) / B16001(P, g) * PUMS(P, l) / PUMS(P, b)

with P the tract's PUMA. English, Spanish, Korean, Vietnamese and Arabic are C16001 groups of one
code: tract counts as published, `measured` (292.7M people). The rest, 23.4M (7.4%), are a
tract's group count shared out by its PUMA's mix (`derived`). Where a PUMA's PUMS sample has
nobody in b although B16001 has people, the state's mix of b is used (177,143 people, 0.25% of
the non-English), then the nation's (1,513).

Why not PUMA counts alone: the tract table puts each group where it is inside the PUMA (a PUMA
is 128,000 people), and English and Spanish need no split at all. Why not C16001 alone: six of
its 12 groups are "other ..." buckets (Hmong, Navajo, Haitian, Hindi, Somali would all be
unnamed). The split assumes the languages inside a group are mixed the same way across a PUMA's
tracts; that is the one modelled step, and the `how`/`note_public` say so.

Under-5s (18,779,951, 5.6% of B01003's 334,922,499) are not asked; not drawn, in `gap`.

## 3. Checks (sources/us_acs.py; numbers from the last run)

1. Tracts (with the Suffolk remainders) sum to the national row in all 13 C16001 rows exactly
   (316,142,548).
2. Tracts sum exactly to 2,414 of 2,462 PUMAs in every row. The other 48 come in pairs off by
   equal and opposite amounts, 2,853 people in all: the 2020 tract-to-PUMA relationship file
   (religiondots/data/geo/tract_to_puma_2020.txt) puts a few tracts in the neighbouring PUMA
   from the one the 2024 tables count them in. Bounded at 5,000 and asserted pairwise (column
   sums zero); not fixed, since it only changes which PUMA's mix those tracts are split by.
3. The 42 B16001 groups sum to the 12 C16001 groups at every PUMA exactly. This asserts the
   B -> C crosswalk, which is not published as a table.
4. PUMS against B16001 nationally per B group: all within 5% for groups over 100,000 (largest:
   Swahili/Central-East-Southern Africa -4.6%, Malayalam/Kannada -2.7%, Punjabi +2.7%). English
   only +0.01%. The LANP -> B crosswalk is from the 2016 user note's examples and the code list;
   two PUMS codes straddle B groups at the detailed level (6795 "Other languages of Africa",
   1025 "Other English-based Creole"); 6795 is put with Western Africa (with "Other and
   unspecified" that group read +13.2%).

Connecticut: the 2024 tables use the 2022 planning-region county codes. CT tracts are keyed to
PUMAs by matching each cb_2024 tract's representative point to the cb_2020 tract holding it,
with the tract codes asserted equal (CT tract codes repeat across old counties, so the code alone
does not work).

**Suffolk County, NY: 14 tracts missing from the C16001 tract rows.** The county and PUMA rows
count their people (72,586 aged 5+); the tract file simply has no row for 36103122406,
...122501, 145601-145605, 145702, 146001, 146105, 146106, 146201, 146204, 201200 (pinned in
`MISSING_TRACTS`). Each PUMA's remainder (PUMA 3603310: 12 tracts, 3603313: 2) is its own unit
over those tracts' polygons, dots shared equally between them. Not looked into further (the
census API needs a key now); worth a look in the 2025 release.

## 4. Geography

`data/geo/us/us_tracts.gpkg`: cb_2024_us_tract_500k (religiondots/data/geo, read in place), 50
states and DC, 84,119 tracts -> 84,107 units; every unit with people has a polygon; 497 tracts
have none in the tables (water, parks). No `pop` column and `place_weight=None`: a unit is one
tract, and inside it a dot lands uniformly (religiondots' US layer does the same). Large rural
tracts therefore spread their dots over empty land (Navajo Nation, Alaska); Kontur hexes keyed
to tracts would fix that and are the obvious next step if it looks wrong. Water clip: 3,513
tracts touched, 0.31% of their area, cached in `data/geo/_waterclip`.

Witness, dots by area (scratch count): Navajo 123 of 145 in the Four Corners box; Inuit-Yupik-
Aleut 22 of 23 in Alaska; Hmong 84 Minnesota, 77 California; Pennsylvania German 75 Ohio and
Indiana, 66 Pennsylvania; Cajun French 11 of 13 Louisiana; Haitian 299 South Florida, 150 NYC;
Ilocano 52 of 89 Hawaii; Yiddish 112 of 219 NYC; Somali 68 of 161 Minnesota.

## 5. Mapping and tree calls

Every PUMS label is its own node, except the remainders (narrowest node; full list in
`taxonomy/us2024.py`'s docstring). The arguable ones:

- **"Chinese" (2.12M)** holds unspecified Chinese plus Hakka, Wu, Gan, Xiang, Min Bei, Min Dong.
  On the group node `sinotibetan.sinitic` (labelled "Chinese"), so it draws as "Chinese, language
  not named". Mandarin (0.80M), Cantonese (0.56M), Min Nan (0.08M) are leaves.
- **"Filipino" kept apart from Tagalog**: often a nationality answer, may hide Cebuano or Ilocano.
- **"Aleut languages"** holds Aleut, Inupiaq, the Yupik languages, Inuktitut and Greenlandic, so
  it sits on the family node `eskimoaleut`, not on an Aleut leaf.
- **"Uto-Aztecan languages"** is the family's name: on `utoaztecan`.
- **"India N.E.C.", "Other Languages of Asia", "Other languages of Africa", "Other and
  unspecified"**: `other` (each spans families).
- **"Other Native North American" and "Other Central and South American"**: `americas_other`,
  a root of its own (ask 003).
- Haitian, Kabuverdianu and Jamaican Creole sit under br.txt's `creole` root (French-, Portuguese-
  and English-based groups). Amharic, Tigrinya, Somali, Oromo, Nilo-Saharan use et.txt's nodes;
  Romance uses br.txt's (repeated identically).

Colours: hand-picked in the fragment for local contrast (AGENT_BRIEF §3): Spanish red-orange,
Haitian amber (Bengali is yellow next door in Brooklyn), Korean pink and Vietnamese purple apart
for Orange County, Samoan deep blue away from Tagalog's cyan. Not changed: Portuguese, which is
br.txt's node and generated a light blue from its pale-blue Romance group; beside English's
near-white in Massachusetts and New Jersey it is the weakest contrast on the US map.

## 6. Not done

- No second source compared (not needed for a whole-population question). The 2016+ ACS tract
  estimates carry large margins of error; a tract's small groups are noisy, and the 5-year
  pooling is what makes them usable at all.
- Kontur placement inside tracts (see §4).
- The 2.3 GB PUMS zip was deleted after the build; `us_acs.py` reads the cached aggregate
  `data/raw/us/pums_lanp_puma.csv` (delete that and run `--fetch` to rebuild it, ~5 min).
