# Northern Mariana Islands: the record

Drawn 2026-10-19 (session eca0185b-terr, "coverage gaps"). 44,111 people aged 5 and over, 21 tracts
with people, 11 nodes from 12 labels, 39 dots and 3 rings at 1:1000. Scripts: `sources/ia2020.py`
(shared reader), `sources/mp_census.py` (table and checks), `sources/mp_geo.py` (placement layer),
`taxonomy/mp2020.py`, `taxonomy/tree.d/mp.txt`, `countries/mp.py`. No asks. Built as Guam
(`sources/gu.md`), whose table is the same.

## 1. Source and question

2020 Island Areas Census of the CNMI, Demographic and Housing Characteristics Summary File (U.S.
Census Bureau, public domain), table PCT24 "Age by language spoken at home for the population 5
years and over". The Island Areas use the long-form questionnaire for everyone, so this is a full
count. The question: does this person speak a language other than English at home, and if so
which. One answer per person, so a Chamorro-and-English home counts under Chamorro and "English"
is English only. Universe: everyone aged 5+ (Guam's table excludes military housing; the CNMI's
does not exclude anyone).

Files (`data/raw/mp/`): `mp2020.dhc.zip`, `2020-iac-dhc-cnmi-table-matrix.xlsx` (which table is in
which segment), `2020-iac-dhc-geographic-header-record-as-cnmi-guam.xlsx`. URLs in
`sources/ia2020.py`. api.census.gov was not used.

PCT24 has 12 rows, as Guam. CNMI totals: Philippine languages 12,726 (28.9%), English only 11,466
(26.0%), Chamorro 9,579 (21.7%), Chinese 3,060 (6.9%), Carolinian 1,851 (4.2%), Korean 1,297,
Chuukese 1,046, other Pacific Island 648, Japanese 551, Palauan 506, other Asian 299, other 1,082.
Not stated is not a category; the census allocates missing answers (PBG83 gives the allocation
counts, not read).

No finer language detail exists for 2020; "Philippine languages" cannot be split.

## 2. Grain

PCT24 is published for the commonwealth, 4 municipalities, 12 district records, 26 tracts and
block groups. Nothing is suppressed (every record carries numbers, unlike Guam's six tracts).
Tracts are used: Saipan 17, Tinian 2 with people, Rota 1, the Northern Islands 1 (7 people).

## 3. Checks (`sources/mp_census.py`, numbers from the last run)

1. PCT24 adds up (bands to total, languages to band) in all 663 records.
2. The 26 tracts, and the 8 populated districts, each sum exactly to their 4 municipalities, and
   those to the commonwealth, in all 12 rows.
3. A second table of the same census: PBG5 (the same universe in five groups, by block group)
   summed per tract equals PCT24 collapsed to those groups in 25 of 26 tracts. Tract 8's block
   group 1 is missing from the geographic header (1,840 in P1 against 1,579 for the one group
   listed), the same header gap as Guam's tracts 9532 and 9543.
4. P1 (whole population) is 47,329; the 44,111 aged 5+ leave 3,218 (6.8%) under 5, the gap.

## 4. Mapping (`taxonomy/mp2020.py`)

Guam's mapping, label for label (`sources/gu.md` §4). Philippine languages on the Philippine
group ("language not named"), other Pacific Island languages on `pacific_other`, Chinese on the
Sinitic group, other Asian and other languages on `other`. The Bureau's "other Pacific Island
languages" here (648 people) are Micronesian languages other than Chamorro, Carolinian, Chuukese
and Palauan (likely Yapese, Pohnpeian, Kosraean, Marshallese), but the table does not say, so it stays a
remainder. No new node; `tree.d/mp.txt` repeats Guam's.
Carolinian is the CNMI's own language (1,841 of 1,851 speakers are in Saipan Municipality, where
it is 4.6% of people aged 5+).

## 5. Geography (`sources/mp_geo.py`)

`data/geo/mp/mp_tracts.gpkg`: cb_2020_us_tract_500k, STATEFP 69 (religiondots/data/geo, read in
place): 23 tracts, all 21 with people have a polygon; the two without (Tinian 9501.02 and 9502.02,
leased land) have nobody. Kontur MP (downloaded to `data/geo/kontur/`; 269 hexes, 49,952 people
against 47,329 in the census, ratio 1.055) cut to tracts as Guam's: 373 pieces, no hex outside a
tract. Kontur against each tract's P1, normalised: p10 0.84, median 1.06, p90 1.46, none outside a
factor of 3, log correlation 0.990 (best over 500 shuffles 0.956). Densest raw hex 3,325/km2, far
under the cap, so no cap rows. Kontur counts everyone; the table does too, so no base-housing lean.
Scatter: 0.19% of placement area was sea; 5,111 people (11.6%) are under one dot per language
nationally and draw as 3 rings.

## 6. Second source

Not searched. The CNMI Department of Commerce's household surveys may hold a language question that
corroborates the Philippine share; not pursued.
