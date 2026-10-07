# Puerto Rico: the record

Drawn 2026-10-04 (session d9e44929-pr). 3,132,790 people aged 5 and over, 921 tracts, 27 nodes
from 28 labels, 3,128 dots and 5 rings at 1:1000. Scripts: `sources/pr_acs.py` (tables and
split), `sources/pr_geo.py` (placement layer), `taxonomy/pr2024.py`, `taxonomy/tree.d/pr.txt`,
`countries/pr.py`. No asks.

## 1. Source and question

Puerto Rico Community Survey (PRCS) 2020-2024 5-year, U.S. Census Bureau, public domain. The
PRCS is the American Community Survey run in Puerto Rico: same question, same tables, same PUMS
code list, released with the ACS. Asked of everyone aged 5+, group quarters included: "does this
person speak a language other than English at home? What is this language?" One answer per
person, so a Spanish-and-English home counts under Spanish and "English" is English only. No
decennial census asks language. The queue's lead was the 2019-2023 release; the 2020-2024
release is used because the US entry is on it and its summary files were already on disk.

Same three products as the US (`sources/us.md` §1): C16001 by tract, B16001 by PUMA, PUMS. The
two summary files are the US build's (`data/raw/us/acsdt5y2024-{c,b}16001.dat` cover Puerto
Rico too), read in place. The only new download is Puerto Rico's own PUMS person file,
`csv_ppr.zip` (18 MB, `data/raw/pr/`), aggregated to `data/raw/pr/pums_lanp_puma.csv`.

National figures (C16001, Puerto Rico's row): English only 141,844 (4.53%), Spanish 2,986,236
(95.32%), everything else 4,710 (0.15%): French/Haitian/Cajun 1,234, Other Indo-European 1,189,
Chinese 861, German/West Germanic 451, Arabic 422, Other 191, Other Asian 132, Korean 85,
Slavic 75, Tagalog 44, Vietnamese 26. Under-5s: B01003 total 3,234,309, so 101,519 (3.1%) not
asked, in `gap`.

## 2. How the counts are built

Exactly the US method (`sources/us.md` §2), on Puerto Rico's rows: each tract's C16001 group is
shared out by its PUMA's mix (B16001 group within the C group, PUMS code within the B group).
`pr_acs.py` imports the US script's group lists and both crosswalks (`B_GROUPS`, `LANP_B`)
rather than repeating them. English, Spanish, Korean, Vietnamese and Arabic are tract counts as
published (`measured`); the rest, about 4,400 people, `derived`. Where a PUMA's PUMS sample has
nobody in a B group that B16001 has people in, Puerto Rico's own mix of that group is used (206
people), then the 50 states' (none needed).

At 1:1000 only Spanish (2,986 dots), English (141) and French (1) draw dots; the scatter reports
4,790 people under one dot per language nationally. So the split matters for the legend and for
anyone reading the counts, not for what the map shows. It was kept because it is the same code
path as the US and costs one 18 MB download.

## 3. Checks (`sources/pr_acs.py`; numbers from the last run)

1. Tracts sum to Puerto Rico's row exactly in all 13 C16001 rows (981 tract rows, 921 with
   people).
2. Tracts sum exactly to 22 of 24 PUMAs. PUMAs 7200500 and 7200600 are off by +33/-33 Spanish
   speakers: the 2020 tract-to-PUMA relationship file puts one tract in the neighbouring PUMA
   from the one the 2024 tables count it in, the same thing the US build found 24 times.
   Asserted pairwise (column sums zero), bounded at 500; it only changes which PUMA's mix that
   tract's (Spanish-only) groups are split by, which for Spanish is the identity.
3. The 42 B16001 groups sum to the 12 C16001 groups at all 24 PUMAs exactly.
4. PUMS against B16001 for Puerto Rico: Spanish -0.04%, English only +1.07%. The small groups
   read -60% to +50% (Russian 75 published, 30 in PUMS), which is sampling: PUMS is 126,569
   persons, so a group of 75 people is about three respondents.
5. Every LANP code in Puerto Rico's PUMS is in the US crosswalk (asserted).

## 4. Geography

`data/geo/pr/pr_tracts.gpkg`: cb_2024_us_tract_500k, STATEFP 72 (religiondots/data/geo, read in
place): 939 tracts, all 921 with people have a polygon; 18 have nobody in the tables.

The median tract is 3.32 km2, 4.5 Kontur hexes' worth, p10 0.51 km2, so a centroid join would
leave the small San Juan-area tracts without a hex (playbook: "Below the floor, cut the hexes to
the units"). Kontur PR (November 2023, religiondots/data/raw/pr/, read in place; 14,908 hexes,
3,260,315 people) is cut to the tracts and each hex's people shared over its land pieces by area:
24,193 pieces, median 15 per tract, minimum 2, none where Kontur has nobody. 24 hexes (85 people)
touch no tract and are dropped. Raw Kontur's densest hex is 9,391/km2, far under the 46,200 cap,
so no cap rows; the file is named `_tracts`, not `_hexes`, because religiondots' cap check would
measure density on pieces.

Kontur against the PRCS's people 5+ per tract, normalised (national ratio 1.036): p10 0.73,
median 0.99, p90 1.33; 3 of 921 outside a factor of 3, the extreme two special-use tracts
(72033980005 in Catano at 57x, a port and industrial tract with few residents; 72113980001 in
Ponce at 0.08). Log correlation 0.832 against a best of 0.103 over 500 shuffles. The ratio only
shapes dots inside a tract; tract totals are the survey's.

Water clip at scatter time: 0.11% of placement area was sea.

## 5. Mapping and tree calls

`taxonomy/pr2024.py` is the US mapping (`us2024.NAMES`) cut to the 28 labels Puerto Rico's rows
carry; a new label fails loudly rather than passing through on the US mapping unseen. The calls
are the US ones (`taxonomy/us2024.py` docstring): "Chinese" (580, variety not named) on the
Sinitic group, "India N.E.C." (42) and "Other and unspecified languages" (77) on `other`, "Other
Indo-European languages" (78) on Indo-European, "Other Bantu languages" (32) on Bantu, "Filipino"
(39) apart from Tagalog.

No new nodes. `tree.d/pr.txt` repeats bare (no colour) the fragment nodes Puerto Rico uses, so
the country stands alone under `build.py --only`; colours come from us.txt and the others, and a
retune there cannot clash with this file. Colours not touched: Spanish (red-orange) against
English (near white) is the only pairing that shows on the map, and it is the US one.

## 6. Not done

- No second source compared. English-only share by municipio runs 3-8% (Sabana Grande 7.7%, Culebra
  7.3%), flat rather than clustered.
- The 5-year tract estimates carry margins of error; for Puerto Rico only the English-only count
  per tract is small enough for that to matter.
