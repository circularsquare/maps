# Qatar (`qa`): the 2004 census's religion by municipality

Drawn 2026-09-15 by session `d743fc47-qa`, from Table 6 of the 2004 census, at the ten
municipalities of 2004. 744,029 people; 743 dots at 1:1,000 and 73 at 1:10,000. Code:
`sources/qa.py` (the tables), `sources/qa_geo.py` (units and placement), `taxonomy/qa2004.py`,
`countries/qa.py`. sources.md §qa-2026-09-15 is the summary.

## 1. The source, and what else was looked at

**Table No (6), "Population By Religion, Gender And Municipality, March 2004"**, from the 2004
census's population tables, published by the Planning Council's Statistics Department and later
hosted by the Ministry of Development Planning and Statistics:
`http://www.mdps.gov.qa/en/statistics/census/Census2004/Population/Pages/Tables/Pubulation/T06.aspx`
(the folder is spelled `Pubulation`), Wayback capture 20170627120742, CDX digest
`BSAO3D3IWND3W6IKEG3DSUCZGN37AZWU`, 84,269 bytes. One HTML table, English and Arabic labels
side by side: persons, males and females in Muslim, Christian and Other, for the ten
municipalities and the nation. The ministry's site is gone (it is now the National Planning
Council, `npc.qa`); only Wayback has the tree.

Tables 1 to 5 of the same folder are read as witnesses (population by sex and municipality; by
sex, municipality and zone; age by municipality for persons, males and females), each pinned by
its CDX digest in `sources/qa.py`.

What else was looked at, 2026-09-15:
- **The rest of the 2004 tree, by folder name only.** A CDX prefix query on
  `mdps.gov.qa/en/statistics/census/Census2004/` lists population table folders `Pubulation`
  (T01-T06), `Educational-Status`, `Economic-Activity`, `Marital-Status`,
  `Using-Computer-And-Internet`, `-qa` twins of several of those, `Fertility-qa`,
  `Special-Needs-qa` and `HouseholdsAnd-Individuals`, plus the Buildings, Establishment and
  ResidentialUnits trees. No folder is named for religion, and only `Pubulation` was opened, so
  a religion table by nationality inside another folder is not ruled out.
- **Later censuses.** The 2010 results booklet, the archived 2010 table tree and the 2020 results
  workbook have no religion table (sources.md §11ao, which read all 209 sheets of the 2020
  workbook). The 2020 workbook's contents sheet was read again here: no religion title among
  its first 52 tables. Not checked: the 2015 simplified census, the 1986 and 1997 censuses, and
  whether the 2010 or 2020 forms asked religion.
- **UNSD Demographic Yearbook table 28** has Qatar 2004 only, national, and equal to Table 6.

## 2. Reconciliation

All pass (`python sources/qa.py`):
- all six pages match their pinned digests and carry their captions;
- Table 6 parsed off the page equals the transcription in `sources/qa.py`;
- persons = males + females in all 44 cells; the ten municipalities sum to the national row in
  all 12 columns; the three religions sum to the totals in every row, for both sexes;
- Table 1 equals Table 6's total columns;
- Table 2's 87 zones sum to their municipality's printed total, and those totals equal Table 1,
  persons, males and females;
- Tables 3, 4 and 5: the 18 age rows sum to the total row in all 11 columns, the totals equal
  Table 1, and Table 3 = Table 4 + Table 5 cell by cell;
- UNSD table 28, Qatar 2004: Muslim 576,391, Christian 63,212, Other 104,426, total 744,029.

| municipality (2004) | people | Muslim | Christian | Other | men among Muslim / Christian / Other |
|---|---:|---:|---:|---:|---|
| Doha | 339,847 | 79.13% | 10.15% | 10.73% | 64% / 55% / 80% |
| Al Rayyan | 272,860 | 78.31% | 6.03% | 15.66% | 62% / 71% / 96% |
| Al Wakra | 31,441 | 85.14% | 4.03% | 10.84% | 55% / 53% / 84% |
| Umm Salal | 31,605 | 87.80% | 6.38% | 5.82% | 55% / 67% / 84% |
| Al Khor | 31,547 | 56.10% | 11.07% | 32.83% | 71% / 78% / 96% |
| Al Shamal | 4,915 | 86.19% | 6.78% | 7.04% | 61% / 76% / 91% |
| Al Ghuwairiya | 2,159 | 77.17% | 2.41% | 20.43% | 79% / 77% / 98% |
| Al Jemailya | 10,303 | 65.83% | 9.27% | 24.91% | 69% / 73% / 85% |
| Jeryan Al Batna | 6,678 | 60.18% | 33.98% | 5.84% | 76% / 98% / 96% |
| Mesaieed | 12,674 | 38.51% | 15.01% | 46.47% | 79% / 94% / 94% |
| Qatar | 744,029 | 77.47% | 8.50% | 14.04% | 63% / 64% / 89% |

## 3. What the numbers are: Qataris counted, everyone else sampled

The census's population introduction page
(`.../Census2004/Population/Pages/Introduction.aspx`, Wayback 20170625221218) says:
- the universe is everyone "residing within the boundaries of the State at the reference point
  of time on the 16th March 2004, at their place of presence whether in residential units or
  labor gatherings, excluding those living in public houses, those on transit at air parts, at
  land customs outlets and on board vessels in territorial waters";
- "Data was fully collected in regards to building, dwellings, units, Qatari population and
  establishments while non-Qatari population residing in the State, whether in residential
  units or labor gatherings, were enumerated via a sample";
- the non-Qatari household sample (with small labour gatherings) was stratified by municipality,
  selected systematically after sorting by zone and household size, with square-root allocation
  across the ten municipalities; people in large labour gatherings were sampled individually in
  each municipality;
- the final weight is design weight x response weight x census control weight, and the control
  weights "calibrate the sample-based estimates to the March 2004 census results based on
  complete counts of the non-Qatari population", separately for males and females, by
  municipality, for households and for compounds.

So each municipality's total and sex split in Table 6 is a count, and the religion split among
its non-Qataris is a weighted estimate. The page gives no sample sizes, so a small cell such as
Al Ghuwairiya's 52 Christians may rest on a handful of sampled people. No table read here gives
the non-Qatari share of the population, or crosses religion with nationality.

**Tier: `measured`, my call.** Spec §7's measured tier is a question asked and answered in the
unit; here it was, of everyone for Qataris and of a sample for non-Qataris, and the published
unit is the sample's own stratum and calibration level. That is the footing of a census
long-form sample, not of a project laying a survey's shares on someone else's count (American
Samoa's `modelled` rows). A split tier is not possible: Table 6 does not separate Qataris from
non-Qataris. Reversing it is one line in `countries/qa.py::_qa_counts`; tiers change no colour.

The people the universe leaves out (hotel residents, travellers in transit, crews) are not
quantified anywhere read, so there is no `gap`.

## 4. The questionnaire and the mapping

UNSD's copy, `QAT2004en.pdf` (four pages), is the Qatari form: "Register of Qatari
Characteristics" and "Questionnaire of Qatari Characteristics", form 4 P.C., headed "Buildings,
Dwellings, Households and Establishments Census And Sample Population Characteristics Survey
2003/2004". Column 6, Religion, read from the page rendered at 7x: **1 Muslim, 2 Christian,
3 Other.** There is no code for no religion or for not stated. The introduction page lists a
non-Qatari form (4BPC) and a labour gatherings form; neither was found.

| Table 6 | node | why |
|---|---|---|
| Muslim | `islam` | the form asks only Muslim; no Sunni or Shia split is published |
| Christian | `christianity` | no church is asked |
| Other | `other.qa` (new) | code 3, the only box for Hindus, Buddhists and anyone with no religion |

`Other` is not handled as a lumped no-religion box (the draft procedure, playbooks/census_table.md):
its label is Other, not None, and it is the only home of the Hindus and Buddhists of a largely
South and South-East Asian workforce, so `unknown` would hide what the category mostly is.
`tools/check_no_religion.py` raises nothing for it. REVIEW in `taxonomy/qa2004.py` carries each
call with its figures.

## 5. Geography: the 2004 municipalities rebuilt from zones

Qatar had ten municipalities in 2004 and has eight. COD-AB's membership shows four of 2004's are
gone: Al Ghuwairiya (zone 76) is in Al Khor and Al Thakhira; Al Jemailya (72, 73, 84-86) in Al
Sheehaniya; Jeryan Al Batna split between Al Sheehaniya (82) and Al Rayyan (83, 96, 97); Mesaieed
(92-94) in Al Wakra. Zone 57, the Industrial Area, was Al Rayyan's in 2004 and is Doha's now;
zones 69 and 70 were Doha's and Umm Salal's and are Al Daayen's.

The zones kept their numbers. **COD-AB Qatar v02** (HDX `cod-ab-qat`, shapefile zip, zones last
edited 30 November 2015, from the Planning and Statistics Authority) has 91 zones; each pcode
ends in the zone number and each name ends in the same number. **Table 2** lists 87 zones under
their 2004 municipality. So each 2004 municipality is the union of the COD-AB zones with its 2004
zone numbers, and `sources/qa_geo.py` asserts the two lists' differences exactly:
- **2004 zones with no polygon: 10 and 11**, Wadi Al Sail (East) and Al Rumeila (East). They are
  folded into zones 20 (COD-AB `Wadi Al Sail 20`) and 21 (`Rumaila 21`), both in Doha, which
  moves 9,157 people's placement within central Doha and no count.
- **COD-AB zones with no 2004 row: 46, 49, 50, 58, 98, 99.** A zone of 9 people is listed in
  2004, so these had nobody or did not exist. They carry no weight; each polygon is given to a
  2004 municipality only so the units tile the country: 46 and 50 (`Al Thumama`, COD-AB's name
  for 47 too) and 49 (Hamad International Airport, beside 48) to Doha, 58 (Wholesale Market,
  beside 57) to Al Rayyan, 98 (Al Adaid, beside 95) to Al Wakra, 99 (2.7 km2, unnamed) to Al
  Shamal.

Two witnesses the zone number does not decide:
1. **Names.** 65 of the 85 shared zone numbers have a 2004 name and a COD-AB name that share a
   word or are nearly the same spelling. The 20 that do not are pinned in `ZONES_RENAMED` as
   read, none explained by a source: eight of them are 2004's `New District Of Doha nn` or
   COD-AB's `Zone nn`, one is a translation (`Al Matar Al Qadeem`, `Old Airport`), and the rest
   are other neighbourhood names for the same number.
2. **Kontur's rank of the zones.** Kontur 2023 summed per zone against Table 2's 2004 people,
   over the 85 zones with people: Spearman **0.876**; 1,000 random reassignments of the 2004
   populations reach at most 0.365. The bar, 0.5 and above every shuffle, was set before the
   run. Qatar has grown about fourfold, so this tests the pattern, not a level.

Rebuilt areas: Doha 240 km2, Al Rayyan 844, Al Wakra 1,876, Umm Salal 567, Al Khor 980, Al
Shamal 863, Al Ghuwairiya 636, Al Jemailya 2,577, Jeryan Al Batna 2,372, Mesaieed 682. No 2004
area table was found to check these against (not checked: the 2004 and 2005 Annual Statistical
Abstracts). What could still be wrong: a zone whose boundary changed between 2004 and 2015 while
keeping its number (zone 91 Al Wukair beside the new 46 and 50 is the obvious candidate). That
moves dots between neighbouring zones, and between municipalities only on a 2004 municipal
border.

## 6. Placement

Kontur QA (2023-11-01, 8,028 populated hexes, 2,717,091 people) is cut to the zones and each
piece's people shared by area; 0.51% of Kontur falls in no zone. Every zone's pieces are then
scaled to that zone's 2004 population, so the weights inside a municipality are 2004's, zone by
zone, and Kontur only says where inside a zone. Every municipality's weights are asserted to sum
to its Table 1 population. No zone with 2004 people lacked a Kontur piece.

**Why not plain Kontur.** The dots per municipality are Table 6's either way, but inside a
municipality plain Kontur 2023 would put **16.6%** of the dots in a different zone than 2004's
populations do: Doha 18.0% (Onaiza, Leqtaifiya and Al Qassar, zone 66, is 1.3% of Doha's 2004
people and 4.0% of its Kontur), Al Rayyan 14.8% (the Industrial Area, zone 57, 22.9% against
33.6%), Al Khor 18.8% (Al Thakhira and Ras Laffan, zone 75, 42.8% against 61.7%), Mesaieed
30.5% (the Mesaieed Industrial Area, zone 93, held 9 people in 2004 and 20.9% of the
municipality's Kontur people). Measured by a scratch script over `data/geo/qa/qa_lookup.csv`.

Kontur's share against the census's, by 2004 municipality: Doha 45.7% of people in 2004 and
30.4% of Kontur; Al Rayyan 36.7% and 42.3%; Al Khor 4.2% and 11.8%; the rest within about a
point.

Scatter: water clipped 466 of 9,074 pieces (0.85% of their area was sea), left 12 entirely-water
pieces and 3 over the 95% rule whole; `kontur_cap.py qa` found no block to stop on. 743 dots in
510 polygons at 1:1,000, with 1,029 people (0.14%) under one dot nationally; 73 dots at 1:10,000.

## 7. The note

Three paragraphs: what the census published and the headline shares (Other explained as the
form's third answer, with Mesaieed, Al Khor and Jeryan Al Batna); that these are 2004 figures,
with the 2020 census's **2,846,118** (Planning and Statistics Authority, *Census of Population,
Housing and Establishments, December 2020, Detailed Results*, `Census_Final_Results.xlsx` sheet
`1`, Table 1, from `npc.qa` with `curl -k`) and the zone weighting; and that non-Qataris'
religion comes from a weighted sample.

## 8. Calls someone might reverse

1. **Drawing a 22-year-old table at all.** 2004's 744,029 is 26% of the 2020 count. The
   Northern Mariana Islands were closed on vintage by a scout at about a third of today's
   population (queue.md, Asia and Oceania sweep), while Senegal is drawn from 1988. I drew Qatar
   because this is the only religion table below the nation that any census published, it counts
   labour camps as well as households, and the note says the year in its second paragraph. To
   reverse: take `qa` out of `ORDER`, delete `countries/qa.py`, and run the build tail.
2. **`measured` for a table whose non-Qatari half is a calibrated sample** (§3).
3. **The six empty zones' municipalities** (§5), which move no dots.

## 9. Not checked, and where it reopens

The other 2004 table folders; the 1986 and 1997 censuses; the 2015 simplified census; whether
the 2010 and 2020 forms asked religion; the non-Qatari and labour gatherings forms; a 2004 area
table; the sample sizes. Reopens on any newer religion table by municipality or zone.

## 10. Review, 2026-09-15 (`d743fc47-rev13`)

Full pass. `check_md.py` clean, `built_countries.py --check` ok, rollup clean (744,029, all
`measured`). Every `note_public` figure and the grain recompute off `qa.csv`: 77.5% Muslim, 8.5%
Christian, 14.0% Other; Mesaieed 46.5% Other and Al Khor 32.8%; Jeryan Al Batna 34.0% Christian;
744,029 is 26.1% of 2,846,118.

- **The Gulf ruling does not touch it.** I read the introduction page myself (Wayback
  20170625221218). Its "Scope of the Census" covers everyone present, labour gatherings included,
  and says non-Qataris "were enumerated via a sample", whose data "was appraised ... and added to
  the Qatari population data to achieve estimates to the States' overall level"; the planning list
  on the same page says it again. So Table 6 is the whole population, with the migrant majority
  measured by a sample calibrated to its full count by municipality and sex. That is neither a
  citizens-only figure nor a large unmeasured share. `measured` agreed.
- **The vintage shows more than it misleads, so it stays.** The one newer composition on disk is
  Pew's modelled 2020 figure (`data/raw/estimates/pew.zip`, `Religious Composition 2010-2020
  (percentages).csv`; its inputs for Qatar were not read): Muslim 75.9%, Christian 12.5%, Hindu
  10.6%, everything else 1.0%, so 11.6% outside Islam and Christianity against Table 6's 14.0%.
  The 2004 levels are within about 4 points of that, and they do not all lean one way (Christian
  lower, Other higher). What the table mostly gets wrong is magnitude (a quarter of today's people)
  and 2004's settlement pattern, and the note's second paragraph says both. Senegal (1988, drawn
  without rescaling) is the precedent; the Northern Mariana Islands closure was a 1973 table.
- **Mapping: agreed.** `other.qa` for a box that holds Hindus, Buddhists and anyone with no religion
  follows Brunei's `Others` on `other.bn`.
- **One wording point, not edited.** "The form offered only those three answers" rests on the
  Qatari form; §4 says the non-Qatari and labour gatherings forms were not found. The conclusion
  holds either way, because Table 6 prints only three columns. "The census published only those
  three answers" would be exact.
- §14: nothing, at 2004 municipalities of 74,000 people on average.

Screenshot clean: Doha and Al Rayyan dense, a thin scatter north to Ras Laffan and south to
Mesaieed, nothing offshore.
