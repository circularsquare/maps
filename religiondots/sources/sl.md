# Sierra Leone — 2015 Population and Housing Census, religion by district

**Drawn 2026-09-15** (session `d743fc47-sl`). 14 districts (the 2015 set), 6 categories,
7,076,119 people in households, every row `measured`. 7,073 dots at 1:1,000, 705 at 1:10,000.

- `sources/sl.py` -> `data/normalized/sl.csv` (the PDF is
  `data/raw/sl/sl_2015_phc_thematic_report_on_pop_structure_and_pop_distribution.pdf`, pinned)
- `sources/sl_geo.py` -> `data/geo/sl/sl_districts.gpkg`, `sl_hexes.gpkg`, `sl_lookup.csv`
  (COD-AB v02 chiefdoms regrouped to 2015, geoBoundaries ADM2 as witness, Kontur 400 m)
- `taxonomy/sl2015.py` -> the mapping; `countries/sl.py` -> the entry; `taxonomy/branches.py`
  `other.sl` is the one new node
- sources.md **§sl-2026-09-15** is the summary; **§11aq** was the scout's row.

```
python sources/sl.py     --fetch
python sources/sl_geo.py --fetch
```

## 1. What Statistics Sierra Leone publishes

| release | religion | tier |
|---|---|---|
| **2015 PHC, *Thematic Report on Population Structure and Population Distribution*** (Oct 2017, 54 pp) | **Table 5.3, region and district x six answers, % of household population, one decimal** (PDF pp.37-38) | **14 districts** |
| 2015 PHC, *National Analytical Report* (584 pp) | Table 3.25, the eleven codes nationally with sex ratio and % under 15; Table 4.6, households and household population by the HEAD's religion; Tables 8.10a-b, marital status by religion | national |
| *2015 population census data for 16 districts, 5 regions* (7 pp) | none; total population by the 2017 chiefdoms, male and female | 190 chiefdoms (2017) |
| 2015 PHC, Census Atlas | ethnicity by chiefdom, no religion (scout, §11aq) | |
| 2004 census, SLUPS/UNFPA *Population Profile* Table 8A | 14 categories (Ahmadi, Sunni, Shia apart) in counts, for the nation, Bo District and Bo Town, and archived for Western Area Urban and Bombali (scout, §11aq) | 3 districts |
| 2021 mid-term census | asked religion with denominational codes; nothing published (scout, §11aq) | |
| SLIHS 2018 microdata | asks religion; posted openly on the Stats SL site (Wayback 2019-11-13); not used | |
| UNSD Demographic Yearbook table 28 | absent | |

**Not opened this session, so not a negative:** the other fourteen 2015 thematic reports listed in
the Wayback CDX for `www.statistics.sl/images/StatisticsSL/Documents/Census/2015/` (agriculture;
children, adolescents and youth; disability; economic characteristics; education and literacy;
elderly; gender; housing conditions; life tables; migration and urbanisation; mortality; nuptiality
and fertility; population projections; poverty and durables; Ebola). A religion-by-chiefdom table
would most plausibly be an annex of one of these; the analytical report's disability chapter
points to a chiefdom-level Appendix Table A14.1, so chiefdom annexes exist in that series.

## 2. The construction

    shares        Table 5.3, district x religion, one decimal               thematic report pp.37-38
    denominators  Table 2.2, total population by district, counts           thematic report p.17
    universe      household population 7,076,119 of 7,092,113               analytical report Tables 4.1, 4.6

Each district's six shares are divided by their own printed sum (99.9 to 100.1) and applied to its
total population times 7,076,119 / 7,092,113. There is no per-religion rescale to a national count
because none exists in persons: Table 3.25 is percentages, and Table 4.6's 5,506,388 "Islam" is the
people living in Muslim-headed households, not Muslims. Household population by district is not
printed in counts anywhere found (Table 4.3 gives households, Table 4.5 a one-decimal mean size,
Table 4.10 a one-decimal head share), so the 15,994 in institutions are taken off every district
at the national rate. `gap` is those 15,994, 0.23%, hand-written: they are in no table, so
`tools/gap_share.py` correctly refuses.

## 3. The checks (`sources/sl.py::check`)

| check | result |
|---|---|
| PDF pinned | live file 2026-09-15 = Wayback 2019-04-10 capture, digest `H4DQRV6Q6CHUWXAOK43BI4NRANWBZ7J2`, 4,192,879 bytes |
| Table 5.3 parsed off the page = transcription | 19 rows (nation, 4 regions, 14 districts), identical |
| district and region rows close | within 0.10 pp |
| national row | sums to 100.5; 100.0 with Bahai read as 0.0 (pinned misprint) |
| district totals | all 14 on Table 2.2's page; sum by region = Table 5.4's four region totals; sum 7,092,113, also in Table 3.2 |
| household share | 7,076,119 / 7,092,113 = 99.77%, the report's prose "99.8 per cent" |
| districts weighted to regions | worst 0.07 pp (Southern, Islam) |
| districts weighted to the nation | worst 0.06 pp; Bahai weights to 0.036 |
| Table 3.25 witness | Islam, Traditional, Other, No Religion equal Table 5.3's national row; its six Christian codes sum to 21.9 without the stray row |

## 4. The questionnaire and the codes

Household form (IPUMS `enum_form_sl2015a.pdf`, p.1): P05 *What is (NAME's) religion?* "Write the
code of the religion, use code list". Code list (`Census_2015_tools/2015-slphc_codes_list_web.pdf`,
Wayback 20201114085519, rendered and read; its text layer interleaves the columns): **01 Catholic,
02 Anglican, 03 Methodist, 04 SDA, 05 Pentecostal, 06 Other Christian, 07 Islam, 08 Bahai,
09 Traditional, 10 Other, 11 No Religion.** Table 5.3 is those codes with 01-06 added together, in
code order. No code for no answer, and no table prints a religion non-response, so blanks are inside
some column, most plausibly `Other`; nothing measures it, and no dots are moved.

`No Religion` has `Traditional` beside it on the list, so it is `unaffiliated` under the 2026-09-14
no-religion rule. Table 3.25 says **87.0% of the no-religion population is under 15** (40.9% of all
household members are), so the box is mostly children recorded without a religion.

**Denominations are national only.** Table 3.25's Christian block reads Catholic 7.0, Anglican 1.2,
Methodist 3.0, `SDA 8.0` (sex ratio 210.2), an unlabelled 0.7 (sex ratio 96.6), Pentecostal 5.3,
Other Christian 4.7. Dropping the 8.0 row closes the block on 21.9, so SDA is 0.7 and 8.0 is a stray
row. The scout called the table unusable; it is usable as a national figure once that row goes, but
it has no geography, so Christianity stays one node and nothing is spread from it.

## 5. Geography

**The census has 14 districts; COD-AB v02 (`cod-ab-sle`, reviewed 30 October 2025) has 16.** Karene
(2017) came out of Bombali and Port Loko, Falaba out of Koinadugu. COD's ADM3 is still the
pre-2017 set of 167 whole chiefdoms, so `sl_geo.py` gives Karene's eight and Falaba's five back:
Buya Romende, Dibia, Sanda Magbolontor -> Port Loko; Libeisaygahun, Sanda Loko, Sanda Tendaran,
Sella Limba, Tambakha -> Bombali; Dembelia Sinkunia, Folosaba Dembelia, Mongo, Neya, Sulima ->
Koinadugu. All thirteen appear under those districts in the census's Table 3.3b.

| witness | result |
|---|---|
| chiefdoms per 2015 district = Table 3.3a-c's row count | all 12 provincial districts (Western Area's two kept at COD's 4 and 9) |
| dissolved area = COD's national polygon | 72,438 km2 both |
| IoU with geoBoundaries gbOpen SLE ADM2 (14 districts, older GoSL/OCHA release) | 0.963 (Kambia) to 1.000; no district overlaps another's namesake above 0.008 |
| Western Area Urban | 0.878 as drawn; **0.968 without COD's Tasso Island ward** (7.7 km2), which geoBoundaries leaves out |
| Koinadugu via the 16-district sheet | Falaba 205,353 + Koinadugu 204,019 = 409,372, the 2015 figure |

Tasso Island stays in Western Area Urban: the 16-district sheet prints eight Western Area Urban
wards summing to the district's 1,055,964, so its people are counted inside one of them.

**Placement.** Kontur SL 2023-11, 43,562 hexes. 533 centroids fell offshore; 497 (122,320 people)
snapped to a district within 500 m, 36 (2,794) dropped. Kontur 2023 is 1.245x the 2015 household
population, from **0.76x in Kono** (the only district Kontur has below the census) to 1.61x in Bo;
Western Area Urban 1.37x in 155 hexes. `kontur_cap.py sl` found no block at the cap. Median unit
5,400 km2, thousands of hexes each, so no grid-floor concern.

## 6. What the table shows

Every district has a Muslim majority; Kono is the lowest at 54.3%. Christianity is 43.5% of Kono,
34.0% of Kailahun, 31.3% of Western Area Urban and 27.3% of Western Area Rural, and 4.8% of
Pujehun, 5.3% of Kambia, 5.9% of Port Loko. Western Area Urban holds 21.3% of the country's
Christians, Kono 14.2%, Kailahun 11.5%. Kenema, beside Kono and Kailahun, is 12.8%. `Other` is 1.8%
of Kono and 1.4% of Kailahun and Bonthe. Traditional religion is printed at 0.2% at most (Kailahun,
Kono), a floor: one code per person.

## 7. §14 was considered and no ask was filed

Not escalated. The tier is districts of 505,000 people on average, coarser than the units asks 001,
017 and 018 cleared; the table is the office's own publication, the same bytes on its site since at
least April 2019; and nothing here locates a group more finely than Stats SL already has. No search
for attacks on religious groups in Sierra Leone was made for this build, so this rests on the grain
and the publisher, not on a safety check; a session that finds such a record should reread it.

## 8. Gotchas

- **Table 5.3's national row prints Bahai 0.5.** Its districts weight to 0.036 and Table 3.25
  prints 0.0. Pinned in `check()`; nothing drawn reads the national row.
- **Table 3.25 has a stray `SDA 8.0` row.** See §4.
- **Table 4.6's "Household population" by religion is by the head's religion.** 5,506,388 "Islam"
  is not a Muslim count; do not rescale to it.
- **A CDX `length` is the compressed capture, not the file.** The 2019 capture lists 4,052,621 and
  is the same 4,192,879 bytes as the live file.
- **Table 3.3 misspells chiefdoms** (`Sabda Tendaren`, `Bibia`, `Sando loko`, `Knike Sanda`), and its
  Bo 2004 column prints thousands (17, 54, 388) where every other district prints percentages.
- **COD spells Falaba `Fabala`** (adm2 SL0206).
- **The 16-district sheet is not a 2015-district witness except for Koinadugu.** It re-cuts the
  2015 count onto the 2017 chiefdoms, and moves Mara (17,451) from Tonkolili to Bombali and several
  split chiefdoms into Karene, so its Bombali, Port Loko, Karene and Tonkolili do not regroup to
  2015.
- **The religion question is P05 on the form and `P07` on the code list.**

## 9. Reopen when

- The 2021 mid-term census publishes religion (it asked denominations).
- Any 2015 thematic report annex turns out to print religion by chiefdom (§1's unopened list).
- A district denominational table appears; Christianity would then split.

## 10. Terms

Statistics Sierra Leone's reports are public PDFs on its own site with no licence text. OCHA COD-AB
is CC BY-IGO; geoBoundaries gbOpen SLE is CC BY 3.0 IGO (ADM2); Kontur Population is CC BY 4.0. The
IPUMS questionnaire is a public enumeration-materials page and was read, not redistributed.

## 11. Review, 2026-09-15 (`d743fc47-rev6`)

Light pass. Read the mapping, `sl.csv`, the entry and the State Department's 2023 religious
freedom report. `check_md` clean, `built_countries --check` OK, `check_rollup sl` 0 orphaned
(every row measured). One screenshot: dots over the whole country, dense on the Freetown
peninsula, Christian yellow showing in Kailahun, Kono and Freetown against green elsewhere, nothing
in the sea. No ask, nothing rebuilt.

- **`note_public` figures match** `sl.csv` or the printed national row. Its 0.1% traditional is
  the printed national figure (`T53_NATIONAL`); drawn it is 0.037%, because eleven districts print
  0.0. Fine as written.
- **Mapping: agreed.** Christianity bare, Islam bare and traditional on `indigenous.african`, as
  in `gn`, `lr`, `gw`, `bf`, `td` and `ml`. `other.sl` is the usual `other.<cc>` residual.
- **Gap: honest.** The 0.23% in institutions is hand-written because it is in no table.
- **§14, searched now, and no ask.** The State Department's *2023 Report on International
  Religious Freedom: Sierra Leone* (read via ecoi.net document 2111951) reports no physical attack
  on any religious group. Its one incident is verbal: an Ahmadi leader said Tablighi preachers in
  Waterloo (Western Area Rural), Moyamba, Kono and other communities called Ahmadis nonbelievers
  and said they should be killed in Sierra Leone. The map draws no Ahmadi geography, because 2015
  has one Islam code, and the report mentions nothing about Bahá'ís, the other small group placed.
  So §7's conclusion stands and no further search is needed.
- **Reopen under §14.4 rule 2 if a later build splits Ahmadis out.** The 2021 mid-term census
  asked denominations, and the 2004 Table 8A split survives for Bo District, Bo Town, Western Area
  Urban and Bombali (§1). Such a build should start from the incident above.
