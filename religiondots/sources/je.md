# Jersey (`je`)

Session `cb8b206e-je`, 2026-09-15, under a supervisor. Drawn: Statistics Jersey's Opinions and
Lifestyle Survey 2023 for the level, the 2018 round for the churches, one unit, every row
`modelled`. Code `sources/je.py`, mapping `taxonomy/je2023.py`, entry `countries/je.py`, node
`other.je` in `taxonomy/branches.py`. Found from `sources.md` §scout-2026-09-15-europe's row,
which had read only the 2023 report and so saw no church split.

## 1. Sources

- **No census asks.** The *Report on the 2021 Jersey Census* never mentions religion (scout);
  `tools/oracle.py Jersey`: absent from UNSD table 28. The 2011 and 2001 census reports were not
  opened.
- **Which rounds asked.** Statistics Jersey's contents workbook
  (`stats.je/wp-content/uploads/2025/05/D-JOLS-Contents-2005-2024-SJ20101222.xlsx`, TOPICS sheet,
  row `Demographics | Religion`) marks 2015, 2018 and 2023 and no other year from 2005 to 2024.
  The 2025 results table has no religion question (only discrimination by religion and
  volunteering for a church). The survey was the Jersey Annual Social Survey (JASS) to 2015 and
  the Jersey Opinions and Lifestyle Survey (JOLS) from 2016.
- **The form, every round**: "Do you regard yourself as having a religion? Please leave blank if
  you do not wish to answer", 01 Yes, 02 No, 03 Not sure; then "If yes, which?", a write-in.
  Read in the 2023 form (Q17.5-17.6, p.24) and the 2015 form (Q1.7-1.8, p.5), both from the
  Wayback Machine because gov.je answers 404 for both; the 2018 form (Q1.7-1.8, p.2) says the
  same and is not saved.
- **2023 level**: `opendata.gov.je` dataset *Jersey Opinions and Lifestyle Survey data tables*,
  `jols_2023_results.csv` (weighted proportions, all adults and by sex, age and parish type).
  Q17.5: Yes 0.39, No 0.50, Not sure 0.11. Report (`gov.je/.../Opinions and Lifestyle Survey 2023
  Report.pdf`) p.79, Figure 9.6: 39 / 49 / 11, which sums to 99; the table sums to 1.00 and is
  what is drawn. "Those who reported having a religion were invited to specify which one: the
  majority (93%) specified 'Christian' or a denomination of Christianity." No church split is
  printed anywhere in the report or the table.
- **2023 design** (report pp.4, 100-103): over 4,000 households drawn at random, stratified by
  parish with proportional allocation; the adult (16+) with the next birthday answers, by post or
  online; June and July 2023; 1,514 respondents, 37% response; weighted by age, sex and tenure to
  the 2021 census's adults in private households (84,742). ±2 points on all adults, ±4 for St
  Helier and the suburban parishes, ±5 for the rural ones. The section lookup says Q17.5 was asked
  of everyone.
- **2018** (`R Opinions and Lifestyle Survey 2018 Report 20181205 SU.pdf`) p.74: 47% have a
  religion; 96% of them named Christianity; "of those who stated a specific denomination of
  Christianity, 50% specified that they were Catholic, 39% specified Church of England, and 12%
  specified other denominations" (footnote: amended 21 November 2023 to make clear these are of
  respondents who named a denomination). Not sure is not printed in the report (Humanists UK's
  6% is not from a page read here). 1,074 respondents (Table A1).
- **2015** (`R JASS 2015 20151202 SU.pdf`, found through the Wayback CDX, live on gov.je) p.8:
  54% yes, 39% no, 7% not sure; four-fifths of the religious wrote which; 97% of those named
  Christianity, and Buddhist, Hindu, Jewish, Muslim or Sikh were "each specified by very small
  numbers of respondents"; of those naming a Christian denomination, Catholic or Roman Catholic
  43%, Anglican or Church of England 44%, other 13%. Figure 1.3 gives the churches by place of
  birth (below). Over 1,600 respondents, 52% response, about 3,200 households.
- **Population base**: Statistics Jersey, *Population and migration: Total population, December
  2024* (24 September 2025) p.3: end-2023 revised to 104,030 from a provisional 103,650. The
  fieldwork year's figure, as `sources/ad.py` chose.

## 2. What is drawn

Shares of 104,030: No religion 50.00% (52,015); Catholic 17.96% (18,679); Church of England
14.01% (14,570); Other Christian denomination 4.31% (4,483); Religion other than Christianity
2.73% (2,840); Not sure 11.00% (11,443), not drawn. Christian is 0.39 x 93%, and the three
churches are 2018's 50, 39 and 12 over their sum, 101. Mapping and the reasons in
`taxonomy/je2023.py`: Catholic to `christianity.catholic.latin`, Church of England to
`christianity.anglican`, other Christian to the `christianity` parent (Gibraltar's shape),
other religions to a new `other.je`, not sure excluded into `gap` (ESS's don't-know precedent;
`tools/gap_share.py` confirms 11.00% both ways).

## 3. The churches are from 2018, and how far that can be off

The one call here someone might reverse. The 2023 report prints no church split, so the 2018
shares are laid on 2023's Christians (ask 003 allows a mixed vintage where the method is sound).
Two assumptions: the churches moved little in five years, and people who wrote plain "Christian"
split as those who named a church did.

`sources/je.py::witness` prints three readings of the split, % of those naming a church:

| | Catholic | Anglican | other |
|---|---:|---:|---:|
| 2015 report | 43 | 44 | 13 |
| 2018 report, drawn | 49.5 | 38.6 | 11.9 |
| 2023 religion by place of birth x 2015 churches by place of birth | 44.3 | 42.4 | 13.3 |

The third row takes Q1.4's 2023 place-of-birth shares (Jersey 0.45, British Isles 0.35, Portugal
or Madeira 0.06, other European 0.06, elsewhere 0.09), 2023's share with a religion by place of
birth (Figure 9.8: 30, 38, 68, 37, 73) and 2015's churches within each (Figure 1.3: Jersey 52
Anglican, 34 Catholic, 14 other; British Isles 58/29/14; Portugal or Madeira 0/100/0; other
European 9/37/54; elsewhere 29/65/5). It rebuilds 2023's yes at 0.397 against 0.39. It
understates Catholics a little, because 2015's Poland box (100% Catholic) is not in 2023's form
and 2015's other European excludes Poles.

The readings disagree by about 5 points on Catholics. Since 2018 the share with a religion fell
among the Jersey-born (39% to 30%) and British-born (51% to 38%) and rose among the Portugal-
and Madeira-born (59% to 68%), which moves Catholics up; the birthplace reading moves them down.
Neither is decisive. The note says the church split is rough and names both rounds.

## 4. Grain: one unit

- The results table splits Q17.5 by parish type: St Helier (`4 Urban`) 0.39 / 0.48 / 0.13, St
  Brelade, St Clement and St Saviour (`3 Suburban`) 0.40 / 0.50 / 0.10, the other eight
  (`2 Rural`) 0.38 / 0.50 / 0.11. All inside the report's ±4 to ±5 points of each other, and no
  church or other-religion split exists by parish, so three units would draw sampling noise.
- Drawn as one unit under Anita's microstate ruling (2026-09-08), as Andorra (76,000) and the
  Isle of Man (84,000) are; Jersey is 104,000.
- Kontur (`kontur_population_JE_20231101`): 219 populated hexes, 111,802 people, 1.07x the base.
  `kontur_cap.py je`: no stops. Natural Earth's countries file has `Jersey` with `ISO_A2` and
  `ISO_A2_EH` `JE`, so `country_shapes.py` needs no entry.

## 5. Build

`python sources/je.py --fetch` pins every figure above against the saved reports, forms and
table. `check_mapping.py je`: 7 categories, 5 nodes, 0 unmapped; the population row and Not sure
resolve to nothing. `gap_share.py je --write`: 11.00% by both routes. 1:1,000 scatter: 90 dots on
54 hexes, no rings; 12 coastal hexes lose over 95% to the sea and stay unclipped (water.py).
1:10,000: 7 dots, 2 rings. `coverage.py` ok, which before the build tail says nothing about a
new country. `taxonomy/build_tree.py` run for `other.je`. The build tail is the supervisor's.

Traps met, none general enough for a playbook: the section lookup CSV is Windows-1252 while the
results CSV beside it is UTF-8; both forms put a checkbox glyph (U+F0A1) after each answer code
in the PDF text; the 2023 report's Figure 9.6 sums to 99 where the table sums to 1.00.

## 6. Not checked, with where to start

- **The 2023 write-ins.** Statistics Jersey coded them (93% Christian) and printed no church
  split; the office (`info@stats.je`) may hold one. The microdata are not published.
- **The 2018 not-sure share**, which the report does not print; the 2018 results table is not in
  the open data set (it starts at 2020).
- **Religion by parish** at any finer grain than parish type: not published in any round read.
- **The 2001 and 2011 census reports**: not opened for a religion item (the 2021 report has
  none).
- **JOLS 2026 onward**: religion has run every three to five years (2015, 2018, 2023); a later
  round with a church split replaces §3.

## 7. Review, 2026-09-15 (session `cb8b206e-rev6`)

Nothing to rebuild. `check_md`, `built_countries --check` and `check_rollup je` clean (92,587
modelled, nothing orphaned). The mapping matches precedent: other Christian on the parent as
`gi2022`, the residual on `other.je` as every country's, not sure excluded as `fr2024`'s ESS
don't-know. Screenshot at 1:1,000: dots on the island only, heaviest in St Helier and along the
south coast. Two readings of §3 from the report pages themselves, for whoever next touches it:

- **The plain "Christian" assumption probably leans one way.** 2015 p.8 and 2018 p.74 both give
  churches only "of those that specified a denomination", and 2015 says only four-fifths of the
  religious gave any detail at all. Anyone who wrote plain "Christian" is split as the named
  ones were. In a British Isles write-in that answer is more likely Anglican-lapsed or free
  church than Catholic, so the assumption more likely raises the drawn Catholic share than
  lowers it. The §3 witness uses 2015's churches, which exclude the same people, so it cannot
  test this. Neither report says how many there were.
- **The 5-point gap to the witness is inside sampling error.** 2018's split rests on roughly
  1,074 x 47% x 96%, under 500 Christians, fewer once those who named no church drop out. A 50%
  share on ~400 is about ±5 points at 95% before any design effect, so the witness does not
  show that 2018 leans Catholic; it shows the two agree within noise.

Neither changes a figure; the note already calls the split rough.
