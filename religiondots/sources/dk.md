# Denmark — `sources/dk.py`, `sources/dk_geo.py`, `taxonomy/dk2024.py`

Drawn 2026-09-14. **11 units, 50 nodes, 5,805,207 people, 99.40% of the country.** sources.md
§9dg is the summary. The Church of Denmark roll Danmarks Statistik publishes is printed beside the map, not
drawn (§1).

| | |
|---|---|
| counting geography | **the 11 landsdele (NUTS 3 2021)**, 530,000 people each |
| citizen composition | the 5 regioner (NUTS 2), which the landsdele nest in exactly |
| placement | 98 kommuner and Christiansø, GISCO LAU 2021, weighted by population |
| basis | self-identification (survey) for citizens; nationality-derived for foreign residents |
| tier | **`modelled` throughout** |
| vintage | ESS 2010-2019 (rounds 5, 6, 7, 9), census 2021, Pew 2020; DST's roll 2014 and 2021 printed beside |

---

## 1. The office check, which finds one church's roll

The queue's ESS block makes the office's own tables the first step (§9cu), and Sweden and Norway
add: look for commissioned work, and look for membership statistics.

**Danmarks Statistik has a roll of the Church of Denmark and nothing else.** Its Statbank API lists
5,721 tables (2026-09-14, `api.statbank.dk/v1/tables`, inactive tables included). Filtering the
titles for `folkekirke`, `trossamfund`, `religi`, `kirke`, `muslim` finds:

| table | what | geography | years |
|---|---|---|---|
| `KM1` | population by Church of Denmark membership, quarterly | parish | 2007-2026 |
| `KM5` | the same by sex and age, 1 January | parish | 2007-2026 |
| **`KM6`** | **the same, 1 January** | **98 kommuner** | 2011-2026 |
| `KM2`, `KM22` | joining and leaving | parish, deanery | 2007-2026 |
| `KM4`, `KM44` | church ceremonies | parish, deanery | 2006-2025 |
| `KV2FR5` | visits to places of worship (culture survey) | national | 2024-2025 |

DST's information page on religion (`dst.dk/da/informationsservice/oss/religion`) says it holds no
figures on other faith communities and sends readers to Aarhus University's Center for
Samtidsreligion. **No public shelf of commissioned tables was found.** Two web searches for DST's
commissioned or special-run tables on religion returned nothing of the kind; one surfaced DST's
terms for service tasks
(`dst.dk/da/TilSalg/skraeddersyede-loesninger/forretningsvilkaar-og-betingelser`), which by the
search summary deliver tables to the customer for publication only by agreement. That page was
not opened, so this is the weakest line in the section. Sweden's lesson (§9cz) points at the
customer's website, and the obvious customer, the Church of Denmark, is already served by KM6.

**KM6 is not drawn**, for Norway's first two reasons (sources/no.md §1). It is a `roll` (spec §3.1)
and Finland, Sweden and Norway are all self-identification; on it Denmark would draw 73.8%
Lutheran in 2021 beside Sweden's 21.5% answered. And a roll cannot say who belongs to no religion.
It is printed beside the survey in §7.

## 2. Rounds and variable

Probed on 2026-09-14 against every integrated file from round 5 to 11:

| round | fieldwork | Denmark | `region` | `rlgdndk` |
|---|---|---|---|---|
| 5 | 2010-11 | 1,576 | NUTS 2, 5 codes | **exists** |
| 6 | 2012-13 | 1,650 | NUTS 2, 5 codes | E201VariableNotFound |
| 7 | 2014-15 | 1,502 | NUTS 2, 5 codes | E201VariableNotFound |
| 8 | 2016-17 | **absent** | | |
| 9 | 2018-19 | 1,572 | NUTS 2, 5 codes, **in a different order (§3)** | E201VariableNotFound |
| 10, 11 | 2020-23 | **absent** | | |

`rlgdnadk` and `rlgdnbdk` do not exist in any round. Rounds 1-4 have no `region` (Belgium's wall)
and were not probed further; they would push the vintage back to 2002.

**`rlgdndk` is the harmonised card in Danish, code for code**, which is Belgium's case and not
Norway's: `Romersk-katolsk`, `Protestantisk`, `Ortodoks`, `Andre kristne religioner`, `Jødisk`,
`Islam`, `Østlige religioner`, `Andre ikke kristne religioner`. `_check_card` proves the one-to-one
nesting from round 5's cross-tab. So the four rounds pool on `rlgdnm`, and **the Church of Denmark
and the free churches are one answer**, `Protestant`, where Norway's card split them.

## 3. Round 9's regions, which are in Danmarks Statistik's order

**The trap worth carrying out of Denmark.** Round 9 returns the same five codes as rounds 5-7, with
the same labels, and nothing errors. But the respondents behind the codes are in the wrong regions.
The first sign was the sample shares: Hovedstaden holds 32% of Denmark and was 25-28% of the sample
in rounds 5-7, and in round 9 `DK01 Hovedstaden` is 11.5% while `DK02 Sjælland` is 23.5%.

Every profile the survey carries says the same thing (all respondents, unweighted):

| round 9 as published | n | big city | suburbs | Muslims | Protestant | the vote | is really |
|---|---:|---:|---:|---:|---:|---|---|
| DK01 Hovedstaden | 181 | 18.8% | 14.4% | 3 | 60.2% | Social Democrats 37.1% | **Nordjylland** |
| DK02 Sjælland | 369 | 11.7% | 22.5% | 4 | 59.9% | Alternativet 5.9% | **Midtjylland** |
| DK03 Syddanmark | 378 | 18.8% | 13.0% | 8 | 58.5% | Dansk Folkeparti **22.0%** | **Syddanmark** |
| DK04 Midtjylland | 436 | **34.9%** | **37.4%** | **16** | **38.8%** | Enhedslisten 13.3% | **Hovedstaden** |
| DK05 Nordjylland | 208 | 3.8% | 5.8% | 0 | 52.9% | | **Sjælland** |

In rounds 5-7 Hovedstaden is 27-37% big city and 35-45% suburbs, Sjælland 1-7% big city, and
Nordjylland the most Protestant at 64-70%. Three of the five are unambiguous. The two Jutland
regions are separated by the recalled 2015 vote (`prtvtddk`): Dansk Folkeparti is highest in the
published DK03 (22.0%, against 14.9% in DK02), Alternativet is 5.9% in DK02 against 1.4% in DK03
(and 6.6% in the published DK04, Hovedstaden), and German spoken at home turns up in DK03 in
round 9 as it does in Syddanmark in round 7.

**The permutation is not random.** Danmarks Statistik numbers the regions 1081 Nordjylland, 1082
Midtjylland, 1083 Syddanmark, 1084 Hovedstaden, 1085 Sjælland. Round 9's DK01-DK05 are those five
in that order. The recode is `RECODE` in `dk.py`.

**ESS's own weight was fitted to the wrong labels.** `pspwght` for round 9 puts the published DK01
at 31.3% of Denmark, which is Hovedstaden's population share, carried by 181 people who live in
Nordjylland. So post-stratification raked the round to the wrong regional targets. Round 9 is
weighted by `dweight` instead, which for Denmark's simple random sample from CPR is equal weights.
It barely matters nationally (composition with round 9 on each):

| | dweight (drawn) | pspwght |
|---|---:|---:|
| Protestant | 52.04% | 52.19% |
| No religion | 43.78% | 43.25% |
| Islam | 1.62% | 1.79% |
| Roman Catholic | 0.75% | 0.98% |

`_check_recode` runs on every build: after recoding, Hovedstaden must have the highest big-city and
suburb share and Sjælland the lowest big-city share in all four rounds; **round 9 as published must
fail that test**, so the recode stops the build if ESS ever fixes the file; and round 9's Dansk
Folkeparti vote must be higher in Syddanmark than in Midtjylland, with Enhedslisten highest in
Hovedstaden.

No ESS release note mentions it. ESS9 edition 3.0 is the release that added Denmark and Iceland,
and its announcement lists no corrections.

## 4. The level, and why nothing is rescaled

Weighted, citizens, by round:

| | r5 2010-11 | r6 2012-13 | r7 2014-15 | r9 2018-19 |
|---|---:|---:|---:|---:|
| Protestant | 53.22% | 49.81% | 50.77% | 54.40% |
| No religion | 42.60% | 46.20% | 44.76% | 41.51% |
| Islam | 1.79% | 1.77% | 1.47% | 1.45% |

**The survey's Protestant share is flat across the pool**, where Norway's fell twelve points and
forced spec §3.4. So Denmark is not rescaled, and the map's citizen half is a 2010-2019 average,
which `how` says. The roll fell over the same years (78.44% in 2014, 73.82% in 2021), and the survey
does not show that as a fall in stated belonging.

## 5. Which categories carry their own geography, and Islam

The test is Norway's `_stability`, imported: median Spearman over every round split, a 2,000-draw
per-round permutation of the unit labels (seed 0), and Sweden's spatial chi-square, both at 0.05.
Four rounds give three splits of two against two. **6,041 answered citizens** over the 5 regions
(706 in Nordjylland to 1,591 in Hovedstaden); 0.49% declined (weighted).

| category | n | median rho | null 95th | p | chi² p | drawn at |
|---|---:|---:|---:|---:|---:|---|
| **Protestant** | 3,236 | +0.900 | +0.700 | 0.0200 | 6.0e-38 | **5 regions** |
| No religion | 2,567 | +0.900 | +0.700 | 0.0145 | 5.0e-30 | residual (passes; §12 nested units) |
| Islam | 85 | +0.300 | +0.700 | 0.3248 | 4.1e-05 | residual |
| Other Christian denomination | 53 | +0.400 | +0.700 | 0.2094 | 1.3e-01 | residual |
| Roman Catholic | 47 | −0.600 | +0.700 | 0.9200 | 3.5e-01 | residual |
| Other Non-Christian religions | 22 | +0.500 | +0.783 | 0.2079 | 1.1e-03 | residual |
| Eastern religions | 19 | +0.100 | +0.700 | 0.4583 | 3.7e-01 | residual |
| Eastern Orthodox | 6 | +0.363 | +0.725 | 0.2424 | 5.2e-01 | residual |
| Jewish | 6 | +0.125 | +0.875 | 0.5192 | 1.9e-01 | residual |

`No religion` stays the residual; fixed at its region's share the tail would not go negative
anywhere (worst +2.82%).

### Islam, which the chi-square wants and the rank test cannot see

The regions differ on Islam at 4.1e-05, and the rank test finds nothing. **Both are right.** Among
citizens the survey has Hovedstaden at 2.98% and the other four at 0.80-1.35% in no stable order,
so the difference is one unit standing apart, and a rank correlation over five units is decided by
how the four small ones shuffle between splits. `Other Non-Christian religions` has the same shape
(Hovedstaden 0.90%, chi-square 1.1e-03).

**Not overridden.** Norway's override (sources/no.md §6) stood on a rank p of 0.0555 and an
independent roll that ordered all seven regions the same way at +0.929. Denmark has a rank p of 0.32
and no roll of Muslims. And the residual construction already carries part of the contrast, because
Islam shares out each region's non-Protestant remainder:

| citizens | Hovedstaden | Sjælland | Syddanmark | Midtjylland | Nordjylland |
|---|---:|---:|---:|---:|---:|
| Islam as drawn | 2.07% | 1.64% | 1.45% | 1.44% | 1.14% |
| Islam, survey | 2.98% | 1.03% | 1.35% | 0.80% | 1.06% |

With the foreign half added, Københavns omegn draws 6.24% Muslim and Byen København 5.04%, against
2.78% for Nordjylland. What an override could still add is the named follow-up: a witness at the
5 regions, such as Danish citizens of Muslim-majority origin from DST's origin tables (FOLK1C
against FOLK1B), which is a new source and was not fetched.

## 6. The mapping, two calls

**`Protestant` goes to `christianity.lutheran`.** The Danish card names no church, but the Church of
Denmark had 78.44% of the country on its roll in 2014, and the largest Protestant free churches on
Center for Samtidsreligion's list of approved communities (members at 1 January 2009, read from
`samtidsreligion.au.dk/religion-i-danmark/rel-aarbog09/statistik/alle`) are the Baptists 5,260, the
Pentecostals 5,158, the Apostolic Church 3,000, the Adventists 2,537, the Mission Covenant 2,200 and
the Methodists 2,006. About 20,000 together, under 0.4% of Denmark.

**The Orthodox answer is not split.** The same list (§9cz's check):

| community, 1 January 2009 | members | |
|---|---:|---|
| Serbian Orthodox Church in Denmark | 7,000 | Eastern |
| Russian Orthodox congregation, Copenhagen | 1,000 | Eastern |
| Russian Orthodox congregation, Hobro | 100 | Eastern |
| Romanian Orthodox congregation | 500 | Eastern |
| Macedonian Orthodox Church | 500 | Eastern |
| Coptic Orthodox Church | 250 | Oriental |
| Armenian Apostolic Church | none printed | Oriental |
| Assyrian Church of the East | 270 | Church of the East |

**9,100 of 9,620 is Eastern, 94.6%**, so the whole cell goes to `christianity.orthodox`, which is
Austria's and the UK's call. The list is old and self-reported and leaves out unapproved
congregations (the Ethiopian church was approved in 2011), which is the direction the error runs;
the cell is 6 citizens in four rounds.

## 7. What the finished country looks like, and the roll beside it

| landsdel | Lutheran | unaffiliated | Islam | Catholic | Orthodox | foreign citizens |
|---|---:|---:|---:|---:|---:|---:|
| Byen København | 32.64% | 50.06% | 5.04% | 4.64% | 1.61% | 15.78% |
| Københavns omegn | 33.68% | 50.22% | 6.24% | 3.24% | 1.69% | 13.12% |
| Nordsjælland | 35.76% | 52.65% | 4.17% | 2.58% | 1.22% | 7.77% |
| Bornholm | 36.40% | 53.42% | 3.26% | 2.10% | 0.94% | 6.14% |
| Østsjælland | 47.90% | 41.80% | 4.15% | 2.32% | 0.93% | 7.33% |
| Vest- og Sydsjælland | 48.50% | 42.02% | 3.67% | 2.14% | 0.98% | 6.19% |
| Fyn | 53.41% | 37.25% | 3.43% | 2.18% | 1.04% | 6.80% |
| Sydjylland | 52.22% | 36.76% | 3.65% | 2.75% | 1.71% | 8.87% |
| Vestjylland | 52.94% | 36.80% | 3.23% | 2.39% | 1.98% | 7.70% |
| Østjylland | 52.98% | 37.00% | 3.53% | 2.43% | 1.29% | 7.62% |
| Nordjylland | 61.85% | 29.63% | 2.78% | 2.02% | 1.35% | 6.63% |

Nationally: Lutheran 47.06%, unaffiliated 41.34%, Sunni Islam 3.57%, Latin Catholic 2.70%.

**The roll and the survey agree about the order, which is the opposite of Sweden.** KM6 at 1
January 2014, summed from kommuner to regions, against the survey's Protestant share among citizens:

| region | roll 2014 | survey, Protestant | survey, no religion |
|---|---:|---:|---:|
| Nordjylland | 86.1% | 66.26% | 29.65% |
| Syddanmark | 83.7% | 57.34% | 39.37% |
| Midtjylland | 83.2% | 57.38% | 39.80% |
| Sjælland | 82.1% | 51.72% | 44.72% |
| Hovedstaden | 67.1% | 38.80% | 54.97% |

**Spearman +0.900**, the only discordant pair being Syddanmark and Midtjylland, 0.04 points apart
on the survey. Sweden's register and survey ordered the län at +0.247 (sources/se.md §4); Norway's
at +0.718 over 11 counties. The gap between the columns is 25-30 points in every region.

**And the roll shows what five regions hide.** At the 11 landsdele (2014, 2021):

| landsdel | 2014 | 2021 |
|---|---:|---:|
| Vestjylland | 87.4% | 83.4% |
| Nordjylland | 86.1% | 82.6% |
| Sydjylland | 84.4% | 80.1% |
| **Bornholm** | **83.5%** | 78.3% |
| Vest- og Sydsjælland | 83.4% | 79.4% |
| Fyn | 82.6% | 78.7% |
| Østjylland | 81.1% | 77.0% |
| Østsjælland | 79.0% | 74.3% |
| Nordsjælland | 77.3% | 72.6% |
| Københavns omegn | 67.5% | 61.1% |
| Byen København | 59.7% | 54.9% |

Hovedstaden runs from Byen København, last, to Bornholm, fourth. **Bornholm is drawn with the
capital region's citizen composition** (36.40% Lutheran), because the survey places nobody finer
than the region. Using the roll to spread the survey's Protestant share inside regions was
considered and not done: the part of the roll that does not answer Protestant is not published
per landsdel, and Sweden shows the two geographies can part. `note_public` names Bornholm.

The join is by kommune code and checked without names: every KM6 code must be one of the 99 LAUs
in `dk_lau.gpkg`, and each landsdel's KM6 population for 2021 must be within 2% of the census's.
Both pass; KM6's national 2021 total is 5,840,045 against the census's 5,840,046.

## 8. What the build cannot do

- **Five regions for the citizen half.** Landsdele inside one region draw the same citizen
  composition and differ only through their foreign residents. Bornholm is the visible case.
- **Everything except the Protestant answer is at the national rate inside each region's
  remainder**, so the map says nothing about where Muslim, Catholic or Orthodox citizens live
  beyond how secular their region is.
- **The free churches are inside `Protestant`**, and the Inner Mission and Grundtvigian movements
  inside the church are invisible, as the EFS and the Norwegian lay missions are.
- **The Shia and the Alevis are not separable among citizens**; their dots come from the foreign
  half only (19,041 Shia and 5,571 Alevi there).
- **The newest interview is from 2019.** Denmark has not been in ESS since.
- **Placement is population.** Muslim dots spread over Københavns omegn by where anyone lives, not
  over Ishøj, Albertslund and Brøndby Strand. DST publishes population by citizenship per kommune,
  which would give Denmark the Italy weighter for the foreign half; not built, for Belgium's reason
  (sharpening one half only).

## 9. Numbers to check a rebuild against

```
rounds 5, 6, 7, 9; Denmark absent from 8, 10, 11; region NUTS 2 in all four
round 9 region order = DST 1081-1085: DK01->DK05, DK02->DK04, DK03->DK03, DK04->DK01, DK05->DK02
rlgdndk round 5 only, = rlgdnm one to one
6,041 answered citizens (706 to 1,591 per region); 0.49% declined (weighted)
5 regions: Protestant and No religion pass; Islam p 0.3248, chi-square 4.1e-05, not overridden
5,840,046 in cens_21ctz_r3 = 5,300,552 NAT + 530,887 FOR + 8,556 STLS + 51 UNK
drawn 5,805,207, 99.40%; citizen half 5,274,320; foreign half 530,887
lutheran 47.06%  unaffiliated 41.34%  islam.sunni 3.57%  catholic.latin 2.70%
KM6 roll: 78.44% (2014), 73.82% (2021); Spearman with the survey at 5 regions +0.900
GISCO LAU 2021: 99 LAUs, 5,840,045 people
```

## 10. Review, 2026-09-14 (f95259a4-dkrev)

**The round 9 recode holds, tested from the raw files rather than from §3.** Every labelling of
round 9's five codes (120 of them) was scored as a multinomial log-likelihood against a
reference. Figures are the difference from the best labelling.

| evidence | RECODE | Jutland pair swapped | as published |
|---|---:|---:|---:|
| domicile, religion, citizenship and sample share, against rounds 5-7 pooled | rank 1, 0.0 | rank 2, −5.4 | rank 33, −457.5 |
| recalled 2015 vote, against DST's actual 2015 result by region | rank 1, 0.0 | rank 13, −10.9 | rank 31, −51.3 |

Hovedstaden and Sjælland are settled by domicile alone. The only close rival swaps Midtjylland and
Syddanmark, and the two tests together put it about 16 log units behind. The vote side is DST
`FVKOM` for 2015 over all 98 kommuner, joined to regions through `dk_lau.gpkg`, and models each
party's regional departure from its national share, so the survey's over-recall of winners
cancels. Kristendemokraterne is a witness `_check_recode` does not use: DST has it at 1.83 times
its national share in Midtjylland and 0.97 in Syddanmark, and round 9 at 1.70 in published DK02 and
1.16 in DK03. Religion and sample share alone lean slightly to the swap (−2.7, −1.1); domicile and
the vote outweigh them.

Rounds 5, 6 and 7, each scored against the other two, pick their own published labels first
(runner-up −4.3, −10.9, −15.0), so the reference rounds are not scrambled themselves.
`_check_recode` only tests Hovedstaden and Sjælland in those rounds, so this was worth doing once.

`pspwght` confirmed from the files: round 9's published DK01 is 11.5% of respondents and 31.3%
weighted. `dweight` equals the unweighted count in every round. The scripts were throwaways and
are not in the tree.

**Islam not overridden is consistent with Norway.** Norway's override stood on a near miss (p
0.0555) and an independent roll ordering the regions; Denmark has p 0.32 and nothing independent on
disk. Overriding would move roughly 15,000 citizens' dots into Hovedstaden (2.07% to 2.98% of its
citizens) from the other four. Left as it is.
