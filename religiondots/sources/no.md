# Norway — `sources/no.py`, `sources/no_geo.py`, `taxonomy/no2024.py`

Drawn 2026-09-14. **11 units, 50 nodes, 5,372,689 people, 99.65% of the country.** sources.md
§9dd is the summary. The roll SSB publishes is printed beside the map rather than drawn (§1).

| | |
|---|---|
| counting geography | **the 11 counties of 2020-2023 (NUTS 3 2021)**, 490,000 people each |
| placement | 356 kommuner, GISCO LAU 2021, weighted by kommune population |
| basis | self-identification (survey) for citizens; nationality-derived for foreign residents |
| tier | **`modelled` throughout** |
| vintage | ESS 2010-2018 (7 regions) and 2010-2023 (4 shared units), census 2021, Pew 2020, grant list 2018; SSB's rolls 2020 printed beside |

---

## 1. The office check, which finds a roll and not a question

`queue.md`'s ESS block makes the office's custom-table shelf the first step (§9cu), and the
Swedish and Belgian records add two more: look for work the office did for someone else, and
look for the membership statistics a grant system produces. For Norway the brief named both:
SSB *oppdrag*, and SSB's own statistics on religious communities.

**SSB has no public shelf of commissioned tables.** Its site search for `religion` on 2026-09-14
returned 1,067 items; the first fifteen are the statistic pages for faith communities and for
the living-conditions survey, three statbank tables (06340, 06326, 08531), the 2011 needs
analysis (*Notater 2011/07*) and five articles, and none is a commissioned religion table or a
regional self-identification figure. The needs analysis itself names two sources of religion
statistics, registers and interview surveys, and describes SSB's work with the Church of Norway
without naming any tabulation for anyone.

**What SSB does publish is membership, and it is richer than §11k said.** The state pays every
faith and life-stance community a grant per member, the county governors check the member lists
against the population register, and SSB publishes the result:

| table | what | geography | years |
|---|---|---|---|
| `12025` | Church of Norway members and affiliated, % of inhabitants; members of other communities, % | **356 kommuner** and the counties | 2015-2025 |
| `08531` | members of every other grant-receiving community, by five groups: Buddhism, Islam, Christianity, other religion, life stance | **the counties** | 2010-2020 |
| `06340` | the same, older county vintages | counties | 2006-2020 |
| `06326` | ten groups (Baha'i, Buddhism, Hinduism, Islam, Judaism, Christianity, Sikhism, life stance, other) | national | 2006-2026 |

§11k's line was *"Table 06326 has ten religions, national only"* and that is true; what it
missed is `08531`, which has all of them by county in five groups. **The county series ends in
2020**, and SSB's statistic page says why: *"Frå 2021 vart det ikkje lengre bustadkommune knytt
til organisasjonane"* (from 2021 the communities no longer report members' home municipality).

**It is not drawn, for three reasons in order of weight.**

1. **It is a roll (spec §3.1), and both neighbours are self-identification.** Finland (§9by) and
   Sweden (§9cz) are drawn from ESS with their registers printed beside. Norway on its roll
   would draw 67.7% Church of Norway (2020) beside Sweden's 21.5% answered, a step at the border
   that is a step between two questions. §3.5a re-based the United States for exactly that.
2. **A roll cannot say who belongs to nothing.** Its residual is "on no roll", 19.4% in 2020,
   and that is not the unaffiliated: SSB's own living-conditions survey found **47%** of adults
   saying they belong to a religion that year, when the roll had 80.6% on one.
3. **Its categories are five groups at county level**, and `Christianity` outside the Church of
   Norway puts the Catholics, the Pentecostals and the Eritrean Orthodox in one cell.

**Two other things turned up and neither is a source.** SSB's article *Hvor kristent er Norge?*
(27 August 2024) ranks the counties and kommuner by how Christian they are, which looked like a
regional self-identification figure and is **church attendance per inhabitant** from the Church
of Norway's own returns (Agder 0.93 visits a year, Oslo 0.33). And the living-conditions
survey's religion item is published nationally only.

## 2. The variable, which is the harmonised card with the right answer split

`rlgdnno` exists in all seven rounds with a region, 5 to 11. `rlgdnano` and `rlgdnbno` raise
`E201VariableNotFound`. Belgium's lesson is to read both cards before choosing:

| `rlgdnno` | inside `rlgdnm` |
|---|---|
| Den norske kirke | Protestant |
| Andre protestantiske trossamfunn (f.eks. frikirker, anglikanske kirke, pinsevenner og andre) | Protestant |
| Katolsk kirke | Roman Catholic |
| Ortodoks kirke (gresk, russisk, andre) | Eastern Orthodox |
| Andre kristne trossamfunn (f.eks. Jehovas vitner, mormonerne) | Other Christian denomination |
| Det mosaiske trossamfunn (jødisk) | Jewish |
| Islam (muslimsk) | Islam |
| Østlige religioner (f.eks. buddhisme, hinduisme, sikh, shintoisme, taoisme, konfutsianisme) | Eastern religions |
| Andre ikke-kristne religioner | Other Non-Christian religions |

So `rlgdnno` is `rlgdnm` with `Protestant` split into the state church and everyone else, which
is the one split Norway needs, and it costs no rounds. `_check_card` proves the nesting from a
fetched `rlgdnno x rlgdnm` cross-tab in every round and stops the build on any off-diagonal
respondent. Like Sweden's `rlgdnase`, the labels arrive in Norwegian with
`metadataLanguage:"en"`.

## 3. The level, which is NUTS 2 twice over in two shapes

Norway is NUTS 2 in every round, which `regunit` confirms. The trap is the vintage:

| rounds | NUTS | regions |
|---|---|---|
| 5-9 (2010-2018) | 2016 | NO01 Oslo og Akershus, NO02 Innlandet, NO03 Sør-Østlandet, NO04 Agder og Rogaland, NO05 Vestlandet, NO06 Trøndelag, NO07 Nord-Norge |
| 10-11 (2020-2023) | 2021 | NO02, NO06, NO07, NO08 Oslo og Viken, NO09 Agder og Sør-Østlandet, NO0A Vestlandet |

**These do not nest in either direction.** NO08 takes Akershus out of NO01 and Østfold and
Buskerud out of NO03; NO09 takes Agder out of NO04; NO0A takes Rogaland out of NO04. Finland's
three vintages recoded cleanly because its maakunnat were renamed and merged; Norway's regions
were cut differently. The only geography both vintages are unions of is **four units**:
Innlandet, Trøndelag, Nord-Norge and everything else. (Jevnaker and Lunner, 15,897 people, moved
from Oppland to Viken in 2020, so the shared Innlandet is 4.3% larger in the older vintage.
Named rather than modelled.)

So Sweden's construction (spec §12, nested units) runs over three levels:

| level | pool |
|---|---|
| the 7 NUTS 2016 regions | rounds 5-9 |
| the 4 shared units | all seven rounds |
| the national rate inside each county's residual | rounds 5-9 |

and a category takes the finest level at which it passes both of Sweden's tests.

**The counting unit is the 11 counties of 2020-2023**, where the census counts citizenship.
Ten sit inside one NUTS 2016 region. **Viken does not**: it is Akershus (NO01), Østfold and
Buskerud (NO03) and two kommuner from Oppland (NO02), so its citizen composition is those
regions' shares blended by their 1 January 2019 populations (SSB 07459): 624,055 / 587,353 /
15,897. Asker is in the blend correctly, because Røyken and Hurum were Buskerud kommuner and are
inside Buskerud's figure.

## 4. The Orthodox answer, split two ways, and why the Swedish argument runs backwards here

sources.md §9cz: before sending an undifferentiated Orthodox answer to `christianity.orthodox`,
check the country. Norway's check is the ministry's list of grant-counted members per community,
`antall tilskuddstellende medlemmer i tros- og livssynssamfunn`. **regjeringen.no answers 403**
to a script and to WebFetch alike; the Wayback Machine's copies of the 2018 and 2024 lists are
whole (%%EOF present) and were read from there. The 2018 list is used because it is the last year
of rounds 5-9.

| 2018 | members | share |
|---|---:|---:|
| Eastern Orthodox | 15,279 | 59.49% |
| Oriental Orthodox | 10,403 | 40.51% |

Eastern is the Serbian parish (4,507), four Russian parishes (5,726 between them), the Greek
parish (1,596), the Norwegian-language parishes, Romanian and Bulgarian. Oriental is about forty
Eritrean congregations from Kristiansand to Nordreisa, the Ethiopian churches, the Armenian
community (198) and the Copts. No Syriac church and no Church of the East, which is where Norway
differs from Sweden.

**The borrowed ratio is biased, and in the opposite direction from Sweden's.** Sweden's Syriac
population is old and naturalised, so the resident ratio understated the Oriental share of the
citizen cell there. Norway's Eritreans arrived mostly after 2008 and largely after 2014, and
naturalisation takes seven years, so in 2010-2018 many were not yet citizens; they are drawn from
the census half instead, where Eritrea's Christians go to `christianity.oriental`. **40.51%
overstates the Oriental share of the citizen cell.** Left unadjusted, as Sweden's was, because
nothing publishes the adjustment; it moves a few thousand people between two sibling nodes and
nobody between counties.

Two smaller things. Three rows are assigned by naming convention rather than a stated church
(`ENGELEN SANKT GABRIEL ORTODOKSE KIRKE` 57, `ST. GEORGS ORTODOKSE, KRISTNE KIRKE` 198,
`ST.MICHEAL` 52; 1.2% of the total). Four `apostolisk` rows are Pentecostal and are excluded. And
`Hellige Irina menighet` (384) names no church at all and is the Russian Orthodox parish in
Bryne, per the parish's own site.

The card's own examples are *gresk, russisk*, so some Eritrean respondents will have answered
`Andre kristne trossamfunn` instead. That moves people out of the cell, not between its halves.

## 5. The geography, and one check that looked like a failed join

GISCO LAU 2021 has all 356 kommuner. **The correspondence workbook beside it does not have
Norway**: `EU-27-LAU-2021-NUTS-2021.xlsx` is the 27 member states, so the Swedish route of reading
the NUTS 3 code off the workbook does not exist. The county is read off the kommune number, whose
first two digits are the county number.

**The first version of the join check failed six counties, and the join was right.** It compared
each county's kommune populations with the 2021 census at 0.5% tolerance. GISCO's `POP_2021` for
Norway is **the 1 January 2020 population** (5,367,580, SSB's figure for that date to the person),
and every county was off in the direction one year of growth gives: Oslo and Viken low, Nordland
and Innlandet high. A mis-assigned kommune moves two counties by the same amount in opposite
directions instead. The check is now exact equality against SSB 07459 at 1 January 2020, all
eleven counties, and passes.

The census's stateless people (1,700) are a separate code from `FOR` in Norway's
`cens_21ctz_r3`, so they are in neither half, with the 30 of unknown citizenship. Svalbard is not
a kommune and the census counts nobody there.

## 6. Which categories carry their own geography, and the one drawn against the test

Sweden's test unchanged: median Spearman over every round split, a 2,000-draw permutation of
the unit labels per round (seed 0), and a spatial chi-square, both at 0.05. The statistic is
imported from `sources/be.py`, not copied. Unweighted counts.

**At the 7 NUTS 2016 regions, rounds 5-9, 10 splits of 2 against 3, 7,045 answered citizens**
(503 in Innlandet to 1,558 in Oslo og Akershus):

| category | n | median rho | null 95th | p | chi² p | drawn at |
|---|---:|---:|---:|---:|---:|---|
| No religion | 3,432 | +0.750 | +0.485 | 0.0025 | 1.7e-27 | residual (passes; §12 nested units) |
| **Den norske kirke** | 3,151 | +0.714 | +0.500 | 0.0060 | 1.1e-36 | **7 regions** |
| **Andre protestantiske trossamfunn** | 180 | +0.804 | +0.536 | 0.0015 | 1.8e-13 | **7 regions** |
| **Islam** | 91 | +0.529 | +0.536 | 0.0555 | 1.3e-10 | **7 regions, OVERRIDE** |
| Katolsk kirke | 54 | +0.387 | +0.514 | 0.1154 | 2.6e-03 | residual |
| Østlige religioner | 39 | +0.234 | +0.503 | 0.2424 | 2.5e-01 | residual |
| Andre kristne trossamfunn | 40 | +0.243 | +0.522 | 0.2424 | 1.6e-01 | residual |
| Andre ikke-kristne religioner | 36 | −0.691 | +0.500 | 0.9885 | 9.6e-01 | residual |
| Ortodoks kirke | 20 | −0.076 | +0.524 | 0.6057 | 5.2e-01 | residual |
| Det mosaiske trossamfunn | 2 | −0.167 | +1.000 | 1.0000 | 7.2e-01 | residual |

**At the 4 shared units, all seven rounds, 35 splits of 3 against 4, 9,611 citizens**, only No
religion passes (p = 0.0450), and the Church of Norway misses at p = 0.0555 with a median rho of
+0.800 against a null 95th of +0.800. Four units give a rank correlation a handful of possible
values, so this level cannot say much about anything; it is reported because §12 says to run it.

`No religion` passes at the 7 regions and stays the residual, per spec §12 (nested units). Fixed
at its regional share it would not have gone negative anywhere (worst +2.08%, Innlandet), so
here the rule costs nothing either way; as the residual it still follows each county's measured
Lutheran, free-church and Muslim shares.

### The Islam override

**The rank test misses by 0.0055 and the chi-square says the regions differ at 1.3e-10.** Drawn
at the national rate, the first build put Oslo at 4.43% Muslim with both halves together, against
**9.64%** on SSB's 2020 roll. So the question was whether the survey's ordering of the 7 regions
is real, and that is something an independent instrument can answer. SSB's 2019 county roll
(`08531` by the old counties, `07459` for population), summed to the same 7 NUTS 2016 regions:

| region | ESS Islam, citizens | respondents | roll Islam, residents |
|---|---:|---:|---:|
| NO01 Oslo og Akershus | 3.48% | 47 | 6.80% |
| NO03 Sør-Østlandet | 2.06% | 22 | 3.71% |
| NO02 Innlandet | 1.13% | 5 | 1.56% |
| NO04 Agder og Rogaland | 0.88% | 7 | 2.35% |
| NO05 Vestlandet | 0.42% | 5 | 1.32% |
| NO06 Trøndelag | 0.32% | 2 | 1.51% |
| NO07 Nord-Norge | 0.29% | 3 | 1.19% |

**Spearman +0.929, p = 0.003.** The roll's levels are higher because it counts residents and the
children of members; the order is the survey's order with Innlandet and Agder og Rogaland
swapped. So Islam is drawn at the 7 regions, in `OVERRIDE` with that reason printed on every
build, and `EXPECT_FINE_PASS` asserts the verdict set so a re-fetch cannot quietly change it.

The same comparison for the other categories, for the record:

| pair, 7 regions | Spearman | |
|---|---:|---|
| ESS free churches vs roll Christianity outside the church | +0.857 (p 0.014) | Agder og Rogaland top on both |
| ESS Church of Norway vs roll membership | +0.571 (p 0.180) | Trøndelag is 39.9% on the survey and 75.9% on the roll |
| ESS No religion vs roll "on no roll" | +0.286 (p 0.535) | not the same quantity at all |

The Church of Norway row is Sweden's finding again (§9cz, +0.247 there): membership and stated
belonging are two geographies of the same church. Trøndelag is the Norwegian case of it, high on
the rolls and one of the least Lutheran regions by what people say.

## 7. The level, which moved twelve points inside the pool

Rounds 10-11 cannot be placed at the 7 regions, so they only enter the 4-unit test. Nationally,
weighted, they say something the 7-region pool does not:

| | rounds 5-9 (2010-2018) | rounds 10-11 (2020-2023) | |
|---|---:|---:|---:|
| No religion | 48.74% | 60.01% | +11.27 |
| Den norske kirke | 44.14% | 32.24% | **−11.90** |
| Andre protestantiske trossamfunn | 2.60% | 3.08% | +0.48 |
| Islam | 1.55% | 1.99% | +0.44 |
| everything else | under 0.4 points each | | |

Sweden's equivalent drift was under 3.5 points and was left in. Twelve points on the largest
religious category, on a map whose other half is the 2021 census, is not a vintage footnote. **So
spec §3.4 is applied**: the fine categories keep rounds 5-9's regional pattern and are scaled to
their rounds 10-11 national share, weighted by the census's citizen population per county
(not by the survey's own unit mix, which is the Nigeria entry in spec §12).

**Factor or shift is a claim about the shape of the decline, and two witnesses agree it is a
factor.**

| roll, Church of Norway % | 2020 | 2025 | ratio | points |
|---|---:|---:|---:|---:|
| Norway | 67.7 | 60.9 | 0.900 | −6.8 |
| Oslo | 46.4 | 40.7 | 0.877 | −5.7 |
| Rogaland | 68.6 | 61.9 | 0.902 | −6.7 |
| Møre og Romsdal | 77.8 | 72.3 | 0.929 | −5.5 |
| Nordland | 79.1 | 72.6 | 0.918 | −6.5 |
| Innlandet | 78.3 | 72.1 | 0.921 | −6.2 |
| Agder | 66.1 | 60.7 | 0.918 | −5.4 |
| Vestland | 73.4 | 67.4 | 0.918 | −6.0 |
| Trøndelag | 75.1 | 68.9 | 0.917 | −6.2 |

Coefficient of variation over the eight counties whose codes survived 2024: **0.018 for the
ratio, 0.077 for the point drop.** And the survey's own four shared units, rounds 5-9 against
10-11: ratios 0.728, 0.752, 0.707, 0.691; drops −14.1, −9.9, −15.6, −13.2.

The residual categories are not scaled. They share what is left of each county at the
all-seven-round national proportions, because rounds 10-11 alone put 7 respondents on Eastern
religions and 4 on Orthodox.

The scaling factors: Church of Norway x0.7317 (44.06% to 32.24%), free churches x1.1778 (2.61%
to 3.08%), Islam x1.2717 (1.56% to 1.99%). The drawn citizen composition then matches rounds
10-11 to the hundredth on those three and No religion comes out at 59.37% against 60.01%.

## 8. What the finished country looks like

| | national | lowest county | highest county |
|---|---:|---|---|
| unaffiliated | 54.43% | Rogaland 48.02% | Trøndelag 60.49% |
| Lutheran (Church of Norway) | 28.64% | Oslo 19.98% | Nordland 35.90% |
| Protestant (free churches) | 4.44% | Nordland 2.05% | Rogaland 8.00%, Agder 7.99% |
| Catholic | 4.74% | Nordland 3.20% | Oslo 5.62% |
| Islam family | 3.74% | Møre og Romsdal 2.04% | Oslo 6.53% |
| Orthodox and Oriental | about 1.5% | Vestfold og Telemark 1.17% | Oslo 2.29% |
| foreign citizens | 11.13% | Innlandet 7.68% | Oslo 16.29% |

Two geographies overlap and neither is Flanders-against-Wallonia simple. **The capital region is
the secular and Muslim pole** (Oslo and Viken lowest on the Lutheran row, highest on Islam), and
**the south-west is the free-church pole**: Rogaland and Agder at 8% against 2% in the north,
which is the Bible belt, and the lowest unaffiliated shares in the country. The north and
Innlandet are the most Lutheran by answer as well as by roll.

### The roll beside the map, at the 11 counties, 2020

| county | roll, Church of Norway | map, Church of Norway among citizens | roll, Islam | map, Islam |
|---|---:|---:|---:|---:|
| Nordland | 79.1% | 38.95% | 1.03% | 2.12% |
| Innlandet | 78.3% | 37.83% | 1.61% | 2.98% |
| Møre og Romsdal | 77.8% | 39.31% | 0.97% | 2.04% |
| Troms og Finnmark | 76.5% | 38.95% | 1.31% | 2.22% |
| Trøndelag | 75.1% | **29.20%** | 1.51% | 2.26% |
| Vestland | 73.4% | 39.31% | 1.41% | 2.12% |
| Vestfold og Telemark | 68.6% | 29.50% | 2.63% | 4.28% |
| Rogaland | 68.6% | 36.05% | 2.57% | 2.88% |
| Agder | 66.1% | 36.05% | 2.25% | 3.18% |
| Viken | 64.4% | 26.75% | 4.15% | 5.10% |
| Oslo | 46.4% | 23.89% | **9.64%** | **6.53%** |

Spearman over the 11: Church of Norway **+0.718** (p 0.013), Islam **+0.964**, other Christians
+0.782. Nationally the roll has 67.7% in the church, 12.85% in another community and 19.44% on
no roll; the map has 28.64% Lutheran and 54.43% unaffiliated.

**Read the church column as Sweden's §4, not as a failed validation.** Membership is conferred
by baptism and ended by a form; the survey asks what people consider themselves. The two agree
that Oslo is last and disagree about Trøndelag, which is fifth on the rolls and among the least
Lutheran counties by answer. Paired counties sharing a value (Nordland and Troms og Finnmark,
Rogaland and Agder, Vestland and Møre og Romsdal) are paired because they sit in one NUTS 2016
region; the map cannot separate them.

**The Islam column is where the roll's counts run above the map's**, by about a third in Oslo.
The roll counts members' children and residents who joined a mosque, and the map's citizen half
is adults' answers applied to everyone; nothing here says which is nearer the people.

## 9. What the build cannot do, in one list

- **Seven regions, not eleven counties, for the citizen half.** Counties inside one NUTS 2016
  region draw the same citizen composition, so Agder and Rogaland differ only through their
  foreign residents.
- **Viken is a blend.** Bærum and Halden draw the same composition, although Akershus and
  Østfold are different places.
- **Placement is population, not religion.** Oslo is one kommune of 693,000 people, so its
  Muslim dots spread over the whole city rather than concentrating in Groruddalen and Søndre
  Nordstrand. SSB publishes population by country background per bydel; that would be the
  Italy weighter for the foreign half and is not built.
- **The humanists are invisible.** Human-Etisk Forbund members are on the grant roll with the
  churches and inside `unaffiliated` here, because ESS offers no humanist answer.
- **The lay-mission movement is inside the church** (Normisjon, NLM, Indremisjonsforbundet), and
  the free churches are one box.
- **Among citizens, Catholics, Orthodox, Jews and the Eastern religions are at the national
  rate.** The Catholics' real geography (Oslo, Rogaland's oil coast, the Polish and Lithuanian
  workers) comes through only in the foreign half.
- **The regional pattern is 2010-2018's.** The level is 2020-2023's; if the geography of
  secularisation changed shape as well as size in between, this cannot see it.

## 10. Numbers to check a rebuild against

```
rlgdnno nests in rlgdnm, all 7 rounds, 9 substantive answers
7,045 answered citizens, rounds 5-9, 7 NUTS 2016 regions (503 to 1,558 per region)
9,611 answered citizens, all seven rounds
0.35% of citizens declined (weighted, rounds 5-9)
5,391,370 in cens_21ctz_r3 = 4,789,815 NAT + 599,825 FOR + 1,700 STLS + 30 UNK
7 regions: No religion, Den norske kirke, free churches pass; Islam by OVERRIDE
factors x0.7317 church, x1.1778 free churches, x1.2717 Islam
Orthodox split 15,279 : 10,403 (2018 grant list)
drawn 5,372,689 of 5,391,370, 99.65%
unaffiliated 54.43%  lutheran 28.64%  catholic.latin 4.74%  protestant 4.44%  islam.sunni 3.49%
roll 2020: Church of Norway 67.7%, other communities 12.85%, no roll 19.44%
Spearman roll vs map, 11 counties: church +0.718, Islam +0.964, other Christian +0.782
```

## 11. Review, 2026-09-14

rd-review, session `f95259a4-norev`. Read from `data/raw/no/`, the two normalized CSVs and the 2018
PDF, not from §1-10. `check_md`, `built_countries --check` and `check_rollup no` are clean. A
screenshot at Norway's bbox draws dots where people live and none in the sea.

**The crossing vintages and Viken's blend hold out of sample.** Rounds 10-11 never feed the
regional pattern, so they can test it. The drawn citizen composition summed to the six NUTS 2021
regions, against rounds 10-11 read at those regions directly (weighted shares, unweighted n):

| region | n | Church of Norway, drawn / R10-11 | no religion, drawn / R10-11 |
|---|---:|---:|---:|
| NO02 Innlandet | 204 | 37.83 / 37.64 | 56.46 / 57.33 |
| NO06 Trøndelag | 260 | 29.20 / 30.00 | 64.91 / 63.64 |
| NO07 Nord-Norge | 186 | 38.95 / 37.63 | 56.47 / 58.33 |
| NO08 Oslo og Viken | 913 | 25.76 / 23.58 | 64.25 / 66.46 |
| NO09 Agder og Sør-Østlandet | 373 | 32.27 / 30.71 | 57.39 / 59.46 |
| NO0A Vestlandet | 630 | 38.19 / 37.24 | 53.71 / 53.33 |

Every gap is inside about 1.5 standard errors. The free churches agree where they are large (NO09
5.14 / 5.16, NO0A 4.36 / 4.83). Trøndelag is low on the church by answer in both pools, so the
note's Trøndelag sentence does not rest on 2010-2018 alone.

**The rescale is a trend and not a break at round 10.** Church of Norway among answered citizens,
weighted, by round 5 to 11: 51.0, 44.9, 42.4, 45.9, 35.7, then 35.1, 29.1. Round 9 (2018) is
already where round 10 is, so a mode or card change at round 10 does not explain the drop.

**The Islam override holds, and its stated reason had one error.** Spearman +0.929 reproduces from
the raw ESS files against §6's roll column, but over seven units +0.929 is two swapped pairs, not
one: Innlandet and Agder og Rogaland, and also Vestlandet (0.42 ESS, 1.32 roll) and Trøndelag
(0.32, 1.51). §6's "with Innlandet and Agder og Rogaland swapped" is short a pair. Fixed in
`OVERRIDE`'s text and the no.py docstring, and note_public no longer says the rolls order the
regions "the same way". Out of sample, rounds 10-11 agree on the capital (Oslo og Viken 3.83%
drawn, 3.98%) and say nothing usable elsewhere; Trøndelag comes out 3.18% on about eight
respondents against 0.40% drawn. **The 2019 old-county roll the +0.929 was computed on is not in
`data/raw/no/` and `fetch()` does not get it**, so the override's evidence is §6's seven figures
and cannot be regenerated from the repo.

**The Orthodox split is confirmed from the PDF.** The 2018 list, re-read from Wayback (full URL now
in no2024.py's comment; 26 pages, `%%EOF` present): all 60 transcribed rows match a line of the
PDF exactly, and a keyword sweep (ortodoks, tewahedo, koptisk, eritreisk, etiopisk, armensk,
syrisk, the Eritrean dedications, the national churches) finds nothing left out except
`DEN ERITREISKE EVANGELISK LUTHERSKE KIRKE I NORGE` and three Muslim communities, all rightly
excluded.

**One note figure was the wrong half.** note_public said the free-church answer "is 8.0% of
Rogaland and Agder against 2.1% of Nordland". Those are both halves, and the foreign half's
`christianity.protestant` is about half of Nordland's figure. The survey answer among citizens is
7.18% and 1.04%, and the note now says 7.2% of citizens against 1.0%.
