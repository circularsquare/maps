# Latvia — `sources/lv.py`, `sources/lv_geo.py`, `taxonomy/lv2024.py`

Drawn 2026-09-14. **6 units, 50 nodes, 1,872,263 people, 98.89% of the country.** sources.md §9dj is the
summary. The Ministry of Justice's membership figures are printed beside the map, not drawn (§1).

| | |
|---|---|
| counting geography | **the 6 statistical regions (NUTS 3 2021)**, 315,000 people each |
| placement | 119 LAUs, GISCO 2021 (the municipalities before the July 2021 reform), weighted by population |
| basis | self-identification (survey) for citizens and recognised non-citizens; nationality-derived for foreign citizens |
| tier | **`modelled` throughout** |
| vintage | ESS 2008-2024 (rounds 4, 9, 11 for the pattern; 9-11 for the level), census 2021, Pew 2020; Justice Ministry 2013 and 2025 printed beside |

---

## 1. The office check: congregations, and a ministry that counts members

**The Central Statistical Bureau (CSP) counts congregations, not people.** Its PxWeb search
(`data.stat.gov.lv/api/v1/lv/OSP_PUB?query=reli*`, 2026-09-14) returns one religion table, `KUR010`,
registered congregations by denomination, national, 1990-2021; sources.md §11a had found the same.
CSP's statistics that are not published are a paid individual service (its services page,
`csp.gov.lv/lv/pakalpojumi/statistikas-un-saistitas-informacijas-pieejamibas-nodrosinasana`), and
nothing like a shelf of past orders was found. Latvia's censuses do not ask religion (§11a).

**The Ministry of Justice publishes every religious organisation's reported members, with no
geography.** The 2025 report (`tm.gov.lv/lv/media/29203`, 7 pages, `%%EOF` present; attachments
`29206` for the unions and `29209` for the autonomous congregations) gives 1,286,431 members in all:
Lutheran 701,118, Catholic 304,759, Orthodox 250,240, Baptist 6,544, Old Believer 2,060. The Catholic
Church files its four dioceses as one figure and the Orthodox Church its two.

**The 2013 report is the one with any geography, and it explains the Old Believers.** Read from the
page itself (`tm.gov.lv/lv/publiskais-parskats-...-2013gada`), §2.5:

| Catholic diocese, 2013 | members |
|---|---:|
| Riga Metropolitan Curia | 222,910 |
| Rezekne-Aglona | 88,000 |
| Jelgava | 50,760 |
| Liepaja | 28,000 |
| total | 389,670 |

and *"Latvijas Vecticībnieku Pomoras baznīca uzskaita tikai tos draudzes locekļus, kuriem ir
balsstiesības draudzes kopsapulcēs"*: the Pomor Church counts only members with a vote at
congregational meetings (2,355 that year), while 41,877 people attend its services (36,712 in the
church, 5,165 in the Rezekne cemetery congregation).

**None of it is drawn.** It is a `roll` (spec §3.1), and Estonia and Lithuania are drawn from their
censuses' self-identification. On the roll Latvia would be 37% Lutheran; by what citizens say it is
11.8%.

## 2. Rounds, files and variables

ESS's Latvia page lists rounds 3, 4, 7, 9, 10 and 11. Probed 2026-09-14:

| round | fieldwork | file | region | belonging | card | citizenship named | weight | used for |
|---|---|---|---|---|---|---|---|---|
| 3 | 2006-07 | `ess3lv`, separate | **`regionlv`** | `rlgblg` | `rlgdnm` only | no | none | **witness** (§5) |
| 4 | 2008-09 | integrated | **`regionlv`** | `rlgblg` | `rlgdnlv` | `ctzshipb` | `dweight` only | pattern, non-citizens |
| 7 | 2014-15 | integrated | no Latvian response | | | | | nothing |
| 9 | 2018-20 | integrated | `region` | `rlgblg` | `rlgdnlv` | `ctzshipd` | `pspwght` | pattern, level, non-citizens |
| 10 | 2020-22 | self-completion | `region`, **no geography** (§3) | `scrlgblg` | `rlgdnlv` | no | `pspwght` | level only |
| 11 | 2023-24 | integrated | `region` | `rlgblg` | `rlgdnlv` | no | `pspwght` | pattern, level |

**Rounds 1-4 have a region variable after all.** Greece, Sweden, Belgium and Denmark all recorded
rounds 1-4 as having no `region`, which is true. Latvia's rounds 3 and 4 carry `regionlv`, the six
statistical regions by name, so the others very likely carry `regiongr`, `regionse` and so on. Not
checked for them; it would add up to four rounds each.

Three smaller things. Round 7's integrated file has no Latvian response and `doi.org/10.21338/ess7lv`
is 404, although the country page lists the round. Round 11's `regunit` says NUTS level 2, which for
Latvia is the whole country; the codes are NUTS 3. Round 10's `scrlgblg` answers `No, never` and
`No, but did in the past`, both read as No.

**`rlgdnlv` is finer than `rlgdnm` and nests in it** (`_check_card`, rounds 4, 9, 11): Lutheran,
Baptist and other Protestant are three answers inside `Protestant`, and Russian or Greek Orthodox and
Other Orthodox two inside `Eastern Orthodox`. No `a`/`b` revision exists.

## 3. Every round's regions against the people behind them

Denmark's round 9 (§9dg) published scrambled labels. Each Latvian round checked on three things the
survey carries (all respondents, unweighted):

| round | Riga, big city | Latgale, Russian at home | Latgale, Catholic of those naming one | language chi-square across regions |
|---|---:|---:|---:|---:|
| 3 | 88% | 55% | 56% | 5.0e-42 |
| 4 | 94% | 52% | 66% | 1.2e-66 |
| 9 | 81% | 55% | 62% | 6.9e-36 |
| **10** | **37%** | **23%** | **29%** | **0.86** |
| 11 | 95% | 36% | 69% | 1.0e-18 |

**Round 10's `region` carries no geography.** Every region is 34-41% big city, 23-30% Russian at home
and 28-40% Catholic; Pieriga and Zemgale are more big-city than Riga. It is not a permutation, since
no relabelling of six identical rows recovers anything, so the round cannot be recoded the way
Denmark's was. `_check_labels` requires Riga first on big city, Latgale first on Russian outside Riga
and on Catholic, and Vidzeme last on Russian: rounds 3, 4, 9 and 11 must pass and round 10 must fail,
so a corrected re-issue stops the build. Its national composition agrees with rounds 9 and 11, and it
is used for the level (§6).

**NUTS vintages.** The NUTS 2013-2016 correspondence has no Latvian row. Latvia redrew its regions on
1 January 2024 (Riga and Pieriga merged, Vidzeme and Kurzeme recoded LV00C, LV00B; CSP IRE031's
metadata lists both sets); round 11 and the 2021 census both use the earlier six, so nothing crosses
here. Rounds 3 and 4 were fielded under NUTS 2006, whose correspondence to 2013 was not checked;
their labels pass the check above.

## 4. Three parts, and the non-citizens inside Eurostat's `FOR`

| region | people | citizens | recognised non-citizens | foreign citizens |
|---|---:|---:|---:|---:|
| Kurzeme | 236,022 | 89.56% | 7.37% | 3.07% |
| Latgale | 252,682 | 88.37% | 8.40% | 3.23% |
| Riga | 614,618 | 78.73% | 15.83% | 5.44% |
| Pieriga | 378,982 | 90.06% | 7.65% | 2.29% |
| Vidzeme | 183,399 | 95.90% | 3.54% | 0.56% |
| Zemgale | 227,520 | 90.11% | 8.48% | 1.41% |

**`cens_21ctz_r3` reports Latvia's 190,544 recognised non-citizens as `RNC`, and `FOR` includes
them**: NAT 1,640,782 + FOR 252,305 + STLS 136 = TOTAL 1,893,223. The named citizenships under `FOR`
sum to 61,472. Scaling them up to `FOR`, as every other two-half country does, would multiply Russia,
Ukraine and Belarus by four and draw Latvia's non-citizens as Russian citizens on Pew's Russia. The
foreign half's target is `FOR - RNC`, and the build stops if the named citizenships stop covering it.
`be.py`, `fi.py`, `fr.py`, `gr.py` and `it.py` spread `RNC` over the named citizenships, which is
harmless where it is tiny.

**The non-citizens are drawn from the survey that sampled them.** Rounds 4 and 9 ask non-citizens
which citizenship they hold and Latvia's answer with the code for an alien's passport (65, 6500): 252
answered respondents, 184 of them in round 4. National composition, weighted: Russian or Greek
Orthodox **45.04%**, no religion 33.88%, Catholic 8.68%, Christian unspecified 5.37%, Other Orthodox
3.72%, Lutheran 0.41%. It goes into each region's `RNC` + `STLS` count. Rounds 10 and 11 do not name
the citizenship, so this part is neither regional nor rescaled, and it is mostly 2008-09.

## 5. Which categories carry their own geography, and two drawn against the test

Norway's `_stability`, imported. Three rounds give three splits of one against two; **3,681 answered
citizens** (440 in Kurzeme to 975 in Riga); 1.23% declined (weighted).

| category | n | median rho | null 95th | p | chi² p | drawn at |
|---|---:|---:|---:|---:|---:|---|
| No religion | 2,050 | +0.714 | +0.600 | 0.0295 | 1.4e-55 | residual (passes) |
| **Catholic** | 650 | +0.829 | +0.657 | 0.0105 | 1.3e-114 | **6 regions** |
| **Lutheran** | 544 | +0.086 | +0.600 | 0.4293 | 4.0e-14 | **6 regions, OVERRIDE** |
| **Russian or Greek Orthodox** | 238 | +0.600 | +0.600 | 0.0755 | 3.1e-15 | **6 regions, OVERRIDE** |
| Christian, denomination not specified | 64 | +0.143 | +0.600 | 0.3723 | 2.1e-02 | residual |
| **Other Orthodox Denominations** | 71 | +0.771 | +0.657 | 0.0250 | 5.0e-20 | **6 regions** |
| **Baptist** | 29 | +0.698 | +0.638 | 0.0380 | 8.0e-06 | **6 regions** |
| **Other Christian Denominations** | 19 | +0.754 | +0.600 | 0.0150 | 4.99e-02 | **6 regions** (likeliest false pass) |
| Other Protestant Denominations | 8 | −0.393 | +0.548 | 0.8721 | 8.7e-01 | residual |
| Eastern religions | 3 | +0.465 | +0.775 | 0.3543 | 4.4e-01 | residual |
| Other Non-Christian Religions | 4 | +0.220 | +0.775 | 0.4063 | 4.8e-01 | residual |

### Why two categories are drawn against the test

**The residual construction reversed them.** With both at the national rate inside each region's
residual (level already scaled, §6), citizens came out:

| drawn (survey) | Kurzeme | Latgale | Riga | Vidzeme |
|---|---:|---:|---:|---:|
| Lutheran | 12.96% (14.66%) | **7.04% (2.66%)** | 12.49% (13.85%) | 12.78% (16.84%) |
| Russian or Greek Orthodox | **7.10% (2.85%)** | **3.86% (11.03%)** | 6.84% (11.35%) | 7.00% (3.16%) |

Latgale's non-Catholic remainder is Orthodox and Kurzeme's is Lutheran, so a tail shared at national
proportions moves each church's people into the other's region. Denmark's residual softened a
contrast (§9dg §5); Latvia's reverses one. Both categories are Denmark's shape, a unit or two standing
apart and the rest shuffling: Latgale is the least Lutheran region in every round (2.7%, 3.5%, 2.0%,
against 6.8% or more), and Riga and Latgale are the two most Orthodox in every round. spec §12 says to
leave such a category at the national rate **unless an independent witness orders the units**. Two do.

**Round 3**, a separate sample outside the pool and the test, on the harmonised card, unweighted,
1,728 citizens. Exact p over all 720 orderings:

| round 3 answer | against the pool's | Spearman | exact p |
|---|---|---:|---:|
| Protestant | Lutheran + Baptist + other Protestant | **+0.886** | 0.0167 |
| Eastern Orthodox | Russian or Greek + other Orthodox | +0.657 | 0.0875 |
| Roman Catholic | Catholic | +0.943 | 0.0083 |
| No religion | No religion | +0.829 | 0.0292 |

**The population register** (CSP `IRE031`, 1 January 2021; its regional totals equal the census's to
the person), Russians, Belarusians and Ukrainians as a share of each region, against the pool's
Russian or Greek Orthodox among citizens:

| | Riga | Latgale | Pieriga | Zemgale | Kurzeme | Vidzeme |
|---|---:|---:|---:|---:|---:|---:|
| East Slavic, register | 43.1% | 42.3% | 22.1% | 21.4% | 17.7% | 10.0% |
| Orthodox, survey | 11.35% | 11.03% | 4.92% | 3.61% | 2.85% | 3.16% |

**Spearman +0.943, exact p 0.0083**, Kurzeme and Vidzeme swapped at the bottom. Ethnicity is not
religion and the register counts non-citizens; it is evidence about where, not how many.

So Lutheran is drawn at the regions on round 3's +0.886, and Orthodox on a near miss plus the
register's +0.943. Both are in `OVERRIDE` with the reason printed, and `EXPECT_PASS` asserts the five
that pass on their own. The witness files are fetched by `lv.py --fetch` (`ess_r3_*.json`,
`csp_ire031_2021.csv`), so the evidence regenerates, which Norway's review found missing for Norway.

## 6. The level, which moved eleven points

Citizens, weighted:

| | r4 2008-09 | r9 2018-20 | r10 2020-22 | r11 2023-24 | pool (4, 9, 11) | **late (9-11)** |
|---|---:|---:|---:|---:|---:|---:|
| No religion | 51.79% | 59.66% | 61.99% | 65.86% | 57.97% | **62.87%** |
| Catholic | 19.04% | 14.73% | 13.70% | 12.76% | 16.10% | **13.62%** |
| Lutheran | 16.52% | 12.08% | 12.21% | 11.29% | 13.88% | **11.81%** |
| Russian or Greek Orthodox | 7.44% | 7.25% | 7.17% | 5.34% | 6.74% | **6.47%** |
| Other Orthodox | 1.46% | 1.45% | 0.32% | 1.81% | 1.57% | **1.23%** |

Round 4 is 46% of the pool, so the pool is a 2015 figure beside a 2021 census. **spec §3.4 is
applied, Norway's way**: every category drawn at the regions keeps the pool's regional pattern and is
scaled by one factor to the late national share, at census citizen weights (Catholic x0.8580, Lutheran
x0.8483, Orthodox x0.9200, Other Orthodox x0.7715, Baptist x0.7163, Other Christian x1.7879); the
residual shares what is left at the late proportions. The drawn citizen composition then matches the
late rounds to the hundredth.

**A factor, not a shift**, on the survey's own regions, round 4 against rounds 9+11: the ratios vary
less than the point differences for No religion (coefficient of variation 0.24 against 0.54), Catholic
(0.24 against 1.11) and Lutheran (0.34 against 0.91). Noisy on these sample sizes, and all three
point the same way.

## 7. The mapping, three calls

- **Other Orthodox Denominations -> `christianity.orthodox.oldbeliever`**, lt2021.py's and pl2021.py's
  node. The other Orthodox body of any size is the Pomor Church. The survey puts 71 respondents there,
  Latgale 6.1% of citizens and nowhere else above 1.3%; the map draws 27,029 (1.44%), against 41,877
  worshippers the church reported in 2013. Among all respondents the answer is 12.7% of the Orthodox
  in round 4, 13.9% in round 9 and **31.8% in round 11**, after the Saeima declared the Latvian
  Orthodox Church independent of Moscow in 2022; some round 11 respondents of that church may have
  answered `other`. Among citizens the jump is smaller (1.81% against 1.46%) and it is left in.
- **Russian or Greek Orthodox -> `christianity.orthodox.canonical`**, Estonia's and Lithuania's node.
- **Islam -> `islam`, the root**, Estonia's call: Tatars, Azerbaijanis (mostly Shia) and Central
  Asians, 175 members in 2025. Zero citizens answered it in the pool; its dots are the foreign half's.

## 8. What the finished country looks like, and the rolls beside it

| region (all three parts) | unaffiliated | Catholic | Lutheran | Orthodox | Old Believer | non-citizens | foreign |
|---|---:|---:|---:|---:|---:|---:|---:|
| Kurzeme | 69.16% | 5.87% | 11.16% | 7.43% | 0.28% | 7.37% | 3.07% |
| Latgale | 35.31% | **39.31%** | **2.03%** | 14.64% | **4.46%** | 8.40% | 3.23% |
| Riga | 56.49% | 9.12% | 9.30% | **17.77%** | 1.35% | 15.83% | 5.44% |
| Pieriga | 63.57% | 8.61% | 13.77% | 8.62% | 1.14% | 7.65% | 2.29% |
| Vidzeme | 68.96% | 8.02% | 13.71% | 4.61% | 0.26% | 3.54% | 0.56% |
| Zemgale | 61.42% | 12.53% | 12.46% | 7.52% | 1.01% | 8.48% | 1.41% |

Nationally: unaffiliated 58.46%, Latin Catholic 12.91%, canonical Orthodox 11.73%, Lutheran 10.26%,
`christianity` 3.09%, Old Believer 1.44%.

| roll against map, share of 1,893,223 | Justice Ministry 2025 | map |
|---|---:|---:|
| Lutheran | 37.0% | 10.3% |
| Catholic | 16.1% | 12.9% |
| Orthodox | 13.2% | 11.7% |
| Old Believer | 0.1% (voting members) | 1.4% |
| Baptist | 0.3% | 0.5% |

The Lutheran gap is Sweden's and Denmark's: a church roll against stated belonging. The Catholic and
Orthodox gaps are small, and the Old Believer one runs the other way for the reason §1 quotes.

**The Catholic dioceses order the regions the same way as the survey** (2013 roll against the drawn
citizen share, paired approximately): Rezekne-Aglona 34.8% / 43.4%, Jelgava 22.3% / 12.9%, Riga
archdiocese 18.9% / 8.8%, Liepaja 11.9% / 5.6%. Spearman +1.000 over four units, an ordering and not a
test.

## 9. What the build cannot do

- **Six regions.** Daugavpils, Latvia's most Russian city, draws Latgale's composition, and Riga is
  one LAU of 615,000 people, so its Orthodox dots spread evenly over the city.
- **The non-citizens are one national composition** from 252 respondents, mostly 2008-09, unscaled.
- **Christian unspecified, other Protestant and the non-Christian answers are at the national rate**
  inside each region's residual. Nothing among citizens says where Jews or Muslims live (1 and 0
  respondents); their dots are the foreign half's.
- **The Dievturi are inside `other.lv`**, with the Baha'i and the Eastern congregations.
- **Round 10 cannot be placed**, and a round 12 will be on the 2024 regions, which do not nest in the
  six used here (Riga and Pieriga merged).
- **Placement is population.** Sweden's and Denmark's named improvement (citizenship per municipality
  for the foreign half) would matter less here, since the foreign citizens are 3.3%.

## 10. Numbers to check a rebuild against

```
rounds 4, 9, 11 pooled at 6 NUTS 3 regions; round 3 witness only; round 10 level only (region void)
3,681 answered citizens (440 to 975 per region); 1.23% declined (weighted)
252 answered alien's-passport holders (r4 184, r9 68); 0.41% declined
1,893,223 in cens_21ctz_r3 = 1,640,782 NAT + 252,305 FOR (incl. 190,544 RNC) + 136 STLS
foreign target FOR - RNC = 61,761; 200 named citizenships cover 61,472 (99.53%)
pass: No religion, Catholic, Other Orthodox, Baptist, Other Christian; OVERRIDE: Lutheran, Russian or Greek Orthodox
witnesses: round 3 Protestant +0.886 (p 0.0167); CSP IRE031 East Slavic vs Orthodox +0.943 (p 0.0083)
late level r9-11: none 62.87, Catholic 13.62, Lutheran 11.81, Orthodox 6.47
drawn 1,872,263, 98.89%; unaffiliated 58.46%  catholic.latin 12.91%  orthodox.canonical 11.73%  lutheran 10.26%
GISCO LAU 2021: 119 LAUs, 1,892,623 people
```

## 11. Review, 2026-09-14 (session `f95259a4-lvrev`)

- **The odd-round halving bug (spec §12, sources.md §9dk) does not reach Latvia.** `_citizen_shares`
  calls `no._stability`, whose `no._splits` (since 2026-09-14 `stability.py::halvings`) gives all three
  halvings of three rounds (r4 against r9+r11, r9 against r4+r11, r11 against r4+r9). The `if 0 in a`
  filter `cab.stability` and `se._stability` used then would have kept only the first; both call
  `stability.halvings` now. Re-run read-only, §5's table reproduces exactly. Per halving,
  Lutheran is +0.086, +0.143, +0.029 and Russian or Greek Orthodox +0.600, +0.657, +0.600, against a
  null 95th of +0.600: no halving rescues Lutheran, and the Orthodox near miss is the same on every
  one. Both failures are real, so both overrides are still needed.
- **Both witnesses reproduce** (+0.886, p 0.0167; +0.943, p 0.0083), and what they actually establish
  is narrower than an order of six. Round 3's Protestant share is 4.4% in Latgale, 9.4% in Riga and
  15.0-16.2% in the other four; round 3 and the register agree that Riga and Latgale are the most
  Orthodox, not on the order below them (round 3 puts Kurzeme third). For the regions nothing
  orders, the override draws the pool's own shares, which is noise with no direction. The national
  rate reverses Latgale. The override is the better of the two.
- **Round 10 re-checked from the raw files.** Big city is 34-41% and Russian at home 23-30% in every
  region (chi-square p 0.86), while the sample is spread in proportion to population (Riga 34%,
  Pieriga 21%, Kurzeme and Latgale 13%, Vidzeme and Zemgale 9%). That is a region assigned
  independently of the respondent; no recode recovers it. Level only is right.
- **`FOR` includes `RNC` in Eurostat's own hierarchy**: FOR 252,305 = EU_FOR 6,343 + NEU 245,962, and
  NEU's European part EUR_NEU (239,926) holds RNC 190,544. The 289 between the named citizenships
  (61,472) and FOR - RNC (61,761) are exactly `EUR_OTH`, spread over the named ones by the scale.
- **The non-citizen mix is stable across its two rounds**: alien's-passport holders, weighted, are
  44.4% and 46.8% Orthodox and 32.8% and 37.1% no religion in r4 and r9. Rounds 10 and 11 cannot
  separate non-citizens, but all their `ctzcntr = No` respondents (67 and 83, foreign citizens
  included) give no religion 42.0% and 38.9%. The drawn 33.9% is probably a few points low for 2021.
  Samples too small to act on; not changed.
- **`note_public`**: cut "The churches count far more", which the Catholic roll (16.1% against the
  survey's 13.6%) does not support; the Lutheran figure stays. Not pushed to counts.json
  (`tiles.py --refresh-meta`) while other builders are live. Screenshot: dots on land in all six
  regions, heaviest in Riga, Daugavpils and Liepaja, none in the sea.
