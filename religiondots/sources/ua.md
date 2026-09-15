# Ukraine — `sources/ua.py`, `sources/ua_geo.py`, `taxonomy/ua2013.py`

Drawn 2026-09-14. **27 units, 9 nodes, 42,007,553 people, 96.52% of 43,520,435.** sources.md §9dw is
the summary. Ask 016 holds the three calls that are Anita's (§3, §6, §7).

| | |
|---|---|
| counting geography | **COD-AB's 27 first-level units** (24 oblasts, Kyiv city, Crimea, Sevastopol), 1.6 million people each |
| survey geography | 26 units (ESS never names Sevastopol, which takes Crimea's composition) |
| placement | Kontur H3 r8 hexes (2023-11-01), 295,398 cells |
| basis | self-identification, sample survey, everyone (no foreign half) |
| tier | **`modelled` throughout** |
| vintage | ESS 2005-2013 (rounds 2-6) for pattern and level; round 11 (2023-24) a witness; Ukrstat population 1 January 2022 (Crimea and Sevastopol 1 January 2014) |

---

## 1. The office check: organisations, not people

**No Ukrainian census has asked religion in 1989 or 2001, and there has been no census since.** The
State Statistics Service publishes no religion table (sources.md's census list already had Ukraine
as "genuinely not asked").

**The State Service for Ethnic Policy and Freedom of Conscience (DESS) publishes Form 1 every year**,
on data.gov.ua under CC BY (dataset `5f62ea97-4248-4916-99ae-41aa1df54c44`): registered religious
organisations (centres, communities, monasteries, missions, schools) by denomination for every region,
89 template rows per region. The 1 January 2024 workbook is fetched as
`dess_form1_2024-01-01.xlsx`; sheet `За областями`, communities in column 4. It counts institutions
(spec §3.6), in 2024, and has no Crimea or Sevastopol block, so it is a witness and is not drawn.

`data.gov.ua` and `razumkov.org.ua` answer **403 to WebFetch and 200 to a script with a full browser
header set**. `ukrstat.gov.ua` serves an **expired TLS certificate**; the 2022 population volume is
the same PDF on `db.ukrcensus.gov.ua`, and the 2014 express release came from the Wayback Machine.
**COD-PS Ukraine on HDX is marked restricted** ("should be treated as 'restricted' and shared for this
specified purpose", UNFPA, July 2026) and has no public resources, so it is not the population base.

## 2. Rounds, files, variables

Ukraine is in ESS rounds 2-6 and 11 and in none of 7-10 (probed 2026-09-14: no `UA` response in the
main or the self-completion files). **`search.searchDatafiles` no longer exists in the ESS schema**
(sources/fr.py's comment is stale); the main datafile ids come from
`search.seriesMetadata(id:"321b06ad-…", version:985){studies{mainDataFiles{id version}}}`.

| round | fieldwork | respondents | region | card | not sampled |
|---|---|---:|---|---|---|
| 2 | 2005 | 2,031 | `regionua` (Latin names) | `rlgdnm` only | Vinnytsia, Ternopil |
| 3 | 2006-07 | 2,002 | `regionua` | `rlgdnm` only | Ternopil, Khmelnytskyi, Cherkasy, Chernivtsi |
| 4 | 2009 | 1,845 | `regionua` | `rlgdnm`, `rlgdnua` | Kyiv oblast, Poltava, Khmelnytskyi |
| 5 | 2011 | 1,931 | `region` UA11-UA83 (Ukrainian names) | `rlgdnm`, `rlgdnua` | Kyiv oblast, Poltava, Khmelnytskyi |
| 6 | 2013 | 2,178 | `region` | `rlgdnm`, `rlgdnua` | none |
| 11 | 2023-24 (ESS11's span; the file has no interview-year variable) | 2,661 | `region` | `rlgdnm`, `rlgdnaua` | Crimea, Donetsk, Luhansk |

Rounds 2-6: **9,641 answered** of 9,987 with an oblast; 3.48% declined (weighted). `pspwght` exists in
every round. `rlgdnua` and `rlgdnaua` nest in `rlgdnm` (`_check_card`), except that round 11's
`Other (Write in)` was coded into several harmonised answers, 3 of them `Protestant`.
**`rlgdnm`'s `Roman Catholic` holds the Greek Catholics** (round 4: 142 of 146).

Region labels are matched to COD-AB pcodes by stem and asserted unique; the first digit of rounds
5, 6 and 11's codes is asserted against the eight groups used as the coarse level (§5).

## 3. Population, boundaries, and occupied territory (ask 016)

- **Population.** Ukrstat's present population at 1 January 2022, table 1 of *Чисельність наявного
  населення України на 1 січня 2022 року*, for the 24 oblasts and Kyiv city, 41,167,335, with
  Donetsk (4,059,372) and Luhansk (2,102,921) as whole oblasts. The volume itself says it excludes
  Crimea and Sevastopol. For those two, the last Ukrstat figure, 1 January 2014: 1,967.2 and 385.9
  thousand. Total 43,520,435.
- **Boundaries.** COD-AB Ukraine v05 (valid from 2025-09-01, CC BY-IGO, from the State Scientific
  Production Enterprise "Kartographia"): all 27 pcodes, names checked against `ua.UNIT_NAMES`.
- **Occupied territory is drawn**, from the rounds that sampled it before occupation: Crimea has 502
  respondents in rounds 2-6, Donetsk 957 and Luhansk 571. Georgia's Abkhazia is blank because the
  census never enumerated it; these places were sampled. `note_public` says the map shows neither
  displacement nor change under occupation.
- **Kontur against Ukrstat**: Spearman +0.990 over 27, 37.9M against 43.5M, ratios 0.78-1.00 except
  **Sevastopol 0.51x**: COD-AB's Sevastopol polygon is 57 km², the city, not the city council's
  territory, so its outer settlements' hexes sit in Crimea. A placement fact; no count moves.
- `kontur_cap.py ua`: no block at the cap.

## 4. Every round's oblasts against the people behind them

| round | big city top | Catholic top | Russian at home top | Donetsk/Luhansk Catholic | sample vs population Spearman |
|---|---|---|---|---:|---:|
| 2 | Kyiv 100% | Lviv 67% | Crimea 97% | 0.5% | +0.69 |
| 3 | Kyiv 97% | Ivano-Frankivsk 78% | Donetsk 97% | 0.8% | +0.67 |
| 4 | Kyiv 98% | Ivano-Frankivsk 76% | Donetsk 96% | 0.5% | +0.64 |
| 5 | Kyiv 99% | Ivano-Frankivsk 73% | Crimea 97% | 0.0% | +0.67 |
| 6 | Kyiv 97% | Ivano-Frankivsk 66% | Luhansk 97% | 0.6% | +0.89 |
| 11 | Kyiv 100% | Lviv 77% | Dnipropetrovsk 76% | | +0.94 |

No scramble and no void round. The sample-share correlations are lower in rounds 2-5, the rounds
with missing oblasts, where some neighbours carry more respondents than their population (round 5's
Chernihiv is 5.6% of the sample and 2.2% of the people). The ESS sampling documentation for those
rounds was not read, so why is not established; every profile lands where it should. `_check_labels` asserts the three tops and the Donbas ceiling every build.

## 5. Which categories carry their own geography

Norway's `_stability`, imported: five rounds, all ten halvings of 2 against 3, per-round permutation
null, chi-square gate. **A unit absent from a round makes `stability.median_rho` (then `be._median_rho`) skip every halving whose
half lacks it**, and under the per-round permutation the absent unit's zero row moves, so the null
would skip different halvings from the statistic. So the oblast test runs on the **19 oblasts sampled
in all five rounds**; the shares are still drawn for all 26.

| 19 oblasts | n | median rho | null 95th | p | chi² p | drawn at |
|---|---:|---:|---:|---:|---:|---|
| Eastern Orthodox | 4,820 | +0.690 | +0.293 | 0.0005 | 6e-134 | residual (passes; the partition's big category) |
| No religion | 2,226 | +0.709 | +0.290 | 0.0005 | 7e-124 | **oblast** |
| Roman Catholic | 712 | +0.686 | +0.335 | 0.0005 | 0 | **oblast** |
| Other Christian denomination | 109 | +0.237 | +0.312 | 0.1084 | 3e-05 | residual |
| Protestant | 106 | +0.260 | +0.306 | 0.0890 | 3e-14 | residual |
| Islam | 43 | +0.522 | +0.327 | 0.0025 | 1e-36 | **oblast** |
| Other Non-Christian religions | 34 | −0.154 | +0.332 | 0.7976 | 4e-11 | residual |
| Jewish | 6 | +0.422 | +0.396 | 0.0495 | 0.10 | residual (refused on the chi-square) |
| Eastern religions | 9 | +0.370 | +0.393 | 0.0570 | 0.12 | macro-region |

**At the 8 groups of ESS's region codes** (Poltava/Sumy/Kharkiv/Chernihiv; Donetsk/Luhansk;
Dnipropetrovsk/Zaporizhzhia/Kirovohrad; Odesa/Mykolaiv/Kherson/Crimea; Vinnytsia/Ternopil/Khmelnytskyi;
Kyiv city/Kyiv oblast/Cherkasy; Lviv/Ivano-Frankivsk/Zakarpattia/Chernivtsi; Volyn/Rivne/Zhytomyr), all
sampled in every round: No religion, Catholic and **Eastern religions** (9 respondents, p 0.0040,
chi² 0.031) pass; Orthodox misses (p 0.081). Eastern religions is the likeliest false pass
(Belgium's and Latvia's pattern), 0.03% of the map, on `other.ua` with the next answer.

- **Largest (round, oblast) cell** (spec §12, Uzbekistan): Islam 18% (round 3, Crimea), Eastern
  religions 22%, the rest 2-11%. Nothing refused.
- **Standouts** (Honduras): none; Protestant tops both halves in Volyn in 3 of 10 halvings.
- **2x rule**: worst 1.50x (Other Non-Christian religions in Rivne). The residual stands.
- **Orthodox as the residual** never goes negative fixed at its oblast share (worst +0.31%).

**Protestant is not overridden.** It is Denmark's shape (chi² 3e-14, rank p 0.089) and the residual
flattens it (Zakarpattia 1.79% drawn against 7.36% measured, Volyn 2.11% against 6.43%). DESS's
Protestant communities per head order the survey's own oblast shares at **+0.450 (p 0.024) over 25**,
far from Norway's +0.929 or Latvia's +0.886, and counts of congregations are a weaker instrument than a
roll of members. Left in the residual.

## 6. The level, which moved and is not rescaled (ask 016)

Round 11 on the 23 oblasts it sampled, both weighted by 2022 population:

| | pool | drawn | round 11 | drift | Spearman over 23 |
|---|---:|---:|---:|---:|---:|
| Eastern Orthodox | 57.55% | 57.98% | 54.54% | −3.44 | +0.398 |
| No religion | 27.51% | 27.51% | 33.85% | **+6.34** | +0.628 |
| Roman Catholic | 10.78% | 10.78% | 9.54% | −1.24 | +0.825 |
| Protestant | 1.84% | 1.51% | 0.89% | −0.61 | +0.596 |
| Other Christian | 1.56% | 1.53% | 0.91% | −0.62 | +0.093 |

Rounds 2-6 are flat nationally (No religion 28.2-31.2%, Orthodox 57.0-58.5%). **No religion is past the
3.5-point bar**, and Norway and Latvia rescaled there. Not here: the only measurement of the change
excludes Crimea, Sevastopol, Donetsk and Luhansk, so scaling the 23 draws a vintage step along the
2014 line and scaling all 27 asserts a change nobody measured under occupation. `note_public` gives
round 11's 34% beside the map's 28%.

## 7. The Orthodox jurisdictions, printed and not drawn (ask 016)

Share of the Orthodox answer, weighted:

| rounds 4-6 (2009-2013) | Moscow Patriarchate | Kyiv Patriarchate | Autocephalous | other |
|---|---:|---:|---:|---:|
| national | 47.9% | 46.8% | 1.5% | 3.8% |
| Donetsk, Luhansk | 75% | 23% | | 2% |
| Odesa, Mykolaiv, Kherson, Crimea | 57% | 33% | 2% | 8% |
| Dnipropetrovsk, Zaporizhzhia, Kirovohrad | 28% | 69% | | 4% |
| Kyiv city, Kyiv oblast, Cherkasy | 28% | 65% | 3% | 4% |
| Lviv, Ivano-Frankivsk, Zakarpattia, Chernivtsi | 38% | 50% | 8% | 4% |

| round 11 (2023-24) | OCU | no patriarchate | Moscow Patriarchate |
|---|---:|---:|---:|
| national (23 oblasts) | 74.7% | 15.0% | 10.3% |
| Odesa, Mykolaiv, Kherson | 77% | 4% | 20% |
| Lviv, Ivano-Frankivsk, Zakarpattia, Chernivtsi | 78% | 3% | 19% |
| Dnipropetrovsk, Zaporizhzhia, Kirovohrad | 57% | 40% | 3% |

Razumkov (November 2025, n=2,009): OCU 42.1%, Moscow Patriarchate 5.4%, just Orthodox 10.2% of all
adults. DESS 1 January 2024: **UOC 10,586 communities, OCU 8,075**; OCU is 11% of the two in
Donetsk/Luhansk and 62% in the far west. Institutions and answers point opposite ways.

**Why it is one node.** The five pooled rounds share only the harmonised card; the two Ukrainian cards
are different cards on either side of the 2018 merger; self-identification with the Moscow
Patriarchate fell by about three quarters between 2013 and 2025 (Razumkov); and drawing a church by
oblast in wartime, when the state has legislated against organisations affiliated with Russia's
church, is §14. `christianity.orthodox` rather than `.canonical`, because canonical status is itself
contested.

## 8. The Catholic split

`rlgdnm`'s Catholic answer is split Eastern/Latin on the Ukrainian cards of rounds 4, 5, 6 and 11
(761 Catholic respondents; Eastern 94.3% nationally), at the oblast where it has 10 Catholic
respondents, else its macro-region, else nationally: Lviv 98.4% Eastern (329), Ivano-Frankivsk 99.3%
(205), Ternopil 100% (112), Zakarpattia 91.8% (52), **Khmelnytskyi 18.2% (16), Vinnytsia 27.3% (13),
Zhytomyr 27.3% (11)**. Volyn and Rivne take their group's 50%, pulled down by Zhytomyr, on 3 and 2
respondents. DESS orders the Eastern shares at +0.719 and the Latin at +0.565 over 25 (communities
per head; Zhytomyr and Khmelnytskyi top the Latin list on both).

## 9. What the finished country looks like

| unit | Orthodox | Greek Catholic | Latin Catholic | Protestant | Sunni | unaffiliated |
|---|---:|---:|---:|---:|---:|---:|
| Ivano-Frankivsk | 17.88% | **72.13%** | 0.49% | 0.46% | | 8.39% |
| Lviv | 30.54% | 63.30% | 1.05% | 0.79% | | 3.24% |
| Ternopil | 35.10% | 59.31% | | 0.91% | | 3.45% |
| Zakarpattia | 68.95% | 16.70% | 1.48% | 1.79% | | 8.66% |
| Khmelnytskyi | 78.25% | 1.95% | **8.77%** | 2.03% | | 6.25% |
| Rivne | **86.89%** | 0.34% | 0.34% | 2.26% | | 7.12% |
| Kyiv city | 57.38% | 1.64% | 0.10% | 1.49% | 0.19% | 37.19% |
| Crimea, Sevastopol | 53.80% | | | 1.40% | **7.69%** | 35.22% |
| Donetsk | 54.39% | 0.44% | 0.03% | 1.41% | 0.59% | 41.22% |
| Kirovohrad | 46.32% | | | 1.20% | | **50.77%** |

Nationally: Orthodox 57.61%, unaffiliated 29.53%, Greek Catholic 8.14%, `christianity` 1.52%,
Protestant 1.50%, Sunni 0.59%, Latin Catholic 0.58%, `other.ua` 0.43%, Jewish 0.10%.

**Razumkov's four regions, 2025**, drawn against its table (a different answer list with `just
Christian`, so ordering only): Orthodoxy +0.80, Greek Catholicism +0.80, no religion +1.00, Roman
Catholicism +0.60, Protestant +0.20 (the map's Protestant is flat by construction).

## 10. What the build cannot do

- **The pattern and the level are 2005-2013's.** Round 11 says the country is less religious now and
  the Orthodox answer has changed church.
- **No displacement and no occupation.** Everyone is drawn where the 2022 (and 2014) estimates put them.
- **The Orthodox are one colour** (§7), and so are the Protestants: Volyn and Rivne's Baptist and
  Pentecostal belt, the Hungarian Reformed of Zakarpattia and the rest are flattened by the residual.
- **The Jews are at the national rate** (6 respondents); Dnipro, Kyiv and Odesa are not visible.
- **Crimean Tatars are the Muslim dots**, placed by population across Crimea, not by where Tatars live.
- **Kyiv city is one unit**, and Sevastopol is only the city polygon.

## 11. Numbers to check a rebuild against

```
ESS rounds 2-6 pooled (rounds 7-10 have no Ukraine); round 11 witness only
9,987 with an oblast, 9,641 answered, 3.48% declined (weighted), 0.47% non-citizens kept
test units: 19 oblasts in all five rounds; coarse: 8 groups of ESS region codes
oblast: Orthodox, No religion, Roman Catholic, Islam pass (Orthodox kept as residual); Jewish refused on chi2
macro: No religion, Roman Catholic, Eastern religions pass
round 11 on 23 oblasts: No religion +6.34 points, not rescaled (ask 016)
population 43,520,435 = Ukrstat 2022 41,167,335 + Crimea 1,967,200 + Sevastopol 385,900 (2014)
drawn 42,007,553 (96.52%): orthodox 57.61  unaffiliated 29.53  catholic.eastern 8.14  protestant 1.50
Catholic split: 761 respondents, Eastern 94.3% nationally
Kontur r8 295,398 hexes; Spearman with Ukrstat +0.990; Sevastopol 0.51x (city polygon)
DESS 2024: UOC 10,586 communities, OCU 8,075
```
