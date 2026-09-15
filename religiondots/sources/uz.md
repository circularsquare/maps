# Uzbekistan: DRAWN 2026-09-14 from the Central Asia Barometer read by ethnic group on the 2026 census, 14 regions, every row `modelled`

Built by session `f95259a4-uz` on the route §11ao reopened. `sources.md` §9di is the write-up.
**Redrawn the same day by ethnic group (§9, session `f95259a4-uz2`, `sources.md` §9dp).** §0
describes the first build: its population row, its sampling-point paragraph and its Tashkent
figures are superseded by §8 and §9.
Sections 1 to 7 below are the 2026-09-08 closure record and stay true of the census, the office
and LiTS; they no longer close the country.

## 0. The build

| | |
|---|---|
| survey | Central Asia Barometer waves 1-6, face-to-face, spring 2017 to autumn 2019, 1,500 a wave |
| files | `data/raw/cab/CAB-Survey-Wave-{1..6}-*.zip`, the Uzbek `.dta` read out of each zip |
| columns | `Region_M` (4001-4014), `Religion_M`, `totwt` |
| population | National Statistics Committee, SIAT indicator 2.01.02.0001, 1 January 2026, **38,236,700** |
| boundaries | OCHA COD-AB `uzb_admbnda_adm1_2018b`, 14 units |
| placement | Kontur 400 m hexes, 2023-11 (`sources/uz_grid.py`) |
| modules | `sources/cab.py` (shared reader and tests), `sources/uz_geo.py`, `sources/uz_grid.py`, `sources/uz.py`, `taxonomy/uz2019.py`, node `other.uz` |

**The split-half resamples waves.** The design is settlements as PSUs, ten interviews each, in 27
region x urban/rural strata (wave 4 methods report, pp. 17-19). This build believed no cluster
column was released; `SamPt` is, and §8's PSU re-run kept every verdict.

**The decode has three witnesses.** SOATO `17NN` is COD's `UZNN` for all 14; the office's Russian
names equal COD's `ADM1_RU`; and the barometer's codes 4001-4014 run in exactly the order the office's
own table lists its region rows (Karakalpakstan first, Tashkent region eleventh, Tashkent city last),
with each label also name-matching its own pcode best among units of its kind (worst margin 0.198).
Held-out: survey respondent share against population share r = +0.995 over 14 units, best of 20,000
random pairings +0.918.

**Quota test: all 15 wave pairs compared**, worst pair 2 vs 6 at 5 of 23 free cells identical,
p = 0.848, 1.0 after Bonferroni. That needed `arabbarometer.quota_agreement` to take `waves=`:
before 2026-09-14 it only paired waves named in `WAVE_NAMES` and passed a non-Arab-Barometer frame
without comparing anything.

**Split-half, run 2026-09-14** (six waves, ten halvings, 2,000-draw per-wave permutation null,
spatial chi-square):

```
                                        14 regions                      5 DHS 1996 survey regions
    answer                        n   rho   null95      p   chi2 p      rho   null95      p
    Muslim                    8,532 +0.543 +0.361 0.0035   0          +0.900 +0.600 0.0040
    Christian                   251 +0.536 +0.358 0.0065   0          +0.937 +0.650 0.0010
    A non-believer               59 +0.624 +0.379 0.0025   1e-50      +0.900 +0.650 0.0025
    A believer of no part. faith 66 +0.404 +0.360 0.0335   2e-30      +0.900 +0.600 0.0025
    Other (vol.)                 18 +0.454 +0.442 0.0430   6e-21      +0.112 +0.688 0.4123
    Jewish                        4 -0.113 +0.734 1.0000   0.45       -0.395 +0.791 1.0000
    A believer of another faith   1   no test                          no test
```

What is drawn where:

- **Christian, non-believer, no particular faith: region share.**
- **Muslim: the residual.** It passes, but Andijan's 840 answers are all Muslim and a placed Muslim
  share leaves no room for the tail there. Sweden's rule (spec §12, nested units). Drawn Muslim
  differs from measured by at most 2.0 points, in Bukhara, and that is the refused `Other` below.
- **`Other (vol.)`: refused, national rate.** Eleven of its eighteen respondents are wave 4 in
  Bukhara, all Uzbek, one or two sampling points. Its largest (wave, region) cell is 61% of the
  answer; for every placed answer it is 22% or less. The chi-square cannot see a cluster. Written
  as `OVERRIDE` in `sources/uz.py`; spec §12 has the general rule.
- **Jewish, another faith: national rate.**
- **The coarse level placed nothing.** The DHS 1996 survey regions (Karakalpakstan and Khorezm;
  Navoi, Bukhara, Kashkadarya and Surkhandarya; Samarkand, Jizzakh, Syrdarya and Tashkent region;
  the three Fergana Valley oblasts; Tashkent City: DHS final report FR84, chapter 1) were used
  because they are a published design grouping. Nothing that fails at 14 passes there.

**As drawn:** Muslim 95.93%, Christian 2.41% (920,976), non-believer 0.72%, no particular faith
0.70%, other 0.17%, Jewish 0.06%. **Tashkent city holds 74.7% of the Christians** at 21.65%, with
6.49% non-believer and 5.08% no particular faith. Christian is zero in Andijan (n=840), Namangan
(640) and Khorezm (550); by the rule of three that still allows up to 0.36%, 0.47% and 0.55%.
Karakalpakstan's 1.86% no particular faith (12 respondents, spread over four waves) is the one
irreligion figure outside Tashkent worth noticing.

**The excluded 0.75%** (50 don't know, 19 refused, weighted) sits mostly in Khorezm (22 don't
know) and Samarkand (10 refused). Its §3.5 lean against the drawn Christian share is r = -0.20
(don't know) and -0.23 (refused); both regions are over 99% Muslim, so the exclusion moves nothing
visible.

**Not done, and worth a look by whoever returns:**

- **The adult-only lean.** Tashkent city has the most Christians and, very likely, the fewest
  children, which is Kyrgyzstan's -0.85 again; no regional age table was pulled to measure it.
- **The boundary vintage.** COD-AB is 2018 and Tashkent city took territory from Tashkent region in
  2020. Kontur against the office reads 0.57x for the city and 1.25x for the region, which is that
  line. It moves where some suburban dots sit inside Tashkent region, not either unit's totals.
- **Turkmenistan** (§11ao) can reuse `sources/cab.py` as it stands, with waves `[4, 5, 6]`.

Scouted 2026-09-08 against the §9cd instruction (an office that has ruled out a
cross-tabulation on the evidence of its *publications* has ruled out nothing). Kazakhstan's
route was applied here in full and came back empty for a different reason: **Uzbekistan does
not hold the cross-tabulation either.** No census here has ever asked religion, the office's
own queryable database has neither religion nor ethnicity in it, and the LiTS fallback the
queue points at cannot be used at ADM1.

| | |
|---|---|
| office | National Committee on Statistics (`stat.uz`), formerly the State Committee |
| census | **2026, the first since 1989** — enumerated 15 January 2026, fielded 15 Jan to 28 Feb |
| religion question | **none.** Question 12 is nationality, 13 native language, and Form 2 runs 1 to 23 |
| office database | `siat.stat.uz`, open and keyless, **3,231 indicators, none religion, none ethnicity** |
| dashboards | none. Every iframe on every stat.uz property is YouTube |
| LiTS III | has the religion question, **has no PSU in four of fourteen regions** |
| LiTS IV | covers all fourteen regions, **has no religion question** |
| what reopens it | the 2026 census's **nationality x region** table, unpublished; nationality was collected |

---

## 1. The census asks nationality and language and does not ask religion

Uzbekistan ran its first post-independence census from 15 January to 28 February 2026, 37
years after the Soviet one, with 15 January 2026 as the census moment. Preliminary results
were presented at a conference in Tashkent on 30 June 2026: 82.3% of the population completed
the online stage, post-enumeration coverage was 97.3%, and the population, sex and age by
region were released.

The instrument itself is public. `aholi.stat.uz` links, under *Технические условия*, the
National Committee's own **«Инструкция о порядке заполнения форм переписных вопросников
через сеть Интернет»** (75 pages, Tashkent 2025), cached at
`data/raw/uz/uz_census2026_instruction_ru.pdf`. It walks every question on all four forms.
Form 2, the individual questionnaire, runs 1 to 23:

```
     7 Сколько детей Вы родили?            15 Ваш уровень образования?
     8 Место рождения?                     16 Получали ли Вы образование?
     9 Проживаете с рождения?              17 Источники дохода
    10 Проживали ли за границей?           18 Занимались ли Вы работой?
    11 Ваше гражданство?                   19 Где осуществлялась основная работа?
    12 Ваша национальность?                20 Кем Вы являлись на основной работе?
    13 Ваш родной язык?                    21 Вид экономической деятельности
    14 На каких языках Вы можете           22 Где находится место основной работы?
       разговаривать?                      23 Искали ли Вы работу?
```

**«Вероисповедание» occurs exactly once in the whole 133,000-character document**, and it is
in the definition of refugee status under question 11a ("persecution on grounds of race,
nationality, religion, political conviction, sex..."). There is no religion item on any form.

This is a check on the questionnaire, not on a publication, which is the distinction §9cd
turns on. `[[reference_census_questionnaire]]`.

## 2. The office's database was queried, and it is the §9cd move coming back empty

`stat.uz`'s demography page links its figures not to files but to
`api.siat.stat.uz/media/uploads/sdmx/sdmx_data_<n>.{xlsx,csv,json,xml}`. That names **SIAT**,
the Committee's statistical information-analytical system, at `siat.stat.uz`. It is a Vue SPA
and its bundle names four routes (`[[reference_spa_hidden_apis]]`):

```
    https://siat.stat.uz/api/sdmx/              the indicator tree
    https://siat.stat.uz/api/sdmx/last-layer/   every leaf indicator, paginated
    https://siat.stat.uz/api/sdmx/search/       elasticsearch-backed
    https://siat.stat.uz/api/common/counters/   {"indicators":3231,"sdmxs":91,...}
```

No key, no login, no terms gate. `last-layer` returns **all 3,231 indicators** with names in
four languages (uz, uzc, ru, en); page it with `?page=<n>&size=500` until `next` is null, which
is seven requests. Cached at `data/raw/uz/uz_siat_indicators.json` (gitignored, so re-pull it).

Grepped over all four name fields for `вероиспов|религиозн|конфесси|diniy|e'tiqod|religio|
confession|faith`: **zero hits.** For `национальн(ый|ости)|этническ|millat|milliy tarkib|
ethnic|nationalit`: **two hits**, both "share of women/men playing national sports". For
`владение язык|language`: zero.

So the answer is not "BNS publishes it and I have not found the document". The office's own
database, queried directly, does not contain religion or ethnicity as an indicator at all.

## 3. There is no BI engine to ask

Fetched as **raw HTML** rather than through WebFetch, which strips iframes (§9cd):
`stat.uz` in all three languages, `aholi.stat.uz`, `siat.stat.uz`, `gender.stat.uz`,
`nsdp.stat.uz`, `gov.uz/ru/pages/2026` (the government's census page) and
`gov.uz/ru/religions` (the Committee for Religious Affairs). Every `<iframe src>` on all of
them is a YouTube embed. No Qlik, Power BI, Tableau, Superset, Metabase or ArcGIS anywhere.

> **The one `qlik` hit was the Uzbek word for openness.** A substring grep for BI vendors on
> the Uzbek-language pages matches **ochiqlik**, which is in the main menu twice. Anchor the
> pattern or read the match.

## 4. Three stat.uz subdomains are geo-fenced, and the tell is that every port is open

`census.stat.uz` (the online census system), `module.stat.uz` (electronic reporting) and
`hudud.stat.uz` (regional statistics) all resolve, all on `89.249.62.139/140`, a different
block from `stat.uz`'s `185.74.7.250`. From here every one of them accepts a TCP connection
on **80, 443, 3000, 5000, 8000, 8080, 8083, 8443 and 9000** and then resets the TLS
handshake. `curl` returns 35, `requests` returns `ConnectionResetError`, and WebFetch returns
`ECONNRESET` from a different continent.

**A host where a port scan says everything is open is not a host with nine services on it.**
It is a firewall answering every SYN and completing no handshake, which is what a geo-fence
looks like from outside. Wayback has `census.stat.uz` from 2021 and 2026 and it is a React
SPA for filling in the census form, not a results site, so nothing is lost here; the point is
the diagnostic. `data.egov.uz`, the national open-data portal linked from every stat.uz page,
is different again: it resolves to `195.158.28.138` and that host *refuses* TCP on 80 and 443,
which reads as retired rather than fenced.

## 5. LiTS closes, and it takes two rounds to see why

`queue.md` §B sends Uzbekistan to the EBRD Life in Transition Survey. Both rounds were pulled
and read (`data/raw/lits/lits_iii.dta`, 170 MB; `data/raw/lits/lits_iv_csv.zip`, 23.6 MB;
both open, no auth, no terms gate, exactly as §11ag found).

**LiTS III (2015-16) has the question and not the country.** `q922`, *Religion of the
respondent*, n = 1,506 for Uzbekistan: 1,440 Muslim, 46 Orthodox Christian, 10
atheist/agnostic/none, 5 other Christian, 3 Buddhist, 1 Jewish, 1 refusal. But its 75 PSUs
fall in **ten of the fourteen regions**, and the four with no PSU at all are

| region | population, 1 Jan 2024 |
|---|---:|
| Fergana | 4,061,500 |
| Kashkadarya | 3,560,600 |
| Andijan | 3,394,400 |
| Surkhandarya | 2,877,100 |
| **total** | **13,893,600 of 36,799,800 = 37.8%** |

Nothing about that is recoverable, and the weights do not tell you it is missing.
`weight_population` sums to **20,370,978** for Uzbekistan, and the four unsampled regions
contribute nothing to that total. Whether those weights are the ten regions' own populations
or the national adult total spread proportionally across the ten cannot be told apart, because
in Uzbekistan in 2016 those two quantities coincide to within 4%; in Kazakhstan, the Kyrgyz
Republic and Tajikistan the same variable sits at roughly 55% of each region's population, so
it is the adult reading there. Either way there is no observation of any kind in 37.8% of the
country, and the four are the worst to lose: the two southern provinces and two of the three
Fergana Valley provinces, the most rural and most observant part of Uzbekistan, against a
sample whose one distinctive cell is Tashkent city.

**LiTS IV (2022-23) has the country and not the question.** n = 1,006 across **all fourteen
regions** (Fergana 120, Samarkand 120, Kashkadarya 101, Surkhandarya 82, Andijan 81, Tashkent
city 81, ... Syrdarya 20). `queue.md` left it open whether LiTS IV kept the religion item.
**It did not.** Its 1,319 columns contain no religion variable. The only religious content is
the country-specific identity battery `q811`, where `q811522` / `q811523` / `q811524` are
*Uzbekistan - Muslim* / *Orthodox Christian* / *Other religious identity* as yes/no ticks
alongside *Uzbek*, *Tajik*, *Central Asian*, *Soviet* and eleven others. **274 of 1,006 tick
Muslim, 2 tick Orthodox, 1 ticks other religious.** That is a question about which identities
matter to you, not about what your religion is, and reading 27% as Uzbekistan's Muslim share
would be nonsense.

**So pooling cannot rescue it**, which was the queue's plan: the round that asks is missing a
third of the country and the round that covers the country does not ask.

### The split-half test, run anyway, on the ten regions LiTS III does cover

Split on PSUs rather than rows, 400 random halves, weighted shares, Spearman across the ten
regions (spec §9bi/§9bl):

```
    MUSLIM                       median rho  0.279   12% of halves negative
    ORTHODOX CHRISTIAN           median rho  0.585    1% of halves negative
    ATHEISTIC/AGNOSTIC/NONE      median rho -0.300   87% of halves negative
```

Orthodoxy is the only category with a stable geography, and inspecting it shows why: 28 of the
46 Orthodox respondents are in Tashkent city and 8 each in Bukhara and Tashkent oblast. The
signal is one city, which the split-half reproduces because one city is always in both halves.
Muslim share is noise around 98%, and non-belief is pure noise. Even setting the coverage hole
aside, this survey supports "Tashkent has Russians in it" and nothing finer.

### §3.5 lean check, leave-one-out

The excluded residual is `Refusal`, and it is **one respondent**, in Tashkent city. With a
single non-zero unit there is no correlation to compute: leave-one-out drops the only
informative row and every one of the ten leave-one-out correlations is either undefined or
identically zero. The honest reading is that LiTS III Uzbekistan has no measurable
non-response lean because it has no measurable non-response, and that is a fact about the
sample size rather than about Uzbekistan. Nothing is drawn from it either way.

## 6. What reopens Uzbekistan, and it is dated

**The 2026 census collected nationality by region and has not published it.** Question 12 went
to 39 million people at 15 January 2026 and the individual record carries region; the
July 2026 preliminary release covered population, sex and age only. The Committee's stated
plan, given at the 30 June 2026 conference, is final results in 2027; the presidential decree
UP-173 of 19 September 2025 sets the field dates and no publication deadline, so the year is
the Committee's word and not a statutory one.

When that table lands, Uzbekistan becomes buildable on the **§9aq pattern** — a `modelled`
country, religion spread across regions by ethnic composition — and it becomes buildable well,
for a reason specific to Uzbekistan. §9cd measured what that model gets wrong in Kazakhstan:
Islam and Orthodoxy were within 4.5% of the truth region by region, and the large errors were
all in the **refusal** (27.8% misplaced) and **non-belief** (22.9%) layers. Uzbekistan has no
refusal or non-belief layer to get wrong, because nobody asked. The model's proven failure
mode does not exist here and its proven strength is the whole of what Uzbekistan needs.

Fragments are already out and are not enough to build from. The Committee's July 2026 release
gave national composition (Uzbeks 34.9M / 89.4%, Tajiks 1.28M, Karakalpaks 841,400, Kazakhs
707,300, Russians 606,500, Turkmens 194,800) with the largest regional cells for four groups
only (Tajiks: Surkhandarya 273,700, Namangan 254,700, Fergana 210,200; Kazakhs: Tashkent
region 335,700, Karakalpakstan 236,600; Russians: Tashkent city 299,700, almost half the
national total). A fourteen-by-N matrix cannot be reconstructed from six numbers, and
assembling one out of press-release cells would be inventing the rest.

## 7. What was NOT tried, so the next session does not repeat it

- **OCR on the decrees.** `up-173-19_09_2025_p37893.pdf` (10 pp) and the Cabinet resolution
  `vm-arori-629-07_10_2025_p91662.pdf` (65 pp) are image-only scans; PyMuPDF gets 389 and 64
  characters. The Cabinet resolution is where a publication schedule would be, and it is
  worth OCR if the 2027 date matters to someone.
- **The regional statistical offices.** Each viloyat has its own site (`farstat.uz` for
  Fergana was seen in passing). If any of them publishes its own region's national
  composition, fourteen of those would assemble the matrix ahead of the national release.
  Untested.
- **The Committee for Religious Affairs** at `gov.uz/ru/religions` keeps the register of
  religious organisations, which is buildings and not people. `[[feedback_proxy_residual_nameable]]`
  rules that out as a proxy: the non-matching part is not a published number anyone can weight
  by.
- **`data.egov.uz`** refuses connections from here. If it answers from somewhere else, it is
  the one national catalogue that was not read.

## 8. Review, 2026-09-14 (session `f95259a4-uzrev`), tested from the raw files

- **The shared `arabbarometer.py` change is clean.** Egypt, Jordan and Iraq were rebuilt into a
  scratch directory with the current module: sha256 identical to `data/normalized/{eg,jo,iq}.csv`.
  On this country's frame the current code compares all 15 wave pairs (worst 2 vs 6, p = 0.848);
  git HEAD's code returns `(None, 0)` on the same frame; a copy of wave 2 planted as wave 7 stops
  at 5.7e-12 after Bonferroni; a CAB frame passed without `waves=` stops on the stray-wave check.
- **The public file does carry the sampling point.** `SamPt` ("Sample Point Number") is in all
  six Uzbek `.dta` files, never null: 902 PSUs, each inside one region, 5 to 10 interviews
  (median 10). `IntCode` (interviewer) is there too. §0, the docstrings of `cab.py`, `uz.py` and
  `_uz_counts`, and the `note` field in `countries.py` all say there is no cluster column; they
  need correcting whenever this country is next touched.
- **Re-run on PSUs, every verdict holds.** 60 halvings of PSUs stratified by wave and region;
  the null shuffles whole PSUs across regions within each wave (150 draws for rho, 300 for the
  chi-square, so 0.0033 is the floor):

```
    answer                      median rho   null 95th      p   PSU-shuffle chi2 p
    Muslim                          +0.675      +0.375  0.007   0.0033
    Christian                       +0.492      +0.383  0.013   0.0033
    A non-believer                  +0.579      +0.364  0.007   0.0033
    A believer of no part. faith    +0.510      +0.336  0.013   0.0033
    Other (vol.)                    +0.494      +0.387  0.013   0.0033
    Jewish                          +0.500      +0.500  0.113   0.40
```

  So the wave split reached the same answers a PSU split does. Only 6 of 152 `SamPt` numbers
  keep one region in all six waves, and the wave 4 methods report has points drawn and replaced
  per wave (12 replaced in Uzbekistan), so the waves are not a fixed panel of settlements.
- **`Other (vol.)` stays refused, for a different reason than the one written.** Its 18
  respondents are in 12 PSUs, the largest holding 3. Bukhara's eleven in wave 4 are in five
  sampling points (19, 20, 22, 23, 24), not one or two, and **all eleven were recorded by
  interviewer 4**; the other interviewer in Bukhara that wave recorded none in 46 interviews.
  That is an interviewer effect, which neither a wave split nor a PSU split can see. `OVERRIDE`
  in `uz.py` should say so.
- **What shows on the map at Tashkent is a Kontur artefact, not the boundary.** The raw
  `kontur_population_UZ_20231101.gpkg` has 73 hexes over 15,000 people (about 110,000 per km2
  at 400 m), most of them at 40,66x, in one block at roughly 69.19-69.35 E, 41.12-41.23 N, south
  of the centre. 36 are inside UZ26 and hold **1,033,223 of its 1,813,965 Kontur people, 57.0%**;
  37 are in UZ27 and hold 24.5% of Tashkent region. Hexes at Amir Timur square hold about 1,400
  each. A headless shot at city zoom shows it: a dense patch of dots south of the city, yellow
  mixed through it, with the centre thinly covered. On the weights that is about 390,000 of the
  688,175 Christians drawn in the city. The fix is a cap or smoothing on hex weight before the
  next scatter. `kg_hexes.gpkg` has 46 hexes over 15,000 holding 13.2% of Kyrgyzstan's weight and
  deserves the same look; `kz_hexes.gpkg` has none.
- **The boundary vintage is small.** SIAT 2.01.02.0001: Zangiata and Kibray districts fall by
  29.4k and 29.1k from 2021 to 2022, Tashkent region falls 34.4k that year against 34-50k of growth
  in the years around it, and Tashkent city rises 166.2k against 41-62k a year before 2020. About
  100,000 people, 3% of the city. At the city-region Christian gap (21.65% against 4.17%) that is
  about 17,000 Christians drawn a few kilometres on the region side. §0's reading of Kontur's
  0.57x and 1.25x as "that line" is wrong; the artefact above is most of it.
- **Tashkent city's 21.7% Christian is likely too high for the 2026 population it is applied
  to.** In the survey the city is 21.65% Christian (95% PSU bootstrap 16.8-27.4%) and 22.96%
  Russian by `Ethnic_M` (17.9-28.5%); 83% of its Russian respondents answer Christian, and
  Russians supply 17.8 of the 21.65 points. §6 gives the 2026 census's Russians in the city as
  299,700 of 3,178,100, 9.4%; nationally the survey is 2.39% Russian against the census's 1.59%.
  The city's gap (2.4x) is larger than the national one (1.5x). With the Russian part scaled to the
  census the city is about 11.7% Christian, roughly 370,000 people against 688,175 drawn. Either
  the city changed that much since 2017-19 or the sample leans Russian there; both push the same
  way. Wave 1 had 60 city respondents in 6 PSUs at 39.5% Christian and lifts the pooled share by
  3 to 4 points. The census figures are §6's press-release numbers and are not in any of the
  three cached `uz_census2026_prelim_*.xlsx`, so they were not re-checked here.
- **The hard zeros in Andijan, Namangan and Khorezm are right to draw.** Precedent draws a zero
  on a placed survey share in `eg` (Red Sea), `jo` (Jerash, Tafilah), `iq` (Najaf Sunni; Duhok and
  Erbil Shia) and `kg` (Batken and Osh Orthodox). The three regions have one Russian respondent
  between them (Andijan, who answered Muslim), so the zero agrees with who was interviewed, and
  the rule-of-three ceilings come to 11,000 to 15,000 people a region, 11 to 15 dots at 1:1,000
  spread over two to three million. `note_public` says it plainly.
- Checks clean: `check_md.py`, `built_countries.py --check`, `check_rollup.py uz`,
  `review_dump.py uz` (the seven mappings follow `al`, `cr`, `ru` and `kz`; `other.uz` is the
  usual per-country node). The country-wide screenshot is clean.
- **Edited:** `note_public` loses "From 2020 the survey was run by telephone and stopped asking
  Uzbeks about religion.", which left "to say that" pointing at the telephone switch rather
  than at the survey. Not pushed with `--refresh-meta`, because other builders may hold the lock.

## 9. Redrawn by ethnic group, 2026-09-14 (session `f95259a4-uz2`)

Anita's ruling on §8: where a census counts a group whose religion differs sharply, the map draws
by group, as the citizenship splits do (`be`, `se`, `no`, `dk`) and Latvia's non-citizens
(`lv.md` §4). So religion within ethnic group comes from the survey, and each group's people per
region from the 2026 census. Kontur placement was not touched; that fix waits on Anita.

### The table is published, in the office's compilation PDF

| | |
|---|---|
| document | *Preliminary Results of the Population and Agriculture Census of the Republic of Uzbekistan, 2026*, National Statistics Committee, Tashkent 2026, 94 pp., English edition |
| URL | `https://stat.uz/img/news/english_natija_merged-2_p42445.pdf` (curl exit 60 on the certificate; `-k` works) |
| cached | `data/raw/uz/uz_census2026_results_en.pdf`, 2,305,677 bytes, 89 PDF pages, `%%EOF` present, text layer on every table page |
| tables | printed p. 62 *Ethnic composition of the population, by region* (PDF p. 64); printed pp. 63-70 *by ethnic composition and sex, by region* (PDF pp. 65-72) |
| groups | eight: Uzbeks, Karakalpaks, Kazakhs, Tajiks, Kyrgyz, Russians, Turkmens, Other |
| census moment | 15 January 2026, **39,047,321** |

`sources/uz.py` parses both tables from the text layer. They agree on all 15 rows by 9 columns,
every group's males and females sum to the group, every region's groups to its total, and the 14
regions to the national row.

- **Tashkent city: 3,224,838 people, 299,725 Russians, 9.29%.** §8's 9.4% is the same Russian
  count over SIAT's 3,178,100. Every regional cell §6 quoted from the press also matches (Tajiks
  273,695 / 254,723 / 210,203; Kazakhs 335,706 / 236,563).
- **The three `uz_census2026_prelim_*.xlsx` cached earlier are the agriculture volume** (sheets
  1.1 to 1.11, holdings and livestock), which is why §8 found no population tables in them.
  `aholi.stat.uz` lists a separate xlsx of the results (item 6258); not pulled, the PDF's two
  tables witness each other.
- **Census over SIAT's 1 January 2026 estimate:** 0.97 to 1.07 per region, except Tashkent region
  at 1.191 (3,763,093 against 3,160,700), which is most of the 810,617 the office said its
  estimates had missed. The population base is now the census.
- **How it was found:** WebSearch in Russian, English and Uzbek; kun.uz, gazeta.uz (ru, en),
  dunyo.info/Kazinform, UzMetronom and both Wikipedias gave national figures only. The Times of
  Central Asia's article links the PDF.

### The groups, sized to what both sides carry

Survey `Ethnic_M` (*What is your ethnic identity?*), one label set in all six waves: Uzbek 7,651,
Karakalpak 297, Tajik 291, Russian 261, Kazakh 165, Other (vol.) 130, Tatar 110, Kyrgyz 44, don't
know 41, refused 10. There is no Turkmen, Korean, Ukrainian or Armenian code.

| group | census rows | survey answers | answered n |
|---|---|---|---:|
| central | Uzbeks, Karakalpaks, Kazakhs, Tajiks, Kyrgyz, Turkmens | Uzbek, Karakalpak, Kazakh, Tajik, Kyrgyz | 8,391 |
| russian | Russians | Russian | 261 |
| other | Other | Tatar, Other (vol.) | 235 |

- **"Other Slavic" cannot be a group.** The census has no Ukrainian or Belarusian row; they are in
  Other.
- **Turkmens do not line up.** The census counts them; the survey has no code, so its Turkmens
  answer Other (vol.): Karakalpakstan's 37, 35 of them Muslim. Census Turkmens go to `central`.
  The survey's stay in `other` and lean its rest-of-country mix Muslim: 88.1% against 84.8%
  without Karakalpakstan and Khorezm, Christian 7.4% against 10.2%, about 5,000 to 6,000 people
  over 185,696.
- 44 answered respondents gave no ethnicity and sit out of the within-group shares (0.33%
  weighted).

Survey against census, weighted group share:

| | survey nat. | census nat. | survey city | census city | survey Tashkent reg. | census Tashkent reg. |
|---|---:|---:|---:|---:|---:|---:|
| central | 94.80% | 97.68% | 63.04% | 87.16% | 88.60% | 94.13% |
| russian | 2.42% | 1.55% | 23.05% | 9.29% | 4.05% | 3.95% |
| other | 2.78% | 0.77% | 13.91% | 3.55% | 7.35% | 1.92% |

The overshoot is the capital's minorities, not Russians outside it.

### Religion within group, and what carries geography

Weighted within-group mixes: central Muslim 99.10%, no particular faith 0.29%, Christian 0.20%,
non-believer 0.20%; russian Christian 73.98%, non-believer 10.62%, Muslim 7.76%, no particular
faith 7.41%; other Muslim 69.88%, Christian 12.51%, non-believer 8.92%, no particular faith 7.80%.

**central** takes the first build's construction: split-half on waves plus spatial chi-square.

```
    14 regions                         n   median rho  null 95th       p    chi2 p
    Muslim                         8,309       +0.645     +0.353  0.0005   7e-22   KEPT AS RESIDUAL
    A believer of no part. faith      29       +0.462     +0.360  0.0145   5e-09   region share
    Christian                         17       +0.421     +0.395  0.0425   1e-04   region share
    A non-believer                    19       +0.567     +0.373  0.0095   4e-12   region share
    Other (vol.)                      14       +0.411     +0.500  0.1849   2e-26   national rate
    Jewish                             3                                            national rate
```

At the five DHS 1996 survey regions the same four pass and `Other (vol.)` fails, so the
mixed-level construction places nothing only there. Central Christian is the likeliest false pass
(17 respondents, p = 0.0425); it is 0.20% of the group.

**russian and other cannot take a split-half.** 56 of 84 (wave, region) cells hold no Russian and
37 of 84 no `other` respondent; at the DHS regions 11 of 30 and 1 of 30. They are tested on two
units, **Tashkent city against the rest of the country**: the city flag shuffled across whole
sampling points (`SamPt`) within each wave, 2,000 draws, on the weighted share, with the
chi-square as a veto. A passing answer takes the city's or the rest's share; the rest of the group
shares the remainder at the group's national proportions, its largest answer included.

```
    russian, city 199 / rest 62      city    rest   perm p   chi2 p
    Christian                       77.7%   63.3%   0.069    0.033    THE GROUP'S RESIDUAL
    A non-believer                  11.3%    8.6%   0.60     0.92     national rate
    Muslim                           1.3%   26.4%   0.0005   1e-07    city or rest
    A believer of no part. faith     9.4%    1.7%   0.14     0.86     national rate

    other, city 94 / rest 141        city    rest   perm p   chi2 p
    Muslim                          41.4%   88.1%   0.0005   2e-10    THE GROUP'S RESIDUAL
    Christian                       20.5%    7.4%   0.034    2e-04    city or rest
    A non-believer                  19.2%    2.3%   0.0005   0.030    city or rest
    A believer of no part. faith    18.2%    1.2%   0.0005   3e-04    city or rest
```

Russians outside the city answer Muslim at 26.4% (11 of 62: Tashkent region 5, Bukhara 2,
Surkhandarya 2, Andijan 1, Samarkand 1). Drawn as found.

**Against spec §14.25.** Kazakhstan's ethnicity model misplaced non-belief by 22.9% because it
spread the attitude layers by ancestry alone. Here only the group mix comes from ancestry; the
non-belief and no-particular-faith answers keep the survey's own geography (central at the region,
the minorities city against rest), so that failure is not built in. It is not measured either.

Checks unchanged from the first build: quota test all 15 wave pairs (worst 2 vs 6, p = 0.848);
held-out r = +0.994 against the census counts, best random pairing +0.919; 902 sampling points,
each inside one region. §3.5 lean: don't know against Christian r = -0.20, refused -0.24.

### Result

National: Muslim 97.47%, **Christian 1.40% (545,878)**, no particular faith 0.46%, non-believer
0.46%, other 0.15%, Jewish 0.06%. First build: Christian 2.41% (920,976).

| region | census | Russian | Muslim | Christian | non-believer | no part. faith | first build Christian |
|---|---:|---:|---:|---:|---:|---:|---:|
| Tashkent city | 3,224,838 | 9.29% | 85.06% | **9.32%** | 3.44% | 1.95% | 21.65% |
| Tashkent region | 3,763,093 | 3.95% | 95.83% | 3.24% | 0.38% | 0.33% | 4.17% |
| Navoi | 1,184,591 | 2.29% | 97.57% | 1.39% | 0.21% | 0.63% | 0.29% |
| Syrdarya | 954,361 | 1.61% | 97.91% | 1.29% | 0.47% | 0.11% | 1.95% |
| Bukhara | 2,104,874 | 0.57% | 98.64% | 0.71% | 0.06% | 0.38% | 0.34% |
| Samarkand | 4,404,575 | 0.64% | 98.73% | 0.50% | 0.07% | 0.49% | 0.17% |
| Karakalpakstan | 2,149,932 | 0.35% | 96.80% | 0.38% | 0.68% | 1.94% | 0.14% |
| Fergana | 4,238,911 | 0.59% | 99.33% | 0.37% | 0.06% | 0.04% | 0.55% |
| Surkhandarya | 2,984,084 | 0.27% | 99.21% | 0.35% | 0.21% | 0.02% | 0.44% |
| Jizzakh | 1,524,055 | 0.43% | 99.11% | 0.28% | 0.04% | 0.35% | 0.97% |
| Khorezm | 2,140,746 | 0.26% | 99.29% | 0.16% | 0.03% | 0.31% | 0.00% |
| Andijan | 3,531,777 | 0.22% | 99.58% | 0.16% | 0.03% | 0.02% | 0.00% |
| Namangan | 3,149,161 | 0.23% | 99.43% | 0.15% | 0.14% | 0.06% | 0.00% |
| Kashkadarya | 3,692,323 | 0.21% | 99.09% | 0.14% | 0.33% | 0.23% | 0.27% |

- **Tashkent city: 300,678 Christians, 55.1% of the country's**, against 688,175 and 74.7%. By
  group: Russians 237,246, central 40,009, other 23,423.
- **The hard zeros in Andijan, Namangan and Khorezm are gone.** Each region's census Russians and
  others now carry the rest-of-country rates. That is the construction, not a new measurement.
- **Navoi rises from 0.29% to 1.39%**: the census counts 27,127 Russians there (2.29%) and the
  survey met two.
- Karakalpakstan's 1.94% no particular faith is the central group's region share, as before.

### Not done

- Kontur placement (the dense block south of the city, §8) is untouched; it waits on Anita.
- The adult-only lean is still unmeasured: the survey is adults, the census counts all ages.
- The census's mother-tongue-by-region table (printed p. 71) would witness the Russian share; not
  used.

## 10. Tashkent's block at Kontur's density cap, capped 2026-09-14 (session `f95259a4-kontur`)

§8's patch south of the city is Kontur's density limit. Kontur caps every hex at 46,200
people/km², and a block at that limit is either a real core or a false concentration (spec §12,
"KONTUR'S DENSITY CAP"). Tashkent's block, as `kontur_cap.py` defines it (contiguous hexes at
15,000/km² or more), is **54 hexes, 11 at the limit, holding 1,535,105 people across the city
line**, with its peak 15.6 km south of the centre, where Kontur is 2,432/km². Its city side held
**58.0% of Tashkent city's placement weight**. A 4-hex piece on the region side of the line held
112,217 people, 2.8% of Tashkent region. (§8's 36 hexes and 57.0% count hexes over 15,000 people
inside UZ26; the definitions differ, and the finding does not.)

Both are `capped` in `kontur_cap.csv`, and `scatter.py` now lowers each hex to the median density
of the populated hexes within 3 km: 2,010/km² for the city block, which leaves its city side 8.1%
of the city's weight, and 7,497/km² for the region piece, 0.7% of the region.

Re-scattered on the counts from session `f95259a4-uz2`'s ethnicity reweight. On those counts a
capped and an uncapped run draw identical dots per node, 39,043 at 1:1,000, so the cap moved
placement and nothing else.

§8's question about Kyrgyzstan is answered and `kg` is unchanged: Bishkek's dome peaks at
26,781/km², below the limit, 1.5 km from the centre, where Kontur is at that same density. It is
a real core drawn a little steeply, not this artefact.

**Checked after the build, 2026-09-14 (session `f95259a4-kontur`).** Of the 3,092 dots inside
Tashkent city's polygon, 256 (8.3%) now fall in the capped hexes, where the weights before put
58%, and the city's weight within 6 km of Amir Timur square rose from 9.7% to 21.2%. A headless
shot at zoom 10.4 shows dots across the city's street grid, where before the centre was thinly
covered. **A dense patch south of the city is still visible, and it is not the block.** It is
Kontur's wider surface there: 13 more blocks at 15,000/km² or more within 15 km, all below the
cap and mostly single hexes (the largest, 8 hexes at 38,377/km², is on the region side), and a
ramp of hexes at 10,000-15,000/km². Capping all 13 would move the city's weight in the box
69.15-69.36 E, 41.10-41.21 N only from 29.3% to 27.4% (the region's from 23.7% to 15.4%), so they
were not registered. Kontur is thin across central Tashkent as a whole, and half the city's weight
still sits 10 km or more from the centre; that is the grid, and no block rule reaches it.
