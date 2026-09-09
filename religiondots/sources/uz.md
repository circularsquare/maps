# Uzbekistan — NOT DRAWN, closed 2026-09-08. Nobody asks, and the one survey that asks is missing 37.8% of the country

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
