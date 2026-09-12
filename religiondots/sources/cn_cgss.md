# cn_cgss — Chinese General Social Survey, self-identified religion by province

**What this is for.** spec §14.15 ranked "CLDS, then pooled CGSS" as the only route to a Han
layer that could sit beside the rest of the map, and §14.13 rejected CGSS 2021 alone (19
provinces of 31, per-province Buddhist cell of 1 to 55). This is the mirror hunt §6b asks for,
and the pooling test. **Three waves found openly, 32,495 respondents over 29 provinces.**

Basis is `self_id` throughout — CGSS asks *您的宗教信仰是什么* (which religion do you belong
to), which is §3.1's `self_id` and the same question Vietnam's census, Korea's and Russia's
Arena answer. This is the point of chasing CGSS rather than CFPS: CFPS asks about *belief in
Buddha or a bodhisattva* and returns 33% where CGSS returns 4%, and §3.1 forbids mixing.

## The three waves, and the exact re-fetch

| wave | n | provinces | licence | file |
|---|---|---|---|---|
| CGSS 2012 | 11,765 | 29 | **CC0**, Harvard Dataverse | `cgss2012_14.dta`, 57.3 MB |
| CGSS 2017 | 12,582 | 28 | **CC0**, Harvard Dataverse | `cgss2017.dta`, 48.2 MB |
| CGSS 2021 | 8,148 | 19 | CC BY 4.0, figshare | `CGSS2021.dta`, 8.0 MB |

```
curl -L -o data/raw/cn/cgss/cgss2017.dta \
  "https://dataverse.harvard.edu/api/access/datafile/5414235?format=original"
curl -L -o data/raw/cn/cgss/cgss2012_14.dta \
  "https://dataverse.harvard.edu/api/access/datafile/5408954?format=original"
curl -L -o data/raw/cn/cgss/CGSS2021.dta \
  "https://ndownloader.figshare.com/files/57617965"
```

- 2017 — `doi:10.7910/DVN/SZUSBS`. 2012 — `doi:10.7910/DVN/R1UDF2`. Both sit loose in the root
  Harvard Dataverse rather than in a collection, which is why a collection crawl misses them;
  `api/search?q=title:CGSS` finds exactly these two and nothing else.
- 2021 — figshare `10.6084/m9.figshare.30041488`. **The CC BY is the uploader's claim, not
  CNSDA's**, per §6b's caveat. Cite CNSDA as the origin.
- `?format=original` matters on Dataverse: without it you get the ingested `.tab` and lose every
  value label, including the province names.

## Variables

| | 2012 | 2017 | 2021 |
|---|---|---|---|
| province of interview | `s41` | `s41` | `provinces` |
| religion | `a501`+`a511`–`a521` | `a51`+`a511`–`a521` | `A5` |
| weight | `weight` | `weight` | `weight_raking` (2015 mini-census raking) |

**2012 and 2017 ask religion as a MULTI-SELECT block** (您的宗教信仰（多选）), one binary per
religion; 2021 asks a single-choice `A5`. That is a §3.1a answer-set difference and it was
measured rather than assumed: **of respondents naming any religion in 2012/2017, only 2.5% name
more than one** (73 of 2,961 name two, 2 name three). Collapsing the multi-select to a single
category is therefore safe and the three waves are comparable on that axis.

The answer set is richer than Pew's summary suggests: 佛教, 道教, 民间信仰（拜妈祖、关公等）,
回教/伊斯兰教, 天主教, 基督教, 东正教, 其他基督教, 犹太教, 印度教, 其他, 不信仰宗教. Orthodox,
Judaism and Hinduism return 0–1 respondents in 32,495 and are not usable.

## THE LEVELS MOVE MONOTONICALLY DOWN AND IT IS NOT A ROUNDING ARTEFACT

Weighted national shares, % of adults:

| | 2012 | 2017 | 2021 |
|---|---|---|---|
| **any religion** | **14.47** | **10.61** | **7.50** |
| Buddhism | 6.02 | 4.66 | 3.76 |
| folk | 3.43 | 2.11 | **0.27** |
| Islam | 2.56 | 2.23 | 1.87 |
| Protestantism | 2.31 | 1.41 | 1.03 |
| Daoism | 0.26 | 0.23 | 0.19 |
| Catholicism | 0.15 | 0.18 | 0.34 |

**Every category except Catholicism falls, and Islam falls 27% in a population whose Muslim
nationalities grew.** Two mechanisms and they are not exclusive: a real decline in willingness
to report a religion across the 2018 Regulations on Religious Affairs and the sinicization
campaign; and a multi-select-to-single-choice instrument change at 2021, which reliably lowers
affirmatives. **The 2012→2017 fall happens with the instrument held constant**, so the
instrument cannot explain all of it.

**Consequence for any layer built on this: pooling buys precision and spends currency.** The
pooled share is an average over a moving target and is not "China in year X".

## What survives the §14.10 test, and what does not

Pooled, 32,495 respondents, 29 provinces. `province_shares.csv` in this directory carries the
per-province weighted shares (pooled and 2021-only) beside each province's census population.

| | respondents | provinces with cell <10 | rank stability 2012↔2021 | verdict |
|---|---|---|---|---|
| **Buddhism** | 1,592 | **3 / 29** | **+0.63** | **drawable** |
| Protestantism | 585 | 13 / 29 | **+0.17** | marginal |
| Islam | 698 | 21 / 29 | +0.64 | **do not draw — see below** |
| folk | 681 | 18 / 29 | +0.45 | no |
| Daoism | 80 | 28 / 29 | +0.52 | no |
| Catholicism | 65 | **29 / 29** | +0.16 | no |

**Buddhism passes cleanly.** χ² homogeneity across provinces p = 4×10⁻¹⁸⁴; Zhejiang 15.7%
(CI 14.0–17.5) against Anhui 0.9% (0.4–1.4), non-overlapping by a wide margin. The pattern is
the southeastern coastal belt the literature describes — Zhejiang 14.8%, Fujian 11.5%,
Jiangxi 9.0%, Shanghai 7.0% — against Shanxi 1.0%, Shandong 1.1%, Anhui 1.1%, Chongqing 1.4%.
**This is a gradient, not sampling noise**, which is exactly what §14.13 said the 2021 wave
alone could not deliver.

**Protestantism is the open call.** Henan at 6.4% (CI 5.2–7.6) is China's Protestant heartland
and the data finds it unaided, with Heilongjiang, Jiangsu, Zhejiang and Jilin behind it. But 13
provinces have fewer than 10 Protestant respondents and the 2012↔2021 rank correlation is
**+0.17** — the geographic pattern is not stable across the nine years, where Buddhism's is.
§14.10's fifth condition is not met.

### ISLAM MUST STAY ON THE ETHNIC DERIVATION, AND CGSS IS THE EVIDENCE FOR THAT

The single most useful finding here. CGSS's provincial subsamples are drawn from a handful of
PSUs, and where a minority is spatially concentrated *within* a province the sample lands on it
or misses it entirely:

| | census, Muslim nationalities | CGSS pooled, self-id Islam | ratio |
|---|---|---|---|
| Ningxia | 34.5% | **90.4%** | 2.6× |
| Xinjiang | 58.3% | **92.0%** | 1.6× |
| Yunnan | 1.5% | **10.4%** | **7×** |

**A weight cannot repair a sample drawn from the wrong places.** §14.5's county-level ethnic
derivation is straightforwardly better here and stays. Note this cuts the other way too, as a
caution on Buddhism and Protestantism: they are more evenly spread within a province so the
effect is far smaller, but it is not zero, and it is the main residual risk in the layer.

**Nationally, though, CGSS and the derivation agree**, and this is the first external check
§14.5 has ever had: CGSS self-id Islam is 1.87% (2021) to 2.56% (2012), and the derivation's
23.07M over 1.259bn is **1.83%**. The check is valid at national level and only there.

## Magnitudes, if drawn

29 of 31 provinces, **99.2% of China's population** (only Hainan and Tibet uncovered; CGSS has
never sampled Tibet, and §14.5 already draws it from ethnicity).

| | pooled level | 2021 level |
|---|---|---|
| Mahayana Buddhists | **58.4M** (58,393 dots) | ~32.7M |
| Protestants | **21.3M** (+0.88M §14.9 mission peoples) | ~11.0M |

For scale, **all colour on China before this layer was 31.5M** (30.67M derived + 0.88M
modelled). The Buddhist layer alone is nearly twice everything that was drawn before it.

**These figures are ~5% larger than this layer first drew, because of spec §14.17 the same
day.** The county join was repaired — 168 census counties holding 67.8M people had been read
out of the volumes and discarded — and since these shares are applied to the `unknown` residual,
a bigger residual draws more. That also removes a bias this layer could not have seen: **142 of
the 168 missing counties were urban districts (市辖区)**, so a provincial share was being applied
to a denominator missing 5.5% of the province, concentrated in its cities. China's drawn
population went from 1,259,316,206 to 1,332,810,852, within 17 people of the published census.

The 2× spread between the two columns is the levels problem above, stated as dots. The obvious
resolution is the one China already uses for ethnicity under §3.4 — **shape from the pooled
waves, level from the most recent** — but that is a decision, not a default, and it is not
taken here.

## Waves and sources checked and NOT obtained

Per [[feedback_nothing_is_truly_dead]], what was searched, so it is not re-searched:

- **CLDS — FOUND, and blocked by a login.** Science Data Bank
  `doi:10.57760/sciencedb.02333`, "China Labor-force Dynamic Survey", 77.5 MB, **advertised
  CC BY 4.0 and `conditionsOfAccess: PUBLIC`**, description covering the 2011 Guangdong pilot
  and the 2012, 2014, 2016 and 2018 waves. Deposited by a third party at Hebei University in
  2022, not by SYSU. The listing API
  (`/api/sdb-dataset-service/dataset/details/<id>`, found by grepping the Nuxt bundles per
  [[reference_spa_hidden_apis]]) answers **`70001 无访问权限 / PERMISSION NO ACCESS`**. A free
  ScienceDB account is the obvious next step and is email-only — unlike CNSDA it does not want a
  Chinese ID. Landing page:
  `https://www.scidb.cn/en/detail?dataSetId=36d6d9d24afc4bc8a5ce0da5feaf22ff`
  Note CLDS's own use agreement forbids third-party redistribution, so §6b's caveat applies with
  more force than usual: cite SYSU CSS, not the depositor.
- **CGSS 2006 — CC0 and unrestricted, but the server is broken.** PKU Open Research Data,
  `doi:10.18170/DVN/21HKLB`, `CGSS2006.tab` 26.3 MB, `restricted: false`. Every access route
  returns 500 (`/api/access/datafile/1421`, with and without `format=original`, with a browser
  UA) or 403 (dataset-level zip). Metadata API works fine, so it is the file server specifically.
  Worth one retry later: `https://opendata.pku.edu.cn/dataset.xhtml?persistentId=doi:10.18170/DVN/21HKLB`
- **CGSS 2003 — Borealis (Canadian Dataverse), `doi:10.5683/SP3/IP5MPK`.** Not fetched: 2003 is
  urban-only and ~5,900 respondents, so it adds the least and differs the most.
- **The PLOS ONE supplementary §14.13 flagged as an unchecked lead — DEAD, and now checked.**
  `pone.0318221.s001.sav`, 22.2 MB, CC BY on figshare
  (`10.1371/journal.pone.0318221.s001`). It is a **single** CGSS wave, not multi-wave, and it
  **carries no religion variable at all** — the only match for "religio" in the whole file is
  the ISCO occupation code *religious professionals*. Its A-block is demographics
  (出生年月, 户口状况, 民族, 教育). Do not re-open this lead.
- **Searched and empty:** Zenodo (both surveys, exact-phrase and Chinese-name queries); Harvard
  Dataverse file-level search for `clds` (82 files, none CLDS); PKU for CLDS; ICPSR holds only
  the EASS cross-national sets, not CGSS waves. Replication packages on Harvard carry
  `CGSS2010_all_pool.tab` and `CGSS2006_all_pool.tab` (Chen & Zhan) at ~1 MB — variable subsets,
  not full waves, and not checked for the religion column.
- **CNSDA (`cnsda.org`) and `cgss.ruc.edu.cn`** are the official archives and want a free
  account. Not attempted, per [[feedback_gated_data_last_resort]] — but note this is an
  ordinary email registration, not the Korean-ID wall, so it is a cheap ask rather than a dead
  end. **It is NOT the only route to 2010, 2011 and 2013 — that sentence was wrong and is
  corrected below.** It remains the only route to 2015, 2018, 2023 and the household files.

## 2010, 2011 AND 2013 ARE OPEN AFTER ALL — FOUND 2026-09-08

Anita registered at CNSDA and hit a wall the free account does not clear: **every download
needs a separate data APPLICATION**, reviewed, which for an unaffiliated applicant abroad is
the shape of thing that already refused this project once (see [[reference_cfps_terms]]).

So §6b's mirror rule got one more run, and went five for five. **Three of the walled waves are
sitting unrestricted in a replication package**: `doi:10.7910/DVN/R1S5RP`, *"Meritocracy as
Authoritarian Co-Optation"*, which carries `cgss2010.tab`, `cgss2011.tab`, `cgss2013.tab` and a
second copy of 2012. `sources/cn_cgss_fetch.py` pulls them and its docstring is the record.

**These are near-complete waves, not the variable subsets a replication package usually is** —
963, 592 and 650 variables — and all three keep the two columns that matter, which is what had
to be checked rather than hoped:

| | n | provinces | religion variable | weight |
|---|---|---|---|---|
| 2010 | 11,783 | **31** | `a5`, single choice, 8 categories | `WEIGHT` |
| 2011 | 5,620 | 26 | `a501` + `a511`–`a521`, multi-select | **none** |
| 2013 | 11,438 | 28 | `a501` + `a511`–`a521`, multi-select | **none** |

**CGSS 2010 IS THE ONLY WAVE ANYWHERE HERE THAT COVERS ALL 31 PROVINCES.** It has Hainan (100
respondents) and Xizang (79), the two this file has always listed as uncovered. Tibet's cell is
far too thin to draw and §14.5 draws it from ethnicity anyway; Hainan's is thin but is the only
survey reading that province has ever had.

Pooling all six waves would take the sample from **32,495 to 61,336**, and roughly double both
drawn categories: Buddhism 1,592 → ~3,133 respondents, Protestantism **585 → ~1,145**. That is
aimed squarely at the thing §14.10 called marginal, which was 13 of 29 provinces holding fewer
than ten Protestant respondents.

**Three things to settle before any of that is drawn**, and they are why it has not been:

- **2011 and 2013 carry no weight column in these copies.** Unweighted provincial shares are
  biased by the sample design, and this project has never drawn one. Either find the weights,
  or draw those waves unweighted and say so, or use them for the cross-wave stability test only.
- **2013's multi-select overlap is 4.0% of religious respondents**, against the 5% ceiling
  `cn_cgss.py::check` enforces for collapsing multi-select to one category. Inside the rule, but
  it is the closest any wave has come.
- **Adding three older waves moves the pooled level UP**, because reported religiosity falls
  across the whole period. Pooling six waves centred on ~2014 is a different object from pooling
  three centred on ~2017, and it makes the unmade levels decision (§14.20) sharper rather than
  softer.

Cite **CNSDA / Renmin University** as the origin, never the replication package. Its own terms
are *"not to be distributed/posted outside of the Harvard Dataverse"*, which is a
no-redistribution clause and not a no-use one; `cn_cgss_fetch.py` downloads from Harvard
directly, as it asks.
