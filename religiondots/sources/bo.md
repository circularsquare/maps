# Bolivia — LAPOP AmericasBarometer single-country files, waves 2010–2023

Built 2026-09-14 by session `f95259a4-bo`, from `sources.md` §11ap's scouting. Write-up in
`sources.md` §9dm. Code: `sources/bo.py` (build), `sources/bo_census.py` (the 2024 count and 1992's
religion item off INE's REDATAM), `sources/bo_geo.py` (boundaries), `sources/bo_grid.py` (Kontur
hexes), `sources/bo_checks.py` (1992 against the survey), `taxonomy/bo2023.py` (mapping),
`countries.py::_bo_counts`.

## 1. Why a survey

No Bolivian census since 1992 has asked religion. §11ap read the 2024 questionnaire (59 questions,
none) and searched the variables of all 120 studies on INE's ANDA, the 2001 and 2012 censuses
included; the only religion items are ENDSA 1989, a coca-leaf survey and discrimination and
festival-spending items. The 1992 census did ask (household item 16) and its counts are now in hand
(§6), but they are 18 to 31 years older than the rounds pooled here.

## 2. The files

| file | what | from |
|---|---|---|
| `data/raw/bo/lapop_bo_<year>.dta`, 2008-2023 | LAPOP single-country Stata files, ids 1595 (2008), 1475, 1860, 2041, 2177, 2215, 2430 (2023) | `vanderbilt.edu/center-for-global-democracy/data/directory/?lp_download=<id>`; Free Tier behind the site usage agreement (research use, no redistribution, the attribution sentence), the terms the grand merge is already used under |
| `data/raw/bo/bol_admin_boundaries.shp.zip` | COD-AB v02: 9 departments, 112 provinces, 339 municipalities, valid from 2024-09-16 | HDX `cod-ab-bol` |
| `data/raw/bo/cpv2024_pop_{depto,provin}.csv` | 2024 census count by sex | INE REDATAM base `PHCCEN24ESPV1`, via `bo_census.py` |
| `data/raw/bo/cpv1992_religion_{depto,provin}.csv` | 1992 item 16, persons per group | INE REDATAM base `PHCCEN92ESP`, via `bo_census.py` |
| `data/raw/bo/redatam/*.htm` | the raw result pages behind both | same |
| `data/raw/bo/bol_admpop_adm1_2022.csv` | COD-PS 2022, printed against the count and nothing else | HDX `cod-ps-bol` |
| `data/raw/bo/boletin_censo_2024.pdf` | INE's results bulletin, 11,365,333 on p. 1 | UNFPA Bolivia's copy |
| `data/raw/bo/kontur_population_BO_20231101.gpkg.gz` | placement | Kontur |

**The population is the 2024 census, not COD-PS.** COD-PS 2022 is a projection from the 2012 census
that totals 12,006,031 against 11,365,333 counted, and misses unevenly, from Oruro 0.97x to Pando
1.22x (Ecuador's case, §9bn). No INE page found prints the department counts in a table, and
Wikipedia's table cites press, so the count is read off the REDATAM base and the bulletin's national
total, 5,682,835 women and 5,682,498 men are asserted against it.

The 2024 base's province break has 113 areas: the 112 provinces and **the Territorio Indígena
Multiétnico** (0809, Beni, 3,973 people), created by Ley 1497 of 2023-03-01 out of San Ignacio de
Moxos and Santa Ana de Yacuma (CIPCA's report of the law). No split is published and `bo_geo.py`
puts all of it in Moxos. Only the province files see this, and nothing drawn uses them.

## 3. The department labels, checked without trusting them

Honduras's merge printed one wave's labels on every wave (§11ap). These files carry their own labels
per wave and were checked anyway, on every run (`bo.py::decode_wave`):

- **`prov` is LAPOP's own department order in 2008-2018**: 1001 La Paz, 1002 Santa Cruz, 1003
  Cochabamba, 1004 Oruro, 1005 Chuquisaca, 1006 Potosí, 1007 Pando, 1008 Tarija, 1009 Beni. INE's
  order is 01 Chuquisaca to 09 Pando, so a code join on `prov - 1000` misplaces seven of nine.
  **In 2023 `prov` is a province code**, 10 plus INE's department and province, 52 of them.
- **`municipio` names, 2010-2023**, against COD-AB's 339 municipalities: for each prov code in each
  wave, the departments its municipality names can belong to intersect in exactly the labelled
  department (120, 86, 76, 64, 63 and 65 municipalities). Sixteen short names needed COD's official
  long form (`MUNI_ALIAS`: La Paz is Nuestra Señora de La Paz, Rurrenabaque is Puerto Menor de
  Rurrenabaque), and 2010's `cercado` is the province name standing for Cochabamba city. In 2023 the
  municipality code minus 1,000,000 is INE's code and COD's pcode for all 65, and it settles the one
  prov code (10606, Burnet O'Connor) whose only municipality name two departments share.
- **`estratopri`** equals `prov` to 2014, and from 2016 is the stratum of the labelled department for
  every respondent.
- **Sample share against the 2024 census**, exhaustive over 9! orderings (`lits.held_out`):
  **reported, not asserted.** 2008's design weights put La Paz at 28.4% and Santa Cruz at 24.5%
  against 26.7% and 27.5% in 2024, and Oruro, Tarija, Beni and Chuquisaca sit within a point of each
  other, so 14 of 362,879 orderings beat 2008's r=+0.991 with no label wrong. The weights describe
  an older population; the names are the witness.
- 2008 has no `municipio` and is not pooled.

## 4. Weights

2008-2014 carry one design weight per department, on about 300 interviews each, Pando included. From
2016/17 `wt` is 1 for every respondent although the design is six strata with the small ones
oversampled (Beni 8.4% of the sample, 4.2% of the people). `bo.py::poststratify` scales each (wave,
department) to the department's 2024 census share of 1,500, keeping the file's weight inside the
department, so every wave counts equally inside every department whatever its allocation was.

Respondents with an answer: 2010 2,944; 2012 2,977; 2014 3,043; 2016/17 1,635; 2018/19 1,639; 2023
1,644. Pando has 270-309 a wave to 2014 and 48-49 after.

## 5. Answer codes by wave

Weighted %, post-stratified:

| code | 2010 | 2012 | 2014 | 2016 | 2018 | 2023 |
|---|---|---|---|---|---|---|
| 1 Católico | 80.26 | 76.69 | 70.47 | 65.65 | 65.48 | 64.84 |
| 5 Evangélica y Pentecostal | 8.52 | 8.12 | 8.14 | 16.82 | 11.98 | 17.91 |
| 2 Protestante tradicional | 5.54 | 8.00 | 10.71 | 4.62 | 10.50 | 4.78 |
| 4 Ninguna (creyente) | 3.23 | 5.26 | 7.31 | 5.34 | 8.84 | 7.21 |
| 11 Agnóstico o ateo | 0.93 | 0.27 | 0.37 | 0.69 | 1.57 | 1.99 |
| 12 Testigos de Jehová | 0.86 | 0.47 | 1.38 | 1.13 | 0.00 | 0.00 |
| 6 Mormones | 0.41 | 0.83 | 0.68 | 1.10 | 0.00 | 0.00 |
| 3 Orientales | 0.19 | 0.13 | 0.52 | 1.12 | 0.61 | 0.78 |
| 77 Otro | 0.00 | 0.00 | 0.00 | 3.50 | 0.87 | 2.22 |

**No Honduras-2016-style shift.** 2016 has a low Protestant box and a high evangelical one, but 2023
repeats it and 2018 sits between; the two together run 14.1, 16.1, 18.9, 21.4, 22.5 and 22.7.
Bolivians trade the two boxes between rounds, not by place (§7). The card change reaches Bolivia a
round after Ecuador: 2016/17 still offers Witnesses and Mormons beside `Otro`.

## 6. The 1992 census

Found 2026-09-14 by a helper session: **INE's REDATAM server still hosts the 1992 base
(`PHCCEN92ESP`), with its link commented out of the home page**, which lists 2001, 2012 and 2024. The
server's certificate chain is incomplete. Its frequency form gives households by how many members
answered each way; persons are the sum of k times households, which is exact, and each area's
household total equals its own Total row. The universe is 6,292,819 people in private households of
6,420,792 counted. Catholic, evangelical, other and none equal INE's 2008 web page (Wayback capture
of `ine.gov.bo/censo/censo1992.aspx`) to the person; unknown is 416,424 against 416,514 there. The
department shares ANF Fides printed agree to 0.01.

**Against LAPOP 2010-2023** (`bo_checks.py`), as shares of the four named groups:

- **Across departments the ordering held.** Catholic Spearman +0.750 (exact p 0.013), non-Catholic
  Christian (1992's `evangélicos` against LAPOP's Protestant, evangelical, Mormon and Witness
  answers) +0.917 (p 0.0007); none +0.37 and other +0.55 do not. The level moved: Catholic 85.0% to
  70.6%, non-Catholic Christian 10.8% to 20.4%.
- **Inside departments it did not.** Each sampled province's departure from its department (86
  provinces with at least 10 respondents, in all nine departments), LAPOP against 1992, provinces
  shuffled within departments for the null, 20,000 draws: Catholic r=+0.262 (p 0.045, null 95th
  +0.254), non-Catholic Christian +0.200 (p 0.10), none +0.17, other +0.14. The power is thin:
  1992's between-province spread is 1.45 and 1.68 times LAPOP's sampling variance for the two big
  groups and under 1 for the others, and each small province is one or two sampling points, so the
  Catholic p is optimistic. The largest cell disagrees outright: Cordillera (Santa Cruz) was 26.1%
  non-Catholic Christian in 1992 and is 10.8% in LAPOP's 169 interviews there.

**So 1992 is a witness, not a source.** It confirms the department geography the survey draws for
the two groups that carry most of it. It is not used for §3.4's construction (the survey's department
level split by 1992's 112 provinces), because nothing shows its pattern inside departments still
holds, and a province map would draw that pattern as if it did. The municipality level (339) is
offered by the same form and was not pulled.

## 7. Which answers carry their own geography

§9cy's median over every halving (six waves, ten 3-against-3 halvings, `co.py`'s enumeration)
against a 2,000-draw per-wave permutation null, with Sweden's chi-square veto, Uzbekistan's
largest-cell refusal at 50% and Honduras's which-department-tops-both-halves, on unweighted counts.
At the nine departments:

| answer | n | share | median rho | null 95th | p | chi-square p | verdict |
|---|---|---|---|---|---|---|---|
| Católico | 10,188 | 70.57% | +0.950 | +0.442 | 0.0005 | 5e-81 | department share |
| Evangélica y Pentecostal | 1,545 | 11.91% | +0.850 | +0.433 | 0.0005 | 1e-36 | department share |
| Protestante tradicional | 917 | 7.36% | +0.575 | +0.434 | 0.009 | 1e-26 | department share |
| Ninguna (creyente) | 762 | 6.20% | +0.708 | +0.442 | 0.0025 | 9e-30 | department share |
| Agnóstico o ateo | 105 | 0.97% | +0.815 | +0.451 | 0.0005 | 8e-17 | department share |
| Otro | 102 | 1.10% | +0.170 | +0.460 | 0.28 | 4e-05 | national rate |
| Testigos de Jehová | 95 | 0.64% | +0.225 | +0.447 | 0.24 | 1e-03 | national rate |
| the other four | | | | | ≥0.36 | | national rate |

No placed answer has more than 3% of its respondents in one (wave, cluster) cell, and no failing
answer has one department on top in both halves of any halving. At LAPOP's six design strata,
Catholic, evangelical, Ninguna and agnostic pass and nothing passes that failed at the department,
so the mixed-level construction places nothing at the coarse level. The two Protestant boxes as one
pass at +0.875, but their department shares are unrelated (Spearman +0.03) and each passes alone, so
they are kept apart. The 2008 round, on its older card and not pooled, orders the departments like the
pool for Catholics (+0.667, p 0.029) and less clearly for evangelicals (+0.550, p 0.066).

The residual tail is 1.9% of Beni and 4.2% of La Paz against 3.0% nationally, and no national-rate
answer comes out reversed against its own department shares by more than a few tenths of a point
(Latvia's check).

## 8. The level is the pool's, deliberately

Catholic identification falls 15.4 points across the pool. §12's Norway entry would take the level
from the recent rounds. Not done: Colombia (§9dk), same instrument and 8.8 points, kept the pooled
level, and in Bolivia a recent level per answer is unstable because the Protestant and evangelical
boxes swap between rounds (Protestant 4.6%, 10.5%, 4.8% in the last three). `note_public` gives 2010's
and 2023's figures instead. **Worth revisiting** with a second recent instrument: Latinobarómetro
2023's `REG` is all nine departments (§11ap).

## 9. Mapping calls (`taxonomy/bo2023.py`)

- **`Religiones Tradicionales` -> `other.bo`, not `indigenous`.** 24 respondents, one to four in every
  department (chi-square p 0.86), 10 of them in 2014; the card's examples are Candomblé, Vudú,
  Rastafari, Maya religions and Umbanda. Colombia's call, reached on Bolivia's own data.
- `other.bo` added to `taxonomy/branches.py`, `build_tree.py` run (684 nodes).
- Otherwise as `co2023.py`, with traditional Protestant and agnostic/atheist placed where Colombia's
  were not.

## 10. What was drawn

11,365,333 people on nine departments. As drawn: Católico 70.57%, Evangélica 11.91%, Protestante
tradicional 7.36%, Ninguna (creyente) 6.20%, Otro 1.10%, agnostic or atheist 0.97%, Witnesses 0.64%,
Eastern 0.56%, Mormon 0.50%, Traditional 0.15%, Jewish 0.04%.

| department | n | Catholic | evangelical | traditional Protestant | believer, no church | agnostic or atheist |
|---|---|---|---|---|---|---|
| La Paz | 2,241 | 60.3% | 14.1% | 9.2% | 10.0% | 2.2% |
| Pando | 1,018 | 64.3% | 24.0% | 5.7% | 3.7% | 0.0% |
| Cochabamba | 2,011 | 70.3% | 11.8% | 8.3% | 6.0% | 0.6% |
| Oruro | 1,182 | 70.5% | 9.2% | 11.5% | 5.7% | 1.1% |
| Beni | 1,305 | 73.4% | 17.9% | 2.9% | 3.6% | 0.4% |
| Potosí | 1,321 | 75.2% | 9.6% | 3.4% | 8.6% | 0.6% |
| Santa Cruz | 2,285 | 75.3% | 11.6% | 7.2% | 3.3% | 0.1% |
| Tarija | 1,255 | 79.3% | 7.5% | 4.7% | 5.4% | 0.3% |
| Chuquisaca | 1,264 | 83.2% | 5.5% | 3.9% | 3.4% | 1.9% |

**Kontur's density cap, checked 2026-09-14 at the supervisor's request.** `plateau_scan2.py bo` over
145,425 hexes: the densest is 18,208 people per km², none is at Kontur's 46,200 cap, and no block is
listed. Nothing for the pending fix to touch.

## 11. Open

- **The whole-map build tail was not run for Bolivia (2026-09-14, 16:10).** Steps 1-9 are done: dots
  in both editions, `coverage.py` ok, `built_countries.py --check` ok. `data/build.lock` is held by
  session `cc923721`, whose pid 18996 is gone; its buffers step stopped writing at 15:11 on `id.bin`,
  after its tiles archive (15:09) and before its manifest (still 14:59), so the buffers are half
  rewritten. No python was running. Breaking the lock with `build_tail.py --force` was refused by
  this session's permissions, so it needs someone allowed to: `python tools/build_tail.py --id <sid>
  --force`, which also repairs that half-written state and picks Bolivia up.
- **The non-Christian tail**, as for every LAPOP country (`queue.md`'s refinement list).
- **Latinobarómetro 2023** (a direct zip, `REG` is all nine departments, n=1,200, with Adventists and
  three evangelical subtypes on its card): a recent level and a second instrument at the same units.
- **1992 at the 339 municipalities**, from the same REDATAM form, if a later source ever licenses a
  pattern below the department.
- **WVS wave 7** (2017, nine departments) behind the WVS licence form, which is Anita's.

## 12. Review, 2026-09-14 (`f95259a4-borev`)

Read from the raw `.dta` files, the REDATAM CSVs and `data/normalized/bo.csv`, not from the sections
above.

- **Population base: confirmed.** All nine department counts in `cpv2024_pop_depto.csv` equal INE's
  final 2024 results (released 2025-08-28) to the person, as tabled on es.wikipedia citing INE; El
  Deber's report of the release gives Santa Cruz 3,122,605 and La Paz 3,030,917.
- **Department labels: confirmed round by round, with a witness that uses no label.** `decode_wave`
  runs per wave. Mother tongue (`leng1`) by decoded department puts Aymara in La Paz (22-31%) and
  Oruro (14-22%), Quechua in Chuquisaca, Cochabamba, Potosí and Oruro (29-50%), and 85-99% Spanish in
  Beni, Pando, Santa Cruz and Tarija, in 2008, 2010, 2012, 2014 and 2023 alike. 2016/17 and 2018/19
  have no `leng1`; `etid` there makes La Paz the most indigenous department (52%, 45%), weaker but
  consistent. 2023 decodes off INE's numeric codes as well as names.
- **Weights: the rescale is right.** Inside a department it only sets how the rounds count, and they
  count equally. Pooling respondents instead of rounds keeps the ordering (Spearman +0.97 for Catholic
  and for evangelical) but moves Pando's evangelical share from 24.0% to 17.9%, Chuquisaca's Catholic
  from 83.2% to 87.0% and Potosí's from 75.2% to 78.8%. Pando's 24.0% in `note_public` rests one sixth
  on 48 interviews in 2023, 45.8% of them evangelical. Equal rounds is the better choice for a time
  average; left as built.
- **The level: the map understates a real decline.** §8's reason, that the Protestant and evangelical
  boxes swap, is true of those two boxes and not of the Catholic share, which is 65.7%, 65.5% and
  64.8% in the last three rounds against 70.6% drawn; the two boxes together run 21.4%, 22.5%, 22.7%.
  Not a design artefact: the weighted urban share is 67-71% in every round, and rural respondents
  fell too (84.3% Catholic in 2010, 63.0% in 2023). Rounds 2016-2023 alone give 65.3% Catholic, 22.2%
  Protestant plus evangelical, 7.1% believer without a church and 1.4% agnostic or atheist. That is
  5.3 points, above the 3.5 at which spec §12's Norway entry reaches for §3.4, and every LAPOP build
  shares the choice, so it went to Anita as `ask/015`. Added to `note_public`: "so this map is more
  Catholic than any round since 2016 found", Colombia's clause. `tiles.py --refresh-meta` not run.
- **1992 as a check only: confirmed, on firmer ground than §6 gives.** A re-run of `bo_checks.py`
  reproduces §6 exactly. Had 1992's pattern inside departments survived, sampling noise alone would
  still leave LAPOP's province departures correlating with it at about +0.77 for Catholics and +0.79
  for other Christians (the square root of spread over spread plus sampling variance, from §6's 1.45
  and 1.68; +0.65 and +0.68 at a design effect of 2). They correlate at +0.26 and +0.20, so most of
  the pattern is gone, not merely hard to see. Catholic's p=0.045 is nominal.
- **Checks and map.** `check_md`, `built_countries --check` and `check_rollup` clean; the mapping
  follows `co2023.py`. The build tail has since run (runlog, the Kontur fix), and a screenshot shows
  Bolivia drawn on La Paz-El Alto, Cochabamba, Santa Cruz, Sucre, Potosí and Tarija, nothing in the
  sea.
