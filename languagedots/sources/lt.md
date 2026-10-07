# Lithuania: Gyventojų ir būstų surašymas 2021, mother tongue

Built 2026-10-05 (session d9e44929-lt). Rebuild:

```
python sources/lt_census.py [--fetch]   -> data/normalized/lt.csv
python taxonomy/build.py
python tools/check_country.py lt
python scatter.py --country lt
```

Drawn: 2,810,758 people on 60 municipalities (savivaldybės), 10 labels on 10 nodes, 2,805 dots,
2 rings, of them 861 people (0.03%) in 168 municipality cells withheld as confidential and
estimated from the totals, tier `derived` (Anita's ruling, 2026-10-05; section 2). Nobody is
recorded as not stated.

## 1. The table

Statistics Lithuania (Valstybės duomenų agentūra), 2021 census. The web UI `osp.stat.gov.lt` is
behind Cloudflare (403); the data are on the open SDMX host `osp-rs.stat.gov.lt`, as
`religiondots/sources/lt.py` found. The dataflow catalogue (`/rest_xml/dataflow/`, follows a 301
to `/ords/ipospp/ospp/rest_xml/dataflow/`, 7.4 MB) lists each cube's dimensions in its English
description; searching it for "Mother tongue" gives five cubes, three of them census:

| dataflow | dimensions | use |
|---|---|---|
| `S3R778_GBS010509` (28 KB) | territory (73: country, 2 regions, 10 counties, 60 municipalities) x mother tongue, 2021 | drawn |
| `S3R778_GBS010302_1` (8 KB) | ethnicity x mother tongue, national, 2001/2011/2021 | check |
| `S3R778_GBS010302` | territory x ethnicity x mother tongue x sex, 2001-2011 only | not used |

No key, no login, browser User-Agent, TLS verifies. Nothing finer than municipality was found
(the religion cube has the same ceiling).

**The question**: gimtoji kalba, mother tongue. A person could give two. The drawn cube has
Lithuanian, Polish, Russian, Belarusian, Ukrainian, Latvian, German, Romani, `Kitos` (other),
`Dvi gimtosios kalbos` (two mother tongues) and the total.

**How the 2021 figures were made.** 2021 was a register-based census, and registers do not hold
language, ethnicity or religion. For those Statistics Lithuania ran a supplementary survey,
online in January-February 2021 (about 56,000 people) and by interviewers in April-June (about
115,000 people in 40,000 households), and states that "mathematical methods were also used to
determine the population according to ethnocultural characteristics". Source: the office's
results page
https://osp.stat.gov.lt/en/2021-gyventoju-ir-bustu-surasymo-rezultatai/tautybe-gimtoji-kalba-ir-tikyba,
read only through a search engine's extract because the page 403s here (Wayback was rate-limited
at the time); the survey sizes are also on the English Wikipedia page for the census. So the
municipality counts are the office's estimates for everyone, built on a ~6% survey, not a full
count. That explains why "not stated" is 0 in 2021 (it was 43,906, 1.4%, in 2011): everyone was
given a value. It is the official census table and the only one there is; drawn as published,
and `how` and `note_public` say it. The same holds for religiondots' Lithuania, which uses the
same office's religion cube from the same survey.

## 2. Reading the cube

`sources/lt_census.py`, after religiondots' reader. One geography dimension holding four nested
levels told apart by code shape (`00`, `LT01`/`LT02`, `01`-`10`, `11`-`94`). Only 2021.

Nulls: this cube has one OBS_STATUS value, `konfidencialūs duomenys` (confidential), on 168 of
the 600 municipality cells and 4 of the 100 county cells (Latvian in Tauragė and Marijampolė,
German in Marijampolė and Klaipėda). The national cube's nulls mean the opposite, "no such
phenomenon", a true zero, and are read as 0.

**Withheld cells are estimated** (Anita, 2026-10-05: estimate the suppressed cells from the
totals, as Romania and Finland do; they held 56% of German and 18% of Latvian speakers). First
built without them, as religiondots leaves the religion cube's; that call is reversed.
`lt_census.py impute()` uses both margins of each cell, each known exactly because nothing is
not-stated: the unit's total less its published cells, and the parent's cell less its published
children. The 4 county cells (41 people) are fitted from the national row first, then each
county's municipality cells from its county row (published or just estimated), by iterative
proportional fitting from 1 per cell (`ro_census.py`'s method). Every margin is met (largest miss
under 1e-14; the two margins agree exactly in every county, Vilkaviškis's 3-person slip included
because it sits in both its total and its Kitos), and the municipalities then reproduce every
language's national total (asserted). The estimates are an even spread consistent with the
totals, not a guess at which municipality holds the few German or Latvian speakers; where a unit
has a single withheld cell the estimate is exact. Rows are tier `derived`; `how` and
`note_public` say so.

| withheld | municipalities | estimated | share of the language |
|---|---|---|---|
| Latvian | 38 of 60 | 140 | 17.9% |
| German | 48 | 161 | 55.7% |
| Belarusian | 21 | 111 | 1.7% |
| Ukrainian | 14 | 141 | 2.9% |
| Romani | 19 | 74 | 3.8% |
| Polish | 18 | 139 | 0.1% |
| Kitos | 4 | 42 | 0.3% |
| Two mother tongues | 6 | 56 | 0.1% |
| total | | 864 (861 in municipalities; the other 3 are Vilkaviškis's slip, section 3) | 0.03% of the country |

## 3. Checks (all pass)

- Level counts 1, 2, 10, 60; national total 2,810,761, the same as religiondots' religion cube.
- Region and county totals sum to the country exactly. Municipality totals sum to 2,810,758:
  **Vilkaviškis (39) is published 3 short**, its total 35,365 and its `Kitos` 32, against its
  county (Marijampolė: 138,292 and 210, 3 more than its municipalities) and against the religion
  cube's 35,368. The unit is internally consistent, so it is drawn as published. The check
  allows exactly this slip and nothing else.
- Every county equals the sum of its municipalities in all 51 fully published (county,
  category) cells, Vilkaviškis aside. The county membership list is in the script and is
  asserted by this.
- The national row against the second cube (ethnicity x mother tongue, total over ethnicity;
  its `Kitos` holds Latvian, German and Romani): Lithuanian 2,398,352 vs 2,398,353, Russian
  equal, Polish +1, two mother tongues equal (49,066), Kitos 17,134 vs 17,135, Belarusian
  6,711 vs 6,708, Ukrainian equal, not stated 0 vs 0. Two tabulations of the same census, a few
  people apart; tolerance 3.
- All 60 municipality totals equal religiondots' religion cube except Vilkaviškis's 3.
- In the 6 municipalities with nothing withheld, and nationally, the categories exhaust the
  total: no hidden not-stated.

## 4. Mapping (taxonomy/lt2021.py, tree.d/lt.txt)

Lithuanian, Latvian, Russian, Belarusian, Ukrainian, Polish, German: the existing leaves.
Romani: `romani.romani`, variety not stated (Lithuanian Roma mostly speak Baltic Romani, but the
answer is just "Romani"), as cz2021 and hr2021. `Kitos`: `other`; it holds Tatar, Karaim,
Armenian, Yiddish, Hebrew and migrants' languages alike, and the cube cannot tell indigenous
from foreign.

**Two mother tongues** (49,066, 1.75%) is the one call. The census publishes that a person gave
two, never which two. It is not a language and not a remainder of any one group, so it is a new
leaf `other.two_mother_tongues`, "Two mother tongues (the census does not say which)", beside
az2019's `other.jewish` and Ethiopia's unclassifiable names; it draws light grey (#bababa)
against `other`'s mid grey. The national cube says who they are by ethnicity: 17,822
Lithuanians, 12,417 Poles, 8,905 Russians, 3,219 Belarusians, 1,728 Ukrainians, 894 others, 4,081
not stated, so the pairs are almost all within Lithuanian, Polish, Russian and Belarusian. Where:
Vilnius city 28,747 (5.2%), Klaipėda city 7,140 (4.7%), Trakai 4.5%, Vilnius district 3.3%.
Sharing each person half-and-half across a guessed pair (spec §3.6's treatment) would need the
pairs; no table of them was found, and inferring them from ethnicity would change the counts
with a proxy, which is Anita's to allow. Reversing: map the label to a split in lt2021 and
countries/lt.py once a pair table turns up (the census publication may print national pairs; not
searched past the SDMX catalogue).

## 5. Geography

religiondots' `data/geo/lt/lt_grid_400m.gpkg` (read only): Kontur 400 m hexes, 63,766, keyed
`unit` = the two-digit savivaldybė code (religiondots joined GISCO LAU_ID to the census code; its
`countries/lt.py` describes the city/ring blur Kontur shows and why the join is right). Our
`geo_id` is the same code, so the join is the identity; check_country.py confirms all 60 units
have hexes. Placement by the layer's population.

## 6. What the map shows

Polish is Šalčininkai (61%), Vilnius district (44%), Trakai and Švenčionys (20%) and 63,000 in
Vilnius city (11%). Russian is Visaginas (67%), Klaipėda (20%), Zarasai (20%) and 81,000 in
Vilnius city (14.5%). Romani is Panevėžys, Jonava and Vilnius city. Mother tongue does not just
follow ethnicity: 19,260 ethnic Poles and 14,052 ethnic Belarusians named Russian.

**Colour.** Lithuanian is orange-tan (#cf8e52), clear of everything around it. Russian
(#54b85b), Polish (#00a76c) and Belarusian (#00bc99) are three greens, and they share
Šalčininkai, Švenčionys and Vilnius city. Not changed: Russian and Polish are hand-picked and
drawn in many countries, and Belarusian is under 3% everywhere here. Worth a look when colours
are next tuned (Belarus and Latvia will meet the same three).

## 7. Second sources

None needed for the remainder (the question was asked of everyone). The 2011 census
(GBS010302, by territory, a full enumeration) would be the corroboration for the 2021 survey
estimates at municipality level; not compared.
