# Haiti — the record

Built 2026-09-08. `sources.md` §9ck is the write-up; this file is the working record.

    source        IHSI, Enquête sur les Conditions de Vie des Ménages Après le Séisme 2012
    tier          10 departments (ADM1)
    universe      17,977 people aged 10 and over, in 4,951 households, in 500 clusters
    categories    12
    population    COD-PS 2024, 11,899,555
    every row     modelled (§7)

---

## 1. What was asked of Haiti, in order, and what came back

The queue reached Haiti through LAPOP (§11ad, 7,252 AmericasBarometer respondents) with a
note from Anita that the 4.4% Vodou figure is a self-identification floor and wanted a second
source before it was drawn. **The first question asked here was not about LAPOP.** It was
what IHSI itself publishes, which is §9ce's Uruguay move and §9cf's Dominican one.

| asked | answer |
|---|---|
| Does the census ask religion? | **Yes.** The 4ème RGPH 2003 has a twelve-category religion question, and `ihsi.gouv.ht/recensement/resultat_rgph_2003` prints the national shares in prose. |
| Does the census publish it below the nation? | **No.** The full table set is online as 111 table images and six thematic PDFs (`ihsi.gouv.ht/public/tableau/POPULATION.pdf`, 295 pages, with a text layer). Every religion table — 206, 207, 208, 226, 241, 261 — is headed `ENSEMBLE DU PAYS`. A Wayback CDX prefix sweep of `ihsi.ht/tableau/*` returns the same six PDFs and 173 files, no departmental variant. |
| Were departmental volumes printed? | **Yes, and they are not online.** IHSI published *4ème Recensement Général de la Population et de l'Habitat 2003, Résultats définitifs, Département du Nord* and its nine siblings in 2005; abebooks has them as paper. Nothing in either IHSI domain's archived file tree carries them. |
| Is there a REDATAM instance? | **No.** `prod.redatam.org`'s root install is the CELADE demo stub (`Nueva Miranda`), eight candidate Haitian CGI directories all 404, and `celade.cepal.org/redatam/` redirects to `redatam.org`. Haiti is not in CELADE's collection. |
| Is Haiti in the USCB HDX set? | **Yes, and without religion.** `Haiti Subnational Population and Housing Data Tables` exists ([[reference_uscb_country_gdb]]) and its sheets are Population Estimates, Education, Health, Info and Com Technology, Population, Housing Units, Transportation. Haiti is one of the 26 of 34 with no religion layer. |
| Is Haiti in the UNSD oracle? | **No.** Absent from all 117 rows, checked by name and against `--list`, which proves only that no tabulation was forwarded (§12). |
| Does an IHSI **survey** carry religion? | **Two of them do.** ECVMAS 2012 (`I_H04`) and ECVH 2001 (`hr7`). |

**ECVMAS 2012 is the build.** ECVH 2001 is a published table and is the outside witness the
overrides rest on. The census is the national anchor and the check on the age universe.

    LAPOP AmericasBarometer 2010-2016 pooled      7,252 respondents, 10 departments, 18+
    IHSI ECVMAS 2012                            17,977 respondents, 10 departments, 10+
    IHSI ECVH 2001                              32,840 respondents,  9 departments, all ages
    IHSI 4eme RGPH 2003 (census)             8,373,750 people,   NATIONAL ONLY, all ages

## 2. Two squatted Haitian domains, and the office moved without its microdata

**`www.ihsi.ht` is gone.** It answers 301 and lands on `znaki.fm`, an unrelated commercial
site. **`rgph-haiti.ht`, the fifth census's own site, answers 200 with a real page and it is
a betting site.** Both are exactly the failure the brief warns about: a dead-looking host is
usually a bot wall, and a live-looking host is sometimes a squatter.

IHSI itself is at `ihsi.gouv.ht` and it did **not** bring the microdata across. What the old
host published, openly and with no account or request form, was the whole ECVMAS database:

    http://ihsi.ht/pdf/ecvmas/ecvmas_base_donnees/2_ECVMAS_BASE DE DONNEES.zip   9.0 MB

32 SPSS files, including `2_ECVMAS_Individus_e h i j k n p_ok.sav` (23,775 people, 271
variables) and `0_ECVMAS_ECHANTILLON_ok.sav` (4,951 households with `Poids_Finaux`). It is
fetched from the Wayback Machine's byte-identical copy, which is `sources/do.py`'s route for
ONE's Cloudflare-walled files put to a different use: there the office is alive and blocking,
here the office is alive and the file is not.

`sources/ht_geo.py` still reads the live host for nothing, but `sources/ht.py` fetches the
2003 census tables from `ihsi.gouv.ht/public/tableau/POPULATION.pdf`, which is live.

## 3. The department join is zero for ten and looks fine

ECVMAS numbers the departments **1 to 10 in French alphabetical order**: Artibonite 1, Centre
2, Grand'Anse 3, Nippes 4, Nord 5, Nord-Est 6, Nord-Ouest 7, Ouest 8, Sud 9, Sud-Est 10.

COD's pcodes run **HT01 to HT10 in Haiti's traditional order**: Ouest 01, Sud-Est 02, Nord
03, Nord-Est 04, Artibonite 05, Centre 06, Sud 07, Grand'Anse 08, Nord-Ouest 09, Nippes 10.

Both are dense 1..10 integer sequences over the same ten units, so `DEPT -> HT{n:02d}` runs
without a warning and **mispairs all ten**. Artibonite's respondents go to Ouest, Nord's go
to Artibonite, and every national total is untouched. `sources/ht_geo.py::check_code_join`
asserts the correct count is still zero, because a re-cut that made it *mostly* right is the
dangerous direction ([[reference_name_join_wrong_neighbour]]).

The join is on the **French** name (`adm1_name1`); COD's `adm1_name` is English, so joining
on that loses six of the ten silently. One alias: ECVMAS's `Grand'Anse` for COD's
`Grande'Anse`.

**COD's spelling is the one that ends up in `ht_lookup.csv` and in `ht.csv`'s `geo_name`, and
it is wrong.** Haiti writes **Grand'Anse**; COD writes `Grande'Anse`. It reaches no reader —
`_ht_counts` passes only `unit`, `node`, `count`, `congregations` and `tier` to the viewer, so
department names are never displayed — and it was left alone rather than re-running the whole
build tail for a console string. Anything written for a person says Grand'Anse.

Held-out check, which touches nothing in the religion column: ECVMAS's weighted department
share of people against COD-PS 2024, **r = +0.9961 over ten**, and none of 20,000 random
pairings reaches it.

## 4. The universe is ten and over, and the census says what that costs

`I_H04` is in ECVMAS's individual module and the module's universe is **10+**. Of 23,775
people on the household rosters, 5,792 have no answer and 5,714 of them are under ten; the
other 78 are 10+ attrition. Six of the 10-and-overs said `Ne sait pas` or `Refus de réponse`.

The shares are applied to each department's whole population, which is what `sources/lapop.py`
does with an 18+ survey everywhere else on this map. **Here it is measured rather than
assumed, because the 2003 census tabulated religion by age group.** Recomputed over the
10-and-overs only:

| census category | all ages | aged 10+ | shift |
|---|---:|---:|---:|
| Catholique | 54.68% | 55.02% | +0.34 |
| Baptiste | 15.38% | 15.47% | +0.09 |
| Aucune religion | 10.22% | 9.40% | **-0.82** |
| Pentecôtiste | 7.94% | 7.97% | +0.03 |
| Autre religion | 4.00% | 3.99% | -0.01 |
| Adventiste | 2.96% | 3.07% | +0.11 |
| Vaudouïsant | 2.11% | 2.31% | +0.20 |
| Méthodiste | 1.48% | 1.47% | -0.01 |

Nothing moves a full point. Children are recorded as having no religion rather more often
than adults, which is the only systematic thing in it.

## 5. The split-half has almost no power on ten units

The bar is 1.96/sqrt(n-1), which is **+0.65** on ten departments against +0.35 on the
Dominican Republic's thirty-two. The split is on the parity of ECVMAS's 500 sampling
clusters; because a single parity split is one noisy draw on ten units, `stability()` also
reports the **median over 400 random cluster half-splits** and takes the verdict off that.
The bar is not moved.

| | national | parity | median of 400 | chi-square | verdict |
|---|---:|---:|---:|---:|---|
| Catholique | 47.53% | +0.59 | +0.65 | 1.6e-117 | own geography, OVERRIDE |
| Baptiste | 17.46% | +0.54 | +0.55 | 5.7e-56 | own geography, OVERRIDE |
| Pentecôtiste | 10.88% | +0.53 | +0.42 | 3.2e-70 | own geography, OVERRIDE |
| Autre protestant | 9.18% | +0.45 | **+0.68** | 2.0e-59 | own geography |
| Aucune | 6.75% | +0.84 | +0.56 | 5.6e-23 | national rate |
| Adventiste | 3.25% | +0.50 | **+0.67** | 6.5e-28 | own geography |
| Vaudou | 1.51% | +0.63 | +0.56 | 3.0e-62 | own geography, OVERRIDE |
| Méthodiste | 1.33% | +0.75 | **+0.67** | 3.6e-40 | own geography |
| Autre | 1.11% | -0.67 | -0.25 | 1.5e-03 | national rate |
| Episcopale | 0.54% | +0.29 | +0.33 | 2.1e-14 | national rate |
| Témoin de Jéhovah | 0.46% | -0.28 | +0.21 | 4.5e-02 | national rate |
| Musulman | 0.004% | nan | nan | 1.0 | national rate |

**`Catholique` lands within 0.003 of the bar with a 24-point spread and a chi-square of
1e-117.** That is the whole argument for not treating this test as the decision here: it is
not measuring whether Haiti's departments differ, it is measuring whether ten of them can be
ranked twice out of one sample, and they cannot.

## 6. What the overrides rest on: ECVH 2001 as a second instrument

§9bi's rule is that the bar is never moved and an override is a named decision with its
reason printed on every run. There are four, and every one rests on the same thing: **a
different IHSI household survey, a different questionnaire, a different sample and eleven
years earlier, ranking the same departments the same way.** That is replication across
instruments, which is what the split-half is a cheap proxy for.

ECVH 2001's Tableau 2.2.4.2 is transcribed into `sources/ht.py::ECVH_2001` from
`ecvh_volume_I_(juillet2003).pdf` pages 76-77. Nine departments, because **Nippes was created
out of Grand'Anse in September 2003**, eight months after the census; ECVMAS's Grand'Anse and
Nippes are pooled back for the comparison.

| answer | ECVH 2001 | ECVMAS 2012 | spearman on 9 | of 20,000 pairings |
|---|---:|---:|---:|---:|
| Catholique | 59.05% | 47.53% | **+0.85** | 29 reach it |
| Baptiste | 16.59% | 17.46% | **+0.78** | 35 |
| Adventiste | 2.94% | 3.25% | **+0.95** | 0 |
| Pentecostal bloc | 14.08% | 20.06% | **+0.90** | 8 |
| Episcopale | 0.50% | 0.54% | +0.50 | 443 |
| Vodou | 1.94% | 1.51% | +0.47 | 522 |
| Témoin de Jéhovah | 0.40% | 0.46% | +0.08 | 2,043 |

**The two answer cards are not the same and three of the rows above are affected.** ECVH
offered `Eglise de Dieu` (9.4% of Haiti, larger than its own Pentecostal box) and `Eglise
wesleyenne`, and ECVMAS offers neither; ECVMAS offers `Autre protestant`, `Méthodiste` and
`Aucune`, and ECVH offers none of those. **ECVH's card has no no-religion answer at all**,
which is why `Aucune` has no outside witness and is drawn flat. The Pentecostal comparison is
therefore run as the bloc both cards can express: ECVH's Pentecôtiste + Église de Dieu +
Église wesleyenne against ECVMAS's Pentecôtiste + Autre protestant, **+0.90**.

`Vaudou` is the fourth override and it is §9bi's Guatemala case rather than a whole-ordering
one: see §7.

## 7. Vodou — the §3.5 lean check fires, and that is the finding

Anita's queue note wanted a second source before Vodou was drawn. There are now three, and
they all say the same thing about the level and the same thing about the shape.

| instrument | national Vodou |
|---|---:|
| IHSI ECVH 2001 | 2.1% |
| IHSI 4ème RGPH 2003 census | 2.11% |
| IHSI ECVMAS 2012 | 1.51% |
| LAPOP `Religiones Tradicionales`, pooled | 4.39% |

**Every one is a floor**, and `branches.py`'s note on `afrodiasporic.vodou` says why: Vodou
in Haiti is overwhelmingly served alongside Catholicism rather than instead of it, so a
questionnaire offering one box collects the church. The three IHSI instruments agree with
each other and LAPOP's card, which offers a generic `Religiones Tradicionales` beside a much
shorter Christian list, is the outlier upward.

**The §3.5 lean check fires hard and it should be read rather than dismissed.** Drop the
single most extreme department and the national Vodou spread goes from **5.7 points to 1.2**.
The whole geography is Artibonite.

    Vaudou           HT09  0.0% to HT05  5.9%, a 5.9 point spread
                     without the top:    1.2      without the bottom: 5.9

And Artibonite is where both surveys put it: **7.3% in 2001 and 5.9% as drawn from 2012,
against 2.0% and 1.2% for the next highest**, with Nord-Est at the bottom in both. So what is
replicated is one heartland and one floor, and nothing in between. `note_public` says to read
the two ends of the Vodou colour and not the ordering, in those words, which is §9bi's
Guatemala wording applied to a much smaller category.

## 8. LAPOP is not wired, and two traps came out of finding out why

§11ad's Haiti row is **`pais=22`** in the Grand Merge, 7,252 respondents over `prov` 2201 to
2210, waves 2010, 2012, 2014 and 2016.

**Its `prov` cannot be decoded from the merged file.** Mapping `prov - 2200` to Haiti's own
department numbering, which is what the Dominican Republic's code did successfully (§9cf),
gives **r = +0.23** against COD-PS and puts **40.7% of Haiti in Centre**, a department with
8.5% of the population. Every candidate ordering that fits is a guess, and rank-matching the
respondent shares to the population shares and then validating on the population shares is
circular. So this is Honduras's problem in §11ad's own table, and Haiti should be listed
there beside it.

**And `pais=41` is Canada, not Haiti.** Canada has ten first-order units and its LAPOP `prov`
codes are `4101` to `4110`, so an agent guessing that Haiti is 41 gets ten units with plausible
codes and a join that looks clean. It fails visibly here only because the Canadian rows carry
no religion answer at all; had they carried one, the check would have been a comparison of
Haiti against Canada with every unit count correct.

## 9. Population, and what a 2003 census does to a 2024 map

Haiti has not counted since January 2003 and the fifth census has not been held, so there is
no census to prefer to COD-PS the way §9cf preferred the Dominican Republic's 2022 count.
COD-PS 2024 is the base, 11,899,555 people, and `sources/ht_geo.py` prints the drift:

    Centre                6.9% of Haiti in 2003 ->  8.5% in 2024  (+1.6 pt)
    Sud-Est               5.8%                  ->  7.0%          (+1.2)
    Sud                   7.4%                  ->  8.3%          (+0.8)
    Grand'Anse + Nippes   7.5%                  ->  7.9%          (+0.4)
    ...
    Artibonite           15.5%                  -> 14.9%          (-0.6)
    Ouest                37.0%                  -> 33.4%          (-3.6)

The direction is the surprising one: the projection has Port-au-Prince's department growing
more slowly than the country. The comparison is on nine units, because the census predates
Nippes.

Kontur's HT extract is the placement layer (`sources/ht_grid.py`, 35,526 hexes). Its
per-department ratio against COD-PS runs 0.69x in Sud to 1.16x in Ouest, which is a wide
spread for a weight and is why it sets no levels; it is the only thing in this build that has
seen Haiti since 2010.

## 10. What would improve this country

1. **The ten departmental census volumes from 2005.** They are the only known source of
   religion by commune for Haiti, they exist on paper, and no library scan is online.
   140 communes instead of 10 departments.
2. **EMMUS-VI 2016-17** (DHS, run by IHE with MSPP). Newer than ECVMAS and larger, and DHS
   asks religion, but the microdata needs a DHS Program account and its final report does not
   cross religion with department. Not chased.
3. **ECVMAS phase 2 (2013)** and any later IHSI round. The 2013 phase was fielded and its
   database was not on the old host's file tree.
4. **A question ECVMAS can answer and this build does not use.** `I_H04` sits beside a full
   individual module, so religion by education, by employment and by displacement-camp
   residence is one crosstab away and none of it is on this map.


---

## 11. Review, 2026-09-08 (`rd-review`, session `...-ht-rev`)

A second pass over the material rather than over §1-10. Checks clean: `check_md.py`,
`built_countries.py --check`, `check_rollup.py ht` (nothing orphaned), `review_dump.py ht`.
Map glance: Haiti draws, dots on land, none in the sea, density on Port-au-Prince and
Cap-Haïtien. `afrodiasporic.vodou` is in the built output at 178 dots, which is 178,476
people at 1,000 a dot.

### 11.1 The zip is the genuine article, and here is the record of it

Asked once and written down so nobody asks again. The local file is **10,055,634 bytes**,

    sha256  dfa9783b2d54bdd2c89a892590df8e30c28ef2076a68c3302d609e86446efcac
    sha1    412b8f2e6f363d259c011622e9f0cc866a02e539
    sha1 in Wayback's base32 form:  IEVY6LTPGY6SLHABCYROT4GMQZVAFZJZ

The CDX API lists **four captures of that URL and every one carries that same digest**:
`20170707203941` on `ihsi.ht`, then `20180803225506`, `20190604185405` and `20200131102832`
on `www.ihsi.ht`. Byte-identical over two and a half years and across both hostnames. The
homepage kept answering `200 text/html` through 2024, so all four sit well inside IHSI's
own tenure of the domain rather than near the handover.

Internal metadata agrees. 32 members, SPSS timestamps **2014-01-28 to 2014-02-12**, French
IHSI filenames, and a `0_ECVMAS_Description des fichiers de la base de donnees_29012014` in
the zip. `2_ECVMAS_Individus` is 23,775 rows by 271 columns, created 2014-01-29, and its
`I_H04` carries the label **`Quelle est votre religion ?`** with the twelve answers. Nothing
here is consistent with a repackaged or substituted file.

**And the department decode is in the file rather than inferred.** Both `.sav` carry `DEPT`
value labels: `1 Artibonite, 2 Centre, 3 Grand'Anse, 4 Nippes, 5 Nord, 6 Nord-Est,
7 Nord-Ouest, 8 Ouest, 9 Sud, 10 Sud-Est`. §3's French-alphabetical claim is the file's own
label set, which is stronger evidence than §3 gives itself. Worth wiring:
`check_labels()`'s docstring says it asserts *"CATEGORY and the department labels"* and it
only reads `I_H04`. The `DEPT` labels are what the whole join rests on and they are the one
thing not asserted.

### 11.2 The census 2.11% checks out against the PDF, read first-hand

Not taken from `sources/ht.py`. Tableau 206, `ENSEMBLE DU PAYS`, Deux sexes, out of
`POPULATION.pdf`: total **8,373,750**, `Vaudouïsant` **176,976** = **2.114%**. The rest of
§4's column agrees too: Aucune 855,878 (10.22%), Catholique 4,578,842 (54.68%), Baptiste
1,287,742 (15.38%), Pentecôtiste 664,860 (7.94%), Adventiste 248,063 (2.96%), Méthodiste
123,944 (1.48%), Autre religion 335,308 (4.00%), Musulman 2,013, Mormon 5,683.

### 11.3 The ECVH 2001 nationals are sample-weighted, and Vodou is quoted three ways

`ecvh_check()` computes each national as `(department % × respondent count) / 32,840`, and
**ECVH's sample is a long way from proportional**: Ouest is 25.4% of it against 37.0% of
Haiti, Artibonite 11.9% against 15.5%. Since Vodou's whole geography is Artibonite, that
weighting pulls it down. Re-weighting the same nine departments by the 2003 census:

    Vodouisant       1.94% sample-weighted    2.18% population-weighted
    Eglise de Dieu   9.14%                    9.42%
    Catholique      59.05%                   58.08%

So §6's table prints **1.94%**, §7's table and `note_public` both say **2.1%**, and the
population-weighted figure is 2.18%. The reader-facing 2.1% is the more nearly right number
for "what share of Haitians"; what is missing is any line saying which weighting produced
it, when the file's own computed value is the other one. Same shape for the Church of God:
`note_public`'s 9.4% is the population-weighted figure and the code prints 9.14%.

**None of it touches an override.** All four rest on Spearman rank correlations between the
two surveys, and no weighting of the national margin can move a rank. Recorded because the
levels are quoted to a reader, not because the argument moves.

### 11.4 The 5.7 / 5.9 pair is raw survey against as-drawn, and nothing says so

Recomputed from the `.sav` directly. Raw weighted survey, Artibonite Vodou **5.743%**,
national spread 5.74 points falling to 1.22 without the top. As drawn, after the flat tail
is set to its national rate and the carried categories rescale to fill each department,
**5.93%**, spread 5.93 falling to 1.21.

Both numbers are visible to a reader and they disagree. `branches.py`'s `afrodiasporic.vodou`
note, which is the legend row, says **5.7%** and *"from 5.7 points to 1.2"*; `countries.py`'s
`note_public`, which is the country panel, says **5.9%**. §7 of this file has the same split
inside one section, prose against the block below it. The same thing again in `OVERRIDE[2]`,
*"7.9% of Grand'Anse to 25.8% of Nord"* against the drawn 7.80 and 24.99. Neither number is
wrong; they answer different questions and nothing labels which. Whoever next touches these
should pick one convention and say so once, and the as-drawn one is the one a reader can
check against the map.

### 11.5 The 400-split median is sound method, and the proof is that it demotes as well

Judged on the question of whether it is a bar quietly lowered. It is not, for three reasons.

The bar is untouched at `1.96/sqrt(9)` = **+0.653** and the estimand is unchanged: the
median over 400 random cluster half-splits estimates the same half-sample test-retest
Spearman the parity split estimates, with less draw-to-draw variance. Reducing the variance
of an estimate is symmetric and cannot favour passing.

**And it demonstrably did not.** Against the single parity split the median moves three
verdicts, and one of them goes the hard way:

    Autre protestant   +0.45 -> +0.68   gains its own geography
    Adventiste         +0.50 -> +0.67   gains its own geography
    Aucune             +0.84 -> +0.56   LOSES it

`Aucune` is 6.75% of Haiti, the largest thing this map declines to place, and it had a
**passing** parity split that the median took away. A builder shopping for a looser test
keeps that one.

Third, the design is conservative to begin with: each half carries half the sample, so the
statistic understates the reliability of the full-sample ordering rather than flattering it.

### 11.6 The overrides, and which of the four is thin

Three are strong and there is no circularity in them: ECVH 2001 is used nowhere in the build
except as this witness, and the held-out population check touches no religion column.
Catholic +0.85 (29 of 20,000), Baptist +0.78 (35), Adventist +0.95 (0), Pentecostal bloc
+0.90 (8). Rank correlation is also immune to §11.3's weighting problem, which is the reason
that problem does not propagate.

**Vodou is the thin one and the file says so.** Its whole-ordering Spearman is +0.47 with a
permutation tail of 522 in 20,000, p = 0.026, uncorrected for the seven tests run beside it.
The override does not claim the ordering, it claims the ends, and that is stated. Fair.

### 11.7 The drawn bottom end is not the end that replicates

This is the one finding worth a builder's attention. `note_public` says *"read the two ends
of the Vodou colour and not the ordering"*, and names Artibonite and **Nord-Est** as the pair
both surveys agree on, which is correct: 7.3% and 0.0% in 2001 against 5.9% and 0.09% as
drawn.

But the department a reader's eye takes as the bottom is **Nord-Ouest, drawn at exactly
0.00%** — no Vodou at all in a department of 760,000 people — and Nord-Ouest is where the
two instruments disagree most. ECVH 2001 put it at **1.5%, fourth of nine and above its own
national figure**. Its neighbour Nord is the other disagreement, 1.9% in 2001 against 0.19%
as drawn.

The magnitude is small: Nord-Ouest has **zero Vodou respondents in 1,134**, so a one-sided
95% bound is about 0.26%, roughly 2,000 people, two dots, invisible either way. Vodou's
whole national geography rests on **178 respondents**, 74 of them in Artibonite, and four
departments have three or fewer. So the drawn zero is not really a false claim. What is off
is that *"the two ends"* points at three departments and only two of them are the replicated
pair. `note_public` already names Nord-Est once; naming it again in place of *"the two ends"*
would close it, and that is a sentence rather than a rebuild.

### 11.8 Reader-facing figures, recomputed from people

Every figure in `note_public` reproduces off `data/normalized/ht.csv` by summing people, not
dots. Catholic 47.639, non-Catholic churches 42.992, four departments under half holding
60.10% of Haiti, and Artibonite, Centre and Ouest the only three where the Protestant bloc
outnumbers Catholicism. Baptist Nord 24.99 and Nord-Ouest 24.31 against Grand'Anse 7.80;
Adventist Nord-Ouest 7.16 against Grand'Anse 0.66; Pentecostal Ouest 16.24 against
Nord-Ouest 3.73; Methodist 1.303 and Artibonite 4.51; Autre protestant 9.221; Aucune 6.755;
Vodou 1.500 with Artibonite 105,171 of 178,476, which is 4.9x the next department. Ouest
33.41% of the 2024 base. The Dominican comparison is apples to apples off `do.csv`: Catholic
52.706 and the non-Catholic churches 22.393 + 1.968 + 0.995 = 25.356.

**One slip, fixed here**: Baptists are **17.549%**, and `note_public` said 17.6%. Changed to
17.5% in `countries.py`, `check_md.py` re-run clean. Nothing else moved.

### 11.9 §14 does not apply, and I agree, with a fourth ground and a qualification

The three grounds given are right. Rule 1 holds, the magnitude is IHSI's own survey estimate
and nothing is invented. Rule 2 is not even engaged as a limit: the drawn tier **is** the
state's own publication and the state has published nothing finer for any religion, ever. Ten
units at 1.19 million people identify nobody. Vodou has been legally recognised since 2003.

**The fourth ground is §14.2's own test.** Its third risk turns on whether a map REVEALS or
REFLECTS, and the Artibonite valley lakou are the most publicly documented Vodou sites in
the country. The single loud thing this map says about Vodou is already in every account of
the religion, which is the Borough Park side of that distinction rather than the other one.

**The qualification is that §14.2's SECOND risk is engaged**, in the erasure direction and
not the targeting one. Drawing Vodou at 1.5% in Haiti is the failure mode where getting a
community wrong is itself the harm. §14.2's stated remedy is not to withhold the country; it
is that *"the honesty of the labelling matters more here than on an ordinary data map"* and
that the panel text is *"load-bearing, not boilerplate"*. So the note is doing §14 work. That
raises the bar on the note, it does not raise a question for Anita, and no ask is filed.

### 11.10 On whether the note carries the measured-against-practised distinction

`branches.py`'s node note does, explicitly and well: *"this node counts the people who chose
Vodou over the alternatives, which is a much smaller and differently distributed group than
the people who serve the lwa"*, with the 90%-Catholic-100%-Vodou saying and the *rejete*
campaigns for why an answer carries cost. That is exactly the difference between what the
question measured and what people do, and it sits where a reader meets the legend row.

`note_public` is weaker on the same point, and this is the paragraph on this country most
worth a second look. It does say floor, and it does give the mechanism, *"the people who
named it first"*, which is the right sentence. What it then does is quote **three numbers
between 1.5% and 2.1%**. Three near-identical figures read as convergence on a true value,
and the next sentence saying they share an instrument bias has to fight that impression
rather than being helped by it. A reader could finish the paragraph thinking the honest
number is about 2%.

Nothing in it tells the reader how far the floor sits from practice, and there is a figure
on this project's own record that would: **LAPOP's 4.4%** (§11ad, 7,252 respondents, a card
offering a generic `Religiones Tradicionales` beside a much shorter Christian list). §8
argues correctly that LAPOP's Haitian geography cannot be decoded and must not be drawn, but
its national level is a published number from a different instrument and is the only thing
here that gives "floor" a second end. Two things would fix the paragraph: name the 4.4% as a
survey with a differently worded card, and carry one clause of the node note's sentence about
the group being different from the people who serve the lwa.

Both are `note_public` wording and therefore the builder's under §2, so nothing is edited
here. Recorded because the level on this one category is the thing a reader is most likely to
carry away wrong, and §11.9 is why that matters more here than usual.

### 11.11 Two calls I checked and agree with

**`afrodiasporic.vodou` needs no ask.** The family already names five children and three are
single-country: `revival` is Jamaica only, `orisha` and `spiritualbaptist` Trinidad only, all
added without asking. `afrodiasporic.orisha`'s own note already says *"Read the number as a
floor, not a count… this counts people who chose Orisha over the alternatives"* for the
identical situation, which makes Haiti the fourth instance of a settled pattern rather than a
new question. The census counted `Vaudouïsant` by name at 176,976 people, which is
`AGENT_BRIEF` §2's test for a node a builder owns. And a family named after Vodou not
containing it would be the odder outcome.

**`Autre protestant` -> `christianity.protestant` is right.** The card names six Protestant
bodies and then offers a residual, which is that node's definition, and the contrast drawn
against `do2019.py`'s `EVANGÉLICA` is the correct one: there the single box was the only
general Protestant answer on the card.

## Placement: Petit-Goave's block at Kontur's density cap, capped 2026-09-14 (session `f95259a4-kontur`)

Kontur limits every hex to 46,200 people/km², and a block of hexes at that limit is either a real
dense core or a false concentration (spec §12, "KONTUR'S DENSITY CAP"). **Ouest (HT01) had a
block of 8 hexes at Petit-Goave, 2 of them at the limit, holding 157,473 people and 3.4% of the
department's placement weight.** Kontur put 110,836 people within 1 km of the town's centre, more
than within 1 km of central Port-au-Prince (100,774).

It is `capped` in `kontur_cap.csv`, and `scatter.py` now lowers every hex in the block to the
median density of the populated hexes within 3 km, here 377/km², which leaves it 1,719 people.
**What it costs:** the ring around a small town is farmland, so Petit-Goave itself is now drawn
at farmland density and gets fewer dots than it has people. Before, it was drawn with Kontur's
3.4% of Ouest, many times too many.

Counts did not move: dots per node are identical before and after, 11,894 at 1:1,000 and 1,184
at 1:10,000.

**Superseded the same evening by §12.6: the cap no longer applies to Petit-Goâve.** The row is
`real` and the commune is scaled to its COD-PS share instead (135 to 196 dots).

Port-au-Prince's blocks are registered `real`. **Anse-a-Galets on La Gonave looks like Petit-Goave
and is not changed**: 5 hexes, 3 at the limit, 100,109 people, 2.2% of Ouest, 1.6 times the
town's worldcities figure on an island of two communes. It is `unreviewed`, so every scatter
warns about it. **Superseded the same evening by §12: La Gonâve is corrected at commune level
and the row is now `real`.**

## 12. Placement review after Anita's map look, 2026-09-14 (session `f95259a4-clht`)

Anita, on Chile and Haiti together: *"population distributions look a bit more artificial than
i'd expect."* Haiti already placed dots on Kontur, so the question was which part reads as
artificial. Screenshots of the whole country and of Ouest were taken before changing anything.

### 12.1 What reads as artificial, candidate by candidate

| candidate | verdict | changed? |
|---|---|---|
| Kontur's own blockiness | **No.** At country zoom the rural speckle is dense and even, which is what Haiti's dispersed *lakou* settlement looks like; towns and the Port-au-Prince strip stand out. | no |
| **Anse-à-Galets, La Gonâve** | **Yes, and the loudest thing on the Ouest view.** Two tight clumps on an island otherwise almost empty. | **yes, §12.3** |
| Kontur against COD-PS by department (0.69x to 1.16x) | Real, but it is the population base, not placement. §12.4. | no, recorded |
| Colour steps at department lines | The 10-department religion grain by construction. Finer religion needs the 2005 commune volumes (§10). | no, recorded |

### 12.2 COD-PS prints communes, and what that shows about Kontur

`ht_grid.py` said the projection had no sub-departmental detail. **It has 140 communes**
(`hti_admpop_adm2_2024.csv`, now in `data/raw/ht/`), summing exactly to the department figures.
The workbook's own methodology sheet: a cohort-component projection at ADM-1 from the 2003
census, disaggregated to ADM-2 by iterative proportional fitting, on the 140 communes that predate
the current 149.

Kontur, as drawn (after `kontur_cap.py`), put **6.9% of dots in a different commune** from the one
COD-PS would, population-weighted; log-log r over 140 communes 0.929. The within-department
outliers:

    commune              COD-PS 2024    Kontur    within-dept share
    Anse-à-Galets             70,156   273,795    3.49x
    Gressier                   3,987    45,707   10.24x
    Beaumont                  35,380    70,239    2.25x
    Croix-des-Bouquets       129,879   325,992    2.24x
    Anse Rouge                47,668    83,149    1.71x
    Cabaret                   65,302   119,398    1.63x
    Petit-Goâve              194,867   149,067    0.68x   (after this morning's cap)

**These are three different kinds of disagreement, and only one is Kontur's.** Croix-des-Bouquets
and Cabaret are where **Canaan** was settled after 2010, which a projection off the 2003 census
cannot see; Kontur is right there. Gressier at 3,987 is COD-PS, not Kontur. Anse-à-Galets is the
one where the grid is high and nothing since 2003 explains it.

### 12.3 La Gonâve: scaled to its COD-PS share, not capped

As drawn, Kontur gave **La Gonâve 6.70% of Ouest's weight, about 266 dots, against COD-PS's
2.44%, about 97**, piled into a few hexes: 87% of the island's weight in its densest 40 km², where
the island's median populated hex is 27/km² against 132 on the Ouest mainland.

**Capping the block would not fix it**, measured with `kontur_cap.py`'s own ring-median arithmetic
without touching the registry: the island falls to 4.56% (about 181 dots) and Anse-à-Galets
commune to 174,396, still 2.5 times COD-PS, because a second patch inland (-72.94, 18.78) at
8,000-14,000/km² sits below the 15,000/km² block line. And the cap would draw the island's main
town at 252/km², the ring median, which is Petit-Goâve's known cost.

**What was done instead**: `countries.py::_HT_COMMUNE_LEVEL` names commune `HT0151`, and
`_ht_place_weight` scales its hexes so the commune holds exactly its COD-PS share of Ouest,
keeping Kontur's shape inside it. The block is the town and keeps about 37% of the commune, so
the town is drawn at roughly 26,000 people. `sources/ht_grid.py` now writes `commune` and
`commune_pop` on every hex and asserts the join (every hex in a commune of its own department,
all 140 communes present, communes summing to the departments). The registry row is now `real`
with the reason.

    scatter log   ht: commune HT0151 scaled to its COD-PS share, 6.15% -> 1.76% of HT01's weight
    dots          Anse-à-Galets 240 -> 72 (COD-PS 70); Pointe-à-Raquette 20 -> 22 (27)
                  the 168 moved went to mainland Ouest in Kontur's proportions:
                  Port-au-Prince +35, Delmas +20, Carrefour +19, Croix-des-Bouquets +16 ...
    counts        identical per node, 11,894 dots at 1:1,000 and 1,184 at 1:10,000

**Screenshots, before and after** (1400x900, 1:1,000, same camera; session scratchpad, not the
tree). Whole country: the only visible change is La Gonâve, where the bright clump at
Anse-à-Galets is now a small town cluster; the mainland is unchanged to the eye. Ouest close up:
Anse-à-Galets goes from two dense clumps on an empty island to a modest cluster with a scatter of
dots inland, Pointe-à-Raquette stays small, and the Port-au-Prince strip, Arcahaie and Léogâne are
as before. Petit-Goâve town still reads thin, which is the cap's cost noted below.

**The bar for adding a commune to that dict**: every grid tested is over by more than 2x AND
nothing since 2003 explains it. Croix-des-Bouquets and Cabaret fail it (Canaan). Gressier fails
it the other way. Beaumont (Grand'Anse, 2.25x on Kontur) was not added: constrained WorldPop has
it at 18,188, against Kontur's 70,239 and unconstrained WorldPop's 70,765, so the grids do not
agree and there is no case.

**Petit-Goâve is the same shared error and was left as capped.** All three grids have it at
292,000 to 305,000 against COD-PS's 194,867, and the cap leaves the commune at 0.68x its share
(138 dots against 195) with the town at farmland density. A commune scale would do better, as it
did for La Gonâve, but the cap was the supervisor's decision this morning and is recorded here as
the candidate rather than reversed. **Superseded the same evening by §12.6: Petit-Goâve is now
commune-scaled and its row is `real`.**

### 12.4 Would another grid be better? Measured: no

WorldPop 2020 for Haiti, constrained (`BSGM/HTI/hti_ppp_2020_UNadj_constrained.tif`, 0.9 MB) and
unconstrained (`Global_2000_2020/2020/HTI/hti_ppp_2020_UNadj.tif`, the control, spec §12's
two-raster test), zonal sums on the 140 COD-AB communes. Scratch only; nothing in the build reads
them.

    department / COD-PS, normalised     Kontur   WP constrained   WP unconstrained
    Ouest                                1.17        1.22             1.19
    Artibonite, Nord-Ouest               1.07        1.05-1.06        1.06-1.07
    Nord                                 0.98        0.93             0.96
    Grand'Anse                           0.93        0.75             0.92
    Nord-Est                             0.83        0.76             0.80
    Centre                               0.77        0.77             0.76
    Sud-Est                              0.74        0.74             0.73
    Sud                                  0.73        0.75             0.73
    spread, max/min                      1.62        1.66             1.64

    in another commune than COD-PS       6.9%        7.4%             6.4%
    Anse-à-Galets                     273,795     244,402          254,858     (COD-PS 70,156)
    Croix-des-Bouquets                325,992     292,096          285,251     (129,879)
    Gressier                           45,707      41,513           42,267     (3,987)
    Petit-Goâve                       149,067*    292,091          292,951     (194,867)
    La Gonâve in its densest 10 km²     55.1%       99.9%            37.4%

    * after this morning's cap; 305,000 before it

**The three grids agree with each other on exactly the communes where they disagree with the
census**, so the La Gonâve and Petit-Goâve errors are an input they share, not something Kontur
did alone, and switching grids would move neither. Constrained WorldPop is also worse on shape,
putting 99.9% of La Gonâve on one town's building cells. **GHSL was not tested.**

### 12.5 The department density step, recorded and not changed

All three grids put the same departments low against COD-PS, **and they are the ones the
projection grew fastest since 2003** (§9: Centre +1.6 points, Sud-Est +1.2, Sud +0.8), while Ouest,
which the projection shrank by 3.6 points, is high on every grid. Because dots per department are
fixed by COD-PS and placed by the grid, a rural hex of the same Kontur density draws about 1.6
times as many dots in Sud, Sud-Est or Centre as in Ouest, so the speckle can step up across those
department lines. That is the population base, which sets counts, so it is not a placement fix
and nothing here changes it. If the base were ever revisited, the COD-PS metadata names a
**"2022 Haiti Population Data Grid" from IHSI and UNFPA** as the thing it was checked against; not
chased.

### 12.6 Petit-Goâve: moved from the cap to the commune scale, 2026-09-14 (session `f95259a4-house`)

The supervisor's decision, on §12.3's finding. Commune `HT0122` is added to
`countries.py::_HT_COMMUNE_LEVEL` and its `kontur_cap.csv` row goes from `capped` to `real`, so the
block keeps Kontur's shape inside the commune and the commune holds its COD-PS share of Ouest.

**It does not clear the dict's own 2x bar, and the comment above the dict now says so.** Raw
Kontur has the commune at 304,821, 6.62% of Ouest's weight against COD-PS's 4.90%, so 1.35x
(Anse-à-Galets is 3.37x on the same measure). It comes in on a second route, written above the
dict: its town block was already registered false, both WorldPop 2020 rasters have the same excess
(292,091 and 292,951), and the cap drew the town at farmland density while leaving the commune
under its share.

    scatter log   ht: commune HT0122 scaled to its COD-PS share, 6.91% -> 4.90% of HT01's weight
                  (6.91% rather than 6.62% because La Gonâve's scale runs first)
    dots          Petit-Goâve commune 135 -> 196 (COD-PS 194,867)
                  within 1 km of the town centre 8 -> 48, within 3 km 50 -> 134;
                  within 1 km of central Port-au-Prince, 109
                  the 61 came out of the rest of Ouest: Port-au-Prince -15, Carrefour -13,
                  Grand-Goâve -13, Tabarre -8, Cabaret -7, Croix-des-Bouquets -7
    counts        identical per node and per tier, 11,894 dots at 1:1,000 and 1,184 at 1:10,000
    cap           the Haiti scatter now reports 5 blocks registered `real` and caps none

The rest of Ouest should each lose about 1.7%, and most do. Grand-Goâve's 9% is more than that,
but it is within what placement moves between runs: Anse-à-Galets, whose share is fixed, went
74 -> 71 in the same pair. Dots were counted per commune by joining each dot to its hex's
`commune`; 4-6 dots per run land in no hex because coordinates are rounded to 4 decimals.
No screenshot was taken.
