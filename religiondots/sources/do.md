# Dominican Republic — ONE's ENHOGAR-MICS6 2019, 32 provinces

Built 2026-09-08. `sources.md` §9cf is the write-up; this file is the working record.

    sources/do_geo.py    COD-AB ADM2 (32 provinces) + the 2022 census populations + the join
    sources/do_grid.py   Kontur 400 m hexes, the placement layer
    sources/do.py        the decode, the checks, and data/normalized/do.csv
    taxonomy/do2019.py   six answers -> six nodes, one of them new (`other.do`)

---

## 1. What this country was going to be built from, and why it is not

`queue.md` §11ad listed the Dominican Republic as the largest of the nine AmericasBarometer
countries: **8,904 respondents pooled over six waves, 29 of 32 provinces**. §9bn refused it.
Three provinces are never sampled at all and seven more appear in one wave-half only, several
on n under 20, so applying Ecuador's rule would have drawn ten of thirty-two provinces at the
national rate. Anita, 2026-09-08: *"probably cant do DR without additional data."*

The additional data was never far away and nobody had asked for it. **ENHOGAR** — *Encuesta
Nacional de Hogares de Propósitos Múltiples* — is the Dominican household survey series, run
by ONE since 2005, and the 2019 round was fielded as **MICS6** in partnership with UNICEF.
MICS6's household questionnaire carries `HC1A`:

> ¿A cuál religión pertenece el jefe o la jefa del hogar?
> 1 CATÓLICA · 2 EVANGÉLICA · 3 ADVENTISTA · 4 TESTIGO DE JEHOVÁ ·
> 6 OTRA RELIGIÓN (Especifique) · 7 NINGUNA RELIGIÓN

    LAPOP AmericasBarometer 2010-2023 pooled     8,904 respondents,  29 of 32 provinces
    ENHOGAR-MICS6 2019                          31,488 households,   32 of 32 provinces
                                                96,968 people in them

Eleven times the sample and every province. The thinnest province, **San José de Ocoa at
1,759 people**, is a fifth of LAPOP's whole national pool on its own, and it is one of the
three LAPOP never sampled.

---

## 2. How the source was found, because the route is reusable

Neither `one.gob.do` nor `anda.one.gob.do` can be read by anything this project has: both sit
behind a Cloudflare challenge that answers **403 to curl and to WebFetch alike**, on HTML and
on PDFs and on the microdata itself. That is a wall and not a dead host, and the way round it
was three steps.

**1. CELADE, not ONE.** `sources.md` §11's REDATAM sweep had already named `reddom` on the
shared host `prod.redatam.org` and never opened it. It is a real Dominican deployment, on
`/bindom/`, and it is not behind Cloudflare because it is hosted by ECLAC. Its index lists
four census bases and eighteen ENHOGAR rounds — and **only `CPV2010` is actually installed**;
every other link answers `Directorio no encontrado`, which is Albania's empty-folder pattern
(§9bs) again. So the instance is useless as a tabulator here and decisive as a dictionary: the
CPV2010 base ships its own downloadable `.dic`, and there is no religion variable in it.

**2. The Wayback CDX, prefix-matched on the file tree rather than on the site.** ONE publishes
its microdata as a plain directory tree at `one.gob.do/catalogo-datos/`, not through ANDA, and
a CDX prefix sweep of that path returns **every file the archive has ever seen there**: 18
ENHOGAR rounds with their SPSS bases, questionnaires and HTML codebooks, the 2022 census
microdata, ENIGH, ENI, ENESIM, the vital statistics. Fetching all 129 archived codebooks and
grepping them for `religi|creenc|culto|confesi` is what found `HC1A`, and it took one script.

**3. The `.sav` for the labels, the `.csv` for the data.** The published CSVs carry codes and
no labels. The `.sav` does, and the archive's copy is **truncated at exactly 1,048,576 bytes**
— its capture cap, not a damaged file. SPSS writes the label dictionary into the header, so
`pyreadstat.read_sav(..., metadataonly=True)` reads all 298 variables' labels out of a file
whose cases are unreachable. Both `sources/do.py` and `sources/do_geo.py` assert against it.

The same sweep answered three other questions for free, and they are in §5.

---

## 3. The census half, closed twice

The Dominican Republic is absent from the UNSD oracle, which proves only that no tabulation
was forwarded (§12). Two harder checks:

* **2010.** `cpv2010_repdom_pub.dic`, downloaded from the REDATAM portal's own link, is 121 KB
  and has **no religion variable**. The only strings matching `religi` are a dwelling type
  (`Institución religiosa`) and a community-form question about which building is the
  emergency shelter (`ALIAS IGLESIA`).
* **2022.** ONE publishes the X CNPV person microdata openly. Its codebook runs `P25_ORDEN`
  through `P67_ANO` and there is **no religion question**. `P64` is the self-identification by
  *facciones, color de piel y otras características culturales*, which is an ethnoracial
  question and must not be used as if it were religion.

---

## 4. The three decisions worth arguing with

### 4.1 The provinces are COD's ADM2, and the population is the census

**ADM1 is the ten planning regions.** Reading `dom_admin1.shp` the way every other Latin
American country here does gives ten polygons and no error, and ten units for 10.8 million
people is a country drawn three times coarser than its own survey measured it.

**COD-PS 2023 is not the population base**, and the reason is a sharper version of Ecuador's
(§9bn). HDX's own methodology field calls it a projection from a "2015 census" that does not
exist; the Dominican Republic counted in 2010 and again in November 2022. Nationally it is
**0.56% under ONE's census, better than the 3.4% error that disqualified Ecuador's**. Per
province:

    San José de Ocoa   census    69,082   COD-PS    52,687   -23.7%
    La Altagracia      census   446,060   COD-PS   375,872   -15.7%
    Hato Mayor         census   100,133   COD-PS    85,738   -14.4%
    ...
    Distrito Nacional  census 1,029,110   COD-PS 1,062,476    +3.2%
    Santo Domingo      census 2,769,588   COD-PS 3,054,470   +10.3%

A 34-point spread under a national agreement of half a point. The projection carried forward
the drift into the capital the 2010 census had been seeing, and the 2022 count did not find
it. Drawing on COD-PS would have put 285,000 people in Santo Domingo who are not there.

### 4.2 The join is on the name, and the code join is safe-wrong rather than dangerous-wrong

ENHOGAR's `HH7A` is ONE's official province numbering (Distrito Nacional first, then the other
thirty-one alphabetically), which is also what the 2022 census microdata's `PROVINCIA` uses.
COD's ADM2 pcodes are ordered by region and then within it. Pairing sorted pcodes with 1..32
gets **3 of 32 right**, all coincidence, and the three (Dajabón, Espaillat, San Pedro de
Macorís) are scattered rather than clustered at either end, so no spot check would survive it.
That is the opposite of El Salvador's two of fourteen (§9bl) and Uruguay's nine of nineteen
(§9ce), where the coincidences were exactly where a reviewer looks. `check_code_join()` pins
it at three anyway, because a re-cut that made it *mostly* right is the dangerous direction.

One alias, and only one: ENHOGAR writes **Bahoruco**, COD-AB and the census both write
**Baoruco**.

### 4.3 The Witnesses are drawn at the national rate, and there is no override

Split-half on cluster parity across ENHOGAR's 1,747 sampling units, bar +0.35 on 32 provinces:

    CATÓLICA                     52.78%   +0.94
    EVANGÉLICA                   22.34%   +0.94
    NINGUNA RELIGIÓN             20.57%   +0.91
    ADVENTISTA                    1.94%   +0.55
    OTRA RELIGIÓN                 1.38%   +0.54
    ------------------------------------------------ bar +0.35
    TESTIGO DE JEHOVÁ             0.99%   -0.02

**The split is on the CLUSTER and not on the person**, which matters: two halves drawn
person-by-person out of the same 1,747 neighbourhoods would agree far better than two real
samples of the country and the test would pass everything.

The Witnesses' chi-square is p=4e-27 and it is **not** a licence. It is computed on unweighted
persons inside clusters and takes no account of the design, so for a category at 1% it is
measuring household clustering as much as geography. Uruguay's `Judio` exception (§9ce) rested
on one department holding nine tenths of the cell on 80,196 respondents; there is no such
argument here. No `UNDER_BAR` in this file.

**And a country with exactly ONE failing category cannot use `lapop.build()`'s residual
construction.** That function gives each unit's residual to the failing categories at their
national relative proportions, which is right for several and a no-op for one: with a single
failing category the residual *is* its own measured share, so it would have drawn the
Witnesses exactly where the survey put them. It also divides by zero on Hato Mayor, whose
measured Witness share is 0.00%. `sources/do.py` sets the tail to its national share directly
and rescales the five stable shares to fill the rest.

---

## 5. What the sweep found that this country did not use

* **`Estadísticas Vitales — Matrimonios`, 2001 to 2023, carries a religion variable.** It is
  the *rite* of the marriage, not the religion of a person: `Católico`, `Evangelico`,
  `Adventista del Séptimo día`, `Testigo de Jehová`, `Otro religioso`, civil. It is a
  register rather than a sample, it is published per year and per province, and it therefore
  offers something no survey can: an annual, complete, provincial series on exactly the four
  bodies `HC1A` names. It is **not** a population measure (it counts marriages, and who
  marries in church is not who belongs to one), so it cannot replace this build. As a
  cross-check on the Adventist and Witness geographies it would be excellent, and it is
  the first thing to try if anyone wants to reopen §4.3.
* **`ENHOGAR 2014` (MICS5) has the same question at a coarser card** — `ethnicity`, labelled
  *Religión del jefe del hogar*, with only Católica / Evangélica / Otra religión. Two rounds
  five years apart would give a trend and a real split-half across waves.
* **`ENHOGAR 2018` asked adolescents directly**: `AD118. ¿A cuál religión pertenece usted?`,
  which is the one Dominican source found that asks a *person* rather than a household head.
  Adolescents only, so it cannot build the map, but it is the obvious way to measure how much
  the household-head assumption costs.
* **`ENI 2017`, the Encuesta Nacional de Inmigrantes, asks `¿Cuál es su religión?`** of
  immigrants, with Católica / Evangélica / Adventista and an `¿Asiste usted a la iglesia o
  templo?` follow-up. That is the Haitian-origin population asked directly, and it is the
  only source located that could say anything about the group this map's south-western
  provinces are largely about.

---

## 6. Open

**The household head is the ceiling of this build.** Everything drawn is a household
attribute applied to its members. `ENHOGAR 2018`'s adolescent module and `ENI 2017` are the
two located sources that could measure the cost of that, neither is wired, and both are small.

**The level is 2019.** Dominican Catholic identification has been falling fast on every
instrument that has measured it twice, so a round fielded now would draw a different country.
ENHOGAR 2021, 2022 and 2024 are all published and **none of them carries the religion
question**; it belongs to the MICS rounds. The next MICS is the next chance.

**`other.do` cannot be read.** The specify text for `OTRA RELIGIÓN` is not in the public
microdata, unlike Uruguay's `e29_2`. Dominican Vodú has no box anywhere on the card and its
practitioners answer Catholic, which is the Haitian and Cuban pattern; nothing here counts it
and `other.do` should not be read as where it is hiding.

**The Cloudflare wall is the operational risk.** Every ONE URL in `sources/do.py` and
`sources/do_geo.py` fetches through `web.archive.org/web/<ts>id_/`, because the canonical URLs
403. If someone with a browser can reach `one.gob.do` directly, the canonical URLs are the
tails of those strings and the full (untruncated) `.sav` files are worth having.

---

## 7. Review, 2026-09-08 (reviewer, session `967ffe99…-do-rev`)

Nothing here needs rebuilding. Everything in §1 to §6 that could be recomputed was recomputed
from the raw files rather than from this file's account of them, and it all reproduces. What
follows is the part that is new.

### 7.1 The household head costs 12.7 points of Catholicism, and it is measurable

§6 says the head question is the ceiling and that `ENHOGAR 2018`'s adolescent module is the
way to price it. It was fetched and it prices it. `Adolescentes_ENH2018.csv` is on the archive
at `20211114160336` and its `AD118` is answered by **4,934 women aged 15 to 19**, which is
98% of the women that age on the 2018 roster; despite the file's name the module is women
only, `H202` is 2 for every respondent and the one weight in the file is
`Mujeres15_19_Factor_Exp`, so the comparison has to be cut to women and not to adolescents.
Its card is 1 Católica, 2 Evangélica, 3 Adventista, 4 Pentecostal, 5 Ninguna, 6 Otra, with no
Witness box, so Evangélica and Pentecostal are added and the Witnesses sit inside Otra.

Against the same group as this map draws them, women aged 15 to 19 in ENHOGAR-MICS6 2019:

    cell                              drawn (head's answer)   their own answer    cost
    Catholic                                        49.76%             37.11%   -12.66
    Evangelical incl Pentecostal                    23.40%             23.46%    +0.06
    Adventist                                        2.48%              2.12%    -0.36
    Other and Witnesses                              2.72%              2.53%    -0.19
    No religion                                     21.64%             34.78%   +13.14

**So it is biased, and the bias is one axis: Catholic against no religion.** The evangelical
cell is the one the head question gets right, to six hundredths of a point, which is the
opposite of what a reader might assume about a country with a large evangelical population.
Nothing moves between the religions; what moves is that young people in religious households
who report no religion themselves are drawn in their household's column.

**And it is a level shift rather than a distortion.** Regressing the 2018 self-reported
provincial Catholic share on the 2019 attributed one over all 32 provinces gives slope
**1.007**, intercept **-12.66**, r **0.872**, and the size of the gap is uncorrelated with the
level (Spearman +0.04). The province ranking is +0.87 for Catholic and +0.81 for no religion.
The map's shape survives the head question; its levels do not.

**Corroborated on a third instrument.** LAPOP's DR respondents self-report Catholic 43.7% at
18-24 against 69.4% at 65 and over, and no religion 26.5% against 7.0%; in the 2018 and 2023
waves alone the 18-24s are 38.1% Catholic and 30.8% none, which is AD118's 37.1% and 34.8% on
a different survey with a different card. The age gradient in *self-reported* Dominican
religion is real and steep, and the head question flattens it.

**Scale.** Women 15 to 19 are 4.24% of the drawn population, so the measured piece alone is
worth -0.54 points of the national Catholic share. It is a floor and not an estimate: 26.6%
of the country is under 15, 17.4% is 20 to 29, and **52.4% of Dominican adults are non-heads
drawn on somebody else's answer** (67.5% of all drawn people; 20.6% of households are one
person and carry no attribution at all). Nothing published prices those bands and §14.4 says
not to invent the magnitude, so it is not corrected and not extrapolated.

`note_public`'s second paragraph previously ended by saying nothing in the source said how
often the mismatch happened or which way it leaned. That sentence has been replaced with the
measurement above, in the form §3.5 asks for. Its example was also changed: it used a Catholic
mother with an evangelical son, and the evangelical cell is precisely the one that does not
move.

### 7.2 The §3.5 lean check was not run, and the hole does not lean

The only thing this build excludes is **3,494 households of 34,982 that did not complete the
interview**, 9.99%, and they bring no members into the person file at all. Their provincial
rate against each drawn share over the 32 provinces, Spearman with a 2,000-draw permutation p:

    CATOLICA    -0.10  p 0.58      ADVENTISTA  +0.19  p 0.30
    EVANGELICA  +0.11  p 0.53      OTRA        +0.26  p 0.15
    NINGUNA     -0.02  p 0.91      TESTIGO     +0.34  p 0.06

Nothing survives being one of six tests at n=32, and the only one near the line is the
category that is not drawn on its own geography anyway. So there is no lean sentence to write,
which is the answer §3.5 wants recorded rather than left unasked.

### 7.3 The Witness decision is right, and it is right for a second reason

§4.3 rests on one split, cluster parity. A single split can be unlucky, so the same test was
re-run on **400 random cluster splits**:

    CATOLICA    parity +0.938   mean +0.945   passes 100.0% of splits
    NINGUNA     parity +0.910   mean +0.914   passes 100.0%
    EVANGELICA  parity +0.937   mean +0.883   passes 100.0%
    OTRA        parity +0.535   mean +0.573   passes  99.0%
    ADVENTISTA  parity +0.554   mean +0.527   passes  96.5%
    TESTIGO     parity -0.024   mean +0.156   passes   3.5%

Parity was slightly unkind to the Witnesses, whose central value is +0.16 rather than -0.02,
and slightly kind to the evangelicals. Neither changes a decision: the Witnesses fail on 96.5%
of splits and their central estimate is less than half the bar, and Adventists pass on 96.5%,
which is a real pass but not a comfortable one and is worth knowing before anyone leans on the
Adventist geography.

**A second instrument agrees.** LAPOP has its own Witness box (`q3c` = 12, 56 respondents) and
its provincial Witness shares correlate with ENHOGAR's *measured* ones at Spearman **+0.11**
(Pearson -0.02) over the 29 shared provinces. On the same provinces and the same test the
Catholic cells agree at +0.85 and the evangelical ones at +0.81. Two unrelated surveys both
fail to find a Dominican Witness geography, which is stronger evidence for the flat draw than
the split-half alone.

On the question of whether this should have an `UNDER_BAR` entry: no, and the mechanism is the
other way round. `UNDER_BAR` in `sources/uy.py` and `OVERRIDE` in `sources/gt.py` exist to draw
a category on its own geography *despite* failing, which is why they carry an argument.
`lapop.stability`'s docstring states the default in terms: a failing category gets "the national
rate inside each unit's own residual and a sentence in `note_public`, not deletion". This file
did exactly that and the sentence is there, so there is nothing to declare.

### 7.4 What reproduces, and one figure that did not

Every reader-facing figure recomputes from `data/normalized/do.csv`: 52.7055% Catholic, La
Romana 25.315% to Hermanas Mirabal 82.771% (57.5 points), under half in 13 of 32 provinces
holding 50.93% of the population, exactly six provinces above 76% and they are the six named,
La Romana 49.341% / Samaná 43.330% / La Altagracia 33.112% / San Pedro de Macorís 32.439%,
Pedernales 42.909% and Baoruco 42.704% against La Vega 4.902%, evangelicals ahead of Catholics
in exactly five provinces and they are the five named, Adventists 1.968% with Hato Mayor
5.092% and San Cristóbal 4.113% against San José de Ocoa 0.151%, Witnesses 107,147 at a flat
0.9947%, and 336,610 people per province against the entry's "337,000". The Adventist
comparisons hold on the map's own drawn shares: Mexico 0.630%, Peru 1.524%, Grenada 13.266%,
Antigua 12.488%, Jamaica 12.305%, so the correction §9cf describes landed in all four places
and no "largest in Latin America" survives anywhere in the tree.

The one that did not: `taxonomy/do2019.py` said La Romana was "the only province in the country
where this cell is the largest". `EVANGÉLICA` is the largest cell in four provinces (La Romana,
Samaná, San Pedro de Macorís, La Altagracia); La Romana is where it is highest. Corrected in
place, and `countries.py` already had it right.

### 7.5 §4.1 and §4.2 verified independently

COD-PS 2023 against the census, recomputed off `dom_admpop_adm2_2023.csv` and
`do_lookup.csv`: national **-0.5603%**, per province **-23.73%** (San José de Ocoa) to
**+10.29%** (Santo Domingo), a 34.0-point spread, and Santo Domingo's own surplus is
**284,882** people, so §12's citation of "about 285,000" is exact rather than rounded. Across
all 32 provinces the total misplacement is 353,294 people on the half-absolute convention.
The ADM1 point holds literally: `dom_admin1.shp` has **10** features and they are the planning
regions, `dom_admin2.shp` has **32** and they are the provinces.

The LAPOP cross-check reproduces on a rebuild from `lapop_slim.feather` rather than from
`sources/do.py`: 8,904 respondents, 29 of 32 provinces, Catholic **+0.837**, non-Catholic
Christian **+0.767**, no religion **+0.759**, none of 20,000 permutations reaching any of the
three, and the three provinces LAPOP never sampled are `hh7a` 10, 16 and 31, Independencia,
Pedernales and San José de Ocoa. Two details worth having for the LAPOP-built countries:
§9cf's non-Catholic Christian level of 25.9% excludes LAPOP's Mormon box (`q3c` = 6, 0.29%),
which is right against a card that has no Mormon answer but is worth stating; and the
agreement is limited by LAPOP's thin cells rather than by disagreement, since dropping the
eight provinces with the fewest LAPOP respondents (Dajabón has **7**) lifts the three to
+0.929, +0.852 and +0.892 on the remaining 21. Its ENHOGAR-side levels of 52.8 and 25.3 are
the survey's own weighted figures; the drawn CSV gives 52.71 and 25.36 because it lays the
province shares on census populations.
