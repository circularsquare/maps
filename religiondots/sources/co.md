# Colombia — LAPOP AmericasBarometer, waves 2010–2023

Built 2026-09-14 by session `f95259a4-co`, from `sources.md` §11ap's scouting. Write-up in
`sources.md` §9dk. Code: `sources/co.py` (build), `sources/co_geo.py` (boundaries and population),
`sources/co_grid.py` (Kontur hexes), `taxonomy/co2023.py` (mapping), `countries.py::_co_counts`.

## 1. Why a survey

No Colombian census asks religion (§11ac, §11ae). §11ap corrected §11ae on DANE's catalogue: the
Encuesta de Cultura Política asks `P6945` in 2015-2023, but publishes five regions and leaves out
the *nuevos departamentos* and San Andrés, so it cannot place anything at department level. The
Encuesta Nacional de Diversidad Religiosa 2019 sampled all 33 units and publishes seven regions
with no microdata. LAPOP is the only department-level source found.

## 2. The files

| file | what | from |
|---|---|---|
| `data/raw/lapop/Grand_Merge_2004-2023_LAPOP_AmericasBarometer_v1.0_FREE.dta` | the merge, already on disk | LAPOP |
| `data/raw/co/lapop_co.feather`, `lapop_co_labels.csv` | Colombia's rows with `municipio`, `upm`, `cluster`, `estratopri`, `tamano`, which `lapop.USECOLS` does not carry | `python sources/co.py --fetch` |
| `data/raw/co/col-administrative-divisions-shapefiles.zip` | COD-AB, MGN 2020-04-16, ADM1 33 units | HDX `cod-ab-col` |
| `data/raw/co/col_admpop_adm1_2025.csv`, `col_admpop_adm2_2025.csv` | COD-PS 2025, 53,216,592 people; ADM2 is the municipality name witness | HDX `cod-ps-col` |
| `data/raw/co/kontur_population_CO_20231101.gpkg.gz` | placement | Kontur |

COD-PS 2025 is DANE's own post-census projection (HDX explanatory note: source DANE, baseline 2018
census, reference year 2025), so there is no office count to prefer over it, unlike Ecuador.

COD-AB's ADM1 shapefile cannot be read with `engine="fiona"`: a date field holds year 0 and fiona
raises `year 0 is out of range`. `co_geo.py` reads it with pyogrio and asserts the 33 features.

## 3. The department labels, checked three ways

The parent brief asked for this after §11ap found that Honduras's merge stamps the 2023 wave's
labels on every wave. Colombia is not affected, and `co.py::check_labels` asserts it on every run.

1. **`municipio`, 2012-2023.** 800,000 plus the DANE municipality code. Its first three digits
   equal `prov` for every respondent except **24 interviews in Florida (876275, Valle del Cauca)
   printed under 852 Nariño in 2012.** Their `upm` is 7627500, Florida again, and the same
   municipality sits under 876 in 2014, 2018 and 2023. They are moved to Valle del Cauca; the
   exact set is asserted. All 51 sampled municipalities' labels match COD-PS ADM2's name at the
   DANE code, with three short-form aliases (Bogotá, Cartagena, Tumaco for Bogotá D.C., Cartagena
   de Indias, San Andrés de Tumaco).
2. **`upm`, 2010**, the wave with no `municipio`. It holds DANE municipality codes (5001 Medellín,
   76001 Cali, 97001006 Mitú). For all 1,266 respondents outside Bogotá, in 52 codes, the
   department prefix equals `prov - 800`. Bogotá's 231 have `upm` 1-4 and `tamano` = *Capital
   Nacional* on every row. 2012's `upm` is the same code times 100.
3. **Sample share against COD-PS, wave by wave**: 2010 r=+0.887, 2012 +0.961, 2014 +0.959, 2018
   +0.956, 2023 +0.970; none of 20,000 random pairings reaches any of them. Pooled over 26 units,
   `lapop.held_out`: r=+0.964, best random +0.867.

Also: DANE's department codes are not contiguous, and the merge's codes include 885 and 897, which
no private 1..N ordering like Honduras's would produce. LAPOP's six design regions (`estratopri`)
nest the departments exactly in every wave, and that is asserted too.

## 4. Answer codes by wave

Weighted %, the five waves:

| code | 2010 | 2012 | 2014 | 2018 | 2023 |
|---|---|---|---|---|---|
| 1 Católico | 75.75 | 75.55 | 71.46 | 67.81 | 66.95 |
| 5 Evangélica y Pentecostal | 6.21 | 5.37 | 6.89 | 9.87 | 9.65 |
| 2 Protestante tradicional | 7.55 | 9.74 | 5.48 | 7.66 | 5.71 |
| 4 Ninguna (creyente) | 9.22 | 6.38 | 9.56 | 10.06 | 9.44 |
| 12 Testigos de Jehová | 0.40 | 1.07 | 2.94 | 0.00 | 0.00 |
| 77 Otro | 0.00 | 0.00 | 0.00 | 2.76 | 4.93 |

**No Honduras-2016-style shift.** 2014 is the odd wave for the small codes (Witnesses 2.94%,
Eastern religions 1.40%, Traditional 1.54%), but the Christian total and the Catholic series move
smoothly through it, and no answer is concentrated in one sampling cluster (the largest (wave,
cluster) cell holds at most 6% of any answer with more than five respondents). The 2018 card
change (`ec2023.py`) applies: Witnesses, Mormons and Jews have no box in 2018 and 2023, and `Otro`
has none before 2018. Colombia has no 2016 wave.

## 5. Which answers carry their own geography

§9cy's construction plus the Sweden chi-square veto (spec §12), on unweighted counts, 2,000 per-wave
permutations, seed 0. **Five waves give ten distinct 2-against-3 halvings.** `cab.stability`'s
`if 0 in a` filter would keep four of them: it drops mirror images correctly when the wave count is
even and discards distinct halvings when it is odd. `co.py::_halvings` enumerated all ten; since
2026-09-14 that is `sources/stability.py::halvings`, which `co.py::stability` and `cab.stability` both call.

At the **22 departments sampled in every wave**:

| answer | n | median rho | null 95th | p | chi-square p | verdict |
|---|---|---|---|---|---|---|
| Católico | 5,335 | +0.619 | +0.257 | 0.0005 | 1e-22 | department share |
| Evangélica y Pentecostal | 568 | +0.687 | +0.267 | 0.0005 | 9e-21 | department share |
| Ninguna (creyente) | 668 | +0.470 | +0.267 | 0.0020 | 4e-11 | department share |
| Protestante tradicional | 533 | +0.033 | +0.281 | 0.44 | 4e-06 | national rate |
| Testigos de Jehová | 66 | +0.327 | +0.320 | 0.048 | 0.29 | **refused**: rank passes, chi-square fails |
| the other six | | | | ≥0.095 | | national rate |

The single chronological halving (2010-2012 against 2014-2023, weighted) gives §11ap's figures
again: Catholic +0.71, evangelical +0.70, Ninguna +0.56, Protestant +0.06, Witnesses +0.34.

At **LAPOP's 6 design regions** (Atlántica, Bogotá, Central, Oriental, Pacífica, Antiguos
territorios nacionales), all 26 sampled departments: Catholic (+0.83, p=0.002) and Ninguna (+0.66,
p=0.019) pass; evangelical +0.54, p=0.06, fails; everything else fails. **Nothing passes at the
region that failed at the department**, so the mixed-level construction places no answer at the
coarse level (Uzbekistan's result, §9di).

**Protestant and evangelical were tested together, as §11ap suggested, and kept apart.** The union
passes at the department (+0.576) but more weakly than the evangelical box alone (+0.687), and the
two boxes' pooled department shares are unrelated (Spearman +0.08). If respondents were swapping
boxes by place, those shares would be negatively related. So the union's pass is the evangelical
box's, and drawing Protestants on the union's geography would give them a pattern the survey did
not find for them.

## 6. Four departments assumed, seven blank

Ecuador's line (§9bn; Anita, 2026-09-08, on Carchi and Galápagos): whether anything measured the
place.

- **Measured in one round only**: La Guajira (2023 only, n=23, all Manaure), Quindío (2010, n=12),
  Casanare (2010, n=26), Vaupés (2010, n=14). 2,139,687 people, 4.02%. One wave measured each, so
  there is a reading to anchor on, and no second one to test it against.
- **Redrawn 2026-09-14 evening (session `f95259a4-co2`) on spec §12's one-round rule**, after §10's
  review and the supervisor's decision in `queue.md` (Anita deferred it, leaning to "use the most
  granular thing available"). For the three answers drawn on department shares, a one-round
  department takes its LAPOP design region's shares when the region predicts its own every-round
  departments better than the country does. Leave each out, predict its pooled shares from the rest
  of the region's every-round departments and from every other every-round department, sum the
  absolute error over the three answers, and require a lower mean **and** closer for more than half.
  `co.py::region_fallback`, asserted as `ON_REGION`. The other eight answers are the national rate
  inside the residual, as in every department.

  | region | one-round | every-round depts | region closer | mean error, region / country | random-group p | drawn |
  |---|---|---|---|---|---|---|
  | Atlántica | La Guajira | 6 | 6 | 6.11 / 15.15 | 0.001 | **region** |
  | Oriental | Casanare | 5 | 4 | 9.67 / 12.62 | 0.059 | **region** |
  | Central | Quindío | 5 | 2 | 15.35 / 15.41 | 0.279 | national |
  | Antiguos territorios nacionales | Vaupés | 2 | 1 | 19.66 / 15.90 | 0.417 | national |
  | Pacífica | none | 3 | 0 | 18.31 / 12.95 | 0.800 | |

  Errors are summed points over Catholic, evangelical and Ninguna; the random-group p (printed,
  not deciding) is how often a random set of as many every-round departments beats the country by
  as much. Drawn now, Catholic / evangelical / Ninguna: **La Guajira 64.4 / 13.0 / 9.2** (was the
  national 71.6 / 7.6 / 8.9; about 74,000 fewer Catholics and 56,000 more evangelicals), Casanare
  77.5 / 6.0 / 6.8 (about 27,000 more Catholics). Their tails are 13.3% and 9.7% against 11.9%.
- **Quindío is the close call.** The means are 0.06 points apart and the majority clause keeps it
  national. On the mean alone it would switch, and its own 12 interviews (10 Catholic) sit nearer
  Central than the country (summed error 10.0 against 13.1).
- **Casanare is the weaker of the two region draws, and drawn on the rule anyway.** Oriental's win
  comes from its Andean departments. Meta, the one llanos department and Casanare's neighbour, is
  the one it misses worst (11.7 against 2.7), the random-group check is p=0.059, and Casanare's own
  26 interviews (16 Catholic, 8 traditional Protestant) sit nearer the country (21.2 against 19.0).
  All three are weak on their own; reversing it is one constant.
- **The first build's version of this test, and why it came out even.** It averaged the
  leave-one-out over all 21 departments outside Bogotá and four answers (Catholic 5.88 vs 7.26 pp,
  evangelical 2.78 vs 3.90, Ninguna 3.40 vs 3.20, Protestant 2.72 vs 2.44) and kept the national
  rate. That mixed Atlántica's clear win with Pacífica's clear loss, and counted traditional
  Protestant, which is at the national rate whichever fallback is used. §10's Atlántica figures
  (5.0-7.6 against 8.3-19.2) put La Guajira's 23 in the region pool; the rule leaves one-round
  respondents out of both pools (4.4-7.1 against 8.3-19.2).
- **Carchi (`ec`), reported and not changed.** Same rule on Ecuador's design regions (`estratopri`
  Costa, Sierra, Oriente, which nest the provinces) and the 20 provinces in both halves: Sierra is
  closer for 8 of 10 (11.7 against 16.1, p=0.025), so Carchi would draw 79.6% Catholic, 7.1%
  evangelical, 4.0% Ninguna against the national 75.5%, 11.0%, 6.0%; its own 20 interviews (18
  Catholic) point the same way. Pastaza and Orellana would stay national (Oriente closer for 0 of
  4). Carchi's line was Anita's call; scratch script, not in the tree.
- **Not drawn**: Chocó (566,358), Arauca (314,513), Vichada (119,447), Guaviare (96,051),
  Amazonas (87,452), San Andrés, Providencia y Santa Catalina (66,269), Guainía (53,839). LAPOP
  has no code for any of them in `prov_es`. **1,303,929 people, 2.45%** of COD-PS 2025, in `gap=`
  with `gap_share=0.0245` hand-written (nobody was in any table). They keep polygons and hexes.

## 7. Mapping calls (`taxonomy/co2023.py`)

- **`Religiones Tradicionales` -> `other.co`, not `indigenous`.** 18 of its 37 respondents are in
  Bogotá and 23 are the 2014 wave; Cauca, Nariño and La Guajira, the sampled departments with large
  indigenous populations, contribute none. It is 0.48% as drawn and at the national rate anyway.
  The four earlier LAPOP countries put it on `indigenous`; this is the first to differ, on the data.
- `other.co` added to `taxonomy/branches.py` (the per-country LAPOP pattern), `build_tree.py` run.
- Everything else as `gt2023.py` / `ec2023.py`.

## 8. What was drawn

51,912,663 people on 26 departments. As drawn after §6's redraw: Católico 71.53%, Ninguna
(creyente) 8.91%, Evangélica 7.96%, Protestante tradicional 7.02%, Otro 1.46%, agnostic/atheist
0.98%, Witnesses 0.87%, Eastern 0.67%, Traditional 0.48%, Jewish 0.07%, Mormon 0.04% (first build:
Católico 71.62%, Ninguna 8.92%, Evangélica 7.87%).

Catholic runs 61.1% (Cesar), 63.7% (Atlántico), 63.9% (Sucre), 64.0% (Magdalena), 64.8% (Córdoba)
up to 81.2% (Boyacá), 81.4% (Cauca), 83.5% (Caldas, Norte de Santander), 83.6% (Huila).
Evangelical 17.5% Magdalena, 17.3% Putumayo, 15.2% Atlántico down to 2.2% Boyacá, 2.4% Caldas.
Ninguna 17.9% Risaralda, 13.4% Tolima, 13.0% Valle down to 1.8% Huila. **Several departments are
one municipality in most waves** (Huila is Neiva, Norte de Santander Cúcuta, Risaralda Dosquebradas,
Caquetá Florencia, Putumayo Puerto Asís, Tolima Ibagué from 2012), so their own extremes stand for
that city; `note_public` quotes only departments sampled in two or more municipalities.

The 1:1,000 dots were tallied back against `co.csv`: every node matches to the thousand. **5 dots
land inside blank departments (2 Chocó, 2 Guaviare, 1 Vichada) and 34 outside every polygon, and
all 39 are within 425 m of a drawn department** (median 30 m for the 34). That is a whole 400 m
hex assigned to a department by its centroid and straddling the line, not anybody drawn in a
department the survey never reached.

## 9. Open

- **The non-Christian tail**, as for every LAPOP country (`queue.md`'s refinement list).
- **ENDR 2019** (Beltrán and Larotta, 11,034 adults in all 33 units): department interview counts
  in its Annex 2 and seven-region shares in Gráfica 8. Microdata would replace this build; the
  seven regions are a national and regional level check nobody has wired.
- **DANE's ECP `P6945`** (2015-2023, five regions) is a second instrument on the same card at the
  region level: the natural test of whether traditional Protestant really has no geography.
- The seven blank departments need any Colombian source that sampled them; ENDR did.

## 10. Review, 2026-09-14 (session `f95259a4-corev`)

Checked from `lapop_co.feather`, `lapop_slim.feather` and `co.csv`, not from this record.

- **The seven blank departments: confirmed.** No `prov`, no `municipio` prefix and no 2010 `upm`
  prefix for any of them in any Colombian row, 2004-2023. La Guajira's `prov` appears in 2023
  only; Quindío, Casanare and Vaupés in 2004-2010 only.
- **The Florida refile: confirmed, on a witness §3 does not use.** `municipio` and `upm` carry the
  same DANE code, and Nariño has its own La Florida (CO52381), so they are closer to one witness
  than two. The cluster numbers settle it: 2012 and 2014 share one numbering for every Nariño and
  Valle municipality (Cumbal 209-212, Ipiales 213-216, Tumaco 221-224, Cali 225-230, Buenaventura
  237-240, Ginebra 241-244), and 217-220 are Florida in both rounds, filed under Valle in 2014.
  The design agrees (Nariño three municipalities and Valle four in 2010, 2014 and 2018; 2012 as
  printed would be four and three), and so does the sample share (2012 as printed, Nariño 6.3%
  against 3.2% of the population and 3.3-4.8% in the other rounds; 4.7% after the move).
- **`_halvings` (now `stability.py::halvings`): correct.** Checked against a brute-force list of every floor/ceiling split for
  2 to 7 rounds; five rounds give the ten.
- **`Religiones Tradicionales` on `other.co`: holds, but the stated reason needs its comparison.**
  Every LAPOP country's code-7 answers lean somewhere, so "18 of 37 in Bogotá" means little alone.
  Ratio of a unit's share of code 7 to its share of the sample: Bogotá **2.8x**, and 23 of the 37
  are 2014. Guatemala's lean the other way, Quiché 4.8x, Petén 3.2x, Alta Verapaz 1.7x, Guatemala
  department 0.8x, so there the box does sit where indigenous people are. **Panama (Panamá 1.4x)
  and Costa Rica (San José 1.5x) lean to the capital like Colombia, more weakly, and kept
  `indigenous`.** So the line is the strength of the lean plus the one-round spike, not capital
  concentration as such. §7, `co2023.py` and `other.co` name Guatemala, El Salvador and Ecuador as
  the countries that differ, and leave out Panama and Costa Rica, which also map it to `indigenous`.
- **La Guajira at the national rate is visibly wrong.** Drawn 71.6% Catholic and 7.6% evangelical,
  it is more Catholic and less evangelical than every sampled Caribbean department (Cesar,
  Magdalena, Atlántico, Bolívar, Sucre, Córdoba: 61.1-67.9% and 10.6-17.5%), which reverses
  `note_public`'s own Caribbean sentence on a million people. Its 23 Manaure interviews (17 rural)
  are 10 Catholic, 5 Protestant, 4 evangelical, 3 Ninguna, 1 agnostic; the national rate expects
  about 16 Catholics, and Atlántica without La Guajira in the same round is 57.0% Catholic. §6's
  leave-one-out averaged every region and counted traditional Protestant, which is drawn at the
  national rate whichever fallback is used. On the three answers actually drawn on their own
  shares, and within Atlántica alone, the region beats the country for **all six** sampled
  departments (summed error 5.0-7.6 pp against 8.3-19.2). It loses in Pacífica and the old
  national territories, which is why the country-wide average came out even. The other three
  one-round departments show no consistent direction (Quindío and Vaupés read above the national
  Catholic rate, Casanare below it, with 8 of 26 Protestant). **Recommended: La Guajira on
  Atlántica's shares for the three placed answers, or one sentence in the note.** In `queue.md`.
  Not rebuilt here. **Acted on 2026-09-14 evening (`f95259a4-co2`): La Guajira and Casanare now
  take their region's shares, Quindío and Vaupés stay national, on spec §12's rule; §6.**
