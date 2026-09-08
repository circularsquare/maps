# Peru — INEI, Censos Nacionales 2017, variable C5P26

Wired 2026-09-07. 23,196,391 people, 1,874 census districts on 1,873 polygons, 8 categories,
**100% of the universe drawn**.

| | |
|---|---|
| source | INEI, **Censos Nacionales 2017: XII de Población, VII de Vivienda y III de Comunidades Indígenas**, variable `C5P26` — *P12a+: Religión que profesa* — tabulated by us at district on INEI's own Redatam webserver |
| basis | `self_id`, **population aged 12 and over** (23,196,391 of a 29,381,884 census count) |
| geography | **1,874 districts** — 12,378 people each. INEI also serves 196 provinces and 25 departments; all three reconcile to the person |
| categories | **8** plus the universe total; all 8 drawn |
| drawn | **23,196,391 of 23,196,391, 100%** of the universe — no `no especificado` cell exists |
| licence | INEI public dissemination server, no account, no terms accepted to query it |

**The reason to draw Peru is that its census names eight religions and its own published
tables name four.** `sources.md` §11t declined the country on the UNSD oracle's four; §11y
reopened it. This file is the build, and it also corrects two things §11y got wrong.

---

## 1. Access — the source is a query, and the prize is *categories*, not resolution

`censos2017.inei.gob.pe/bininei/` is INEI's Redatam webserver over the 2017 census microdata.
Unauthenticated, no terms gate, two useful endpoints:

```
POST .../RpWebStats.exe/CmdSet?     ITEM=PROGRED  MODE=RUN  BASE=CPV2017DI
     CMDSET=RUNDEF Job / SELECTION ALL / TABLE T / AS AREALIST / OF DISTRITO, Poblacio.C5P26
POST .../RpWebStats.exe/Frequency?  ITEM=FREQPOB  ROW=Poblacio.C5P26  AREABREAK=Distrito
```

Both answer with a shell whose iframe names a per-session temp file; fetch it from
`RpWebUtilities.exe/Text?LFN=<path>&TYPE=TMP`. Six queries build this country.

**The variable dictionary ships inline in the `CmdSet` page**, as a plain HTML table, which is
how `C5P26` was found and how the geographic hierarchy was read without guessing:

```
  1    PERU       CPV2017 - Distrital
  2    Departam   Departamento     2.1 CCDD / 2.2 NCCDD
  3    Provinci   Provincia        3.1 CCPP / 3.2 NCCPP
  4    Distrito   Distrito         4.1 CCDI / 4.2 NCCDI
  ...
  8.x  Poblacio.C5P26   P12a+: Religión que profesa
```

The base is named *"CPV2017 - **Distrital**"* and the hierarchy bottoms out at `Distrito`, so
**1,874 is the floor here and not a choice** — unlike Nicaragua, where INIDE serves comarcas
that cannot be plotted. Peru's ceiling and floor are the same number.

> **§5a again: a bad program returns `Tabla vacía` with HTTP 200.** No error, no non-200.
> `sources/pe.py` asserts the body contains a `<table>` and does not contain `Tabla vac`
> before saving anything. This fired for real during the build — `AS AREALIST OF DEPARTAM,
> Departam.NCCDD` returns an empty table because `NCCDD` is a *string* variable, which is
> what sent the names to the `AREABREAK` route instead. See §3.

## 2. Eight categories, and the four the world sees

`AS AREALIST OF DISTRITO, Poblacio.C5P26` returns:

```
  Católica            17,635,339   76.03%   -> christianity.catholic.latin
  Evangélica           3,264,819   14.07%   -> christianity.protestant
  Ninguna              1,180,361    5.09%   -> unaffiliated
  Cristiano              381,031    1.64%   -> christianity
  Adventista             353,430    1.52%   -> christianity.adventist
  Testigo de Jehová      173,602    0.75%   -> christianity.witnesses
  Mormones               113,659    0.49%   -> christianity.latterday
  Otra                    94,150    0.41%   -> other.pe
  ------------------------------------------------------------------
  Total               23,196,391            universe (12+), EXCLUDED
```

INEI's own headline release for the 2017 census prints **four**: Católica 17,635,339,
Evangélica 3,264,819, *otra religión* 1,115,872, Ninguna 1,180,361. And

```
  Otra 94,150 + Cristiano 381,031 + Adventista 353,430
       + Testigo de Jehová 173,602 + Mormones 113,659  =  1,115,872   exactly
```

**So the oracle's four is not a different count, it is this count with six columns collapsed
into one.** The UNSD Demographic Yearbook return is a summary of a deeper microdata variable,
and §11w's *"ask the oracle with sixteen columns, it ranks as well as exists"* meets its limit
here: **the oracle ranks what was reported, and a country can be deeper than its own return.**

## 3. Where the names come from, and the join that was refused

Redatam gives geographic **codes** in the `AREALIST` crosstab and geographic **names** only in
two places, neither convenient:

* `AS FREQUENCY OF Distrito.NCCDI` returns the names with no codes beside them.
* `AS FREQUENCY OF Distrito.CCDI` returns the codes with no names beside them.

Pairing those two by row order would be a §12 shape-2 join — *a confident wrong pairing* —
and it is not done, even though for Peru's 25 departments the two orders happen to coincide.
The `AREABREAK` route pairs them **at source**, in a single heading row:

```
  AREA # 010101 | Amazonas, Chachapoyas, distrito: Chachapoyas
```

which is where every district, province and department name in `pe.csv` comes from (§12's
Chile rule: take the name from the statistical source, not the boundary file). The one
irregular label is Callao, the *Provincia Constitucional*, which is its own department and
carries a two-part heading at district level and a bare one at province level.

## 4. Three geographies, all reconciling to the person

| level | units | people/unit | summed total |
|---|---:|---:|---:|
| `DEPARTAM` | 25 | 928,000 | 23,196,391 |
| `PROVINCI` | 196 | 118,000 | 23,196,391 |
| `DISTRITO` | **1,874** | **12,378** | 23,196,391 |

Three separate queries, aggregated independently by the engine. `sources/pe.py` asserts that
the 1,874 rebuild the 196 on all 1,764 cells and the 196 rebuild the 25 on all 225 — which is
what would catch a district dropped or double-counted.

**But none of those identities would catch a column landing in the wrong place**, because
every one of them reconciles whichever order the columns are read in. The check that would is
external, and it is §2's four published figures. All four are asserted.

## 5. The universe is 12+, and that is not an undercount

23,196,391 against a 2017 census population of **29,381,884**. The missing 6,185,493 are the
under-twelves: the variable's own label is `P12a+`, and C5P26 was simply not asked of them.
That is a **different thing from a §3.5 undercount** — nobody declined, nothing was
suppressed — and it is carried in `countries.py`'s `gap=` rather than drawn as a hole. At
21.1% it is the largest age floor on this map; Nicaragua's is 11.8% at age 5.

Within the universe the table is an **exact partition**: the eight categories sum to the
district total on all 1,874 rows, there is no `no especificado` category at all, and 100% of
the table resolves to a node.

## 6. The join is on **code**, which reverses Nicaragua on purpose

`sources/ni_geo.py` joins on name and calls the code join a trap. Peru is the opposite case,
and the difference was measured rather than assumed.

COD-AB's `adm3_pcode` is `PE` + the six-digit **ubigeo**, the same identifier INEI tabulates
on. Of the census's 1,874 districts:

* **1,872 codes are present in COD**, and **1,870 of those agree on the district name**
  character-for-character after folding accents.
* The two that do not are spelling, each confirmed by reading the whole province's district
  list out of both sources and finding them otherwise identical, position for position:

| ubigeo | census | COD | province |
|---|---|---|---|
| `051010` | `Hualla` | `Huaya` | Víctor Fajardo, Ayacucho |
| `150712` | `San Pedro de Laraos` | `Laraos` | Huarochirí, Lima |

Neither spelling occurs anywhere else in its province, so neither pairing is ambiguous.

**A name join would be the risky one here.** Peru has many districts sharing a name across
provinces — `San Juan`, `Santa Rosa`, `Pachía`, `Huarochirí` the district inside `Huarochirí`
the province — and disambiguating them would require the code. So the code carries the join
and the name is demoted to the check, which is Nicaragua's arrangement turned around. Both
files say which one is doing the work and why.

## 7. The 1,874-vs-1,873 gap is a **merge**, not a missing district

§11y read COD's ADM3 count of 1,873 against the census's 1,874 as a vintage difference —
"the COD boundaries were created 2015-07-24 and Peru creates districts by law between
censuses". **That is not what it is.** The whole difference is one pair, in Satipo province,
Junín:

```
  census 120604 Mazamari (24,193 people 12+) ─┐
                                              ├─> COD 120699 `Mazamari - Pangoa`, 5,675 km²
  census 120606 Pangoa   (38,036 people 12+) ─┘
```

COD carries **one** polygon spanning both districts and no separate polygon for either; the
ubigeo `...99` is the convention for a unit whose internal boundary is unsettled. Nothing is
dropped and nobody is lost: `pe_lookup.csv` sends both census codes to the single `PE120699`
unit and `countries.py` sums them there.

**The cost is stated rather than hidden.** 62,229 people, **0.27% of the universe**, draw at
half Peru's usual resolution, and Mazamari and Pangoa cannot be told apart on the map.

## 8. The witness that uses neither name nor code — and its first version was wrong

Nicaragua's third witness is *"the Moravians are the Caribbean coast"*, checked against COD's
own centroid longitudes. The same move was tried here on the Adventists, whose altiplano
history is well documented — the mission at Platería on the Puno shore of Lake Titicaca opened
in 1898 and ran the schools that made the Aymara altiplano Adventist. **It fired.** Three of
the ten most Adventist districts came out 800 km north:

```
  Yantalo       Moyobamba          17.8%   lat -5.97
  Omia          Rodríguez de M.    17.1%   lat -6.40
  San Fernando  Rioja              16.8%   lat -5.84
```

**The prior was wrong, not the join.** Those three are the **Alto Mayo**, the San Martín and
Amazonas colonisation frontier, and they are Peru's *other* historic Adventist region.
Adventist Peru is two places, and a witness that asserts one of them would have rejected a
correct join for being surprising.

So the check is now the property that made the naive version tempting in the first place,
stated **without naming anywhere**: *religion shares are spatially smooth*. Neighbouring
districts resemble each other, wherever the clusters happen to be, and a permuted join
destroys that while leaving every name, code and total intact. The threshold is not a guess —
it is calibrated against random re-pairings of the same shares on every run:

```
  Católica     r = 0.7537    best of 200 random re-pairings 0.1139   0/200 reach it
  Evangélica   r = 0.7502                                   0.1221   0/200
  Adventista   r = 0.6903                                   0.1003   0/200
```

**And the strongest witness is the placement grid**, because it shares no lineage with either
INEI's counts or OCHA's boundaries: a modelled 2023 population grid has to agree with a 2017
census about how many people are inside each of 1,873 polygons. `r = 0.9589 on 1,871 units,
against a best of 0.0666 over 500 shuffles.`

## 9. Placement — Kontur, and the smallest vintage gap on this map

`sources/pe_grid.py`, 258,279 Kontur 400 m hexagons. Peru needs a population grid for both of
§8.2's reasons at once, and harder than most:

* **Emptiness.** The Amazonian districts run to thousands of km² at well under one person per
  km²; Loreto alone is 29% of the land and 3% of the people.
* **The desert, which is the same problem inverted.** The coastal districts are mostly
  rainless waste with the entire population in an irrigated valley a few km wide.

And they compound where it matters most: the Adventist altiplano districts are large, high and
mostly empty.

**Counts 2017, grid 2023 — six years, the smallest vintage gap on this map** (Nicaragua's is
eighteen). National ratio 1.106.

**The ratio band is not the check here, and the reason is size rather than vintage.** Kontur
models population from GHSL and building footprints, which on a district of 237 people is
noise. The dependence was measured:

| census 12+ | districts | 1st pct | 99th pct | worst |
|---|---:|---:|---:|---:|
| < 500 | 133 | 0.51 | 7.18 | **12.60** |
| 500–2k | 602 | 0.57 | 7.05 | 8.85 |
| 2k–10k | 771 | 0.71 | 4.95 | 8.38 |
| > 10k | 365 | 0.57 | 2.97 | 5.15 |

Every unit above a factor of 8 is a small district and the largest 365 sit inside 5, so the
band is set to 15 as a tripwire against wholesale mispairing only. This is Nicaragua's
conclusion reached by a different route — there the band was ruined by an eighteen-year
vintage gap and the outliers were the agricultural frontier.

**Two districts are too small to contain a hex centroid.** Chisquilla (229 people 12+) and
Recta (174), both in Bongará, Amazonas. `scatter.py` drops a unit with no placement polygon
with a warning that is easy to miss, so they fall back to their own polygon as one uniform
placement cell — §8.2's equal-share default, applied to the two units where there is no grid
to do better with. It cannot move a dot across a boundary.

Of the Kontur hexes, 1,332 have centroids outside every district. Measured against COD's own
ADM0 outline, **646,575 of those people are in Ecuador, Colombia, Brazil, Bolivia and Chile**
and only **1,594 (0.005% of Peru)** fall in slivers between district boundaries.

## 10. The categories, and what the form does not ask

**`Adventista` is the content, and §11y overstated it by a factor of four.** §11y says
"Adventists are 1.5M nationally in the department run's arithmetic". The department run gives
**353,430, which is 1.52%** — a percentage misread as millions. The case never rested on the
total:

```
  San Anton                  Azángaro, Puno      23.41%     altiplano
  Crucero                    Carabaya, Puno      22.05%     altiplano
  Amantani                   Puno                20.87%     altiplano
  Yantalo                    Moyobamba, S.M.     17.82%     Alto Mayo
  San Pedro de Putina Punco  Sandia, Puno        17.67%     altiplano
  Huacullani                 Chucuito, Puno      17.48%     altiplano
  ... and 399 districts have none at all.
```

**Peru is the first census on this map to print a box for the Latter-day Saints.** 113,659
people named by the state rather than counted by the church or buried in an `other` bucket.
Its geography is the southern coast — Pacocha 2.23%, Islay 2.08%, Mollendo 1.85%, Pocollay
1.71% — and it is absent from 914 districts.

**`Cristiano` is a separate box from both `Católica` and `Evangélica`**, and 381,031 people
chose it while looking at the other two. It names no body, so it lands on the bare
`christianity` family node. Its geography is urban and coastal — Callao 4.52%, La Perla 4.16%,
Bellavista 4.12%, Villa el Salvador 4.02% — the Lima-Callao conurbation and not the
evangelical countryside. See `taxonomy/pe2017.py`'s REVIEW.

**`Ninguna` is 5.09% and reading it as secularity would be wrong.** It peaks in indigenous
Amazonia — Puerto Bermúdez 37.73%, Awajun 30.49%, Pinto Recodo 27.77%, Raymondi 26.94%, Rio
Santiago 26.27% — which are Asháninka, Awajún and Wampís districts. The census gives Amazonian
indigenous religions no box, and `Ninguna` is where a form with no box for your religion puts
you. It is drawn as given; §14.4 forbids splitting it on a reading, and `note_public` says so
in the open.

**`Otra` is the smallest residual with the sharpest geography.** 0.41% nationally; 20.66% in
Yavarí, 19.24% in Tournavista, 18.92% in San Pablo, 13.04% in Iberia, 12.04% in Pebas — Amazon
river districts on the Brazilian and Colombian frontier and the Amazon colonisation zones. A
spread of fifty to one, which by §9r's rule makes it a **missing category rather than a
mixture**. The **Israelitas del Nuevo Pacto Universal**, a Peruvian millenarian church founded
in 1968 whose settlement colonies are concentrated in exactly these districts, is the
strongest single candidate. It is not split; see `taxonomy/branches.py`'s `other.pe` and
spec §14.4.

> Note the difference between the two ways of having no listed religion: in the Awajún and
> Asháninka districts the answer that rises is **`Ninguna`**, and in the colonist frontier
> districts it is **`Otra`**. That is a real distinction and worth not flattening.

## 11. What else is on this server, unused

- The 2017 base carries **69 person variables**, including `C5P25` — *"Por sus costumbres y
  sus antepasados Ud. se considera"*, the self-identified ethnicity question, at the same
  1,874 districts. Peru therefore *can* cross indigenous identity with religion at district
  level, which is what would turn §10's `Ninguna` reading from an inference into a
  measurement. Not read here.
- `WEIGHT=Poblacio.FACTORPOND` offers *"Población total"* against the default *"Población
  censada"*. The unweighted censused count is used, which is what the published figures in §2
  are on.
- `FILTER` offers urban/rural (`Vivienda.ENCAREA`) and the base also carries housing and
  household entities. No religion cross-tabulation was needed from them.
- Only one base is served, `CPV2017DI`. The **2007 census** also asked religion (the oracle
  lists Peru for 2007 and 2017) and is not on this server.
