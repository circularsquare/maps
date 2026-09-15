# Chile — INE, Censo de Población y Vivienda 2024

Wired 2026-09-04. 15,205,784 people aged 15+, 346 comunas, 13 named categories.

| | |
|---|---|
| source | Instituto Nacional de Estadísticas, CPV **2024**, `P6_Religion-o-credo.xlsx` sheet 2 |
| basis | `self_id`, **people aged 15 or over** |
| geography | 346 comunas |
| categories | 13 named, plus Ninguna and a non-response line |
| drawn | **15,118,269 people — 81.8% of the population**, 99.4% of the 15+ universe |
| licence | INE open data, attribution |

**The deepest category list per head of any country added since Croatia**, and it cost one
download. Chile asked about religion for the first time since 2002 — the 2017 census was
abbreviated and dropped it — and rewrote the question with the Oficina Nacional de Asuntos
Religiosos rather than inheriting it.

---

## 1. Getting it

`https://censo2024.ine.gob.cl/estadisticas/` lists the thematic result workbooks: P1
disability, P2 indigenous peoples, … **P6 religion**. One open xlsx, no auth, no bot
protection, no API needed. Sheet 2 is the comuna table; sheets 1/3/5 are regional and
sheets 4/6 are the same comunas by sex and age band.

`D1_Poblacion-censada-por-sexo-y-edad-en-grupos-quinquenales.xlsx` is fetched too. **It is
not used to scale anything** — see §3 — only as `cl_geo.py`'s independent join check and
for the coverage figure.

## 2. The categories

```
  Católica                                        8,168,978   53.72%
  Evangélica o protestante                        2,466,607   16.22%
  Ninguna                                         3,903,308   25.67%
  Testigo de Jehová                                 121,805    0.80%
  Iglesia de Jesucristo de los Santos de los ÚD      94,266    0.62%
  Otros cristianos y tradiciones … con Cristo       199,426    1.31%
  Judía                                              28,153    0.19%
  Budista                                            18,221    0.12%
  Católica Ortodoxa                                  10,912    0.07%
  Musulmana                                          10,197    0.07%
  Hinduista                                           4,530    0.03%
  Fe Bahá'í                                           2,010    0.01%
  Otras religiones o credos                          89,856    0.59%
  Religión o credo no declarado                      87,515    0.58%
```

Two of these are not questionnaire options. **`Otros cristianos y tradiciones relacionadas
con Cristo` is derived by INE**, by coding the free-text answers given under option 11 and
pulling out the ones naming a Christian movement not on the list; `Otras religiones o
credos` is what stayed in option 11 afterwards. So Chile's residual has been actively
*reduced* by the source before publication, which is unusual on this map and is why
`other.cl` is unusually clean.

**The one thing it does not split is the biggest thing about Chilean religion.**
`Evangélica o protestante` is one cell holding 2.47M people, and Chilean Protestantism is
overwhelmingly Pentecostal — the Iglesia Metodista Pentecostal and the Iglesia Evangélica
Pentecostal are the largest religious bodies in the country after the Catholic Church.
`christianity.pentecostal` exists on the tree and is drawn for the US and Brazil. Chile is
not filed there, because the census does not say so and §2 forbids inventing the split at
ingest; see the REVIEW note in `taxonomy/cl2024.py`.

## 3. The 15+ universe is NOT scaled up — DECIDED 2026-09-04, and this is the interesting call

Question 31 was put to residents **aged 15 or over**. 3,274,648 people — 17.7% of Chile —
are outside the table. They are left outside it.

Scaling each comuna by `population / population 15+` was considered, and **Chile's own data
says what that would get wrong**:

```
  share of each age band professing a religion or creed
     65+      96.0%
     45-64    79.3%
     30-44    69.4%
     15-29    63.9%
     ---------------
     15+      75.1%    <- what a flat scale-up would assign to under-15s
```

A 32-point gradient. Under-15s are the children of the two youngest bands, so a uniform
scale-up would overstate religion among children by six to eleven points and understate
`Ninguna` by the same — roughly a quarter of a million people pushed the wrong way. It is a
bias in a known direction, not a coin flip.

**The precedent that looks like it permits scaling does not.** §3.5a applies Pew's adult
answers to American children, and the reason is that the alternative there was drawing
**51.6% of the country as nothing at all** — the country was otherwise not drawable. Chile
has no hole: INE publishes an exact, complete partition of its own universe down to the
comuna. Scaling would invent a magnitude the source already publishes properly, which is
what §14.4 forbids.

So Chile is drawn at **81.8% of its population**, and `note_public` says so in as many
words. That is the same kind of declared partial coverage North Macedonia carries at 92.5%
and Sri Lanka's `Other` bucket carries in a different form — not a special case.

If it is ever revisited, the correct version is **not** a flat scale but assigning under-15s
their parents' age-band rate, which is a larger modelling claim and would need its own §7
tier rather than passing as `measured`.

## 4. The arithmetic

Every check in `sources/cl.py` is an equality and all hold:

- each of the 346 comunas: its 14 category columns sum to its own published 15+ total;
- the 346 comunas sum to the workbook's national row on all 15 columns exactly;
- the national 15+ total is 15,205,784, the published CPV 2024 figure;
- **no suppression marks, no blanks, no sentinels anywhere.** The script raises on any cell
  that is not an integer rather than coercing, precisely because there is nothing to coerce
  today and a new mark must not slip in silently.

## 5. What the map shows — measured off the drawn data

**A country changing fast.** Catholics 53.7% of adults against 70.0% in 2002 and 76.9% in
1992; no-religion 25.7% against 8.3% in 2002. The age gradient in §3 says much of that is
cohort replacement rather than conversion.

**And the pattern is economic as much as regional.**

```
  most evangelical comunas (15+ pop >= 5,000)      least religious comunas
    Los Álamos    62.3% ev  (Biobío)                 Providencia   43.2% none  (Santiago)
    Curanilahue   62.1%     (Biobío)                 Ñuñoa         40.3%       (Santiago)
    Lota          61.0%     (Biobío)                 Villa Alemana 34.6%       (Valparaíso)
    Lebu          58.6%     (Biobío)                 La Reina      34.6%       (Santiago)
    Coronel       55.3%     (Biobío)                 Valparaíso    33.4%
```

Evangelical Chile is the **coal and forestry coast of Biobío**, where it is the majority
faith, thinning northward through La Araucanía (region shares: Biobío 33.5%, La Araucanía
27.5%, Los Ríos 25.8%, against 8.4% in Coquimbo). The least religious places are the wealthy
eastern comunas of Santiago. The most Catholic are rural Maule and the islands of Chiloé —
Curepto 80.9%, Quinchao 80.6%, Quemchi 79.7%.

**The small named groups are a map of immigration and of eastern Santiago.** A third of
Chile's Jews live in Las Condes, Lo Barnechea and Vitacura; the Orthodox and the Bahá'í
cluster the same way. The exception is Islam, whose largest single community is **Iquique**
in the far north, not the capital.

## 6. Not done

- The Pentecostal split, which no table carries (§2).
- The 2002 census, which would make the change measurable rather than quoted; §13 rules out
  a time slider.
- INE publishes indigenous belonging (P2) at the same geography. Crossing it with religion
  to bring Mapuche practice out of `other.cl` is exactly what §14.4 forbids, and the census
  does not cross the two questions itself.

## 7. Placement on Kontur, added 2026-09-14 (session `f95259a4-clht`)

Anita, looking at the map: *"population distributions look a bit more artificial than i'd
expect. could we improve population density granularity here?"*

**Confirmed first: until today every comuna's dots were spread evenly over its polygon.**
`countries.py` had no `place_weight` for Chile and `place` was `cl_comunas.gpkg`, so
`scatter.py` split each comuna's dots uniformly across its area (§8.2's default). Measured on
the dots as they were, **40.6% of Chile's 15,112 dots fell in no hex Kontur has as populated**,
46.3% on land under 5 people/km², and the median dot sat at 13 people/km². The screenshot of La
Araucanía and Los Ríos showed an even speckle over farmland and up the Andes at Lonquimay and
Curarrehue, with only Temuco and Valdivia visible, and only because they are their own small
comunas; around Santiago, Lampa, Colina, Pudahuel and Paine were filled edge to edge.

**§8.2e's floor is nowhere near.** Comunas run 6.3 km² (San Ramón) to 51,456 km², median 630
km² (EPSG:6933), so a median comuna is about 850 Kontur hexes at 0.74 km² and the smallest still
takes eight. None is under one hex.

**Built the way Haiti and Peru are**: `sources/cl_grid.py` joins Kontur's CL 2023 extract
(148,550 hexes) to the 345 comuna polygons on hex centroids and writes
`data/geo/cl/cl_hexes.gpkg`, 145,865 hexes; `countries.py::_cl_place_weight` uses it. The checks:

    hexes outside every comuna        2,685, 141,094 people (0.72%), the extract overruns the
                                      borders; dropped
    comunas with no hex centroid      none (fewest hexes: 10)
    Kontur 2023 / CPV 2024 total      19,492,352 / 18,480,432 = 1.055
    log census vs log Kontur, 345     r = 0.970; best of 500 random pairings 0.205
    comunas of 50,000+ people         0.33x to 1.90x after normalising

The per-comuna band is set at **12x** and is a tripwire, not evidence. It is Peru's shape: every
comuna past 3x has under 7,000 people. Timaukel (157 people, 9.2x) and Torres del Paine (203,
9.1x) are Magallanes estancias and park buildings; Pica (6,272, 5.9x) holds the Collahuasi mine
camp, whose workers the census counts at home. At 1:1,000 all of these draw at most six dots.
**Central Santiago runs low on Kontur (Santiago comuna 0.33x, San Miguel 0.44x, Ñuñoa 0.54x)**,
which does not matter here because each is its own unit and the weight only works inside one.

**`python kontur_cap.py cl` finds no block at the density cap.** Nothing to register.

**After**, same dots re-placed: **0.0%** in no populated hex, 0.5% under 5/km², median
6,333/km². Dots per node identical at 1:1,000 (15,112) and 1:10,000 (1,505); `scatter.py` placed
all 1,356 and 614 (comuna, node) rows on hex population and none on equal shares.

**What the screenshots show** (1400x900, 1:1,000, before and after, same camera; kept in the
session scratchpad, not in the tree). La Araucanía and Los Ríos: the even speckle over farmland
and the Andes is gone, and the dots now sit on Temuco, Valdivia, Osorno, Angol, Victoria,
Villarrica, Pucón and Loncoche and along the roads between them, with Lonquimay and Curarrehue
nearly empty. Santiago: the city now has an edge, and the ring of evenly spread dots across
Lampa, Colina, Pudahuel and Paine has become Colina, Lampa, Talagante, El Monte and Buin.

**What it does not fix.** It is a population weight, not a religion one: a Catholic and an
evangelical dot inside one comuna are spread the same way, so read clusters as "this comuna,
drawn where its people live". The religion geography is still the comuna's.
