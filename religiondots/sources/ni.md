# Nicaragua — INIDE, VIII Censo de Población y IV de Vivienda 2005, variable P13

Wired 2026-09-07. 4,537,200 people, 153 municipios, 8 categories, **100% of the universe
drawn**.

| | |
|---|---|
| source | INIDE, **VIII Censo de Población y IV de Vivienda 2005**, variable `P13` — *A qué religión pertenece* — tabulated by us at municipio on INIDE's own Redatam webserver |
| basis | `self_id`, **population aged 5 and over** (4,537,200 of a 5,142,098 census count) |
| geography | **153 municipios** — ~29,700 people each. INIDE serves 2,579 comarcas as well; see §3 |
| categories | **8** plus the universe total; all 8 drawn |
| drawn | **4,537,200 of 4,537,200, 100%** of the universe — no `no especificado` cell exists |
| licence | INIDE public dissemination server, no account, no terms accepted to query it |

**One category is the reason to draw the country.** `Morava` is **73,902 people, 1.63% of
Nicaragua and 53.3% of Prinzapolka** — the first place on this map where the Moravian Church
is anybody's plurality. Jamaica, Trinidad and the Caribbean small islands have all carried
`christianity.moravian` as a national rounding of well under 1%; none of them can show it as
a geography.

---

## 1. Access — the source is a query, and that is the finding

`redatam.inide.gob.ni` is **INIDE's own Redatam webserver**, linked from the main navigation
of `inide.gob.ni` as *Sistema en línea Redatam*. It is unauthenticated, has no terms gate,
and it will run an **arbitrary Redatam+SP program** against the 2005 census microdata. Three
bases are served: `VIVPOB1995`, `VIVPOB05`, `CENAGRO1`.

```
POST http://redatam.inide.gob.ni/redbin/RpWebStats.exe/CmdSet?
     MAIN=WebServerMain.inl  BASE=VIVPOB05  LANG=esp  CODIGO=XXUSUARIOXX
     ITEM=PROGRED  MODE=RUN  Submit=Ejecutar
     CMDSET=RUNDEF Job / SELECTION ALL / TABLE T / AS AREALIST / OF MUN05, PERS05.P13
```

The engine answers with a shell naming a per-session temp file it has just minted; fetch it
from `RpWebUtilities.exe/Text?LFN=<path>&TYPE=TMP` and the crosstab is an ordinary HTML
table. Four queries build this country — municipio counts, department counts, and a
`FREQUENCY` on each level's name variable, which returns `0505-Jalapa` strings and is where
the municipality **names** come from (§12's Chile rule: take the name from the statistical
source, not the boundary file).

**The variable list ships inline in the CmdSet page's `VARLIST` textarea**, which is how P13
was found and how the geographic hierarchy was read without guessing:

```
  Entity DEP05    I01 Código de Departamento, CDepto Departamento
  Entity MUN05    I02 Codigo de Municipio,    CMuni  Municipio
  Entity COMR05   I03 Codigo de Comarca,      CComarc Comarca
  Entity CMND05   I04 Codigo de Barrio o Comunidad
  Entity PERS05   ... P13 [I 1-8] "A que religión pertenece"
```

> **§5a in a new disguise: a bad program returns `Tabla vacía` with HTTP 200.** No error, no
> non-200, just an empty table — which would write an empty csv and reconcile against
> nothing. `sources/ni.py` asserts the body contains a `<table>` and does not contain
> `Tabla vac` before saving anything.

## 2. What INIDE prints, and why it is not what is drawn

The published route was worked first and it caps four times coarser than the served one.

INIDE's 2005 results pages are a JS accordion; the links live in `/js/censo.js` and resolve
to POST endpoints under `/Estadisticas/` (`censo2005Resultados`, `censo2005Municipio`,
`censo2005Departamento`, …). What they give:

| document | religion? | geography |
|---|---|---|
| `Vol.I Poblacion-Caracteristicas Generales.pdf` (335 pp) | **yes**, pp192–286 | **department only** |
| `Vol.IV Poblacion-Municipios.pdf` (546 pp) | **no** — the only hit is the p9 glossary | municipio |
| 468 per-municipality booklets under `CifrasMun/` | **no** | municipio |
| `VolVivienda`, `VolHogar` | no | — |

Volume I carries two religion tables and both are departmental: **CUADRO 12** (religion ×
department × urban/rural × sex × age) on pp193–245, and **CUADRO 13** (religion × indigenous
people or ethnic community × department) on pp247–286.

> **So the office PRINTS 17 units and SERVES 153 or 2,579, on the same website, for the same
> census.** That is the rule `sources.md` §11x adds: **ask whether a statistical office runs
> a Redatam instance before reading its PDFs.**

## 3. The comarca table is real, free, and unplottable

`OF COMR05, PERS05.P13` returns **2,579 comarcas at 1,759 people each** and reconciles to
4,537,200 exactly, like every other level. It would be among the three finest geographies on
this map. It is not drawn because **no comarca boundaries are published anywhere**:

- geoBoundaries `NIC` returns ADM1=17 and ADM2=153 and **404s on ADM3**.
- HDX `cod-ab-nic` says in its own description: *"structured into 2 levels: Admin 1: 17 …
  Admin 2: 153 municipality"*.

§11w's rule — *the one thing the oracle cannot see is the geography* — biting on the
resolution instead of on the country. **For once the boundaries are the ceiling and the
counts are not**, which is the reverse of nearly everything else in this file. If INIDE's
census cartography ever surfaces, `sources/ni.py` changes one identifier.

## 4. The universe is 5+, and that is not an undercount

4,537,200 against a 2005 census population of **5,142,098**. The missing 604,898 are the
under-fives: CUADRO 12's own title is *POBLACIÓN DE 5 AÑOS Y MÁS, POR RELIGIÓN*, and P13 was
simply not asked of them. That is a **different thing from a §3.5 undercount** — nobody
declined, nothing was suppressed, no answer was lost — and it is carried in `countries.py`'s
`gap=` line rather than drawn as a hole.

Within the universe the table is an **exact partition**: the eight categories sum to the
municipio total on all 153 rows, there is no `no especificado` category at all, and 100% of
the table resolves to a node. The fourth source here of which that is true, after Malawi,
Guyana and Zimbabwe.

## 5. The check is the printed volume, and it is genuinely independent

Every internal identity in this data reconciles whichever order the columns are read in, so
none of them would catch a column landing in the wrong place. The check that would is
external: **all nine of CUADRO 12's `LA REPÚBLICA` figures, typeset in 2006, are reproduced
exactly by a 2026 query against the microdata.**

```
  Total 4,537,200 | Católica 2,652,985 | Evangélica 981,795 | Ninguna 711,310
  Otra 74,101 | Morava 73,902 | Testigo de Jehová 42,587 | Musulmán 321 | Judaísmo 199
```

`sources/ni.py` asserts all nine. It also checks that the **153 municipios rebuild the 17
departments on all 153 cells** — two separate queries the engine aggregated independently,
which is what would catch a municipality dropped or double-counted.

## 6. The join is on name, and the code join is a trap

**This is §12's shape-2 failure — *a confident wrong pairing* — in its most inviting form.**

COD's `adm2_pcode` is `NI` + a four-digit code in INIDE's own format, and **145 of 153 match
the 2005 census codes exactly**. Ten municipalities were renumbered between the census and
COD's 2023 vintage, mostly around municipalities created in the 2000s, and the renumbering
cascaded. Eight go missing and would be noticed. **Five collide and would not:**

| INIDE 2005 | COD 2023, same code | |
|---|---|---|
| `9105` **Waspám** | `NI9105` **Mulukukú** | **different places** |
| `6545` **El Coral** | `NI6545` **San Francisco de Cuapa** | **different places** |
| `0515` El Jícaro | `NI0515` Jícaro | same place, spelling |
| `2020` San Juan de Río Coco | `NI2020` San Juan del Río Coco | same place, spelling |
| `5525` Municipio de Managua | `NI5525` Managua | same place, INIDE prefix |

**The first line is the whole argument.** Waspám is the Río Coco Miskito municipality —
38,926 people, **43.6% Moravian**, one of the four units carrying the category this country
is drawn for. Mulukukú is an interior mining-triangle municipality created in 2005, 0.3%
Moravian. A code join sends Waspám's Moravians inland and puts a mestizo Catholic profile on
the Honduran border, **and every total in `ni.py` still reconciles**, because a permutation
of units preserves every sum.

So `sources/ni_geo.py` joins on **name** — unique on both sides, 150 of 153 folding to the
same string, three spelling aliases listed with reasons — and then confirms it three ways:

1. **department.** All ten renumberings stay inside the same department, so INIDE's
   two-digit prefix must equal COD's on all 153. It does.
2. **what the code join would have done.** Reported rather than assumed, and the file
   **refuses to run if the code join ever stops mispairing anything** — that would mean
   either COD has re-aligned to 2005 (good news, to be acted on deliberately) or the census
   read has changed.
3. **the data's own geography.** Moravian Nicaragua is the Caribbean coast; after the join
   the six most Moravian municipios are checked to be east of 85°W using COD's own centroid
   longitudes. A witness that uses neither name nor code, and the one that would catch a
   systematic east/west swap.

## 7. Placement — Kontur, and an eighteen-year vintage gap

`sources/ni_grid.py`, 47,270 Kontur 400 m hexagons. Nicaragua needs a population grid for
**both** of §8.2's reasons at once:

- **Emptiness.** The two Caribbean autonomous regions are **46% of the land and 12% of the
  people**. Waspám alone is 9,341 km², larger than eleven whole departments. And the emptiest
  units are exactly the ones carrying `Morava`, so an equal share per polygon would paint the
  Moravian coast across uninhabited rainforest.
- **Water.** Lake Cocibolca is 8,264 km² and Xolotlán 1,042, and the municipal boundaries run
  out into both. A population grid has no hexes on open water, so §8.2c's problem does not
  arise rather than being patched — Malawi's finding again.

> **THE VINTAGE GAP IS THE LARGEST ON THIS MAP: counts 2005, grid 2023.** National ratio
> 1.355, and most of that is real growth. It moves dots **within** a unit and never between
> units, so no count is affected — but in the eastern frontier municipios the dots land in
> settlements that had barely begun in 2005. Prinzapolka is both the most Moravian
> municipality and the second most re-settled (4.79x), and that is worth knowing when reading
> its dots.

**It also decides which check carries the join.** This is Zimbabwe's case reversed
(`sources/zw_grid.py`): there the ratio band was tight and the correlation weak on 10 units.
Here the band is ruined by the vintage gap — the outliers are the agricultural frontier and
are *correct* — while the correlation on 153 units is decisive:

```
  band          all 153 inside a factor of 8 (wide, deliberately; a tripwire only)
  correlation   r = 0.9482, against a best of 0.2964 over 2,000 random pairings
```

## 8. The categories, and what the form does not ask

```
  Católica            2,652,985   58.47%   -> christianity.catholic.latin
  Evangélica            981,795   21.64%   -> christianity.protestant
  Ninguna               711,310   15.68%   -> unaffiliated
  Otra                   74,101    1.63%   -> other.ni
  Morava                 73,902    1.63%   -> christianity.moravian
  Testigo de Jehová      42,587    0.94%   -> christianity.witnesses
  Musulmán                  321    0.01%   -> islam
  Judaísmo                  199    0.00%   -> judaism
```

**The list is short in a specific direction.** Latin American censuses that go past
`Católica / Evangélica / Otra` normally add Adventists, Jehovah's Witnesses and the
Latter-day Saints — Mexico's list and Chile's. Nicaragua adds **Jehovah's Witnesses and the
Moravians and nothing else**. A national census printing a box for a church of 74,000 while
printing none for Adventists, Anglicans or Baptists is describing the Moravian Church's
standing on the Caribbean coast rather than its size.

`Otra` is 1.63% nationally and **44.05% on Corn Island**, 21.44% in Laguna de Perlas, 17.48%
in Bluefields — against 0.02% in San José de Cusmapa. Every municipality above 8% is on the
Caribbean. The strong reading is the **Anglican church of the Mosquito Coast** and the
**Jamaican Baptist mission** that worked the same Creole towns from the 1840s, neither of
which has a box. It is not split; see `taxonomy/branches.py`'s `other.ni` and spec §14.4.

Judaísmo (199 people in 57 municipios) and Musulmán (321 in 50) are both **under one dot at
1:1,000 and draw as §4.3 presence rings**. Both peak in Managua.

## 9. What else is on this server, unused

- **`VIVPOB1995` also carries religion** — `Poblac.Religion`, *P06-Religión*, 34 person
  variables, geography down to `Localidad`. A second Nicaraguan census with the question,
  not read here.
- `CENAGRO1` is the 2001 agricultural census; no religion.
- The **2005 questionnaire** (`Boleta Censal`) is linked from the accordion but the path in
  `CifrasMun/` 404s; the variable dictionary made it unnecessary.
- INIDE announced a **IX Censo de Población** (`/docs/cepov2023/`). Nothing published yet, so
  2005 is still the current census as of 2026-09-07.
