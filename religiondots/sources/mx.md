# Mexico — INEGI Censo de Población y Vivienda 2020

`source_id: mx_censo_2020` · ingested 2026-08-30 · `sources/mx.py` → `data/normalized/mx.csv`

**The headline, because it changes what can be drawn:** Mexico collects religion at
46 denominations and publishes it two different ways that never meet. The
**detailed 24-category table stops at entidad federativa (32 units)**; the
**municipio and locality data has only 4 aggregate groups**. There is no municipio ×
denomination table anywhere in INEGI's open output, and I looked hard (§ "What was
checked and ruled out"). So `mx.csv` carries **two geo levels at two different
category depths**, and they nest — see §Nesting.

---

## 1. What was downloaded, and how to re-fetch

`python sources/mx.py --fetch` re-downloads anything missing into `data/raw/mx/`, then
normalises. Direct download, no login, no captcha, no rate limiting.

| file in `data/raw/mx/` | bytes | URL |
|---|---:|---|
| `cpv2020_b_eum_12_religion.xlsx` | 290,935 | `https://www.inegi.org.mx/contenidos/programas/ccpv/2020/tabulados/cpv2020_b_eum_12_religion.xlsx` |
| `iter_00_cpv2020_csv.zip` | 36,615,814 | `https://www.inegi.org.mx/contenidos/programas/ccpv/2020/datosabiertos/iter/iter_00_cpv2020_csv.zip` |
| `clasificaciones_cpv2020_702825198701.pdf` | 2,669,914 | `https://www.inegi.org.mx/contenidos/productos/prod_serv/contenidos/espanol/bvinegi/productos/nueva_estruc/702825198701.pdf` |

- The **xlsx** is the *Tabulados del Cuestionario Básico*, topic 12 "Religión", national
  file (`eum` = Estados Unidos Mexicanos). Sheet `01` is by locality size, sheet `02` is
  the one used: **entidad federativa × sexo × denominación religiosa × age group**. Dated
  25/01/2021.
- The **ITER** zip is *Principales resultados por localidad*, national, fourth edition
  (2022-05-19). It holds one row per locality plus roll-up rows per municipio, per
  entidad and national, 286 columns, of which four are religion. Bulk, one download,
  no API.
- The **PDF** is *Clasificaciones del Censo de Población y Vivienda 2020*; it is the
  authority for the nesting claims below and is the only place the full 46-denomination
  classification is written down.

**Trap when probing INEGI URLs:** their IIS answers a missing file with **HTTP 200 and a
~2 KB HTML page**, not a 404. A `HEAD` sweep will look like every filename you invent
exists. `mx.py --fetch` rejects any download under 100 KB for this reason. This is how I
initially "found" 240 non-existent religion tabulados.

## 2. Geography level and vintage

| level in `mx.csv` | units | `geo_id` | from |
|---|---:|---|---|
| `entidad` | 32 | 2-digit INEGI entidad code, zero-padded (`01`…`32`) | xlsx sheet 02 |
| `municipio` | **2,469** | 5-char CVEGEO = entidad + municipio (`01001`, `07115`) | ITER, rows with `LOC == '0000'` and `MUN != '000'` |

**Vintage: the census's own geography, the Marco Geoestadístico of the 2020 census**
(catálogo de claves de áreas geoestadísticas, 2020). 2,469 units = 2,453 municipios plus
Ciudad de México's 16 demarcaciones territoriales. Oaxaca alone holds 570 of them;
the smallest is **Santa Magdalena Jicotlán, Oaxaca (20047), population 81**.

Per spec §8.1 the boundary file must be this vintage, and **the project's default
boundary source is the wrong one**:

> geoBoundaries `gbOpen/MEX/ADM2` is **`boundaryYearRepresented: 2012`, 2,457 units,
> sourced from a World Bank 2012 municipality file**, licence CC BY 4.0.

2,457 against 2,469 is a 12-unit shortfall, and the join key is geoBoundaries' own
`shapeID`, not CVEGEO. This is the Connecticut failure from spec §8.1 waiting to happen,
and the drift is easy to see in the census file itself: **Morelos runs to `17036`, its
last three being `034 Coatetelco`, `035 Xoxocotla` and `036 Hueyapan`** — the indigenous
municipios its congress created in 2017–19, absent from any pre-2020 boundary set.
Chiapas did the same in 2011 and 2017. So a newer file has units the census has no row
for, and an older one is missing units the census does have. **The right join target is
INEGI's own** *Marco Geoestadístico. Censo de Población y Vivienda 2020*
(ficha `https://www.inegi.org.mx/app/biblioteca/ficha.html?upc=889463807469`, ~2.65 GB).
I did not download it — `data/geo/` is not mine — but the widely-circulated direct zip
URL for it now soft-404s, so it has to be reached from the ficha page. Whatever is used,
spec §8.1's both-directions check applies: 2,469 data rows in, 2,469 polygons out.

**Finer geography is available and was left on the table.** ITER's 193,094 locality rows
carry the same four religion columns *plus* `LONGITUD`/`LATITUD` per locality — which
would place dots without any population grid at all (spec §8.2). Two reasons it is not in
`mx.csv`: **42% of locality-level religion cells are confidentiality-masked** (328,044
`*` and 608 `N/D` out of 772,376), and it would be ~770k rows. At municipio level there
is **no masking at all** — every one of the 9,876 cells is a number, which is why
`mx.py` treats a mask at that level as a hard error rather than a case to handle.

## 3. Categories and national counts

### 3.1 `entidad` — 24 categories, INEGI's published cut

Every one reconciles exactly against the national row of the same tabulado; `count` is
the "Población total" column with `Sexo = Total`, so all ages.

Total (población total) = **126,014,024**

| source_category | national | share |
|---|---:|---:|
| Católica | 97,864,218 | 77.66% |
| Bautista | 143,133 | 0.11% |
| Presbiteriana | 428,651 | 0.34% |
| Iglesia del Dios Vivo, Columna y Apoyo de la Verdad, la Luz del Mundo | 190,005 | 0.15% |
| Adventista del Séptimo Día | 791,109 | 0.63% |
| Iglesia de Jesucristo de los Santos de los Últimos Días (Mormón) | 337,998 | 0.27% |
| Testigo de Jehová | 1,530,909 | 1.21% |
| Cristiana | 6,778,435 | 5.38% |
| Evangélica | 2,387,133 | 1.89% |
| Pentecostal | 1,179,415 | 0.94% |
| Otro Protestante/cristiano evangélico | 328,519 | 0.26% |
| Judía | 58,876 | 0.05% |
| Islámica | 7,982 | 0.01% |
| Origen oriental | 29,985 | 0.02% |
| New Age y Escuelas esotéricas | 11,816 | 0.01% |
| Raíces étnicas | 33,372 | 0.03% |
| Raíces afro | 40,799 | 0.03% |
| Espiritualista | 36,764 | 0.03% |
| Cultos populares | 19,481 | 0.02% |
| Otras religiones o movimientos religiosos | 9,094 | 0.01% |
| Ninguna religión | 9,488,671 | 7.53% |
| Ateos/Agnósticos | 722,381 | 0.57% |
| Sin adscripción religiosa (creyente) | 3,103,464 | 2.46% |
| No especificado | 491,814 | 0.39% |

Two categories carry a footnote in the source; the superscript digit is stripped from
`source_category` (it is a marker, not part of the name) and the footnote text is in
`note`:

- **Origen oriental** — *"Incluye las denominaciones religiosas: budista, hinduista y
  otras de origen oriental."* So Buddhism and Hinduism in Mexico are **not separable in
  any published table**: 29,985 people, one category.
- **Otras religiones o movimientos religiosos** — *"Incluye la denominación religiosa
  católica ortodoxa."*

### 3.2 `municipio` — 4 aggregate groups

`source_category` is INEGI's own `Indicador` wording, read verbatim out of the data
dictionary inside the ITER zip rather than retyped.

| ITER column | source_category | national |
|---|---|---:|
| `PCATOLICA` | Población con religión católica | 97,864,218 |
| `PRO_CRIEVA` | Población con grupo religioso protestante/cristiano evangélico | 14,095,307 |
| `POTRAS_REL` | Población con otras religiones diferentes a las anteriores | 248,169 |
| `PSIN_RELIG` | Población sin religión o sin adscripción religiosa | 13,314,516 |

## 4. Nesting, and what was kept — READ THIS BEFORE SUMMING ANYTHING

**The two levels nest exactly.** The four ITER groups are the four aggregates of the 24
denominations, and `mx.py` proves it on every run rather than asserting it:

```
nesting: denominations -> PCATOLICA         97,864,218  =  97,864,218  ok
nesting: denominations -> PRO_CRIEVA        14,095,307  =  14,095,307  ok
nesting: denominations -> POTRAS_REL           248,169  =     248,169  ok
nesting: denominations -> PSIN_RELIG        13,314,516  =  13,314,516  ok
ITER omits "No especificado"                  491,814  =     491,814  ok
```

with the roll-up being

| ITER group | the entidad denominations it contains |
|---|---|
| `PCATOLICA` | Católica |
| `PRO_CRIEVA` | Bautista · Presbiteriana · Luz del Mundo · Adventista del Séptimo Día · Mormón · Testigo de Jehová · Cristiana · Evangélica · Pentecostal · Otro Protestante/cristiano evangélico |
| `POTRAS_REL` | Judía · Islámica · Origen oriental · New Age · Raíces étnicas · Raíces afro · Espiritualista · Cultos populares · Otras religiones o movimientos religiosos |
| `PSIN_RELIG` | Ninguna religión · Ateos/Agnósticos · Sin adscripción religiosa (creyente) |
| *(nothing)* | No especificado — **ITER does not publish it** |

**What was kept:** *both*, at different `geo_level`s, because they answer different
questions and neither is derivable from the other. **Within** a level the categories are
a clean partition; **across** levels they are the same people counted twice.

> **Rule for anything consuming `mx.csv`: filter on `geo_level` first. Never sum
> `entidad` rows and `municipio` rows together, and never treat an entidad total as
> additional to its municipios.**

The `ROLLUP` dict in `sources/mx.py` exists only to run that check. It is **not** a
taxonomy mapping and nothing downstream should read it — spec §2.4 defers cross-source
denomination matching, and this is a within-source aggregation, not a taxonomy node.

**Nothing else double counts.** The tabulado's own `Total` row is dropped on read, so no
parent row survives into the CSV; ITER's `LOC` codes `9998`/`9999` (small-locality
roll-ups) and its `MUN == '000'` entidad totals are all skipped for the same reason.

## 5. Basis

**`self_id` on every row.** Census question, open-ended, asked of the whole resident
population: *"escriba el nombre de la religión tal como la declare el informante,
textualmente y sin abreviaturas"*, then post-coded to the 46-denomination classifier.
No `roll`, `estimate` or `attendance` rows in this source — unlike ASARB, which mixed
bases within one file (spec §3.1), INEGI's religion output is homogeneous.

The honest asterisk on that, and it belongs in the unit panel: footnote 1 of the tabulado
says the population total *"incluye una estimación de población de 6 337 751 personas que
corresponden a 1 588 422 viviendas sin información de ocupantes y menores omitidos"* —
**5.0% of the national population is an imputed count**. Nationally that imputation has
been carried through into the religion categories (they sum to 126,014,024 with only
491,814 in "No especificado"), so the national and entidad figures are not visibly
affected. Locally it is a different story — see §6.

## 6. Surprises

**1. La Magdalena Tlaltelulco, Tlaxcala (29048) has no religion data at all, and the file
does not say so.** Population 19,036; the four religion columns sum to 1,562. The other
**91.8% — 17,474 people — is simply missing**, and there is no null, no asterisk, no
flag: the columns just do not add up to `POBTOT`. The cause is visible in the same row:
`VIVPAR_HAB` is 395 against `TOTHOG` 4,614, and `P_3YMAS` is 1,620 against a population
of 19,036, so nearly every dwelling in the municipality was counted but not interviewed
and the residents are footnote-1 estimates with no characteristics attached.

This bit me before I noticed it: ranking municipios by Catholic share over `POBTOT` put
La Magdalena Tlaltelulco top of the *least Catholic* list at 7.0%, which is nonsense —
it is a devoutly Catholic Tlaxcalan town whose data is missing. **Shares must be taken
over the sum of the four classified groups, not over `POBTOT`.**

Scale of the problem, because it is otherwise small and worth not over-reacting to:

| unclassified share of POBTOT | municipios |
|---|---:|
| under 1% | 2,403 |
| 1–5% | 57 |
| 5–20% | 7 |
| 20–50% | 1 (Santa María Chimalapa, Oaxaca, 21.2%) |
| over 50% | 1 (La Magdalena Tlaltelulco, 91.8%) |

Nine municipios hold 44,902 of the 491,814 nationally unclassified. So this is
spec §3.2's residual, it is real, and it is concentrated: nationally 0.39%, in one
municipality 92%.

**2. Orthodox Christians are filed under "other religions", not under Christianity.** The
official classifier puts *12 Católico ortodoxo* under credo *1 Cristiano*, a sibling of
*11 Católico*. The published tabulado does not: footnote 3 folds "católica ortodoxa" into
**Otras religiones o movimientos religiosos**, and the arithmetic confirms it lands in
ITER's `POTRAS_REL` alongside Judaism and Islam. A name-based or credo-based mapping
would put it in the wrong branch.

**3. The collected classification is nearly twice as deep as the published one.** INEGI
codes religion to **46 denominations under 14 grupos and 3 credos** (plus 9999, not
specified) — and publishes 24. Named in the classifier but in **no** published table:

> Anabautista/Menonita · Anglicana/Episcopal · Luterana · Metodista · Otras protestantes ·
> Amistad Cristiana · Asambleas de Dios · Iglesia Apostólica de la Fe en Cristo Jesús ·
> Iglesia de Dios · Iglesia de Dios de la Profecía · Iglesia de Dios en México del
> Evangelio Completo · Príncipe de Paz · Otras asociaciones pentecostales · Iglesia
> Cristiana Interdenominacional · Iglesia de Cristo · Iglesia del Nazareno · Movimientos
> Sincréticos Judaicos Neoisraelitas · Otras cristianas evangélicas · Protestante ·
> Católica ortodoxa · Budista · Hinduista · Otras de origen oriental · Ateos and
> Agnósticos separately

That is a **Mexican Mennonite count that exists in INEGI's database and is not
published** — directly relevant given spec §2.4's finding that the Anabaptist branch is
the deepest thing in the US data. Reaching it needs the Laboratorio de Microdatos or
procesamiento remoto (`https://www.inegi.org.mx/rnm/index.php/catalog/632`); the
full-count microdata is **not** downloadable.

**4. "Cristiana" is the second-largest religion in Mexico**, 6.78M people, 5.4% — bigger
than every named denomination combined. With "Evangélica" (2.39M), "Pentecostal" (1.18M)
and "Otro Protestante/cristiano evangélico" (0.33M) that is **10,673,502 of the
14,095,307** protestant/evangelical total sitting in categories that are self-descriptions
rather than bodies (spec §2.3's "whole tradition" row). Only **3,421,805 — 24% —** of
Mexico's protestants and evangelicals are in a category that names an organisation.

**5. The non-Christian minorities are statistically invisible below entidad.**
`POTRAS_REL` is 248,169 people — **0.2% of Mexico** — and **775 of 2,469 municipios
report exactly zero**. Judaism (58,876), Islam (7,982), all of Buddhism+Hinduism
(29,985) and every folk and Afro-Mexican tradition live inside that one municipal column.
On this source, R2 for Mexico is entidad-level or nothing.

**6. Where the interesting geography actually is.** Shares below are over the classified
base:

| | |
|---|---|
| least Catholic | Maravilla Tenejapa, Chiapas **14.5%** (53.2% protestant/evangelical, 32.2% no religion) |
| most protestant/evangelical | **Riva Palacio, Chihuahua 71.6%** — Old Colony Mennonite country, and the census cannot say so because "Anabautista/Menonita" is unpublished |
| most "sin religión" | Mecayapan, Veracruz **47.3%** |
| most Catholic | Bacadéhuachi, Sonora 99.8% |
| Catholics under half | **102 of 2,469 municipios** |

The low-Catholic map is Chiapas, the Oaxacan sierra and southern Veracruz — indigenous
municipios — plus the Chihuahua Mennonite campos. Nothing here is a `roll` artefact:
unlike ASARB (spec §3.6) **no municipio reports more adherents than residents**, because
this is self-ID by residence, not a congregation roll.

## 7. Licence

**INEGI Términos de Libre Uso de la Información** —
`https://www.inegi.org.mx/inegi/terminos.html`. Reproduction, redistribution, adaptation
and reordering, partial or whole extraction, and **commercial use are all permitted**.
Conditions: credit INEGI as the source in the form *"Fuente: INEGI, <nombre del
producto>"*; disclose any transformation applied so end users do not attribute it to
INEGI; do not imply INEGI endorses the result.

Suggested attribution strings:

- `Fuente: INEGI, Censo de Población y Vivienda 2020, Tabulados del Cuestionario Básico.`
- `Fuente: INEGI, Censo de Población y Vivienda 2020, Principales resultados por localidad (ITER).`

## 8. What was checked and ruled out

Each of these was tested, not assumed. Municipio × denomination does not exist in the
open output.

| route | result |
|---|---|
| per-entity tabulados `cpv2020_b_<ent>_NN_<tema>.xlsx` | 12 topics exist per entity, all with municipio detail — **religion is not one of them**. Swept `chh` over NN 01–25 × 18 topic words and all 32 entity abbreviations × `NN_religion`; the only hit anywhere is `eum` (national). |
| the religion tabulado itself | its own index sheet lists exactly two tabulados: by locality size, and by entidad federativa. |
| Tabulados Interactivos (PX-Web, `/app/tabulados/pxwebapi/api`) | database `Religion` holds three tables, `Religion_01/02/03`, all **entidad × sexo × age × census year 1990–2020**, and only católica / no católica / ninguna. Coarser than the xlsx. |
| SCINCE 2020 | same four aggregate groups as ITER, down to AGEB/manzana. Finer geography, no more categories. |
| ITER locality rows | four groups, and 42% confidentiality-masked. |
| full-count microdata (RNM catalog 632) | religion is there at the 46-denomination level; access is **Laboratorio de Microdatos / procesamiento remoto only**, not downloadable. |
| *Panorama de las religiones en México 2020* (INEGI, 2023) | a per-entidad sociodemographic profile publication. Municipios appear only as a count and a top-3-by-population list. No municipal religion table. Nice datum in passing: in Aguascalientes 89.0% of censal households had all members declare the same religion. |

**Leads not followed, in rough order of value:**

1. **Laboratorio de Microdatos / procesamiento remoto** for the 46-denomination cut — the
   only route to Mexican Mennonites, Lutherans, Buddhists and Hindus as separate numbers.
   Application process, not a download.
2. **Muestra censal (cuestionario ampliado) microdata**, which is freely downloadable and
   carries every basic-questionnaire variable including religion. It would give a
   *sampled* municipio × denomination table for the larger municipios — a spec §3.4-shaped
   derivation, `estimate`-flavoured despite the census origin, and worth doing only if
   entidad-level detail proves too coarse to draw.
3. **ITER locality rows + their lon/lat** as the placement layer for the four groups
   (spec §8.2), accepting the 42% masking at that level.
