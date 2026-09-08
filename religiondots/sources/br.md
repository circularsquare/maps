# Brazil — IBGE Censo Demográfico, religion, at município

Ingested 2026-09-02. `sources/br.py` rebuilds `data/normalized/br.csv` from
`data/raw/br/`. All of `data/` is gitignored, so this file is the record.

**Brazil is spec.md §3.4's worked example** — structure from the detailed source, totals
from the recent one — and it is the cleanest case of it in the project, because IBGE
published the two halves twelve years apart and has said it may never publish them
together.

---

## 1. Two tables, and why neither is enough alone

| | Censo 2010, SIDRA **2094** | Censo 2022, SIDRA **9537** |
|---|---|---|
| categories at município | **56** (65 defined, 9 unused) | **9** |
| municípios | 5,565 | 5,570 |
| universe | whole resident population | **persons aged 10 or over** |
| national total | 190,755,799 | 176,600,150 |
| rows written | 311,640 | 55,700 |

2010 has the categories and is fifteen years stale. 2022 has the totals and no
denominations. Both are written to `br.csv`, kept apart by `year` and `source_id`, and
**must not be summed** — the universes differ as well as the vintages.

IBGE stated on release (6 June 2025) that the evangelical denominational breakdown for
2022 is withheld over data quality and that it is still evaluating whether it can be
published at all. So the 2010 structure is not a stopgap pending a better table; it is
currently the only municipal-level denominational data Brazil has.

Per §3.4 the intended use is: take the 2022 municipal total, split it by 2010 municipal
shares, and record `structure_year: 2010, total_year: 2022` on every resulting figure.
`br.py` does not do that — it is a normaliser, and the interpolation is a downstream step.

**That step exists as of 2026-09-04: `br_rescale.py`, writing
`data/normalized/br_municipio_rescaled.csv`, which is what `countries.py` now reads.** See
§12 below. `br.py` is unchanged and still writes both censuses side by side, because the
rescale needs both and because the raw years should stay available.

## 2. The 2022 universe is 10+, and the gap is not a residual

**The single easiest mistake available here.** The 2022 religion question was asked only of
persons aged 10 or over, so the table totals **176,600,150 against a 2022 population of
203,080,756**. The 26.5M difference is children who were never asked.

That is not a not-stated residual and must not be drawn as one. 2022 has its own
not-stated categories (`Não sabe` 95,355 and `Sem declaração` 199,472) and they are small.

2010 asked everybody, so the 2010 total *is* the population and there is no gap.

## 3. 2022's second loss: "Outras religiosidades" swallows five traditions

The 2022 nine-category list is not simply 2010's tree pruned at the top. **`Outras
religiosidades` is a 7,079,101-person catch-all** that contains Judaism, Islam, Buddhism,
Jehovah's Witnesses, LDS, Hinduism, Orthodoxy and the esoteric traditions — every one of
which has its own row in 2010.

In 2010 the same label means something completely different: `Outras religiosidades` is a
**11,307-person** leftover. Same name, 626× the size, different meaning. Anything joining
the two years by category name will silently produce nonsense.

So it is not only the evangelical breakdown that 2022 loses. The full 2022 list is:

| category | 2022 |
|---|---|
| Católica Apostólica Romana | 100,216,153 |
| Evangélicas | 47,418,024 |
| Sem religião | 16,385,342 |
| **Outras religiosidades** | **7,079,101** |
| Espírita | 3,257,455 |
| Umbanda e Candomblé | 1,849,824 |
| Sem declaração | 199,472 |
| Tradições indígenas | 99,425 |
| Não sabe | 95,355 |

### One real finding in the 2022 numbers

**Umbanda e Candomblé went 588,797 → 1,849,824, a 3.1× rise**, while the population grew
6%. Whatever that is — real growth, reduced stigma, or a changed question — it is the
largest proportional move between the two censuses and it is at município level in both.

## 4. Neither year sums across geographic levels — measured, not assumed

**Both censuses are sample tabulations, and IBGE expands the sample independently at each
geographic level.** So SIDRA's own national row does not equal the sum of SIDRA's own
municipal rows for the same table and category:

| | sum of municípios | IBGE national | drift |
|---|---|---|---|
| 2010 Católica | 123,280,184 | 123,280,172 | +12 |
| 2010 Espírita | 3,848,897 | 3,848,876 | +21 |
| 2022 Evangélicas | 47,417,990 | 47,418,024 | −34 |
| 2022 Católica | 100,216,170 | 100,216,153 | +17 |

Under 1 part in 3,000,000, and the same species as Canada's base-5 rounding (§3.8):
parent and child disagree **by construction**. A reconciliation written to demand equality
fails on every category, which is what the first run did.

`br.py` therefore checks within a tolerance (100 absolute or 1 ppm) and prints the drift so
it stays visible. **The 2010 grand total is a universe count rather than an expansion and
is exact** — it is checked exactly, and it is what would catch a genuinely missing município.

The internal nesting drifts the same way: 2010 `Umbanda e Candomblé` 588,810 against its
three children summing to 588,808.

## 5. The category tree is in the label text, not in any code

Classification 133 is **nested, not a partition**. `Evangélicas de origem pentecostal -
Igreja Assembléia de Deus` is a child of `Evangélicas de origem pentecostal`, which is a
child of `Evangélicas`. **Summing every row of a município triple-counts.**

IBGE encodes this only in the label string, with a `-` separator that also appears inside
real names. So `br.py` restates the tree explicitly in `CATEGORY_PARENT` rather than
parsing it, and writes `level=` and `parent=` into the `note` column for `allocate.py`.

Depths: 20 categories at level 0, 15 at level 1, 21 at level 2.

## 6. Full 2010 category list, with presence

`L` is tree depth. Presence is municípios with a non-zero count, out of 5,565.

| L | national | present in | category |
|---|---|---|---|
| 0 | 123,280,184 | 5,565 | Católica Apostólica Romana |
| 0 | 42,275,449 | 5,565 | Evangélicas |
| 1 | 25,370,472 | 5,563 | Evangélicas de origem pentecostal |
| 0 | 15,335,521 | 5,490 | Sem religião |
| 1 | 14,595,984 | 5,456 | Sem religião - Sem religião |
| 2 | 12,314,404 | 5,553 | …Igreja Assembléia de Deus |
| 1 | 9,218,051 | 5,328 | Evangélica não determinada |
| 1 | 7,686,827 | 5,386 | Evangélicas de Missão |
| 2 | 5,267,020 | 5,312 | …de origem pentecostal - outras |
| 0 | 3,848,897 | 4,178 | Espírita |
| 2 | 3,723,852 | 4,587 | …Igreja Evangélica Batista |
| 2 | 2,289,645 | 4,495 | …Igreja Congregação Cristã do Brasil |
| 2 | 1,873,234 | 4,678 | …Igreja Universal do Reino de Deus |
| 2 | 1,808,385 | 3,399 | …Igreja Evangelho Quadrangular |
| 2 | 1,561,047 | 4,234 | …Igreja Evangélica Adventista |
| 0 | 1,461,502 | 3,877 | Outras religiosidades cristãs |
| 0 | 1,393,226 | 4,274 | Testemunhas de Jeová |
| 2 | 999,480 | 2,324 | …Igreja Evangélica Luterana |
| 2 | 921,241 | 2,810 | …Igreja Evangélica Presbiteriana |
| 2 | 845,371 | 4,076 | …Igreja Deus é Amor |
| 0 | 643,624 | 3,688 | Não determinada e multiplo pertencimento |
| 1 | 628,253 | 3,646 | …Religiosidade não determinada ou mal definida |
| 1 | 615,105 | 3,070 | Sem religião - Ateu |
| 0 | 588,810 | 1,598 | Umbanda e Candomblé |
| 0 | 560,792 | 1,582 | Católica Apostólica Brasileira |
| 1 | 407,333 | 1,298 | Umbanda |
| 2 | 356,033 | 1,214 | …Igreja Maranata |
| 2 | 340,940 | 1,165 | …Igreja Evangélica Metodista |
| 0 | 243,971 | 1,177 | Budismo |
| 0 | 226,506 | 1,017 | Igreja de Jesus Cristo dos Santos dos Últimos Dias |
| 2 | 196,661 | 1,585 | …Igreja o Brasil para Cristo |
| 0 | 196,103 | 1,896 | Não sabe |
| 2 | 180,116 | 940 | …Comunidade Evangélica |
| 1 | 167,366 | 845 | Candomblé |
| 0 | 155,969 | 1,464 | Novas religiões orientais |
| 0 | 131,592 | 2,342 | Católica Ortodoxa |
| 2 | 125,564 | 975 | …Igreja Casa da Benção |
| 1 | 124,428 | 752 | Sem religião - Agnóstico |
| 2 | 109,592 | 686 | …Igreja Evangélica Congregacional |
| 0 | 107,325 | 588 | Judaísmo |
| 1 | 103,736 | 1,244 | …Igreja Messiânica Mundial |
| 2 | 90,576 | 328 | …Igreja Nova Vida |
| 0 | 74,020 | 1,080 | Tradições esotéricas |
| 0 | 63,083 | 422 | Tradições indígenas |
| 0 | 61,736 | 752 | Espiritualista |
| 1 | 52,240 | 590 | …Outras novas religiões orientais |
| 0 | 45,841 | 222 | Sem declaração |
| 0 | 35,168 | 369 | Islamismo |
| 2 | 30,660 | 273 | Evangélicas de Missão - outras |
| 2 | 23,476 | 405 | …Evangélica renovada não determinada |
| 1 | 15,387 | 326 | …Declaração de múltipla religiosidade |
| 1 | 14,109 | 163 | Outras declarações de religiosidades afrobrasileira |
| 0 | 11,307 | 287 | Outras religiosidades |
| 0 | 9,681 | 173 | Outras religiões orientais |
| 0 | 5,678 | 129 | Hinduísmo |

### Nine categories are defined but never published at município

`br.py` carries them in `KNOWN_ABSENT` and warns if a category *not* on that list is
absent. They belong to the **2000** census's version of classification 133, which shares
the same classification id — so the metadata advertises 65 categories and the 2010 data
has 56. The absent ones are `Evangélicas sem vínculo institucional` (+2 children),
`Evangélicas - outras religiões evangélicas`, `Outras cristãs` (+2 children),
`Não determinadas`, and the two `outras …` leaves at codes 100408 / 100416.

Nothing is suppressed for disclosure: **0 suppressed cells in either year.**

## 7. Exact URLs and re-fetch

    python sources/br.py --fetch      # 54+ SIDRA calls, ~15 min, ~53 MB raw

No key, no login. The API is `https://apisidra.ibge.gov.br/values` with path segments
`/t/<table>/n6/in n3 <uf>/v/<var>/p/<year>/<fixed classifications>/c133/<codes>`.

    2010:  t/2094  v/93   p/2010  c86/0                 c133/<codes>
    2022:  t/9537  v/140  p/2022  c2/6794/c58/95253     c133/<codes>

**The fixed classification filters are load-bearing.** 2094 is cross-tabulated by
colour/race, so `c86/0` takes the Total column; 9537 is cross-tabulated by sex and age
group, so `c2/6794` and `c58/95253` take theirs. Getting these wrong returns a
plausible-looking file that is one demographic slice, with no error.

Category codes are read from
`https://servicodados.ibge.gov.br/api/v3/agregados/<table>/metadados` rather than
hardcoded.

### Two transport gotchas, both of which cost a run

- **SIDRA answers 400, not a partial result, when a request exceeds ~50,000 values.**
  Minas Gerais has 853 municípios, so 853 × 66 categories = 56,298 trips the cap while
  every other state fits. `fetch_uf()` halves the category list on refusal and merges,
  rather than hardcoding which states are too big.
- **`servicodados` gzips unconditionally and ignores `Accept-Encoding: identity`**, while
  `apisidra` does not compress at all. `urllib` decompresses neither. The symptom is a
  `UnicodeDecodeError` on byte 0x8b that reads like a charset problem and is not one.
  `read_url()` checks the magic number rather than the header.

Per sources.md §5a, a 200 is not a download: a SIDRA error arrives as a 200 carrying a
short JSON object rather than an array, so `get()` checks the shape.

## 8. Value sentinels

IBGE uses four, and only two of them are data:

| | meaning | handling |
|---|---|---|
| `-` | zero | written as 0 |
| `...` | not available at this level | dropped, and counted against `KNOWN_ABSENT` |
| `..` | not applicable | dropped |
| `X` | suppressed to avoid identifying an informant | dropped and counted — **0 occurrences** |

## 9. Geography

Counts are on **município**, IBGE 7-digit code, in `geo_id`. 5,565 for 2010 and 5,570 for
2022 — Brazil created five municipalities between the censuses, so the two years are not
the same unit set and a join between them will lose those five.

**Boundaries are not downloaded yet.** Per spec §8.1 they must be the vintage the data was
published on, which here means two different vintages for the two years. sources.md §9a
already records that geoBoundaries BRA ADM2 will need its vintage checked; IBGE publishes
its own malha municipal per census year, which is the safer choice.

No placement layer has been chosen yet either (§8.2). Brazil's analogue of the US tract is
the **setor censitário** (~310,000 for 2022), which is finer than needed but is designed to
a population target.

## 10. Licence

IBGE data is public and free to reuse with attribution ("IBGE, Censo Demográfico 2010 /
2022"). **To confirm before shipping.**

## 11. Surprises, collected

- The 2022 universe is 10+, not everyone; the 26.5M gap is children, not non-response.
- `Outras religiosidades` means a 11,307-person leftover in 2010 and a 7,079,101-person
  catch-all in 2022. Same label, 626×, different contents.
- Umbanda + Candomblé tripled between the censuses.
- Municipal figures do not sum to national figures in either year, by up to 34 people,
  because each level is an independent sample expansion.
- The category tree lives in the label text, with a `-` separator that also occurs inside
  real names.
- The metadata advertises 65 categories; 9 of them are the 2000 census's and never appear.
- Minas Gerais alone is big enough to trip the API's value cap.
- Zero disclosure suppression in a 190M-person sample tabulation.

---

## 12. The §3.4 rescale — built 2026-09-04

`br_rescale.py` reads `br.csv` and writes `br_municipio_rescaled.csv`:

    est[m, leaf] = total_2022[m, group(leaf)]
                   x count_2010[m, leaf] / sum(count_2010[m, l] for l in group(leaf))

**Per município, not per nation.** Each município's 2022 group total is split by *its own*
2010 mix, so Assembleia de Deus keeps the regional shape it has in the north-east instead of
being smeared to a national average. Fallback where a município has no 2010 people in that
group: its state, then the nation.

### How often the local shape was missing

| 2022 category | 2010 leaves | people | state fallback | national |
|---|---|---|---|---|
| Católica Apostólica Romana | 1 | 100,216,170 | — | — |
| Evangélicas | 20 | 47,417,990 | 5 | 0 |
| Sem religião | 3 | 16,385,362 | 80 | 0 |
| Espírita | 1 | 3,257,448 | — | — |
| **Umbanda e Candomblé** | 3 | 1,849,835 | **3,972** | 0 |
| Tradições indígenas | 1 | 99,434 | — | — |
| Outras religiosidades | 17 | 7,079,124 | 261 | 0 |

**Umbanda e Candomblé needs the state fallback in 3,972 of 5,570 municípios**, and that is
not a defect — it is §3's finding restated. The tradition tripled, so in most municípios
there was nobody in 2010 to take a shape from, and the Umbanda/Candomblé split there is the
state's rather than the town's. Anyone reading that split at município level should know it.

### The category map, and the one entry that inverts

`GROUPS` in `br_rescale.py` is written out by hand and asserted to cover all 23 of 2010's
root categories exactly once. Six entries are one-to-one. The seventh is the trap from §3:
2022's `Outras religiosidades` is **fifteen** 2010 roots — Outras religiosidades cristãs,
Testemunhas de Jeová, Não determinada e multiplo pertencimento, Católica Apostólica
Brasileira, Budismo, os Santos dos Últimos Dias, Novas religiões orientais, Católica
Ortodoxa, Judaísmo, Tradições esotéricas, Espiritualista, Islamismo, Outras religiosidades,
Outras religiões orientais, Hinduísmo — and **not** the identically-named 2010 one.

The largest judgement in that list is `Não determinada e multiplo pertencimento` (643,624).
It is an answer rather than a non-response — 2010 keeps `Não sabe` and `Sem declaração`
separate and much smaller — so it belongs in a religion category, and 2022's catch-all is
the only cell it can be in. IBGE does not say. Arguable, and flagged in the script.

### Tier: 58.7% measured, 41.3% derived

Three 2022 categories map to a single 2010 leaf, so nothing is split and the 2022 count
passes through untouched: **Católica Apostólica Romana, Espírita, Tradições indígenas —
103,573,052 people, `measured`**. Everything else is `derived` and, per §3.10, may never
become a presence ring. Tier is per row here, exactly as `basis` is per row in §3.1.

### What it changed about the drawn map

| | before (2010 as it stood) | after (§3.4) |
|---|---|---|
| totals | 2010 | **2022** |
| universe | everyone, 190.8M | **10+, 176.3M drawn** |
| Catholic share | 64.6% | **56.7%** |
| Evangelical share | 22.2% | **26.9%** |
| Umbanda + Candomblé | 588,810 | **1,849,835** |
| dots at 1:1,000 | 190,498 | 176,291 |
| boundaries | malha 2010, 5,565 | **malha 2022, 5,570** |

Every município's rescaled rows sum to that município's own 2022 drawn total exactly (worst
gap 0.0000 over 5,570 units), and every 2022 category's national total is preserved exactly.

### The boundary vintage moved with it

The drawn geography is now 2022, so `countries.py` reads `br_municipios_2022.gpkg` and
`sources/br_geo.py --year 2022` builds it from IBGE's single national file (194 MB, one
`.zip`, unlike 2010's 27 per-state files). Keeping the 2010 mesh would have been the mirror
image of the trap `br_geo.py` was written to avoid: the five municípios created since 2010
would have had no polygon at all and their 49,483 people would have been dropped, while
their five parents were drawn holding territory that is no longer theirs. The 2010 mesh is
still built by the default invocation and is still correct for the 2010 census as it stands.

The two IBGE lagoon pseudo-municípios (`4300001` Lagoa Mirim, `4300002` Lagoa dos Patos)
recur in the 2022 mesh and are dropped by the same rule.

---

## 13. Placement moved to setores — 2026-09-05

The counts are still municipal and cannot be finer; what changed is where inside a município
the dots sit. **`sources/br_setores.py` builds a placement layer of 466,996 census setores
weighted by setor population**, and `countries.py` reads it instead of the municipal mesh.
Full notes in `sources/br_setores.md`; the general lesson is spec §8.2d.

The case in one number: São Paulo is a single município holding 11.5M people, so its ~11,500
dots were being spread evenly over the Serra da Cantareira, the Billings and Guarapiranga
reservoirs and Avenida Paulista alike.

```
                         before              after
  placement polygons     5,565 municípios    466,996 setores
  dots                   176,291             176,291   <- unchanged, by construction (§4.1)
  polygons holding a dot 5,565               150,088
  fallbacks to equal shares                  0
```

**It is a population weight, not a religion one.** Nothing measures which setor a given
church's members live in, so a Catholic dot and an Assembleia de Deus dot spread identically
inside a município, and every municipal total is exactly IBGE's either way. §14.4 permits
refining placement and forbids inventing magnitude; this is the first half only.

Two traps from that build are in `br_setores.md` and are worth knowing here: **914 setores
arrive as several rows each** (river islands), so a population joined by código gives them
several times their weight unless the parts are dissolved first; and **the lagoon
pseudo-municípios appear a third time**, now as setores with a null `CD_MUN`.

## 14. IBGE has the denominational detail and is not publishing it

Found 2026-09-05 while looking for setor population.
`ftp.ibge.gov.br/Censos/Censo_Demografico_2022/Religioes/` contains exactly one file:
`Banco_descritor-Resultados_preliminares_de_Religiao.xlsx`, **4,327 rows** mapping the
free-text answers people gave to the nine published GRANDE GRUPO categories — CARMELITA,
CATOLICA MARONITA and CATOLICA MELQUITA all folding into *Católica Apostólica Romana*;
dozens of named Lutheran and Presbyterian bodies folding into *Evangélicas*.

It is a coding dictionary, not data, and it publishes no counts at all. What it does is
settle the question §1 leaves open: **the detail was collected and coded at four-figure
granularity, and only the nine-way roll-up was released.** So the 2010 structure is not
standing in for something that was never measured; it is standing in for something that
exists and is withheld. That does not change what this map can draw, and it does change how
long the §3.4 rescale is likely to be needed — if IBGE ever releases the coded detail, the
rescale becomes unnecessary rather than merely improvable.
